import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from model_params_def import preprocess_params
import improvelib.applications.drug_response_prediction.drug_utils as drugs_utils
import improvelib.applications.drug_response_prediction.omics_utils as omics_utils
import improvelib.applications.drug_response_prediction.drp_utils as drp

# [MODEL] Model-specific imports, as needed
import os
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from itertools import islice, combinations
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
import tensorflow.compat.v1 as tf
from codes.models.model_merge_simple_reg_3decoders import mgdc
from codes.models.optimizer_merge_reg_3decoders import optimizer
from train_embeddings import train_model, construct_synergy_networks, preprocess_graph, sparse_to_tuple
from fast_synergy_dataset import FastSynergyDataset
from sklearn.preprocessing import StandardScaler


# Tensorflow configurations
tf.compat.v1.disable_eager_execution()  # Required for TF 1.x-style execution
tf.compat.v1.disable_v2_behavior()  # Completely disables TensorFlow 2.x behavior
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True  # Allows CPU fallback for unsupported GPU ops

# [Req]
filepath = Path(__file__).resolve().parent

# [Req]
def run(params: Dict):
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the ML data files.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val/test sets
    # --------------------------------------------------------------------
    data_train_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")
    data_val_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")
    data_test_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")
    
    # --------------------------------------------------------------------
    # [Req] Create dataloaders and get response data - DRP specific
    # --------------------------------------------------------------------
    omics_obj = omics_utils.OmicsLoader(params)
    drugs_obj = drugs_utils.DrugsLoader(params)
    response_train = drp.DrugResponseLoader(params,
                                    split_file=params["train_split_file"],
                                    verbose=False).dfs["response.tsv"]
    response_val = drp.DrugResponseLoader(params,
                                    split_file=params["val_split_file"],
                                    verbose=False).dfs["response.tsv"]
    response_test = drp.DrugResponseLoader(params,
                                    split_file=params["test_split_file"],
                                    verbose=False).dfs["response.tsv"]
    # --------------------------------------------------------------------
    # [Req] Load X data (feature representations)
    # --------------------------------------------------------------------
    drug_feat_raw = drugs_obj.dfs['drug_infomax.tsv']
    cell_feat = omics_obj.dfs['cancer_gene_expression.tsv']
    
    # TODO Add check for IDs between response dataframe and features dataframes
    
    # --------------------------------------------------------------------
    # [MODEL] Preprocess X data
    # --------------------------------------------------------------------
    # Merge all response data
    response_all = pd.concat([response_train, response_val, response_test], ignore_index=True)
    # Extract unique drug and cell names
    drugslist = sorted(set(response_all[params['drug_col_name_1']]).union(set(response_all[params['drug_col_name_2']])))
    drugscount = len(drugslist)
    cellslist = sorted(set(response_all[params['canc_col_name']]))
    cellscount = len(cellslist)
    print(f"Total unique drugs: {drugscount}, Total unique cell lines: {cellscount}")
    
    drug_feat = sp.csr_matrix( drug_feat_raw )
    drug_feat = sparse_to_tuple(drug_feat.tocoo())
    num_drug_feat = drug_feat[2][1]
    num_drug_nonzeros = drug_feat[1].shape[0]
    
    # convert to numpy array
    drug_feat_array = drug_feat_raw.values
    cell_feat_array = cell_feat.iloc[:, 1:].values
    
    indexs_all = []
    for idx1 in range(drugscount):
        for idx2 in range(drugscount):
            indexs_all.append([idx1, idx2])

    # Create directory for embeddings
    resultspath = f"{params['output_dir']}/{params['embeddings_dir']}"
    if not os.path.isdir(resultspath):
        os.makedirs(resultspath)
        
    # Construct networks separately
    train_networks = construct_synergy_networks(response_train, cellslist, drugslist, indexs_all, params)
    val_networks = construct_synergy_networks(response_val, cellslist, drugslist, indexs_all, params)
    
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    
    placeholders.update({'net1_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
    placeholders.update({'net2_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})
    placeholders.update({'net3_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})

    # Create model
    model = mgdc(placeholders, num_drug_feat, num_drug_nonzeros, params["embedding_dim"], fncellscount=cellscount , name='mgdc')
    
    # Create optimizer
    with tf.name_scope('optimizer'):
        opt = optimizer(model=model, 
                preds_specific=model.reconstructions_specific, 
                preds_common=model.reconstructions_common, 
                lr=params["learning_rate"], 
                d_net1_indexs=train_networks["d_net1_index"], 
                d_net2_indexs=train_networks["d_net1_index"], 
                d_net3_indexs=train_networks["d_net1_index"], 
                fncellscount=cellscount)

   # Initialize session
    sess = tf.Session()
        # Train the model
    loss_history, min_loss, feed_dict = train_model(
                                            sess=sess,
                                            model=model,
                                            opt=opt,
                                            placeholders=placeholders,
                                            drug_feat=drug_feat,
                                            train_networks=train_networks,
                                            val_networks=val_networks,
                                            params=params,
                                            resultspath=resultspath,
                                            cellscount=len(cellslist))
    
    # Write min_loss to stats_loss.txt
    with open(os.path.join(resultspath, 'stats_loss.txt'), 'w') as f:
        f.write(f"{min_loss}\n")

    # --------------------------------------------------------------------
    # [MODEL] Save X data
    # --------------------------------------------------------------------      
    embeddings_common, embeddings_specific = sess.run( [model.embeddings_common, model.embeddings_specific], feed_dict=feed_dict)
    
    # save embeddings specific
    d_topology_specifics = {}
    for cellidx in range(cellscount):
        cellname = cellslist[cellidx]
        embeddings = pd.DataFrame(embeddings_specific[cellidx])
        embeddings.index = drugslist
        # scale specific topology
        scaler = StandardScaler().fit(embeddings)
        drug_topology_specific = pd.DataFrame(scaler.transform(embeddings), index=embeddings.index)
        d_topology_specifics[cellname] = drug_topology_specific
        file_path = os.path.join(resultspath, f"results_embeddings_specific_{cellidx}.txt")
        embeddings.to_csv(file_path, sep="\t",header=None, index=True)
    
    # save embeddings common
    embeddings_common = pd.DataFrame(embeddings_common)
    embeddings_common.index = drugslist
    embeddings_common.to_csv(os.path.join(resultspath, "results_embeddings_common.txt"), sep="\t",header=None, index=True)
    # scale common topology
    scaler = StandardScaler().fit(embeddings_common.values)
    embeddings_common.iloc[:, :] = scaler.transform(embeddings_common.values)
        
    # Dictionary mapping each stage to its respective response dataset
    response_datasets = {
        "train": response_train,
        "val": response_val,
        "test": response_test
    }

    # Define stage-specific filenames if needed
    stage_filenames = {
        "train": data_train_fname,
        "val": data_val_fname,
        "test": data_test_fname
    }

    for stage in ["train", "val", "test"]:
        print(f"Processing {stage} dataset...")

        if stage == "train":
            dataset = FastSynergyDataset(
                drug_feat=drug_feat_array,
                drug_feat_topology_common=embeddings_common,
                drug_feat_topology_specifics=d_topology_specifics,
                cell_feat=cell_feat_array,
                synergy_data=response_datasets[stage],
                params=params
            )
        else:
            dataset = FastSynergyDataset(
                drug_feat=drug_feat_array,
                drug_feat_topology_common=embeddings_common,
                drug_feat_topology_specifics=d_topology_specifics,
                cell_feat=cell_feat_array,
                synergy_data=response_datasets[stage],
                train=False,
                params=params
            )
        # Save dataset using the predefined filename
        dataset.save_to_npy(output_dir=params["output_dir"], stage_fname=stage_filenames[stage])

        print(f"Saved {stage} dataset to {params['output_dir']}")

    # --------------------------------------------------------------------
    # [Req] Save response data (Y data)
    # --------------------------------------------------------------------  
    frm.save_stage_ydf(ydf=response_train, stage="train", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=response_val, stage="val", output_dir=params["output_dir"])
    frm.save_stage_ydf(ydf=response_test, stage="test", output_dir=params["output_dir"])

    return params["output_dir"]

# [Req]
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="mgaedc_original_params.txt",
        additional_definitions=preprocess_params)
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])