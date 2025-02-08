import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from model_params_def import infer_params

# [MODEL] Model-specific imports, as needed
import torch
import torch.nn as nn
import numpy as np
import os
from fast_synergy_dataset import FastSynergyDataset
from codes.models.PRODeepSyn_datasets import FastTensorDataLoader
from model_definition import DNN

filepath = Path(__file__).resolve().parent # [Req]

# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for test set and build model path
    # --------------------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="test")

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["input_model_dir"])

    # --------------------------------------------------------------------
    # Load inference data (ML data)
    # --------------------------------------------------------------------
    test_data = FastSynergyDataset.load_from_npy(load_from=params["input_data_dir"], 
                                                 file_name=test_data_fname,
                                                 params=params)

    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data))
    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    device = torch.device(params["cuda_name"] if torch.cuda.is_available() else "cpu")
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------
    model = DNN(test_data.drug_feat1_len(), test_data.drug_feat2_len(), test_data.drug_feat2_len(), test_data.cell_feat_len(), params["hidden"])
    model.to(device)
    model.load_state_dict(torch.load(modelpath))
    with torch.no_grad():
        for test_each in test_loader:
            test_each = [x.to(device).float() for x in test_each]
            drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = test_each
            yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2,drug2_feats3, cell_feats)
            yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2,drug1_feats3, cell_feats)
            y_pred = (yp1 + yp2) / 2
            y_pred = y_pred.cpu().numpy().flatten()
            y_true = y_true.cpu().numpy().flatten()
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_true,
        y_pred=y_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_data_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    if params["calc_infer_scores"]:
        test_scores = frm.compute_performance_scores(
            y_true=y_true,
            y_pred=y_pred,
            stage="test",
            metric_type=params["metric_type"],
            output_dir=params["output_dir"]
        )

    return True


# [Req]
def main(args):
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="mgaedc_original_params.txt",
        additional_definitions=infer_params,
    )
    status = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])