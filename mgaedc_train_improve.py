import sys
from pathlib import Path
from typing import Dict

# [Req] Core improvelib imports
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics

# [Req] Application-specific imports
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from model_params_def import train_params

# [MODEL] Model-specific imports, as needed
import torch
import torch.nn as nn
import numpy as np
import os
from fast_synergy_dataset import FastSynergyDataset
from codes.models.PRODeepSyn_datasets import FastTensorDataLoader
from codes.models.PRODeepSyn_utils import save_best_model, find_best_model, random_split_indices
from model_definition import DNN

filepath = Path(__file__).resolve().parent # [Req]

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------   
def read_map(map_file):
    d = {}
    with open(map_file, 'r') as f:
        f.readline()
        for line in f:
            k, v = line.rstrip().split('\t')
            d[k] = int(v)
    return d
    
def create_model(data, hidden_size, device):
    # model = DNN(data.cell_feat_len() + 2 * data.drug_feat_len(), hidden_size)
    model = DNN(data.drug_feat1_len(), data.drug_feat2_len(), data.drug_feat2_len(), data.cell_feat_len(), hidden_size)
    model.to(device)
    return model

def step_batch(model, batch, loss_func, device, train=True):
    batch = [x.to(device).float() for x in batch]  # Move batch tensors to device
    drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = batch

    if train:
        y_pred = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
    else:
        yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats)
        yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2, drug1_feats3, cell_feats)
        y_pred = (yp1 + yp2) / 2

    loss = loss_func(y_pred, y_true)
    return loss

def train_epoch(model, loader, loss_func, optimizer, device):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(loader):
        optimizer.zero_grad()
        loss = step_batch(model, batch, loss_func, device)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss

def eval_epoch(model, loader, loss_func, device):
    model.eval()
    with torch.no_grad():
        epoch_loss = 0
        for batch in loader:
            loss = step_batch(model, batch, loss_func, device, train=False)
            epoch_loss += loss.item()
    return epoch_loss

def train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, device,
                sl=False, mdl_dir=None):
    min_loss = float('inf')
    angry = 0
    for epoch in range(1, n_epoch + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, device)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, device)
        val_loss /= valid_loader.dataset_len

        if val_loss < min_loss:
            angry = 0
            min_loss = val_loss
            if sl:
                save_best_model(model.state_dict(), mdl_dir, epoch, keep=1)
        else:
            angry += 1
            if angry >= patience:
                break

    if sl:
        model.load_state_dict(torch.load(find_best_model(mdl_dir)))
    return min_loss

def eval_model(model, optimizer, loss_func, train_data, test_data,
               batch_size, n_epoch, patience, device, mdl_dir):
    tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(tr_indices), batch_size=batch_size, shuffle=True)
    valid_loader = FastTensorDataLoader(*train_data.tensor_samples(es_indices), batch_size=len(es_indices) // 4)
    test_loader = FastTensorDataLoader(*test_data.tensor_samples(), batch_size=len(test_data) // 4)

    train_model(model, optimizer, loss_func, train_loader, valid_loader, n_epoch, patience, device,
                sl=True, mdl_dir=mdl_dir)

    test_loss = eval_epoch(model, test_loader, loss_func, device)
    test_loss /= len(test_data)
    return test_loss

# [Req]
def run(params: Dict):
    """ Run model training.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on validation data
            according to the metrics_list.
    """
    # --------------------------------------------------------------------
    # [Req] Create data names for train/val sets and build model path
    # --------------------------------------------------------------------
    train_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_file_name(data_format=params["data_format"], stage="val")  # [Req]

    modelpath = frm.build_model_path(
        model_file_name=params["model_file_name"],
        model_file_format=params["model_file_format"],
        model_dir=params["output_dir"])

    # --------------------------------------------------------------------
    # Load model input data (ML data) for train and val
    # --------------------------------------------------------------------
    train_data = FastSynergyDataset.load_from_npy(load_from=params["input_dir"], 
                                                 file_name=train_data_fname,
                                                 params=params)
    valid_data = FastSynergyDataset.load_from_npy(load_from=params["input_dir"], 
                                                 file_name=val_data_fname,
                                                 params=params)

    # tr_indices, es_indices = random_split_indices(len(train_data), test_rate=0.1)
    train_loader = FastTensorDataLoader(*train_data.tensor_samples(), batch_size=params["batch_size"], shuffle=True)
    valid_loader = FastTensorDataLoader(*valid_data.tensor_samples(), batch_size=len(valid_data))
    # --------------------------------------------------------------------
    # CUDA/CPU device, as needed
    # --------------------------------------------------------------------
    # Set the device
    device = torch.device(params["cuda_name"] if torch.cuda.is_available() else "cpu")
    # --------------------------------------------------------------------
    # Prepare model
    # --------------------------------------------------------------------
    model = create_model(train_data, params["hidden"], device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
    loss_func = nn.MSELoss(reduction='sum')
    # --------------------------------------------------------------------
    # Train. Iterate over epochs.
    # --------------------------------------------------------------------
    print("Training model...")
    min_loss = float('inf')
    best_epoch = None
    for epoch in range(1, params["epochs"] + 1):
        trn_loss = train_epoch(model, train_loader, loss_func, optimizer, device)
        trn_loss /= train_loader.dataset_len
        val_loss = eval_epoch(model, valid_loader, loss_func, device)
        val_loss /= valid_loader.dataset_len
        if epoch % 100 == 0: 
            print("epoch: {} | train loss: {} valid loss {}".format(epoch, trn_loss, val_loss))
        if val_loss < min_loss:
            min_loss = val_loss
            best_epoch = epoch
            save_best_model(model.state_dict(), params["output_dir"], epoch, keep=1)
            
    # Rename the best model after training completes
    if best_epoch is not None:
        best_model_path = os.path.join(params["output_dir"], f"{best_epoch}.pkl")

        if os.path.exists(modelpath):
            os.remove(modelpath)  # Remove old best model if it exists

        os.rename(best_model_path, modelpath)
        print(f"Best model from epoch {best_epoch} saved as {modelpath}")
    # --------------------------------------------------------------------
    # Load best model and compute predictions
    # --------------------------------------------------------------------
    model.load_state_dict(torch.load(modelpath))
    with torch.no_grad():
        for val_each in valid_loader:
            val_each = [x.to(device).float() for x in val_each]
            drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2, drug2_feats3, cell_feats, y_true = val_each
            yp1 = model(drug1_feats1, drug1_feats2, drug1_feats3, drug2_feats1, drug2_feats2,drug2_feats3, cell_feats)
            yp2 = model(drug2_feats1, drug2_feats2, drug2_feats3, drug1_feats1, drug1_feats2,drug1_feats3, cell_feats)
            y_pred = (yp1 + yp2) / 2
            y_pred = y_pred.cpu().numpy().flatten()
            y_true = y_true.cpu().numpy().flatten()
    # --------------------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # --------------------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_true,
        y_pred=y_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"],
        input_dir=params["input_dir"]
    )

    # --------------------------------------------------------------------
    # [Req] Compute performance scores
    # --------------------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=y_true,
        y_pred=y_pred,
        stage="val",
        metric_type=params["metric_type"],
        output_dir=params["output_dir"]
    )

    return val_scores


# [Req]
def main(args):
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="mgaedc_original_params.txt",
        additional_definitions=train_params)
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])