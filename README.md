# IMPROVE - MGAE-DC: Drug Synergy Prediction
This is the IMPROVE implementation of the original model with original data.

## Dependencies and Installation
### Conda Environment
```
conda create -n mgaedc_improve python=3.8
conda activate mgaedc_improve
conda install pandas=1.3.5 numpy=1.21.2 tensorflow-gpu=2.4.1
conda install networkx scikit-learn matplotlib pytorch
conda install pyyaml # IMPROVE-dependency
```

### Clone Repository
```
git clone https://github.com/JDACS4C-IMPROVE/MGAE-DC.git
cd MGAE-DC
git checkout IMPROVE-original
```

### Clone IMPROVE Repository
```
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop # default branch
cd ..
```

### Download Original Data

The original data files necessary for this implementation are provided in this repository. Please refer to the `rawdata/` directory (or specify the exact path) for access.

## Running the Model

### 1. Activate the conda environment
```
conda activate mgaedc_improve
```

### 2. Set environment variables
```
export PYTHONPATH=$PYTHONPATH:/your/path/to/IMPROVE
```

### 3. Prepare dataset in IMPROVE format
```
python prepare_oneil_data.py
```
This script processes the O'Neil datasets and reformats them to match the IMPROVE framework. The processed data is saved in the `improve_oneil` folder, following the required structure with `y_data`, `x_data`, and `splits` folders.

*Note: The cell features data originally comes from PRODeepSyn.*

### 4. Preprocess raw data to construct model input data (ML data)
```
python mgaedc_preprocess_improve.py --input_dir improve_oneil --output_dir original/ml_data
```
Preprocesses the IMPROVE-formatted data and creates train, validation (val), and test datasets.

Generates:

 * three model input data files: `train_data.npy`, `val_data.npy`, `test_data.npy`
 * three tabular data files, each containing the drug response values (i.e. Loewe) and corresponding metadata: `train_y_data.csv`, `val_y_data.csv`, `test_y_data.csv`

This script includes the training of the cell line-specific and common drug embeddings. 

### 5. Train model
```
python mgaedc_train_improve.py --input_dir original/ml_data --output_dir original/out_models
```
Trains the MGAE-DC model using the model's input data: `train_data.npy` (training), `val_data.npy` (for early stopping).

Generates:

 * trained model: `model.pkl`
 * predictions on val data (tabular data): `val_y_data_predicted.csv`
 * prediction performance scores on val data: `val_scores.json`

### 6. Run inference on test data with the trained model
```
python mgaedc_infer_improve.py --input_data_dir original/ml_data --input_model_dir original/out_models --output_dir original/out_infer --calc_infer_score true
```

Evaluates the performance on a test dataset, `train_data.npy`, with the trained model.

Generates:

 * predictions on test data (tabular data): `test_y_data_predicted.csv`
 * prediction performance scores on test data: `test_scores.json`

## References

Original GitHub: https://github.com/yushenshashen/MGAE-DC

Original Paper: https://doi.org/10.1371/journal.pcbi.1010951

If you use this repository in your research or projects, please cite the original work:
```   
Zhang P, Tu S (2023) MGAE-DC: Predicting the synergistic effects of drug combinations through multi-channel graph autoencoders. PLoS Comput Biol 19(3): e1010951. https://doi.org/10.1371/journal.pcbi.1010951
```
