# MGAE-DC: Drug Synergy Prediction
This branch reproduces the implementation of MGAE-DC, a deep learning framework for predicting drug combination synergistic effects using multi-channel graph autoencoders.

## Dependencies and Installation
### Conda Environment
```
conda create -n mgaedc python=3.8
conda activate mgaedc
conda install pandas=1.3.5 numpy=1.21.2 tensorflow-gpu=2.4.1
conda install networkx scikit-learn matplotlib pytorch
```

### Clone Repository
```
git clone https://github.com/JDACS4C-IMPROVE/MGAE-DC.git
cd MGAE-DC
git checkout original-reproduce
```

### Download Original Data

The original data files necessary for this implementation are provided in this repository. Please refer to the rawdata/ directory (or specify the exact path) for access.

## Running the Model

### Extracting Drug Embeddings with Multi-Channel Graph Autoencoders

The following script extracts cell line-specific and common drug embeddings using multi-channel graph autoencoders from the embedding module.

Usage:
```
python codes/get_oneil_mgaedc_representation.py -learning_rate 0.001 -epochs 10000 -embedding_dim 320 -dropout 0.2 -weight_decay 0 -val_test_size 0.1
```  

Command-Line Arguments:
| Argument          | Default  | Description                                      |
|------------------|---------|--------------------------------------------------|
| `-learning_rate` | 0.001   | Initial learning rate.                           |
| `-epochs`        | 10000   | Number of training epochs.                       |
| `-embedding_dim` | 320     | Number of dimensions for the embedding.          |
| `-dropout`       | 0.2     | Dropout rate (1 - keep probability).             |
| `-weight_decay`  | 0       | Weight for L2 loss on the embedding matrix.      |
| `-val_test_size` | 0.1     | Proportion of validation and test samples.       |


### Predicting Synergistic Effects of Drug Combinations

The following script is used to predict the synergistic effects of drug combinations in the predictor module.

Usage:
```
python codes/get_oneil_mgaedc.py -learning_rate 0.01 -epochs 500 -batch 320 -drop_out 0.2 -hidden 8192 -patience 100 
```

Command-Line Arguments:
| Argument      | Default                          | Description                    |
|--------------|----------------------------------|--------------------------------|
| `--epoch`    | 1                                | Number of training epochs.     |
| `--batch`    | 256                              | Batch size.                    |
| `--gpu`      | 1                                | CUDA device ID.                |
| `--patience` | 100                              | Patience for early stopping.   |
| `--suffix`   | `results_oneil_mgaedc100_folds`  | Model directory suffix.        |
| `--hidden`   | `[2048, 4096, 8192]`             | Hidden layer sizes.            |
| `--lr`       | `[1e-3, 1e-4, 1e-5]`             | Learning rate values.    


## References

Original GitHub: https://github.com/yushenshashen/MGAE-DC

Original Paper: https://doi.org/10.1371/journal.pcbi.1010951

If you use this repository in your research or projects, please cite the original work:
```   
Zhang P, Tu S (2023) MGAE-DC: Predicting the synergistic effects of drug combinations through multi-channel graph autoencoders. PLoS Comput Biol 19(3): e1010951. https://doi.org/10.1371/journal.pcbi.1010951
```
