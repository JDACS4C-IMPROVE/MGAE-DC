from improvelib.utils import str2bool

preprocess_params = [
    {
        "name": "learning_rate",
        "type": float,
        "default": 0.001,
        "help": "Learning rate for model training."
    },
    {
        "name": "epochs",
        "type": int,
        "default": 10000,
        "help": "Number of training epochs."
    },
    {
        "name": "dropout",
        "type": float,
        "default": 0.2,
        "help": "Dropout rate to prevent overfitting."
    },
    {
        "name": "embedding_dim",
        "type": int,
        "default": 320,
        "help": "Dimension of the embedding layer."
    },
    {
        "name": "embeddings_dir",
        "type": str,
        "default": "embeddings/",
        "help": "Directory to save the trained embeddings."
    },
]


train_params = [
    {
        "name": "hidden",
        "type": int,
        "default": 8192,
        "help": "Hidden layer size for the neural network.",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference (e.g., 'cuda:0', 'cuda:1'). Defaults to 'cuda:0'.",
    },
    {
        "name": "drug_col_name_1",
        "type": str,
        "default": "improve_chem_id_1",
        "help": "Column name for the first drug identifier in the dataset."
    },
    {
        "name": "drug_col_name_2",
        "type": str,
        "default": "improve_chem_id_2",
        "help": "Column name for the second drug identifier in the dataset."
    },
    {
        "name": "canc_col_name",
        "type": str,
        "default": "improve_sample_id",
        "help": "Column name for the cell line/sample identifier in the dataset."
    }
]


infer_params = [
    {
        "name": "hidden",
        "type": int,
        "default": 8192,
        "help": "Hidden layer size for the neural network.",
    },
    {
        "name": "cuda_name",
        "type": str,
        "default": "cuda:0",
        "help": "Specify the CUDA device to use for model training and inference (e.g., 'cuda:0', 'cuda:1'). Defaults to 'cuda:0'.",
    },
    {
        "name": "drug_col_name_1",
        "type": str,
        "default": "improve_chem_id_1",
        "help": "Column name for the first drug identifier in the dataset."
    },
    {
        "name": "drug_col_name_2",
        "type": str,
        "default": "improve_chem_id_2",
        "help": "Column name for the second drug identifier in the dataset."
    },
    {
        "name": "canc_col_name",
        "type": str,
        "default": "improve_sample_id",
        "help": "Column name for the cell line/sample identifier in the dataset."
    }
]