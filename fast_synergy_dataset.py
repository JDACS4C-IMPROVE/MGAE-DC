import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os

# -------------------------
# Helper Functions
# -------------------------

def to_numpy(tensor_data):
    """Converts PyTorch tensor to NumPy array if needed."""
    if isinstance(tensor_data, torch.Tensor):
        return tensor_data.numpy()
    return np.array(tensor_data, dtype=np.float32)

def process_sample(drug1, drug2, cellname, score, drug_feat1, drug_feat2, drug_feat3, cell_feat, drugslist, cellslist):
    """Creates a sample from drug and cell features."""
    return [
        torch.from_numpy(np.asarray(drug_feat1[drugslist.index(drug1)])),
        torch.from_numpy(np.asarray(drug_feat2[drugslist.index(drug1)])),
        torch.from_numpy(np.asarray(drug_feat3[cellname][drugslist.index(drug1)])),
        torch.from_numpy(np.asarray(drug_feat1[drugslist.index(drug2)])),
        torch.from_numpy(np.asarray(drug_feat2[drugslist.index(drug2)])),
        torch.from_numpy(np.asarray(drug_feat3[cellname][drugslist.index(drug2)])),
        torch.from_numpy(np.asarray(cell_feat[cellslist.index(cellname)])),
        torch.tensor([float(score)], dtype=torch.float32)
    ]

def save_metadata(output_dir, drugslist, cellslist):
    """Saves drugslist and cellslist as a JSON file."""
    metadata = {"drugslist": drugslist, "cellslist": cellslist}
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f)

def load_metadata(load_from):
    """Loads drugslist and cellslist from a JSON file."""
    with open(f"{load_from}/metadata.json", "r") as f:
        metadata = json.load(f)
    return metadata["drugslist"], metadata["cellslist"]

# -------------------------
# FastSynergyDataset Class
# -------------------------

class FastSynergyDataset(Dataset):
    def __init__(self, drug_feat, drug_feat_topology_common, drug_feat_topology_specifics, 
                 cell_feat, synergy_data, params, train=True, load_from=None, file_name=None):
        """
        Initialize the dataset.

        If `load_from` is provided, it loads the dataset from `.npy` files.
        Otherwise, it uses the provided in-memory drug and cell features.

        Args:
            drug_feat (np.ndarray): Primary drug feature matrix.
            drug_feat_topology_common (dict): Common topology features per drug.
            drug_feat_topology_specifics (dict): Specific topology features per cell line.
            cell_feat (np.ndarray): Cell feature matrix.
            synergy_data (pd.DataFrame): DataFrame containing synergy scores with columns [drug1, drug2, cell_line, score].
            params (dict): Dictionary of parameter values.
            train (bool): Whether to include augmented (drug2-drug1) pairs.
            load_from (str): Directory path to load saved `.npy` files.
            file_name (str): Name of `.npy` file.
        """
        self.params = params
        self.samples = []
        self.raw_samples = []
        self.train = train
        self.drug1 = params['drug_col_name_1']
        self.drug2 = params['drug_col_name_2']
        self.cell_line = params['canc_col_name']
        self.score = params['y_col_name']
        
        if load_from:  # Load from pre-saved .npy files
            print(f"Loading dataset from {load_from}...")
            self.samples = np.load(f"{load_from}/{file_name}", allow_pickle=True)            
            # Load cell and drug mappings
            self.drugslist, self.cellslist = load_metadata(load_from)
            
            # Convert NumPy arrays to float32 tensors
            self.samples = [
                [torch.from_numpy(np.asarray(feature, dtype=np.float32)) if isinstance(feature, np.ndarray) else feature for feature in sample]
                for sample in self.samples
            ]

            print(f"Loaded {file_name} samples from {load_from}.")
            
            self.drug_feat1 = self.samples[0][0]
            self.drug_feat2 = self.samples[0][1]
            self.drug_feat3 = self.samples[0][2]
            self.cell_feat = self.samples[0][-2]
        else:  # Use provided data (no file loading)
            self.drug_feat1 = drug_feat
            self.drug_feat2 = drug_feat_topology_common
            self.drug_feat3 = drug_feat_topology_specifics
            self.cell_feat = cell_feat

            self.drugslist = sorted(set(synergy_data[self.drug1]).union(set(synergy_data[self.drug2])))
            self.cellslist = sorted(set(synergy_data[self.cell_line]))
            
            valid_drugs = set(self.drugslist)
            valid_cells = set(self.cellslist)

            for _, row in synergy_data.iterrows():
                drug1, drug2, cellname, score = row[[self.drug1, self.drug2, self.cell_line, self.score]]
                if drug1 in valid_drugs and drug2 in valid_drugs and cellname in valid_cells:
                    sample = process_sample(drug1, drug2, cellname, score, 
                                            self.drug_feat1, self.drug_feat2, 
                                            self.drug_feat3, self.cell_feat, 
                                            self.drugslist, self.cellslist)
                    self.samples.append(sample)
                    self.raw_samples.append([self.drugslist.index(drug1), self.drugslist.index(drug2), 
                                             self.cellslist.index(cellname), float(score)])

                    if train:  # Data augmentation (reverse order)
                        sample = process_sample(drug2, drug1, cellname, score, 
                                                self.drug_feat1, self.drug_feat2, 
                                                self.drug_feat3, self.cell_feat, 
                                                self.drugslist, self.cellslist)
                        self.samples.append(sample)
                        self.raw_samples.append([self.drugslist.index(drug2), self.drugslist.index(drug1), 
                                                 self.cellslist.index(cellname), float(score)])

    def save_to_npy(self, output_dir, stage_fname):
        """Saves processed dataset to .npy files along with metadata."""
        # Construct file names with stage prefix
        samples_fpath = os.path.join(output_dir, stage_fname)
        np.save(samples_fpath, np.array(self.samples, dtype=object))
        
        # Save metadata (drug & cell mappings)
        save_metadata(output_dir, self.drugslist, self.cellslist)

    @classmethod
    def load_from_npy(cls, load_from, file_name, params):
        """Class method to initialize dataset directly from `.npy` files.

        Args:
            load_from (str): Directory where the `.npy` files are stored.
            file_name (str): Name of the `.npy` file to load.
            params (dict, optional): Dictionary of parameters used in processing.

        Returns:
            FastSynergyDataset: An instance of the dataset.
        """
        return cls(drug_feat=None, 
                drug_feat_topology_common=None, 
                drug_feat_topology_specifics=None, 
                cell_feat=None, 
                synergy_data=None,
                load_from=load_from, 
                file_name=file_name, 
                params=params)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

    def drug_feat1_len(self):
        return self.drug_feat1.shape[-1]

    def drug_feat2_len(self):
        return self.drug_feat2.shape[-1]

    def drug_feat3_len(self):
        return self.drug_feat2.shape[-1]

    def cell_feat_len(self):
        return self.cell_feat.shape[-1]

    def tensor_samples(self, indices=None):
        if indices is None:
            indices = list(range(len(self)))
        d1_f1 = torch.cat([torch.unsqueeze(self.samples[i][0], 0) for i in indices], dim=0)
        d1_f2 = torch.cat([torch.unsqueeze(self.samples[i][1], 0) for i in indices], dim=0)
        d1_f3 = torch.cat([torch.unsqueeze(self.samples[i][2], 0) for i in indices], dim=0)
        d2_f1 = torch.cat([torch.unsqueeze(self.samples[i][3], 0) for i in indices], dim=0)
        d2_f2 = torch.cat([torch.unsqueeze(self.samples[i][4], 0) for i in indices], dim=0)
        d2_f3 = torch.cat([torch.unsqueeze(self.samples[i][5], 0) for i in indices], dim=0)
        c = torch.cat([torch.unsqueeze(self.samples[i][6], 0) for i in indices], dim=0)
        y = torch.cat([torch.unsqueeze(self.samples[i][7], 0) for i in indices], dim=0)
        return d1_f1, d1_f2, d1_f3, d2_f1, d2_f2, d2_f3, c, y
