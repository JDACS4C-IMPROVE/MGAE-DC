import numpy as np
import pandas as pd
import os

output_dir = "improve_oneil"

# --------------------------------------------------------------------
# Prepare response data
# --------------------------------------------------------------------
# Read in response data
loewe_df = pd.read_csv("rawdata/oneil_loewe_cutoff30.txt", sep="\t")
bliss_df = pd.read_csv("rawdata/oneil_synergy_bliss.txt", sep="\t")
hsa_df = pd.read_csv("rawdata/oneil_synergy_hsa.txt", sep="\t")
zip_df = pd.read_csv("rawdata/oneil_synergy_zip.txt", sep="\t")

# Create list of response dataframes
dfs = [loewe_df, bliss_df, hsa_df, zip_df]

# Standardize column names before merging
for i, df in enumerate(dfs):
    df.rename(columns={"synergy": f"synergy_{i}"}, inplace=True)  # Rename synergy columns

# Merge all dataframes on common columns (outer join to retain all data)
merged_df = dfs[0]
for df in dfs[1:]:
    df = df.drop(columns=["fold"])
    merged_df = pd.merge(merged_df, df, on=["drugname1", "drugname2", "cell_line"], how="outer")

# Rename columns to match the required format
merged_df.rename(columns={
    "drugname1": "improve_chem_id_1",
    "drugname2": "improve_chem_id_2",
    "cell_line": "improve_sample_id",
    "synergy_0": "loewe",
    "synergy_1": "bliss",
    "synergy_2": "hsa",
    "synergy_3": "zip",
}, inplace=True)

# Add missing columns
merged_df["source"] = "ONEIL"
merged_df["study"] = "fake_exp"  # Replace with actual study name if available

# Reorder columns to match desired format
final_columns = ["source", "improve_sample_id", "improve_chem_id_1", "improve_chem_id_2", "study", "loewe", "bliss", "hsa", "zip", "fold"]
merged_df = merged_df[final_columns]

# Drop the fold column
merged_df_no_fold = merged_df.drop(columns=["fold"])

# Ensure the y_data directory exists
response_dir = f"{output_dir}/y_data"
os.makedirs(response_dir, exist_ok=True)

# Save as TSV
rsp_file = os.path.join(response_dir, "response.tsv")

# Save as TSV
merged_df_no_fold.to_csv(rsp_file, sep="\t", index=False)

print("Saved to y_data/: response.tsv")

# --------------------------------------------------------------------
# Prepare splits files
# --------------------------------------------------------------------
# Ensure the splits directory exists
splits_dir = f"{output_dir}/splits"
os.makedirs(splits_dir, exist_ok=True)

# Define the test fold manually based on the original code
test_fold = 0
valid_fold = list(range(10))[test_fold-1]
train_folds = [ x for x in list(range(10)) if x != test_fold and x != valid_fold ]

# Get row indices for each split
test_indices = merged_df[merged_df["fold"] == test_fold].index.tolist()
valid_indices = merged_df[merged_df["fold"] == valid_fold].index.tolist()
train_indices = merged_df[merged_df["fold"].isin(train_folds)].index.tolist()

# Save to text files in splits folder
with open(os.path.join(splits_dir, f"ONEIL_split_{test_fold}_test.txt"), "w") as f:
    f.write("\n".join(map(str, test_indices)))

with open(os.path.join(splits_dir, f"ONEIL_split_{test_fold}_val.txt"), "w") as f:
    f.write("\n".join(map(str, valid_indices)))

with open(os.path.join(splits_dir, f"ONEIL_split_{test_fold}_train.txt"), "w") as f:
    f.write("\n".join(map(str, train_indices)))

print(f"Saved to splits/: ONEIL_split_{test_fold}_test.txt, ONEIL_split_{test_fold}_val.txt, ONEIL_split_{test_fold}_train.txt")

# --------------------------------------------------------------------
# Prepare drug data
# --------------------------------------------------------------------
# Ensure the x_data directory exists
x_data_dir = f"{output_dir}/x_data"
os.makedirs(x_data_dir, exist_ok=True)

# Load the SMILES file
drug_smiles_df = pd.read_csv('rawdata/oneil_drug_smiles.txt',sep='\t', header=None)

# Rename columns (assuming first column is improve_chem_id and second is canSMILES)
drug_smiles_df = drug_smiles_df.rename(columns={0: "improve_chem_id", 1: "canSMILES"})

# Save as TSV file
smiles_file = os.path.join(x_data_dir, "drug_SMILES.tsv")
drug_smiles_df.to_csv(smiles_file, sep="\t", index=False, header=True)

print(f"Saved to x_data: drug_SMILES.tsv")

# Load the infomax .npy file
drug_data = np.load("rawdata/oneil_drug_feat.npy")

# Convert to Pandas dataframe
drug_df = pd.DataFrame(drug_data)

# Generate new column names as "infomax.1", "infomax.2", ..., "infomax.N"
num_columns = drug_df.shape[1]  # Get total number of columns
new_column_names = [f"infomax.{i+1}" for i in range(num_columns)]  # Start from 1

# Assign new column names
drug_df.columns = new_column_names

# Concatenate improve_chem_id from drug_smiles_df to drug_df
drug_infomax_df = pd.concat([drug_smiles_df[["improve_chem_id"]], drug_df], axis=1)

# Save as TSV file
infomax_file = os.path.join(x_data_dir, "drug_infomax.tsv")
drug_infomax_df.to_csv(infomax_file, sep="\t", index=False, header=True)

print(f"Saved to x_data: drug_infomax.tsv")

# --------------------------------------------------------------------
# Prepare omics data
# --------------------------------------------------------------------
# Load cell info file from PROSynDeep - same as the one in MGAE-DC
cell_id_df = pd.read_csv('rawdata/cell2id.tsv',sep='\t')

# Load the cell features .npy file used in the code
cell_data = np.load("rawdata/oneil_cell_feat.npy")

# Convert to Pandas dataframe
cell_feat_df = pd.DataFrame(cell_data)

# Create sorted list of names
cells_list = sorted(cell_id_df.cell.tolist())  # Replace with your actual sorted list

# Concatenate with existing dataframe (adds to the left)
cells_df = pd.concat([pd.DataFrame(cells_list, columns=["improve_sample_id"]), cell_feat_df], axis=1)

# Extract the number of gene-related columns (excluding 'improve_sample_id')
num_gene_cols = cells_df.shape[1] - 1  # Subtracting 1 for the sample ID column

# Generate fake Gene IDs, Entrez IDs, and Gene Names
fake_gene_ids = [f"ENSG00000000{str(i).zfill(3)}" for i in range(num_gene_cols)]
fake_entrez_ids = np.random.randint(5000, 70000, size=num_gene_cols)
fake_gene_names = [f"GENE{i}" for i in range(num_gene_cols)]

# Create MultiIndex Columns for gene expression data
multi_index_columns = pd.MultiIndex.from_arrays(
    [fake_gene_ids,  # First level: Gene IDs
     fake_entrez_ids,  # Second level: Entrez IDs
     fake_gene_names],  # Third level: Gene Names
    names=["Gene ID", "Entrez ID", "Gene Name"]
)

# Reassign the column headers while keeping the first column unnamed
cells_df.columns = pd.MultiIndex.from_tuples([("", "", "")] + list(zip(fake_gene_ids, fake_entrez_ids, fake_gene_names)))

# Save the merged DataFrame to a TSV file in the x_data folder
gene_file = os.path.join(x_data_dir, 'cancer_gene_expression.tsv')
cells_df.to_csv(gene_file, sep='\t', index=False)

print(f"Saved to x_data: cancer_gene_expression.tsv")