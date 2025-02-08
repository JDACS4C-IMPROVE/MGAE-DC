import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------
# Model architecture
# --------------------------------------------------------------------
class DNN(nn.Module):
    def __init__(self, drug_feat1_len:int,  drug_feat2_len:int, drug_feat3_len:int, cell_feat_len:int, hidden_size: int):
        super(DNN, self).__init__()

        self.drug_network1 = nn.Sequential(
            nn.Linear(drug_feat1_len, drug_feat1_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat1_len*2),
            nn.Linear(drug_feat1_len*2, drug_feat1_len),
        )

        self.drug_network2 = nn.Sequential(
            nn.Linear(drug_feat2_len, drug_feat2_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat2_len*2),
            nn.Linear(drug_feat2_len*2, drug_feat2_len),
        )

        self.drug_network3 = nn.Sequential(
            nn.Linear(drug_feat3_len, drug_feat3_len*2),
            nn.ReLU(),
            nn.BatchNorm1d(drug_feat3_len*2),
            nn.Linear(drug_feat3_len*2, drug_feat3_len),
        )

        self.cell_network = nn.Sequential(
            nn.Linear(cell_feat_len, cell_feat_len),
            nn.ReLU(),
            nn.BatchNorm1d(cell_feat_len ),
            nn.Linear(cell_feat_len, 768),
        )

        self.fc_network = nn.Sequential(
            nn.BatchNorm1d(2*(drug_feat1_len + drug_feat2_len + drug_feat3_len)+ 768),
            nn.Linear(2*(drug_feat1_len + drug_feat2_len + drug_feat3_len)+ 768, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, drug1_feat1: torch.Tensor, drug1_feat2: torch.Tensor, drug1_feat3: torch.Tensor, drug2_feat1: torch.Tensor, drug2_feat2: torch.Tensor, drug2_feat3: torch.Tensor, cell_feat: torch.Tensor):
        drug1_feat1_vector = self.drug_network1( drug1_feat1 ) 
        drug1_feat2_vector = self.drug_network2( drug1_feat2 )
        drug1_feat3_vector = self.drug_network3( drug1_feat3 )
        drug2_feat1_vector = self.drug_network1( drug2_feat1 ) 
        drug2_feat2_vector = self.drug_network2( drug2_feat2 )
        drug2_feat3_vector = self.drug_network3( drug2_feat3 )
        cell_feat_vector = self.cell_network(cell_feat)
        # cell_feat_vector = cell_feat
        feat = torch.cat([drug1_feat1_vector, drug1_feat2_vector,drug1_feat3_vector , drug2_feat1_vector, drug2_feat2_vector, drug2_feat3_vector, cell_feat_vector], 1)
        out = self.fc_network(feat)
        return out