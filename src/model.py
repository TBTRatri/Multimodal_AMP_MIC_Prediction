"""
model.py
PyTorch implementation of the Biophysical Fusion Architecture and Ensemble Wrapper.
"""

import torch
import torch.nn as nn
import numpy as np

class BiophysicalFusionExpert(nn.Module):
    """
    Biophysical Fusion Network consisting of:
    - Sequence Branch (320-D ESM-2 embedding)
    - Biophysical Branch (12 normalized descriptors)
    - Species Embedding Branch
    - Fusion Regression Head
    """
    def __init__(self, num_species=300, species_emb_dim=32, esm_dim=320, bio_dim=12):
        super(BiophysicalFusionExpert, self).__init__()
        
        # 1. Species Branch
        self.species_embedding = nn.Embedding(num_species, species_emb_dim)
        
        # 2. Sequence (ESM) Branch
        self.esm_branch = nn.Sequential(
            nn.Linear(esm_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 3. Biophysical Branch
        self.bio_branch = nn.Sequential(
            nn.Linear(bio_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 4. Fusion Layer & Regression Head
        fusion_dim = 128 + 32 + species_emb_dim
        self.regression_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predicts log10(MIC)
        )

    def forward(self, esm_x, bio_x, species_idx):
        h_esm = self.esm_branch(esm_x)
        h_bio = self.bio_branch(bio_x)
        h_species = self.species_embedding(species_idx)
        
        # Multimodal Concatenation
        h_fusion = torch.cat([h_esm, h_bio, h_species], dim=1)
        out = self.regression_head(h_fusion)
        return out.squeeze(-1)


class BiophysicalFusionEnsemble:
    """Ensemble wrapper managing 5 independently trained experts."""
    def __init__(self, models):
        self.models = models
        for m in self.models:
            m.eval()

    def predict(self, esm_x, bio_x, species_idx):
        predictions = []
        with torch.no_grad():
            for m in self.models:
                pred = m(esm_x, bio_x, species_idx)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)  # Shape: (5, N)
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        return mean_pred, uncertainty
