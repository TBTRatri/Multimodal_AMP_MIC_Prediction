"""
esm_embeddings.py
Generates mean-pooled 320-dimensional sequence embeddings using ESM-2 (ESM2-T6-8M-UR50D).
"""

import torch
import esm
import numpy as np

class ESMFeatureExtractor:
    def __init__(self, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load ESM-2 8M model (6 layers, 320 embedding dimensions)
        self.model, self.alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_embedding(self, sequence):
        """Extracts 320-D mean-pooled representation for a single peptide sequence."""
        data = [("protein", sequence.strip().upper())]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[6], return_contacts=False)
            token_representations = results["representations"][6]

        # Exclude start/end tokens and mean pool over residues
        seq_len = len(sequence)
        seq_rep = token_representations[0, 1 : seq_len + 1].mean(0).cpu().numpy()
        return seq_rep.astype(np.float32)

    def extract_batch(self, sequences):
        """Extracts embeddings for a list of sequences."""
        return np.vstack([self.get_embedding(seq) for seq in sequences])
