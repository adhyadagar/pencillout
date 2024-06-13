from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from functools import lru_cache
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return Chem.MolToSmiles(mol, isomericSmiles=True)
    else:
        return None

class ConplexModel:
    def __init__(self):
        loaded_data = torch.load('drug_projector_weights.pth')
        loaded_weights = loaded_data['weights']
        loaded_bias = loaded_data['bias']

        self.drug_projector = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU()
        )

        self.drug_projector[0].weight.data.copy_(loaded_weights)
        self.drug_projector[0].bias.data.copy_(loaded_bias)

    def smiles_to_morgan(self, smile: str):
        """
        Convert smiles into Morgan Fingerprint.
        :param smile: SMILES string
        :type smile: str
        :return: Morgan fingerprint
        :rtype: np.ndarray
        """
        try:
            smile = canonicalize(smile)
            mol = Chem.MolFromSmiles(smile)
            features_vec = AllChem.GetMorganFingerprintAsBitVect(
                mol, 2, nBits=2048
            )
            features = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(features_vec, features)
        except Exception as e:
            features = np.zeros((2048,))
        return features


    def embed(self, smiles_list):
        feats = torch.Tensor([self.smiles_to_morgan(smi) for smi in smiles_list])
        return self.drug_projector(feats).detach().cpu().numpy()

# smiles_list = np.array(["CCO", "CCN", "CCO", "ClC3=CC=C2NC=C(C2=C3)CC1NC(=O)C(C(O)C(N)=O)NC(=O)C(CCCNC(=O)C(C(C)C)NC(=O)C(CCC(C)C)NC(=O)C(C(C)C)NC1=O)N"])
# model = ConplexModel()
# with torch.set_grad_enabled(False):
#     vals = model.embed(smiles_list)
#     results = [np.sum(x) for x in model.embed(smiles_list)]
