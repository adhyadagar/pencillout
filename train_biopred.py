import numpy as np
import argparse
import os
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from torch.nn.functional import sigmoid
from torch.utils.data import DataLoader, Dataset

from conplex import ConplexModel

conplex_model = ConplexModel()
 
# Want to train a model per task.

# Also, want to train 3 different model architectures, and take the average bioactivity prediction probability across the three.
class MolDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def get_fingerprint(smiles):
    mol = Chem.MolFromSmiles(smiles)
    fpt = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return np.array(fpt)

def get_conplex(smiles):
    return conplex_model.embed(smiles)

# Define a model class for each architecture
class MLPModel(nn.Module):
    def __init__(self, input_size):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.elu1 = nn.ELU()
        self.fc2 = nn.Linear(256, 1)


    def fit(self, smiles_train, y_train):
        """
        This function trains on smiles and bioactivity data.
        """
        # compute conplex embeddings for X_train
        conplex_embeddings = get_conplex(smiles_train)
        X_train = torch.tensor([get_fingerprint(smi) for smi in smiles_train], dtype=torch.float32)

        X_train = np.concatenate([X_train, conplex_embeddings], axis=1)
        train_dataset = MolDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

        optimizer = torch.optim.AdamW(self.parameters(), lr=0.00005)
        criterion = nn.BCEWithLogitsLoss()
        self.train()
        for epoch in range(15):
            for data, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(data).reshape(-1)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()
        self.eval()

    def forward(self, x):
        x = self.elu1(self.fc1(x))
        return self.fc2(x)

# This is a meta model that uses an MLP, random forest, and logistic regression model to predict bioactivity (one of antibacterial, antifungal, antiviral, anticancer) and toxicity.
class MetaModel():
    def __init__(self):
        self.mlp = MLPModel(2048)
        self.lr = LogisticRegression()
        self.rf = RandomForestClassifier()

    def fit(self, smiles_train, y_train):
        """
        This function trains on smiles and bioactivity data
        """

        self.mlp.fit(smiles_train, y_train)

        conplex_embeddings = get_conplex(smiles_train)
        X_train = torch.tensor([get_fingerprint(smi) for smi in smiles_train], dtype=torch.float32)
        combined_train = np.concatenate([X_train, conplex_embeddings], axis=1)
        self.lr.fit(combined_train, y_train)

        self.rf.fit(X_train, y_train)

    def predict(self, smiles_test):
        # get conplex embeddings
        conplex_embeddings = get_conplex(smiles_test)

        # get morgan embeddings
        X_test = np.array([get_fingerprint(smi) for smi in smiles_test])

        combined_test = np.concatenate([X_test, conplex_embeddings], axis=1)

        # take softmax for prob for mlp
        mlp_preds = self.mlp(torch.tensor(combined_test, dtype=torch.float32))
        mlp_preds = sigmoid(mlp_preds).detach().cpu().numpy().reshape(-1)

        # predict proba for lr and rf
        lr_preds = np.array(self.lr.predict_proba(combined_test)[:, 1]).reshape(-1)
        rf_preds = np.array(self.rf.predict_proba(X_test)[:, 1]).reshape(-1)

        # return the average probability across the 3 methods
        return (mlp_preds + lr_preds + rf_preds) / 3

# Example usage
def __main__():
    train_smiles = ["CCO", "CCN", "CCO"]
    train_y = torch.tensor([1, 0, 1])

    test_smiles = ["CCO", "CCN", "CCO"]
    test_y = torch.tensor([1, 0, 1])

    model = MetaModel()
    model.fit(train_smiles, train_y)
    preds = model.predict(test_smiles)

    print(preds)

if __name__ == "__main__":
    __main__()
