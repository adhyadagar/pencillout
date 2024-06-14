
from flask import Flask, jsonify, request
import joblib
from conplex import ConplexModel  # This assumes you need the ConplexModel for processing

app = Flask(__name__)

# Load the saved model
model = joblib.load('antibacterial_model.joblib')

# Assume ConplexModel is used to generate embeddings
embedding_model = ConplexModel()

smiles_string = "CCC(C)C1N=CC(CC=2C=3C(C(O)C=CC=3)NC=2)NC(=O)C(C(C)C)NC(=O)C(CC(N)=O)NC(=O)C(C(CC)C)NC(=O)C(CCC(N)=O)NC1=O"
print("SMILES string:", smiles_string)
# Generate embedding
smiles_string = smiles_string
# embedding = embedding_model.embed([smiles_string])
# Reshape embedding if necessary, assuming model expects 2D array
# embedding = embedding.reshape(1, -1)
# Make prediction
prediction = model.predict([smiles_string]) # Get probability of the positive class
print("Prediction:", prediction)