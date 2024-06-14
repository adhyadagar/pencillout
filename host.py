from flask import Flask, jsonify, request
import joblib
from conplex import ConplexModel  # This assumes you need the ConplexModel for processing
import numpy as np
import chromadb
from chromadb.config import Settings


app = Flask(__name__)

# Load the saved model
model_antibacterial = joblib.load('antibacterial_model.joblib')
model_antitumour    = joblib.load('antifungal_model.joblib')
toxicity_model = joblib.load('toxicity_model.joblib')

# Assume ConplexModel is used to generate embeddings
embedding_model = ConplexModel()

@app.route('/predict', methods=['POST'])
def predict():
    # Check if JSON data is available and has the 'smiles' key
    if not request.json or 'smiles' not in request.json:
        return jsonify({'error': 'Missing SMILES data'}), 400

    smiles_string = request.json['smiles']

    # Check if the SMILES data is a string
    if not isinstance(smiles_string, str):
        return jsonify({'error': 'SMILES data should be a string'}), 400

    print("SMILES string:", smiles_string)



    try:
                # Match the embedding with the Chroma vector database
        client_final = chromadb.PersistentClient(
            path="./chroma_db",
            settings=Settings())
        
        collection_embeddings = client_final.get_collection(name="adhya_embeddings")

        # Generate embedding
        embedding = embedding_model.embed([smiles_string]) 
        results = collection_embeddings.query(query_embeddings=embedding.tolist(), n_results=3)

        prediction_antibacterial = model_antibacterial.predict([smiles_string])  # Get probability of the positive class
        prediction_antibacterial = prediction_antibacterial.tolist()  # Convert to list if it's not already
        print("Antibacterial prediction:", prediction_antibacterial)
        prediction_antifungal = model_antitumour.predict([smiles_string])  # Get probability of the positive class
        prediction_antifungal = prediction_antifungal.tolist()  # Convert to list if it's not already
        print("Antitfungal prediction:", prediction_antifungal)  
        prediction = toxicity_model.predict([smiles_string])  # Get probability of the positive class
        prediction = prediction.tolist()  # Convert to list if it's not already
        print("Toxicity prediction:", prediction)


        # Convert prediction to string if necessary
        prediction_antibacterial_str = str(prediction_antibacterial[0])
        prediction_antitumour_str = str(prediction_antifungal[0])
        prediction_str = str(prediction[0])
        return jsonify({'antibacterial': prediction_antibacterial_str, 'antitumour': prediction_antitumour_str, 'toxicity': prediction_str, 
                        'matched_protein_ids': results['ids']}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
