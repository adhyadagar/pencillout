from flask import Flask, jsonify, request
import joblib
from conplex import ConplexModel  # This assumes you need the ConplexModel for processing

app = Flask(__name__)

# Load the saved model
model = joblib.load('antibacterial_model.joblib')

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
        # Generate embedding
        # embedding = embedding_model.embed([smiles_string])  # Assuming embed needs a list
        # embedding = np.array([embedding])  # Ensure it is in the correct format
        prediction = model.predict([smiles_string])  # Get probability of the positive class
        prediction = prediction.tolist()  # Convert to list if it's not already
        print("Prediction:", prediction)

        # Convert prediction to string if necessary
        prediction_str = str(prediction[0])
        return jsonify({'probability': prediction_str})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, port=5000)
