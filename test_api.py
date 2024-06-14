import requests

url = 'http://localhost:5000/predict'
data = {
    "smiles": "CCC(C)C1N=CC(CC=2C=3C(C(O)C=CC=3)NC=2)NC(=O)C(C(C)C)NC(=O)C(CC(N)=O)NC(=O)C(C(CC)C)NC(=O)C(CCC(N)=O)NC1=O"
}

response = requests.post(url, json=data)
print("Status Code:", response.status_code)
print("Response:", response.json())
