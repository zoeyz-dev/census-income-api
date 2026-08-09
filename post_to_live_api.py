"""
Sends one record to the live, deployed census income API and prints the
response. Run this after the app is deployed on Render.
"""

import requests

# Change this to the actual Render URL once the app is deployed.
API_URL = "https://census-income-api-trxw.onrender.com/predict"

record = {
    "age": 45,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Doctorate",
    "education-num": 16,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}

if __name__ == "__main__":
    response = requests.post(API_URL, json=record)
    print(f"Status code: {response.status_code}")
    print(f"Result: {response.json()}")
