from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

LOW_INCOME_RECORD = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

HIGH_INCOME_RECORD = {
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


def test_welcome_message():
    r = client.get("/")
    assert r.status_code == 200
    expected = {"message": "Welcome to the census income classifier API."}
    assert r.json() == expected


def test_predict_low_income():
    r = client.post("/predict", json=LOW_INCOME_RECORD)
    assert r.status_code == 200
    assert r.json() == {"salary": "<=50K"}


def test_predict_high_income():
    r = client.post("/predict", json=HIGH_INCOME_RECORD)
    assert r.status_code == 200
    assert r.json() == {"salary": ">50K"}


def test_predict_rejects_malformed_body():
    bad_record = dict(LOW_INCOME_RECORD)
    bad_record["age"] = "not-a-number"
    r = client.post("/predict", json=bad_record)
    assert r.status_code == 422
    assert "detail" in r.json()
