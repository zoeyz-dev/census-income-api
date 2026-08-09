# Census Income API

This project trains a RandomForestClassifier on the UCI Census Income (Adult) dataset to
predict whether a person's annual income exceeds $50,000, and serves predictions through a
FastAPI REST API. It covers the full pipeline: data processing, model training, unit
testing, slice-based performance evaluation, and API deployment.

**GitHub repository:** https://github.com/zoeyz-dev/census-income-api

## Setup

Python version is pinned in `.python-version`. Create and activate a virtual
environment, then install dependencies:

```
python -m venv .venv
pip install -r requirements.txt
```

Activate the venv first: `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on macOS/Linux.

## Training the model

```
python train_model.py
```

This loads `data/census.csv`, splits it 80/20 into train/test sets, trains the
model, prints overall precision/recall/fbeta, saves the model, encoder, and label
binarizer to `model/`, and writes per-slice metrics to `slice_output.txt`.

## Running the API locally

```
uvicorn main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, where the `/predict`
endpoint can be tried directly in the browser. `GET /` returns a welcome message, and
`POST /predict` accepts a single census record and returns the predicted salary category.

## Running tests

```
pytest -v
```

This runs the tests covering data processing, model training, inference, slice metrics,
and both API endpoints.

## Model card

See `model_card.md` for details on the model, training/evaluation data, metrics, and known
limitations across data slices.
