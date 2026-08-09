import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from main import CATEGORICAL_FEATURES, CensusRecord
from ml.data import load_data, process_data
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    train_model,
)

DATA_PATH = "data/census.csv"


def test_train_model_returns_fitted_random_forest():
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "estimators_")
    assert model.n_estimators == 100
    assert model.max_depth == 16


def test_inference_returns_binary_predictions():
    X_train = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    y_train = np.array([0, 1, 0, 1])
    model = train_model(X_train, y_train)
    preds = inference(model, X_train)
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_train.shape[0]
    assert set(np.unique(preds)) <= {0, 1}


def test_compute_model_metrics_matches_hand_computed_values():
    # TP=2, FP=1, FN=1 -> precision = recall = fbeta = 2/3
    y = np.array([1, 0, 1, 1, 0])
    preds = np.array([1, 0, 0, 1, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == pytest.approx(2 / 3)
    assert recall == pytest.approx(2 / 3)
    assert fbeta == pytest.approx(2 / 3)


def test_compute_model_metrics_zero_division_convention():
    y = np.array([0, 0, 0])
    preds = np.array([0, 0, 0])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def _toy_frame():
    return pd.DataFrame(
        {
            "age": [25, 40, 35, 50, 22, 60, 45, 30],
            "hours-per-week": [40, 40, 35, 45, 20, 50, 40, 30],
            "workclass": [
                "Private",
                "Private",
                "Self-emp",
                "Private",
                "Self-emp",
                "Private",
                "Self-emp",
                "Private",
            ],
            "salary": [
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
                "<=50K",
                ">50K",
            ],
        }
    )


def test_compute_slice_metrics_reports_counts_per_value():
    df = _toy_frame()
    cat_features = ["workclass"]
    X, y, encoder, lb = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)

    slices = compute_slice_metrics(
        model, encoder, lb, df, "workclass", cat_features, "salary"
    )

    assert isinstance(slices, list)
    assert len(slices) == df["workclass"].nunique()
    assert sum(row["n"] for row in slices) == len(df)
    by_value = {row["value"]: row for row in slices}
    assert by_value["Private"]["n"] == 5
    assert by_value["Private"]["n_pos"] == 4
    assert by_value["Self-emp"]["n"] == 3
    assert by_value["Self-emp"]["n_pos"] == 0
    for row in slices:
        assert row["feature"] == "workclass"
        assert row["n"] == (df["workclass"] == row["value"]).sum()


def test_process_data_train_and_infer_end_to_end_on_toy_frame():
    df = _toy_frame()
    cat_features = ["workclass"]

    X, y, encoder, lb = process_data(
        df, categorical_features=cat_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    # 2 continuous columns + 2 one-hot categories (Private, Self-emp)
    assert X.shape == (8, 4)
    assert set(np.unique(y)) == {0, 1}
    assert lb.classes_.tolist() == ["<=50K", ">50K"]
    assert preds.shape[0] == df.shape[0]


def test_inference_column_order_matches_api_field_order():
    alias_order = [
        field.alias or name
        for name, field in CensusRecord.model_fields.items()
    ]
    continuous_from_api = [
        name for name in alias_order if name not in CATEGORICAL_FEATURES
    ]

    data = load_data(DATA_PATH)
    continuous_from_data = [
        col
        for col in data.columns
        if col not in CATEGORICAL_FEATURES and col != "salary"
    ]

    assert continuous_from_api == continuous_from_data
