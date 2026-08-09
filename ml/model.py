from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.ndarray
        Training data.
    y_train : np.ndarray
        Labels.
    Returns
    -------
    model : RandomForestClassifier
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100, max_depth=16, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.ndarray
        Known labels, binarized.
    preds : np.ndarray
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.ndarray
        Data used for prediction.
    Returns
    -------
    preds : np.ndarray
        Predictions from the model.
    """
    return model.predict(X)


def compute_slice_metrics(
    model, encoder, lb, data, feature, categorical_features, label
):
    """
    Compute precision, recall, and fbeta for each unique value of one
    categorical feature, holding that value fixed.

    Reuses the encoder and label binarizer fit during training, so each
    slice is processed the same way the full test set was.

    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Encoder fit during training.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Label binarizer fit during training.
    data : pd.DataFrame
        Data to slice, containing `feature`, the other categorical
        features, and `label`.
    feature : str
        Name of the categorical column to slice on.
    categorical_features : list[str]
        Full list of categorical feature names, as passed to
        process_data during training.
    label : str
        Name of the label column.

    Returns
    -------
    slices : list[dict]
        One dict per unique value of `feature`, with keys "feature",
        "value", "n", "n_pos", "n_pred_pos", "precision", "recall", "fbeta".
        "n_pos" is the count of actual positive labels in the slice and
        "n_pred_pos" is the count of positive predictions; both are 0 for
        a slice where compute_model_metrics falls back to its
        zero_division convention.
    """
    slices = []
    for value in sorted(data[feature].unique(), key=str):
        slice_df = data[data[feature] == value]
        X_slice, y_slice, _, _ = process_data(
            slice_df,
            categorical_features=categorical_features,
            label=label,
            training=False,
            encoder=encoder,
            lb=lb,
        )
        preds = inference(model, X_slice)
        precision, recall, fbeta = compute_model_metrics(y_slice, preds)
        slices.append(
            {
                "feature": feature,
                "value": value,
                "n": len(slice_df),
                "n_pos": int(y_slice.sum()),
                "n_pred_pos": int(preds.sum()),
                "precision": precision,
                "recall": recall,
                "fbeta": fbeta,
            }
        )
    return slices


def save_artifacts(model, encoder, lb, model_dir=MODEL_DIR):
    """
    Save the trained model, encoder, and label binarizer to model_dir.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "model.pkl")
    joblib.dump(encoder, model_dir / "encoder.pkl")
    joblib.dump(lb, model_dir / "lb.pkl")


def load_artifacts(model_dir=MODEL_DIR):
    """
    Load the trained model, encoder, and label binarizer from model_dir.

    Returns
    -------
    model : RandomForestClassifier
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
    """
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.pkl")
    encoder = joblib.load(model_dir / "encoder.pkl")
    lb = joblib.load(model_dir / "lb.pkl")
    return model, encoder, lb
