# Script to train machine learning model.

from pathlib import Path

from sklearn.model_selection import train_test_split

from ml.data import load_data, process_data
from ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_artifacts,
    train_model,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "census.csv"
SLICE_OUTPUT_PATH = BASE_DIR / "slice_output.txt"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def main():
    data = load_data(DATA_PATH)
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)

    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Fbeta: {fbeta:.4f}")

    save_artifacts(model, encoder, lb)

    with open(SLICE_OUTPUT_PATH, "w") as f:
        f.write("Model performance on slices of the test set\n")
        f.write("=" * 60 + "\n")
        f.write(
            "Note: precision/recall/fbeta are computed with "
            "zero_division=1. A slice with n_pos=0 (no actual >50K "
            "cases) or n_pred_pos=0 (no predicted >50K cases) reports "
            "1.0000 on the corresponding metric by convention, not by "
            "measurement.\n\n"
        )
        for feature in CAT_FEATURES:
            f.write(f"Feature: {feature}\n")
            f.write("-" * 60 + "\n")
            slices = compute_slice_metrics(
                model, encoder, lb, test, feature, CAT_FEATURES, "salary"
            )
            for row in slices:
                f.write(
                    f"  Value: {row['value']} "
                    f"(n={row['n']}, n_pos={row['n_pos']}, "
                    f"n_pred_pos={row['n_pred_pos']})\n"
                )
                f.write(f"    precision: {row['precision']:.4f}\n")
                f.write(f"    recall:    {row['recall']:.4f}\n")
                f.write(f"    fbeta:     {row['fbeta']:.4f}\n")
            f.write("\n")


if __name__ == "__main__":
    main()
