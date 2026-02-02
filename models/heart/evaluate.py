import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from .config import DATA_PATH, MODEL_PATH


def main():
    df = pd.read_csv(DATA_PATH)
    df = df[df["dataset"] == "Cleveland"].copy()
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["id", "num"])

    X = df.drop(columns=["target"])
    y = df["target"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = joblib.load(MODEL_PATH)
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    out = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
