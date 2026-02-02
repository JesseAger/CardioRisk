import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

from .config import DATA_PATH, MODEL_PATH, METRICS_PATH


def load_data(use_only_cleveland: bool = True) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)

    # Starting with Cleveland 
    if use_only_cleveland:
        df = df[df["dataset"] == "Cleveland"].copy()

    # Binary target: 1 if num > 0 else 0
    df["target"] = (df["num"] > 0).astype(int)

    # Dropping columns not used as predictors
    df = df.drop(columns=["id", "num", "dataset"])

    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    # Identify categorical vs numeric
    categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(exclude=["object", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = LogisticRegression(max_iter=2000, class_weight="balanced")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def main():
    df = load_data(use_only_cleveland=True)

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)

    # Evaluate
    proba = pipeline.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "dataset_filter": "Cleveland",
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "roc_auc": float(auc),
        "confusion_matrix": cm,
        "classification_report": report,
        "features_used": X.columns.tolist(),
    }

    # Save artifacts
    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    print(f"Saved model -> {MODEL_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
