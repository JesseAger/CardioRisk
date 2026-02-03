import json
from pathlib import Path
import joblib
import numpy as np

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]  # backend/
PROJECT_ROOT = BASE_DIR.parent                 # CardioRisk/

MODEL_PATH = PROJECT_ROOT / "models" / "heart" / "artifacts" / "heart_model_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "heart" / "artifacts" / "metrics.json"


class HeartModelService:
    """
    Loads the trained sklearn pipeline and uses metrics.json to:
    - enforce feature set
    - use tuned threshold for classification
    """
    _pipeline = None
    _threshold = 0.5
    _features = None

    @classmethod
    def load(cls):
        if cls._pipeline is None:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
            cls._pipeline = joblib.load(MODEL_PATH)

        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text())
            cls._threshold = float(metrics["holdout"]["threshold"])
            cls._features = metrics["data"]["features_used"]
        else:
            # Fallback (shouldn't happen in your setup)
            cls._threshold = 0.5
            cls._features = [
                "age","sex","cp","trestbps","chol","fbs","restecg","thalch",
                "exang","oldpeak","slope","ca","thal"
            ]

    @classmethod
    def predict(cls, payload: dict) -> dict:
        cls.load()

        # Ensure required fields exist
        missing = [f for f in cls._features if f not in payload]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")

        # Build single-row dict in correct feature order
        row = {f: payload[f] for f in cls._features}

        # Pipeline expects a dataframe-like input; dict-of-lists works for sklearn
        X = {k: [v] for k, v in row.items()}

        proba = float(cls._pipeline.predict_proba(X)[0][1])
        label = int(proba >= cls._threshold)

        return {
            "risk_probability": proba,
            "threshold": cls._threshold,
            "prediction": label,  # 1=high risk, 0=low risk
            "risk_label": "High risk" if label == 1 else "Low risk",
            "features_used": cls._features,
        }
