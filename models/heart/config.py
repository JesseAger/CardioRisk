from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # CardioRisk/
DATA_PATH = PROJECT_ROOT / "dataset" / "heart_disease_uci.csv"

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "heart_model_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"
