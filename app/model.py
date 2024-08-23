from pathlib import Path
import joblib, pandas as pd

_ARTIFACTS = Path(__file__).parent / "artifacts"
_PIPELINE  = joblib.load(_ARTIFACTS / "kmeans_pipeline.joblib")
_META      = joblib.load(_ARTIFACTS / "meta.joblib")

def predict(raw_row: dict) -> int:
    """
    raw_row: dict from HTML form  →  returns cluster int label.
    """
    df = pd.DataFrame([raw_row])
    return int(_PIPELINE.predict(df)[0])

def meta() -> dict:
    return _META
