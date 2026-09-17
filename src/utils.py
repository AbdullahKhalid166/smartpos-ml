"""Small shared helpers for API and model artifact access."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_model_artifact(filename: str) -> Any:
	"""Load one trained artifact from the project model directory."""
	return joblib.load(MODEL_DIR / filename)


def load_processed_csv(filename: str, **kwargs: Any) -> pd.DataFrame:
	"""Load one generated dataset from the processed-data directory."""
	return pd.read_csv(PROCESSED_DATA_DIR / filename, **kwargs)
