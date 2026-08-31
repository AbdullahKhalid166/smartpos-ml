from pathlib import Path

import joblib

from src.models.insights import generate_insights
from src.models.peak_hour import predict_busy_hour, train_peak_hour_classifier


def test_peak_hour_busy_class_recall_is_improved():
    result = train_peak_hour_classifier()
    assert result["model_name"] == "xgb_weighted"
    assert result["report"]["1"]["recall"] >= 0.5


def test_peak_hour_model_uses_hour_signal_and_keeps_precision():
    result = train_peak_hour_classifier()
    assert "Hour" in result["features"]
    assert result["report"]["accuracy"] >= 0.7
    assert result["report"]["1"]["precision"] >= 0.45
    assert predict_busy_hour(18, 5, 1) in {0, 1}


def test_generate_insights_saves_summary_artifact():
    artifact_path = Path(__file__).resolve().parents[1] / "models" / "insights_summary.joblib"
    if artifact_path.exists():
        artifact_path.unlink()

    result = generate_insights()

    assert "insights" in result
    assert result["insights"]
    assert artifact_path.exists()
    saved = joblib.load(artifact_path)
    assert saved["insights"] == result["insights"]
