from src.models.peak_hour import train_peak_hour_classifier


def test_peak_hour_busy_class_recall_is_improved():
    result = train_peak_hour_classifier()
    assert result["model_name"] == "xgb_weighted"
    assert result["report"]["1"]["recall"] >= 0.5
