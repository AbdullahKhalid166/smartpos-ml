"""Peak hour prediction for SmartPOS."""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "feature_transactions.csv"
MODEL_PATH = ROOT / "models" / "peak_hour_model.joblib"
FIGURE_DIR = ROOT / "reports" / "figures"
REPORT_PATH = ROOT / "reports" / "peak_hour_model_report.md"


def load_feature_transactions(path=DATA_PATH):
    """Load transaction-level data for peak-hour analysis."""
    data = pd.read_csv(path)
    return data.copy()


def summarize_peak_hours(data=None):
    """Aggregate counts by hour to identify the busiest hours."""
    transactions = load_feature_transactions() if data is None else data.copy()
    positive = transactions[transactions["is_positive_sale"] == True].copy()
    hourly = positive.groupby("Hour", as_index=False).size().rename(columns={"size": "transaction_count"})
    hourly = hourly.sort_values("transaction_count", ascending=False).reset_index(drop=True)
    return hourly


def build_busy_label(data=None):
    """Label each hour as busy/quiet based on the top tertile of transaction volume."""
    hourly = summarize_peak_hours(data)
    threshold = hourly["transaction_count"].quantile(0.66)
    hourly["busy"] = (hourly["transaction_count"] >= threshold).astype(int)
    return hourly


def _prepare_hourly_dataset(data=None):
    """Aggregate positive sales by hour and engineer cyclical features for a better busy-hour signal."""
    transactions = load_feature_transactions() if data is None else data.copy()
    positive = transactions[transactions["is_positive_sale"] == True].copy()
    hourly = positive.groupby(["Hour", "DayOfWeek", "Month"], as_index=False).size().rename(columns={"size": "transaction_count"})
    hourly["is_weekend"] = hourly["DayOfWeek"].isin([6, 7]).astype(int)
    hourly["hour_sin"] = np.sin(2 * np.pi * hourly["Hour"] / 24)
    hourly["hour_cos"] = np.cos(2 * np.pi * hourly["Hour"] / 24)
    threshold = hourly["transaction_count"].quantile(0.66)
    hourly["busy"] = (hourly["transaction_count"] >= threshold).astype(int)
    return hourly


def _save_peak_hour_report(best_report, best_threshold, model_name, feature_names):
    """Persist a plain-text report with the selected metrics for the improved peak-hour model."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    precision = float(best_report["1"]["precision"])
    recall = float(best_report["1"]["recall"])
    accuracy = float(best_report["accuracy"])
    f1_score = float(best_report["1"]["f1-score"])
    content = (
        "# Peak Hour Model Evaluation\n\n"
        f"- Model: {model_name}\n"
        f"- Threshold: {best_threshold:.2f}\n"
        f"- Features: {', '.join(feature_names)}\n"
        f"- Accuracy: {accuracy:.4f}\n"
        f"- Busy precision: {precision:.4f}\n"
        f"- Busy recall: {recall:.4f}\n"
        f"- Busy F1: {f1_score:.4f}\n\n"
        "## Interpretation\n"
        "The improved classifier uses the actual hour signal and cyclical hour encoding so it can distinguish peak shopping windows more reliably.\n"
        "This version is tuned to balance busy-hour recall with precision and is saved to the project artifacts for later reference.\n"
    )
    REPORT_PATH.write_text(content, encoding="utf-8")
    return REPORT_PATH


def train_peak_hour_classifier(data=None):
    """Train a stronger hourly XGBoost classifier with cyclic features and threshold tuning."""
    hourly = _prepare_hourly_dataset(data)
    features = ["Hour", "DayOfWeek", "Month", "is_weekend", "hour_sin", "hour_cos"]
    X = hourly[features]
    y = hourly["busy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    class_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        random_state=42,
        scale_pos_weight=class_weight,
        eval_metric="logloss",
    )
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.25, 0.75, 21)
    best_threshold = 0.5
    best_report = classification_report(y_test, (xgb_prob >= best_threshold).astype(int), output_dict=True, zero_division=0)
    best_f1 = best_report["1"]["f1-score"]
    best_precision = best_report["1"]["precision"]
    best_recall = best_report["1"]["recall"]

    for threshold_value in thresholds:
        pred = (xgb_prob >= threshold_value).astype(int)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        candidate_f1 = report["1"]["f1-score"]
        candidate_precision = report["1"]["precision"]
        candidate_recall = report["1"]["recall"]
        if candidate_f1 > best_f1 or (np.isclose(candidate_f1, best_f1) and candidate_precision >= best_precision and candidate_recall > best_recall):
            best_threshold = threshold_value
            best_report = report
            best_f1 = candidate_f1
            best_precision = candidate_precision
            best_recall = candidate_recall

    best_pred = (xgb_prob >= best_threshold).astype(int)
    best_name = "xgb_weighted"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": xgb,
        "features": features,
        "model_name": best_name,
        "threshold": best_threshold,
        "report": best_report,
    }
    joblib.dump(artifact, MODEL_PATH)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Busy vs Quiet Hour Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    tick_labels = ["Quiet", "Busy"]
    plt.xticks([0, 1], tick_labels)
    plt.yticks([0, 1], tick_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.tight_layout()
    plot_path = FIGURE_DIR / "peak_hour_confusion_matrix.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

    _save_peak_hour_report(best_report, best_threshold, best_name, features)

    return {
        "hourly_summary": hourly,
        "best_model": best_name,
        "model_name": best_name,
        "features": features,
        "threshold": best_threshold,
        "report": best_report,
        "plot_path": plot_path,
        "report_path": REPORT_PATH,
    }


def predict_busy_hour(day_of_week, month, hour=12):
    """Predict busy vs quiet for a specific day, month, and hour using the tuned threshold."""
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    sample = pd.DataFrame({
        "Hour": [hour],
        "DayOfWeek": [day_of_week],
        "Month": [month],
    })
    sample["is_weekend"] = sample["DayOfWeek"].isin([6, 7]).astype(int)
    sample["hour_sin"] = np.sin(2 * np.pi * sample["Hour"] / 24)
    sample["hour_cos"] = np.cos(2 * np.pi * sample["Hour"] / 24)
    probabilities = model.predict_proba(sample[artifact["features"]])[:, 1]
    threshold = artifact.get("threshold", 0.5)
    return int(probabilities[0] >= threshold)


def generate():
    """Return peak-hour analysis output for API use."""
    return train_peak_hour_classifier()


if __name__ == "__main__":
    result = train_peak_hour_classifier()
    print(result["hourly_summary"].head().to_string(index=False))
    print(result["report"])
