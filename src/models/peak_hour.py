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


def train_peak_hour_classifier(data=None):
    """Train a weighted XGBoost classifier to improve recall for the busy class."""
    transactions = load_feature_transactions() if data is None else data.copy()
    positive = transactions[transactions["is_positive_sale"] == True].copy()
    hourly = positive.groupby(["Hour", "DayOfWeek", "Month"], as_index=False).size().rename(columns={"size": "transaction_count"})
    threshold = hourly["transaction_count"].quantile(0.66)
    hourly["busy"] = (hourly["transaction_count"] >= threshold).astype(int)

    features = ["DayOfWeek", "Month"]
    X = hourly[features]
    y = hourly["busy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    class_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42,
        scale_pos_weight=class_weight,
    )
    xgb.fit(X_train, y_train)
    xgb_prob = xgb.predict_proba(X_test)[:, 1]

    thresholds = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    best_threshold = 0.5
    best_report = classification_report(y_test, (xgb_prob >= best_threshold).astype(int), output_dict=True, zero_division=0)
    best_recall = best_report["1"]["recall"]
    for threshold_value in thresholds:
        pred = (xgb_prob >= threshold_value).astype(int)
        report = classification_report(y_test, pred, output_dict=True, zero_division=0)
        recall = report["1"]["recall"]
        if recall > best_recall:
            best_threshold = threshold_value
            best_recall = recall
            best_report = report

    best_pred = (xgb_prob >= best_threshold).astype(int)
    best_name = "xgb_weighted"

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": xgb, "features": features, "model_name": best_name, "threshold": best_threshold, "report": best_report}, MODEL_PATH)

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

    return {
        "hourly_summary": hourly,
        "best_model": best_name,
        "model_name": best_name,
        "threshold": best_threshold,
        "report": best_report,
        "plot_path": plot_path,
    }


def predict_busy_hour(day_of_week, month):
    """Predict busy vs quiet for a hour-like feature row using the tuned probability threshold."""
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    sample = pd.DataFrame({"DayOfWeek": [day_of_week], "Month": [month]})
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
