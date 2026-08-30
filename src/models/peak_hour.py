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
    """Train a classifier to predict busy vs quiet hours from calendar features."""
    transactions = load_feature_transactions() if data is None else data.copy()
    positive = transactions[transactions["is_positive_sale"] == True].copy()
    hourly = positive.groupby(["Hour", "DayOfWeek", "Month"], as_index=False).size().rename(columns={"size": "transaction_count"})
    threshold = hourly["transaction_count"].quantile(0.66)
    hourly["busy"] = (hourly["transaction_count"] >= threshold).astype(int)

    features = ["DayOfWeek", "Month"]
    X = hourly[features]
    y = hourly["busy"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        objective="binary:logistic",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_report = classification_report(y_test, xgb_pred, output_dict=True, zero_division=0)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)

    best_name = "xgb" if xgb_report["accuracy"] >= rf_report["accuracy"] else "rf"
    best_model = xgb if best_name == "xgb" else rf
    best_pred = xgb_pred if best_name == "xgb" else rf_pred
    best_report = xgb_report if best_name == "xgb" else rf_report

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_model, "features": features, "model_name": best_name, "report": best_report}, MODEL_PATH)

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
        "report": best_report,
        "plot_path": plot_path,
    }


def predict_busy_hour(day_of_week, month):
    """Predict busy vs quiet for a hour-like feature row."""
    artifact = joblib.load(MODEL_PATH)
    model = artifact["model"]
    sample = pd.DataFrame({"DayOfWeek": [day_of_week], "Month": [month]})
    prediction = model.predict(sample[artifact["features"]])
    return int(prediction[0])


def generate():
    """Return peak-hour analysis output for API use."""
    return train_peak_hour_classifier()


if __name__ == "__main__":
    result = train_peak_hour_classifier()
    print(result["hourly_summary"].head().to_string(index=False))
    print(result["report"])
