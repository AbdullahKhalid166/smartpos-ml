"""Velocity-based stockout risk indicators and alert generation."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DAILY_DATA_PATH = ROOT / "data" / "processed" / "daily_product_features.csv"
VELOCITY_OUTPUT_PATH = ROOT / "data" / "processed" / "stockout_velocity_risk.csv"
ALERT_OUTPUT_PATH = ROOT / "data" / "processed" / "low_stock_alerts.csv"


def estimate_velocity(data=None, recent_days=28, risk_quantile=0.75):
    """Estimate product velocity and flag unusually high recent demand."""
    if data is None:
        data = pd.read_csv(DAILY_DATA_PATH, parse_dates=["Period"])
    data = data.sort_values("Period").copy()
    latest_date = data["Period"].max()
    recent = data[data["Period"] > latest_date - pd.Timedelta(days=recent_days)]

    summary = data.groupby(["StockCode", "Description"], as_index=False).agg(
        first_day=("Period", "min"),
        last_day=("Period", "max"),
        total_units=("Units", "sum"),
        observed_days=("Period", "nunique"),
    )
    recent_summary = recent.groupby("StockCode", as_index=False).agg(
        recent_units=("Units", "sum"),
        recent_observed_days=("Period", "nunique"),
    )

    summary = summary.merge(recent_summary, on="StockCode", how="left").fillna(0)
    summary["average_daily_units"] = summary["total_units"] / summary["observed_days"].clip(lower=1)
    summary["recent_daily_units"] = summary["recent_units"] / summary["recent_observed_days"].clip(lower=1)
    threshold = summary["recent_daily_units"].quantile(risk_quantile)
    summary["risk_threshold"] = threshold
    summary["stockout_risk"] = summary["recent_daily_units"] >= threshold
    summary["risk_basis"] = "high recent sales velocity; no stock-on-hand data"

    summary.to_csv(VELOCITY_OUTPUT_PATH, index=False)
    return summary


def build_low_stock_alerts(data=None, recent_days=28, risk_quantile=0.75):
    """Build a clean alert dataset with human-readable reasons."""
    velocity = estimate_velocity(data=data, recent_days=recent_days, risk_quantile=risk_quantile)
    alert_rows = []
    for _, row in velocity.iterrows():
        risk = bool(row["stockout_risk"])
        if not risk:
            continue

        reason = (
            f"Recent average daily demand is {row['recent_daily_units']:.2f} units versus a "
            f"{row['risk_threshold']:.2f}-unit threshold; this indicates unusually high demand pressure."
        )
        alert_rows.append(
            {
                "StockCode": row["StockCode"],
                "Description": row["Description"],
                "risk_flag": True,
                "reason": reason,
                "recent_daily_units": row["recent_daily_units"],
                "risk_threshold": row["risk_threshold"],
            }
        )

    alerts = pd.DataFrame(alert_rows)
    if alerts.empty:
        alerts = pd.DataFrame(columns=["StockCode", "Description", "risk_flag", "reason", "recent_daily_units", "risk_threshold"])
    alerts.to_csv(ALERT_OUTPUT_PATH, index=False)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"alerts": alerts, "velocity": velocity}, MODEL_DIR / "stockout_alerts.joblib")
    return alerts


def generate():
    """Return a clean alert dataset in API-ready form."""
    return build_low_stock_alerts()


if __name__ == "__main__":
    result = build_low_stock_alerts()
    print(result.head().to_string(index=False))
