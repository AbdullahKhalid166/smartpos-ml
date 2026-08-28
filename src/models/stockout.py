"""Velocity-based stockout risk indicators.

This is a demand-pressure proxy. The dataset has no stock-on-hand column, so
the output cannot confirm that inventory has actually reached zero.
"""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "daily_product_features.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "stockout_velocity_risk.csv"


def estimate_velocity(data=None, recent_days=28, risk_quantile=0.75):
    """Estimate product velocity and flag unusually high recent demand."""
    if data is None:
        data = pd.read_csv(DATA_PATH, parse_dates=["Period"])
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
    summary.to_csv(OUTPUT_PATH, index=False)
    return summary


if __name__ == "__main__":
    result = estimate_velocity()
    print(result["stockout_risk"].value_counts().to_dict())