"""Rule-based AI sales insights generator for SmartPOS outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .segmentation import load_latest_customer_snapshot, fit_customer_segmentation
from .stockout import build_low_stock_alerts
from .profit_forecast import train_profit_model
from .peak_hour import train_peak_hour_classifier

ROOT = Path(__file__).resolve().parents[2]


def _segment_summary():
    segmentation = fit_customer_segmentation()
    summary = segmentation["data"].groupby("segment")["Customer ID"].count().reset_index()
    return summary.rename(columns={"Customer ID": "customer_count"})


def generate_insights():
    """Generate short rule-based summaries from the Week 3 outputs."""
    segmentation = fit_customer_segmentation()
    stockout = build_low_stock_alerts()
    profit = train_profit_model()
    peak = train_peak_hour_classifier()

    segment_summary = segmentation["data"].groupby("segment")["Customer ID"].count().reset_index()
    largest_segment = segment_summary.sort_values("Customer ID", ascending=False).iloc[0]
    top_risk = stockout.sort_values("recent_daily_units", ascending=False).head(1).iloc[0]
    peak_hour = peak["hourly_summary"].sort_values("transaction_count", ascending=False).head(1).iloc[0]

    insights = [
        f"Customer base is led by the {largest_segment['segment']} segment with {int(largest_segment['Customer ID'])} customers.",
        f"The highest-risk product is {top_risk['StockCode']} ({top_risk['Description']}) with recent daily demand of {top_risk['recent_daily_units']:.2f} units.",
        f"The busiest hour is {int(peak_hour['Hour'])}:00, with {int(peak_hour['transaction_count'])} transactions recorded in the positive-sale set.",
        f"Profit forecasting achieved a best XGBoost RMSE of {profit['best_model']['metrics']['RMSE']:.2f}, indicating the estimated margin forecast remains the most stable business signal.",
        f"Peak-hour classification reached {peak['report']['accuracy']:.2f} accuracy, suggesting operational staffing should prioritize the top-volume hour windows.",
    ]

    surprise = {
        "segment": largest_segment["segment"],
        "product": top_risk["StockCode"],
        "hour": int(peak_hour["Hour"]),
        "largest_segment_customer_count": int(largest_segment["Customer ID"]),
        "top_risk_product_units": float(top_risk["recent_daily_units"]),
        "peak_hour_volume": int(peak_hour["transaction_count"]),
    }

    return {"insights": insights, "surprise": surprise}


def generate():
    """Compatibility wrapper for API-style use."""
    return generate_insights()


if __name__ == "__main__":
    result = generate_insights()
    for line in result["insights"]:
        print(line)
    print("Surprise:", result["surprise"])
