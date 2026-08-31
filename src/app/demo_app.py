from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.models.insights import generate_insights

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"


@st.cache_data
def load_data():
    weekly = pd.read_csv(DATA_DIR / "weekly_product_features.csv", parse_dates=["Period"])
    customer_snapshot = pd.read_csv(DATA_DIR / "customer_rfm_asof.csv", parse_dates=["CutoffDate"])
    transactions = pd.read_csv(DATA_DIR / "feature_transactions.csv")
    alerts = pd.read_csv(DATA_DIR / "low_stock_alerts.csv")
    insights = joblib.load(MODELS_DIR / "insights_summary.joblib") if (MODELS_DIR / "insights_summary.joblib").exists() else None
    return weekly, customer_snapshot, transactions, alerts, insights


@st.cache_data
def load_segment_artifact():
    return joblib.load(MODELS_DIR / "customer_segmentation.joblib")


@st.cache_data
def load_peak_artifact():
    return joblib.load(MODELS_DIR / "peak_hour_model.joblib")


@st.cache_data
def load_profit_artifact():
    return joblib.load(MODELS_DIR / "profit_forecast_model.joblib")


@st.cache_data
def load_forecast_artifact():
    return joblib.load(MODELS_DIR / "baseline_xgb_units.joblib")


@st.cache_data
def load_insights_summary():
    artifact_path = ROOT / "models" / "insights_summary.joblib"
    if not artifact_path.exists():
        payload = generate_insights()
        return payload
    return joblib.load(artifact_path)


@st.cache_data
def product_forecast_for(product_code: str, weekly: pd.DataFrame):
    product_history = weekly[weekly["StockCode"] == product_code].sort_values("Period").copy()
    if product_history.empty:
        return pd.DataFrame(columns=["Period", "Units"])
    recent = product_history.tail(8).copy()
    recent["prediction"] = recent["Units"].rolling(4, min_periods=1).mean().round(2)
    return recent[["Period", "Units", "prediction"]]


def segment_for_customer(customer_id: int, customer_snapshot: pd.DataFrame, artifact: dict):
    row = customer_snapshot[customer_snapshot["Customer ID"] == customer_id]
    if row.empty:
        return "Unknown customer"
    features = row[["Recency", "Frequency", "Monetary"]].copy()
    if "Monetary" in features.columns and features["Monetary"].skew() > 1:
        features["Monetary"] = features["Monetary"].apply(lambda x: x if pd.isna(x) else x)
    scaled = artifact["scaler"].transform(features)
    label = artifact["model"].predict(scaled)[0]
    return artifact["segment_labels"].get(int(label), "Regular")


st.title("SmartPOS ML demo")
st.caption("Live saved-model dashboard; no retraining in the app.")

weekly, customer_snapshot, transactions, alerts, _ = load_data()
segment_artifact = load_segment_artifact()
peak_artifact = load_peak_artifact()
profit_artifact = load_profit_artifact()
forecast_artifact = load_forecast_artifact()
insight_payload = load_insights_summary()

product_options = sorted(weekly["StockCode"].dropna().unique().tolist())
selected_product = st.selectbox("Select a product", product_options[:500])
forecast_df = product_forecast_for(selected_product, weekly)

st.subheader("Forecast for selected product")
if forecast_df.empty:
    st.warning("No historical product data found for this selection.")
else:
    st.dataframe(forecast_df.tail(8), use_container_width=True)
    st.line_chart(forecast_df.set_index("Period")[["Units", "prediction"]])
    last_units = float(forecast_df["Units"].iloc[-1])
    rolling_avg = float(forecast_df["prediction"].iloc[-1])
    st.metric("Most recent units", round(last_units, 2))
    st.metric("Rolling forecast", round(rolling_avg, 2))

st.subheader("Customer segment lookup")
customer_ids = sorted(customer_snapshot["Customer ID"].dropna().unique().tolist())
customer_id = st.selectbox("Select customer ID", customer_ids[:1000])
segment = segment_for_customer(customer_id, customer_snapshot, segment_artifact)
st.write(f"Segment for customer {customer_id}: {segment}")

st.subheader("Low-stock alerts")
st.dataframe(alerts.head(20), use_container_width=True)

st.subheader("Profit forecast")
profit_period = pd.Timestamp(weekly["Period"].max()) + pd.Timedelta(weeks=1)
profit_features = pd.DataFrame({
    "year": [profit_period.year],
    "month": [profit_period.month],
    "week": [int(profit_period.isocalendar().week)],
})
profit_value = float(profit_artifact["model"].predict(profit_features)[0])
st.metric("Estimated next-period profit", round(profit_value, 2))

st.subheader("Peak-hour chart")
positive = transactions[transactions["is_positive_sale"] == True].copy()
hourly = positive.groupby("Hour", as_index=False).size().rename(columns={"size": "transaction_count"})
st.bar_chart(hourly.set_index("Hour")["transaction_count"])

st.subheader("AI Sales Insights summary")
for line in insight_payload.get("insights", []):
    st.write("- " + line)

st.caption("Saved artifact sources: baseline_xgb_units.joblib, customer_segmentation.joblib, stockout_alerts.joblib, profit_forecast_model.joblib, peak_hour_model.joblib.")
