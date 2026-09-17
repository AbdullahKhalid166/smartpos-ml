from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from src.models import forecast, insights, peak_hour, profit_forecast, recommendations, segmentation, stockout
from src.api.schemas import ApiResponse, ForecastRequest, PeakHourRequest, ProfitRequest, SegmentRequest


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"
TRAINING_LAST_DATE = pd.Timestamp("2011-12-04")
MAX_FORECAST_DATE = TRAINING_LAST_DATE + pd.Timedelta(days=365)
MODEL_VERSIONS = {
	"forecast": "enhanced_product_v1",
	"profit": "profit_forecast_v1",
	"segment": "customer_segmentation_v1",
	"peakhour": "peak_hour_v1",
	"stockout": "stockout_velocity_v1",
	"recommendations": "association_rules_v1",
	"insights": "insights_summary_v1",
}


def _load_artifacts():
	return {
		"forecast": {
			"enhanced_product": joblib.load(MODEL_DIR / "final_product_xgb_units.joblib"),
		},
		"profit": joblib.load(MODEL_DIR / "profit_forecast_model.joblib"),
		"segment": joblib.load(MODEL_DIR / "customer_segmentation.joblib"),
		"peakhour": joblib.load(MODEL_DIR / "peak_hour_model.joblib"),
		"stockout": joblib.load(MODEL_DIR / "stockout_alerts.joblib"),
		"recommendations": pd.read_csv(DATA_DIR / "association_rules.csv"),
		"insights": joblib.load(MODEL_DIR / "insights_summary.joblib"),
	}


@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.artifacts = _load_artifacts()
	app.state.weekly_product_data = pd.read_csv(DATA_DIR / "weekly_product_features.csv", parse_dates=["Period"])
	app.state.stock_codes = set(pd.read_csv(DATA_DIR / "weekly_product_features.csv", usecols=["StockCode"])["StockCode"].astype(str))
	yield


app = FastAPI(title="SmartPOS ML API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def api_root():
	return {"message": "SmartPOS ML API is running", "docs": "/docs"}


def _response(result, model_key, notes):
	return ApiResponse(result=result, model_version=MODEL_VERSIONS[model_key], notes=notes)


@app.post("/predict/forecast", response_model=ApiResponse)
def predict_forecast(request: ForecastRequest):
	if request.end_date < request.start_date:
		raise HTTPException(status_code=422, detail="end_date must be on or after start_date")
	if pd.Timestamp(request.end_date) > MAX_FORECAST_DATE:
		raise HTTPException(status_code=422, detail=f"Forecast dates cannot exceed {MAX_FORECAST_DATE.date()}")
	if request.StockCode not in app.state.stock_codes:
		raise HTTPException(status_code=404, detail=f"Unknown StockCode: {request.StockCode}")
	result = forecast.predict_forecast(
		request.StockCode,
		request.start_date,
		request.end_date,
		app.state.artifacts["forecast"],
		app.state.weekly_product_data,
	)
	return _response(result, "forecast", "Enhanced product-level forecast using volatility and lag features.")


@app.post("/predict/profit", response_model=ApiResponse)
def predict_profit(request: ProfitRequest):
	if request.end_date < request.start_date:
		raise HTTPException(status_code=422, detail="end_date must be on or after start_date")
	periods = pd.DataFrame({
		"Period": pd.date_range(request.start_date, request.end_date, freq="W-MON"),
	})
	if periods.empty:
		periods = pd.DataFrame({"Period": [pd.Timestamp(request.start_date)]})
	result = profit_forecast.predict_profit(periods, artifact=app.state.artifacts["profit"])
	return _response({"periods": periods["Period"].dt.date.astype(str).tolist(), "profit": [float(value) for value in result]}, "profit", "Weekly profit forecast.")


@app.post("/predict/segment", response_model=ApiResponse)
def predict_segment(request: SegmentRequest):
	try:
		result = segmentation.predict_customer_segment(request.CustomerID, app.state.artifacts["segment"])
	except KeyError:
		raise HTTPException(status_code=404, detail=f"Unknown Customer ID: {request.CustomerID}")
	return _response({"CustomerID": request.CustomerID, "segment": result}, "segment", "Segment predicted from the latest customer RFM snapshot.")


@app.post("/predict/peakhour", response_model=ApiResponse)
def predict_peakhour(request: PeakHourRequest):
	result = peak_hour.predict_busy_hour(request.DayOfWeek, request.Month, request.Hour, app.state.artifacts["peakhour"])
	return _response({"prediction": "busy" if result else "quiet", "is_busy": bool(result)}, "peakhour", "Busy classification from the saved peak-hour model.")


@app.get("/predict/stockout", response_model=ApiResponse)
def predict_stockout():
	alerts = stockout.generate(app.state.artifacts["stockout"])
	return _response(alerts.to_dict(orient="records"), "stockout", "Current low-stock velocity alerts.")


@app.get("/recommend/{stock_code}", response_model=ApiResponse)
def recommend(stock_code: str):
	if stock_code not in app.state.stock_codes:
		raise HTTPException(status_code=404, detail=f"Unknown StockCode: {stock_code}")
	result = recommendations.recommend(stock_code, app.state.artifacts["recommendations"])
	return _response({"StockCode": stock_code, "recommendations": result}, "recommendations", "Top associated products from the saved association rules.")


@app.post("/insights", response_model=ApiResponse)
def get_insights():
	return _response(insights.generate(app.state.artifacts["insights"]), "insights", "Saved AI Sales Insights summary.")
