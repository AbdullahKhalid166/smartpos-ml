from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import forecast, insights, peak_hour, profit_forecast, recommendations, segmentation, stockout
from src.api.schemas import (
	ApiResponse,
	ForecastRequest,
	ForecastResponse,
	HealthResponse,
	InsightsResponse,
	PeakHourRequest,
	PeakHourResponse,
	ProfitRequest,
	ProfitResponse,
	RecommendationResponse,
	SegmentRequest,
	SegmentResponse,
	StockoutResponse,
)
from src.utils import load_model_artifact, load_processed_csv


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
			"enhanced_product": load_model_artifact("final_product_xgb_units.joblib"),
		},
		"profit": load_model_artifact("profit_forecast_model.joblib"),
		"segment": load_model_artifact("customer_segmentation.joblib"),
		"peakhour": load_model_artifact("peak_hour_model.joblib"),
		"stockout": load_model_artifact("stockout_alerts.joblib"),
		"recommendations": load_processed_csv("association_rules.csv"),
		"insights": load_model_artifact("insights_summary.joblib"),
	}


@asynccontextmanager
async def lifespan(app: FastAPI):
	app.state.model_registry = _load_artifacts()
	app.state.weekly_product_data = load_processed_csv("weekly_product_features.csv", parse_dates=["Period"])
	app.state.stock_codes = set(app.state.weekly_product_data["StockCode"].astype(str))
	yield


app = FastAPI(title="SmartPOS ML API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=False,
	allow_methods=["*"],
	allow_headers=["*"],
)


@app.get("/")
def api_root():
	return {"message": "SmartPOS ML API is running", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health_check():
	return {"status": "ok", "models_loaded": sorted(app.state.model_registry)}


def _response(result, model_key, notes):
	return ApiResponse(result=result, model_version=MODEL_VERSIONS[model_key], notes=notes)


@app.post("/predict/forecast", response_model=ForecastResponse)
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
		app.state.model_registry["forecast"],
		app.state.weekly_product_data,
	)
	return _response(result, "forecast", "Enhanced product-level forecast using volatility and lag features.")


@app.post("/predict/profit", response_model=ProfitResponse)
def predict_profit(request: ProfitRequest):
	if request.end_date < request.start_date:
		raise HTTPException(status_code=422, detail="end_date must be on or after start_date")
	periods = pd.DataFrame({
		"Period": pd.date_range(request.start_date, request.end_date, freq="W-MON"),
	})
	if periods.empty:
		periods = pd.DataFrame({"Period": [pd.Timestamp(request.start_date)]})
	result = profit_forecast.predict_profit(periods, artifact=app.state.model_registry["profit"])
	return _response({"periods": periods["Period"].dt.date.astype(str).tolist(), "profit": [float(value) for value in result]}, "profit", "Weekly profit forecast.")


@app.post("/predict/segment", response_model=SegmentResponse)
def predict_segment(request: SegmentRequest):
	try:
		result = segmentation.predict_customer_segment(request.CustomerID, app.state.model_registry["segment"])
	except KeyError:
		raise HTTPException(status_code=404, detail=f"Unknown Customer ID: {request.CustomerID}")
	return _response({"CustomerID": request.CustomerID, "segment": result}, "segment", "Segment predicted from the latest customer RFM snapshot.")


@app.post("/predict/peakhour", response_model=PeakHourResponse)
def predict_peakhour(request: PeakHourRequest):
	result = peak_hour.predict_busy_hour(request.DayOfWeek, request.Month, request.Hour, app.state.model_registry["peakhour"])
	return _response({"prediction": "busy" if result else "quiet", "is_busy": bool(result)}, "peakhour", "Busy classification from the saved peak-hour model.")


@app.get("/predict/stockout", response_model=StockoutResponse)
def predict_stockout():
	alerts = stockout.generate(app.state.model_registry["stockout"])
	return _response(alerts.to_dict(orient="records"), "stockout", "Current low-stock velocity alerts.")


@app.get("/recommend/{stock_code}", response_model=RecommendationResponse)
def recommend(stock_code: str):
	if stock_code not in app.state.stock_codes:
		raise HTTPException(status_code=404, detail=f"Unknown StockCode: {stock_code}")
	result = recommendations.recommend(stock_code, app.state.model_registry["recommendations"])
	return _response({"StockCode": stock_code, "recommendations": result}, "recommendations", "Top associated products from the saved association rules.")


@app.post("/insights", response_model=InsightsResponse)
def get_insights():
	return _response(insights.generate(app.state.model_registry["insights"]), "insights", "Saved AI Sales Insights summary.")
