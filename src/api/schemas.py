from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field


class ForecastRequest(BaseModel):
	StockCode: str = Field(min_length=1)
	start_date: date
	end_date: date


class ProfitRequest(BaseModel):
	start_date: date
	end_date: date


class SegmentRequest(BaseModel):
	CustomerID: str = Field(min_length=1)


class PeakHourRequest(BaseModel):
	DayOfWeek: int = Field(ge=1, le=7)
	Month: int = Field(ge=1, le=12)
	Hour: int = Field(ge=0, le=23)


class ForecastResult(BaseModel):
	StockCode: str
	periods: list[str]
	units: list[float]
	revenue: list[float]


class ProfitResult(BaseModel):
	periods: list[str]
	profit: list[float]


class SegmentResult(BaseModel):
	CustomerID: str
	segment: str


class PeakHourResult(BaseModel):
	prediction: Literal["busy", "quiet"]
	is_busy: bool


class StockoutAlert(BaseModel):
	StockCode: str
	Description: str
	risk_flag: bool
	reason: str
	recent_daily_units: float
	risk_threshold: float


class RecommendationResult(BaseModel):
	StockCode: str
	recommendations: list[str]


class InsightsResult(BaseModel):
	insights: list[str]
	surprise: dict[str, Any]


ResultT = TypeVar("ResultT")


class ApiResponse(BaseModel, Generic[ResultT]):
	model_config = ConfigDict(extra="forbid")

	result: ResultT
	model_version: str
	notes: str


class ForecastResponse(ApiResponse[ForecastResult]):
	pass


class ProfitResponse(ApiResponse[ProfitResult]):
	pass


class SegmentResponse(ApiResponse[SegmentResult]):
	pass


class PeakHourResponse(ApiResponse[PeakHourResult]):
	pass


class StockoutResponse(ApiResponse[list[StockoutAlert]]):
	pass


class RecommendationResponse(ApiResponse[RecommendationResult]):
	pass


class InsightsResponse(ApiResponse[InsightsResult]):
	pass


class HealthResponse(BaseModel):
	status: Literal["ok"]
	models_loaded: list[str]
