from datetime import date
from typing import Any

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


class ApiResponse(BaseModel):
	model_config = ConfigDict(extra="forbid")

	result: Any
	model_version: str
	notes: str


class RecommendationResponse(ApiResponse):
	pass
