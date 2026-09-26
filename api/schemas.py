"""Pydantic response models for the API (§18).

Pydantic's job here: describe the exact shape of each JSON response as a
Python class, so FastAPI can (1) validate that what a route returns
actually matches that shape, (2) auto-generate the interactive API docs at
/docs, and (3) serialize Python/Decimal/datetime values into plain JSON
automatically.
"""
from datetime import date, datetime

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    currency_pair: str
    as_of_date: date
    current_rate: float
    predicted_rate: float
    expected_movement_pct: float
    direction: str
    lower_bound: float
    upper_bound: float
    confidence: float
    model_version: str
    generated_at: datetime


class HistoricalPoint(BaseModel):
    date: date
    close: float


class IndicatorValue(BaseModel):
    name: str
    value: float
    as_of_date: date


class ModelPerformanceEntry(BaseModel):
    model: str
    mae: float | None
    rmse: float | None
    directional_accuracy: float | None
