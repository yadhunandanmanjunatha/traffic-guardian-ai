"""
Traffic Guardian AI - Route Planning Schemas

Request/response models for /api/landmarks, /api/route, /api/best-time.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class LandmarkResponse(BaseModel):
    id: str
    name: str
    lat: float
    lon: float


class RouteRequest(BaseModel):
    source: str
    destination: str
    hour: int = Field(default_factory=lambda: datetime.now().hour, ge=0, le=23)
    # Leave both null to auto-fetch REAL live weather (Open-Meteo) for
    # the route. Set either to override with a manual value instead.
    rainfall_mm: Optional[float] = Field(None, ge=0)
    visibility_km: Optional[float] = Field(None, ge=0, le=10)


class BestTimeRequest(BaseModel):
    source: str
    destination: str
    hours_ahead: int = Field(6, ge=1, le=12)
    rainfall_mm: Optional[float] = Field(None, ge=0)
    visibility_km: Optional[float] = Field(None, ge=0, le=10)
