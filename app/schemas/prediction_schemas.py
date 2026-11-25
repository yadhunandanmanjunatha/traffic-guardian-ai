"""
Traffic Guardian AI - Prediction Schemas

Request/response models for /api/predict and /api/health.
Only OBSERVABLE inputs (camera/sensor-plausible) are exposed on the
request schema - this is exactly what the YOLO-on-Raspberry-Pi edge
device will send later.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # vehicle counts (this is exactly what YOLO on the Pi will send later)
    car_count: int = Field(20, ge=0)
    bike_count: int = Field(8, ge=0)
    bus_count: int = Field(1, ge=0)
    truck_count: int = Field(1, ge=0)

    average_speed: float = Field(28.0, ge=0, le=140)

    # road context (fixed per camera location, set once during setup)
    road_type: str = Field("arterial", pattern="^(highway|arterial|collector|local)$")
    number_of_lanes: int = Field(3, ge=1, le=8)
    speed_limit: int = Field(50, ge=10, le=120)
    road_length_km: float = Field(1.5, gt=0)
    junction: bool = False
    intersection_type: str = Field("signalized", pattern="^(signalized|roundabout|uncontrolled|none)$")

    # weather (from a weather API in production; typed here for the demo)
    rainfall_mm: float = Field(0.0, ge=0)
    visibility_km: float = Field(10.0, ge=0, le=10)
    wind_speed: float = Field(10.0, ge=0)
    temperature: float = Field(26.0)

    # time
    hour: int = Field(default_factory=lambda: datetime.now().hour, ge=0, le=23)
    is_weekend: bool = False
    is_holiday: bool = False

    # incident (optional - only needed for clearance-time prediction)
    incident_exists: bool = False
    severity: int = Field(0, ge=0, le=5)
    blocked_lanes: int = Field(0, ge=0, le=4)
    emergency_vehicle_required: bool = False


class TopReason(BaseModel):
    feature: str
    impact: float
    direction: str


class DerivedInputs(BaseModel):
    vehicle_count: int
    occupancy_rate: float
    road_capacity: float
    heavy_vehicle_ratio: float


class PredictResponse(BaseModel):
    congestion_class: str
    congestion_class_index: int
    confidence: float
    class_probabilities: dict
    top_reasons: list[TopReason]
    clearance_time_minutes: Optional[float] = None
    traffic_health_score: dict
    derived_inputs: DerivedInputs


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]
