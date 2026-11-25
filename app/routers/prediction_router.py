"""
Traffic Guardian AI - Prediction Router

    python3 -m uvicorn app.main:app --reload --port 8000
"""
from fastapi import APIRouter

from app.schemas.prediction_schemas import PredictRequest
from app.services.prediction_service import predict_traffic

router = APIRouter(tags=["prediction"])


@router.post("/api/predict")
def predict(r: PredictRequest):
    return predict_traffic(r)
