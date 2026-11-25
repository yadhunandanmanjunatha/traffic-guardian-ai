"""
Traffic Guardian AI - Health Router
"""
from fastapi import APIRouter

from app.repositories.model_repository import model_repository
from app.schemas.prediction_schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/health", response_model=HealthResponse)
def health():
    loaded = []
    try:
        model_repository.congestion_bundle
        loaded.append("congestion_model")
    except Exception:
        pass
    try:
        model_repository.clearance_bundle
        loaded.append("clearance_model")
    except Exception:
        pass
    return {"status": "ok", "models_loaded": loaded}
