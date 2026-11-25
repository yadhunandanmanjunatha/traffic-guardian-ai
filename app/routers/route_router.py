"""
Traffic Guardian AI - Route Planning Router

Bangalore source -> destination prediction endpoints.
"""
from fastapi import APIRouter

from app.repositories.graph_repository import LANDMARKS
from app.schemas.route_schemas import BestTimeRequest, RouteRequest
from app.services import external_apis
from app.services.route_service import best_time_to_leave, get_route_alternatives, predict_route
from app.utils.traffic_math import daily_traffic_profile

router = APIRouter(tags=["route-planning"])


@router.get("/api/traffic-pattern")
def traffic_pattern():
    """The real daily traffic-volume curve (same formula used to drive
    every fallback prediction) - exposed for the UI's context
    sparkline, not a separately invented number."""
    return {"hours": list(range(24)), "profile": [round(daily_traffic_profile(h), 3) for h in range(24)]}


@router.get("/api/live-status")
def live_status():
    """Which real data sources are actually reachable right now - used
    by the UI to show LIVE / FALLBACK badges honestly."""
    return external_apis.get_live_status()


@router.get("/api/landmarks")
def get_landmarks():
    """List of Bangalore landmarks for the route planner dropdowns."""
    return {
        "landmarks": [
            {"id": k, "name": v["name"], "lat": v["lat"], "lon": v["lon"]}
            for k, v in LANDMARKS.items()
        ]
    }


@router.post("/api/route")
def route(r: RouteRequest):
    return predict_route(r.source, r.destination, hour=r.hour,
                          rainfall_mm=r.rainfall_mm, visibility_km=r.visibility_km)


@router.post("/api/route/alternatives")
def route_alternatives(r: RouteRequest):
    """Up to 3 real alternative routes (not just the single fastest
    path), each with its own live congestion/weather overlay."""
    return {"routes": get_route_alternatives(r.source, r.destination, hour=r.hour,
                                              rainfall_mm=r.rainfall_mm, visibility_km=r.visibility_km)}


@router.post("/api/best-time")
def best_time(r: BestTimeRequest):
    return best_time_to_leave(r.source, r.destination, hours_ahead=r.hours_ahead,
                               rainfall_mm=r.rainfall_mm, visibility_km=r.visibility_km)
