"""
Traffic Guardian AI - Shared Traffic Math Helpers

Small, pure, stateless helper functions used by more than one service
(prediction_service and route_service). Kept here so the same physics
formulas aren't duplicated across the codebase.
"""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in km between two lat/lon points. Used to
    check whether a real (OSRM) route passes near a known Bangalore
    hotspot (Silk Board, Bellandur, etc.) without needing another API."""
    R = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(a)))


def daily_traffic_profile(hour: float, minute: float = 0) -> float:
    """Plausible daily traffic-volume multiplier (0.15-1.0), peaking at
    the morning (~8:30) and evening (~18:30) rush hours. Used to
    simulate believable per-segment vehicle counts along a route."""
    hour_frac = hour + minute / 60.0
    morning = np.exp(-0.5 * ((hour_frac - 8.5) / 1.3) ** 2)
    evening = np.exp(-0.5 * ((hour_frac - 18.5) / 1.5) ** 2)
    return float(np.clip(0.25 + 0.75 * morning + 0.9 * evening, 0.15, 1.0))


def weather_severity_score(rainfall_mm: float, visibility_km: float, wind_speed: float) -> float:
    """Weighted 0-100 weather severity score (rain 40%, visibility 40%, wind 20%)."""
    rain_norm = min(rainfall_mm / 20, 1.0)
    vis_risk = min(max(1 - visibility_km / 10, 0), 1)
    wind_norm = min(wind_speed / 40, 1.0)
    return (rain_norm * 0.4 + vis_risk * 0.4 + wind_norm * 0.2) * 100


def road_complexity_score(road_type: str, number_of_lanes: int, junction: bool, intersection_type: str) -> float:
    """Weighted 0-100 road complexity score from road-type/lane/junction risk."""
    type_risk = {"local": 0.2, "collector": 0.4, "arterial": 0.7, "highway": 0.5}[road_type]
    lane_risk = min(1 / (number_of_lanes + 1), 1)
    junction_risk = 0.3 if junction else 0.0
    intersection_risk = {
        "signalized": 0.3, "roundabout": 0.4, "uncontrolled": 0.8, "none": 0.1,
    }[intersection_type]
    return float(np.clip(
        (type_risk * 0.3 + lane_risk * 0.2 + junction_risk + intersection_risk * 0.3) * 100, 0, 100
    ))
