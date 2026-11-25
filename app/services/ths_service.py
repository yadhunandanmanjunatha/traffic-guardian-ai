"""
Traffic Guardian AI - Traffic Health Score (THS) Service

Standalone function: given a single live reading (from a sensor or
Raspberry Pi + YOLO), compute the Traffic Health Score WITHOUT needing
the full batch pipeline. Used directly by prediction_service and can
also be called by the dashboard.
"""
from app.config import settings as config


def score_reading(
    vehicle_count, road_capacity, average_speed, speed_limit,
    occupancy_rate, rainfall, visibility, wind_speed,
    incident_exists=False, severity=0, blocked_lanes=0,
    max_rainfall_ref=20.0, max_wind_ref=40.0,
):
    """
    Returns dict with traffic_health_score, band, and the 3 sub-scores
    (congestion_index, weather_severity, incident penalty) so the
    dashboard can show WHY the score is what it is - not just the number.
    """
    eps = 1e-6

    traffic_density = vehicle_count / (road_capacity + eps)
    speed_ratio = min(average_speed / (speed_limit + eps), 1.5)
    congestion_index = max(traffic_density * (1 - speed_ratio), 0)
    queue_density_score = (traffic_density * 0.5 + occupancy_rate * 0.5) * 100

    rain_norm = min(rainfall / max_rainfall_ref, 1.0)
    vis_risk = min(max(1 - visibility / 10, 0), 1)
    wind_norm = min(wind_speed / max_wind_ref, 1.0)
    weather_severity_score = (rain_norm * 0.4 + vis_risk * 0.4 + wind_norm * 0.2) * 100

    congestion_penalty = (
        min(congestion_index, 1) * 40 +
        (1 - speed_ratio) * 15 +
        min(queue_density_score / 100, 1) * 15
    )
    weather_penalty = min(weather_severity_score / 100, 1) * 15
    incident_penalty = ((severity / 5) * 10 + (blocked_lanes / 3) * 5) if incident_exists else 0

    ths = max(min(100 - congestion_penalty - weather_penalty - incident_penalty, 100), 0)

    band = next(
        (label for lo, hi, label in config.THS_BANDS if lo <= ths <= hi),
        "Unknown",
    )

    return {
        "traffic_health_score": round(ths, 1),
        "band": band,
        "congestion_index": round(congestion_index, 3),
        "weather_severity_score": round(weather_severity_score, 1),
        "incident_penalty": round(incident_penalty, 1),
    }


if __name__ == "__main__":
    # example: moderately busy road, light rain, no incident
    result = score_reading(
        vehicle_count=180, road_capacity=250, average_speed=28, speed_limit=50,
        occupancy_rate=0.65, rainfall=2.5, visibility=7, wind_speed=10,
    )
    print(result)
