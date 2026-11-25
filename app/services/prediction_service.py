"""
Traffic Guardian AI - Prediction Service

Business logic behind POST /api/predict. Builds the exact same
engineered feature row that feature_engineering.py builds for
training (zero train/serve skew), runs it through the trained
congestion + clearance models, and explains the congestion call with
real SHAP values (not simulated).

Design: the person (or later, the Raspberry Pi) only provides OBSERVABLE
values - the kind a camera/sensor could actually produce. This service
computes every engineered feature (weather_severity_score,
road_complexity_score, etc.) from those observable values.
"""
from datetime import datetime

import numpy as np

from app.config import settings as config
from app.repositories.model_repository import model_repository
from app.schemas.prediction_schemas import PredictRequest
from app.services.clearance_physics import estimate_clearance_time
from app.services.ths_service import score_reading
from app.utils.traffic_math import road_complexity_score, weather_severity_score

CONGESTION_LABELS = config.CONGESTION_LABELS
ROAD_CAPACITY_LOOKUP = config.ROAD_CAPACITY_LOOKUP


def build_feature_row(r: PredictRequest) -> dict:
    """Mirrors feature_engineering.py exactly - single-row version.
    No sensor history exists in manual/live mode, so growth-rate and
    rolling features default to the current reading (documented, not hidden)."""
    vehicle_count = r.car_count + r.bike_count + r.bus_count + r.truck_count
    heavy_vehicle_ratio = (r.bus_count + r.truck_count) / max(vehicle_count, 1)

    road_capacity = ROAD_CAPACITY_LOOKUP[r.road_type] * r.number_of_lanes
    occupancy_rate = float(np.clip(vehicle_count / road_capacity, 0.01, 0.99))
    traffic_volume = vehicle_count
    travel_time = (r.road_length_km / max(r.average_speed, 3)) * 60  # minutes

    peak_hour = r.hour in range(7, 11) or r.hour in range(17, 21)
    precipitation_probability = float(np.clip(r.rainfall_mm / 10, 0, 1))
    humidity = float(np.clip(50 + r.rainfall_mm * 2, 20, 100))
    weather_condition = (
        "heavy_rain" if r.rainfall_mm > 5 else
        "light_rain" if r.rainfall_mm > 0 else
        "fog" if r.visibility_km < 5 else "clear"
    )

    w_score = weather_severity_score(r.rainfall_mm, r.visibility_km, r.wind_speed)
    rc_score = road_complexity_score(r.road_type, r.number_of_lanes, r.junction, r.intersection_type)

    now = datetime.now()

    row = {
        "hour": r.hour, "minute": now.minute, "day_of_week": now.weekday(),
        "month": now.month, "is_weekend": int(r.is_weekend), "is_holiday": int(r.is_holiday),
        "peak_hour": int(peak_hour),
        "vehicle_count": vehicle_count, "average_speed": r.average_speed,
        "travel_time": travel_time, "occupancy_rate": occupancy_rate,
        "traffic_volume": traffic_volume,
        "temperature": r.temperature, "humidity": humidity, "rainfall": r.rainfall_mm,
        "visibility": r.visibility_km, "wind_speed": r.wind_speed,
        "precipitation_probability": precipitation_probability,
        "weather_condition": weather_condition,
        "road_type": r.road_type, "number_of_lanes": r.number_of_lanes,
        "speed_limit": r.speed_limit, "road_capacity": road_capacity,
        "junction": int(r.junction), "intersection_type": r.intersection_type,
        "road_length": r.road_length_km,
        "incident_exists": int(r.incident_exists), "severity": r.severity,
        "blocked_lanes": r.blocked_lanes,
        "emergency_vehicle_required": int(r.emergency_vehicle_required),
        "car_count": r.car_count, "bike_count": r.bike_count,
        "bus_count": r.bus_count, "truck_count": r.truck_count,
        "heavy_vehicle_ratio": heavy_vehicle_ratio,
        "traffic_growth_rate": 0.0,               # no history in single-reading mode
        "rolling_avg_speed": r.average_speed,      # defaults to current reading
        "rolling_avg_volume": vehicle_count,       # defaults to current reading
        "rolling_speed_variance": 0.0,
        "weather_severity_score": w_score,
        "road_complexity_score": rc_score,
    }
    return row


def encode_for_model(row: dict, bundle: dict) -> np.ndarray:
    features = bundle["features"]
    encoders = bundle["encoders"]
    x = []
    for f in features:
        v = row[f]
        if f in encoders:
            le = encoders[f]
            v = v if v in le.classes_ else le.classes_[0]
            v = int(le.transform([v])[0])
        x.append(v)
    return np.array(x, dtype=float).reshape(1, -1)


def predict_traffic(r: PredictRequest) -> dict:
    """Runs the full live-prediction pipeline: feature build -> congestion
    classification (+ SHAP explanation) -> optional clearance-time
    regression -> Traffic Health Score."""
    row = build_feature_row(r)

    congestion_bundle = model_repository.congestion_bundle
    explainer = model_repository.congestion_explainer

    # ---- Model 1: Congestion classification ----
    X_cong = encode_for_model(row, congestion_bundle)
    proba = congestion_bundle["model"].predict_proba(X_cong)[0]
    pred_class = int(np.argmax(proba))
    confidence = float(proba[pred_class])

    # ---- SHAP explainability (real, not simulated) ----
    shap_values = explainer.shap_values(X_cong)
    # shap_values shape: (n_classes, 1, n_features) or (1, n_features, n_classes) depending on version
    sv = np.array(shap_values)
    if sv.ndim == 3 and sv.shape[0] == len(CONGESTION_LABELS):
        class_shap = sv[pred_class, 0, :]
    elif sv.ndim == 3:
        class_shap = sv[0, :, pred_class]
    else:
        class_shap = sv[0]

    contribs = sorted(
        zip(congestion_bundle["features"], class_shap.tolist()),
        key=lambda x: -abs(x[1]),
    )[:5]
    top_reasons = [
        {"feature": f, "impact": round(v, 4), "direction": "increases" if v > 0 else "decreases"}
        for f, v in contribs
    ]

    result = {
        "congestion_class": CONGESTION_LABELS[pred_class],
        "congestion_class_index": pred_class,
        "confidence": round(confidence, 4),
        "class_probabilities": {
            CONGESTION_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)
        },
        "top_reasons": top_reasons,
    }

    # ---- Clearance time: REAL physics-based estimate (HCM residual-capacity
    #      queueing model) is now the primary number. The ML model trained
    #      on synthetic data is kept as a secondary cross-check only. ----
    if r.incident_exists:
        physics = estimate_clearance_time(
            vehicle_count=row["vehicle_count"], occupancy_rate=row["occupancy_rate"],
            road_type=r.road_type, number_of_lanes=r.number_of_lanes,
            blocked_lanes=r.blocked_lanes, severity=r.severity,
        )
        result["clearance_time_minutes"] = physics["clearance_time_minutes"]
        result["clearance_range_minutes"] = physics["clearance_range_minutes"]
        result["clearance_breakdown"] = physics["breakdown"]
        result["clearance_source"] = physics["source"]

        try:
            clearance_bundle = model_repository.clearance_bundle
            X_clear = encode_for_model(row, clearance_bundle)
            ml_minutes = float(clearance_bundle["model"].predict(X_clear)[0])
            result["clearance_time_ml_crosscheck_minutes"] = round(ml_minutes, 1)
        except Exception:
            result["clearance_time_ml_crosscheck_minutes"] = None
    else:
        result["clearance_time_minutes"] = None
        result["clearance_range_minutes"] = None
        result["clearance_breakdown"] = None
        result["clearance_source"] = None
        result["clearance_time_ml_crosscheck_minutes"] = None

    # ---- THS - bonus explainable indicator, computed from same values (not a model) ----
    ths = score_reading(
        vehicle_count=row["vehicle_count"], road_capacity=row["road_capacity"],
        average_speed=row["average_speed"], speed_limit=row["speed_limit"],
        occupancy_rate=row["occupancy_rate"], rainfall=row["rainfall"],
        visibility=row["visibility"], wind_speed=row["wind_speed"],
        incident_exists=r.incident_exists, severity=r.severity, blocked_lanes=r.blocked_lanes,
    )
    result["traffic_health_score"] = ths

    result["derived_inputs"] = {
        "vehicle_count": row["vehicle_count"],
        "occupancy_rate": round(row["occupancy_rate"], 3),
        "road_capacity": row["road_capacity"],
        "heavy_vehicle_ratio": round(row["heavy_vehicle_ratio"], 3),
    }

    return result
