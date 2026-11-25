"""
Traffic Guardian AI - Route Prediction Service (REAL DATA)

Given a source and destination landmark, this now:
  1. Calls OSRM for the REAL OpenStreetMap driving route (real roads,
     real distance, real duration, real named segments) - not the old
     27-node hand-made graph.
  2. Samples live points along that real route and calls TomTom
     Traffic Flow for the REAL currentSpeed/freeFlowSpeed ratio - not
     a random number generator.
  3. Calls Open-Meteo for REAL live weather at the route - not a
     manually typed guess.
  4. Overlays known Bangalore hotspots (Silk Board, Bellandur, etc.)
     onto the real route geometry.

Every response says exactly which pieces are "live" vs
"fallback_estimate" (`data_sources` field) - if OSRM/TomTom/Open-Meteo
are unreachable (e.g. no internet at the venue), the app falls back to
the original fixed-graph + physics-based simulation so the demo never
breaks, but it never claims fallback numbers are live.
"""
import heapq
from datetime import datetime, timedelta

import numpy as np

from app.config import settings as config
from app.exceptions.custom_exceptions import RouteNotFoundError, UnknownLandmarkError
from app.repositories.graph_repository import LANDMARKS, build_graph, nearby_known_issues
from app.repositories.model_repository import model_repository
from app.schemas.prediction_schemas import PredictRequest
from app.services import external_apis
from app.services.clearance_physics import estimate_clearance_time
from app.services.prediction_service import build_feature_row, encode_for_model
from app.services.ths_service import score_reading
from app.utils.traffic_math import daily_traffic_profile

GRAPH = build_graph()
CONGESTION_LABELS = config.CONGESTION_LABELS


def _speed_ratio_to_class(ratio):
    for lo, hi, label in config.SPEED_RATIO_BANDS:
        if lo <= ratio < hi:
            return label
    return "Severe"


def _resolve_landmark(landmark_id):
    if landmark_id not in LANDMARKS:
        raise UnknownLandmarkError(f"Unknown landmark id: {landmark_id}")
    return LANDMARKS[landmark_id]


# =================================================================
# REAL PATH: OSRM route + TomTom live flow + Open-Meteo weather
# =================================================================
def _build_real_route(source_id, dest_id, rainfall_override=None, visibility_override=None):
    src, dst = _resolve_landmark(source_id), _resolve_landmark(dest_id)

    real_route = external_apis.get_real_route(src["lat"], src["lon"], dst["lat"], dst["lon"])
    if real_route is None:
        return None  # caller falls back

    return _process_route_with_live_data(real_route, src, dst, rainfall_override, visibility_override)


def _process_route_with_live_data(real_route, src, dst, rainfall_override=None, visibility_override=None):
    """Takes an already-fetched OSRM route (real_route) and overlays
    live weather + live congestion + known hotspots. Shared by the
    single-route path and the alternate-routes path so both get
    identical real-data treatment."""
    # --- live weather (unless the user explicitly overrode it) ---
    if rainfall_override is not None or visibility_override is not None:
        weather = {
            "rainfall_mm": rainfall_override if rainfall_override is not None else 0.0,
            "visibility_km": visibility_override if visibility_override is not None else 10.0,
            "wind_speed": 8.0, "temperature": 27.0, "source": "user_override",
        }
    else:
        weather = external_apis.get_real_weather(
            (src["lat"] + dst["lat"]) / 2, (src["lon"] + dst["lon"]) / 2
        )

    # --- live congestion: sample points along the REAL route geometry ---
    sample_points = external_apis.sample_points_along_route(real_route["geometry"], n_points=6)
    live_samples = []
    for lat, lon in sample_points:
        flow = external_apis.get_real_traffic_flow(lat, lon)
        if flow:
            live_samples.append({"lat": lat, "lon": lon, **flow})

    tomtom_configured = bool(config.TOMTOM_API_KEY)
    if live_samples:
        ratios = [s["speed_ratio"] for s in live_samples]
        avg_ratio = sum(ratios) / len(ratios)
        worst = min(live_samples, key=lambda s: s["speed_ratio"])
        congestion_source = "live"
    else:
        # fall back to the physics-based daily profile as a stand-in
        # ratio, clearly labeled as an estimate, not invented as live.
        # Calibrated against realistic Bangalore peak-hour slowdowns
        # (worst case ~45-55% of free-flow speed, not near-standstill).
        hour = datetime.now().hour
        profile = daily_traffic_profile(hour)  # 0.15 (quiet) .. 1.0 (peak)
        avg_ratio = float(np.clip(1.0 - 0.55 * profile, 0.4, 1.0))
        worst = {"lat": (src["lat"] + dst["lat"]) / 2, "lon": (src["lon"] + dst["lon"]) / 2,
                 "speed_ratio": avg_ratio}
        congestion_source = "not_configured" if not tomtom_configured else "fallback_estimate"

    weather_penalty = 1.0
    if weather["rainfall_mm"] > 10:
        weather_penalty = 0.75
    elif weather["rainfall_mm"] > 2:
        weather_penalty = 0.9
    effective_ratio = max(min(avg_ratio * weather_penalty, 1.0), 0.1)

    live_duration_min = real_route["duration_min"] / effective_ratio

    # --- named road segments from OSRM, each tagged with the nearest
    #     live sample's congestion class ---
    steps = real_route["steps"] or [{
        "road_name": f"{src['name']} to {dst['name']}",
        "distance_km": real_route["distance_km"], "duration_min": real_route["duration_min"],
        "maneuver": None,
    }]
    n_samples = max(len(live_samples), 1)
    segments = []
    for i, step in enumerate(steps):
        sample_idx = min(int(i / max(len(steps), 1) * n_samples), n_samples - 1)
        ratio = live_samples[sample_idx]["speed_ratio"] if live_samples else avg_ratio
        segments.append({
            "road_name": step["road_name"],
            "distance_km": step["distance_km"],
            "congestion_class": _speed_ratio_to_class(ratio),
            "speed_ratio": round(ratio, 2),
        })

    # --- overlay known Bangalore hotspots on the real geometry ---
    known_issues = []
    for lat, lon in sample_points or [(worst["lat"], worst["lon"])]:
        known_issues.extend(nearby_known_issues(lat, lon, rainfall_mm=weather["rainfall_mm"]))
    seen = set()
    dedup_issues = []
    for issue in known_issues:
        if issue["landmark"] not in seen:
            seen.add(issue["landmark"])
            dedup_issues.append(issue)

    # --- real Traffic Health Score for this route, using the same
    #     formula as the Predict console, fed by the live speed ratio ---
    occupancy_proxy = max(min(1 - effective_ratio, 0.98), 0.02)
    nominal_capacity, nominal_speed_limit = 2000, 50
    ths = score_reading(
        vehicle_count=round(occupancy_proxy * nominal_capacity), road_capacity=nominal_capacity,
        average_speed=round(effective_ratio * nominal_speed_limit, 1), speed_limit=nominal_speed_limit,
        occupancy_rate=occupancy_proxy, rainfall=weather["rainfall_mm"], visibility=weather["visibility_km"],
        wind_speed=weather.get("wind_speed", 10),
        incident_exists=effective_ratio < 0.35, severity=3 if effective_ratio < 0.35 else 0,
    )

    # --- illustrative "if an incident happened at the bottleneck right
    #     now" clearance estimate - clearly labeled hypothetical, not a
    #     live incident report ---
    hypothetical_clearance = estimate_clearance_time(
        vehicle_count=round(occupancy_proxy * nominal_capacity), occupancy_rate=occupancy_proxy,
        road_type="arterial", number_of_lanes=3, blocked_lanes=1, severity=2,
    )

    return {
        "source": src["name"],
        "destination": dst["name"],
        "source_coords": {"lat": src["lat"], "lon": src["lon"]},
        "destination_coords": {"lat": dst["lat"], "lon": dst["lon"]},
        "total_distance_km": real_route["distance_km"],
        "total_time_min": round(live_duration_min, 1),
        "free_flow_time_min": real_route["duration_min"],
        "overall_congestion_class": _speed_ratio_to_class(effective_ratio),
        "overall_speed_ratio": round(effective_ratio, 2),
        "traffic_health_score": ths,
        "hypothetical_incident_clearance_min": hypothetical_clearance["clearance_time_minutes"],
        "bottleneck_point": {"lat": worst["lat"], "lon": worst["lon"],
                              "speed_ratio": round(worst["speed_ratio"], 2)},
        "segments": segments,
        "weather": weather,
        "known_issues": dedup_issues,
        "route_geometry": real_route["geometry"],
        "data_sources": {
            "routing": "live (OSRM real road network)",
            "congestion": {
                "live": "live (TomTom real-time traffic flow)",
                "fallback_estimate": "fallback_estimate (TomTom key present but call failed - physics-based estimate used)",
                "not_configured": "fallback_estimate (add TOMTOM_API_KEY in .env for live congestion)",
            }[congestion_source],
            "weather": weather["source"],
        },
    }


# =================================================================
# FALLBACK PATH: old fixed-graph Dijkstra + physics simulation
# (used only if OSRM is unreachable, e.g. no internet at the venue)
# =================================================================
def _fallback_find_path(source_id, dest_id):
    if source_id not in GRAPH or dest_id not in GRAPH:
        raise UnknownLandmarkError(f"Unknown landmark id: {source_id if source_id not in GRAPH else dest_id}")

    dist = {n: float("inf") for n in GRAPH}
    prev = {}
    dist[source_id] = 0
    pq = [(0, source_id)]
    visited = set()

    while pq:
        d, node = heapq.heappop(pq)
        if node in visited:
            continue
        visited.add(node)
        if node == dest_id:
            break
        for edge in GRAPH[node]:
            nd = d + edge["distance_km"]
            if nd < dist[edge["to"]]:
                dist[edge["to"]] = nd
                prev[edge["to"]] = (node, edge)
                heapq.heappush(pq, (nd, edge["to"]))

    if dist[dest_id] == float("inf"):
        return None

    path_edges = []
    node = dest_id
    while node != source_id:
        prev_node, edge = prev[node]
        path_edges.append(edge)
        node = prev_node
    path_edges.reverse()
    return path_edges


def _simulate_segment_vehicle_counts(edge, hour, rng):
    profile = daily_traffic_profile(hour)
    bottleneck_boost = 1.35 if edge["chronic_bottleneck"] else 1.0
    base = profile * bottleneck_boost * edge["lanes"] * 8
    car = max(int(base * rng.uniform(0.85, 1.15)), 1)
    bike = max(int(base * 0.35 * rng.uniform(0.8, 1.2)), 0)
    bus = max(int(base * 0.04 * rng.uniform(0.5, 1.5)), 0)
    truck = max(int(base * 0.03 * rng.uniform(0.5, 1.5)), 0)
    return car, bike, bus, truck


def _fallback_predict_segment(edge, hour, rainfall_mm, visibility_km, seed=None):
    rng = np.random.default_rng(seed)
    car, bike, bus, truck = _simulate_segment_vehicle_counts(edge, hour, rng)
    req = PredictRequest(
        car_count=car, bike_count=bike, bus_count=bus, truck_count=truck,
        average_speed=edge["speed_limit"] * 0.7,
        road_type=edge["road_type"], number_of_lanes=edge["lanes"],
        speed_limit=edge["speed_limit"], road_length_km=edge["distance_km"],
        rainfall_mm=rainfall_mm, visibility_km=visibility_km,
        hour=hour, is_weekend=False, is_holiday=False,
    )
    row = build_feature_row(req)
    occ = row["occupancy_rate"]
    bottleneck_penalty = 0.15 if edge["chronic_bottleneck"] else 0.0
    refined_speed = edge["speed_limit"] * max(1 - 0.75 * occ - bottleneck_penalty, 0.08)
    req.average_speed = round(refined_speed, 1)
    row = build_feature_row(req)

    congestion_bundle = model_repository.congestion_bundle
    X = encode_for_model(row, congestion_bundle)
    proba = congestion_bundle["model"].predict_proba(X)[0]
    pred_class = int(np.argmax(proba))
    travel_time_min = (edge["distance_km"] / max(refined_speed, 3)) * 60

    return {
        "to": LANDMARKS[edge["to"]]["name"], "corridor": edge["corridor"],
        "distance_km": edge["distance_km"], "congestion_class": CONGESTION_LABELS[pred_class],
        "predicted_speed_kmh": round(refined_speed, 1), "travel_time_min": round(travel_time_min, 1),
        "chronic_bottleneck": edge["chronic_bottleneck"], "flood_prone": edge["flood_prone"],
    }


def _build_fallback_route(source_id, dest_id, hour, rainfall_mm, visibility_km):
    path = _fallback_find_path(source_id, dest_id)
    if path is None:
        raise RouteNotFoundError(f"No route found between {source_id} and {dest_id}")

    segments = []
    for edge in path:
        seg = _fallback_predict_segment(edge, hour, rainfall_mm, visibility_km,
                                         seed=hash((edge["from"], edge["to"], hour)) % (2**31))
        seg["road_name"] = f"{LANDMARKS[edge['from']]['name']} -> {seg['to']} ({seg['corridor']})"
        seg["speed_ratio"] = round(seg["predicted_speed_kmh"] / edge["speed_limit"], 2)
        segments.append(seg)

    total_distance = sum(s["distance_km"] for s in segments)
    total_time = sum(s["travel_time_min"] for s in segments)
    severity_rank = {"Low": 0, "Moderate": 1, "High": 2, "Severe": 3}
    bottleneck_idx = max(range(len(segments)), key=lambda i: severity_rank[segments[i]["congestion_class"]])
    bottleneck = segments[bottleneck_idx]
    bottleneck_edge = path[bottleneck_idx]
    known_issues = [{"landmark": s["to"], "issue": "known chronic bottleneck (offline knowledge base)",
                      "flood_prone": s["flood_prone"], "lat": LANDMARKS[edge["to"]]["lat"], "lon": LANDMARKS[edge["to"]]["lon"]}
                     for edge, s in zip(path, segments) if s["chronic_bottleneck"] or (s["flood_prone"] and rainfall_mm > 0)]

    occupancy_proxy = max(min(1 - bottleneck["speed_ratio"], 0.98), 0.02)
    nominal_capacity, nominal_speed_limit = 2000, 50
    ths = score_reading(
        vehicle_count=round(occupancy_proxy * nominal_capacity), road_capacity=nominal_capacity,
        average_speed=bottleneck["predicted_speed_kmh"], speed_limit=nominal_speed_limit,
        occupancy_rate=occupancy_proxy, rainfall=rainfall_mm, visibility=visibility_km, wind_speed=10,
        incident_exists=bottleneck["congestion_class"] == "Severe", severity=3 if bottleneck["congestion_class"] == "Severe" else 0,
    )
    hypothetical_clearance = estimate_clearance_time(
        vehicle_count=round(occupancy_proxy * nominal_capacity), occupancy_rate=occupancy_proxy,
        road_type="arterial", number_of_lanes=3, blocked_lanes=1, severity=2,
    )

    return {
        "source": LANDMARKS[source_id]["name"], "destination": LANDMARKS[dest_id]["name"],
        "source_coords": {"lat": LANDMARKS[source_id]["lat"], "lon": LANDMARKS[source_id]["lon"]},
        "destination_coords": {"lat": LANDMARKS[dest_id]["lat"], "lon": LANDMARKS[dest_id]["lon"]},
        "total_distance_km": round(total_distance, 1), "total_time_min": round(total_time, 1),
        "free_flow_time_min": round(total_time, 1),
        "overall_congestion_class": bottleneck["congestion_class"],
        "traffic_health_score": ths,
        "hypothetical_incident_clearance_min": hypothetical_clearance["clearance_time_minutes"],
        "bottleneck_point": {"lat": LANDMARKS[bottleneck_edge["to"]]["lat"], "lon": LANDMARKS[bottleneck_edge["to"]]["lon"],
                              "speed_ratio": bottleneck["speed_ratio"]},
        "segments": segments,
        "weather": {"rainfall_mm": rainfall_mm, "visibility_km": visibility_km, "source": "user_override"},
        "known_issues": known_issues, "route_geometry": None,
        "data_sources": {
            "routing": "fallback_estimate (OSRM unreachable - offline 27-landmark graph used)",
            "congestion": "fallback_estimate (ML model trained on synthetic data)",
            "weather": "user_override",
        },
    }


# =================================================================
# PUBLIC API
# =================================================================
def predict_route(source_id, dest_id, hour=None, rainfall_mm=None, visibility_km=None):
    if hour is None:
        hour = datetime.now().hour

    real = _build_real_route(source_id, dest_id, rainfall_mm, visibility_km)
    if real is not None:
        real["hour"] = hour
        return real

    return _build_fallback_route(
        source_id, dest_id, hour,
        rainfall_mm if rainfall_mm is not None else 0.0,
        visibility_km if visibility_km is not None else 10.0,
    )


def get_route_alternatives(source_id, dest_id, hour=None, rainfall_mm=None, visibility_km=None):
    """Real alternative routes (not just the single fastest path),
    each with its own live congestion/weather overlay. Falls back to
    a single route (wrapped in a 1-item list) if OSRM alternatives
    aren't available - the frontend can treat the response uniformly
    either way."""
    if hour is None:
        hour = datetime.now().hour
    src, dst = _resolve_landmark(source_id), _resolve_landmark(dest_id)

    raw_alternatives = external_apis.get_real_route_alternatives(
        src["lat"], src["lon"], dst["lat"], dst["lon"]
    )
    if not raw_alternatives:
        single = predict_route(source_id, dest_id, hour, rainfall_mm, visibility_km)
        single["is_recommended"] = True
        single["label"] = "Only route available"
        return [single]

    processed = []
    for i, real_route in enumerate(raw_alternatives):
        route = _process_route_with_live_data(real_route, src, dst, rainfall_mm, visibility_km)
        route["hour"] = hour
        processed.append(route)

    # sort by live-adjusted total time, fastest first
    processed.sort(key=lambda r: r["total_time_min"])
    labels = ["Fastest", "Alternate", "Alternate 2"]
    for i, route in enumerate(processed):
        route["is_recommended"] = (i == 0)
        route["label"] = labels[i] if i < len(labels) else f"Alternate {i}"
    return processed


def best_time_to_leave(source_id, dest_id, hours_ahead=6, rainfall_mm=None, visibility_km=None):
    """Scans the next N hours. The *current* hour uses a real live
    sample (OSRM+TomTom+Open-Meteo); future hours are estimated by
    scaling that live sample against the known Bangalore daily traffic
    curve - i.e. calibrated by one real data point rather than pure
    simulation, and clearly labeled as such."""
    now = datetime.now()
    now_route = predict_route(source_id, dest_id, hour=now.hour,
                               rainfall_mm=rainfall_mm, visibility_km=visibility_km)
    live_ratio_now = now_route.get("overall_speed_ratio")
    is_live = "live" in now_route["data_sources"]["congestion"]

    results = []
    for h in range(hours_ahead + 1):
        dep_time = now + timedelta(hours=h)
        if h == 0:
            total_time = now_route["total_time_min"]
            cls = now_route["overall_congestion_class"]
        elif is_live and live_ratio_now:
            profile_now = daily_traffic_profile(now.hour)
            profile_h = daily_traffic_profile(dep_time.hour)
            scale = (profile_h + 0.2) / (profile_now + 0.2)
            projected_ratio = float(np.clip(1 - (1 - live_ratio_now) * scale, 0.1, 1.0))
            total_time = round(now_route["free_flow_time_min"] / projected_ratio, 1)
            cls = _speed_ratio_to_class(projected_ratio)
        else:
            route = predict_route(source_id, dest_id, hour=dep_time.hour,
                                   rainfall_mm=rainfall_mm, visibility_km=visibility_km)
            total_time = route["total_time_min"]
            cls = route["overall_congestion_class"]

        results.append({
            "departure_hour": dep_time.hour, "departure_label": dep_time.strftime("%I:%M %p"),
            "total_time_min": total_time, "congestion_class": cls,
        })

    best = min(results, key=lambda r: r["total_time_min"])
    worst = max(results, key=lambda r: r["total_time_min"])

    return {
        "source": now_route["source"], "destination": now_route["destination"],
        "options": results, "recommended": best, "worst": worst,
        "potential_savings_min": round(worst["total_time_min"] - best["total_time_min"], 1),
        "calibration": "live_sample_scaled_by_daily_profile" if is_live else "fallback_estimate",
    }


if __name__ == "__main__":
    r = predict_route("whitefield", "electronic_city", hour=9)
    print(f"{r['source']} -> {r['destination']}: {r['total_distance_km']} km, {r['total_time_min']} min")
    print(f"Data sources: {r['data_sources']}")
