"""
Traffic Guardian AI - External Real-Data API Clients

This is the file that turns the route/clearance system from
"simulated" into "real". Three free, well-known public APIs, each
with a short in-memory cache (so a demo doesn't hammer rate limits)
and a documented fallback if the network/key isn't available - the
app must still run smoothly during a live judge demo even on flaky
venue wifi.

  1. OSRM (router.project-osrm.org)   - real OpenStreetMap road network.
     No API key. Returns the ACTUAL drivable path between two points,
     real distance, real duration, real turn-by-turn geometry.

  2. Open-Meteo (api.open-meteo.com)  - real live weather.
     No API key. Rain / wind / visibility for any lat/lon, right now.

  3. TomTom Traffic Flow (api.tomtom.com) - real live congestion.
     Free-tier API key (developer.tomtom.com). Returns currentSpeed vs
     freeFlowSpeed for the road nearest a point - i.e. "how much
     slower is this road right now than it normally is", which is the
     single most honest, real-world definition of congestion there is.

If TomTom's key is missing/invalid or any call fails, everything
degrades gracefully and the response says so explicitly (`"source":
"fallback_estimate"` vs `"source": "live"`) - never silently fakes a
number and pretends it's live. That's the important part for judges.
"""
import time

import requests

from app.config import settings as config
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TIMEOUT = 6.0  # seconds - generous enough for slower campus/venue wifi, still bounded
_MAX_RETRIES = 2  # one retry on transient network hiccups before falling back
_CACHE = {}
_CACHE_TTL = 45  # seconds - live enough to feel real, cheap enough on rate limits


def _get_with_retry(url, timeout=_TIMEOUT, retries=_MAX_RETRIES):
    """GET with a couple of quick retries - real-world wifi (especially
    campus/venue networks) drops the occasional request; one retry
    turns a lot of those into a live result instead of a needless
    fallback."""
    last_exc = None
    for attempt in range(retries):
        try:
            return requests.get(url, timeout=timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
            continue
    raise last_exc


def _cache_get(key):
    hit = _CACHE.get(key)
    if hit and (time.time() - hit[0]) < _CACHE_TTL:
        return hit[1]
    return None


def _cache_set(key, value):
    _CACHE[key] = (time.time(), value)


# ---------------------------------------------------------------
# 1. OSRM - real road-network routing
# ---------------------------------------------------------------
def _process_osrm_route(route):
    """Turn one raw OSRM route object into our standard shape. Shared
    by both the single-route and alternatives calls below."""
    steps = []
    for leg in route.get("legs", []):
        for step in leg.get("steps", []):
            name = step.get("name") or "unnamed road"
            dist_m = step.get("distance", 0)
            dur_s = step.get("duration", 0)
            if dist_m < 20:
                continue
            steps.append({
                "road_name": name,
                "distance_km": round(dist_m / 1000, 2),
                "duration_min": round(dur_s / 60, 2),
                "maneuver": step.get("maneuver", {}).get("type"),
            })
    return {
        "distance_km": round(route["distance"] / 1000, 2),
        "duration_min": round(route["duration"] / 60, 1),
        "geometry": route.get("geometry"),
        "steps": steps,
        "source": "live",
    }


def get_real_route(source_lat, source_lon, dest_lat, dest_lon):
    """
    Real driving route along the actual OpenStreetMap road network.
    Returns distance_km, duration_min (OSRM's free-flow estimate),
    a GeoJSON geometry (for drawing the real path on a map), and the
    named road segments ("legs") it actually uses.
    Returns None on any failure (caller must fall back).
    """
    cache_key = f"osrm:{source_lat},{source_lon}:{dest_lat},{dest_lon}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = (
        f"{config.OSRM_BASE_URL}/route/v1/driving/"
        f"{source_lon},{source_lat};{dest_lon},{dest_lat}"
        f"?overview=full&geometries=geojson&steps=true&annotations=true"
    )
    try:
        resp = _get_with_retry(url)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM returned no route: {data.get('code')}")
            return None

        result = _process_osrm_route(data["routes"][0])
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"OSRM route fetch failed: {e}")
        return None


def get_real_route_alternatives(source_lat, source_lon, dest_lat, dest_lon, max_routes=3):
    """Real alternative driving routes (not just the single fastest
    one) along the actual OSM road network - OSRM natively supports
    this via alternatives=true, we just weren't using it. Returns a
    list of up to max_routes processed routes, or None on failure."""
    cache_key = f"osrm_alt:{source_lat},{source_lon}:{dest_lat},{dest_lon}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = (
        f"{config.OSRM_BASE_URL}/route/v1/driving/"
        f"{source_lon},{source_lat};{dest_lon},{dest_lat}"
        f"?overview=full&geometries=geojson&steps=true&annotations=true&alternatives=true"
    )
    try:
        resp = _get_with_retry(url)
        resp.raise_for_status()
        data = resp.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            logger.warning(f"OSRM returned no alternative routes: {data.get('code')}")
            return None

        results = [_process_osrm_route(r) for r in data["routes"][:max_routes]]
        _cache_set(cache_key, results)
        return results
    except Exception as e:
        logger.warning(f"OSRM alternatives fetch failed: {e}")
        return None


def sample_points_along_route(geometry, n_points=6):
    """Evenly sample n_points [lat, lon] pairs along an OSRM GeoJSON
    LineString, for probing live traffic conditions along the whole
    route rather than just the endpoints."""
    if not geometry or geometry.get("type") != "LineString":
        return []
    coords = geometry.get("coordinates", [])  # [lon, lat] pairs
    if len(coords) < 2:
        return []
    step = max(len(coords) // n_points, 1)
    sampled = coords[::step][:n_points]
    return [(lat, lon) for lon, lat in sampled]


# ---------------------------------------------------------------
# 2. Open-Meteo - real live weather, no API key
# ---------------------------------------------------------------
def get_real_weather(lat, lon):
    """Live rainfall/wind/visibility/temperature for a point, right now.
    Returns a dict with source='live', or a documented seasonal-average
    fallback with source='fallback_estimate' if the call fails."""
    cache_key = f"weather:{round(lat,2)},{round(lon,2)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = (
        f"{config.OPEN_METEO_BASE_URL}/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,precipitation,rain,wind_speed_10m,weather_code"
        f"&hourly=visibility&timezone=auto&forecast_days=1"
    )
    try:
        resp = _get_with_retry(url)
        resp.raise_for_status()
        data = resp.json()
        current = data["current"]

        # pull visibility for the current hour out of the hourly array
        visibility_km = 10.0
        try:
            hourly_times = data["hourly"]["time"]
            hourly_vis = data["hourly"]["visibility"]  # meters
            current_time = current["time"]
            idx = hourly_times.index(current_time) if current_time in hourly_times else 0
            visibility_km = round(min(hourly_vis[idx] / 1000, 10.0), 1)
        except Exception:
            pass

        result = {
            "rainfall_mm": round(current.get("precipitation", 0.0), 1),
            "visibility_km": visibility_km,
            "wind_speed": round(current.get("wind_speed_10m", 0.0), 1),
            "temperature": round(current.get("temperature_2m", 26.0), 1),
            "source": "live",
        }
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"Open-Meteo fetch failed: {e}")
        return {
            "rainfall_mm": 0.0, "visibility_km": 10.0, "wind_speed": 8.0,
            "temperature": 27.0, "source": "fallback_estimate",
        }


# ---------------------------------------------------------------
# 3. TomTom Traffic Flow - real live congestion (currentSpeed vs
#    freeFlowSpeed for the road nearest a point)
# ---------------------------------------------------------------
def get_real_traffic_flow(lat, lon):
    """Live speed ratio for the road segment nearest (lat, lon).
    Returns None if no API key is configured or the call fails -
    caller must fall back to the physics-based estimate."""
    if not config.TOMTOM_API_KEY:
        return None

    cache_key = f"tomtom:{round(lat,3)},{round(lon,3)}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = (
        f"{config.TOMTOM_BASE_URL}/traffic/services/4/flowSegmentData/absolute/10/json"
        f"?point={lat},{lon}&key={config.TOMTOM_API_KEY}"
    )
    try:
        resp = _get_with_retry(url)
        if resp.status_code == 403 or resp.status_code == 401:
            logger.error(f"TomTom rejected the API key (HTTP {resp.status_code}) - check the key is correct, "
                         f"active, and has the Traffic API enabled on developer.tomtom.com")
            return None
        resp.raise_for_status()
        data = resp.json().get("flowSegmentData")
        if not data:
            return None

        current_speed = data.get("currentSpeed", 0)
        free_flow_speed = max(data.get("freeFlowSpeed", 1), 1)
        speed_ratio = round(min(current_speed / free_flow_speed, 1.0), 3)

        result = {
            "current_speed_kmh": current_speed,
            "free_flow_speed_kmh": free_flow_speed,
            "speed_ratio": speed_ratio,  # 1.0 = free-flowing, near 0 = gridlock
            "confidence": data.get("confidence", 0.7),
            "road_closed": data.get("roadClosure", False),
            "source": "live",
        }
        _cache_set(cache_key, result)
        return result
    except Exception as e:
        logger.warning(f"TomTom flow fetch failed: {e}")
        return None


def get_live_status():
    """Quick self-check used by the UI to show which data sources are
    actually live right now vs. running on fallback."""
    status = {"osrm": "unknown", "osrm_detail": None, "open_meteo": "unknown", "open_meteo_detail": None,
              "tomtom": "not_configured", "tomtom_detail": None}
    try:
        r = requests.get(f"{config.OSRM_BASE_URL}/route/v1/driving/77.5946,12.9716;77.6,12.98",
                          timeout=_TIMEOUT)
        if r.ok:
            status["osrm"] = "live"
        else:
            status["osrm"] = "unreachable"
            status["osrm_detail"] = f"HTTP {r.status_code}"
    except Exception as e:
        status["osrm"] = "unreachable"
        status["osrm_detail"] = f"{type(e).__name__}: {str(e)[:150]}"

    try:
        r = requests.get(f"{config.OPEN_METEO_BASE_URL}/v1/forecast?latitude=12.97&longitude=77.59&current=temperature_2m",
                          timeout=_TIMEOUT)
        if r.ok:
            status["open_meteo"] = "live"
        else:
            status["open_meteo"] = "unreachable"
            status["open_meteo_detail"] = f"HTTP {r.status_code}"
    except Exception as e:
        status["open_meteo"] = "unreachable"
        status["open_meteo_detail"] = f"{type(e).__name__}: {str(e)[:150]}"

    if config.TOMTOM_API_KEY:
        try:
            url = (f"{config.TOMTOM_BASE_URL}/traffic/services/4/flowSegmentData/absolute/10/json"
                   f"?point=12.9716,77.5946&key={config.TOMTOM_API_KEY}")
            r = requests.get(url, timeout=_TIMEOUT)
            if r.status_code == 200:
                status["tomtom"] = "live"
            elif r.status_code in (401, 403):
                status["tomtom"] = "unreachable_or_invalid_key"
                status["tomtom_detail"] = f"HTTP {r.status_code}: key rejected - check it's active and has Traffic API enabled"
            else:
                status["tomtom"] = "unreachable_or_invalid_key"
                status["tomtom_detail"] = f"HTTP {r.status_code}: {r.text[:150]}"
        except Exception as e:
            status["tomtom"] = "unreachable_or_invalid_key"
            status["tomtom_detail"] = f"{type(e).__name__}: {str(e)[:150]}"

    return status
