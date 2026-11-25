"""
Traffic Guardian AI - Raw Data Loading & Merging (database layer)

Loads raw sources, merges into a single timeline, applies the
missing-value strategy defined per-feature in Feature Store v2.0.

Lineage: Raw Data -> Preprocessing -> (feature_engineering.py continues)
"""
import pandas as pd
from app.config import settings as config


def load_raw():
    traffic = pd.read_csv(f"{config.RAW_DIR}/traffic_sensors.csv", parse_dates=["timestamp"])
    weather = pd.read_csv(f"{config.RAW_DIR}/weather.csv", parse_dates=["timestamp"])
    road = pd.read_csv(f"{config.RAW_DIR}/road_network.csv")
    holiday = pd.read_csv(f"{config.RAW_DIR}/holiday_calendar.csv", parse_dates=["date"])
    incident = pd.read_csv(f"{config.RAW_DIR}/incidents.csv", parse_dates=["timestamp"])
    yolo = pd.read_csv(f"{config.RAW_DIR}/yolo_edge.csv", parse_dates=["timestamp"])
    return traffic, weather, road, holiday, incident, yolo


def merge_sources(traffic, weather, road, holiday, incident, yolo):
    # Traffic + Road (static attributes per segment)
    df = traffic.merge(road, on="road_segment_id", how="left")

    # + Weather (city-level, joined on timestamp)
    df = df.merge(weather, on="timestamp", how="left")

    # + Holiday flag
    df["date"] = df["timestamp"].dt.normalize()
    df = df.merge(holiday, on="date", how="left")
    df["is_holiday"] = df["is_holiday"].fillna(False)
    df = df.drop(columns=["date"])

    # + Incidents (sparse - most rows have no incident at that timestamp/segment)
    df = df.merge(
        incident.drop(columns=["clearance_time_minutes"]),
        on=["timestamp", "road_segment_id"], how="left",
    )
    df["incident_exists"] = df["incident_exists"].fillna(False)
    df["severity"] = df["severity"].fillna(0).astype(int)
    df["blocked_lanes"] = df["blocked_lanes"].fillna(0).astype(int)
    df["emergency_vehicle_required"] = df["emergency_vehicle_required"].fillna(False)
    df["incident_type"] = df["incident_type"].fillna("none")

    # + YOLO edge data (only exists for segments with a deployed Pi device)
    df = df.merge(yolo, on=["timestamp", "road_segment_id"], how="left")
    for col in ["car_count", "bike_count", "bus_count", "truck_count", "heavy_vehicle_ratio"]:
        df[col] = df[col].fillna(0)
    df["has_edge_device"] = df["heavy_vehicle_ratio"].notna() | (df["car_count"] > 0)

    return df


def apply_missing_value_strategy(df):
    """Per Feature Store v2.0 missing-value strategy table."""
    median_cols = ["vehicle_count", "occupancy_rate"]
    mean_cols = ["average_speed", "travel_time", "traffic_volume"]

    for col in median_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    for col in mean_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # weather - forward fill (weather is temporally continuous)
    weather_cols = ["temperature", "humidity", "rainfall", "visibility",
                     "wind_speed", "precipitation_probability"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill()

    # categorical "unknown" fallback
    for col in ["sensor_id", "road_segment_id", "weather_condition",
                "road_type", "intersection_type"]:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # drop rows with no timestamp at all (per spec: timestamp missing -> drop)
    df = df.dropna(subset=["timestamp"])

    return df


def run(save=True):
    traffic, weather, road, holiday, incident, yolo = load_raw()
    merged = merge_sources(traffic, weather, road, holiday, incident, yolo)
    cleaned = apply_missing_value_strategy(merged)

    if save:
        cleaned.to_csv(f"{config.PROCESSED_DIR}/merged_clean.csv", index=False)
        # keep incident-with-target table separate for the clearance model
        incident.to_csv(f"{config.PROCESSED_DIR}/incident_targets.csv", index=False)

    return cleaned


if __name__ == "__main__":
    df = run()
    print("Merged & cleaned shape:", df.shape)
    print("Missing values remaining:\n", df.isna().sum()[df.isna().sum() > 0])
