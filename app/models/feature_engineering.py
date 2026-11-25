"""
Traffic Guardian AI - Feature Engineering Stage
Implements Feature Store v2.0 exactly: temporal, engineered features
(Group G), and the Traffic Health Score (novelty feature).

Lineage: Preprocessing -> Transformation -> Feature Engineering ->
         Scaling/Encoding (models/) -> Model
"""
import numpy as np
import pandas as pd
from app.config import settings as config


# ---------------------------------------------------------------
# GROUP A: Temporal features
# ---------------------------------------------------------------
def add_temporal_features(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["minute"] = ts.dt.minute
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = ts.dt.dayofweek >= 5
    hour_frac = df["hour"] + df["minute"] / 60.0
    df["peak_hour"] = ((hour_frac.between(7, 10)) | (hour_frac.between(17, 20)))
    return df


# ---------------------------------------------------------------
# GROUP G: Engineered features (formulas per your spec)
# ---------------------------------------------------------------
def add_engineered_features(df):
    eps = 1e-6

    # Feature 1: Traffic Density
    df["traffic_density"] = df["vehicle_count"] / (df["road_capacity"] + eps)

    # Feature 2: Speed Ratio
    df["speed_ratio"] = df["average_speed"] / (df["speed_limit"] + eps)
    df["speed_ratio"] = df["speed_ratio"].clip(0, 1.5)

    # Feature 3: Congestion Index = density * (1 - speed_ratio), normalized version
    df["congestion_index"] = (df["traffic_density"] * (1 - df["speed_ratio"])).clip(0, None)

    # Feature 4: Weather Severity Score (weighted, 0-100)
    rain_norm = (df["rainfall"] / (df["rainfall"].max() + eps)).clip(0, 1)
    vis_risk = (1 - df["visibility"] / 10).clip(0, 1)
    wind_norm = (df["wind_speed"] / (df["wind_speed"].max() + eps)).clip(0, 1)
    df["weather_severity_score"] = (
        rain_norm * 0.4 + vis_risk * 0.4 + wind_norm * 0.2
    ) * 100

    # Feature 5: Road Complexity Score (0-100)
    type_risk = df["road_type"].map(
        {"local": 0.2, "collector": 0.4, "arterial": 0.7, "highway": 0.5}
    ).fillna(0.4)
    lane_risk = (1 / (df["number_of_lanes"] + 1)).clip(0, 1)
    junction_risk = df["junction"].astype(int) * 0.3
    intersection_risk = df["intersection_type"].map(
        {"signalized": 0.3, "roundabout": 0.4, "uncontrolled": 0.8, "none": 0.1}
    ).fillna(0.3)
    df["road_complexity_score"] = (
        (type_risk * 0.3 + lane_risk * 0.2 + junction_risk + intersection_risk * 0.3) * 100
    ).clip(0, 100)

    # Feature 6: Traffic Growth Rate (per sensor, time-ordered)
    df = df.sort_values(["sensor_id", "timestamp"])
    prev_count = df.groupby("sensor_id")["vehicle_count"].shift(1)
    df["traffic_growth_rate"] = ((df["vehicle_count"] - prev_count) / (prev_count + eps)).fillna(0)
    df["traffic_growth_rate"] = df["traffic_growth_rate"].clip(-5, 5)

    # Feature 7 & 8: Rolling averages (30 min = 6 rows at 5-min freq)
    window = 30 // config.FREQ_MINUTES
    grp = df.groupby("sensor_id")
    df["rolling_avg_speed"] = grp["average_speed"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )
    df["rolling_avg_volume"] = grp["vehicle_count"].transform(
        lambda s: s.rolling(window, min_periods=1).mean()
    )

    # Feature 9: Rolling speed variance (instability)
    df["rolling_speed_variance"] = grp["average_speed"].transform(
        lambda s: s.rolling(window, min_periods=1).std()
    ).fillna(0)

    # Queue Density Score (supports THS)
    df["queue_density_score"] = (
        df["traffic_density"] * 0.5 + df["occupancy_rate"] * 0.5
    ) * 100

    return df.reset_index(drop=True)


# ---------------------------------------------------------------
# GROUP G Feature 10: Traffic Health Score (THS) — the novelty metric
# ---------------------------------------------------------------
def compute_ths(df):
    """
    THS = 100 - Congestion Penalty - Weather Penalty - Incident Penalty
    Each penalty is 0-100 scaled and weighted before subtraction so no
    single factor can push the score negative on its own.
    """
    congestion_penalty = (
        df["congestion_index"].clip(0, 1) * 40 +
        (1 - df["speed_ratio"]).clip(0, 1) * 15 +
        (df["queue_density_score"] / 100).clip(0, 1) * 15
    )  # max 70

    weather_penalty = (df["weather_severity_score"] / 100).clip(0, 1) * 15  # max 15

    incident_penalty = np.where(
        df["incident_exists"],
        (df["severity"] / 5) * 10 + (df["blocked_lanes"] / 3) * 5,
        0,
    )  # max 15

    ths = 100 - congestion_penalty - weather_penalty - incident_penalty
    df["traffic_health_score"] = ths.clip(0, 100).round(1)

    df["ths_band"] = pd.cut(
        df["traffic_health_score"],
        bins=[-1, 49, 69, 89, 100],
        labels=["Critical", "Busy", "Moderate", "Healthy"],
    )
    return df


# ---------------------------------------------------------------
# TARGET: Congestion class (derived from congestion_index quantiles)
# ---------------------------------------------------------------
def add_congestion_target(df):
    q = df["congestion_index"].quantile([0.5, 0.8, 0.95]).values
    conditions = [
        df["congestion_index"] <= q[0],
        df["congestion_index"] <= q[1],
        df["congestion_index"] <= q[2],
    ]
    df["congestion_class"] = np.select(conditions, [0, 1, 2], default=3)
    return df


def run(save=True):
    df = pd.read_csv(f"{config.PROCESSED_DIR}/merged_clean.csv", parse_dates=["timestamp"])
    df = add_temporal_features(df)
    df = add_engineered_features(df)
    df = compute_ths(df)
    df = add_congestion_target(df)

    if save:
        df.to_csv(f"{config.PROCESSED_DIR}/master_feature_store.csv", index=False)

    return df


if __name__ == "__main__":
    df = run()
    print("Master feature store shape:", df.shape)
    print("Columns:", len(df.columns))
    print(df[["timestamp", "sensor_id", "congestion_index", "congestion_class",
              "traffic_health_score", "ths_band"]].head())
