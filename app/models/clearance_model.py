"""
Traffic Guardian AI - Model 2: Clearance Time Regression
Target: clearance_time_minutes (only known incidents have a target)

LEAKAGE NOTE: incident_duration is intentionally excluded (see Feature
Store v2.0 Group E). We only use fields known AT INCIDENT REPORT TIME:
severity, blocked_lanes, incident_type, emergency_vehicle_required,
plus road/weather/traffic context at that moment.
"""
import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from app.config import settings as config

FEATURE_COLS = [
    "severity", "blocked_lanes", "emergency_vehicle_required",
    "road_type", "number_of_lanes", "speed_limit", "road_capacity",
    "junction", "intersection_type", "road_length",
    "temperature", "rainfall", "visibility", "wind_speed",
    "hour", "is_weekend", "peak_hour",
]
CATEGORICAL_COLS = ["road_type", "intersection_type"]
TARGET = "clearance_time_minutes"


def build_training_table():
    """Joins incident reports (which carry the target) back onto the
    feature store at the matching timestamp+segment to pull in
    road/weather/traffic context at report time."""
    incidents = pd.read_csv(f"{config.PROCESSED_DIR}/incident_targets.csv",
                             parse_dates=["timestamp"])
    features = pd.read_csv(f"{config.PROCESSED_DIR}/master_feature_store.csv",
                            parse_dates=["timestamp"])

    context_cols = ["timestamp", "sensor_id", "road_segment_id", "hour",
                     "is_weekend", "peak_hour", "temperature", "rainfall",
                     "visibility", "wind_speed", "road_type", "number_of_lanes",
                     "speed_limit", "road_capacity", "junction",
                     "intersection_type", "road_length"]
    features_ctx = features[context_cols].drop_duplicates(
        subset=["timestamp", "road_segment_id"]
    )

    df = incidents.merge(features_ctx, on=["timestamp", "road_segment_id"], how="left")
    df = df.dropna(subset=FEATURE_COLS + [TARGET])
    return df


def prepare(df):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    for col in ["emergency_vehicle_required", "junction", "is_weekend", "peak_hour"]:
        X[col] = X[col].astype(int)

    return X, y, encoders


def train():
    df = build_training_table()
    X, y, encoders = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED
    )

    model = RandomForestRegressor(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
    )[:10]

    with open(f"{config.MODEL_DIR}/clearance_model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": FEATURE_COLS}, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{config.REPORT_DIR}/clearance_model_report.json", "w") as f:
        json.dump({
            "mae_minutes": mae, "r2": r2, "n_incidents_trained": len(df),
            "top_10_features": [(f, round(v, 4)) for f, v in importances],
        }, f, indent=2)

    print(f"Clearance model - MAE: {mae:.2f} min, R2: {r2:.4f} (n={len(df)} incidents)")
    print("Top features:")
    for f, v in importances:
        print(f"  {f:30s} {v:.4f}")

    return model, mae


if __name__ == "__main__":
    train()
