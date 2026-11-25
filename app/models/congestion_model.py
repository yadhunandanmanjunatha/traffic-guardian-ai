"""
Traffic Guardian AI - Model 1: Congestion Classification
Target: congestion_class (0=Low, 1=Moderate, 2=High, 3=Severe)

Only uses features that pass the "available at prediction time" test.
"""
import pickle
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import LabelEncoder
from app.config import settings as config

# Features available at prediction time (no leakage: excludes anything
# derived FROM congestion_index/congestion_class itself, and excludes
# post-incident fields)
FEATURE_COLS = [
    "hour", "minute", "day_of_week", "month", "is_weekend", "is_holiday", "peak_hour",
    "vehicle_count", "average_speed", "travel_time", "occupancy_rate", "traffic_volume",
    "temperature", "humidity", "rainfall", "visibility", "wind_speed",
    "precipitation_probability", "weather_condition",
    "road_type", "number_of_lanes", "speed_limit", "road_capacity",
    "junction", "intersection_type", "road_length",
    "incident_exists", "severity", "blocked_lanes", "emergency_vehicle_required",
    "car_count", "bike_count", "bus_count", "truck_count", "heavy_vehicle_ratio",
    "traffic_growth_rate", "rolling_avg_speed", "rolling_avg_volume",
    "rolling_speed_variance", "weather_severity_score", "road_complexity_score",
]
CATEGORICAL_COLS = ["weather_condition", "road_type", "intersection_type"]
TARGET = "congestion_class"


def prepare(df):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET].copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    for col in ["is_weekend", "is_holiday", "peak_hour", "junction",
                "incident_exists", "emergency_vehicle_required"]:
        X[col] = X[col].astype(int)

    return X, y, encoders


def train():
    df = pd.read_csv(f"{config.PROCESSED_DIR}/master_feature_store.csv")
    X, y, encoders = prepare(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_SEED, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=120, max_depth=10, min_samples_leaf=10,
        class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    # Numeric-labeled report: keys are the raw class indices (0/1/2/3) that
    # congestion_class actually is in the data, not text names. This is
    # what the model literally predicts/scores against - class_index_map
    # below is only a legend for reading the numeric keys, not part of
    # the scored report itself.
    class_labels = sorted(config.CONGESTION_CLASSES.keys())
    report = classification_report(
        y_test, y_pred, labels=class_labels, output_dict=True
    )
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    importances = sorted(
        zip(FEATURE_COLS, model.feature_importances_), key=lambda x: -x[1]
    )[:10]

    with open(f"{config.MODEL_DIR}/congestion_model.pkl", "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": FEATURE_COLS}, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(f"{config.REPORT_DIR}/congestion_model_report.json", "w") as f:
        json.dump({
            "macro_f1": macro_f1,
            "classification_report": report,
            "class_index_map": config.CONGESTION_CLASSES,  # legend: {0: "Low", 1: "Moderate", ...}
            "top_10_features": [(f, round(v, 4)) for f, v in importances],
        }, f, indent=2)

    print(f"Congestion model - macro F1: {macro_f1:.4f}")
    print("Top features:")
    for f, v in importances:
        print(f"  {f:30s} {v:.4f}")

    return model, macro_f1


if __name__ == "__main__":
    train()
