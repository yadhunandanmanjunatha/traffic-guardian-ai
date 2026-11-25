"""
Traffic Guardian AI - Central Configuration
Single source of truth for all pipeline stages.
Swap DATA_MODE to 'real' once you have actual METR-LA/PeMS-BAY files in data/raw/.
"""
import os

from dotenv import load_dotenv

# ---------------------------------------------------------------
# LOAD .env FIRST - this MUST happen before any os.environ.get() call
# below, otherwise values in .env (like TOMTOM_API_KEY) are silently
# ignored and every os.environ.get(...) falls through to its default.
# This was a real bug: python-dotenv was in requirements.txt but never
# actually invoked anywhere in the codebase, so .env was dead weight.
# ---------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_THIS_DIR, "..", "..", ".env")
load_dotenv(dotenv_path=_ENV_PATH)

# ---------------------------------------------------------------
# DATA MODE
# ---------------------------------------------------------------
# 'synthetic' -> generates realistic placeholder data matching the exact
#                schema/statistics of METR-LA/PeMS-BAY so the full pipeline
#                runs end-to-end today.
# 'real'      -> expects real files in data/raw/ (see README for filenames).
DATA_MODE = os.environ.get("TGA_DATA_MODE", "synthetic")

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
# BASE_DIR is the project root (two levels up from app/config/settings.py)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
REPORT_DIR = os.path.join(BASE_DIR, "outputs", "reports")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
LOG_DIR = os.path.join(BASE_DIR, "logs")

for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR, UPLOAD_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------
# SENSOR NETWORK (mirrors METR-LA/PeMS-BAY structure, scaled down
# for hackathon-speed training; bump N_SENSORS up when you have
# real compute/time budget)
# ---------------------------------------------------------------
N_SENSORS = 40
N_ROAD_SEGMENTS = 40  # 1:1 sensor-to-segment mapping for simplicity

# ---------------------------------------------------------------
# TIME RANGE
# ---------------------------------------------------------------
START_DATE = "2025-01-01"
N_DAYS = 45
FREQ_MINUTES = 5  # matches METR-LA/PeMS-BAY 5-minute intervals

# ---------------------------------------------------------------
# HOLIDAYS (India-focused, matches Bengaluru deployment context)
# ---------------------------------------------------------------
HOLIDAYS_2025 = [
    "2025-01-01", "2025-01-14", "2025-01-26",  # New Year, Makar Sankranti, Republic Day
    "2025-03-14",  # Holi
    "2025-04-14",  # Ambedkar Jayanti / Baisakhi
    "2025-08-15",  # Independence Day
    "2025-10-02",  # Gandhi Jayanti
    "2025-10-20",  # Diwali (approx)
    "2025-12-25",  # Christmas
]

# ---------------------------------------------------------------
# CONGESTION CLASS THRESHOLDS (built from congestion_index quantiles
# at training time, but base labels defined here)
# ---------------------------------------------------------------
CONGESTION_CLASSES = {0: "Low", 1: "Moderate", 2: "High", 3: "Severe"}
CONGESTION_LABELS = ["Low", "Moderate", "High", "Severe"]

# ---------------------------------------------------------------
# THS (Traffic Health Score) BANDS
# ---------------------------------------------------------------
THS_BANDS = [
    (90, 100, "Healthy"),
    (70, 89, "Moderate"),
    (50, 69, "Busy"),
    (0, 49, "Critical"),
]

# ---------------------------------------------------------------
# ROAD CAPACITY LOOKUP (used by the live prediction feature builder)
# NOTE: these are NOT standard traffic-engineering hourly-throughput
# capacities (which run 900-2000+/lane and would make any realistic
# camera snapshot count - tens of vehicles - read as near-zero
# occupancy, since the training data's occupancy_rate feature is the
# single most important feature at 31% importance). Instead these are
# scaled to match "vehicles visible/counted in one camera snapshot vs.
# a full segment", so realistic YOLO-style counts (10-80 vehicles)
# produce a meaningful, non-degenerate occupancy_rate - the same
# relationship the model was actually trained on.
# ---------------------------------------------------------------
ROAD_CAPACITY_LOOKUP = {
    "highway": 40, "arterial": 30, "collector": 20, "local": 12,
}

RANDOM_SEED = 42

# ---------------------------------------------------------------
# REAL EXTERNAL DATA SOURCES (route + clearance realism layer)
# ---------------------------------------------------------------
OSRM_BASE_URL = os.environ.get("OSRM_BASE_URL", "https://router.project-osrm.org")
OPEN_METEO_BASE_URL = os.environ.get("OPEN_METEO_BASE_URL", "https://api.open-meteo.com")
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "").strip()
TOMTOM_BASE_URL = os.environ.get("TOMTOM_BASE_URL", "https://api.tomtom.com")

SPEED_RATIO_BANDS = [
    (0.85, 1.01, "Low"),
    (0.65, 0.85, "Moderate"),
    (0.40, 0.65, "High"),
    (0.00, 0.40, "Severe"),
]

# ---------------------------------------------------------------
# CORS
# ---------------------------------------------------------------
CORS_ALLOW_ORIGINS = ["*"]
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]

# ---------------------------------------------------------------
# APP METADATA
# ---------------------------------------------------------------
APP_TITLE = "Traffic Guardian AI - Prediction API"
