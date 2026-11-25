# Traffic Guardian AI

Production-style ML pipeline for traffic congestion prediction, clearance-time
estimation, and a novel Traffic Health Score (THS) - now organized as a
layered FastAPI application (routers -> services -> repositories/models ->
database), instead of the original flat `src/` script layout.

## Project layout

```
project/
├── app/
│   ├── routers/          # thin FastAPI endpoint handlers (HTTP in/out only)
│   │   ├── health_router.py
│   │   ├── prediction_router.py
│   │   └── route_router.py
│   ├── services/         # business logic
│   │   ├── prediction_service.py     # live feature build + congestion/clearance predict
│   │   ├── route_service.py          # Dijkstra routing + per-segment prediction
│   │   ├── ths_service.py            # Traffic Health Score scorer
│   │   └── training_service.py       # offline pipeline orchestrator (was pipeline.py)
│   ├── repositories/     # data-access layer
│   │   ├── model_repository.py       # loads/holds the trained .joblib models + SHAP explainer
│   │   └── graph_repository.py       # Bangalore landmark/road graph (was bangalore_graph.py)
│   ├── models/           # ML model definitions & training (was src/models/ + feature_engineering.py)
│   │   ├── feature_engineering.py
│   │   ├── congestion_model.py       # was train_congestion.py
│   │   └── clearance_model.py        # was train_clearance.py
│   ├── schemas/          # Pydantic request/response models
│   │   ├── prediction_schemas.py
│   │   └── route_schemas.py
│   ├── database/         # raw data loading/merging + synthetic data generation
│   │   ├── data_loader.py            # was preprocessing.py
│   │   └── synthetic_data.py
│   ├── config/
│   │   ├── settings.py               # was src/config.py
│   │   └── feature_dictionary.yaml   # Feature Store v2.0 spec
│   ├── middleware/
│   │   ├── cors.py
│   │   └── logging_middleware.py
│   ├── exceptions/
│   │   ├── custom_exceptions.py
│   │   └── handlers.py
│   ├── utils/
│   │   ├── logger.py
│   │   └── traffic_math.py           # shared formulas used by 2+ services
│   └── main.py            # FastAPI app: wires routers, middleware, startup model load
├── tests/
├── logs/                  # app.log written here at runtime
├── static/                 # index.html (merged Predict/Route/Dashboard SPA)
├── uploads/                # runtime file uploads (empty by default)
├── data/                   # raw/, processed/, real_topology/ (unchanged from original)
├── outputs/                # models/ (.pkl), reports/ (JSON metrics)
├── requirements.txt
├── .env / .env.example
├── run.bat                 # Windows launcher
├── run.sh                  # macOS/Linux launcher
├── Dockerfile / docker-compose.yml
└── README.md
```

## What changed vs. the original `src/` layout

The code is the same logic, reorganized by responsibility:

| Original file | New location |
|---|---|
| `src/config.py` | `app/config/settings.py` |
| `src/api/server.py` (routes) | `app/routers/*.py` |
| `src/api/server.py` (feature build/predict logic) | `app/services/prediction_service.py` |
| `src/api/server.py` (pydantic models) | `app/schemas/*.py` |
| `src/api/server.py` (model loading) | `app/repositories/model_repository.py` |
| `src/route_predictor.py` | `app/services/route_service.py` |
| `src/bangalore_graph.py` | `app/repositories/graph_repository.py` |
| `src/ths_live.py` | `app/services/ths_service.py` |
| `src/preprocessing.py` | `app/database/data_loader.py` |
| `src/synthetic_data.py` | `app/database/synthetic_data.py` |
| `src/feature_engineering.py` | `app/models/feature_engineering.py` |
| `src/models/train_congestion.py` | `app/models/congestion_model.py` |
| `src/models/train_clearance.py` | `app/models/clearance_model.py` |
| `src/pipeline.py` | `app/services/training_service.py` |

New, previously-absent pieces added to complete the layered architecture:
`app/exceptions/` (custom exceptions + FastAPI handlers so route/model errors
return clean HTTP responses instead of raw tracebacks), `app/middleware/`
(CORS extracted out of `main.py`, plus a request-logging middleware), and
`app/utils/` (shared math helpers that both `prediction_service` and
`route_service` used to duplicate).

## Run it

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Train everything (models are already included under `outputs/models/`,
so this is only needed if you want to retrain):**
```bash
python3 -m app.services.training_service
```

**3. Start the live prediction API:**
```bash
run.bat          # Windows
./run.sh         # macOS/Linux
# or directly: python3 -m uvicorn app.main:app --reload --port 8000
```

**4. Open `http://127.0.0.1:8000`** in your browser - the API now serves
the merged single-page app (Predict Console / Route Planner / Network
Dashboard tabs) directly, no separate HTML file to open.

## About the data

- Road network topology for the training pipeline is real (207 real
  LA-area sensors from METR-LA / DCRNN paper, real lat/lon and inter-sensor
  distances).
- Traffic readings, weather, holidays, and incidents used to train the
  models are simulated (documented in `app/config/feature_dictionary.yaml`),
  since the real METR-LA/PeMS-BAY speed time series isn't reachable from a
  sandboxed environment.
- The Bangalore route planner (`app/repositories/graph_repository.py`) uses
  approximate real landmark coordinates and known chronic-bottleneck /
  flood-prone corridors (Silk Board Junction, Bellandur, etc.) as local
  domain knowledge layered on top of the trained congestion model.
