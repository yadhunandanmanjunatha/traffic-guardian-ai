# Traffic Guardian AI — Complete Project Guide
### Every file, every algorithm, every connection explained — written so you can defend it in front of judges without hesitation.

---

## 1. THE BIG PICTURE — one paragraph you should be able to say from memory

> "Traffic Guardian AI predicts road congestion and clearance time using two trained
> machine learning models — a Random Forest classifier and a Random Forest regressor.
> The road network is real (actual METR-LA sensor geography and measured distances
> from a published traffic-prediction benchmark paper). Live inputs — vehicle counts,
> speed, weather, road context — get fed through a feature-engineering pipeline that
> computes the same derived features the model was trained on, then sent to a FastAPI
> backend that runs the trained model and returns a prediction, a confidence score,
> and a SHAP-based explanation of *why*. Right now those inputs are typed in manually;
> later, a Raspberry Pi running YOLO fills the same fields automatically."

That's the whole project in one breath. Everything below is the detail underneath it.

---

## 2. THE DATA FLOW — two different flows, don't mix them up

**Flow A — TRAINING (happens once, offline, before the demo)**
```
Raw data (real topology + simulated readings)
    -> preprocessing.py (merge + clean)
    -> feature_engineering.py (compute 53 features + targets)
    -> train_congestion.py / train_clearance.py (fit Random Forest models)
    -> .joblib model files saved to outputs/models/
```

**Flow B — LIVE PREDICTION (happens every time you click "Predict")**
```
You type numbers into predict_console.html
    -> browser sends JSON to FastAPI (server.py) over HTTP
    -> server.py computes the SAME engineered features, live, for just your one input
    -> loads the ALREADY-TRAINED model from the .joblib file (no retraining happens here)
    -> model.predict_proba() runs -> class + confidence
    -> SHAP explains which features pushed the prediction
    -> JSON sent back to the browser -> rendered on screen
```

The single most important thing to understand: **training happens once, offline. Predicting happens instantly, using the model that's already trained.** This is exactly how production ML systems work (a judge will recognize this immediately if you say it explicitly).

---

## 3. FILE-BY-FILE WALKTHROUGH

### `src/config.py` — the single source of truth
Every other file imports settings from here instead of hardcoding numbers. Contains:
- File paths (`RAW_DIR`, `PROCESSED_DIR`, `MODEL_DIR`, `REPORT_DIR`)
- `N_SENSORS = 40`, `N_DAYS = 45` — size of the training dataset generated
- `HOLIDAYS_2025` — the holiday calendar list
- `RANDOM_SEED = 42` — makes every run reproducible (same seed = same "random" numbers every time, which is why your F1 score doesn't change between runs)

**Why this matters to say out loud:** "We centralized configuration instead of hardcoding values across files — that's a basic production-engineering practice."

---

### `src/synthetic_data.py` — the data layer (READ THIS SECTION CAREFULLY — most likely question)

This file generates the raw training data. Two very different things happen in it:

**REAL data used directly:**
- `load_real_metr_la_topology()` loads two CSV files sourced from the DCRNN paper's public GitHub repo (Li et al., *ICLR 2018* — the paper that created the METR-LA benchmark): actual GPS coordinates of 207 real Los Angeles traffic sensors, and actual measured road-network distances between them (11,753 real edges).
- `generate_road_network()` builds the road segment table **from that real geography** — `road_length` is a real measured distance, not invented. Road type (highway/arterial/collector/local) is inferred from real sensor spacing (closely-spaced sensors → dense urban road; widely-spaced → highway).

**SIMULATED data (and here is the honest reason why):**
- The actual 5-minute traffic speed readings that made METR-LA famous (`metr-la.h5`) are distributed via Google Drive, not a direct download link. This is true of every public mirror of this dataset, not just one source — it's a real access constraint, not a shortcut we took.
- `generate_traffic_data()` simulates speed/volume/occupancy readings **on top of the real road network**, using correct traffic-flow physics: speed decreases as occupancy increases (`average_speed = speed_limit * (1 - 0.75*occupancy_rate)`), rush-hour has a bimodal peak (morning + evening, via `_daily_profile()`), and weekends have less traffic.
- `generate_weather_data()`, `generate_incident_data()`, `generate_yolo_edge_data()` — all simulated, all documented as such.

**If asked "is your data real?"** — the honest, confident answer: *"The road network topology is real, sourced from the published benchmark paper. The live traffic readings are simulated on top of that real network because the original readings are gated behind Google Drive and unreachable in our build environment — but they follow the actual physical relationships real traffic obeys, not random noise."*

---

### `src/preprocessing.py` — merging and cleaning
Takes the 6 raw tables (traffic, weather, road, holiday, incident, YOLO) and:
1. Merges them all onto one timeline, joined by `timestamp` and `road_segment_id`
2. Applies a **missing-value strategy per feature type** (not one blanket rule):
   - `vehicle_count`, `occupancy_rate` → filled with the **median** (robust to outliers)
   - `average_speed`, `travel_time`, `traffic_volume` → filled with the **mean**
   - Weather → **forward-fill** (weather is temporally continuous — the last known reading is the best guess)
   - Categoricals (sensor IDs, road type) → filled with `"unknown"`
3. Drops any row with no timestamp at all (nothing can be done with it)

**Why this matters:** naive pipelines just do `.fillna(0)` everywhere, which silently corrupts data (a speed of 0 looks like "gridlock" instead of "we don't know"). Choosing the right fill strategy per feature type is a real data-engineering decision.

---

### `src/feature_engineering.py` — where the 53 columns get built
This is the file that turns raw numbers into the actual inputs the model learns from. Grouped:

**Temporal** (`add_temporal_features`): hour, minute, day_of_week, month, is_weekend, is_holiday, and `peak_hour` (a formula: hour between 7–10 AM or 5–8 PM).

**Engineered** (`add_engineered_features`) — the 9 hand-designed features:
| Feature | Formula | What it captures |
|---|---|---|
| `traffic_density` | vehicle_count / road_capacity | how full the road is |
| `speed_ratio` | average_speed / speed_limit | how much slower than allowed |
| `congestion_index` | density × (1 − speed_ratio) | combined congestion signal |
| `weather_severity_score` | 0.4×rain + 0.4×visibility_risk + 0.2×wind, scaled 0–100 | weather's impact |
| `road_complexity_score` | weighted road_type + lanes + junction + intersection | how hard the road is to manage |
| `traffic_growth_rate` | (current − previous) / previous, per sensor | is traffic building up? |
| `rolling_avg_speed` / `rolling_avg_volume` | 30-min rolling mean, per sensor | smooths out noise |
| `rolling_speed_variance` | 30-min rolling std, per sensor | how unstable/erratic traffic is |

**Traffic Health Score** (`compute_ths`) — **not a model, a formula**:
```
THS = 100 − congestion_penalty − weather_penalty − incident_penalty
```
This is intentionally *not* machine learning — it's a transparent, auditable score anyone can hand-check. That's the point: it's explainable by construction, unlike a black-box model.

**Target creation** (`add_congestion_target`): `congestion_class` (0–3) is derived from `congestion_index` quantiles — the bottom 50% is Low, next 30% Moderate, next 15% High, top 5% Severe.

⚠️ **Critical point if asked about data leakage:** `congestion_index`, `traffic_density`, `speed_ratio`, and `traffic_health_score` are **deliberately excluded** from the model's training features (`FEATURE_COLS` in `train_congestion.py`), because `congestion_class` was derived directly from `congestion_index` — including it as an input would let the model "cheat" by looking at its own answer key. This is the single best technical detail to bring up if a judge asks how you avoided leakage.

---

### `src/models/train_congestion.py` — Model 1

**Algorithm: Random Forest Classifier** (`sklearn.ensemble.RandomForestClassifier`)

*What a Random Forest actually is, in plain terms:* it trains many individual decision trees (we use 120), each on a random subset of the training rows and a random subset of features. Each tree makes its own prediction; the forest's final answer is the **average of all 120 trees' vote probabilities**. This is why it outputs a confidence score (`predict_proba`), not just a single label — it's literally counting how many of the 120 trees agree.

*Why Random Forest and not a neural network:* with ~40 structured/tabular features and a dataset this size, tree ensembles reliably outperform deep learning on tabular data, train in seconds instead of hours, need no GPU, and — crucially for this project — are naturally explainable via feature importances and SHAP. A judge who knows ML will respect this choice if you can say why.

**Hyperparameters used and why:**
- `n_estimators=120` — number of trees. More trees = more stable predictions, diminishing returns past ~100-150.
- `max_depth=10` — caps how deep each tree can grow, preventing overfitting to training noise.
- `min_samples_leaf=10` — each leaf node needs at least 10 training examples, another overfitting guard.
- `class_weight="balanced"` — since Low congestion is far more common than Severe in the data, this automatically up-weights rare classes so the model doesn't just always predict "Low" to get a cheap high accuracy.

**Training process** (`train()` function):
1. Load `master_feature_store.csv`
2. Select only the 40 leakage-safe `FEATURE_COLS`
3. Encode categorical columns (`weather_condition`, `road_type`, `intersection_type`) into numbers using `LabelEncoder` — models can't read text
4. `train_test_split(test_size=0.2)` — 80% of rows train the model, 20% are held back to test on unseen data
5. `model.fit(X_train, y_train)` — this is the actual "learning" step
6. Evaluate on the held-out 20% with `classification_report` and `macro_f1`
7. Save the trained model + the label encoders together in one `.joblib` file (you need the *same* encoders at prediction time, or "arterial" might get encoded as a different number than it was during training — this is a common real bug, worth mentioning you thought about it)

**Result:** macro F1 ≈ 0.96 on the held-out test set. **Be upfront that this is on simulated-reading data** — it validates the pipeline end-to-end, not real-world accuracy.

---

### `src/models/train_clearance.py` — Model 2

**Algorithm: Random Forest Regressor** — same underlying idea as above, but instead of voting on a class, each of the 200 trees predicts a number (minutes), and the forest averages them.

**Leakage safeguard, the most important thing to say about this model:** `incident_duration` is **excluded on purpose**. It's only known *after* an incident ends — including it as an input to predict "how long will this take to clear" would mean the model already knows the answer. Instead, only fields known **at the moment the incident is reported** are used: `severity`, `blocked_lanes`, `emergency_vehicle_required`, plus road/weather context.

**Result:** MAE (Mean Absolute Error) ≈ 7.6 minutes, R² ≈ 0.79, trained on 5,184 simulated incident records.

---

### `src/ths_live.py` — standalone score function
A small, dependency-light function (`score_reading()`) that computes THS for a single reading without needing the full batch pipeline. This is what the API calls for instant scoring — it's plain arithmetic, not a model, so it's instant and has zero "confidence" ambiguity.

---

### `src/pipeline.py` — the orchestrator
Runs all training stages in order with one command: `python3 -m src.pipeline`. Generate data → preprocess → engineer features → train both models → print metrics. This is what you'd run once, offline, before the demo — **not** during the live demo itself.

---

### `src/api/server.py` — the live backend (THE FILE THAT MAKES IT "REAL-TIME")

**Framework: FastAPI**, a Python web framework. It exposes HTTP endpoints the browser can call.

**What happens on every `/api/predict` call, step by step:**
1. `PredictRequest` (a Pydantic model) validates the incoming JSON — enforces types and sane ranges (e.g., `hour` must be 0–23), rejecting garbage input automatically
2. `build_feature_row()` computes the same engineered features as training — but for a **single row**, live, in Python instead of pandas. Since there's no sensor history for a one-off manual reading, `traffic_growth_rate` defaults to 0 and rolling averages default to the current reading (documented in the code comments — a fair simplification to explain if asked)
3. `encode_for_model()` applies the **same LabelEncoders saved during training** to convert categories to numbers — this is what guarantees no train/serve mismatch
4. `CONGESTION["model"].predict_proba(X)` — runs the actual trained Random Forest, returns probability per class
5. `shap.TreeExplainer(...).shap_values(X)` — computes **real SHAP values** for this specific prediction (see section 5 below)
6. If `incident_exists` is true, also runs the clearance regressor
7. Calls `ths_live.score_reading()` for the bonus THS number
8. Returns one JSON object with everything

**CORS** (`CORSMiddleware`) is enabled so the HTML file (opened directly in a browser, not served from the same origin) is allowed to call this API — otherwise browsers block "cross-origin" requests by default as a security measure.

---

### `outputs/predict_console.html` — the live demo frontend
Pure HTML/CSS/JS, no framework, no build step — just open it in a browser. Key mechanics:
- On page load, `checkApi()` pings `/api/health` every 8 seconds to show a live connection indicator
- The form's `gatherPayload()` function reads every input field into one JSON object matching `PredictRequest`
- On submit: `fetch(API_BASE + '/api/predict', {method:'POST', body: JSON.stringify(payload)})` — a standard browser HTTP request, no different from any web app calling a backend
- `renderResult()` takes the JSON response and builds the result cards, probability bars, and SHAP reason list dynamically

**This file has zero data baked in** — everything on screen came from the live API call you just made. That's the key difference from the other dashboard file below.

---

### `outputs/dashboard.html` (+ embedded data) — the network-wide view
This one is **different in kind**: it's a static snapshot. A one-time Python script read the real 207-sensor topology and the trained models' saved metrics, computed a plausible live-looking reading for each sensor, and baked all of it into one JSON blob embedded directly in the HTML (`const TGA_DATA = {...}`). It does **not** call the API and does **not** update.

**Use it to show:** the real sensor network, the overall system design, and the "if this were fully deployed, here's what monitoring 207 sensors at once would look like" story. **Don't** claim it's live — say directly "this is a snapshot view of the real network; the prediction console next to it is where inference is actually live."

---

### `config/feature_dictionary.yaml` — the Feature Store spec
A machine-readable document listing every one of the ~51 features: name, type, source, whether it's raw or engineered, which model uses it, whether it's safe at prediction time, and the missing-value/encoding strategy. This is the document real ML teams (the file's comments reference Uber/Google-style practice) write *before* touching code. If a judge asks "how did you decide what features to use," point at this file — it's your proof of a designed process, not a stumbled-into feature set.

---

### `run_api.sh`
One-line convenience script: `python3 -m uvicorn src.api.server:app --reload --port 8000`. Starts the backend so the prediction console has something to talk to.

---

## 4. THE HARDWARE CONNECTION (future work — explain this clearly, it's your strongest "vision" answer)

Right now, `predict_console.html` sends a JSON body like:
```json
{"car_count": 24, "bike_count": 9, "bus_count": 1, "truck_count": 1, "average_speed": 32, ...}
```
That JSON is typed by a human, sent over HTTP to `/api/predict`.

**When the Raspberry Pi is wired up:** the Pi runs YOLO on a camera feed, counts vehicles by class, and sends the **exact same JSON shape** to the **exact same endpoint** — just from a Python script on the Pi (using the `requests` library) instead of a browser form. `server.py` does not need to change at all. That's the entire hardware integration story: **swap the input source, keep everything downstream identical.** This is a genuinely strong architectural point to make — it shows you designed the interface correctly the first time.

---

## 5. WHAT IS SHAP, IN PLAIN LANGUAGE (judges love this question)

SHAP (SHapley Additive exPlanations) answers: *"for this one specific prediction, how much did each feature push the answer up or down?"* It's grounded in game theory (Shapley values, originally from economics — how do you fairly divide credit among players who contributed to a team outcome). For tree models like ours, `shap.TreeExplainer` computes this efficiently and *exactly* (not an approximation) by examining how predictions change as you include/exclude each feature across all paths in every tree.

**Why it matters for your project:** a raw "High Congestion" prediction is a black box. SHAP turns it into: *"High congestion, and specifically because occupancy_rate contributed +0.24, rainfall contributed +0.11, peak_hour contributed +0.08."* That's the difference between a model that predicts and a model that explains — and explainability is explicitly one of the five things judges are told to look for.

---

## 6. LIKELY JUDGE QUESTIONS — rehearse these

**Q: Is your traffic data real?**
A: The sensor network — real GPS locations, real road distances — is from the actual METR-LA benchmark paper. The live readings are simulated on top of that real network because the original data requires Google Drive access we don't have in our build pipeline, but the simulation follows real traffic physics (speed drops as occupancy rises), not random numbers.

**Q: Why Random Forest and not deep learning?**
A: Tabular, structured data with ~40 features and this dataset size — tree ensembles reliably beat neural nets here, train in seconds, and are natively explainable via feature importance and SHAP, which matters for our explainability goal.

**Q: How do you avoid data leakage?**
A: Two concrete examples: `congestion_index` (used to derive the target) is excluded from training features. `incident_duration` is excluded from the clearance model because it's only known after the incident ends.

**Q: How does the hardware connect?**
A: The Pi sends the same JSON shape our form sends now, to the same API endpoint. No backend code changes — only the input source changes.

**Q: What's the accuracy, really?**
A: Macro F1 of 0.96 on our held-out test set — but that's on simulated-reading data validating the pipeline works end-to-end. We're upfront that real-world accuracy on actual sensor data would need retraining once real METR-LA readings are available.

**Q: What's the Traffic Health Score — is that ML too?**
A: No, and deliberately not — it's a transparent formula (100 minus weighted penalties), fully auditable, unlike the model's internal decision-making. It's a complementary explainable indicator, not a third prediction model.

**Q: What would you do with more time?**
A: Get the real METR-LA speed archive (manual download, since it's Google-Drive-gated), retrain on real readings, and wire up the actual Raspberry Pi + YOLO pipeline to replace manual input.

---

## 8. V2 UPGRADE — Bangalore Route Planner (source → destination)

This is the biggest capability upgrade: instead of predicting congestion at
one point, the system now predicts an entire **route**, using a real
Bangalore road graph.

### `src/bangalore_graph.py`
27 real Bangalore landmarks (Silk Board, Marathahalli, Whitefield, Koramangala,
Hebbal, Electronic City, etc.) as graph nodes, connected by 28 real known road
corridors (Outer Ring Road, Hosur Road, Sarjapur Road...) as edges. Each edge
carries `road_type`, `lanes`, `speed_limit`, real approximate `distance_km`,
and two Bangalore-specific flags: `chronic_bottleneck` (Silk Board, Marathahalli
Bridge, Bellandur, etc. — well-documented real problem spots) and `flood_prone`
(Silk Board and Bellandur genuinely flood in Bangalore's monsoon). **Be upfront**:
coordinates and distances are from general geographic knowledge, not a live
mapping API pull — accurate enough for a schematic demo, not survey-grade.

### `src/route_predictor.py`
1. **`find_path()`** — Dijkstra's algorithm finds the real shortest route
   between your source and destination across the graph, weighted by distance.
2. **`predict_segment()`** — for each road segment on that path, simulates a
   plausible current vehicle count (same rush-hour physics as training data,
   boosted if the segment is a known chronic bottleneck), then runs the
   **exact same trained congestion model** used everywhere else in this
   project — no separate model was built for this.
3. **`predict_route()`** — chains segment predictions together: total
   distance, total estimated time, and automatically identifies the
   **bottleneck segment** (worst congestion class along the path).
4. **`best_time_to_leave()`** — runs `predict_route()` repeatedly across the
   next 6 hours for the same source/destination, and recommends the
   departure time with the lowest total travel time.

### New API endpoints (`src/api/server.py`)
- `GET /api/landmarks` — the 27 landmarks, for populating dropdowns
- `POST /api/route` — full route prediction (segments, bottleneck, known issues)
- `POST /api/best-time` — the departure-time comparison

### `outputs/route_planner.html`
The new frontend: pick source + destination from real Bangalore landmarks,
optionally set rainfall/hour, hit **Find Route**. Shows the route as a
visual chain of segments (colored by predicted congestion), a bottleneck
callout, known chronic-bottleneck/flood warnings, and a bar chart of travel
time by departure hour with the best option starred.

**Why this is the strongest thing to lead with to judges:** it's not "here's
a sensor reading," it's "I want to go from Koramangala to Whitefield —
tell me how long it'll take, where I'll get stuck, and when I should leave."
That is the actual problem commuters have, solved end-to-end with the same
trained model, reused across every segment of a real route.



| File | One line |
|---|---|
| `config.py` | Central settings everything else imports |
| `synthetic_data.py` | Real sensor topology + physics-realistic simulated readings |
| `preprocessing.py` | Merges 6 raw sources, fills missing values per-feature-type |
| `feature_engineering.py` | Builds 53 columns incl. 9 engineered features + THS formula |
| `models/train_congestion.py` | Trains Random Forest classifier (Low/Mod/High/Severe) |
| `models/train_clearance.py` | Trains Random Forest regressor (clearance minutes), leakage-safe |
| `ths_live.py` | Instant THS formula for single readings |
| `pipeline.py` | Runs the full offline training sequence in one command |
| `api/server.py` | FastAPI backend: live feature engineering + real model inference + SHAP |
| `predict_console.html` | Live demo frontend — real API calls, zero baked-in data |
| `dashboard.html` | Static snapshot of the real 207-sensor network |
| `feature_dictionary.yaml` | The design spec for every feature, written before coding |
| `run_api.sh` | Starts the backend server |
| `bangalore_graph.py` | Real 27-landmark Bangalore road graph with bottleneck/flood flags |
| `route_predictor.py` | Dijkstra pathfinding + per-segment prediction + best-time-to-leave |
| `route_planner.html` | Source→destination route demo frontend |
