"""
Traffic Guardian AI - Live Prediction API - Entrypoint

Loads the two ALREADY-TRAINED models (congestion_model.pkl,
clearance_model.pkl) at startup and serves real predictions from
real inputs, plus a Bangalore route planner built on top of the same
congestion model.

Run:
    python3 -m uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings as config
from app.exceptions.handlers import register_exception_handlers
from app.middleware.cors import add_cors_middleware
from app.middleware.logging_middleware import add_logging_middleware
from app.repositories.model_repository import model_repository
from app.routers import health_router, operations_router, prediction_router, route_router

app = FastAPI(title=config.APP_TITLE)

# ---------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------
add_cors_middleware(app)
add_logging_middleware(app)

# ---------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------
register_exception_handlers(app)

# ---------------------------------------------------------------
# Routers
# ---------------------------------------------------------------
app.include_router(health_router.router)
app.include_router(prediction_router.router)
app.include_router(route_router.router)
app.include_router(operations_router.router)


# ---------------------------------------------------------------
# Load trained models once at startup
# ---------------------------------------------------------------
@app.on_event("startup")
def load_models():
    model_repository.load()


# ---------------------------------------------------------------
# Single-page app (merged Predict / Route / Dashboard UI)
# Served straight off the API so it's one URL, one command:
# uvicorn app.main:app --reload --port 8000  ->  http://127.0.0.1:8000
# ---------------------------------------------------------------
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.get("/")
def serve_app():
    return FileResponse(f"{config.STATIC_DIR}/index.html")


@app.get("/favicon.ico")
def favicon():
    # Browsers probe this path directly even when a <link rel="icon">
    # data-URI is declared in the HTML - a plain 204 keeps it silent
    # instead of spamming 404s in the server log.
    from fastapi.responses import Response
    return Response(status_code=204)
