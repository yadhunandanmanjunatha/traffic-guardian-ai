"""
Traffic Guardian AI - Exception Handlers

Registers handlers that translate domain exceptions into clean HTTP
error responses. Wired up in app/main.py via register_exception_handlers(app).
"""
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.exceptions.custom_exceptions import (
    DataFileNotFoundError,
    ModelNotLoadedError,
    RouteNotFoundError,
    TrafficGuardianException,
    UnknownLandmarkError,
)


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(RouteNotFoundError)
    async def route_not_found_handler(request: Request, exc: RouteNotFoundError):
        return JSONResponse(status_code=404, content={"error": exc.message})

    @app.exception_handler(UnknownLandmarkError)
    async def unknown_landmark_handler(request: Request, exc: UnknownLandmarkError):
        return JSONResponse(status_code=400, content={"error": exc.message})

    @app.exception_handler(ModelNotLoadedError)
    async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedError):
        return JSONResponse(status_code=503, content={"error": exc.message})

    @app.exception_handler(DataFileNotFoundError)
    async def data_file_not_found_handler(request: Request, exc: DataFileNotFoundError):
        return JSONResponse(status_code=500, content={"error": exc.message})

    @app.exception_handler(TrafficGuardianException)
    async def generic_handler(request: Request, exc: TrafficGuardianException):
        return JSONResponse(status_code=500, content={"error": exc.message})
