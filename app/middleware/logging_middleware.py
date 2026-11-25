"""
Traffic Guardian AI - Request Logging Middleware

Simple pass-through timing/log middleware for API requests. Logs go to
logs/ via the standard logging module (see app/utils/logger.py).
"""
import time

from fastapi import FastAPI, Request

from app.utils.logger import get_logger

logger = get_logger(__name__)


def add_logging_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method, request.url.path, response.status_code, duration_ms,
        )
        return response
