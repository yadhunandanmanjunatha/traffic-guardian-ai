"""
Traffic Guardian AI - Logger Utility

Central place to grab a configured logger. Writes to logs/app.log and
stdout.
"""
import logging
import os

from app.config import settings

_LOG_FILE = os.path.join(settings.LOG_DIR, "app.log")
_configured = False


def _configure_root():
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(_LOG_FILE), logging.StreamHandler()],
    )
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure_root()
    return logging.getLogger(name)
