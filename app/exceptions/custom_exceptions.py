"""
Traffic Guardian AI - Custom Exceptions

Domain-specific exceptions raised by the services/repositories layers.
Handlers for these are registered in app/exceptions/handlers.py.
"""


class TrafficGuardianException(Exception):
    """Base class for all application-specific exceptions."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ModelNotLoadedError(TrafficGuardianException):
    """Raised when a trained model artifact could not be loaded from disk."""


class RouteNotFoundError(TrafficGuardianException):
    """Raised when no path exists between two landmarks in the road graph."""


class UnknownLandmarkError(TrafficGuardianException):
    """Raised when a source/destination landmark id is not in the graph."""


class DataFileNotFoundError(TrafficGuardianException):
    """Raised when an expected raw/processed data file is missing."""
