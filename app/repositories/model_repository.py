"""
Traffic Guardian AI - Model Repository

Data-access layer for trained model artifacts (.pkl files on disk).
Loads the two ALREADY-TRAINED models once at process startup and hands
back the in-memory bundles + a SHAP explainer. Services should go
through this repository rather than calling pickle.load directly.
"""
import pickle
import warnings

import shap

from app.config import settings as config
from app.exceptions.custom_exceptions import ModelNotLoadedError
from app.utils.logger import get_logger

warnings.filterwarnings("ignore", message="X does not have valid feature names")

logger = get_logger(__name__)

CONGESTION_LABELS = config.CONGESTION_LABELS


class ModelRepository:
    """Holds the loaded congestion/clearance model bundles and the SHAP
    explainer for the congestion model. Instantiated once as a
    module-level singleton (`model_repository`, below) and loaded at
    FastAPI startup."""

    def __init__(self):
        self._congestion_bundle = None
        self._clearance_bundle = None
        self._congestion_explainer = None

    def load(self):
        try:
            with open(f"{config.MODEL_DIR}/congestion_model.pkl", "rb") as f:
                self._congestion_bundle = pickle.load(f)
            with open(f"{config.MODEL_DIR}/clearance_model.pkl", "rb") as f:
                self._clearance_bundle = pickle.load(f)
            self._congestion_explainer = shap.TreeExplainer(
                self._congestion_bundle["model"]
            )
            logger.info("Loaded congestion_model and clearance_model (pickle) from %s", config.MODEL_DIR)
        except FileNotFoundError as exc:
            raise ModelNotLoadedError(
                f"Could not load trained model artifacts from {config.MODEL_DIR}: {exc}"
            ) from exc

    @property
    def congestion_bundle(self) -> dict:
        if self._congestion_bundle is None:
            raise ModelNotLoadedError("Congestion model has not been loaded yet.")
        return self._congestion_bundle

    @property
    def clearance_bundle(self) -> dict:
        if self._clearance_bundle is None:
            raise ModelNotLoadedError("Clearance model has not been loaded yet.")
        return self._clearance_bundle

    @property
    def congestion_explainer(self):
        if self._congestion_explainer is None:
            raise ModelNotLoadedError("Congestion SHAP explainer has not been loaded yet.")
        return self._congestion_explainer


# module-level singleton, initialized on app startup (see app/main.py)
model_repository = ModelRepository()
