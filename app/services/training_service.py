"""
Traffic Guardian AI - Training Pipeline Service

Orchestrates the full offline pipeline: raw data -> master feature
store -> trained models -> reports. This is NOT called during normal
API request handling; run it standalone to (re)train the models:

    python3 -m app.services.training_service
"""
import time

from app.config import settings as config
from app.database import data_loader, synthetic_data
from app.models import clearance_model, congestion_model, feature_engineering


def run_pipeline():
    t0 = time.time()
    print("=" * 60)
    print("TRAFFIC GUARDIAN AI - FULL PIPELINE")
    print(f"Data mode: {config.DATA_MODE}")
    print("=" * 60)

    print("\n[1/4] Generating / loading raw data sources...")
    if config.DATA_MODE == "synthetic":
        synthetic_data.generate_all()
    else:
        raise NotImplementedError(
            "DATA_MODE='real' expects files already in data/raw/. "
            "See README.md for expected filenames and columns."
        )

    print("\n[2/4] Preprocessing & merging sources...")
    data_loader.run()

    print("\n[3/4] Feature engineering (lineage pipeline + THS)...")
    df = feature_engineering.run()
    print(f"    Master feature store: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print("\n[4/4] Training models...")
    print("  -> Model 1: Congestion Classification")
    _, macro_f1 = congestion_model.train()
    print("  -> Model 2: Clearance Time Regression")
    _, mae = clearance_model.train()

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"  Congestion model macro F1: {macro_f1:.4f}")
    print(f"  Clearance model MAE:       {mae:.2f} minutes")
    print(f"  Total time:                {elapsed:.1f}s")
    print(f"  Models saved to:           {config.MODEL_DIR}")
    print(f"  Reports saved to:          {config.REPORT_DIR}")
    print(f"  Feature store saved to:    {config.PROCESSED_DIR}/master_feature_store.csv")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
