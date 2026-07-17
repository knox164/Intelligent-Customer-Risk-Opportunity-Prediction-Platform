"""
utils.py
--------
Shared configuration, logging, and I/O helpers used across the
segmentation module (trainer, evaluator, visualizer, report, predictor).

Keeping these in one place means every script agrees on:
  - where data/models/outputs live
  - which columns are used as clustering features
  - how data is loaded / validated
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import joblib
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Project root = two levels up from src/models/segmentation
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = PROJECT_ROOT / "data"/"processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

DEFAULT_INPUT_CSV = DATA_DIR / "customer_features.csv"
MODEL_PATH = MODEL_DIR / "segmentation_model.pkl"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METRICS_PATH = OUTPUT_DIR / "metrics.json"
SEGMENTS_CSV_PATH = OUTPUT_DIR / "customer_segments.csv"
REPORT_MD_PATH = OUTPUT_DIR / "segmentation_report.md"

for _dir in (DATA_DIR, OUTPUT_DIR, MODEL_DIR, PLOT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Feature configuration
# --------------------------------------------------------------------------
# RFM + behavioral features used for clustering. Identifiers, raw timestamps,
# and monetary columns that are redundant with others are excluded.
DEFAULT_FEATURE_COLUMNS: List[str] = [
    "recency",
    "frequency",
    "monetary",
    "average_order_value",
    "customer_lifetime_days",
    "purchase_frequency",
    "basket_size",
]

ID_COLUMN = "customer_id"


@dataclass
class SegmentationConfig:
    """Central config object passed between pipeline stages."""

    input_csv: Path = DEFAULT_INPUT_CSV
    feature_columns: List[str] = field(default_factory=lambda: list(DEFAULT_FEATURE_COLUMNS))
    id_column: str = ID_COLUMN
    random_state: int = 42
    k_min: int = 2
    k_max: int = 10


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# --------------------------------------------------------------------------
# Data loading / validation
# --------------------------------------------------------------------------
def load_customer_features(
    csv_path: Path | str = DEFAULT_INPUT_CSV,
    feature_columns: List[str] | None = None,
    id_column: str = ID_COLUMN,
) -> pd.DataFrame:
    """
    Load customer_features.csv and validate that required columns exist.

    Raises FileNotFoundError / ValueError with clear messages rather than
    letting pandas/sklearn fail deep inside a pipeline.
    """
    csv_path = Path(csv_path)
    feature_columns = feature_columns or DEFAULT_FEATURE_COLUMNS

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Could not find input file at '{csv_path}'. "
            f"Place customer_features.csv there or pass an explicit path."
        )

    df = pd.read_csv(csv_path)

    missing_id = id_column not in df.columns
    missing_features = [c for c in feature_columns if c not in df.columns]
    if missing_id or missing_features:
        missing = ([id_column] if missing_id else []) + missing_features
        raise ValueError(f"Input CSV is missing required column(s): {missing}")

    # Basic hygiene: drop rows with NaNs in the feature set, warn if any dropped.
    before = len(df)
    df = df.dropna(subset=feature_columns).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        get_logger(__name__).warning(
            f"Dropped {dropped} row(s) with missing values in feature columns."
        )

    return df


def save_pickle(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_pickle(path: Path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No file found at '{path}'.")
    return joblib.load(path)


def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    path = Path(path)
    with open(path) as f:
        return json.load(f)
