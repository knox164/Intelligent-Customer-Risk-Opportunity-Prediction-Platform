"""
trainer.py
----------
Part 1 of the segmentation module.

Pipeline:
  1. Load customer_features.csv
  2. Select clustering features
  3. Scale features (StandardScaler)
  4. Run the Elbow Method (inertia vs k)
  5. Compute Silhouette scores vs k
  6. Pick best k, train final KMeans model
  7. Persist segmentation_model.pkl and scaler.pkl

Usage:
    python -m src.models.segmentation.trainer
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .utils import (
    MODEL_PATH,
    SCALER_PATH,
    SegmentationConfig,
    get_logger,
    load_customer_features,
    save_pickle,
)

logger = get_logger(__name__)


def select_features(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    """Return only the columns used for clustering, in a fixed order."""
    return df[feature_columns].copy()


def scale_features(X: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_elbow_method(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> Dict[int, float]:
    """Compute inertia (within-cluster sum of squares) for k in [k_min, k_max]."""
    inertias: Dict[int, float] = {}
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X_scaled)
        inertias[k] = float(km.inertia_)
        logger.info(f"Elbow: k={k} -> inertia={km.inertia_:.2f}")
    return inertias


def run_silhouette_scan(
    X_scaled: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42
) -> Dict[int, float]:
    """Compute silhouette score for k in [k_min, k_max]. Requires k >= 2."""
    scores: Dict[int, float] = {}
    for k in range(max(2, k_min), k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = float(score)
        logger.info(f"Silhouette: k={k} -> score={score:.4f}")
    return scores


def choose_best_k(silhouette_scores: Dict[int, float]) -> int:
    """Best k = highest silhouette score."""
    return max(silhouette_scores, key=silhouette_scores.get)


def train_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    km.fit(X_scaled)
    return km


def run_training_pipeline(config: SegmentationConfig | None = None) -> dict:
    """
    Full Part-1 pipeline. Returns a summary dict (useful for tests / notebooks)
    and writes segmentation_model.pkl + scaler.pkl to disk as a side effect.
    """
    config = config or SegmentationConfig()

    logger.info(f"Loading data from {config.input_csv}")
    df = load_customer_features(config.input_csv, config.feature_columns, config.id_column)
    logger.info(f"Loaded {len(df)} customers.")

    X = select_features(df, config.feature_columns)

    # KMeans needs at least as many samples as clusters; cap k_max to available data.
    k_max = min(config.k_max, max(2, len(df) - 1))
    k_min = min(config.k_min, k_max)

    X_scaled, scaler = scale_features(X)

    inertias = run_elbow_method(X_scaled, k_min, k_max, config.random_state)
    silhouette_scores = run_silhouette_scan(X_scaled, k_min, k_max, config.random_state)
    best_k = choose_best_k(silhouette_scores)
    logger.info(f"Selected best_k={best_k} based on silhouette score.")

    model = train_kmeans(X_scaled, best_k, config.random_state)

    save_pickle(model, MODEL_PATH)
    save_pickle(scaler, SCALER_PATH)
    logger.info(f"Saved model -> {MODEL_PATH}")
    logger.info(f"Saved scaler -> {SCALER_PATH}")

    return {
        "n_customers": len(df),
        "feature_columns": config.feature_columns,
        "k_min": k_min,
        "k_max": k_max,
        "inertias": inertias,
        "silhouette_scores": silhouette_scores,
        "best_k": best_k,
        "model_path": str(MODEL_PATH),
        "scaler_path": str(SCALER_PATH),
    }


if __name__ == "__main__":
    summary = run_training_pipeline()
    print(f"\nTraining complete. Best k = {summary['best_k']}")
    print(f"Model saved to: {summary['model_path']}")
    print(f"Scaler saved to: {summary['scaler_path']}")
