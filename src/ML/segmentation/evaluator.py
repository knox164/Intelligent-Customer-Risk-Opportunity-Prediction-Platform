"""
evaluator.py
------------
Part 2 of the segmentation module.

Compares multiple clustering algorithms on the same scaled feature set:
  - KMeans
  - Agglomerative (Hierarchical) Clustering
  - DBSCAN
  - Gaussian Mixture Model

For each, computes:
  - Silhouette Score       (higher is better, range [-1, 1])
  - Davies-Bouldin Index   (lower is better)
  - Calinski-Harabasz Index (higher is better)

Saves results to metrics.json.

Usage:
    python -m src.models.segmentation.evaluator
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors

from .trainer import scale_features, select_features
from .utils import (
    METRICS_PATH,
    SegmentationConfig,
    get_logger,
    load_customer_features,
    save_json,
)

logger = get_logger(__name__)


def _safe_metrics(X_scaled: np.ndarray, labels: np.ndarray) -> Optional[Dict[str, float]]:
    """
    Compute the three clustering-quality metrics, guarding against
    degenerate cases (e.g. DBSCAN finding only 1 cluster or all noise).
    """
    unique_labels = set(labels)
    unique_labels.discard(-1)  # -1 = noise, used by DBSCAN
    if len(unique_labels) < 2:
        logger.warning("Fewer than 2 clusters found; metrics are undefined for this run.")
        return None

    # Metrics like silhouette are undefined on noise points for DBSCAN;
    # standard practice is to evaluate on non-noise points only.
    mask = labels != -1
    X_eval, labels_eval = (X_scaled[mask], labels[mask]) if -1 in labels else (X_scaled, labels)

    return {
        "silhouette_score": float(silhouette_score(X_eval, labels_eval)),
        "davies_bouldin_index": float(davies_bouldin_score(X_eval, labels_eval)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X_eval, labels_eval)),
        "n_clusters": int(len(unique_labels)),
        "n_noise_points": int((labels == -1).sum()) if -1 in labels else 0,
    }


def evaluate_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> dict:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X_scaled)
    metrics = _safe_metrics(X_scaled, labels)
    return {"algorithm": "KMeans", "params": {"n_clusters": k}, "metrics": metrics}


def evaluate_agglomerative(X_scaled: np.ndarray, k: int) -> dict:
    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(X_scaled)
    metrics = _safe_metrics(X_scaled, labels)
    return {"algorithm": "Agglomerative", "params": {"n_clusters": k}, "metrics": metrics}


def estimate_dbscan_eps(X_scaled: np.ndarray, min_samples: int) -> float:
    """
    Heuristic: eps = the 'knee' of the k-distance graph, approximated here
    as the 90th percentile of each point's distance to its min_samples-th
    nearest neighbor. Good enough as a starting default; tune per dataset.
    """
    n_neighbors = min(min_samples, len(X_scaled) - 1)
    if n_neighbors < 1:
        return 0.5
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_distances = np.sort(distances[:, -1])
    return float(np.percentile(k_distances, 90))


def evaluate_dbscan(
    X_scaled: np.ndarray, eps: Optional[float] = None, min_samples: int = 5
) -> dict:
    eps = eps if eps is not None else estimate_dbscan_eps(X_scaled, min_samples)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    metrics = _safe_metrics(X_scaled, labels)
    return {
        "algorithm": "DBSCAN",
        "params": {"eps": round(eps, 4), "min_samples": min_samples},
        "metrics": metrics,
    }


def evaluate_gmm(X_scaled: np.ndarray, k: int, random_state: int = 42) -> dict:
    model = GaussianMixture(n_components=k, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    metrics = _safe_metrics(X_scaled, labels)
    return {"algorithm": "GaussianMixture", "params": {"n_components": k}, "metrics": metrics}


def compare_algorithms(
    X_scaled: np.ndarray,
    k: int,
    random_state: int = 42,
    dbscan_min_samples: int = 5,
    dbscan_eps: Optional[float] = None,
) -> Dict[str, dict]:
    """Run all four algorithms and return their results keyed by algorithm name."""
    results = {
        "KMeans": evaluate_kmeans(X_scaled, k, random_state),
        "Agglomerative": evaluate_agglomerative(X_scaled, k),
        "DBSCAN": evaluate_dbscan(X_scaled, eps=dbscan_eps, min_samples=dbscan_min_samples),
        "GaussianMixture": evaluate_gmm(X_scaled, k, random_state),
    }
    for name, res in results.items():
        m = res["metrics"]
        if m:
            logger.info(
                f"{name}: silhouette={m['silhouette_score']:.4f} "
                f"davies_bouldin={m['davies_bouldin_index']:.4f} "
                f"calinski_harabasz={m['calinski_harabasz_index']:.2f} "
                f"(n_clusters={m['n_clusters']})"
            )
        else:
            logger.info(f"{name}: metrics undefined (degenerate clustering result)")
    return results


def rank_algorithms(results: Dict[str, dict]) -> list:
    """
    Rank algorithms by silhouette score (descending). Algorithms with
    undefined metrics are placed last.
    """
    scored = [
        (name, res["metrics"]["silhouette_score"] if res["metrics"] else -np.inf)
        for name, res in results.items()
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scored]


def run_evaluation_pipeline(config: SegmentationConfig | None = None, k: Optional[int] = None) -> dict:
    """
    Full Part-2 pipeline: load data, scale, compare algorithms, save metrics.json.
    If k is not given, uses config.k_min as a reasonable default (or the
    best_k discovered by trainer.run_training_pipeline, if you pass it in).
    """
    config = config or SegmentationConfig()

    df = load_customer_features(config.input_csv, config.feature_columns, config.id_column)
    X = select_features(df, config.feature_columns)
    X_scaled, _ = scale_features(X)

    k = k or max(2, min(config.k_min, len(df) - 1))
    logger.info(f"Comparing clustering algorithms with k={k} on {len(df)} customers.")

    results = compare_algorithms(X_scaled, k=k, random_state=config.random_state)
    ranking = rank_algorithms(results)

    output = {
        "k": k,
        "n_customers": len(df),
        "feature_columns": config.feature_columns,
        "results": results,
        "ranking_by_silhouette": ranking,
        "best_algorithm": ranking[0] if ranking else None,
    }

    save_json(output, METRICS_PATH)
    logger.info(f"Saved comparison metrics -> {METRICS_PATH}")
    return output


if __name__ == "__main__":
    summary = run_evaluation_pipeline()
    print(f"\nBest algorithm by silhouette score: {summary['best_algorithm']}")
    print(f"Full ranking: {summary['ranking_by_silhouette']}")
    print(f"Metrics saved to: {METRICS_PATH}")
