"""
visualizer.py
-------------
Part 3 of the segmentation module.

Produces:
  - PCA (2D) projection of the feature space
  - Elbow plot (inertia vs k)
  - Silhouette plot (score vs k)
  - PCA cluster scatter plot (colored by cluster label)
  - Cluster size bar chart

All plots are saved as PNGs under outputs/plots/ and use a non-interactive
backend so this runs fine on a headless server.
 
Usage:
    python -m src.models.segmentation.visualizer
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ....models.reccomendation.segmentation.trainer import run_training_pipeline, scale_features, select_features, train_kmeans
from .utils import PLOT_DIR, SegmentationConfig, get_logger, load_customer_features

logger = get_logger(__name__)

plt.rcParams.update({"figure.figsize": (8, 5), "axes.grid": True, "grid.alpha": 0.3})


def run_pca(X_scaled: np.ndarray, n_components: int = 2) -> tuple:
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(X_scaled)
    return components, pca


def plot_elbow(inertias: Dict[int, float], save_path: Path = PLOT_DIR / "elbow_plot.png") -> Path:
    ks = sorted(inertias.keys())
    values = [inertias[k] for k in ks]

    fig, ax = plt.subplots()
    ax.plot(ks, values, marker="o", linewidth=2)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xticks(ks)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved elbow plot -> {save_path}")
    return save_path


def plot_silhouette(
    scores: Dict[int, float], save_path: Path = PLOT_DIR / "silhouette_plot.png"
) -> Path:
    ks = sorted(scores.keys())
    values = [scores[k] for k in ks]
    best_k = max(scores, key=scores.get)

    fig, ax = plt.subplots()
    ax.plot(ks, values, marker="o", linewidth=2, color="darkorange")
    ax.axvline(best_k, color="gray", linestyle="--", alpha=0.7, label=f"best k={best_k}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Score vs k")
    ax.set_xticks(ks)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved silhouette plot -> {save_path}")
    return save_path


def plot_pca_clusters(
    components: np.ndarray,
    labels: np.ndarray,
    save_path: Path = PLOT_DIR / "pca_cluster_plot.png",
) -> Path:
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        components[:, 0], components[:, 1], c=labels, cmap="tab10", s=60, alpha=0.8, edgecolor="k", linewidth=0.3
    )
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer Segments (PCA-projected)")
    legend1 = ax.legend(*scatter.legend_elements(), title="Cluster", loc="best")
    ax.add_artist(legend1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved PCA cluster plot -> {save_path}")
    return save_path


def plot_cluster_sizes(
    labels: np.ndarray, save_path: Path = PLOT_DIR / "cluster_size_chart.png"
) -> Path:
    unique, counts = np.unique(labels, return_counts=True)
    fig, ax = plt.subplots()
    bars = ax.bar([str(u) for u in unique], counts, color="steelblue", edgecolor="black")
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count),
                ha="center", va="bottom")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of customers")
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved cluster size chart -> {save_path}")
    return save_path


def run_visualization_pipeline(
    config: Optional[SegmentationConfig] = None,
    training_summary: Optional[dict] = None,
) -> Dict[str, Path]:
    """
    Full Part-3 pipeline. If training_summary (from trainer.run_training_pipeline)
    is not supplied, this will train a fresh model to obtain inertias/silhouette
    scores/labels so it can also be run standalone.
    """
    config = config or SegmentationConfig()

    if training_summary is None:
        training_summary = run_training_pipeline(config)

    df = load_customer_features(config.input_csv, config.feature_columns, config.id_column)
    X = select_features(df, config.feature_columns)
    X_scaled, _ = scale_features(X)

    best_k = training_summary["best_k"]
    model = train_kmeans(X_scaled, best_k, config.random_state)
    labels = model.predict(X_scaled)

    components, _ = run_pca(X_scaled)

    paths = {
        "elbow_plot": plot_elbow(training_summary["inertias"]),
        "silhouette_plot": plot_silhouette(training_summary["silhouette_scores"]),
        "pca_cluster_plot": plot_pca_clusters(components, labels),
        "cluster_size_chart": plot_cluster_sizes(labels),
    }
    return paths


if __name__ == "__main__":
    paths = run_visualization_pipeline()
    print("\nGenerated plots:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
