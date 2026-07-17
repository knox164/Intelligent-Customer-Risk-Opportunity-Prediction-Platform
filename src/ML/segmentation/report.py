"""
report.py
---------
Part 4 of the segmentation module.

  - Builds per-cluster profiles (mean/median of each feature, size, % of base)
  - Assigns human-readable names to clusters using simple RFM heuristics
  - Writes customer_segments.csv  (customer_id + assigned cluster + label)
  - Writes segmentation_report.md (readable summary of the segmentation run)

Usage:
    python -m src.models.segmentation.report
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ....models.reccomendation.segmentation.trainer import run_training_pipeline, scale_features, select_features, train_kmeans
from .utils import (
    REPORT_MD_PATH,
    SEGMENTS_CSV_PATH,
    SegmentationConfig,
    get_logger,
    load_customer_features,
)

logger = get_logger(__name__)


def build_cluster_profiles(df: pd.DataFrame, feature_columns: list, cluster_col: str = "cluster") -> pd.DataFrame:
    """
    Mean of each feature per cluster, plus cluster size and share of the
    total customer base. One row per cluster.
    """
    profile = df.groupby(cluster_col)[feature_columns].mean()
    sizes = df[cluster_col].value_counts().sort_index()
    profile["cluster_size"] = sizes
    profile["pct_of_customers"] = (sizes / len(df) * 100).round(2)
    return profile.reset_index()


def name_cluster(row: pd.Series, medians: pd.Series) -> str:
    """
    Simple, transparent RFM-style naming heuristic:
      - low recency + high frequency + high monetary  -> "Champions"
      - low recency + moderate frequency               -> "Loyal Customers"
      - high recency + low frequency                   -> "At Risk" / "Lost"
      - high monetary, low frequency                    -> "Big Spenders (Infrequent)"
      - everything else                                 -> "Standard Customers"

    Comparisons are made against the overall median for each metric so the
    naming adapts to whatever dataset is passed in.
    """
    recency_low = row["recency"] <= medians["recency"]
    frequency_high = row["frequency"] >= medians["frequency"]
    monetary_high = row["monetary"] >= medians["monetary"]

    if recency_low and frequency_high and monetary_high:
        return "Champions"
    if recency_low and frequency_high:
        return "Loyal Customers"
    if (not recency_low) and (not frequency_high):
        return "At Risk / Lost"
    if monetary_high and (not frequency_high):
        return "Big Spenders (Infrequent)"
    if recency_low and (not frequency_high):
        return "New / Promising"
    return "Standard Customers"


def assign_cluster_names(profile: pd.DataFrame, feature_columns: list) -> Dict[int, str]:
    medians = profile[feature_columns].median()
    names: Dict[int, str] = {}
    used = {}
    for _, row in profile.iterrows():
        base_name = name_cluster(row, medians)
        # Disambiguate if two clusters would get the same name.
        count = used.get(base_name, 0)
        used[base_name] = count + 1
        final_name = base_name if count == 0 else f"{base_name} ({count + 1})"
        names[int(row["cluster"])] = final_name
    return names


def write_segments_csv(
    df: pd.DataFrame, id_column: str, cluster_col: str, cluster_names: Dict[int, str], path=SEGMENTS_CSV_PATH
) -> None:
    out = df[[id_column, cluster_col]].copy()
    out["segment_name"] = out[cluster_col].map(cluster_names)
    out = out.rename(columns={cluster_col: "cluster"})
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    logger.info(f"Saved customer segments -> {path}")


def _format_profile_table(profile: pd.DataFrame, feature_columns: list, cluster_names: Dict[int, str]) -> str:
    header_cols = ["Cluster", "Segment Name"] + [c.replace("_", " ").title() for c in feature_columns] + ["Size", "% of Base"]
    lines = ["| " + " | ".join(header_cols) + " |", "|" + "---|" * len(header_cols)]
    for _, row in profile.iterrows():
        cluster_id = int(row["cluster"])
        values = [f"{row[c]:,.2f}" for c in feature_columns]
        line = (
            f"| {cluster_id} | {cluster_names[cluster_id]} | "
            + " | ".join(values)
            + f" | {int(row['cluster_size'])} | {row['pct_of_customers']}% |"
        )
        lines.append(line)
    return "\n".join(lines)


def write_report_md(
    profile: pd.DataFrame,
    feature_columns: list,
    cluster_names: Dict[int, str],
    n_customers: int,
    best_k: int,
    metrics: Optional[dict] = None,
    path=REPORT_MD_PATH,
) -> None:
    table = _format_profile_table(profile, feature_columns, cluster_names)

    lines = [
        "# Customer Segmentation Report",
        "",
        f"- **Total customers analyzed:** {n_customers}",
        f"- **Number of segments (k):** {best_k}",
        "",
        "## Cluster Profiles",
        "",
        "Values below are per-cluster feature means.",
        "",
        table,
        "",
        "## Segment Descriptions",
        "",
    ]

    for cluster_id, name in sorted(cluster_names.items()):
        lines.append(f"**Cluster {cluster_id} — {name}:** ")
        row = profile[profile["cluster"] == cluster_id].iloc[0]
        lines.append(
            f"{int(row['cluster_size'])} customers ({row['pct_of_customers']}% of base). "
            f"Avg recency {row.get('recency', float('nan')):.0f} days, "
            f"avg frequency {row.get('frequency', float('nan')):.1f} orders, "
            f"avg monetary value {row.get('monetary', float('nan')):,.2f}."
        )
        lines.append("")

    if metrics:
        lines += [
            "## Evaluation Metrics (Best Algorithm)",
            "",
            f"- **Algorithm:** {metrics.get('best_algorithm', 'N/A')}",
        ]
        best = metrics.get("results", {}).get(metrics.get("best_algorithm", ""), {})
        m = best.get("metrics")
        if m:
            lines += [
                f"- **Silhouette Score:** {m['silhouette_score']:.4f}",
                f"- **Davies-Bouldin Index:** {m['davies_bouldin_index']:.4f}",
                f"- **Calinski-Harabasz Index:** {m['calinski_harabasz_index']:.2f}",
            ]
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))
    logger.info(f"Saved segmentation report -> {path}")


def run_report_pipeline(
    config: Optional[SegmentationConfig] = None,
    training_summary: Optional[dict] = None,
    metrics: Optional[dict] = None,
) -> Dict[str, str]:
    """Full Part-4 pipeline: profile clusters, name them, write CSV + Markdown report."""
    config = config or SegmentationConfig()

    if training_summary is None:
        training_summary = run_training_pipeline(config)

    df = load_customer_features(config.input_csv, config.feature_columns, config.id_column)
    X = select_features(df, config.feature_columns)
    X_scaled, _ = scale_features(X)

    best_k = training_summary["best_k"]
    model = train_kmeans(X_scaled, best_k, config.random_state)
    df = df.copy()
    df["cluster"] = model.predict(X_scaled)

    profile = build_cluster_profiles(df, config.feature_columns)
    cluster_names = assign_cluster_names(profile, config.feature_columns)

    write_segments_csv(df, config.id_column, "cluster", cluster_names)
    write_report_md(
        profile, config.feature_columns, cluster_names, len(df), best_k, metrics=metrics
    )

    return {
        "segments_csv": str(SEGMENTS_CSV_PATH),
        "report_md": str(REPORT_MD_PATH),
        "cluster_names": cluster_names,
    }


if __name__ == "__main__":
    result = run_report_pipeline()
    print(f"\nSegments CSV: {result['segments_csv']}")
    print(f"Report: {result['report_md']}")
    print(f"Cluster names: {result['cluster_names']}")
