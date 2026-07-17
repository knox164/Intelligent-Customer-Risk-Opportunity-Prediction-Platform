"""
run_pipeline.py
----------------
Convenience entry point that runs the full segmentation pipeline in order:

    1. trainer.py     -> segmentation_model.pkl, scaler.pkl
    2. evaluator.py    -> metrics.json (algorithm comparison)
    3. visualizer.py   -> elbow/silhouette/PCA/cluster-size plots
    4. report.py       -> customer_segments.csv, segmentation_report.md

Run from the project root:
    python run_pipeline.py
    python run_pipeline.py --input path/to/customer_features.csv
"""

import argparse
from pathlib import Path

from src.ML.segmentation.evaluator import run_evaluation_pipeline
from src.ML.segmentation.report import run_report_pipeline
from src.ML.segmentation.trainer import run_training_pipeline
from src.ML.segmentation.utils import SegmentationConfig, get_logger
from src.ML.segmentation.visualizer import run_visualization_pipeline

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run the full customer segmentation pipeline.")
    parser.add_argument("--input", type=Path, default=None, help="Path to customer_features.csv")
    args = parser.parse_args()

    config = SegmentationConfig(input_csv=args.input) if args.input else SegmentationConfig()

    logger.info("=== Step 1/4: Training ===")
    training_summary = run_training_pipeline(config)

    logger.info("=== Step 2/4: Evaluating algorithms ===")
    metrics = run_evaluation_pipeline(config, k=training_summary["best_k"])

    logger.info("=== Step 3/4: Visualizing ===")
    plots = run_visualization_pipeline(config, training_summary=training_summary)

    logger.info("=== Step 4/4: Reporting ===")
    report = run_report_pipeline(config, training_summary=training_summary, metrics=metrics)

    print("\nPipeline complete.")
    print(f"  Best k: {training_summary['best_k']}")
    print(f"  Best algorithm: {metrics['best_algorithm']}")
    print(f"  Model: {training_summary['model_path']}")
    print(f"  Scaler: {training_summary['scaler_path']}")
    print(f"  Metrics: outputs/metrics.json")
    print(f"  Plots: {list(plots.values())}")
    print(f"  Segments CSV: {report['segments_csv']}")
    print(f"  Report: {report['report_md']}")


if __name__ == "__main__":
    main()
