"""
predictor.py
------------
Part 5 of the segmentation module.

Loads the persisted scaler.pkl and segmentation_model.pkl (produced by
trainer.py) and assigns cluster labels to new/unseen customers.

Usage (CLI):
    python -m src.models.segmentation.predictor --input new_customers.csv --output predictions.csv

Usage (library):
    from src.models.segmentation.predictor import SegmentPredictor
    predictor = SegmentPredictor()
    labeled_df = predictor.predict_dataframe(new_customers_df)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .utils import (
    DEFAULT_FEATURE_COLUMNS,
    ID_COLUMN,
    MODEL_PATH,
    SCALER_PATH,
    get_logger,
    load_pickle,
)

logger = get_logger(__name__)


class SegmentPredictor:
    """
    Thin wrapper around the saved scaler + clustering model so callers don't
    need to worry about load order, missing files, or column alignment.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        scaler_path: Union[str, Path] = SCALER_PATH,
        feature_columns: Optional[list] = None,
    ):
        self.model_path = Path(model_path)
        self.scaler_path = Path(scaler_path)
        self.feature_columns = feature_columns or list(DEFAULT_FEATURE_COLUMNS)

        logger.info(f"Loading model from {self.model_path}")
        self.model = load_pickle(self.model_path)
        logger.info(f"Loading scaler from {self.scaler_path}")
        self.scaler = load_pickle(self.scaler_path)

    def _validate_and_select(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Input data is missing required feature column(s): {missing}")
        if df[self.feature_columns].isnull().any().any():
            raise ValueError(
                "Input data contains missing values in feature columns; "
                "impute or drop them before predicting."
            )
        return df[self.feature_columns]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cluster labels for a DataFrame already restricted to feature columns."""
        X_selected = self._validate_and_select(X)
        X_scaled = self.scaler.transform(X_selected)
        return self.model.predict(X_scaled)

    def predict_dataframe(
        self, df: pd.DataFrame, id_column: str = ID_COLUMN, output_col: str = "cluster"
    ) -> pd.DataFrame:
        """
        Predict clusters for new customers and return a copy of the input
        DataFrame with an added cluster-label column.
        """
        labels = self.predict(df)
        result = df.copy()
        result[output_col] = labels
        logger.info(f"Predicted clusters for {len(result)} customers.")
        return result

    def predict_single(self, customer_features: dict) -> int:
        """Predict the cluster for a single customer given as a dict of feature: value."""
        df = pd.DataFrame([customer_features])
        return int(self.predict(df)[0])


def run_prediction_cli(input_csv: Path, output_csv: Path, id_column: str = ID_COLUMN) -> Path:
    df = pd.read_csv(input_csv)
    predictor = SegmentPredictor()
    result = predictor.predict_dataframe(df, id_column=id_column)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    logger.info(f"Saved predictions -> {output_csv}")
    return output_csv


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict customer segments for new data.")
    parser.add_argument("--input", type=Path, required=True, help="Path to new customer features CSV.")
    parser.add_argument(
        "--output", type=Path, default=Path("predictions.csv"), help="Path to write predictions CSV."
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out_path = run_prediction_cli(args.input, args.output)
    print(f"Predictions saved to: {out_path}")
