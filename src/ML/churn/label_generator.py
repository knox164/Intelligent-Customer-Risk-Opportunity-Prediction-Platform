"""
Generate churn labels from customer features.

Business Rule:
--------------
A customer is considered churned if they have not made a purchase
within the last CHURN_THRESHOLD_DAYS of the observation period.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChurnLabelGenerator:

    REQUIRED_COLUMNS = [
        "customer_id",
        "last_purchase",
        "recency"
    ]

    def __init__(
        self,
        input_path="data/processed/customer_features.csv",
        output_path="data/processed/customer_churn.csv",
        churn_threshold_days=180,
    ):

        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.threshold = churn_threshold_days

    # -------------------------------------------------------
    # Load Customer Features
    # -------------------------------------------------------

    def load_data(self):

        logger.info("Loading customer features...")

        if not self.input_path.exists():

            raise FileNotFoundError(
                f"File not found: {self.input_path}"
            )

        df = pd.read_csv(
            self.input_path,
            parse_dates=["first_purchase", "last_purchase"]
        )

        logger.info(
            f"Loaded {len(df):,} customers."
        )

        return df

    # -------------------------------------------------------
    # Validate Required Columns
    # -------------------------------------------------------

    def validate_columns(self, df):

        logger.info("Validating required columns...")

        missing = [
            c
            for c in self.REQUIRED_COLUMNS
            if c not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Missing columns: {missing}"
            )

        logger.info("Column validation passed.")

    # -------------------------------------------------------
    # Generate Labels
    # -------------------------------------------------------

    def generate_labels(self, df):

        logger.info(
            f"Generating churn labels using "
            f"{self.threshold}-day threshold..."
        )

        # ---------------------------------------------------
        # Reference date
        # ---------------------------------------------------

        reference_date = df["last_purchase"].max()

        logger.info(
            f"Reference Date: {reference_date.date()}"
        )

        # ---------------------------------------------------
        # Days since last purchase
        # ---------------------------------------------------

        df["days_since_last_purchase"] = (
            reference_date -
            df["last_purchase"]
        ).dt.days

        # ---------------------------------------------------
        # Binary Target
        # ---------------------------------------------------

        df["churn"] = (
            df["days_since_last_purchase"] >
            self.threshold
        ).astype(int)

        churned = df["churn"].sum()

        active = len(df) - churned

        logger.info(
            f"Active Customers : {active:,}"
        )

        logger.info(
            f"Churned Customers: {churned:,}"
        )

        return df

    # -------------------------------------------------------
    # Save
    # -------------------------------------------------------

    def save_data(self, df):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            self.output_path,
            index=False
        )

        logger.info(
            f"Saved labels to {self.output_path}"
        )

    # -------------------------------------------------------
    # Summary
    # -------------------------------------------------------

    def summary(self, df):

        logger.info("=" * 60)

        logger.info("CHURN LABEL SUMMARY")

        logger.info("=" * 60)

        counts = df["churn"].value_counts()

        active = counts.get(0, 0)
        churned = counts.get(1, 0)

        total = len(df)

        logger.info(
            f"Total Customers : {total:,}"
        )

        logger.info(
            f"Active Customers: {active:,}"
        )

        logger.info(
            f"Churned Customers: {churned:,}"
        )

        logger.info(
            f"Churn Rate: "
            f"{(churned/total)*100:.2f}%"
        )

        logger.info("=" * 60)

    # -------------------------------------------------------
    # Pipeline
    # -------------------------------------------------------

    def run(self):

        try:

            logger.info("=" * 70)
            logger.info("STARTING CHURN LABEL GENERATION")
            logger.info("=" * 70)

            df = self.load_data()

            self.validate_columns(df)

            df = self.generate_labels(df)

            self.save_data(df)

            self.summary(df)

            logger.info(
                "CHURN LABEL GENERATION COMPLETED"
            )

            logger.info("=" * 70)

            return df

        except Exception as e:

            logger.exception(
                f"Label generation failed: {e}"
            )

            raise