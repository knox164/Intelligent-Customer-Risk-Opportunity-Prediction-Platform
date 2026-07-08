from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataCleaner:

    def __init__(
        self,
        input_path="data/interim/raw_transactions.csv",
        output_path="data/interim/clean_transactions.csv",
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_data(self):
        logger.info("Loading raw transaction data...")

        df = pd.read_csv(
            self.input_path,
            parse_dates=["invoicedate"]
        )

        logger.info(f"Loaded {len(df):,} rows.")

        return df

    def clean(self, df):

        logger.info("Starting data cleaning...")

        original_rows = len(df)

        # -------------------------------------------------
        # Remove duplicate rows
        # -------------------------------------------------

        duplicates = df.duplicated().sum()

        df = df.drop_duplicates()

        logger.info(f"Removed {duplicates:,} duplicate rows.")

        # -------------------------------------------------
        # Remove missing customer IDs
        # -------------------------------------------------

        missing_customers = df["customer_id"].isna().sum()

        df = df.dropna(subset=["customer_id"])

        logger.info(
            f"Removed {missing_customers:,} rows with missing customer IDs."
        )

        # -------------------------------------------------
        # Convert invoice date
        # -------------------------------------------------

        df["invoicedate"] = pd.to_datetime(
            df["invoicedate"],
            errors="coerce"
        )

        invalid_dates = df["invoicedate"].isna().sum()

        if invalid_dates > 0:

            df = df.dropna(subset=["invoicedate"])

        logger.info(
            f"Removed {invalid_dates:,} invalid dates."
        )

        # -------------------------------------------------
        # Remove negative prices
        # -------------------------------------------------

        negative_prices = (df["price"] < 0).sum()

        df = df[df["price"] >= 0]

        logger.info(
            f"Removed {negative_prices:,} negative prices."
        )

        # -------------------------------------------------
        # Remove zero quantities
        # -------------------------------------------------

        zero_quantity = (df["quantity"] == 0).sum()

        df = df[df["quantity"] != 0]

        logger.info(
            f"Removed {zero_quantity:,} zero quantity rows."
        )

        # -------------------------------------------------
        # Remove returns
        # -------------------------------------------------

        returns = (df["quantity"] < 0).sum()

        df = df[df["quantity"] > 0]

        logger.info(
            f"Removed {returns:,} returned transactions."
        )

        # -------------------------------------------------
        # Remove cancelled invoices
        # -------------------------------------------------

        cancelled = df["invoice"].astype(str).str.startswith("C")

        cancelled_count = cancelled.sum()

        df = df[~cancelled]

        logger.info(
            f"Removed {cancelled_count:,} cancelled invoices."
        )

        # -------------------------------------------------
        # Remove missing descriptions
        # -------------------------------------------------

        missing_desc = df["description"].isna().sum()

        df = df.dropna(subset=["description"])

        logger.info(
            f"Removed {missing_desc:,} missing descriptions."
        )

        # -------------------------------------------------
        # Standardize text columns
        # -------------------------------------------------

        text_columns = [
            "description",
            "country",
            "invoice",
            "stockcode",
        ]

        for column in text_columns:

            df[column] = (
                df[column]
                .astype(str)
                .str.strip()
                .str.upper()
            )

        logger.info("Standardized text columns.")

        # -------------------------------------------------
        # Create TotalPrice
        # -------------------------------------------------

        df["totalprice"] = (
            df["quantity"] * df["price"]
        )

        logger.info("Created TotalPrice column.")

        logger.info(
            f"Rows before cleaning: {original_rows:,}"
        )

        logger.info(
            f"Rows after cleaning: {len(df):,}"
        )

        logger.info(
            f"Rows removed: {original_rows-len(df):,}"
        )

        return df

    def save(self, df):

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            self.output_path,
            index=False
        )

        logger.info(
            f"Saved cleaned dataset to {self.output_path}"
        )

    def run(self):

        logger.info("=" * 60)
        logger.info("STARTING DATA CLEANING")

        df = self.load_data()

        df = self.clean(df)

        self.save(df)

        logger.info("DATA CLEANING COMPLETED")
        logger.info("=" * 60)

        return df