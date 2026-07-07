from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:

    REQUIRED_COLUMNS = [
        "invoice",
        "stockcode",
        "description",
        "quantity",
        "invoicedate",
        "price",
        "customer_id",
        "country",
    ]

    EXPECTED_DTYPES = {
        "invoice": "object",
        "stockcode": "object",
        "description": "object",
        "quantity": "int64",
        "invoicedate": "datetime64[ns]",
        "price": "float64",
        "customer_id": "float64",
        "country": "object",
    }

    def __init__(
        self,
        input_path="data/interim/raw_transactions.csv",
        report_path="reports/validation_report.txt",
    ):
        self.input_path = Path(input_path)
        self.report_path = Path(report_path)
        self.errors = []
        self.warnings = []

    def load_data(self):
        logger.info("Loading dataset for validation...")

        df = pd.read_csv(
            self.input_path,
            parse_dates=["invoicedate"],
        )

        logger.info(f"Dataset shape: {df.shape}")

        return df

    def check_empty(self, df):
        if df.empty:
            self.errors.append("Dataset is empty.")

    def check_required_columns(self, df):
        missing = [
            col
            for col in self.REQUIRED_COLUMNS
            if col not in df.columns
        ]

        if missing:
            self.errors.append(
                f"Missing required columns: {missing}"
            )

    def check_duplicate_rows(self, df):
        duplicates = df.duplicated().sum()

        if duplicates > 0:
            self.warnings.append(
                f"{duplicates} duplicate rows found."
            )

    def check_missing_customer_ids(self, df):
        missing = df["customer_id"].isna().sum()

        if missing > 0:
            self.warnings.append(
                f"{missing} missing Customer IDs."
            )

    def check_negative_prices(self, df):
        negatives = (df["price"] < 0).sum()

        if negatives > 0:
            self.warnings.append(
                f"{negatives} negative prices found."
            )

    def check_invalid_quantities(self, df):
        invalid = (df["quantity"] == 0).sum()

        if invalid > 0:
            self.warnings.append(
                f"{invalid} zero quantities found."
            )

    def check_invalid_dates(self, df):
        invalid = df["invoicedate"].isna().sum()

        if invalid > 0:
            self.warnings.append(
                f"{invalid} invalid dates."
            )

    def check_dtypes(self, df):
        for column, expected in self.EXPECTED_DTYPES.items():

            actual = str(df[column].dtype)

            if actual != expected:
                self.warnings.append(
                    f"{column}: expected {expected}, got {actual}"
                )

    def generate_report(self):

        self.report_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        status = (
            "PASSED"
            if len(self.errors) == 0
            else "FAILED"
        )

        with open(self.report_path, "w") as f:

            f.write("=" * 60 + "\n")
            f.write("DATA VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"STATUS: {status}\n\n")

            f.write("ERRORS\n")
            f.write("-" * 30 + "\n")

            if self.errors:
                for e in self.errors:
                    f.write(f"- {e}\n")
            else:
                f.write("None\n")

            f.write("\n")

            f.write("WARNINGS\n")
            f.write("-" * 30 + "\n")

            if self.warnings:
                for w in self.warnings:
                    f.write(f"- {w}\n")
            else:
                f.write("None\n")

        logger.info(
            f"Validation report saved to {self.report_path}"
        )

        return status

    def run(self):

        logger.info("=" * 60)
        logger.info("STARTING DATA VALIDATION")

        df = self.load_data()

        self.check_empty(df)
        self.check_required_columns(df)
        self.check_duplicate_rows(df)
        self.check_missing_customer_ids(df)
        self.check_negative_prices(df)
        self.check_invalid_dates(df)
        self.check_invalid_quantities(df)
        self.check_dtypes(df)

        status = self.generate_report()

        logger.info(f"VALIDATION STATUS: {status}")
        logger.info("=" * 60)

        return status