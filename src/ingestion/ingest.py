from pathlib import Path
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """
    Data Ingestion Module

    Responsibilities:
    - Read Excel file
    - Read both sheets
    - Combine datasets
    - Standardize column names
    - Convert dates
    - Save raw CSV
    """

    def __init__(
        self,
        input_path="data/raw/online_retail_II.xlsx",
        output_path="data/interim/raw_transactions.csv"
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def load_excel(self) -> pd.DataFrame:
        """
        Load and combine both sheets.
        """

        logger.info("Loading Excel file...")

        if not self.input_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.input_path}"
            )

        excel_file = pd.ExcelFile(self.input_path)

        logger.info(
            f"Sheets detected: {excel_file.sheet_names}"
        )

        sheet_1 = pd.read_excel(
            excel_file,
            sheet_name=0
        )

        sheet_2 = pd.read_excel(
            excel_file,
            sheet_name=1
        )

        logger.info(
            f"Sheet 1 shape: {sheet_1.shape}"
        )

        logger.info(
            f"Sheet 2 shape: {sheet_2.shape}"
        )

        combined_df = pd.concat(
            [sheet_1, sheet_2],
            ignore_index=True
        )

        logger.info(
            f"Combined shape: {combined_df.shape}"
        )

        return combined_df

    def standardize_columns(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Standardize column names.
        """

        logger.info(
            "Standardizing column names..."
        )

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        logger.info(
            f"Columns: {list(df.columns)}"
        )

        return df

    def convert_dates(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Convert invoice_date to datetime.
        """

        logger.info(
            "Converting invoice_date..."
        )

        if "invoicedate" in df.columns:
            df["invoicedate"] = pd.to_datetime(
                df["invoicedate"],
                errors="coerce"
            )

        elif "invoice_date" in df.columns:
            df["invoice_date"] = pd.to_datetime(
                df["invoice_date"],
                errors="coerce"
            )

        logger.info(
            "Date conversion completed."
        )

        return df

    def save_data(
        self,
        df: pd.DataFrame
    ) -> None:
        """
        Save combined dataset.
        """

        logger.info(
            "Saving raw transactions..."
        )

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            self.output_path,
            index=False
        )

        logger.info(
            f"Saved to {self.output_path}"
        )

    def run(self) -> pd.DataFrame:
        """
        Execute ingestion pipeline.
        """

        try:

            logger.info(
                "=" * 60
            )

            logger.info(
                "STARTING DATA INGESTION"
            )

            df = self.load_excel()

            df = self.standardize_columns(df)

            df = self.convert_dates(df)

            self.save_data(df)

            logger.info(
                "DATA INGESTION COMPLETED"
            )

            logger.info(
                "=" * 60
            )

            return df

        except Exception as e:

            logger.exception(
                f"Data ingestion failed: {e}"
            )

            raise