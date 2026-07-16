from src.utils.logger import get_logger

from src.ingestion.ingest import DataIngestion
from src.validation.validator import DataValidator
from src.preprocessing.cleaner import DataCleaner
from src.feature_engineering.feature_engineer import FeatureEngineer


logger = get_logger(__name__)


class ETLPipeline:
    """
    End-to-End ETL Pipeline

    Pipeline Flow

    Raw Excel
        ↓
    Data Ingestion
        ↓
    Data Validation
        ↓
    Data Cleaning
        ↓
    Feature Engineering
        ↓
    Processed Datasets
    """

    def __init__(self):
        logger.info("Initializing ETL Pipeline...")

    def extract(self):
        logger.info("STEP 1: DATA INGESTION")

        ingestion = DataIngestion()

        return ingestion.run()

    def validate(self):
        logger.info("STEP 2: DATA VALIDATION")

        validator = DataValidator()

        return validator.run()

    def clean(self):
        logger.info("STEP 3: DATA CLEANING")

        cleaner = DataCleaner()

        return cleaner.run()

    def transform(self):
        logger.info("STEP 4: FEATURE ENGINEERING")

        engineer = FeatureEngineer()

        return engineer.run()

    def run(self):

        try:

            logger.info("=" * 70)
            logger.info("STARTING ETL PIPELINE")
            logger.info("=" * 70)

            raw_df = self.extract()

            logger.info(
                f"Raw Dataset Shape: {raw_df.shape}"
            )

            validation_status = self.validate()

            if validation_status == "FAILED":

                logger.error(
                    "Validation failed. Pipeline terminated."
                )

                return

            clean_df = self.clean()

            logger.info(
                f"Clean Dataset Shape: {clean_df.shape}"
            )

            customer_df, product_df, transaction_df = self.transform()

            logger.info(
                f"Customer Features: {customer_df.shape}"
            )

            logger.info(
                f"Product Features: {product_df.shape}"
            )

            logger.info(
                f"Transaction Features: {transaction_df.shape}"
            )

            logger.info("=" * 70)
            logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)

        except Exception as e:

            logger.exception(
                f"Pipeline failed: {e}"
            )

            raise