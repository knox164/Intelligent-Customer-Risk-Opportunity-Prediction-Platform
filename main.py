from src.utils.logger import get_logger
from src.ingestion.ingest import DataIngestion
from src.validation.validator import DataValidator

logger = get_logger(__name__)


def main():
    
    ingestion = DataIngestion()

    df = ingestion.run()
    validator = DataValidator()
    validator.run()

    logger.info(
        f"Final dataset shape: {df.shape}"
    )
    logger.info("Customer Risk Platform Started")

    logger.debug("Debugging application...")

    logger.warning("This is a warning.")

    logger.error("Example error.")

    logger.critical("Critical example.")


if __name__ == "__main__":
    main()