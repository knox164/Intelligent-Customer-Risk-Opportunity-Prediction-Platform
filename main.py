from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    logger.info("Customer Risk Platform Started")

    logger.debug("Debugging application...")

    logger.warning("This is a warning.")

    logger.error("Example error.")

    logger.critical("Critical example.")


if __name__ == "__main__":
    main()