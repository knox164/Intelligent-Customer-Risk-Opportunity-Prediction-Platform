"""
Centralized logging configuration for the Customer Risk Platform.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Information")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical error")
"""

import logging
from pathlib import Path


# -------------------------------------------------------------------
# Create logs directory if it doesn't exist
# -------------------------------------------------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "app.log"


# -------------------------------------------------------------------
# Log format
# -------------------------------------------------------------------
LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


# -------------------------------------------------------------------
# Configure root logger only once
# -------------------------------------------------------------------
def setup_logger(level=logging.INFO):
    """
    Configure the application's root logger.

    Parameters
    ----------
    level : int
        Logging level (default: INFO)
    """

    root_logger = logging.getLogger()

    if root_logger.handlers:
        return

    root_logger.setLevel(level)

    formatter = logging.Formatter(
        LOG_FORMAT,
        datefmt=DATE_FORMAT
    )

    # -----------------------------
    # Console Handler
    # -----------------------------
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # -----------------------------
    # File Handler
    # -----------------------------
    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


# -------------------------------------------------------------------
# Return logger
# -------------------------------------------------------------------
def get_logger(name: str):
    """
    Returns a logger for a module.

    Example:
        logger = get_logger(__name__)
    """
    setup_logger()
    return logging.getLogger(name)