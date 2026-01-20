"""Logging configuration for ToS Inspector."""

import logging
import sys
from tosinspector.config import settings


def setup_logging() -> logging.Logger:
    """
    Configure structured logging for the application.

    Returns:
        logging.Logger: Configured logger instance
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger("tosinspector")


# Global logger instance
logger = setup_logging()
