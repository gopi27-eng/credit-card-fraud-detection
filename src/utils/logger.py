import logging
import sys
from typing import Optional
from pythonjsonlogger import jsonlogger


def get_logger(name: str, log_level: Optional[int] = logging.INFO) -> logging.Logger:
    """Configures and returns a structured JSON logger for Kubernetes/container runtime.

    Args:
        name: Name of the logger (typically the module __name__).
        log_level: Logging severity level (default: logging.INFO).

    Returns:
        logging.Logger: Configured logger emitting JSON-formatted lines to stdout.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers if get_logger is called repeatedly in the same process
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(module)s %(funcName)s %(lineno)d %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger