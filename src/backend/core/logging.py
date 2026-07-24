"""Centralized logging configuration for the BrainTumorAI backend.

This module configures application logging using Python's standard logging
package. It provides console output and rotating file logs while deriving
runtime values from the shared application configuration.
"""

import logging
from logging import Logger
from logging.handlers import RotatingFileHandler
from pathlib import Path

from backend.core.config import settings


LOG_FORMAT: str = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"
LOG_FILE_NAME: str = "backend.log"
MAX_LOG_FILE_BYTES: int = 10 * 1024 * 1024
BACKUP_LOG_FILE_COUNT: int = 5
MANAGED_HANDLER_ATTR: str = "_brain_tumor_ai_managed"


def configure_logging() -> None:
    """Configure console and rotating file logging for the application.

    The function is safe to call multiple times. Existing handlers installed by
    this module are replaced to avoid duplicate log entries while still
    respecting the configured log level and log directory.
    """

    log_level = _resolve_log_level(settings.log_level)
    log_directory = Path(settings.log_directory)
    log_directory.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
    root_logger = logging.getLogger()

    root_logger.setLevel(log_level)
    _remove_managed_handlers(root_logger)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    setattr(console_handler, MANAGED_HANDLER_ATTR, True)

    file_handler = RotatingFileHandler(
        filename=log_directory / LOG_FILE_NAME,
        maxBytes=MAX_LOG_FILE_BYTES,
        backupCount=BACKUP_LOG_FILE_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, MANAGED_HANDLER_ATTR, True)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    """Return a logger configured for the given module name.

    Args:
        name: Name of the logger, typically ``__name__`` from the caller.

    Returns:
        A standard library logger instance.
    """

    return logging.getLogger(name)


def _resolve_log_level(log_level: str) -> int:
    """Resolve a configured log level name to a logging level constant.

    Args:
        log_level: Log level name from application settings.

    Returns:
        The corresponding integer logging level.
    """

    resolved_level = logging.getLevelName(log_level.upper())
    if isinstance(resolved_level, int):
        return resolved_level

    return logging.INFO


def _remove_managed_handlers(logger: Logger) -> None:
    """Remove handlers previously installed by this module.

    Args:
        logger: Logger whose managed handlers should be removed.
    """

    for handler in logger.handlers[:]:
        if getattr(handler, MANAGED_HANDLER_ATTR, False):
            logger.removeHandler(handler)
            handler.close()
