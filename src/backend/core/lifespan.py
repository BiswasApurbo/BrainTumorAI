"""FastAPI lifespan management for the BrainTumorAI backend.

This module defines the application startup and shutdown lifecycle hook. It is
limited to infrastructure initialization that is safe to run when the FastAPI
application starts.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.config import ensure_project_directories, settings
from backend.core.logging import configure_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage FastAPI application startup and shutdown events.

    Startup configures centralized logging, ensures required project
    directories exist, and records application metadata. Shutdown records a
    graceful shutdown message.

    Args:
        app: FastAPI application instance managed by this lifespan context.

    Yields:
        Control back to FastAPI while the application is running.
    """

    configure_logging()
    logger = get_logger(__name__)

    ensured_directories = ensure_project_directories()
    logger.info(
        "Starting %s version %s in %s environment.",
        settings.app_name,
        settings.version,
        settings.environment,
    )
    logger.debug(
        "Ensured project directories: %s",
        ", ".join(str(directory) for directory in ensured_directories),
    )

    try:
        yield
    finally:
        logger.info("Shutting down %s gracefully.", settings.app_name)
