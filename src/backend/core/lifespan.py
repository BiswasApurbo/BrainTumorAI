import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from backend.core.config import ensure_project_directories, settings
from backend.core.logging import configure_logging, get_logger
from backend.dependencies import get_report_service
from backend.services.report_service import ReportService

logger = get_logger(__name__)


async def _periodic_expiration_task(report_service: ReportService) -> None:
    """Periodically sweep and delete analyses older than 24 hours."""

    while True:
        try:
            await asyncio.sleep(3600)  # Check every hour
            purged = report_service.cleanup_expired_reports(max_age_hours=24)
            if purged > 0:
                logger.info("Periodic cleanup swept and purged %d expired report(s).", purged)
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("Error in periodic expiration cleanup task: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage FastAPI application startup and shutdown events.

    Startup configures centralized logging, ensures required project
    directories exist, sweeps expired reports, and launches background
    maintenance tasks.
    """

    configure_logging()

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

    report_service = get_report_service()
    initial_purged = report_service.cleanup_expired_reports(max_age_hours=24)
    if initial_purged > 0:
        logger.info("Startup sweep purged %d expired report(s).", initial_purged)

    cleanup_task = asyncio.create_task(_periodic_expiration_task(report_service))

    try:
        yield
    finally:
        cleanup_task.cancel()
        logger.info("Shutting down %s gracefully.", settings.app_name)

