"""Health endpoint for the BrainTumorAI API.

This module exposes a lightweight health route that returns application
metadata and the current UTC timestamp without performing external dependency
checks.
"""

from datetime import datetime, timezone

from fastapi import APIRouter, status

from ....core.config import settings
from ....schemas.responses import HealthResponse


router = APIRouter()


@router.get(
    "",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Application health check",
)
async def get_health() -> HealthResponse:
    """Return the current health status for the backend application.

    Returns:
        Application health status, metadata, runtime environment, and a
        dynamically generated UTC ISO 8601 timestamp.
    """

    return HealthResponse(
        status="healthy",
        application=settings.app_name,
        version=settings.version,
        environment=settings.environment,
        timestamp=datetime.now(timezone.utc),
    )
