"""Health endpoint for the BrainTumorAI API.

This module exposes a lightweight health route that returns application
metadata and the current UTC timestamp without performing external dependency
checks.
"""

from datetime import datetime, timezone

from fastapi import APIRouter

from ....core.config import settings


router = APIRouter()

HealthResponse = dict[str, str]


@router.get("")
async def get_health() -> HealthResponse:
    """Return the current health status for the backend application.

    Returns:
        A JSON-serializable payload containing health status, application
        metadata, runtime environment, and a dynamically generated UTC ISO 8601
        timestamp.
    """

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
