"""FastAPI application entry point for the BrainTumorAI backend.

This module creates the backend application instance, attaches the application
lifespan manager, and exposes lightweight root and health endpoints.
"""

from typing import Any

from fastapi import FastAPI

from backend.core.config import settings
from backend.core.lifespan import lifespan


ResponsePayload = dict[str, Any]


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan,
)


@app.get("/")
async def root() -> ResponsePayload:
    """Return basic application runtime information.

    Returns:
        A JSON-serializable payload with the application name, version, and
        running status.
    """

    return {
        "application": settings.app_name,
        "version": settings.version,
        "status": "running",
    }


@app.get("/health")
async def health() -> ResponsePayload:
    """Return the current application health status.

    Returns:
        A JSON-serializable payload with health status and application
        metadata.
    """

    return {
        "status": "healthy",
        "application": settings.app_name,
        "version": settings.version,
    }
