"""CORS middleware configuration for the BrainTumorAI backend.

This module configures Cross-Origin Resource Sharing (CORS) middleware for the
FastAPI application using application settings.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings


def setup_cors(app: FastAPI) -> None:
    """Configure CORS middleware on the FastAPI application instance.

    Args:
        app: FastAPI application instance to attach middleware to.
    """

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=settings.allowed_methods,
        allow_headers=settings.allowed_headers,
    )
