"""Global exception handlers for the BrainTumorAI backend.

This module provides exception handlers that convert uncaught application and
HTTP exceptions into standardized ``ErrorResponse`` payloads.
"""

from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from backend.core.logging import get_logger
from backend.schemas.common import ErrorResponse


logger = get_logger(__name__)


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Handle uncaught ValueError exceptions.

    Args:
        request: Incoming HTTP request.
        exc: Raised ValueError exception instance.

    Returns:
        JSON response conforming to ErrorResponse with HTTP 400 Bad Request.
    """

    error_payload = ErrorResponse(
        error="bad_request",
        detail=str(exc) or "Bad request.",
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_payload.model_dump(mode="json"),
    )


async def file_not_found_handler(
    request: Request, exc: FileNotFoundError
) -> JSONResponse:
    """Handle uncaught FileNotFoundError exceptions.

    Args:
        request: Incoming HTTP request.
        exc: Raised FileNotFoundError exception instance.

    Returns:
        JSON response conforming to ErrorResponse with HTTP 404 Not Found.
    """

    error_payload = ErrorResponse(
        error="not_found",
        detail=str(exc) or "Resource not found.",
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content=error_payload.model_dump(mode="json"),
    )


async def http_exception_handler(
    request: Request, exc: HTTPException
) -> JSONResponse:
    """Handle FastAPI HTTPException exceptions.

    Args:
        request: Incoming HTTP request.
        exc: Raised HTTPException instance.

    Returns:
        JSON response conforming to ErrorResponse with the exception's status code.
    """

    detail_str = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    error_payload = ErrorResponse(
        error="http_error",
        detail=detail_str,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump(mode="json"),
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle uncaught general Exception instances.

    Args:
        request: Incoming HTTP request.
        exc: Unexpected exception instance.

    Returns:
        JSON response conforming to ErrorResponse with HTTP 500 Internal Server Error.
    """

    logger.error(
        "Unhandled exception processing request %s: %s",
        request.url,
        exc,
        exc_info=True,
    )
    error_payload = ErrorResponse(
        error="internal_server_error",
        detail="An unexpected internal server error occurred.",
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload.model_dump(mode="json"),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on a FastAPI application instance.

    Args:
        app: FastAPI application instance.
    """

    app.add_exception_handler(ValueError, value_error_handler)
    app.add_exception_handler(FileNotFoundError, file_not_found_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
