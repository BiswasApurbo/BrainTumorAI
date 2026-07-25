"""Integration tests for global exception handlers."""

from unittest.mock import patch

import pytest
from fastapi import APIRouter, HTTPException, status
from fastapi.testclient import TestClient

from backend.main import app

# Create a test-only router to trigger uncaught exceptions
dummy_exception_router = APIRouter(prefix="/test-exceptions", tags=["Testing"])


@dummy_exception_router.get("/value-error")
async def raise_value_error() -> None:
    raise ValueError("Invalid configuration parameter.")


@dummy_exception_router.get("/file-not-found")
async def raise_file_not_found() -> None:
    raise FileNotFoundError("Target model weights file missing.")


@dummy_exception_router.get("/http-exception")
async def raise_http_exception() -> None:
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied.")


@dummy_exception_router.get("/unexpected-error")
async def raise_unexpected_error() -> None:
    raise RuntimeError("Database connection crashed unexpectedly.")


app.include_router(dummy_exception_router)


def test_value_error_handler(client: TestClient) -> None:
    """Test uncaught ValueError is converted to 400 ErrorResponse."""

    response = client.get("/test-exceptions/value-error")
    assert response.status_code == 400
    data = response.json()
    assert data["error"] == "bad_request"
    assert data["detail"] == "Invalid configuration parameter."
    assert "timestamp" in data


def test_file_not_found_handler(client: TestClient) -> None:
    """Test uncaught FileNotFoundError is converted to 404 ErrorResponse."""

    response = client.get("/test-exceptions/file-not-found")
    assert response.status_code == 404
    data = response.json()
    assert data["error"] == "not_found"
    assert data["detail"] == "Target model weights file missing."
    assert "timestamp" in data


def test_http_exception_handler(client: TestClient) -> None:
    """Test HTTPException is converted to ErrorResponse while preserving status code."""

    response = client.get("/test-exceptions/http-exception")
    assert response.status_code == 403
    data = response.json()
    assert data["error"] == "http_error"
    assert data["detail"] == "Access denied."
    assert "timestamp" in data


def test_unhandled_exception_handler() -> None:
    """Test unhandled Exception is converted to 500 ErrorResponse and logged."""

    custom_client = TestClient(app, raise_server_exceptions=False)
    with patch("backend.exceptions.handlers.logger.error") as mock_logger:
        response = custom_client.get("/test-exceptions/unexpected-error")
        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_server_error"
        assert data["detail"] == "An unexpected internal server error occurred."
        assert "timestamp" in data
        mock_logger.assert_called_once()
