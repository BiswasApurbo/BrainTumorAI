"""Pytest configuration and shared fixtures for the BrainTumorAI test suite."""

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure src/ is on Python path so `backend` package can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import importlib.util

_deps_file = os.path.join(os.path.dirname(__file__), "..", "src", "backend", "dependencies.py")
_spec = importlib.util.spec_from_file_location("backend.dependencies", _deps_file)
_backend_dependencies = importlib.util.module_from_spec(_spec)
sys.modules["backend.dependencies"] = _backend_dependencies
_spec.loader.exec_module(_backend_dependencies)

from backend.core.config import settings
from backend.dependencies import (
    get_inference_service,
    get_report_service,
    get_upload_service,
)
from backend.main import app
from backend.services.inference_service import InferenceService
from backend.services.report_service import ReportService
from backend.services.upload_service import UploadService


@pytest.fixture(autouse=True)
def reset_service_caches() -> Generator[None, None, None]:
    """Clear lru_cache on service dependency factories before and after each test."""

    get_upload_service.cache_clear()
    get_inference_service.cache_clear()
    get_report_service.cache_clear()
    yield
    get_upload_service.cache_clear()
    get_inference_service.cache_clear()
    get_report_service.cache_clear()


@pytest.fixture
def temp_upload_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide a temporary directory for file uploads and override settings."""

    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "uploads_directory", upload_dir)
    return upload_dir


@pytest.fixture
def client(temp_upload_dir: Path) -> TestClient:
    """Provide a FastAPI TestClient with temporary upload directory configured."""

    return TestClient(app)


@pytest.fixture
def upload_service(temp_upload_dir: Path) -> UploadService:
    """Provide a fresh UploadService instance pointed at a temporary directory."""

    return UploadService(upload_directory=temp_upload_dir)


@pytest.fixture
def inference_service(upload_service: UploadService) -> InferenceService:
    """Provide a fresh InferenceService instance using the temporary upload_service."""

    return InferenceService(upload_service=upload_service)


@pytest.fixture
def report_service() -> ReportService:
    """Provide a fresh ReportService instance."""

    return ReportService()
