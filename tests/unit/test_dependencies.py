"""Unit tests for backend dependency injection factories."""

from backend.dependencies import (
    get_inference_service,
    get_report_service,
    get_upload_service,
)
from backend.services.inference_service import InferenceService
from backend.services.report_service import ReportService
from backend.services.upload_service import UploadService


def test_get_upload_service_singleton() -> None:
    """Test get_upload_service returns a cached singleton UploadService instance."""

    service1 = get_upload_service()
    service2 = get_upload_service()

    assert isinstance(service1, UploadService)
    assert service1 is service2


def test_get_inference_service_singleton_and_wiring() -> None:
    """Test get_inference_service returns a cached instance wired with upload service."""

    upload_service = get_upload_service()
    inference_service1 = get_inference_service()
    inference_service2 = get_inference_service()

    assert isinstance(inference_service1, InferenceService)
    assert inference_service1 is inference_service2
    assert inference_service1.upload_service is upload_service


def test_get_report_service_singleton() -> None:
    """Test get_report_service returns a cached singleton ReportService instance."""

    service1 = get_report_service()
    service2 = get_report_service()

    assert isinstance(service1, ReportService)
    assert service1 is service2
