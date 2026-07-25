"""Unit tests for backend ReportService."""

from uuid import UUID, uuid4

import pytest

from backend.services.report_service import ReportService


def test_store_report_success(report_service: ReportService) -> None:
    """Test storing a diagnostic report for an upload."""

    upload_id = uuid4()
    content = {"summary": "Tumor detected", "confidence": 0.98}

    metadata = report_service.store_report(upload_id, content=content)

    assert isinstance(metadata.report_id, UUID)
    assert metadata.upload_id == upload_id
    assert metadata.status == "completed"
    assert report_service.report_exists(upload_id) is True


def test_store_report_duplicate_raises_value_error(report_service: ReportService) -> None:
    """Test that storing a second report for the same upload ID raises ValueError."""

    upload_id = uuid4()
    content = {"summary": "Report 1"}

    report_service.store_report(upload_id, content=content)

    with pytest.raises(ValueError, match="A report already exists for upload"):
        report_service.store_report(upload_id, content={"summary": "Report 2"})


def test_get_report_success(report_service: ReportService) -> None:
    """Test retrieving a stored report by upload ID."""

    upload_id = uuid4()
    content = {"summary": "Retrieved report"}

    stored_metadata = report_service.store_report(upload_id, content=content)
    retrieved_report = report_service.get_report(upload_id)

    assert retrieved_report.report_id == stored_metadata.report_id
    assert retrieved_report.upload_id == upload_id
    assert retrieved_report.content == content


def test_get_report_not_found(report_service: ReportService) -> None:
    """Test retrieving a non-existent report raises KeyError."""

    with pytest.raises(KeyError, match="No report found for upload"):
        report_service.get_report(uuid4())


def test_list_reports(report_service: ReportService) -> None:
    """Test listing all stored report metadatas."""

    assert len(report_service.list_reports()) == 0

    u1, u2 = uuid4(), uuid4()
    report_service.store_report(u1, {"id": 1})
    report_service.store_report(u2, {"id": 2})

    reports = report_service.list_reports()
    assert len(reports) == 2
    upload_ids = {r.upload_id for r in reports}
    assert upload_ids == {u1, u2}


def test_delete_report_success(report_service: ReportService) -> None:
    """Test deleting a stored report by upload ID."""

    upload_id = uuid4()
    report_service.store_report(upload_id, {"data": "test"})

    assert report_service.delete_report(upload_id) is True
    assert report_service.report_exists(upload_id) is False

    with pytest.raises(KeyError):
        report_service.get_report(upload_id)


def test_delete_report_not_found(report_service: ReportService) -> None:
    """Test deleting a non-existent report raises KeyError."""

    with pytest.raises(KeyError, match="No report found for upload"):
        report_service.delete_report(uuid4())


def test_invalid_upload_id_normalization(report_service: ReportService) -> None:
    """Test that blank or malformed upload IDs raise ValueError."""

    with pytest.raises(ValueError, match="upload_id must be provided"):
        report_service.store_report("", {"data": 1})

    with pytest.raises(ValueError):
        report_service.get_report("not-a-valid-uuid")
