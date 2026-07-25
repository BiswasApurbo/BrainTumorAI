"""Integration tests for POST /api/v1/upload endpoint."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from backend.dependencies import get_upload_service
from backend.main import app
from backend.services.upload_service import UploadMetadata


def test_upload_file_success(client: TestClient) -> None:
    """Test uploading a valid medical image file."""

    files = {"file": ("brain_scan.nii.gz", BytesIO(b"valid_nifti_data"), "application/octet-stream")}
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 201
    data = response.json()
    assert "upload_id" in data
    assert data["original_filename"] == "brain_scan.nii.gz"
    assert data["stored_filename"].endswith(".nii.gz")
    assert data["file_size_bytes"] == len(b"valid_nifti_data")
    assert data["status"] == "uploaded"


def test_upload_file_unsupported_extension(client: TestClient) -> None:
    """Test uploading an unsupported extension returns 415 Unsupported Media Type."""

    files = {"file": ("image.jpg", BytesIO(b"jpeg_data"), "image/jpeg")}
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 415
    data = response.json()
    assert data["detail"] == "Unsupported file extension."


def test_upload_file_empty(client: TestClient) -> None:
    """Test uploading an empty file returns 400 Bad Request."""

    files = {"file": ("empty.nii", BytesIO(b""), "application/octet-stream")}
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Uploaded file is empty."


def test_upload_file_multiple_files_rejected(client: TestClient) -> None:
    """Test uploading multiple files in the file field returns 400 Bad Request."""

    files = [
        ("file", ("scan1.nii", BytesIO(b"data1"), "application/octet-stream")),
        ("file", ("scan2.nii", BytesIO(b"data2"), "application/octet-stream")),
    ]
    response = client.post("/api/v1/upload", files=files)

    assert response.status_code == 400
    data = response.json()
    assert data["detail"] == "Exactly one file must be uploaded."


def test_upload_file_dependency_override(client: TestClient) -> None:
    """Test replacing UploadService via FastAPI dependency_overrides."""

    mock_service = MagicMock()
    mock_service.save_upload = AsyncMock(
        return_value=UploadMetadata(
            upload_id="12345678-1234-5678-1234-567812345678",
            original_filename="mock.nii",
            stored_filename="12345678-1234-5678-1234-567812345678.nii",
            file_size_bytes=100,
            upload_directory="/tmp/mock",
        )
    )

    app.dependency_overrides[get_upload_service] = lambda: mock_service
    try:
        files = {"file": ("mock.nii", BytesIO(b"mock"), "application/octet-stream")}
        response = client.post("/api/v1/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["original_filename"] == "mock.nii"
    finally:
        app.dependency_overrides.clear()
