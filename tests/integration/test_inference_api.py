"""Integration tests for POST /api/v1/inference endpoint."""

from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient

from backend.dependencies import get_inference_service
from backend.main import app
from backend.services.inference_service import InferenceMetadata, PipelineStatus


def test_create_inference_request_success(client: TestClient) -> None:
    """Test creating an inference request after uploading a file."""

    # First upload a valid file
    upload_res = client.post(
        "/api/v1/upload",
        files={"file": ("brain.nii", BytesIO(b"data"), "application/octet-stream")},
    )
    assert upload_res.status_code == 201
    upload_id = upload_res.json()["upload_id"]

    # Now request inference
    response = client.post(
        "/api/v1/inference",
        json={"upload_id": upload_id},
    )

    assert response.status_code == 202
    data = response.json()
    assert "request_id" in data
    assert data["upload_id"] == upload_id
    assert data["status"] in ("queued", "completed")
    assert "created_at" in data


def test_create_inference_request_upload_not_found(client: TestClient) -> None:
    """Test requesting inference for a non-existent upload returns 404 Not Found."""

    fake_upload_id = str(uuid4())
    response = client.post(
        "/api/v1/inference",
        json={"upload_id": fake_upload_id},
    )

    assert response.status_code == 404
    data = response.json()
    assert data["detail"] == "Uploaded file was not found."


def test_create_inference_request_invalid_uuid(client: TestClient) -> None:
    """Test requesting inference with invalid UUID returns 400 Bad Request or 422."""

    response = client.post(
        "/api/v1/inference",
        json={"upload_id": "not-a-uuid"},
    )

    assert response.status_code in (400, 422)


def test_create_inference_request_dependency_override(client: TestClient) -> None:
    """Test replacing InferenceService via FastAPI dependency_overrides."""

    class MockInferenceService:
        def create_inference_request(self, upload_id: str) -> InferenceMetadata:
            return InferenceMetadata(
                request_id=uuid4(),
                upload_id=uuid4(),
                status=PipelineStatus.QUEUED,
                input_path="mock/path",
                created_at=datetime.now(timezone.utc),
            )

    app.dependency_overrides[get_inference_service] = lambda: MockInferenceService()
    try:
        response = client.post(
            "/api/v1/inference",
            json={"upload_id": str(uuid4())},
        )
        assert response.status_code == 202
    finally:
        app.dependency_overrides.clear()


def test_create_inference_request_from_path_brats_directory(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Test creating inference from a BraTS patient directory via /from-path."""

    patient_dir = tmp_path / "BraTS20_Training_777"
    patient_dir.mkdir()

    for suffix in ("flair", "t1", "t1ce", "t2"):
        (patient_dir / f"BraTS20_Training_777_{suffix}.nii").write_bytes(b"data")

    response = client.post(
        "/api/v1/inference/from-path",
        json={"input_path": str(patient_dir)},
    )

    assert response.status_code == 202
    data = response.json()
    assert "request_id" in data
    assert data["status"] in ("queued", "completed")


def test_create_inference_request_from_path_not_found(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Test /from-path returns 404 for a nonexistent path."""

    response = client.post(
        "/api/v1/inference/from-path",
        json={"input_path": str(tmp_path / "ghost")},
    )

    assert response.status_code == 404


def test_create_inference_request_from_path_empty_directory(
    client: TestClient,
    tmp_path: Path,
) -> None:
    """Test /from-path returns 400 for an empty directory."""

    empty_dir = tmp_path / "empty_patient"
    empty_dir.mkdir()

    response = client.post(
        "/api/v1/inference/from-path",
        json={"input_path": str(empty_dir)},
    )

    assert response.status_code == 400

