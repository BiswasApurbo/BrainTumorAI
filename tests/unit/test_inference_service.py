"""Unit tests for backend InferenceService."""

from pathlib import Path
from uuid import UUID, uuid4

import pytest

from backend.services.inference_service import (
    InferenceJob,
    InferenceService,
    PipelineStatus,
    PipelineStep,
)
from backend.services.upload_service import UploadService


class DummyPipelineStep:
    """Dummy pipeline collaborator implementing PipelineStep protocol."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self.called_count = 0

    def run(self, job: InferenceJob) -> None:
        self.called_count += 1


class FailingPipelineStep:
    """Failing pipeline collaborator for testing step error handling."""

    def run(self, job: InferenceJob) -> None:
        raise RuntimeError("Step execution failed")


def test_create_inference_request_success(
    inference_service: InferenceService,
    upload_service: UploadService,
) -> None:
    """Test creating an inference request for an existing uploaded file."""

    upload_id = uuid4()
    file_path = upload_service.upload_directory / f"{upload_id}.nii"
    file_path.write_bytes(b"data")

    metadata = inference_service.create_inference_request(upload_id)

    assert isinstance(metadata.request_id, UUID)
    assert metadata.upload_id == upload_id
    assert metadata.status == PipelineStatus.QUEUED
    assert metadata.input_path == file_path


def test_create_inference_request_file_not_found(
    inference_service: InferenceService,
) -> None:
    """Test creating an inference request for a missing upload raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="Uploaded file was not found"):
        inference_service.create_inference_request(uuid4())


def test_create_inference_request_invalid_uuid(
    inference_service: InferenceService,
) -> None:
    """Test creating an inference request with invalid UUID raises ValueError."""

    with pytest.raises(ValueError, match="upload_id must be provided"):
        inference_service.create_inference_request("")


def test_run_pipeline_success(
    upload_service: UploadService,
) -> None:
    """Test running a complete inference pipeline with injected collaborators."""

    prep_step = DummyPipelineStep("prep")
    seg_step = DummyPipelineStep("seg")

    service = InferenceService(
        upload_service=upload_service,
        preprocessing_service=prep_step,
        segmentation_service=seg_step,
    )

    upload_id = uuid4()
    (upload_service.upload_directory / f"{upload_id}.nii").write_bytes(b"data")

    metadata = service.create_inference_request(upload_id)
    final_status = service.run_pipeline(metadata.request_id)

    assert final_status == PipelineStatus.COMPLETED
    assert prep_step.called_count == 1
    assert seg_step.called_count == 1


def test_run_pipeline_step_failure(
    upload_service: UploadService,
) -> None:
    """Test that a failing pipeline step sets status to FAILED and records error message."""

    failing_step = FailingPipelineStep()
    service = InferenceService(
        upload_service=upload_service,
        preprocessing_service=failing_step,
    )

    upload_id = uuid4()
    (upload_service.upload_directory / f"{upload_id}.nii").write_bytes(b"data")

    metadata = service.create_inference_request(upload_id)

    with pytest.raises(RuntimeError, match="Step execution failed"):
        service.run_pipeline(metadata.request_id)

    assert service.get_status(metadata.request_id) == PipelineStatus.FAILED


def test_get_status_success(
    inference_service: InferenceService,
    upload_service: UploadService,
) -> None:
    """Test retrieving current status for a queued request."""

    upload_id = uuid4()
    (upload_service.upload_directory / f"{upload_id}.nii").write_bytes(b"data")

    metadata = inference_service.create_inference_request(upload_id)
    status = inference_service.get_status(metadata.request_id)
    assert status == PipelineStatus.QUEUED


def test_get_status_not_found(inference_service: InferenceService) -> None:
    """Test retrieving status for a non-existent request raises KeyError."""

    with pytest.raises(KeyError, match="Inference request was not found"):
        inference_service.get_status(uuid4())


def test_cancel_request_success(
    inference_service: InferenceService,
    upload_service: UploadService,
) -> None:
    """Test cancelling a queued inference request."""

    upload_id = uuid4()
    (upload_service.upload_directory / f"{upload_id}.nii").write_bytes(b"data")

    metadata = inference_service.create_inference_request(upload_id)
    new_status = inference_service.cancel_request(metadata.request_id)

    assert new_status == PipelineStatus.CANCELLED
    assert inference_service.get_status(metadata.request_id) == PipelineStatus.CANCELLED


def test_cancel_request_not_queued(
    inference_service: InferenceService,
    upload_service: UploadService,
) -> None:
    """Test cancelling an already completed request raises ValueError."""

    upload_id = uuid4()
    (upload_service.upload_directory / f"{upload_id}.nii").write_bytes(b"data")

    metadata = inference_service.create_inference_request(upload_id)
    inference_service.run_pipeline(metadata.request_id)

    with pytest.raises(ValueError, match="Only queued inference requests can be cancelled"):
        inference_service.cancel_request(metadata.request_id)


def test_create_inference_request_from_path_brats_directory(
    tmp_path: Path,
) -> None:
    """Test creating inference request from a valid BraTS patient directory."""

    patient_dir = tmp_path / "BraTS20_Training_042"
    patient_dir.mkdir()

    for suffix in ("flair", "t1", "t1ce", "t2", "seg"):
        (patient_dir / f"BraTS20_Training_042_{suffix}.nii").write_bytes(b"data")

    service = InferenceService()
    metadata = service.create_inference_request_from_path(patient_dir)

    assert metadata.input_path == patient_dir
    assert metadata.status == PipelineStatus.QUEUED


def test_create_inference_request_from_path_single_file(
    tmp_path: Path,
) -> None:
    """Test creating inference request from a single NIfTI file."""

    nifti_file = tmp_path / "brain.nii.gz"
    nifti_file.write_bytes(b"data")

    service = InferenceService()
    metadata = service.create_inference_request_from_path(nifti_file)

    assert metadata.input_path == nifti_file
    assert metadata.status == PipelineStatus.QUEUED


def test_create_inference_request_from_path_missing_modality(
    tmp_path: Path,
) -> None:
    """Test that a BraTS directory missing modalities raises AIProcessingError."""

    from ai.exceptions import AIProcessingError

    patient_dir = tmp_path / "BraTS20_Training_099"
    patient_dir.mkdir()

    # Only create flair and t1 — missing t1ce and t2
    for suffix in ("flair", "t1"):
        (patient_dir / f"BraTS20_Training_099_{suffix}.nii").write_bytes(b"data")

    service = InferenceService()

    with pytest.raises(AIProcessingError, match="missing required modalities"):
        service.create_inference_request_from_path(patient_dir)


def test_create_inference_request_from_path_empty_directory(
    tmp_path: Path,
) -> None:
    """Test that an empty BraTS directory raises AIProcessingError."""

    from ai.exceptions import AIProcessingError

    empty_dir = tmp_path / "empty_patient"
    empty_dir.mkdir()

    service = InferenceService()

    with pytest.raises(AIProcessingError, match="empty"):
        service.create_inference_request_from_path(empty_dir)


def test_create_inference_request_from_path_nonexistent(
    tmp_path: Path,
) -> None:
    """Test that a nonexistent path raises FileNotFoundError."""

    service = InferenceService()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        service.create_inference_request_from_path(tmp_path / "ghost")

