"""Unit tests for backend UploadService."""

from io import BytesIO
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from fastapi import UploadFile

from backend.services.upload_service import SUPPORTED_EXTENSIONS, UploadService


@pytest.mark.anyio
async def test_save_upload_valid_file(upload_service: UploadService) -> None:
    """Test saving a valid medical imaging file."""

    file_content = b"header_data_dummy_content"
    upload_file = UploadFile(
        filename="brain_scan.nii.gz",
        file=BytesIO(file_content),
    )

    metadata = await upload_service.save_upload(upload_file)

    assert isinstance(metadata.upload_id, UUID)
    assert metadata.original_filename == "brain_scan.nii.gz"
    assert metadata.stored_filename == f"{metadata.upload_id}.nii.gz"
    assert metadata.file_size_bytes == len(file_content)
    assert (upload_service.upload_directory / metadata.stored_filename).exists()


@pytest.mark.anyio
@pytest.mark.parametrize("ext", [".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"])
async def test_supported_extensions(upload_service: UploadService, ext: str) -> None:
    """Test that all supported medical image extensions are accepted."""

    upload_file = UploadFile(
        filename=f"scan{ext}",
        file=BytesIO(b"data"),
    )

    metadata = await upload_service.save_upload(upload_file)
    assert metadata.stored_filename.endswith(ext)


@pytest.mark.anyio
async def test_save_upload_unsupported_extension(upload_service: UploadService) -> None:
    """Test that unsupported file extensions raise ValueError."""

    upload_file = UploadFile(
        filename="scan.png",
        file=BytesIO(b"png_data"),
    )

    with pytest.raises(ValueError, match="Unsupported file extension"):
        await upload_service.save_upload(upload_file)


@pytest.mark.anyio
async def test_save_upload_empty_file(upload_service: UploadService) -> None:
    """Test that empty files raise ValueError and clean up temporary storage."""

    upload_file = UploadFile(
        filename="empty.nii",
        file=BytesIO(b""),
    )

    with pytest.raises(ValueError, match="Uploaded file is empty"):
        await upload_service.save_upload(upload_file)


@pytest.mark.anyio
async def test_save_upload_missing_filename(upload_service: UploadService) -> None:
    """Test that a missing filename raises ValueError."""

    upload_file = UploadFile(
        filename="",
        file=BytesIO(b"content"),
    )

    with pytest.raises(ValueError, match="Uploaded file must include a filename"):
        await upload_service.save_upload(upload_file)


@pytest.mark.anyio
async def test_save_upload_path_traversal_filename(upload_service: UploadService) -> None:
    """Test that path traversal characters in filenames are sanitized."""

    upload_file = UploadFile(
        filename="../../../etc/passwd.nii",
        file=BytesIO(b"content"),
    )

    metadata = await upload_service.save_upload(upload_file)
    assert metadata.original_filename == "passwd.nii"


def test_get_uploaded_file_success(upload_service: UploadService) -> None:
    """Test looking up an existing uploaded file by UUID."""

    upload_id = uuid4()
    file_path = upload_service.upload_directory / f"{upload_id}.nii.gz"
    file_path.write_bytes(b"data")

    retrieved = upload_service.get_uploaded_file(upload_id)
    assert retrieved == file_path

    # Test string UUID
    retrieved_str = upload_service.get_uploaded_file(str(upload_id))
    assert retrieved_str == file_path


def test_get_uploaded_file_not_found(upload_service: UploadService) -> None:
    """Test looking up a non-existent upload ID raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError, match="Uploaded file was not found"):
        upload_service.get_uploaded_file(uuid4())


def test_get_uploaded_file_invalid_uuid(upload_service: UploadService) -> None:
    """Test looking up an invalid UUID string raises ValueError."""

    with pytest.raises(ValueError, match="upload_id must be provided"):
        upload_service.get_uploaded_file("")

    with pytest.raises(ValueError):
        upload_service.get_uploaded_file("invalid-uuid-string")


def test_delete_upload_success(upload_service: UploadService) -> None:
    """Test deleting an uploaded file by ID."""

    upload_id = uuid4()
    file_path = upload_service.upload_directory / f"{upload_id}.nii"
    file_path.write_bytes(b"data")

    assert upload_service.delete_upload(upload_id) is True
    assert not file_path.exists()


def test_delete_upload_not_found(upload_service: UploadService) -> None:
    """Test deleting a non-existent upload raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        upload_service.delete_upload(uuid4())
