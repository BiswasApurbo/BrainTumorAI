"""Upload endpoint for medical imaging files.

This module accepts a single medical imaging file, validates its extension,
stores it in the configured uploads directory, and returns upload metadata. It
does not perform preprocessing, inference, reporting, or visualization.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ....core.config import settings


router = APIRouter()

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".nii", ".nii.gz", ".mha", ".mhd", ".nrrd"}
)
CHUNK_SIZE_BYTES: int = 1024 * 1024
UploadResponse = dict[str, str | int]


@router.post("", status_code=status.HTTP_201_CREATED)
async def upload_file(
    files: Annotated[list[UploadFile], File(alias="file")],
) -> UploadResponse:
    """Store a single supported medical imaging file for future inference.

    Args:
        files: Uploaded medical imaging files provided as multipart form data.

    Returns:
        A JSON-serializable payload containing upload identifiers, filenames,
        file size, storage location, timestamp, and status.

    Raises:
        HTTPException: If the uploaded file is missing a valid filename, uses
        an unsupported extension, is empty, or cannot be stored.
    """

    file = _get_single_upload(files)
    original_filename = _get_safe_original_filename(file)
    extension = _get_supported_extension(original_filename)
    upload_id = str(uuid4())
    stored_filename = f"{upload_id}{extension}"
    upload_directory = Path(settings.uploads_directory)
    destination = upload_directory / stored_filename

    upload_directory.mkdir(parents=True, exist_ok=True)

    try:
        file_size_bytes = await _store_upload(file, destination)
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uploaded file could not be stored.",
        ) from exc

    if file_size_bytes == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    return {
        "upload_id": upload_id,
        "original_filename": original_filename,
        "stored_filename": stored_filename,
        "file_size_bytes": file_size_bytes,
        "upload_directory": str(upload_directory),
        "upload_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "uploaded",
    }


def _get_single_upload(files: list[UploadFile]) -> UploadFile:
    """Return the uploaded file when exactly one file is provided.

    Args:
        files: Uploaded files received from the multipart ``file`` field.

    Returns:
        The single uploaded file.

    Raises:
        HTTPException: If no file or multiple files are provided.
    """

    if len(files) != 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Exactly one file must be uploaded.",
        )

    return files[0]


def _get_safe_original_filename(file: UploadFile) -> str:
    """Return a sanitized original filename from an uploaded file.

    Args:
        file: Uploaded file object received by FastAPI.

    Returns:
        Filename without any client-supplied directory components.

    Raises:
        HTTPException: If the uploaded file does not include a filename.
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a filename.",
        )

    original_filename = Path(file.filename.replace("\\", "/")).name
    if not original_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must include a valid filename.",
        )

    return original_filename


def _get_supported_extension(filename: str) -> str:
    """Return the supported extension for a filename.

    Args:
        filename: Original uploaded filename.

    Returns:
        The normalized supported extension, preserving compound extensions.

    Raises:
        HTTPException: If the filename extension is unsupported.
    """

    normalized_filename = filename.lower()
    for extension in sorted(SUPPORTED_EXTENSIONS, key=len, reverse=True):
        if normalized_filename.endswith(extension):
            return extension

    raise HTTPException(
        status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        detail="Unsupported file extension.",
    )


async def _store_upload(file: UploadFile, destination: Path) -> int:
    """Persist an uploaded file to disk and return its byte size.

    Args:
        file: Uploaded file object received by FastAPI.
        destination: Filesystem path where the file should be stored.

    Returns:
        Number of bytes written to disk.
    """

    file_size_bytes = 0

    with destination.open("wb") as stored_file:
        while chunk := await file.read(CHUNK_SIZE_BYTES):
            file_size_bytes += len(chunk)
            stored_file.write(chunk)

    return file_size_bytes
