"""Upload endpoint for medical imaging files.

This module accepts a single medical imaging file, delegates validation and
storage to the injected ``UploadService``, and returns upload metadata. It
does not perform preprocessing, inference, reporting, or visualization.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ....dependencies import get_upload_service
from ....schemas.responses import UploadResponse
from ....services.upload_service import UploadService


router = APIRouter()


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a medical imaging file",
)
async def upload_file(
    files: Annotated[list[UploadFile], File(alias="file")],
    upload_service: Annotated[UploadService, Depends(get_upload_service)],
) -> UploadResponse:
    """Store a single supported medical imaging file for future inference.

    Args:
        files: Uploaded medical imaging files provided as multipart form data.
        upload_service: Injected upload service instance.

    Returns:
        Upload metadata including identifiers, filenames, file size, storage
        location, timestamp, and status.

    Raises:
        HTTPException: If the uploaded file is missing a valid filename, uses
        an unsupported extension, is empty, or cannot be stored.
    """

    file = _get_single_upload(files)

    try:
        metadata = await upload_service.save_upload(file)
    except ValueError as exc:
        detail = str(exc)
        status_code = status.HTTP_400_BAD_REQUEST
        if detail == "Unsupported file extension.":
            status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE

        raise HTTPException(
            status_code=status_code,
            detail=detail,
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Uploaded file could not be stored.",
        ) from exc

    return UploadResponse(
        upload_id=metadata.upload_id,
        original_filename=metadata.original_filename,
        stored_filename=metadata.stored_filename,
        file_size_bytes=metadata.file_size_bytes,
        upload_directory=str(metadata.upload_directory),
        upload_timestamp=datetime.now(timezone.utc),
        status="uploaded",
    )


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
