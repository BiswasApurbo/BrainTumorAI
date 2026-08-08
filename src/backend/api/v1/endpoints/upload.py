"""Upload endpoint for medical imaging files.

This module accepts a single medical imaging file, delegates validation and
storage to the injected ``UploadService``, and returns upload metadata. It
does not perform preprocessing, inference, reporting, or visualization.
"""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

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
    flair: UploadFile = File(...),
    t1: UploadFile = File(...),
    t1ce: UploadFile = File(...),
    t2: UploadFile = File(...),
    mode: str = Form("prediction"),
    seg: UploadFile | None = File(None),
    upload_service: UploadService = Depends(get_upload_service),
) -> UploadResponse:
    """Store a single supported medical imaging file for future inference.

    Args:
        flair: FLAIR modality file.
        t1: T1 modality file.
        t1ce: T1CE modality file.
        t2: T2 modality file.
        upload_service: Injected upload service instance.

    Returns:
        Upload metadata including identifiers, filenames, file size, storage
        location, timestamp, and status.

    Raises:
        HTTPException: If the uploaded file is missing a valid filename, uses
        an unsupported extension, is empty, or cannot be stored.
    """

    if mode == "ground_truth" and not seg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="SEG file is required in Ground Truth mode.",
        )

    try:
        metadata = await upload_service.save_upload(
            flair=flair,
            t1=t1,
            t1ce=t1ce,
            t2=t2,
            seg=seg,
        )
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
        upload_directory=str(metadata.upload_directory),
        upload_timestamp=datetime.now(timezone.utc),
        status="uploaded",
    )



