"""Inference request endpoint for the BrainTumorAI API.

This module exposes the API boundary for future brain tumor segmentation
inference. It validates that an uploaded file exists and returns a structured
queued response without running preprocessing, model inference, postprocessing,
reporting, or visualization.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ....core.config import settings


router = APIRouter()


class InferenceRequest(BaseModel):
    """Request body for creating an inference request.

    Attributes:
        upload_id: Identifier returned by the upload endpoint.
    """

    upload_id: str = Field(
        ...,
        min_length=1,
        description="Identifier returned by the upload endpoint.",
    )


class InferenceResponse(BaseModel):
    """Response body for a queued inference request.

    Attributes:
        request_id: Unique identifier for the inference request.
        upload_id: Upload identifier associated with the request.
        status: Current request status.
        message: Human-readable request status message.
        created_at: UTC ISO 8601 timestamp for request creation.
    """

    request_id: str
    upload_id: str
    status: str
    message: str
    created_at: str


@router.post(
    "",
    response_model=InferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_inference_request(
    request: InferenceRequest,
) -> InferenceResponse:
    """Create a queued inference request for an uploaded medical image.

    Args:
        request: Inference request payload containing an upload identifier.

    Returns:
        Structured metadata for the queued inference request.

    Raises:
        HTTPException: If the upload identifier is blank, invalid, or does not
        correspond to an uploaded file.
    """

    upload_id = _normalize_upload_id(request.upload_id)
    if not _uploaded_file_exists(upload_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded file was not found.",
        )

    return InferenceResponse(
        request_id=str(uuid4()),
        upload_id=upload_id,
        status="queued",
        message="Inference pipeline is ready for implementation.",
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def _normalize_upload_id(upload_id: str) -> str:
    """Validate and normalize an upload identifier.

    Args:
        upload_id: Raw upload identifier from the request body.

    Returns:
        Canonical string representation of the upload UUID.

    Raises:
        HTTPException: If the identifier is blank or not a valid UUID.
    """

    candidate = upload_id.strip()
    if not candidate:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload_id must be provided.",
        )

    try:
        return str(UUID(candidate))
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="upload_id must be a valid UUID.",
        ) from exc


def _uploaded_file_exists(upload_id: str) -> bool:
    """Return whether a stored upload exists for an upload identifier.

    Args:
        upload_id: Canonical upload UUID string.

    Returns:
        ``True`` when a matching uploaded file exists, otherwise ``False``.
    """

    upload_directory = Path(settings.uploads_directory)
    return any(
        path.is_file()
        for path in upload_directory.glob(f"{upload_id}.*")
    )
