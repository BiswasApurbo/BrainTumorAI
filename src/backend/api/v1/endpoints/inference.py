"""Inference request endpoint for the BrainTumorAI API.

This module exposes the API boundary for brain tumor segmentation inference.
It delegates validation and job creation to the injected ``InferenceService``
and returns a structured queued response.
"""

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.core.logging import get_logger
from ai.exceptions import AIProcessingError
from ....dependencies import get_inference_service
from ....services.inference_service import InferenceService

logger = get_logger(__name__)

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


class InferenceFromPathRequest(BaseModel):
    """Request body for creating an inference request from a filesystem path.

    Supports both a single NIfTI file path and a BraTS patient directory
    containing FLAIR, T1, T1CE, and T2 modalities.

    Attributes:
        input_path: Absolute filesystem path to a NIfTI file or BraTS patient directory.
    """

    input_path: str = Field(
        ...,
        min_length=1,
        description="Absolute filesystem path to a NIfTI file or BraTS patient directory.",
    )


class InferenceResponse(BaseModel):
    """Response body for a queued or completed inference request.

    Attributes:
        request_id: Unique identifier for the inference request.
        upload_id: Upload identifier associated with the request.
        status: Current request status.
        message: Human-readable request status message.
        created_at: UTC ISO 8601 timestamp for request creation.
        report_id: Optional unique identifier for the generated diagnostic report.
        visualization_path: Optional path to the 3D Plotly HTML visualization.
        tumor_mask_path: Optional path to the tumor segmentation mask.
        anatomy_mask_path: Optional path to the anatomy segmentation mask.
        volumetric_analysis: Optional quantitative volumetric measurements dictionary.
    """

    request_id: str
    upload_id: str
    status: str
    message: str
    created_at: str
    report_id: str | None = None
    visualization_path: str | None = None
    tumor_mask_path: str | None = None
    anatomy_mask_path: str | None = None
    volumetric_analysis: dict[str, Any] | None = None


@router.post(
    "",
    response_model=InferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_inference_request(
    request: InferenceRequest,
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InferenceResponse:
    """Create and execute an inference request for an uploaded medical image.

    Args:
        request: Inference request payload containing an upload identifier.
        inference_service: Injected inference orchestration service instance.

    Returns:
        Structured metadata for the completed or queued inference request.

    Raises:
        HTTPException: If the upload identifier is blank, invalid, or does not
        correspond to an uploaded file.
    """

    logger.info("Received inference request for upload_id: %s", request.upload_id)

    try:
        metadata = inference_service.create_inference_request(request.upload_id)
    except FileNotFoundError as exc:
        logger.warning("Upload file not found for upload_id %s: %s", request.upload_id, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Uploaded file was not found.",
        ) from exc
    except ValueError as exc:
        logger.warning("Invalid inference request for upload_id %s: %s", request.upload_id, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except AIProcessingError as exc:
        logger.error("AI pipeline execution failed for upload_id %s: %s", request.upload_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI pipeline execution failed.",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during inference request for upload_id %s: %s", request.upload_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from exc

    return _build_response(metadata, message="Inference pipeline executed successfully.")


@router.post(
    "/from-path",
    response_model=InferenceResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Create inference request from a filesystem path",
)
async def create_inference_request_from_path(
    request: InferenceFromPathRequest,
    inference_service: Annotated[InferenceService, Depends(get_inference_service)],
) -> InferenceResponse:
    """Create and execute an inference request from a filesystem path.

    Accepts either a single NIfTI file path or a BraTS patient directory
    containing FLAIR, T1, T1CE, and T2 modalities.

    Args:
        request: Request payload containing a filesystem path.
        inference_service: Injected inference orchestration service instance.

    Returns:
        Structured metadata for the completed or queued inference request.

    Raises:
        HTTPException: If the path does not exist, is unreadable, the directory
            is empty, or required BraTS modalities are missing.
    """

    input_path = Path(request.input_path)
    logger.info("Received inference request from path: %s", input_path)

    try:
        metadata = inference_service.create_inference_request_from_path(input_path)
    except FileNotFoundError as exc:
        logger.warning("Input path not found: %s: %s", input_path, exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Input path does not exist: {input_path}",
        ) from exc
    except AIProcessingError as exc:
        logger.error("Validation or pipeline failure for path %s: %s", input_path, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error for path %s: %s", input_path, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected internal server error occurred.",
        ) from exc

    return _build_response(metadata, message="Inference pipeline executed successfully.")


def _build_response(metadata: Any, *, message: str) -> InferenceResponse:
    """Build a standardized InferenceResponse from service metadata.

    Args:
        metadata: InferenceMetadata returned by the inference service.
        message: Human-readable status message.

    Returns:
        Formatted InferenceResponse.
    """

    status_value = (
        metadata.status.value
        if hasattr(metadata.status, "value")
        else str(metadata.status)
    )

    return InferenceResponse(
        request_id=str(metadata.request_id),
        upload_id=str(metadata.upload_id),
        status=status_value,
        message=message,
        created_at=metadata.created_at.isoformat(),
        report_id=str(metadata.report_id) if metadata.report_id else None,
        visualization_path=str(metadata.visualization_path) if metadata.visualization_path else None,
        tumor_mask_path=str(metadata.tumor_mask_path) if metadata.tumor_mask_path else None,
        anatomy_mask_path=str(metadata.anatomy_mask_path) if metadata.anatomy_mask_path else None,
        volumetric_analysis=metadata.volumetric_analysis,
    )

