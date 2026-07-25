"""Domain exception hierarchy for the BrainTumorAI AI subsystem.

This module defines custom exceptions raised during AI pipeline execution,
model inference, postprocessing, 3D reconstruction, and report generation.
"""


class AIProcessingError(Exception):
    """Base exception for all AI subsystem errors."""

    def __init__(self, message: str, detail: str | None = None) -> None:
        """Initialize the AI processing error.

        Args:
            message: High-level error summary.
            detail: Optional detailed context or error log.
        """

        super().__init__(message)
        self.message = message
        self.detail = detail


class PreprocessingError(AIProcessingError):
    """Raised when medical image preprocessing or format normalization fails."""


class ModelInferenceError(AIProcessingError):
    """Raised when an AI model adapter (nnUNet / SynthSeg) fails to execute."""


class PostProcessingError(AIProcessingError):
    """Raised when mask fusion or volumetric analysis calculation fails."""


class VisualizationError(AIProcessingError):
    """Raised when 3D surface mesh extraction or Plotly HTML rendering fails."""


class ReportGenerationError(AIProcessingError):
    """Raised when clinical report aggregation or persistence fails."""
