"""Abstract base adapter for AI model integrations."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ai.contracts import NormalizedScan


class BaseModelAdapter(ABC):
    """Abstract Base Class for wrapping pretrained segmentation model inference engines."""

    def __init__(self, model_directory: Path | str) -> None:
        """Initialize the model adapter.

        Args:
            model_directory: Path to pretrained model artifacts/checkpoints.
        """

        self.model_directory = Path(model_directory)

    @abstractmethod
    def predict(
        self,
        input_scan: NormalizedScan,
        output_mask_path: Path,
    ) -> Any:
        """Run model inference on a normalized scan and save output mask.

        Args:
            input_scan: Normalized scan contract object.
            output_mask_path: Target path to write output NIfTI segmentation mask.

        Returns:
            Segmentation result contract object.

        Raises:
            ModelInferenceError: If inference fails or model output is missing.
        """
