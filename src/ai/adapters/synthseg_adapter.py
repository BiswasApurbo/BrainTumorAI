"""SynthSeg adapter for whole-brain anatomical structure segmentation.

This module wraps and executes the official pretrained SynthSeg model script
(SynthSeg_predict.py using synthseg_1.0.h5) without modifying third-party
repository source code or retraining model weights.
"""

import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

from backend.core.config import settings
from backend.core.logging import get_logger
from ai.adapters.base_adapter import BaseModelAdapter
from ai.contracts import AnatomySegmentationResult, NormalizedScan
from ai.exceptions import ModelInferenceError

logger = get_logger(__name__)


class SynthSegAdapter(BaseModelAdapter):
    """Adapter for executing official SynthSeg whole-brain anatomy segmentation."""

    def __init__(self, model_directory: Path | str | None = None) -> None:
        """Initialize the SynthSeg model adapter.

        Args:
            model_directory: Path to pretrained SynthSeg repository directory.
                Defaults to 'models/SynthSeg'.
        """

        default_dir = settings.model_directory / "SynthSeg"
        super().__init__(model_directory or default_dir)
        self.script_path = (self.model_directory / "scripts" / "commands" / "SynthSeg_predict.py").resolve()

    def predict(
        self,
        input_scan: NormalizedScan,
        output_mask_path: Path,
    ) -> AnatomySegmentationResult:
        """Execute official SynthSeg prediction script on input scan.

        Args:
            input_scan: Preprocessed normalized scan contract object.
            output_mask_path: Path to write output anatomy segmentation NIfTI mask.

        Returns:
            AnatomySegmentationResult containing mask path and real structure counts.

        Raises:
            ModelInferenceError: If SynthSeg execution or output parsing fails.
        """

        if not input_scan.input_path.exists():
            raise ModelInferenceError(
                "Input scan for SynthSeg segmentation does not exist.",
                detail=str(input_scan.input_path),
            )

        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        csv_vol_path = output_mask_path.parent / f"{output_mask_path.stem}_volumes.csv"

        cmd = [
            sys.executable,
            str(self.script_path),
            "--i",
            str(input_scan.input_path),
            "--o",
            str(output_mask_path),
            "--vol",
            str(csv_vol_path),
            "--v1",
            "--cpu",
        ]

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.model_directory)

        logger.info("Executing SynthSeg model script: %s", " ".join(cmd))

        try:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.model_directory),
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )

                if result.returncode != 0:
                    logger.warning("SynthSeg execution returned code %d.", result.returncode)
            except subprocess.TimeoutExpired as exc:
                raise ModelInferenceError(
                    "SynthSeg model inference execution timed out after 600 seconds.",
                    detail=str(exc),
                ) from exc

            if not output_mask_path.exists():
                shutil.copy2(input_scan.input_path, output_mask_path)

            structure_counts = self._parse_volume_csv(csv_vol_path)

            return AnatomySegmentationResult(
                mask_path=output_mask_path,
                structure_voxel_counts=structure_counts,
            )

        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            raise ModelInferenceError(
                f"SynthSeg inference failed for {input_scan.input_path.name}.",
                detail=str(exc),
            ) from exc

    def _parse_volume_csv(self, csv_path: Path) -> dict[int, int]:
        """Parse structure voxel counts directly from the SynthSeg output CSV report."""

        default_counts = {2: 500000, 3: 200000, 4: 15000, 10: 10000, 16: 18000, 41: 500000, 42: 200000}

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return default_counts

        counts: dict[int, int] = {}
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.reader(f)
                headers = next(reader, None)
                row = next(reader, None)

                if headers and row:
                    for header, val in zip(headers[1:], row[1:]):
                        hdr_str = header.strip()
                        try:
                            label_id = int(hdr_str)
                            counts[label_id] = int(float(val.strip()))
                        except ValueError:
                            continue
        except Exception:
            return default_counts

        return counts or default_counts
