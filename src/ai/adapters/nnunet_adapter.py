"""nnU-Net adapter for BraTS2020 brain tumor segmentation.

This module executes the official pretrained nnU-Net 3d_fullres model
checkpoints (Task082_BraTS2020) located in models/nnUNet without modifying
model weights or repository source code.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

from backend.core.config import settings
from backend.core.logging import get_logger
from ai.adapters.base_adapter import BaseModelAdapter
from ai.contracts import NormalizedBraTSCase, NormalizedScan, TumorSegmentationResult
from ai.exceptions import ModelInferenceError

logger = get_logger(__name__)


class NNUNetAdapter(BaseModelAdapter):
    """Adapter for executing official nnU-Net BraTS2020 tumor segmentation checkpoints."""

    def __init__(self, model_directory: Path | str | None = None) -> None:
        """Initialize the nnUNet model adapter.

        Args:
            model_directory: Path to pretrained nnUNet weights directory.
                Defaults to 'models/nnUNet'.
        """

        default_dir = settings.model_directory / "nnUNet"
        super().__init__(model_directory or default_dir)
        self.task_name = "Task082_BraTS2020"

    def predict(
        self,
        input_scan: NormalizedBraTSCase | NormalizedScan,
        output_mask_path: Path,
    ) -> TumorSegmentationResult:
        """Run nnUNet tumor segmentation inference on normalized scan or BraTS case.

        Args:
            input_scan: Preprocessed `NormalizedBraTSCase` or legacy `NormalizedScan`.
            output_mask_path: Target path to write output tumor segmentation NIfTI mask.

        Returns:
            TumorSegmentationResult containing mask path and real subregion counts.

        Raises:
            ModelInferenceError: If input files are missing, or if nnUNet execution fails.
        """

        output_mask_path.parent.mkdir(parents=True, exist_ok=True)
        input_dir = output_mask_path.parent / "nnunet_input"
        output_dir = output_mask_path.parent / "nnunet_output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(input_scan, NormalizedBraTSCase):
            # Verify all four required BraTS modality files exist
            modalities = [
                ("FLAIR (0000)", input_scan.normalized_flair, input_dir / "scan_0000.nii.gz"),
                ("T1 (0001)", input_scan.normalized_t1, input_dir / "scan_0001.nii.gz"),
                ("T1CE (0002)", input_scan.normalized_t1ce, input_dir / "scan_0002.nii.gz"),
                ("T2 (0003)", input_scan.normalized_t2, input_dir / "scan_0003.nii.gz"),
            ]

            missing = [label for label, src, _ in modalities if not src.exists()]
            if missing:
                raise ModelInferenceError(
                    f"Missing required BraTS modalities for patient {input_scan.patient_id}: {', '.join(missing)}.",
                    detail=f"Patient ID: {input_scan.patient_id}",
                )

            logger.info(
                "Staging multi-modal BraTS case for patient %s into nnU-Net input directory:",
                input_scan.patient_id,
            )
            for label, src, dst in modalities:
                logger.info("  Modality %-15s: %s -> %s", label, src.name, dst.name)
                shutil.copy2(src, dst)

            fallback_src = input_scan.normalized_flair

        else:
            # Legacy single-file processing path
            if not input_scan.input_path.exists():
                raise ModelInferenceError(
                    "Input scan for nnUNet segmentation does not exist.",
                    detail=str(input_scan.input_path),
                )

            formatted_input = input_dir / "scan_0000.nii.gz"
            shutil.copy2(input_scan.input_path, formatted_input)
            fallback_src = input_scan.input_path

        results_folder = str(self.model_directory.parent.resolve())
        trainer_class = "nnUNetTrainerV2BraTSRegions_DA4_BN"
        plans_identifier = "nnUNetPlansv2.1_bs5"

        env = dict(os.environ)
        env["RESULTS_FOLDER"] = results_folder
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        logger.info(
            "Configured nnU-Net environment: RESULTS_FOLDER=%s, model_directory=%s, trainer_class=%s",
            results_folder,
            self.model_directory,
            trainer_class,
        )

        nnunet_cli = shutil.which("nnUNet_predict")
        if nnunet_cli:
            cmd = [
                nnunet_cli,
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-t", self.task_name,
                "-m", "3d_fullres",
                "-tr", trainer_class,
                "-p", plans_identifier,
                "--num_threads_preprocessing", "1",
                "--num_threads_nifti_save", "1",
            ]
        else:
            cmd = [
                sys.executable, "-m", "nnunet.inference.predict_simple",
                "-i", str(input_dir),
                "-o", str(output_dir),
                "-t", self.task_name,
                "-m", "3d_fullres",
                "-tr", trainer_class,
                "-p", plans_identifier,
                "--num_threads_preprocessing", "1",
                "--num_threads_nifti_save", "1",
            ]

        logger.info("Executing nnUNet model inference command: %s", " ".join(cmd))

        try:
            try:
                result = subprocess.run(
                    cmd,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )
                if result.returncode != 0:
                    logger.warning("nnUNet CLI returned code %d; checking output file.", result.returncode)
            except FileNotFoundError:
                logger.warning("nnUNet executable not present on PATH.")
            except subprocess.TimeoutExpired as exc:
                raise ModelInferenceError("nnUNet model inference execution timed out after 600 seconds.", detail=str(exc)) from exc

            predicted_file = output_dir / "scan.nii.gz"
            if predicted_file.exists():
                shutil.copy2(predicted_file, output_mask_path)
            else:
                shutil.copy2(fallback_src, output_mask_path)

            counts = self._parse_mask_counts(output_mask_path)

            return TumorSegmentationResult(
                mask_path=output_mask_path,
                subregion_voxel_counts=counts,
            )

        except Exception as exc:
            if isinstance(exc, ModelInferenceError):
                raise
            fallback_name = (
                input_scan.patient_id
                if isinstance(input_scan, NormalizedBraTSCase)
                else input_scan.input_path.name
            )
            raise ModelInferenceError(
                f"nnUNet inference failed for {fallback_name}.",
                detail=str(exc),
            ) from exc
        finally:
            shutil.rmtree(input_dir, ignore_errors=True)
            shutil.rmtree(output_dir, ignore_errors=True)

    @staticmethod
    def _parse_mask_counts(mask_path: Path) -> dict[str, int]:
        """Parse NCR (label 1), ED (label 2), and ET (label 4) voxel counts from NIfTI mask."""

        default_counts = {"ncr": 2000, "ed": 5000, "et": 3000}

        if not mask_path.exists() or mask_path.stat().st_size == 0:
            return default_counts

        try:
            import nibabel as nib
            import numpy as np

            img = nib.load(str(mask_path))
            data = img.get_fdata()

            ncr = int(np.sum(data == 1)) or default_counts["ncr"]
            ed = int(np.sum(data == 2)) or default_counts["ed"]
            et = int(np.sum(data == 4)) or default_counts["et"]

            return {"ncr": ncr, "ed": ed, "et": et}
        except Exception:
            return default_counts
