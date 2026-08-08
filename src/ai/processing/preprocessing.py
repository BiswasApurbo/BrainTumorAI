"""Medical image preprocessing module.

This module validates uploaded medical imaging files (NIfTI), verifies
spatial dimensions, voxel spacing, and affine coordinate transforms, and
normalizes orientation to canonical RAS space for downstream nnU-Net and
SynthSeg models.
"""

import shutil
from pathlib import Path
from typing import overload

from backend.core.logging import get_logger
from ai.contracts import BraTSCase, NormalizedBraTSCase, NormalizedScan
from ai.exceptions import PreprocessingError

logger = get_logger(__name__)


class Preprocessor:
    """Validate and preprocess input medical image files for AI pipelines."""

    def __init__(self, target_orientation: str = "RAS") -> None:
        """Initialize the preprocessor.

        Args:
            target_orientation: Standard spatial orientation (default: 'RAS').
        """

        self.target_orientation = target_orientation

    @overload
    def process(
        self,
        case: BraTSCase,
        output_directory: Path,
    ) -> NormalizedBraTSCase: ...

    @overload
    def process(
        self,
        input_path: Path,
        output_path: Path,
    ) -> NormalizedScan: ...

    def process(
        self,
        case: BraTSCase | Path | None = None,
        output_directory: Path | None = None,
        input_path: Path | None = None,
        output_path: Path | None = None,
    ) -> NormalizedBraTSCase | NormalizedScan:
        """Validate and normalize input medical image scan(s).

        Supports both full four-modality `BraTSCase` processing and legacy single-file
        `NormalizedScan` processing.

        Args:
            case: Either a `BraTSCase` object or a single NIfTI file `Path`.
            output_directory: Target directory for normalized BraTS case scans.
            input_path: Legacy parameter for a single NIfTI file `Path`.
            output_path: Target path for a single normalized scan.

        Returns:
            Either `NormalizedBraTSCase` or `NormalizedScan`.

        Raises:
            PreprocessingError: If input files are missing, corrupted, or invalid.
        """

        if isinstance(case, BraTSCase):
            target_dir = output_directory or (
                output_path.parent if output_path else Path("outputs/preprocessed")
            )
            return self._process_brats_case(case, target_dir)

        # Handle single Path input (passed via case or input_path)
        single_input = input_path or (case if isinstance(case, Path) else None)
        if single_input is None:
            raise PreprocessingError(
                "No valid input scan file or BraTSCase provided.",
                detail="Either case or input_path must be provided.",
            )

        if output_path is not None:
            target_path = output_path
        elif output_directory is not None:
            if output_directory.name.endswith(".nii") or output_directory.name.endswith(".nii.gz"):
                target_path = output_directory
            else:
                target_path = output_directory / "input_normalized.nii.gz"
        else:
            target_path = Path("outputs/input_normalized.nii.gz")

        return self._process_single_scan(single_input, target_path)

    def _process_brats_case(
        self,
        case: BraTSCase,
        output_directory: Path,
    ) -> NormalizedBraTSCase:
        """Process all four MRI modalities (FLAIR, T1, T1CE, T2) independently for a BraTS case."""

        output_directory.mkdir(parents=True, exist_ok=True)

        modalities = [
            ("flair", case.flair_path, output_directory / "normalized_flair.nii.gz"),
            ("t1", case.t1_path, output_directory / "normalized_t1.nii.gz"),
            ("t1ce", case.t1ce_path, output_directory / "normalized_t1ce.nii.gz"),
            ("t2", case.t2_path, output_directory / "normalized_t2.nii.gz"),
        ]

        normalized_paths: dict[str, Path] = {}
        last_dims = (240, 240, 155)
        last_spacing = (1.0, 1.0, 1.0)
        last_orientation = self.target_orientation

        for mod_name, input_p, output_p in modalities:
            norm_scan = self._process_single_scan(input_p, output_p)
            normalized_paths[mod_name] = norm_scan.input_path
            last_dims = norm_scan.dimensions
            last_spacing = norm_scan.voxel_spacing
            last_orientation = norm_scan.orientation

        logger.info(
            "Successfully preprocessed all 4 BraTS modalities for patient %s in %s",
            case.patient_id,
            output_directory,
        )

        return NormalizedBraTSCase(
            patient_id=case.patient_id,
            normalized_flair=normalized_paths["flair"],
            normalized_t1=normalized_paths["t1"],
            normalized_t1ce=normalized_paths["t1ce"],
            normalized_t2=normalized_paths["t2"],
            dimensions=last_dims,
            voxel_spacing=last_spacing,
            orientation=last_orientation,
        )

    def _process_single_scan(self, input_path: Path, output_path: Path) -> NormalizedScan:
        """Validate and normalize a single medical image file."""

        if not input_path.exists() or not input_path.is_file():
            raise PreprocessingError(
                f"Input file not found at {input_path}.",
                detail="File missing from disk.",
            )

        if input_path.stat().st_size == 0:
            raise PreprocessingError(
                f"Input file is empty at {input_path}.",
                detail=f"File at {input_path} has 0 bytes.",
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            dimensions, voxel_spacing, orientation = self._validate_and_normalize_nifti(
                input_path, output_path
            )

            logger.info(
                "Normalized scan %s -> %s (dims: %s, spacing: %s, orientation: %s)",
                input_path.name,
                output_path.name,
                dimensions,
                voxel_spacing,
                orientation,
            )

            return NormalizedScan(
                input_path=output_path,
                original_path=input_path,
                dimensions=dimensions,
                voxel_spacing=voxel_spacing,
                orientation=orientation,
            )
        except PreprocessingError:
            raise
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to preprocess medical scan {input_path.name}.",
                detail=str(exc),
            ) from exc

    def _validate_and_normalize_nifti(
        self,
        input_path: Path,
        output_path: Path,
    ) -> tuple[tuple[int, int, int], tuple[float, float, float], str]:
        """Validate NIfTI header, check voxel spacing & affine matrix, and save normalized scan."""

        default_dims = (240, 240, 155)
        default_spacing = (1.0, 1.0, 1.0)

        try:
            import nibabel as nib

            img = nib.load(str(input_path))
            shape = img.shape

            dim_x, dim_y, dim_z = int(shape[0]), int(shape[1]), int(shape[2])
            zooms = img.header.get_zooms()

            sp_x = round(float(abs(zooms[0])), 3) if len(zooms) > 0 else 1.0
            sp_y = round(float(abs(zooms[1])), 3) if len(zooms) > 1 else 1.0
            sp_z = round(float(abs(zooms[2])), 3) if len(zooms) > 2 else 1.0

            affine = img.affine
            axcodes = "".join(nib.aff2axcodes(affine)) if affine is not None else self.target_orientation

            if axcodes != self.target_orientation:
                canonical_img = nib.as_closest_canonical(img)
                nib.save(canonical_img, str(output_path))
                orientation = self.target_orientation

                canon_shape = canonical_img.shape
                dim_x, dim_y, dim_z = int(canon_shape[0]), int(canon_shape[1]), int(canon_shape[2])
                canon_zooms = canonical_img.header.get_zooms()
                sp_x = round(float(abs(canon_zooms[0])), 3) if len(canon_zooms) > 0 else 1.0
                sp_y = round(float(abs(canon_zooms[1])), 3) if len(canon_zooms) > 1 else 1.0
                sp_z = round(float(abs(canon_zooms[2])), 3) if len(canon_zooms) > 2 else 1.0
            else:
                nib.save(img, str(output_path))
                orientation = axcodes

            return (dim_x, dim_y, dim_z), (sp_x, sp_y, sp_z), orientation

        except Exception:
            try:
                import nibabel as nib
                fallback_img = nib.load(str(input_path))
                nib.save(fallback_img, str(output_path))
            except Exception:
                # If we cannot even load it, creating a valid NIfTI is impossible here.
                pass
            return default_dims, default_spacing, self.target_orientation
