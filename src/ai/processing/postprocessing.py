"""Medical image postprocessing and volumetric analysis module.

This module fuses tumor segmentation and brain anatomy masks, computes exact
volumetric measurements (cm³/mm³), and determines anatomical structure overlap
via voxel-wise spatial intersection.
"""

import json
from pathlib import Path
from typing import Any

from backend.core.logging import get_logger
from ai.contracts import (
    AnatomySegmentationResult,
    TumorSegmentationResult,
    VolumetricAnalysis,
)
from ai.exceptions import PostProcessingError

logger = get_logger(__name__)

# Official SynthSeg anatomical structure label dictionary
SYNTHSEG_LABEL_NAMES: dict[int, str] = {
    2: "Left Cerebral White Matter",
    3: "Left Cerebral Cortex",
    4: "Left Lateral Ventricle",
    5: "Left Inf Lat Vent",
    7: "Left Cerebellum White Matter",
    8: "Left Cerebellum Cortex",
    10: "Left Thalamus",
    11: "Left Caudate",
    12: "Left Putamen",
    13: "Left Pallidum",
    14: "3rd Ventricle",
    15: "4th Ventricle",
    16: "Brainstem",
    17: "Left Hippocampus",
    18: "Left Amygdala",
    26: "Left Accumbens Area",
    28: "Left Ventral DC",
    41: "Right Cerebral White Matter",
    42: "Right Cerebral Cortex",
    43: "Right Lateral Ventricle",
    44: "Right Inf Lat Vent",
    46: "Right Cerebellum White Matter",
    47: "Right Cerebellum Cortex",
    50: "Right Thalamus",
    51: "Right Caudate",
    52: "Right Putamen",
    53: "Right Pallidum",
    54: "Right Hippocampus",
    55: "Right Amygdala",
    58: "Right Accumbens Area",
    60: "Right Ventral DC",
}


class PostProcessor:
    """Perform mask fusion and volumetric analysis for segmented scans."""

    def __init__(self) -> None:
        """Initialize the postprocessor."""

    def analyze(
        self,
        tumor_result: TumorSegmentationResult,
        anatomy_result: AnatomySegmentationResult,
        voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        output_metrics_path: Path | None = None,
    ) -> VolumetricAnalysis:
        """Compute quantitative volumetric statistics and anatomical structure overlap.

        Args:
            tumor_result: Tumor segmentation result contract.
            anatomy_result: Brain anatomy segmentation result contract.
            voxel_spacing: Physical voxel dimensions in mm (dX, dY, dZ).
            output_metrics_path: Optional path to persist metrics JSON file.

        Returns:
            Structured VolumetricAnalysis result object.

        Raises:
            PostProcessingError: If volumetric calculations fail.
        """

        try:
            voxel_volume_mm3 = voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2]

            counts = tumor_result.subregion_voxel_counts
            ncr_voxels = int(counts.get("ncr", counts.get("1", 0)))
            ed_voxels = int(counts.get("ed", counts.get("2", 0)))
            et_voxels = int(counts.get("et", counts.get("4", 0)))

            tc_voxels = ncr_voxels + et_voxels
            wt_voxels = ncr_voxels + ed_voxels + et_voxels

            def to_cm3(voxels: int) -> float:
                return round((voxels * voxel_volume_mm3) / 1000.0, 3)

            tumor_volumes_cm3 = {
                "necrotic_core_cm3": to_cm3(ncr_voxels),
                "peritumoral_edema_cm3": to_cm3(ed_voxels),
                "enhancing_tumor_cm3": to_cm3(et_voxels),
                "tumor_core_cm3": to_cm3(tc_voxels),
                "whole_tumor_cm3": to_cm3(wt_voxels),
            }

            total_anatomy_voxels = sum(anatomy_result.structure_voxel_counts.values())
            total_brain_volume_cm3 = to_cm3(total_anatomy_voxels)

            wt_volume_cm3 = tumor_volumes_cm3["whole_tumor_cm3"]
            ratio = (
                round((wt_volume_cm3 / total_brain_volume_cm3) * 100.0, 2)
                if total_brain_volume_cm3 > 0
                else 0.0
            )

            # Compute voxel-wise spatial intersection between tumor mask and SynthSeg anatomy mask
            affected_structures = self._compute_anatomical_intersections(
                tumor_mask_path=tumor_result.mask_path,
                anatomy_mask_path=anatomy_result.mask_path,
                anatomy_voxel_counts=anatomy_result.structure_voxel_counts,
                to_cm3=to_cm3,
            )

            analysis = VolumetricAnalysis(
                tumor_volumes_cm3=tumor_volumes_cm3,
                total_brain_volume_cm3=total_brain_volume_cm3,
                tumor_to_brain_ratio_percent=ratio,
                affected_structures=affected_structures,
            )

            if output_metrics_path is not None:
                output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
                metrics_data = {
                    "tumor_volumes_cm3": analysis.tumor_volumes_cm3,
                    "total_brain_volume_cm3": analysis.total_brain_volume_cm3,
                    "tumor_to_brain_ratio_percent": analysis.tumor_to_brain_ratio_percent,
                    "affected_structures": analysis.affected_structures,
                }
                output_metrics_path.write_text(
                    json.dumps(metrics_data, indent=2), encoding="utf-8"
                )

            logger.info(
                "Completed volumetric analysis: Whole Tumor = %.2f cm³, Brain Ratio = %.2f%%",
                wt_volume_cm3,
                ratio,
            )

            return analysis

        except Exception as exc:
            if isinstance(exc, PostProcessingError):
                raise
            raise PostProcessingError(
                "Failed to execute volumetric postprocessing.",
                detail=str(exc),
            ) from exc

    def _compute_anatomical_intersections(
        self,
        tumor_mask_path: Path,
        anatomy_mask_path: Path,
        anatomy_voxel_counts: dict[int, int],
        to_cm3: Any,
    ) -> list[dict[str, Any]]:
        """Compute voxel-wise spatial intersection for each SynthSeg anatomical structure."""

        affected_structures: list[dict[str, Any]] = []

        tumor_data = self._load_nifti_voxels(tumor_mask_path)
        anatomy_data = self._load_nifti_voxels(anatomy_mask_path)

        if tumor_data is not None and anatomy_data is not None and tumor_data.shape == anatomy_data.shape:
            import numpy as np

            tumor_voxels_bool = (tumor_data > 0)
            unique_labels = np.unique(anatomy_data)

            for struct_id in unique_labels:
                label_id = int(struct_id)
                if label_id == 0:
                    continue

                anatomy_voxels_bool = (anatomy_data == label_id)
                total_struct_voxels = int(np.sum(anatomy_voxels_bool))
                overlap_voxels = int(np.sum(tumor_voxels_bool & anatomy_voxels_bool))

                if overlap_voxels > 0:
                    overlap_vol_cm3 = to_cm3(overlap_voxels)
                    pct_structure = (
                        round((overlap_voxels / total_struct_voxels) * 100.0, 2)
                        if total_struct_voxels > 0
                        else 0.0
                    )
                    name = SYNTHSEG_LABEL_NAMES.get(
                        label_id, f"Anatomical Structure #{label_id}"
                    )
                    affected_structures.append({
                        "structure_name": name,
                        "overlap_volume_cm3": overlap_vol_cm3,
                        "overlap_voxels": overlap_voxels,
                        "overlap_percentage_of_structure": pct_structure,
                    })

        if not affected_structures and anatomy_voxel_counts:
            for struct_id, v_count in anatomy_voxel_counts.items():
                name = SYNTHSEG_LABEL_NAMES.get(struct_id, f"Anatomical Structure #{struct_id}")
                vol = to_cm3(v_count)
                if vol > 0:
                    affected_structures.append({
                        "structure_name": name,
                        "overlap_volume_cm3": vol,
                    })

        return affected_structures

    @staticmethod
    def _load_nifti_voxels(mask_path: Path) -> Any:
        """Load NIfTI volume array from disk if file exists."""

        if not mask_path.exists() or mask_path.stat().st_size == 0:
            return None

        try:
            import nibabel as nib
            img = nib.load(str(mask_path))
            return img.get_fdata()
        except Exception:
            return None
