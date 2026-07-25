"""Data contracts and DTOs for the BrainTumorAI pipeline.

This module defines typed dataclasses for passing data cleanly between
pipeline stages (preprocessing, segmentation adapters, postprocessing, 3D
reconstruction, visualization, and report generation).
"""

from dataclasses import dataclass, field
from pathlib import Path
from uuid import UUID


@dataclass(frozen=True, slots=True)
class NormalizedScan:
    """Metadata describing a preprocessed, standardized medical scan.

    Attributes:
        input_path: Path to the preprocessed NIfTI scan file.
        original_path: Path to the raw uploaded file.
        dimensions: Voxel matrix dimensions (X, Y, Z).
        voxel_spacing: Physical voxel dimensions in mm (dX, dY, dZ).
        orientation: Spatial orientation string (e.g., 'RAS').
    """

    input_path: Path
    original_path: Path
    dimensions: tuple[int, int, int]
    voxel_spacing: tuple[float, float, float]
    orientation: str = "RAS"


@dataclass(frozen=True, slots=True)
class BraTSCase:
    """Raw BraTS patient case contract representing all four MRI modalities.

    Attributes:
        patient_id: Patient or case identification string (e.g., 'BraTS20_Training_001').
        flair_path: Path to the raw FLAIR MRI scan NIfTI file.
        t1_path: Path to the raw T1-weighted MRI scan NIfTI file.
        t1ce_path: Path to the raw T1-contrast enhanced MRI scan NIfTI file.
        t2_path: Path to the raw T2-weighted MRI scan NIfTI file.
        segmentation_path: Optional path to the expert ground truth segmentation mask NIfTI file.
    """

    patient_id: str
    flair_path: Path
    t1_path: Path
    t1ce_path: Path
    t2_path: Path
    segmentation_path: Path | None = None


@dataclass(frozen=True, slots=True)
class NormalizedBraTSCase:
    """Preprocessed and standardized multi-modal BraTS patient case contract.

    Attributes:
        patient_id: Patient or case identification string.
        normalized_flair: Path to the preprocessed RAS FLAIR NIfTI scan.
        normalized_t1: Path to the preprocessed RAS T1 NIfTI scan.
        normalized_t1ce: Path to the preprocessed RAS T1CE NIfTI scan.
        normalized_t2: Path to the preprocessed RAS T2 NIfTI scan.
        dimensions: Voxel matrix dimensions (X, Y, Z).
        voxel_spacing: Physical voxel dimensions in mm (dX, dY, dZ).
        orientation: Spatial orientation string (e.g., 'RAS').
    """

    patient_id: str
    normalized_flair: Path
    normalized_t1: Path
    normalized_t1ce: Path
    normalized_t2: Path
    dimensions: tuple[int, int, int] = (240, 240, 155)
    voxel_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    orientation: str = "RAS"


@dataclass(frozen=True, slots=True)
class TumorSegmentationResult:
    """Output contract from the nnUNet tumor segmentation adapter.

    Attributes:
        mask_path: Path to the generated multi-class tumor label NIfTI mask.
        subregion_voxel_counts: Voxel count dictionary per tumor subregion
            (e.g., 'et', 'tc', 'wt', 'ncr', 'ed').
    """

    mask_path: Path
    subregion_voxel_counts: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnatomySegmentationResult:
    """Output contract from the SynthSeg brain anatomy segmentation adapter.

    Attributes:
        mask_path: Path to the generated 33+ anatomical structure label NIfTI mask.
        structure_voxel_counts: Voxel count dictionary per anatomical structure ID.
    """

    mask_path: Path
    structure_voxel_counts: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VolumetricAnalysis:
    """Quantitative volumetric measurement results.

    Attributes:
        tumor_volumes_cm3: Volume in cm³ for Whole Tumor (WT), Tumor Core (TC),
            Enhancing Tumor (ET), Peritumoral Edema (ED), and Necrotic Core (NCR).
        total_brain_volume_cm3: Calculated total brain volume in cm³.
        tumor_to_brain_ratio_percent: Tumor volume percentage relative to brain volume.
        affected_structures: List of anatomical brain regions overlapping with tumor.
    """

    tumor_volumes_cm3: dict[str, float]
    total_brain_volume_cm3: float
    tumor_to_brain_ratio_percent: float
    affected_structures: list[dict[str, str | float]] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class Mesh3DData:
    """3D Surface mesh data for visualization rendering.

    Attributes:
        vertices: List of 3D coordinates [[x, y, z], ...].
        faces: List of triangular mesh faces [[v1, v2, v3], ...].
        label: Descriptive label (e.g., 'Enhancing Tumor', 'Ventricles').
        color: Hex color string (e.g., '#FF0000').
        opacity: Surface opacity value between 0.0 and 1.0.
    """

    vertices: list[list[float]]
    faces: list[list[int]]
    label: str
    color: str
    opacity: float = 0.8


@dataclass(frozen=True, slots=True)
class AIPipelineResult:
    """Final output contract returned upon successful AI pipeline execution.

    Attributes:
        request_id: Unique request identifier.
        upload_id: Associated upload identifier.
        report_id: Associated report identifier.
        visualization_path: Filesystem path to the generated 3D Plotly HTML.
        volumetric_analysis: Quantitative volumetric analysis result.
    """

    request_id: UUID
    upload_id: UUID
    report_id: UUID
    visualization_path: Path
    volumetric_analysis: VolumetricAnalysis
