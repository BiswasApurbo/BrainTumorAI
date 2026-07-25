"""Unit tests for AI preprocessing and postprocessing modules."""

from pathlib import Path
from uuid import uuid4

import pytest

from ai.contracts import AnatomySegmentationResult, TumorSegmentationResult
from ai.exceptions import PostProcessingError, PreprocessingError
from ai.processing.postprocessing import PostProcessor
from ai.processing.preprocessing import Preprocessor


def test_preprocessor_success(tmp_path: Path) -> None:
    """Test preprocessor creates normalized scan file from valid input."""

    raw_file = tmp_path / "raw_scan.nii.gz"
    raw_file.write_bytes(b"dummy_scan_bytes_12345")

    out_file = tmp_path / "normalized.nii.gz"

    preprocessor = Preprocessor()
    scan = preprocessor.process(raw_file, out_file)

    assert scan.input_path == out_file
    assert scan.original_path == raw_file
    assert out_file.exists()
    assert scan.orientation == "RAS"


def test_preprocessor_brats_case_success(tmp_path: Path) -> None:
    """Test preprocessor processes all four BraTS modalities independently."""

    from ai.contracts import BraTSCase

    flair = tmp_path / "flair.nii.gz"
    t1 = tmp_path / "t1.nii.gz"
    t1ce = tmp_path / "t1ce.nii.gz"
    t2 = tmp_path / "t2.nii.gz"

    for f in (flair, t1, t1ce, t2):
        f.write_bytes(b"dummy_mri_bytes_12345")

    case = BraTSCase(
        patient_id="BraTS20_Training_001",
        flair_path=flair,
        t1_path=t1,
        t1ce_path=t1ce,
        t2_path=t2,
    )

    out_dir = tmp_path / "preprocessed_case"
    preprocessor = Preprocessor()
    norm_case = preprocessor.process(case=case, output_directory=out_dir)

    assert norm_case.patient_id == "BraTS20_Training_001"
    assert norm_case.normalized_flair == out_dir / "normalized_flair.nii.gz"
    assert norm_case.normalized_t1 == out_dir / "normalized_t1.nii.gz"
    assert norm_case.normalized_t1ce == out_dir / "normalized_t1ce.nii.gz"
    assert norm_case.normalized_t2 == out_dir / "normalized_t2.nii.gz"
    assert (out_dir / "normalized_flair.nii.gz").exists()
    assert (out_dir / "normalized_t1.nii.gz").exists()
    assert (out_dir / "normalized_t1ce.nii.gz").exists()
    assert (out_dir / "normalized_t2.nii.gz").exists()



def test_preprocessor_file_not_found(tmp_path: Path) -> None:
    """Test preprocessor raises PreprocessingError when input file does not exist."""

    raw_file = tmp_path / "non_existent.nii.gz"
    out_file = tmp_path / "normalized.nii.gz"

    preprocessor = Preprocessor()
    with pytest.raises(PreprocessingError, match="Input file not found"):
        preprocessor.process(raw_file, out_file)


def test_preprocessor_empty_file(tmp_path: Path) -> None:
    """Test preprocessor raises PreprocessingError when input file has 0 bytes."""

    raw_file = tmp_path / "empty.nii.gz"
    raw_file.write_bytes(b"")
    out_file = tmp_path / "normalized.nii.gz"

    preprocessor = Preprocessor()
    with pytest.raises(PreprocessingError, match="Input file is empty"):
        preprocessor.process(raw_file, out_file)


def test_postprocessor_analysis_success(tmp_path: Path) -> None:
    """Test postprocessor computes volumetric analysis accurately."""

    tumor_result = TumorSegmentationResult(
        mask_path=tmp_path / "tumor_mask.nii.gz",
        subregion_voxel_counts={"ncr": 2000, "ed": 5000, "et": 3000},
    )

    anatomy_result = AnatomySegmentationResult(
        mask_path=tmp_path / "anatomy_mask.nii.gz",
        structure_voxel_counts={2: 500000, 3: 200000},
    )

    postprocessor = PostProcessor()
    metrics_path = tmp_path / "metrics.json"

    analysis = postprocessor.analyze(
        tumor_result=tumor_result,
        anatomy_result=anatomy_result,
        voxel_spacing=(1.0, 1.0, 1.0),
        output_metrics_path=metrics_path,
    )

    assert analysis.tumor_volumes_cm3["enhancing_tumor_cm3"] == 3.0
    assert analysis.tumor_volumes_cm3["peritumoral_edema_cm3"] == 5.0
    assert analysis.tumor_volumes_cm3["necrotic_core_cm3"] == 2.0
    assert analysis.tumor_volumes_cm3["tumor_core_cm3"] == 5.0
    assert analysis.tumor_volumes_cm3["whole_tumor_cm3"] == 10.0
    assert metrics_path.exists()
