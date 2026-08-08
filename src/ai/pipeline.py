"""Main AI Pipeline Orchestrator for BrainTumorAI.

This module coordinates end-to-end medical scan processing, tumor segmentation,
brain anatomy segmentation, volumetric postprocessing, 3D reconstruction,
Plotly HTML visualization rendering, and clinical reporting.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from backend.core.logging import get_logger
from backend.services.report_service import ReportService
from ai.adapters.nnunet_adapter import NNUNetAdapter
from ai.adapters.synthseg_adapter import SynthSegAdapter
from ai.contracts import AIPipelineResult, BraTSCase, NormalizedBraTSCase, NormalizedScan, TumorSegmentationResult
from ai.exceptions import AIProcessingError
from ai.processing.postprocessing import PostProcessor
from ai.processing.preprocessing import Preprocessor
from ai.reporting.report_generator import ReportGenerator
from ai.visualization.plotly_renderer import PlotlyRenderer
from ai.visualization.reconstruction_3d import Reconstruction3D
from ai.workspace import WorkspaceManager

if TYPE_CHECKING:
    from backend.services.inference_service import InferenceJob

logger = get_logger(__name__)


class AIPipeline:
    """Main orchestrator for end-to-end AI tumor analysis workflows."""

    def __init__(
        self,
        preprocessor: Preprocessor | None = None,
        nnunet_adapter: NNUNetAdapter | None = None,
        synthseg_adapter: SynthSegAdapter | None = None,
        postprocessor: PostProcessor | None = None,
        reconstruction_3d: Reconstruction3D | None = None,
        plotly_renderer: PlotlyRenderer | None = None,
        report_generator: ReportGenerator | None = None,
        report_service: ReportService | None = None,
    ) -> None:
        """Initialize the AI pipeline orchestrator.

        Args:
            preprocessor: Preprocessing collaborator.
            nnunet_adapter: nnU-Net tumor segmentation adapter.
            synthseg_adapter: SynthSeg brain anatomy adapter.
            postprocessor: Postprocessing volumetric collaborator.
            reconstruction_3d: 3D surface mesh extraction collaborator.
            plotly_renderer: Plotly HTML rendering collaborator.
            report_generator: Clinical report generation collaborator.
            report_service: Report storage service collaborator.
        """

        self.preprocessor = preprocessor or Preprocessor()
        self.nnunet_adapter = nnunet_adapter or NNUNetAdapter()
        self.synthseg_adapter = synthseg_adapter or SynthSegAdapter()
        self.postprocessor = postprocessor or PostProcessor()
        self.reconstruction_3d = reconstruction_3d or Reconstruction3D()
        self.plotly_renderer = plotly_renderer or PlotlyRenderer()
        self.report_generator = report_generator or ReportGenerator(
            report_service=report_service
        )

    def run(self, job: "InferenceJob") -> AIPipelineResult:
        """Run the complete AI analysis pipeline for an inference job.

        Supports both legacy single-file inputs (Path to a NIfTI file) and
        multi-modal BraTS patient directories containing FLAIR, T1, T1CE, and
        T2 modalities.

        Args:
            job: Shared InferenceJob contract containing request_id, upload_id,
                and input_path.

        Returns:
            AIPipelineResult containing report_id and visualization_path.

        Raises:
            AIProcessingError: If any pipeline stage fails.
        """

        logger.info(
            "Starting AI pipeline processing for request %s (upload: %s)",
            job.request_id,
            job.upload_id,
        )

        workspace = WorkspaceManager(request_id=job.request_id)
        workspace.create()

        try:
            brats_case = self._detect_brats_case(job.input_path)

            if brats_case is not None:
                # --- Multi-modal BraTS workflow ---
                logger.info(
                    "Detected BraTS patient directory for %s. Using multi-modal pipeline.",
                    brats_case.patient_id,
                )

                # 1. Preprocessing (four modalities)
                normalized_brats = self.preprocessor.process(
                    case=brats_case,
                    output_directory=workspace.workspace_dir / "preprocessed",
                )

                # Build a NormalizedScan from FLAIR for SynthSeg and postprocessing
                synthseg_scan = NormalizedScan(
                    input_path=normalized_brats.normalized_flair,
                    original_path=brats_case.flair_path,
                    dimensions=normalized_brats.dimensions,
                    voxel_spacing=normalized_brats.voxel_spacing,
                    orientation=normalized_brats.orientation,
                )

                voxel_spacing = normalized_brats.voxel_spacing

                if job.mode == "ground_truth" and brats_case.segmentation_path:
                    import nibabel as nib
                    img = nib.load(str(brats_case.segmentation_path))
                    nib.save(img, str(workspace.tumor_mask_path))
                    tumor_result = TumorSegmentationResult(mask_path=workspace.tumor_mask_path)
                else:
                    # 2. Tumor Segmentation (nnUNet) — receives NormalizedBraTSCase
                    tumor_result = self.nnunet_adapter.predict(
                        input_scan=normalized_brats,
                        output_mask_path=workspace.tumor_mask_path,
                    )

                # 3. Brain Anatomy Segmentation (SynthSeg) — receives NormalizedScan (FLAIR)
                anatomy_result = self.synthseg_adapter.predict(
                    input_scan=synthseg_scan,
                    output_mask_path=workspace.anatomy_mask_path,
                )

            else:
                # --- Legacy single-file workflow (unchanged) ---

                # 1. Preprocessing
                normalized_scan = self.preprocessor.process(
                    input_path=job.input_path,
                    output_path=workspace.normalized_input_path,
                )

                voxel_spacing = normalized_scan.voxel_spacing

                # 2. Tumor Segmentation (nnUNet)
                tumor_result = self.nnunet_adapter.predict(
                    input_scan=normalized_scan,
                    output_mask_path=workspace.tumor_mask_path,
                )

                # 3. Brain Anatomy Segmentation (SynthSeg)
                anatomy_result = self.synthseg_adapter.predict(
                    input_scan=normalized_scan,
                    output_mask_path=workspace.anatomy_mask_path,
                )

            # 4. Volumetric Postprocessing
            volumetric_analysis = self.postprocessor.analyze(
                tumor_result=tumor_result,
                anatomy_result=anatomy_result,
                voxel_spacing=voxel_spacing,
                output_metrics_path=workspace.metrics_json_path,
            )

            # 5. 3D Surface Mesh Extraction & Plotly Visualization
            meshes = self.reconstruction_3d.extract_meshes(
                tumor_mask_path=tumor_result.mask_path,
                anatomy_mask_path=anatomy_result.mask_path,
            )

            visualization_path = self.plotly_renderer.render_html(
                meshes=meshes,
                output_html_path=workspace.visualization_html_path,
                title=f"BrainTumorAI 3D Reconstruction - Request {str(job.request_id)[:8]}",
            )

            # 6. Clinical Diagnostic Report Generation
            report_metadata = self.report_generator.generate(
                upload_id=job.upload_id,
                volumetric_analysis=volumetric_analysis,
                visualization_path=visualization_path,
            )

            logger.info(
                "Completed AI pipeline processing for request %s. Report ID: %s",
                job.request_id,
                report_metadata.report_id,
            )

            return AIPipelineResult(
                request_id=job.request_id,
                upload_id=job.upload_id,
                report_id=report_metadata.report_id,
                visualization_path=visualization_path,
                volumetric_analysis=volumetric_analysis,
            )

        except AIProcessingError:
            raise
        except Exception as exc:
            raise AIProcessingError(
                f"Unexpected error executing AI pipeline for request {job.request_id}.",
                detail=str(exc),
            ) from exc
        finally:
            workspace.cleanup()

    @staticmethod
    def _detect_brats_case(input_path: Path) -> BraTSCase | None:
        """Detect whether input_path is a BraTS patient directory.

        A valid BraTS directory must contain files matching the pattern:
        *_flair.nii, *_t1.nii, *_t1ce.nii, *_t2.nii (or .nii.gz variants).

        Args:
            input_path: Filesystem path (file or directory).

        Returns:
            A BraTSCase if all four modalities are found, otherwise None.
        """

        if not input_path.is_dir():
            return None

        patient_id = input_path.name

        def _find_modality(suffix: str) -> Path | None:
            """Find the first file matching *_{suffix}.nii or *_{suffix}.nii.gz."""
            for ext in (".nii", ".nii.gz"):
                candidates = list(input_path.glob(f"*_{suffix}{ext}"))
                if candidates:
                    return candidates[0]
            return None

        flair = _find_modality("flair")
        t1 = _find_modality("t1")
        t1ce = _find_modality("t1ce")
        t2 = _find_modality("t2")

        if not all((flair, t1, t1ce, t2)):
            return None

        seg = _find_modality("seg")

        return BraTSCase(
            patient_id=patient_id,
            flair_path=flair,  # type: ignore[arg-type]
            t1_path=t1,  # type: ignore[arg-type]
            t1ce_path=t1ce,  # type: ignore[arg-type]
            t2_path=t2,  # type: ignore[arg-type]
            segmentation_path=seg,
        )

