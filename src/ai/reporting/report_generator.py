"""Clinical diagnostic report generator module.

This module aggregates volumetric measurements, spatial structure overlap,
and 3D visualization paths into structured clinical reports persisted via ReportService.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from backend.core.logging import get_logger
from backend.services.report_service import ReportMetadata, ReportService
from ai.contracts import VolumetricAnalysis
from ai.exceptions import ReportGenerationError

logger = get_logger(__name__)


class ReportGenerator:
    """Aggregate pipeline output into clinical diagnostic reports."""

    def __init__(self, report_service: ReportService | None = None) -> None:
        """Initialize the report generator.

        Args:
            report_service: Service used to persist diagnostic reports. If
                omitted, a default ReportService instance is initialized.
        """

        self.report_service = report_service or ReportService()

    def generate(
        self,
        upload_id: UUID | str,
        volumetric_analysis: VolumetricAnalysis,
        visualization_path: Path,
    ) -> ReportMetadata:
        """Generate and persist a clinical report for an uploaded scan.

        Args:
            upload_id: Associated upload UUID or UUID string.
            volumetric_analysis: Volumetric analysis results.
            visualization_path: Filesystem path to the 3D Plotly HTML.

        Returns:
            ReportMetadata contract for the newly stored report.

        Raises:
            ReportGenerationError: If input validation, report building, or persistence fails.
        """

        if not upload_id:
            raise ReportGenerationError("upload_id must be provided for report generation.")

        if volumetric_analysis is None or not hasattr(volumetric_analysis, "tumor_volumes_cm3"):
            raise ReportGenerationError(
                "Valid VolumetricAnalysis result object is required for report generation."
            )

        if visualization_path is None or not visualization_path.exists():
            raise ReportGenerationError(
                f"3D visualization file not found at {visualization_path}.",
                detail="Missing visualization HTML file.",
            )

        try:
            wt_cm3 = float(volumetric_analysis.tumor_volumes_cm3.get("whole_tumor_cm3", 0.0))
            et_cm3 = float(volumetric_analysis.tumor_volumes_cm3.get("enhancing_tumor_cm3", 0.0))
            ed_cm3 = float(volumetric_analysis.tumor_volumes_cm3.get("peritumoral_edema_cm3", 0.0))
            ncr_cm3 = float(volumetric_analysis.tumor_volumes_cm3.get("necrotic_core_cm3", 0.0))
            brain_cm3 = float(volumetric_analysis.total_brain_volume_cm3)
            ratio_pct = float(volumetric_analysis.tumor_to_brain_ratio_percent)

            summary = (
                f"Quantitative volumetric analysis identified a whole tumor volume of {wt_cm3:.3f} cm³ "
                f"representing {ratio_pct:.2f}% of total segmented brain volume ({brain_cm3:.3f} cm³). "
                f"Subregion breakdown: Enhancing Tumor = {et_cm3:.3f} cm³, Peritumoral Edema = {ed_cm3:.3f} cm³, Necrotic Core = {ncr_cm3:.3f} cm³."
            )

            report_content = {
                "title": "BrainTumorAI Diagnostic Volumetric Analysis & Segmentation Report",
                "upload_id": str(upload_id),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "volumetric_analysis": {
                    "whole_tumor_cm3": wt_cm3,
                    "enhancing_tumor_cm3": et_cm3,
                    "peritumoral_edema_cm3": ed_cm3,
                    "necrotic_core_cm3": ncr_cm3,
                    "tumor_core_cm3": float(volumetric_analysis.tumor_volumes_cm3.get("tumor_core_cm3", 0.0)),
                },
                "total_brain_volume_cm3": brain_cm3,
                "tumor_to_brain_ratio_percent": ratio_pct,
                "affected_structures": volumetric_analysis.affected_structures,
                "visualization_path": str(visualization_path),
                "visualization_url": f"/outputs/visualizations/{visualization_path.name}",
            }

            metadata = self.report_service.store_report(
                upload_id=upload_id,
                content=report_content,
                status="completed",
            )

            logger.info(
                "Successfully generated and stored clinical report %s for upload %s.",
                metadata.report_id,
                upload_id,
            )
            return metadata

        except ReportGenerationError:
            raise
        except Exception as exc:
            raise ReportGenerationError(
                f"Failed to generate clinical report for upload {upload_id}.",
                detail=str(exc),
            ) from exc
