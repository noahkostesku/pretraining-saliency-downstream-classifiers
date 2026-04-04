"""Explainability methods and saliency map IO helpers."""

from .gradcam import generate_gradcam, generate_gradcampp
from .occlusion import generate_occlusion_map
from .pipeline import (
    Stage4Run,
    discover_stage4_runs,
    generate_explanations_for_run,
    generate_explanations_for_runs,
)
from .qc import run_explanation_qc, write_explanation_qc_report
from .saliency_io import (
    load_saliency_map,
    normalize_saliency_batch,
    normalize_saliency_map,
    read_saliency_metadata,
    resize_saliency_map,
    save_saliency_map,
    write_saliency_metadata,
)
from .targets import (
    gather_target_scores,
    get_predicted_class_target,
    predicted_class_scores,
)

__all__ = [
    "Stage4Run",
    "discover_stage4_runs",
    "gather_target_scores",
    "generate_explanations_for_run",
    "generate_explanations_for_runs",
    "generate_gradcam",
    "generate_gradcampp",
    "generate_occlusion_map",
    "get_predicted_class_target",
    "load_saliency_map",
    "normalize_saliency_batch",
    "normalize_saliency_map",
    "predicted_class_scores",
    "read_saliency_metadata",
    "resize_saliency_map",
    "run_explanation_qc",
    "save_saliency_map",
    "write_explanation_qc_report",
    "write_saliency_metadata",
]
