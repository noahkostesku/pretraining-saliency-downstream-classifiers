"""Explanation curve construction, AUC, and summary helpers."""

from .auc import compute_auc, deletion_auc, insertion_auc
from .bootstrap import bootstrap_ci, paired_bootstrap_ci, paired_permutation_pvalue
from .curves import (
    build_curve_fraction_axis,
    build_patch_slices,
    patch_mean_scores,
    rank_patches_by_saliency,
)
from .insertion_deletion import InsertionDeletionResult, run_insertion_deletion
from .summarize import (
    classify_gradcampp_outcome,
    compute_method_deltas,
    compute_primary_correct_intersection,
    summarize_condition_level_deltas,
    summarize_condition_level_metrics,
    summarize_seed_level_deltas,
    summarize_seed_level_metrics,
)

__all__ = [
    "InsertionDeletionResult",
    "bootstrap_ci",
    "build_curve_fraction_axis",
    "build_patch_slices",
    "classify_gradcampp_outcome",
    "compute_auc",
    "compute_method_deltas",
    "compute_primary_correct_intersection",
    "deletion_auc",
    "insertion_auc",
    "paired_bootstrap_ci",
    "paired_permutation_pvalue",
    "patch_mean_scores",
    "rank_patches_by_saliency",
    "run_insertion_deletion",
    "summarize_condition_level_deltas",
    "summarize_condition_level_metrics",
    "summarize_seed_level_deltas",
    "summarize_seed_level_metrics",
]
