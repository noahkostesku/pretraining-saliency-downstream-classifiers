"""Curve integration helpers for insertion/deletion faithfulness metrics."""

import numpy as np

from .curves import build_curve_fraction_axis


def compute_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute area under a curve using trapezoidal integration."""
    x_values = np.asarray(x, dtype=np.float64)
    y_values = np.asarray(y, dtype=np.float64)

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError(
            f"x and y must be 1-D arrays, got {x_values.ndim} and {y_values.ndim}."
        )
    if x_values.shape[0] != y_values.shape[0]:
        raise ValueError(
            "x and y must have matching lengths, "
            f"got {x_values.shape[0]} and {y_values.shape[0]}."
        )
    if x_values.shape[0] < 2:
        raise ValueError("At least two curve points are required to compute AUC.")
    if not np.all(np.diff(x_values) >= 0):
        raise ValueError("x must be monotonically non-decreasing.")

    return float(np.trapezoid(y_values, x_values))


def insertion_auc(scores: np.ndarray, *, x: np.ndarray | None = None) -> float:
    """Compute insertion AUC (higher is better)."""
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {tuple(score_values.shape)}")

    x_values = (
        np.asarray(x, dtype=np.float64)
        if x is not None
        else build_curve_fraction_axis(score_values.shape[0] - 1).astype(np.float64)
    )
    return compute_auc(x_values, score_values)


def deletion_auc(scores: np.ndarray, *, x: np.ndarray | None = None) -> float:
    """Compute deletion AUC (lower is better)."""
    score_values = np.asarray(scores, dtype=np.float64)
    if score_values.ndim != 1:
        raise ValueError(f"scores must be 1-D, got shape {tuple(score_values.shape)}")

    x_values = (
        np.asarray(x, dtype=np.float64)
        if x is not None
        else build_curve_fraction_axis(score_values.shape[0] - 1).astype(np.float64)
    )
    return compute_auc(x_values, score_values)
