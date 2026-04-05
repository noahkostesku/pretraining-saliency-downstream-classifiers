"""Insertion/deletion patch ranking and curve-axis helpers."""

from collections.abc import Sequence

import numpy as np


PatchSlice = tuple[slice, slice]


def build_patch_slices(
    *,
    image_height: int,
    image_width: int,
    patch_size: int = 16,
    stride: int = 16,
) -> list[PatchSlice]:
    """Return deterministic image-space patch slices.

    Patches are generated row-major (top-to-bottom, left-to-right).
    """
    if image_height <= 0 or image_width <= 0:
        raise ValueError(
            "image_height and image_width must be positive, "
            f"got ({image_height}, {image_width})."
        )
    if patch_size <= 0 or stride <= 0:
        raise ValueError(
            "patch_size and stride must be positive, "
            f"got patch_size={patch_size}, stride={stride}."
        )
    if patch_size > image_height or patch_size > image_width:
        raise ValueError(
            "patch_size must not exceed image dimensions, "
            f"got patch_size={patch_size} for ({image_height}, {image_width})."
        )

    ys = range(0, image_height - patch_size + 1, stride)
    xs = range(0, image_width - patch_size + 1, stride)
    patch_slices: list[PatchSlice] = [
        (slice(y, y + patch_size), slice(x, x + patch_size)) for y in ys for x in xs
    ]
    if not patch_slices:
        raise ValueError(
            "No patch slices were produced. Check image size and patch/stride settings."
        )
    return patch_slices


def patch_mean_scores(
    saliency: np.ndarray,
    *,
    patch_slices: Sequence[PatchSlice],
) -> np.ndarray:
    """Compute mean saliency score for each patch in ``patch_slices``."""
    saliency_2d = np.asarray(saliency, dtype=np.float32)
    if saliency_2d.ndim != 2:
        raise ValueError(
            f"Expected 2-D saliency map, got shape {tuple(saliency_2d.shape)}"
        )
    if not patch_slices:
        raise ValueError("patch_slices must contain at least one patch.")

    means = np.asarray(
        [
            float(np.mean(saliency_2d[y_slice, x_slice]))
            for y_slice, x_slice in patch_slices
        ],
        dtype=np.float32,
    )
    return means


def rank_patches_by_saliency(
    saliency: np.ndarray,
    *,
    patch_slices: Sequence[PatchSlice],
) -> np.ndarray:
    """Rank patch indices by descending mean saliency."""
    means = patch_mean_scores(saliency, patch_slices=patch_slices)
    # Stable sort avoids run-to-run tie jitter.
    return np.argsort(-means, kind="mergesort")


def build_curve_fraction_axis(num_patches: int) -> np.ndarray:
    """Return normalized perturbation fractions from 0.0 to 1.0."""
    if num_patches <= 0:
        raise ValueError(f"num_patches must be positive, got {num_patches}.")
    return np.linspace(0.0, 1.0, num=num_patches + 1, dtype=np.float32)
