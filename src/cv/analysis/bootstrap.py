"""Bootstrap and permutation utilities for paired explanation comparisons."""

import numpy as np


def bootstrap_ci(
    values: np.ndarray,
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return ``(mean, ci_low, ci_high)`` for the sample mean."""
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.ndim != 1:
        raise ValueError(
            f"Expected 1-D values array, got shape {tuple(values_array.shape)}"
        )
    if values_array.size == 0:
        raise ValueError("Expected at least one value for bootstrap CI.")
    if num_resamples <= 0:
        raise ValueError(f"num_resamples must be positive, got {num_resamples}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")

    rng = np.random.default_rng(seed)
    sample_size = values_array.size
    bootstrap_means = np.empty(num_resamples, dtype=np.float64)
    for idx in range(num_resamples):
        indices = rng.integers(0, sample_size, size=sample_size)
        bootstrap_means[idx] = float(np.mean(values_array[indices]))

    alpha = (1.0 - confidence) / 2.0
    ci_low = float(np.quantile(bootstrap_means, alpha))
    ci_high = float(np.quantile(bootstrap_means, 1.0 - alpha))
    return float(np.mean(values_array)), ci_low, ci_high


def paired_bootstrap_ci(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    num_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Return bootstrap CI for paired mean delta ``(b - a)``."""
    a_array = np.asarray(values_a, dtype=np.float64)
    b_array = np.asarray(values_b, dtype=np.float64)
    if a_array.ndim != 1 or b_array.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays, got shapes {a_array.shape} and {b_array.shape}."
        )
    if a_array.shape[0] != b_array.shape[0]:
        raise ValueError(
            "Paired arrays must have equal length, "
            f"got {a_array.shape[0]} and {b_array.shape[0]}."
        )

    deltas = b_array - a_array
    return bootstrap_ci(
        deltas,
        num_resamples=num_resamples,
        confidence=confidence,
        seed=seed,
    )


def paired_permutation_pvalue(
    values_a: np.ndarray,
    values_b: np.ndarray,
    *,
    num_resamples: int = 10000,
    seed: int = 0,
) -> float:
    """Two-sided paired permutation p-value for mean delta ``(b - a)``."""
    a_array = np.asarray(values_a, dtype=np.float64)
    b_array = np.asarray(values_b, dtype=np.float64)
    if a_array.ndim != 1 or b_array.ndim != 1:
        raise ValueError(
            f"Expected 1-D arrays, got shapes {a_array.shape} and {b_array.shape}."
        )
    if a_array.shape[0] != b_array.shape[0]:
        raise ValueError(
            "Paired arrays must have equal length, "
            f"got {a_array.shape[0]} and {b_array.shape[0]}."
        )
    if a_array.size == 0:
        raise ValueError("Expected at least one paired sample.")
    if num_resamples <= 0:
        raise ValueError(f"num_resamples must be positive, got {num_resamples}.")

    deltas = b_array - a_array
    observed = float(np.mean(deltas))
    if np.isclose(observed, 0.0):
        return 1.0

    rng = np.random.default_rng(seed)
    extreme_count = 0
    for _ in range(num_resamples):
        signs = rng.choice(np.asarray([-1.0, 1.0]), size=deltas.size, replace=True)
        permuted = float(np.mean(deltas * signs))
        if abs(permuted) >= abs(observed):
            extreme_count += 1

    # Add-one smoothing prevents returning exactly zero.
    return float((extreme_count + 1) / (num_resamples + 1))
