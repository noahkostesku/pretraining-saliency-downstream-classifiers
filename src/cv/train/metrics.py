"""Top-1 accuracy and seed summary helpers."""

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch


def top1_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy in ``[0, 1]`` for one batch."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch, classes], got {logits.shape}"
        )
    if targets.ndim != 1:
        raise ValueError(f"Expected targets with shape [batch], got {targets.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Logits and targets batch size mismatch: "
            f"{logits.shape[0]} != {targets.shape[0]}"
        )

    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.numel()
    if total == 0:
        raise ValueError("Cannot compute accuracy for an empty target tensor.")
    return float(correct) / float(total)


def top1_num_correct(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Return the number of top-1 correct predictions for one batch."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch, classes], got {logits.shape}"
        )
    if targets.ndim != 1:
        raise ValueError(f"Expected targets with shape [batch], got {targets.shape}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            "Logits and targets batch size mismatch: "
            f"{logits.shape[0]} != {targets.shape[0]}"
        )

    predictions = logits.argmax(dim=1)
    return int((predictions == targets).sum().item())


def mean_and_std(values: Iterable[float]) -> tuple[float, float]:
    """Return mean and sample std (ddof=1) for a sequence."""
    values_array = np.asarray(list(values), dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("Expected at least one value to summarize.")

    mean = float(np.mean(values_array))
    if values_array.size == 1:
        return mean, 0.0
    std = float(np.std(values_array, ddof=1))
    return mean, std


def summarize_test_accuracy_by_condition(
    run_rows: Iterable[dict[str, object]],
) -> list[dict[str, float | str | int]]:
    """Aggregate seed-wise test accuracy into condition-level summaries."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in run_rows:
        condition = row.get("condition")
        test_acc = row.get("test_acc")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in run row: {row}")
        if not isinstance(test_acc, (int, float)):
            raise ValueError(f"Invalid test_acc in run row: {row}")
        grouped[condition].append(float(test_acc))

    summaries: list[dict[str, float | str | int]] = []
    for condition, values in sorted(grouped.items()):
        mean, std = mean_and_std(values)
        summaries.append(
            {
                "condition": condition,
                "num_seeds": len(values),
                "mean_test_acc": mean,
                "std_test_acc": std,
            }
        )

    return summaries
