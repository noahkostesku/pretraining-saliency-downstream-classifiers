"""Accuracy evaluation helpers for validation and test."""

import contextlib
from collections.abc import Iterable

import torch
from torch import nn
from torch.amp import autocast

from .metrics import top1_num_correct


def evaluate_model(
    *,
    model: nn.Module,
    dataloader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, float]:
    """Evaluate a model over one dataloader and return scalar metrics."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if device.type == "cuda":
                forward_ctx = autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                    enabled=amp_enabled,
                )
            else:
                forward_ctx = contextlib.nullcontext()

            with forward_ctx:
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.shape[0]
            batch_correct = top1_num_correct(logits, targets)

            total_examples += batch_size
            total_correct += batch_correct
            total_loss += float(loss.item()) * batch_size

    if total_examples == 0:
        raise ValueError("Dataloader is empty; cannot evaluate metrics.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": float(total_examples),
    }
