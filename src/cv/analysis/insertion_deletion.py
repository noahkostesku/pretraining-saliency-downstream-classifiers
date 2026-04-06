"""Shared perturbation implementation for insertion/deletion faithfulness."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from .curves import (
    build_curve_fraction_axis,
    build_patch_slices,
    rank_patches_by_saliency,
)


@dataclass(frozen=True)
class InsertionDeletionResult:
    """Per-image insertion/deletion outputs under one fixed perturbation protocol."""

    x: np.ndarray
    insertion_scores: np.ndarray
    deletion_scores: np.ndarray
    insertion_pred_classes: np.ndarray
    deletion_pred_classes: np.ndarray
    target_class: int
    target_logit_original: float
    num_patches: int
    drop_top10: float
    drop_top20: float
    flip_at_top10: bool
    flip_at_top20: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "x": [float(value) for value in self.x.tolist()],
            "insertion_scores": [
                float(value) for value in self.insertion_scores.tolist()
            ],
            "deletion_scores": [
                float(value) for value in self.deletion_scores.tolist()
            ],
            "insertion_pred_classes": [
                int(value) for value in self.insertion_pred_classes.tolist()
            ],
            "deletion_pred_classes": [
                int(value) for value in self.deletion_pred_classes.tolist()
            ],
            "target_class": int(self.target_class),
            "target_logit_original": float(self.target_logit_original),
            "num_patches": int(self.num_patches),
            "drop_top10": float(self.drop_top10),
            "drop_top20": float(self.drop_top20),
            "flip_at_top10": bool(self.flip_at_top10),
            "flip_at_top20": bool(self.flip_at_top20),
        }


def _resolve_device(
    model: nn.Module, device: torch.device | str | None
) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _validate_image(image: torch.Tensor) -> None:
    if image.ndim != 3:
        raise ValueError(
            "Expected image with shape [channels, height, width], "
            f"got {tuple(image.shape)}"
        )


def _validate_saliency(
    saliency: np.ndarray,
    *,
    expected_height: int,
    expected_width: int,
) -> None:
    if saliency.ndim != 2:
        raise ValueError(
            f"Expected 2-D saliency map, got shape {tuple(saliency.shape)}"
        )
    if saliency.shape != (expected_height, expected_width):
        raise ValueError(
            "Saliency shape does not match image shape. "
            f"Expected {(expected_height, expected_width)}, got {tuple(saliency.shape)}"
        )
    if not np.isfinite(saliency).all():
        raise ValueError("Saliency map contains NaN or Inf values.")


def _score_target_logits_for_states(
    *,
    model: nn.Module,
    states: list[torch.Tensor],
    target_class: int,
    device: torch.device,
    eval_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if eval_batch_size <= 0:
        raise ValueError(f"eval_batch_size must be positive, got {eval_batch_size}.")

    target_scores_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), eval_batch_size):
            end = min(start + eval_batch_size, len(states))
            batch = torch.stack(states[start:end], dim=0).to(device, non_blocking=True)
            logits = model(batch)
            if logits.ndim != 2:
                raise ValueError(
                    "Expected model logits with shape [batch, classes], "
                    f"got {tuple(logits.shape)}"
                )

            target_classes = torch.full(
                (logits.shape[0],),
                fill_value=target_class,
                device=logits.device,
                dtype=torch.long,
            )
            batch_indices = torch.arange(logits.shape[0], device=logits.device)
            probs = torch.softmax(logits, dim=1)
            target_scores = probs[batch_indices, target_classes]
            predicted_classes = logits.argmax(dim=1)

            target_scores_chunks.append(target_scores.detach().cpu().numpy())
            pred_chunks.append(predicted_classes.detach().cpu().numpy())

    target_scores_np = np.concatenate(target_scores_chunks, axis=0).astype(np.float32)
    predicted_classes_np = np.concatenate(pred_chunks, axis=0).astype(np.int64)
    return target_scores_np, predicted_classes_np


def _build_perturbation_states(
    *,
    image: torch.Tensor,
    baseline: torch.Tensor,
    ranked_patch_ids: np.ndarray,
    patch_slices: list[tuple[slice, slice]],
    mode: str,
) -> list[torch.Tensor]:
    if mode not in {"insertion", "deletion"}:
        raise ValueError(f"Unsupported mode '{mode}'.")

    if mode == "deletion":
        current = image.clone()
        source = baseline
    else:
        current = baseline.clone()
        source = image

    states = [current.clone()]
    for patch_id in ranked_patch_ids.tolist():
        y_slice, x_slice = patch_slices[int(patch_id)]
        current[:, y_slice, x_slice] = source[:, y_slice, x_slice]
        states.append(current.clone())
    return states


def run_insertion_deletion(
    *,
    model: nn.Module,
    image: torch.Tensor,
    saliency: np.ndarray,
    target_class: int,
    patch_size: int = 16,
    stride: int = 16,
    blur_kernel_size: int = 21,
    blur_sigma: float = 5.0,
    eval_batch_size: int = 64,
    device: torch.device | str | None = None,
) -> InsertionDeletionResult:
    """Run insertion/deletion curves for one image and one fixed saliency map."""
    _validate_image(image)
    _, image_height, image_width = image.shape
    saliency_np = np.asarray(saliency, dtype=np.float32)
    _validate_saliency(
        saliency_np,
        expected_height=image_height,
        expected_width=image_width,
    )

    if target_class < 0:
        raise ValueError(f"target_class must be non-negative, got {target_class}.")
    if blur_kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be odd for gaussian_blur.")

    patch_slices = build_patch_slices(
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
        stride=stride,
    )
    ranked_patch_ids = rank_patches_by_saliency(
        saliency_np,
        patch_slices=patch_slices,
    )

    torch_device = _resolve_device(model, device)
    image_dev = image.detach().to(torch_device).float()
    baseline_dev = gaussian_blur(
        image_dev,
        kernel_size=[blur_kernel_size, blur_kernel_size],
        sigma=[blur_sigma, blur_sigma],
    )

    insertion_states = _build_perturbation_states(
        image=image_dev,
        baseline=baseline_dev,
        ranked_patch_ids=ranked_patch_ids,
        patch_slices=patch_slices,
        mode="insertion",
    )
    deletion_states = _build_perturbation_states(
        image=image_dev,
        baseline=baseline_dev,
        ranked_patch_ids=ranked_patch_ids,
        patch_slices=patch_slices,
        mode="deletion",
    )

    insertion_scores, insertion_preds = _score_target_logits_for_states(
        model=model,
        states=insertion_states,
        target_class=target_class,
        device=torch_device,
        eval_batch_size=eval_batch_size,
    )
    deletion_scores, deletion_preds = _score_target_logits_for_states(
        model=model,
        states=deletion_states,
        target_class=target_class,
        device=torch_device,
        eval_batch_size=eval_batch_size,
    )

    x = build_curve_fraction_axis(num_patches=len(patch_slices))
    target_logit_original = float(deletion_scores[0])

    top10_index = max(1, int(round(0.10 * len(patch_slices))))
    top20_index = max(1, int(round(0.20 * len(patch_slices))))
    drop_top10 = float(target_logit_original - deletion_scores[top10_index])
    drop_top20 = float(target_logit_original - deletion_scores[top20_index])

    return InsertionDeletionResult(
        x=x,
        insertion_scores=insertion_scores,
        deletion_scores=deletion_scores,
        insertion_pred_classes=insertion_preds,
        deletion_pred_classes=deletion_preds,
        target_class=int(target_class),
        target_logit_original=target_logit_original,
        num_patches=len(patch_slices),
        drop_top10=drop_top10,
        drop_top20=drop_top20,
        flip_at_top10=bool(deletion_preds[top10_index] != target_class),
        flip_at_top20=bool(deletion_preds[top20_index] != target_class),
    )
