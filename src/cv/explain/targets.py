"""Target score helpers based on the original-image predicted class."""

import torch


def get_predicted_class_target(logits: torch.Tensor) -> torch.Tensor:
    """Return original-image predicted classes as ``[batch]`` class indices."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch, classes], got {tuple(logits.shape)}"
        )
    return logits.argmax(dim=1)


def gather_target_scores(
    logits: torch.Tensor, target_classes: torch.Tensor
) -> torch.Tensor:
    """Gather one target logit per sample from ``[batch, classes]`` logits."""
    if logits.ndim != 2:
        raise ValueError(
            f"Expected logits with shape [batch, classes], got {tuple(logits.shape)}"
        )
    if target_classes.ndim != 1:
        raise ValueError(
            "Expected target_classes with shape [batch], "
            f"got {tuple(target_classes.shape)}"
        )
    if logits.shape[0] != target_classes.shape[0]:
        raise ValueError(
            "Batch mismatch between logits and target_classes: "
            f"{logits.shape[0]} != {target_classes.shape[0]}"
        )

    batch_indices = torch.arange(logits.shape[0], device=logits.device)
    return logits[batch_indices, target_classes]


def predicted_class_scores(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted classes and corresponding logits for each sample."""
    predicted_classes = get_predicted_class_target(logits)
    predicted_scores = gather_target_scores(logits, predicted_classes)
    return predicted_classes, predicted_scores
