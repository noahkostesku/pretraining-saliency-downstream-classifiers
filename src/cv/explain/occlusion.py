"""Occlusion map generation with a fixed masking protocol."""

import torch
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from .saliency_io import normalize_saliency_batch
from .targets import gather_target_scores, get_predicted_class_target


def _build_patch_grid(
    *,
    image_height: int,
    image_width: int,
    patch_size: int,
    stride: int,
) -> list[tuple[int, int]]:
    ys = list(range(0, image_height - patch_size + 1, stride))
    xs = list(range(0, image_width - patch_size + 1, stride))
    return [(y, x) for y in ys for x in xs]


def generate_occlusion_map(
    *,
    model: nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor | None = None,
    patch_size: int = 16,
    stride: int = 16,
    blur_kernel_size: int = 21,
    blur_sigma: float = 5.0,
    occlusion_batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate occlusion saliency maps for a batch.

    Returns ``(saliency_maps, target_classes, logits)``.
    """
    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape [batch, channels, height, width], "
            f"got {tuple(images.shape)}"
        )
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive integers.")
    if blur_kernel_size % 2 == 0:
        raise ValueError("blur_kernel_size must be odd for gaussian_blur.")

    device = next(model.parameters()).device
    images = images.to(device)

    model.eval()
    with torch.no_grad():
        original_logits = model(images)

    if target_classes is None:
        target_classes = get_predicted_class_target(original_logits)
    else:
        target_classes = target_classes.to(device)

    original_scores = gather_target_scores(original_logits, target_classes)
    blurred_images = gaussian_blur(
        images,
        kernel_size=[blur_kernel_size, blur_kernel_size],
        sigma=[blur_sigma, blur_sigma],
    )

    _, _, image_height, image_width = images.shape
    patch_grid = _build_patch_grid(
        image_height=image_height,
        image_width=image_width,
        patch_size=patch_size,
        stride=stride,
    )

    maps = torch.zeros(
        (images.shape[0], image_height, image_width),
        device=device,
        dtype=torch.float32,
    )

    for image_index in range(images.shape[0]):
        source = images[image_index : image_index + 1]
        baseline = blurred_images[image_index : image_index + 1]
        target_class = target_classes[image_index : image_index + 1]
        target_score = original_scores[image_index].item()

        patch_scores: list[float] = []
        for chunk_start in range(0, len(patch_grid), occlusion_batch_size):
            chunk = patch_grid[chunk_start : chunk_start + occlusion_batch_size]
            occluded_batch = source.repeat(len(chunk), 1, 1, 1)

            for patch_idx, (y, x) in enumerate(chunk):
                occluded_batch[
                    patch_idx,
                    :,
                    y : y + patch_size,
                    x : x + patch_size,
                ] = baseline[0, :, y : y + patch_size, x : x + patch_size]

            with torch.no_grad():
                occluded_logits = model(occluded_batch)

            repeated_targets = target_class.repeat(len(chunk))
            occluded_scores = gather_target_scores(occluded_logits, repeated_targets)
            score_drops = target_score - occluded_scores
            patch_scores.extend(float(value) for value in score_drops.detach().cpu())

        if len(patch_scores) != len(patch_grid):
            raise RuntimeError(
                "Patch score count mismatch during occlusion generation."
            )

        map_tensor = torch.zeros((image_height, image_width), device=device)
        for (y, x), patch_score in zip(patch_grid, patch_scores):
            map_tensor[y : y + patch_size, x : x + patch_size] = patch_score
        maps[image_index] = map_tensor

    maps = normalize_saliency_batch(maps)
    return maps.detach(), target_classes.detach(), original_logits.detach()
