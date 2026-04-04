"""Grad-CAM generation using encoder.layer4[-1].conv3."""

import torch
import torch.nn.functional as F
from torch import nn

from .saliency_io import normalize_saliency_batch
from .targets import get_predicted_class_target, gather_target_scores


def _ensure_target_layer(model: nn.Module) -> nn.Module:
    if not hasattr(model, "encoder"):
        raise ValueError("Model must expose an 'encoder' attribute for Grad-CAM.")
    if not hasattr(model.encoder, "gradcam_target_layer"):
        raise ValueError("Model encoder must expose 'gradcam_target_layer'.")
    return model.encoder.gradcam_target_layer


def _compute_cam(
    *,
    activations: torch.Tensor,
    gradients: torch.Tensor,
    method: str,
) -> torch.Tensor:
    if method == "gradcam":
        weights = gradients.mean(dim=(2, 3), keepdim=True)
    elif method == "gradcampp":
        eps = 1e-8
        gradients_2 = gradients.pow(2)
        gradients_3 = gradients.pow(3)
        activation_sum = activations.sum(dim=(2, 3), keepdim=True)
        alpha_denom = 2.0 * gradients_2 + activation_sum * gradients_3 + eps
        alphas = gradients_2 / alpha_denom
        weights = (alphas * torch.relu(gradients)).sum(dim=(2, 3), keepdim=True)
    else:
        raise ValueError(f"Unknown Grad-CAM method '{method}'.")

    cam = torch.relu((weights * activations).sum(dim=1))
    return cam


def _generate_gradcam_impl(
    *,
    model: nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor | None,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if images.ndim != 4:
        raise ValueError(
            "Expected images with shape [batch, channels, height, width], "
            f"got {tuple(images.shape)}"
        )

    device = next(model.parameters()).device
    images = images.to(device)
    images = images.requires_grad_(True)
    target_layer = _ensure_target_layer(model)

    captured: dict[str, torch.Tensor] = {}

    def _forward_hook(_module, _inputs, output):
        captured["activations"] = output

    def _backward_hook(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(_forward_hook)
    backward_handle = target_layer.register_full_backward_hook(_backward_hook)

    model.eval()
    model.zero_grad(set_to_none=True)

    logits = model(images)
    if target_classes is None:
        target_classes = get_predicted_class_target(logits)
    else:
        target_classes = target_classes.to(device)

    target_scores = gather_target_scores(logits, target_classes)
    target_scores.sum().backward()

    forward_handle.remove()
    backward_handle.remove()

    activations = captured.get("activations")
    gradients = captured.get("gradients")
    if activations is None or gradients is None:
        raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

    cam = _compute_cam(activations=activations, gradients=gradients, method=method)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    cam = normalize_saliency_batch(cam)
    return cam.detach(), target_classes.detach(), logits.detach()


def generate_gradcam(
    *,
    model: nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Grad-CAM maps for a batch.

    Returns ``(saliency_maps, target_classes, logits)``.
    """
    return _generate_gradcam_impl(
        model=model,
        images=images,
        target_classes=target_classes,
        method="gradcam",
    )


def generate_gradcampp(
    *,
    model: nn.Module,
    images: torch.Tensor,
    target_classes: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate Grad-CAM++ maps for a batch.

    Returns ``(saliency_maps, target_classes, logits)``.
    """
    return _generate_gradcam_impl(
        model=model,
        images=images,
        target_classes=target_classes,
        method="gradcampp",
    )
