"""Supervised ResNet-50 checkpoint loader."""

import torch
from torch import nn
from torchvision import models

from cv.config import DEFAULT_ENCODER_CHECKPOINTS

from .wrapper import EncoderMetadata, EncoderWrapper, LoadedEncoder, PreprocessConfig


def load_supervised_encoder(
    *,
    weight_enum: str | None = None,
    freeze: bool = True,
    device: torch.device | str | None = None,
) -> LoadedEncoder:
    """Load the supervised ImageNet-pretrained ResNet-50 encoder."""
    if weight_enum is None:
        weight_enum = DEFAULT_ENCODER_CHECKPOINTS.supervised_weight_enum

    try:
        weight_value = getattr(models.ResNet50_Weights, weight_enum)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown torchvision ResNet-50 weight enum '{weight_enum}'."
        ) from exc

    backbone = models.resnet50(weights=weight_value)
    backbone.fc = nn.Identity()
    wrapper = EncoderWrapper(backbone, feature_dim=2048)

    if freeze:
        wrapper.freeze(set_eval=True)

    if device is not None:
        wrapper = wrapper.to(device)

    metadata = EncoderMetadata(
        condition="supervised",
        architecture="ResNet-50",
        pretraining_source="ImageNet-1K",
        pretraining_objective="supervised classification",
        checkpoint_origin="torchvision",
        checkpoint_id=f"ResNet50_Weights.{weight_enum}",
        feature_dim=2048,
        gradcam_target_layer="encoder.layer4[-1]",
        notes="Torchvision supervised baseline checkpoint.",
    )

    return LoadedEncoder(
        encoder=wrapper,
        preprocess_config=PreprocessConfig(),
        metadata=metadata,
    )
