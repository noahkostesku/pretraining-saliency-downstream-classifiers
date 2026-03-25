"""Unified encoder wrapper exposing pooled 2048-D features."""

from dataclasses import asdict, dataclass

import torch
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    """Shared preprocessing config across all encoder conditions."""

    image_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD
    interpolation: str = "bilinear"


@dataclass(frozen=True)
class EncoderMetadata:
    """Checkpoint provenance and structural metadata for one condition."""

    condition: str
    architecture: str
    pretraining_source: str
    pretraining_objective: str
    checkpoint_origin: str
    checkpoint_id: str
    feature_dim: int
    gradcam_target_layer: str
    notes: str = ""


@dataclass(frozen=True)
class LoadedEncoder:
    """Return payload for stage-1 encoder loading API."""

    encoder: "EncoderWrapper"
    preprocess_config: PreprocessConfig
    metadata: EncoderMetadata

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable payload for reports."""
        return {
            "preprocess_config": asdict(self.preprocess_config),
            "metadata": asdict(self.metadata),
        }


class EncoderWrapper(nn.Module):
    """Wrap a pretrained encoder behind a common pooled-feature interface."""

    def __init__(self, encoder: nn.Module, *, feature_dim: int = 2048) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map input images to pooled encoder features."""
        features = self.encoder(images)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"Expected features with shape [batch, {self.feature_dim}], "
                f"got {tuple(features.shape)}"
            )
        return features

    def freeze(self, *, set_eval: bool = True) -> None:
        """Freeze encoder parameters for linear-probe training."""
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        if set_eval:
            self.encoder.eval()

    @property
    def gradcam_target_layer(self) -> nn.Module:
        """Return target layer used by Grad-CAM in later stages."""
        return self.encoder.layer4[-1]
