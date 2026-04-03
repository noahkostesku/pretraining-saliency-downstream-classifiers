"""Shared downstream wrapper with freeze and unfreeze controls."""

from dataclasses import dataclass

import torch
from torch import nn
from torchvision import models

from cv.encoders import EncoderWrapper, load_encoder

VALID_CONDITIONS = {"supervised", "moco", "swav", "random_init"}


@dataclass(frozen=True)
class DownstreamModeConfig:
    """Trainability controls for a downstream run."""

    freeze_encoder: bool
    trainable_layer4: bool = False


def resolve_mode_config(
    *,
    condition: str,
    training_mode: str,
) -> DownstreamModeConfig:
    """Resolve trainability settings from a condition and training mode."""
    if training_mode == "frozen_probe":
        if condition == "random_init":
            raise ValueError("'random_init' cannot use frozen_probe mode.")
        return DownstreamModeConfig(freeze_encoder=True, trainable_layer4=False)

    if training_mode == "full_train_random_init":
        if condition != "random_init":
            raise ValueError(
                "full_train_random_init mode is only valid for random_init."
            )
        return DownstreamModeConfig(freeze_encoder=False, trainable_layer4=False)

    if training_mode in {"ablation_layer4", "ablation_mode", "limited_finetune"}:
        if condition == "random_init":
            raise ValueError("ablation_layer4 is for pretrained conditions only.")
        return DownstreamModeConfig(freeze_encoder=False, trainable_layer4=True)

    valid_modes = (
        "frozen_probe, full_train_random_init, "
        "ablation_layer4/ablation_mode/limited_finetune"
    )
    raise ValueError(
        f"Unknown training_mode '{training_mode}'. Valid values: {valid_modes}"
    )


def _build_random_init_encoder() -> EncoderWrapper:
    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()
    return EncoderWrapper(backbone, feature_dim=2048)


class DownstreamModel(nn.Module):
    """Common downstream model wrapper for all encoder conditions."""

    def __init__(self, *, encoder: EncoderWrapper, num_classes: int = 10) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(self.encoder.feature_dim, num_classes)
        self._force_encoder_eval = False
        self._layer4_only_train = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map input images to class logits."""
        features = self.encoder(images)
        return self.classifier(features)

    def train(self, mode: bool = True):
        """Set module train/eval mode while enforcing frozen-encoder eval mode."""
        super().train(mode)
        if self._force_encoder_eval:
            self.encoder.eval()
        elif self._layer4_only_train and mode:
            self.encoder.eval()
            self.encoder.encoder.layer4.train()
        return self

    def configure_trainable_parameters(
        self,
        *,
        freeze_encoder: bool,
        trainable_layer4: bool = False,
    ) -> None:
        """Configure trainability for frozen probe, full train, or layer4 ablation."""
        if freeze_encoder and trainable_layer4:
            raise ValueError(
                "freeze_encoder=True is incompatible with trainable_layer4=True."
            )

        for parameter in self.classifier.parameters():
            parameter.requires_grad = True

        if freeze_encoder:
            self.encoder.freeze(set_eval=True)
            self._force_encoder_eval = True
            self._layer4_only_train = False
            return

        self._force_encoder_eval = False
        self._layer4_only_train = bool(trainable_layer4)
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False if trainable_layer4 else True

        if trainable_layer4:
            if not hasattr(self.encoder.encoder, "layer4"):
                raise ValueError("Encoder backbone does not expose 'layer4'.")
            self.encoder.eval()
            layer4 = self.encoder.encoder.layer4
            layer4.train()
            for parameter in layer4.parameters():
                parameter.requires_grad = True
            return

        self.encoder.train()


def build_downstream_model(
    *,
    condition: str,
    num_classes: int = 10,
    freeze_encoder: bool,
    trainable_layer4: bool = False,
    device: torch.device | str | None = None,
    allow_remote_download: bool | None = None,
    checkpoint_path: str | None = None,
) -> DownstreamModel:
    """Build a shared downstream model for one encoder condition."""
    if condition not in VALID_CONDITIONS:
        valid = ", ".join(sorted(VALID_CONDITIONS))
        raise ValueError(f"Unknown condition '{condition}'. Valid values: {valid}")
    if condition == "random_init" and trainable_layer4:
        raise ValueError(
            "trainable_layer4=True is only supported for pretrained encoders."
        )

    if condition == "random_init":
        encoder = _build_random_init_encoder()
    else:
        encoder_kwargs: dict[str, object] = {"freeze": False}
        if condition in {"moco", "swav"} and allow_remote_download is not None:
            encoder_kwargs["allow_remote_download"] = allow_remote_download
        if condition in {"moco", "swav"} and checkpoint_path is not None:
            encoder_kwargs["checkpoint_path"] = checkpoint_path

        loaded = load_encoder(condition, **encoder_kwargs)
        encoder = loaded.encoder

    model = DownstreamModel(encoder=encoder, num_classes=num_classes)
    model.configure_trainable_parameters(
        freeze_encoder=freeze_encoder,
        trainable_layer4=trainable_layer4,
    )

    if device is not None:
        model = model.to(device)

    return model
