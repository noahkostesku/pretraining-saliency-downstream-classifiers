"""SwaV ResNet-50 checkpoint loader."""

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.hub import load_state_dict_from_url
from torchvision import models

from cv.config import DEFAULT_ENCODER_CHECKPOINTS

from .wrapper import EncoderMetadata, EncoderWrapper, LoadedEncoder, PreprocessConfig


def _select_state_dict(payload: Mapping[str, object]) -> Mapping[str, Tensor]:
    if "state_dict" in payload:
        state_dict = payload["state_dict"]
        if isinstance(state_dict, Mapping):
            return state_dict  # type: ignore[return-value]
    return payload  # type: ignore[return-value]


def _load_checkpoint(
    checkpoint_path: str | Path | None,
    *,
    allow_remote_download: bool,
) -> tuple[Mapping[str, object], str, str]:
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_ENCODER_CHECKPOINTS.swav_checkpoint_path

    resolved_checkpoint_path = Path(checkpoint_path)
    if resolved_checkpoint_path.exists():
        checkpoint = torch.load(
            resolved_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_id = str(resolved_checkpoint_path)
        checkpoint_origin = "local file"
    elif allow_remote_download:
        try:
            checkpoint = load_state_dict_from_url(
                DEFAULT_ENCODER_CHECKPOINTS.swav_checkpoint_url,
                map_location="cpu",
                progress=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Unable to download SwaV checkpoint from public URL. "
                "Provide a local checkpoint via checkpoint_path."
            ) from exc
        checkpoint_id = DEFAULT_ENCODER_CHECKPOINTS.swav_checkpoint_url
        checkpoint_origin = "fbaipublicfiles"
    else:
        raise FileNotFoundError(
            "SwaV checkpoint was not found at configured path "
            f"'{resolved_checkpoint_path}'. Provide checkpoint_path explicitly "
            "or enable allow_remote_download."
        )

    if not isinstance(checkpoint, Mapping):
        raise ValueError("SwaV checkpoint payload must be a mapping.")

    return checkpoint, checkpoint_origin, checkpoint_id


def _remap_swav_keys(state_dict: Mapping[str, Tensor]) -> dict[str, Tensor]:
    remapped: dict[str, Tensor] = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]

        if key.startswith(("projection_head.", "prototypes.")):
            continue
        if key.startswith("fc."):
            continue

        remapped[key] = value

    if not remapped:
        raise ValueError(
            "No SwaV backbone keys were found in checkpoint. "
            "Expected keys prefixed with 'module.'"
        )

    return remapped


def load_swav_encoder(
    *,
    checkpoint_path: str | Path | None = None,
    allow_remote_download: bool | None = None,
    freeze: bool = True,
    device: torch.device | str | None = None,
) -> LoadedEncoder:
    """Load an ImageNet-pretrained SwaV ResNet-50 encoder."""
    if allow_remote_download is None:
        allow_remote_download = DEFAULT_ENCODER_CHECKPOINTS.allow_remote_download

    checkpoint, checkpoint_origin, checkpoint_id = _load_checkpoint(
        checkpoint_path,
        allow_remote_download=allow_remote_download,
    )

    state_dict = _select_state_dict(checkpoint)
    remapped = _remap_swav_keys(state_dict)

    backbone = models.resnet50(weights=None)
    backbone.fc = nn.Identity()

    incompatible = backbone.load_state_dict(remapped, strict=False)
    allowed_missing = {"fc.weight", "fc.bias"}
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected:
        raise ValueError(f"Unexpected keys in SwaV checkpoint: {sorted(unexpected)}")
    if missing:
        raise ValueError(f"Missing SwaV backbone keys: {sorted(missing)}")

    wrapper = EncoderWrapper(backbone, feature_dim=2048)

    if freeze:
        wrapper.freeze(set_eval=True)

    if device is not None:
        wrapper = wrapper.to(device)

    metadata = EncoderMetadata(
        condition="swav",
        architecture="ResNet-50",
        pretraining_source="ImageNet-1K",
        pretraining_objective="SwaV self-supervised clustering",
        checkpoint_origin=checkpoint_origin,
        checkpoint_id=checkpoint_id,
        feature_dim=2048,
        gradcam_target_layer="encoder.layer4[-1]",
        notes="Projection/prototype heads are dropped during key remapping.",
    )

    return LoadedEncoder(
        encoder=wrapper,
        preprocess_config=PreprocessConfig(),
        metadata=metadata,
    )
