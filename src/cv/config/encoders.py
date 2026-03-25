"""Encoder checkpoint configuration for stage-1 loading."""

from dataclasses import dataclass
from pathlib import Path

from .base import EXTERNAL_CHECKPOINTS_ROOT


SUPERVISED_WEIGHT_ENUM = "IMAGENET1K_V2"
SUPERVISED_WEIGHT_ID = "ResNet50_Weights.IMAGENET1K_V2"

MOCO_V2_800EP_URL = (
    "https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
    "moco_v2_800ep_pretrain.pth.tar"
)
SWAV_800EP_URL = (
    "https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar"
)


@dataclass(frozen=True)
class EncoderCheckpointConfig:
    """Default checkpoint identifiers and local paths for stage 1."""

    supervised_weight_enum: str = SUPERVISED_WEIGHT_ENUM
    supervised_weight_id: str = SUPERVISED_WEIGHT_ID
    moco_checkpoint_path: Path = (
        EXTERNAL_CHECKPOINTS_ROOT / "moco_v2_800ep_pretrain.pth.tar"
    )
    swav_checkpoint_path: Path = (
        EXTERNAL_CHECKPOINTS_ROOT / "swav_800ep_pretrain.pth.tar"
    )
    moco_checkpoint_url: str = MOCO_V2_800EP_URL
    swav_checkpoint_url: str = SWAV_800EP_URL
    allow_remote_download: bool = False


DEFAULT_ENCODER_CHECKPOINTS = EncoderCheckpointConfig()
