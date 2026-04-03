"""Downstream model definitions for probing and evaluation."""

from .downstream import (
    DownstreamModeConfig,
    DownstreamModel,
    build_downstream_model,
    resolve_mode_config,
)
from .linear_probe import LinearProbeModel

__all__ = [
    "DownstreamModeConfig",
    "DownstreamModel",
    "LinearProbeModel",
    "build_downstream_model",
    "resolve_mode_config",
]
