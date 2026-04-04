"""Save and load HxW saliency maps and metadata."""

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cv.utils.io import ensure_parent, read_json, write_json


def resize_saliency_map(
    saliency: torch.Tensor,
    *,
    image_size: tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Resize one saliency map to image resolution."""
    if saliency.ndim != 2:
        raise ValueError(
            f"Expected 2-D saliency map, got shape {tuple(saliency.shape)}"
        )

    resized = F.interpolate(
        saliency.unsqueeze(0).unsqueeze(0),
        size=image_size,
        mode="bilinear",
        align_corners=False,
    )
    return resized.squeeze(0).squeeze(0)


def normalize_saliency_map(saliency: torch.Tensor) -> torch.Tensor:
    """Normalize one saliency map to ``[0, 1]`` with constant-map fallback."""
    if saliency.ndim != 2:
        raise ValueError(
            f"Expected 2-D saliency map, got shape {tuple(saliency.shape)}"
        )

    saliency = saliency.float()
    min_value = torch.min(saliency)
    max_value = torch.max(saliency)
    denominator = max_value - min_value
    is_constant = torch.isclose(denominator, torch.tensor(0.0, device=saliency.device))
    if bool(is_constant.item()):
        return torch.zeros_like(saliency)
    return (saliency - min_value) / denominator


def normalize_saliency_batch(saliency_batch: torch.Tensor) -> torch.Tensor:
    """Normalize a batch of saliency maps to ``[0, 1]`` per sample."""
    if saliency_batch.ndim != 3:
        raise ValueError(
            "Expected saliency batch with shape [batch, height, width], "
            f"got {tuple(saliency_batch.shape)}"
        )

    normalized = [
        normalize_saliency_map(saliency_batch[i])
        for i in range(saliency_batch.shape[0])
    ]
    return torch.stack(normalized, dim=0)


def validate_saliency_array(
    saliency: np.ndarray,
    *,
    expected_size: tuple[int, int] = (224, 224),
) -> None:
    """Validate persisted saliency shape and numeric range."""
    if saliency.shape != expected_size:
        raise ValueError(
            f"Expected saliency shape {expected_size}, got {tuple(saliency.shape)}"
        )
    if not np.isfinite(saliency).all():
        raise ValueError("Saliency map contains NaN or Inf values.")
    if saliency.min() < 0.0 or saliency.max() > 1.0:
        raise ValueError(
            "Saliency map values must be within [0, 1], "
            f"got min={saliency.min():.6f}, max={saliency.max():.6f}"
        )


def save_saliency_map(
    path: str | Path,
    saliency: torch.Tensor | np.ndarray,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save one normalized ``HxW`` saliency map and optional sidecar metadata."""
    destination = Path(path)
    ensure_parent(destination)

    if isinstance(saliency, torch.Tensor):
        payload = saliency.detach().cpu().numpy().astype(np.float32)
    else:
        payload = np.asarray(saliency, dtype=np.float32)

    validate_saliency_array(payload)
    np.save(destination, payload)

    if metadata is not None:
        metadata_path = destination.with_suffix(".json")
        write_json(metadata_path, metadata)

    return destination


def load_saliency_map(path: str | Path) -> np.ndarray:
    """Load one persisted saliency map from ``.npy``."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Missing saliency map: {source}")

    payload = np.load(source)
    payload = np.asarray(payload, dtype=np.float32)
    validate_saliency_array(payload)
    return payload


def write_saliency_metadata(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """Write per-image saliency metadata rows as JSON."""
    destination = Path(path)
    write_json(destination, rows)
    return destination


def read_saliency_metadata(path: str | Path) -> list[dict[str, Any]]:
    """Load per-image saliency metadata rows from JSON."""
    payload = read_json(Path(path))
    if not isinstance(payload, list):
        raise ValueError("Saliency metadata must be a JSON list of row objects.")
    return payload
