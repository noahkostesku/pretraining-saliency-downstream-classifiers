"""STL-10 dataset loading and transform wiring."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import STL10

from cv.config import DATA_ROOT
from cv.transforms import build_eval_transform, build_train_transform

VALID_STL10_SPLITS = {"train", "test", "unlabeled", "train+unlabeled"}


@dataclass(frozen=True)
class DownstreamDatasets:
    """Datasets for one downstream training run."""

    train: Subset
    val: Subset
    test: STL10


def load_stl10_split(
    split: str,
    *,
    transform=None,
    data_root: str | Path | None = None,
    download: bool = False,
) -> STL10:
    """Load one STL-10 split with the configured transform."""
    if split not in VALID_STL10_SPLITS:
        valid = ", ".join(sorted(VALID_STL10_SPLITS))
        raise ValueError(f"Unknown STL-10 split '{split}'. Valid values: {valid}")

    root = Path(data_root) if data_root is not None else DATA_ROOT / "raw"
    return STL10(
        root=str(root),
        split=split,
        transform=transform,
        download=download,
    )


def extract_stl10_labels(dataset: STL10) -> np.ndarray:
    """Return STL-10 labels as an integer NumPy array."""
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise ValueError("STL-10 dataset has no 'labels' attribute for this split.")

    labels_array = np.asarray(labels, dtype=np.int64)
    if labels_array.ndim != 1:
        raise ValueError(f"Expected 1-D label array, got shape {labels_array.shape}")

    return labels_array


def build_downstream_datasets(
    *,
    train_indices: list[int],
    val_indices: list[int],
    data_root: str | Path | None = None,
    download: bool = False,
) -> DownstreamDatasets:
    """Build train/val/test datasets with fixed indices and shared transforms."""
    train_dataset = load_stl10_split(
        "train",
        transform=build_train_transform(),
        data_root=data_root,
        download=download,
    )
    val_dataset = load_stl10_split(
        "train",
        transform=build_eval_transform(),
        data_root=data_root,
        download=download,
    )
    test_dataset = load_stl10_split(
        "test",
        transform=build_eval_transform(),
        data_root=data_root,
        download=download,
    )

    return DownstreamDatasets(
        train=Subset(train_dataset, train_indices),
        val=Subset(val_dataset, val_indices),
        test=test_dataset,
    )
