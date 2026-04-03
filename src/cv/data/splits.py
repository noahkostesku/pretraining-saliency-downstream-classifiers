"""Fixed stratified split creation and loading."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from cv.config import ARTIFACTS_ROOT
from cv.data.stl10 import extract_stl10_labels, load_stl10_split
from cv.utils.io import read_json, write_json

DEFAULT_SPLIT_SEED = 42
DEFAULT_VAL_RATIO = 0.2


@dataclass(frozen=True)
class FixedSplitArtifacts:
    """Materialized fixed train/validation split artifacts."""

    train_indices: list[int]
    val_indices: list[int]
    metadata: dict[str, Any]
    train_indices_path: Path
    val_indices_path: Path
    metadata_path: Path


def _build_split_paths(
    artifacts_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    root = Path(artifacts_root) if artifacts_root is not None else ARTIFACTS_ROOT
    split_root = root / "splits"
    return (
        split_root / "stl10_train_indices.json",
        split_root / "stl10_val_indices.json",
        split_root / "stl10_split_metadata.json",
    )


def _counts_by_class(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    selected = labels[indices]
    unique, counts = np.unique(selected, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def create_fixed_split_indices(
    *,
    data_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    split_seed: int = DEFAULT_SPLIT_SEED,
    val_ratio: float = DEFAULT_VAL_RATIO,
    overwrite: bool = False,
    download: bool = False,
) -> FixedSplitArtifacts:
    """Create and persist the fixed stratified STL-10 train/validation split."""
    train_indices_path, val_indices_path, metadata_path = _build_split_paths(
        artifacts_root
    )
    split_files_exist = (
        train_indices_path.exists()
        and val_indices_path.exists()
        and metadata_path.exists()
    )
    if split_files_exist and not overwrite:
        return load_fixed_split_indices(artifacts_root=artifacts_root)

    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    train_dataset = load_stl10_split("train", data_root=data_root, download=download)
    labels = extract_stl10_labels(train_dataset)
    all_indices = np.arange(len(labels), dtype=np.int64)

    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=val_ratio,
        random_state=split_seed,
        shuffle=True,
        stratify=labels,
    )

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    train_indices = [int(index) for index in train_idx.tolist()]
    val_indices = [int(index) for index in val_idx.tolist()]

    metadata: dict[str, Any] = {
        "dataset": "STL10",
        "source_split": "train",
        "num_classes": 10,
        "split_seed": split_seed,
        "stratified": True,
        "train_ratio": float(1.0 - val_ratio),
        "val_ratio": float(val_ratio),
        "train_count": len(train_indices),
        "val_count": len(val_indices),
        "class_counts": {
            "train": _counts_by_class(labels, train_idx),
            "val": _counts_by_class(labels, val_idx),
        },
        "note": (
            "Fixed 80/20 stratified split sampled once with "
            f"seed {split_seed} and reused across all conditions and seeds."
        ),
    }

    write_json(train_indices_path, train_indices)
    write_json(val_indices_path, val_indices)
    write_json(metadata_path, metadata)

    return FixedSplitArtifacts(
        train_indices=train_indices,
        val_indices=val_indices,
        metadata=metadata,
        train_indices_path=train_indices_path,
        val_indices_path=val_indices_path,
        metadata_path=metadata_path,
    )


def load_fixed_split_indices(
    *,
    artifacts_root: str | Path | None = None,
) -> FixedSplitArtifacts:
    """Load saved train/validation indices."""
    train_indices_path, val_indices_path, metadata_path = _build_split_paths(
        artifacts_root
    )

    missing_paths = [
        path
        for path in (train_indices_path, val_indices_path, metadata_path)
        if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing split artifacts. Run scripts/make_splits.py first. "
            f"Missing: {missing}"
        )

    train_indices_payload = read_json(train_indices_path)
    val_indices_payload = read_json(val_indices_path)
    metadata_payload = read_json(metadata_path)

    if not isinstance(train_indices_payload, list) or not isinstance(
        val_indices_payload, list
    ):
        raise ValueError("Train/val split artifacts must be JSON arrays of indices.")
    if not isinstance(metadata_payload, dict):
        raise ValueError("Split metadata artifact must be a JSON object.")

    train_indices = [int(index) for index in train_indices_payload]
    val_indices = [int(index) for index in val_indices_payload]

    return FixedSplitArtifacts(
        train_indices=train_indices,
        val_indices=val_indices,
        metadata=metadata_payload,
        train_indices_path=train_indices_path,
        val_indices_path=val_indices_path,
        metadata_path=metadata_path,
    )
