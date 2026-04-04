"""Fixed explanation evaluation subset creation and reuse."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cv.config import ARTIFACTS_ROOT
from cv.data.stl10 import extract_stl10_labels, load_stl10_split
from cv.utils.io import read_json, write_json

DEFAULT_EVAL_SUBSET_SEED = 42
DEFAULT_IMAGES_PER_CLASS = 20


@dataclass(frozen=True)
class EvalSubsetArtifacts:
    """Materialized fixed evaluation subset artifacts for explanations."""

    indices: list[int]
    metadata: dict[str, Any]
    indices_path: Path
    metadata_path: Path


def _build_eval_subset_paths(
    artifacts_root: str | Path | None = None,
) -> tuple[Path, Path]:
    root = Path(artifacts_root) if artifacts_root is not None else ARTIFACTS_ROOT
    split_root = root / "splits"
    return (
        split_root / "stl10_eval_subset_indices.json",
        split_root / "stl10_eval_subset_metadata.json",
    )


def _counts_by_class(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    selected = labels[indices]
    unique, counts = np.unique(selected, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(unique, counts)}


def create_eval_subset(
    *,
    data_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    subset_seed: int = DEFAULT_EVAL_SUBSET_SEED,
    images_per_class: int = DEFAULT_IMAGES_PER_CLASS,
    overwrite: bool = False,
    download: bool = False,
) -> EvalSubsetArtifacts:
    """Create and persist the shared explanation evaluation subset."""
    indices_path, metadata_path = _build_eval_subset_paths(artifacts_root)
    subset_files_exist = indices_path.exists() and metadata_path.exists()
    if subset_files_exist and not overwrite:
        return load_eval_subset(artifacts_root=artifacts_root)

    if images_per_class <= 0:
        raise ValueError(f"images_per_class must be positive, got {images_per_class}.")

    test_dataset = load_stl10_split("test", data_root=data_root, download=download)
    labels = extract_stl10_labels(test_dataset)
    num_classes = int(np.unique(labels).size)

    rng = np.random.default_rng(subset_seed)
    selected_indices: list[int] = []
    for class_index in range(num_classes):
        class_indices = np.where(labels == class_index)[0]
        if class_indices.size < images_per_class:
            raise ValueError(
                "Not enough samples for class "
                f"{class_index}: requested {images_per_class}, found {class_indices.size}."
            )

        sampled = rng.choice(class_indices, size=images_per_class, replace=False)
        selected_indices.extend(int(index) for index in sampled.tolist())

    indices = sorted(selected_indices)
    np_indices = np.asarray(indices, dtype=np.int64)

    metadata: dict[str, Any] = {
        "dataset": "STL10",
        "source_split": "test",
        "num_classes": num_classes,
        "subset_seed": subset_seed,
        "images_per_class": images_per_class,
        "subset_count": len(indices),
        "class_counts": _counts_by_class(labels, np_indices),
        "note": (
            "Fixed explanation subset sampled once from STL-10 test split and "
            "reused across all conditions and seeds."
        ),
    }

    write_json(indices_path, indices)
    write_json(metadata_path, metadata)

    return EvalSubsetArtifacts(
        indices=indices,
        metadata=metadata,
        indices_path=indices_path,
        metadata_path=metadata_path,
    )


def load_eval_subset(
    *,
    artifacts_root: str | Path | None = None,
) -> EvalSubsetArtifacts:
    """Load the shared explanation evaluation subset."""
    indices_path, metadata_path = _build_eval_subset_paths(artifacts_root)
    missing_paths = [
        path for path in (indices_path, metadata_path) if not path.exists()
    ]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(
            "Missing explanation subset artifacts. Run scripts/export_eval_subset.py first. "
            f"Missing: {missing}"
        )

    indices_payload = read_json(indices_path)
    metadata_payload = read_json(metadata_path)

    if not isinstance(indices_payload, list):
        raise ValueError("Explanation subset indices artifact must be a JSON list.")
    if not isinstance(metadata_payload, dict):
        raise ValueError("Explanation subset metadata artifact must be a JSON object.")

    indices = [int(index) for index in indices_payload]
    return EvalSubsetArtifacts(
        indices=indices,
        metadata=metadata_payload,
        indices_path=indices_path,
        metadata_path=metadata_path,
    )
