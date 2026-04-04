"""Sample and save the fixed explanation evaluation subset."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.data import (
    DEFAULT_EVAL_SUBSET_SEED,
    DEFAULT_IMAGES_PER_CLASS,
    create_eval_subset,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset-seed",
        type=int,
        default=DEFAULT_EVAL_SUBSET_SEED,
        help="Subset sampling seed.",
    )
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=DEFAULT_IMAGES_PER_CLASS,
        help="Number of test images to sample per class.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="Optional artifacts root override.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional data root override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subset artifacts.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download STL-10 if missing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for exporting the shared evaluation subset."""
    args = _parse_args()
    artifacts = create_eval_subset(
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        subset_seed=args.subset_seed,
        images_per_class=args.images_per_class,
        overwrite=args.overwrite,
        download=args.download,
    )

    print(f"Saved eval subset indices: {artifacts.indices_path}")
    print(f"Saved eval subset metadata: {artifacts.metadata_path}")
    print(f"Subset size: {len(artifacts.indices)}")
    print("Class counts:")
    print(json.dumps(artifacts.metadata.get("class_counts", {}), indent=2))


if __name__ == "__main__":
    main()
