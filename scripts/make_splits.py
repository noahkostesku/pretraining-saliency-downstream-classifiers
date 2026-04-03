"""Create and save the fixed STL-10 train/validation split indices."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.data.splits import create_fixed_split_indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed used for the fixed 80/20 stratified split.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio applied to the STL-10 train split.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate split files even if existing artifacts are found.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download STL-10 if missing under data/raw/.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for materializing split artifacts."""
    args = _parse_args()
    artifacts = create_fixed_split_indices(
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
        download=args.download,
    )

    print(
        "Saved fixed split artifacts:\n"
        f"- train indices: {artifacts.train_indices_path}\n"
        f"- val indices: {artifacts.val_indices_path}\n"
        f"- metadata: {artifacts.metadata_path}"
    )
    print(
        "Counts: "
        f"train={artifacts.metadata['train_count']} "
        f"val={artifacts.metadata['val_count']}"
    )
    print(
        "Per-class counts: "
        f"train={artifacts.metadata['class_counts']['train']} "
        f"val={artifacts.metadata['class_counts']['val']}"
    )


if __name__ == "__main__":
    main()
