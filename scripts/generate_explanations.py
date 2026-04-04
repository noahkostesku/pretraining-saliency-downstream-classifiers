"""Generate Stage-5 explanation maps for Stage-4 checkpoints."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.explain import discover_stage4_runs, generate_explanations_for_runs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["supervised", "moco", "swav", "random_init"],
        choices=["supervised", "moco", "swav", "random_init"],
        help="Condition names to include.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["gradcam", "gradcampp", "occlusion"],
        choices=["gradcam", "gradcampp", "occlusion"],
        help="Explanation methods to generate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for explanation generation dataloader.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, for example 'cpu' or 'cuda'.",
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
        "--allow-remote-download",
        action="store_true",
        help="Allow MoCo/SwaV checkpoint download fallback for model loading.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing method metadata/maps for selected runs.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download STL-10 if missing.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for Stage-5 explanation generation."""
    args = _parse_args()
    runs = discover_stage4_runs(
        artifacts_root=args.artifacts_root,
        conditions=args.conditions,
        seeds=args.seeds,
    )
    rows = generate_explanations_for_runs(
        runs=runs,
        methods=args.methods,
        batch_size=args.batch_size,
        data_root=args.data_root,
        artifacts_root=args.artifacts_root,
        device=args.device,
        allow_remote_download=args.allow_remote_download,
        overwrite=args.overwrite,
        download=args.download,
    )
    print(f"Generated explanation rows: {len(rows)}")


if __name__ == "__main__":
    main()
