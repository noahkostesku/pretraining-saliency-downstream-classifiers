"""Train one downstream run for one condition and seed."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.train.trainer import TrainingRunConfig, train_one_run


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--condition",
        required=True,
        choices=["supervised", "moco", "swav", "random_init"],
        help="Encoder condition to train.",
    )
    parser.add_argument(
        "--seed",
        required=True,
        type=int,
        help="Training seed for this run.",
    )
    parser.add_argument(
        "--recipe-id",
        default=None,
        help="Optional fixed recipe id override.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device string: 'auto', 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader worker count.",
    )
    parser.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Disable dataloader pinned memory.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Optional dataset root override.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="Optional artifacts root override.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download STL-10 if missing.",
    )
    parser.add_argument(
        "--allow-remote-download",
        action="store_true",
        help="Allow MoCo/SwaV checkpoint downloads when local files are missing.",
    )
    parser.add_argument(
        "--moco-checkpoint",
        type=str,
        default=None,
        help="Optional MoCo checkpoint path.",
    )
    parser.add_argument(
        "--swav-checkpoint",
        type=str,
        default=None,
        help="Optional SwaV checkpoint path.",
    )
    parser.add_argument(
        "--skip-sanity-checks",
        action="store_true",
        help="Skip first-batch gradient and BatchNorm sanity checks.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision (train/eval in fp32 on GPU).",
    )
    parser.add_argument(
        "--no-strict-repro",
        action="store_true",
        help=(
            "Disable strict reproducibility: allow cudnn.benchmark and CUDA AMP for speed "
            "(not bitwise reproducible)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for a single downstream training run."""
    args = _parse_args()

    config = TrainingRunConfig(
        condition=args.condition,
        seed=args.seed,
        recipe_id=args.recipe_id,
        device=args.device,
        artifacts_root=args.artifacts_root,
        data_root=args.data_root,
        download=args.download,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        allow_remote_download=args.allow_remote_download,
        moco_checkpoint_path=args.moco_checkpoint,
        swav_checkpoint_path=args.swav_checkpoint,
        sanity_checks=not args.skip_sanity_checks,
        use_amp=not args.no_amp,
        strict_reproducibility=not args.no_strict_repro,
    )

    result = train_one_run(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
