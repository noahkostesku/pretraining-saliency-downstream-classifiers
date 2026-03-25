"""Prepare local encoder checkpoints for stage-1 loading on fresh clones."""

import argparse
from pathlib import Path
import subprocess
import sys

from torch.hub import download_url_to_file

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.config import DEFAULT_ENCODER_CHECKPOINTS
from cv.encoders import load_encoder
from cv.utils.io import ensure_parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["supervised", "moco", "swav"],
        help="Encoder conditions to prepare.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload MoCo/SwaV checkpoints even if local files exist.",
    )
    parser.add_argument(
        "--skip-inspect",
        action="store_true",
        help="Skip running scripts/inspect_encoders.py after preparation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used when warming/loading encoders for validation.",
    )
    return parser.parse_args()


def _download_checkpoint(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and not force:
        print(f"[skip] {destination} already exists")
        return

    ensure_parent(destination)
    print(f"[download] {url} -> {destination}")
    try:
        download_url_to_file(url, str(destination), progress=True)
    except Exception:
        if destination.exists():
            destination.unlink()
        raise


def _prepare_self_supervised_checkpoints(
    *, conditions: list[str], force_download: bool
) -> None:
    config = DEFAULT_ENCODER_CHECKPOINTS

    if "moco" in conditions:
        _download_checkpoint(
            config.moco_checkpoint_url,
            config.moco_checkpoint_path,
            force=force_download,
        )

    if "swav" in conditions:
        _download_checkpoint(
            config.swav_checkpoint_url,
            config.swav_checkpoint_path,
            force=force_download,
        )


def _warm_supervised_if_requested(*, conditions: list[str], device: str) -> None:
    if "supervised" not in conditions:
        return

    print("[prepare] warming supervised torchvision weights")
    load_encoder("supervised", freeze=True, device=device)


def _run_inspection(*, conditions: list[str], device: str) -> None:
    config = DEFAULT_ENCODER_CHECKPOINTS

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "inspect_encoders.py"),
        "--conditions",
        *conditions,
        "--device",
        device,
    ]

    if "moco" in conditions:
        command.extend(["--moco-checkpoint", str(config.moco_checkpoint_path)])
    if "swav" in conditions:
        command.extend(["--swav-checkpoint", str(config.swav_checkpoint_path)])

    print("[inspect] running inspect_encoders.py")
    subprocess.run(command, check=True)


def main() -> None:
    """Prepare encoder checkpoints and run stage-1 sanity checks."""
    args = _parse_args()
    conditions = args.conditions

    _prepare_self_supervised_checkpoints(
        conditions=conditions,
        force_download=args.force_download,
    )
    _warm_supervised_if_requested(conditions=conditions, device=args.device)

    if not args.skip_inspect:
        _run_inspection(conditions=conditions, device=args.device)

    print("[done] encoder preparation complete")


if __name__ == "__main__":
    main()
