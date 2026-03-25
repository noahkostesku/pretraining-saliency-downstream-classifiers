"""Sanity-check encoder loading and pooled feature shapes."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.config import ARTIFACTS_ROOT
from cv.encoders import load_encoder
from cv.utils.io import ensure_parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["supervised", "moco", "swav"],
        help="Encoder condition names to inspect.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Synthetic batch size for feature-shape validation.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device string, for example 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "encoder_prep_report.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--moco-checkpoint",
        type=str,
        default=None,
        help="Optional local path for MoCo checkpoint.",
    )
    parser.add_argument(
        "--swav-checkpoint",
        type=str,
        default=None,
        help="Optional local path for SwaV checkpoint.",
    )
    parser.add_argument(
        "--allow-remote-download",
        action="store_true",
        help="Allow downloading MoCo/SwaV checkpoints when local files are missing.",
    )
    return parser.parse_args()


def _validate_condition(
    condition: str,
    *,
    batch_size: int,
    device: torch.device,
    checkpoint_path: str | None = None,
    allow_remote_download: bool = False,
) -> dict[str, object]:
    loader_kwargs: dict[str, object] = {"freeze": True, "device": device}
    if checkpoint_path is not None:
        loader_kwargs["checkpoint_path"] = checkpoint_path
    if condition in {"moco", "swav"}:
        loader_kwargs["allow_remote_download"] = allow_remote_download

    loaded = load_encoder(condition, **loader_kwargs)

    images = torch.randn(batch_size, 3, 224, 224, device=device)
    with torch.no_grad():
        features = loaded.encoder(images)

    if tuple(features.shape) != (batch_size, loaded.metadata.feature_dim):
        raise ValueError(
            f"Unexpected feature shape for '{condition}': {tuple(features.shape)}"
        )

    gradcam_layer = loaded.encoder.gradcam_target_layer
    encoder_requires_grad = any(
        param.requires_grad for param in loaded.encoder.parameters()
    )

    return {
        "condition": condition,
        "metadata": asdict(loaded.metadata),
        "preprocess_config": asdict(loaded.preprocess_config),
        "checks": {
            "feature_shape": list(features.shape),
            "feature_dim": loaded.encoder.feature_dim,
            "gradcam_layer_type": gradcam_layer.__class__.__name__,
            "encoder_frozen": not encoder_requires_grad,
        },
    }


def main() -> None:
    """Entrypoint for encoder inspection."""
    args = _parse_args()
    device = torch.device(args.device)

    results: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []

    for condition in args.conditions:
        checkpoint_path = None
        if condition == "moco":
            checkpoint_path = args.moco_checkpoint
        elif condition == "swav":
            checkpoint_path = args.swav_checkpoint

        try:
            result = _validate_condition(
                condition,
                batch_size=args.batch_size,
                device=device,
                checkpoint_path=checkpoint_path,
                allow_remote_download=args.allow_remote_download,
            )
            results.append(result)
        except Exception as exc:
            failures.append({"condition": condition, "error": str(exc)})

    report = {
        "stage": "stage-1-encoder-preparation",
        "device": str(device),
        "results": results,
        "failures": failures,
    }

    ensure_parent(args.output)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for result in results:
        checks = result["checks"]
        print(
            f"[{result['condition']}] feature_shape={checks['feature_shape']} "
            f"gradcam_layer={checks['gradcam_layer_type']} "
            f"frozen={checks['encoder_frozen']}"
        )

    for failure in failures:
        print(f"[{failure['condition']}] failed: {failure['error']}")

    print(f"Saved encoder prep report to: {args.output}")

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
