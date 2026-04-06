"""Launch downstream training runs across conditions and seeds."""

import argparse
import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.config import ARTIFACTS_ROOT
from cv.data import build_downstream_datasets, load_fixed_split_indices
from cv.models import build_downstream_model, resolve_mode_config
from cv.train.trainer import (
    RUN_TABLE_COLUMNS,
    TrainingRunConfig,
    build_run_table_row,
    resolve_training_recipe,
    train_one_run,
)
from cv.utils.io import ensure_parent, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["supervised", "moco", "swav", "random_init"],
        choices=["supervised", "moco", "swav", "random_init"],
        help="Conditions to train.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Training seeds to execute for each condition.",
    )
    parser.add_argument(
        "--probe-recipe-id",
        default=None,
        help="Optional recipe override for pretrained frozen-probe conditions.",
    )
    parser.add_argument(
        "--random-init-recipe-id",
        default=None,
        help="Optional recipe override for random_init.",
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
        "--skip-cross-condition-check",
        action="store_true",
        help="Skip one-batch cross-condition output-shape consistency check.",
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
    parser.add_argument(
        "--run-table-json",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_runs" / "run_table.json",
        help="Path to save the run table JSON.",
    )
    parser.add_argument(
        "--run-table-csv",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_runs" / "run_table.csv",
        help="Path to save the run table CSV.",
    )
    return parser.parse_args()


def _recipe_id_for_condition(args: argparse.Namespace, condition: str) -> str | None:
    if condition == "random_init":
        return args.random_init_recipe_id
    return args.probe_recipe_id


def _cross_condition_one_batch_check(args: argparse.Namespace) -> None:
    split_artifacts = load_fixed_split_indices(artifacts_root=args.artifacts_root)
    datasets = build_downstream_datasets(
        train_indices=split_artifacts.train_indices,
        val_indices=split_artifacts.val_indices,
        data_root=args.data_root,
        download=args.download,
    )

    dataloader = DataLoader(datasets.train, batch_size=8, shuffle=False, num_workers=0)
    images, targets = next(iter(dataloader))

    labels_min = int(targets.min().item())
    labels_max = int(targets.max().item())
    if labels_min < 0 or labels_max > 9:
        raise RuntimeError(
            "Unexpected label range in STL-10 training batch: "
            f"min={labels_min}, max={labels_max}"
        )

    device = torch.device(args.device)
    images = images.to(device)

    for condition in args.conditions:
        recipe = resolve_training_recipe(
            condition=condition,
            recipe_id=_recipe_id_for_condition(args, condition),
        )
        mode_config = resolve_mode_config(
            condition=condition,
            training_mode=recipe.training_mode,
        )
        checkpoint_path = None
        if condition == "moco":
            checkpoint_path = args.moco_checkpoint
        elif condition == "swav":
            checkpoint_path = args.swav_checkpoint

        model = build_downstream_model(
            condition=condition,
            num_classes=10,
            freeze_encoder=mode_config.freeze_encoder,
            trainable_layer4=mode_config.trainable_layer4,
            device=device,
            allow_remote_download=args.allow_remote_download,
            checkpoint_path=checkpoint_path,
        )
        model.eval()
        with torch.no_grad():
            logits = model(images)

        expected_shape = (images.shape[0], 10)
        if tuple(logits.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected logits shape for {condition}: {tuple(logits.shape)} "
                f"(expected {expected_shape})"
            )


def _write_run_table(
    rows: list[dict[str, object]], *, json_path: Path, csv_path: Path
) -> None:
    ensure_parent(json_path)
    ensure_parent(csv_path)

    write_json(json_path, rows)

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RUN_TABLE_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Entrypoint for multi-run probe execution."""
    args = _parse_args()

    if not args.skip_cross_condition_check:
        _cross_condition_one_batch_check(args)
        print("[sanity] cross-condition one-batch shape/class check passed")

    run_rows: list[dict[str, object]] = []

    for condition in args.conditions:
        recipe_id = _recipe_id_for_condition(args, condition)
        for seed in args.seeds:
            print(f"[run] condition={condition} seed={seed}")
            config = TrainingRunConfig(
                condition=condition,
                seed=seed,
                recipe_id=recipe_id,
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
            run_rows.append(build_run_table_row(result))
            print(
                "[done] "
                f"condition={condition} seed={seed} "
                f"best_val_acc={result['best_val_acc']:.4f} "
                f"test_acc={result['test_acc']:.4f}"
            )

    _write_run_table(
        run_rows, json_path=args.run_table_json, csv_path=args.run_table_csv
    )
    print(f"Saved run table JSON: {args.run_table_json}")
    print(f"Saved run table CSV: {args.run_table_csv}")


if __name__ == "__main__":
    main()
