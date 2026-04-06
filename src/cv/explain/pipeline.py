"""Stage-5/6 orchestration helpers for explanation generation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from cv.config import ARTIFACTS_ROOT, DEFAULT_ENCODER_CHECKPOINTS
from cv.data import load_eval_subset, load_stl10_split
from cv.models import build_downstream_model, resolve_mode_config
from cv.transforms import build_eval_transform
from cv.utils.io import read_json, write_json

from .gradcam import generate_gradcam, generate_gradcampp
from .occlusion import generate_occlusion_map
from .saliency_io import save_saliency_map, write_saliency_metadata
from .targets import gather_target_scores, get_predicted_class_target

VALID_EXPLANATION_METHODS = {"gradcam", "gradcampp", "occlusion"}


@dataclass(frozen=True)
class Stage4Run:
    """Minimal run metadata needed for explanation generation."""

    condition: str
    seed: int
    training_mode: str
    recipe_id: str
    checkpoint_path: Path
    run_metrics_path: Path


class IndexedSubset(Dataset):
    """Dataset wrapper that returns the original dataset index with each sample."""

    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        source_index = self.indices[item]
        image, label = self.dataset[source_index]
        return image, label, source_index


def _resolve_artifacts_root(artifacts_root: str | Path | None) -> Path:
    if artifacts_root is None:
        return ARTIFACTS_ROOT
    return Path(artifacts_root)


def discover_stage4_runs(
    *,
    artifacts_root: str | Path | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
) -> list[Stage4Run]:
    """Discover Stage-4 run payloads from ``artifacts/metrics/probe_runs``."""
    root = _resolve_artifacts_root(artifacts_root)
    metrics_root = root / "metrics" / "probe_runs"
    if not metrics_root.exists():
        raise FileNotFoundError(
            f"Run metrics root does not exist: {metrics_root}. "
            "Run Stage-4 training first."
        )

    allowed_conditions = set(conditions) if conditions is not None else None
    allowed_seeds = set(seeds) if seeds is not None else None

    runs: list[Stage4Run] = []
    for metrics_path in sorted(
        p for p in metrics_root.glob("*/*.json")
        if not any(p.stem.endswith(s) for s in (".batch_losses", ".epoch_losses"))
    ):
        payload = read_json(metrics_path)
        if not isinstance(payload, dict):
            raise ValueError(
                f"Run metrics file must contain a JSON object: {metrics_path}"
            )

        condition = payload.get("condition")
        seed = payload.get("seed")
        training_mode = payload.get("training_mode")
        recipe_id = payload.get("recipe_id")
        checkpoint_path = payload.get("checkpoint_path")

        if not isinstance(condition, str):
            raise ValueError(f"Run payload missing condition: {metrics_path}")
        if not isinstance(seed, int):
            raise ValueError(f"Run payload missing seed: {metrics_path}")
        if not isinstance(training_mode, str):
            raise ValueError(f"Run payload missing training_mode: {metrics_path}")
        if not isinstance(recipe_id, str):
            raise ValueError(f"Run payload missing recipe_id: {metrics_path}")
        if not isinstance(checkpoint_path, str):
            raise ValueError(f"Run payload missing checkpoint_path: {metrics_path}")

        if allowed_conditions is not None and condition not in allowed_conditions:
            continue
        if allowed_seeds is not None and seed not in allowed_seeds:
            continue

        runs.append(
            Stage4Run(
                condition=condition,
                seed=seed,
                training_mode=training_mode,
                recipe_id=recipe_id,
                checkpoint_path=Path(checkpoint_path),
                run_metrics_path=metrics_path,
            )
        )

    if not runs:
        raise ValueError(
            "No Stage-4 runs matched the requested filters. "
            "Check --conditions/--seeds and Stage-4 artifacts."
        )
    return runs


def _build_model_for_run(
    *,
    run: Stage4Run,
    device: torch.device,
    allow_remote_download: bool,
) -> torch.nn.Module:
    mode_config = resolve_mode_config(
        condition=run.condition,
        training_mode=run.training_mode,
    )

    checkpoint_path = None
    if run.condition == "moco":
        checkpoint_path = str(DEFAULT_ENCODER_CHECKPOINTS.moco_checkpoint_path)
    elif run.condition == "swav":
        checkpoint_path = str(DEFAULT_ENCODER_CHECKPOINTS.swav_checkpoint_path)

    model = build_downstream_model(
        condition=run.condition,
        num_classes=10,
        freeze_encoder=mode_config.freeze_encoder,
        trainable_layer4=mode_config.trainable_layer4,
        device=device,
        allow_remote_download=allow_remote_download,
        checkpoint_path=checkpoint_path,
    )

    if not run.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint for explanation run: {run.checkpoint_path}"
        )

    checkpoint_payload = torch.load(
        run.checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_state_dict" not in checkpoint_payload:
        raise ValueError(
            f"Checkpoint payload missing 'model_state_dict': {run.checkpoint_path}"
        )
    model.load_state_dict(checkpoint_payload["model_state_dict"])
    model.eval()
    return model


def _resolve_method_callable(method: str):
    if method == "gradcam":
        return generate_gradcam
    if method == "gradcampp":
        return generate_gradcampp
    if method == "occlusion":
        return generate_occlusion_map
    raise ValueError(f"Unknown explanation method '{method}'.")


def generate_explanations_for_run(
    *,
    run: Stage4Run,
    methods: list[str],
    batch_size: int,
    data_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    device: str = "cpu",
    allow_remote_download: bool = False,
    overwrite: bool = False,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Generate all requested explanation maps for one Stage-4 run."""
    invalid_methods = [
        method for method in methods if method not in VALID_EXPLANATION_METHODS
    ]
    if invalid_methods:
        raise ValueError(
            f"Unsupported explanation methods: {invalid_methods}. "
            f"Valid methods: {sorted(VALID_EXPLANATION_METHODS)}"
        )

    artifacts_root_path = _resolve_artifacts_root(artifacts_root)
    subset_artifacts = load_eval_subset(artifacts_root=artifacts_root)

    base_test_dataset = load_stl10_split(
        "test",
        transform=build_eval_transform(),
        data_root=data_root,
        download=download,
    )
    indexed_dataset = IndexedSubset(base_test_dataset, subset_artifacts.indices)
    dataloader = DataLoader(
        indexed_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    torch_device = torch.device(device)
    model = _build_model_for_run(
        run=run,
        device=torch_device,
        allow_remote_download=allow_remote_download,
    )

    run_rows: list[dict[str, Any]] = []
    for method in methods:
        method_root = (
            artifacts_root_path
            / "saliency"
            / run.condition
            / f"seed_{run.seed}"
            / method
        )
        metadata_path = method_root / "metadata.json"
        if metadata_path.exists() and not overwrite:
            print(
                f"[skip] metadata exists for {run.condition}/seed_{run.seed}/{method}"
            )
            existing_rows = read_json(metadata_path)
            if not isinstance(existing_rows, list):
                raise ValueError(f"Expected list metadata at {metadata_path}")
            run_rows.extend(existing_rows)
            continue

        rows_for_method: list[dict[str, Any]] = []
        method_callable = _resolve_method_callable(method)

        for images, labels, image_ids in dataloader:
            images = images.to(torch_device)
            labels = labels.to(torch_device)
            with torch.no_grad():
                logits = model(images)
            target_classes = get_predicted_class_target(logits)
            target_scores = gather_target_scores(logits, target_classes)

            saliency_maps, method_targets, _ = method_callable(
                model=model,
                images=images,
                target_classes=target_classes,
            )

            predictions = target_classes.detach().cpu()
            target_scores_cpu = target_scores.detach().cpu()
            method_targets = method_targets.detach().cpu()
            labels_cpu = labels.detach().cpu()

            if not torch.equal(predictions, method_targets):
                raise RuntimeError(
                    "Method returned target classes inconsistent with original-image "
                    "predicted classes."
                )

            for item_index in range(images.shape[0]):
                image_id = int(image_ids[item_index])
                map_path = method_root / f"{image_id:05d}.npy"
                save_saliency_map(map_path, saliency_maps[item_index])

                predicted_class = int(predictions[item_index].item())
                true_class = int(labels_cpu[item_index].item())
                row: dict[str, Any] = {
                    "condition": run.condition,
                    "seed": run.seed,
                    "method": method,
                    "test_image_id": image_id,
                    "target_score_type": "predicted_class_logit",
                    "target_logit_original": float(
                        target_scores_cpu[item_index].item()
                    ),
                    "predicted_class": predicted_class,
                    "true_class": true_class,
                    "correct": bool(predicted_class == true_class),
                    "checkpoint_path": str(run.checkpoint_path),
                    "run_metrics_path": str(run.run_metrics_path),
                    "recipe_id": run.recipe_id,
                    "training_mode": run.training_mode,
                    "saliency_map_path": str(map_path),
                }
                rows_for_method.append(row)

        write_saliency_metadata(metadata_path, rows_for_method)
        run_rows.extend(rows_for_method)
        print(
            "[done] "
            f"condition={run.condition} seed={run.seed} method={method} "
            f"images={len(rows_for_method)}"
        )

    return run_rows


def generate_explanations_for_runs(
    *,
    runs: list[Stage4Run],
    methods: list[str],
    batch_size: int,
    data_root: str | Path | None = None,
    artifacts_root: str | Path | None = None,
    device: str = "cpu",
    allow_remote_download: bool = False,
    overwrite: bool = False,
    download: bool = False,
) -> list[dict[str, Any]]:
    """Generate explanations for all selected Stage-4 runs."""
    all_rows: list[dict[str, Any]] = []
    for run in runs:
        print(
            f"[run] condition={run.condition} seed={run.seed} recipe_id={run.recipe_id}"
        )
        rows = generate_explanations_for_run(
            run=run,
            methods=methods,
            batch_size=batch_size,
            data_root=data_root,
            artifacts_root=artifacts_root,
            device=device,
            allow_remote_download=allow_remote_download,
            overwrite=overwrite,
            download=download,
        )
        all_rows.extend(rows)

    output_root = _resolve_artifacts_root(artifacts_root) / "metrics" / "saliency"
    manifest_path = output_root / "generation_manifest.json"
    write_json(manifest_path, all_rows)
    print(f"Saved saliency generation manifest: {manifest_path}")
    return all_rows
