"""Shared training loop and validation checkpointing logic."""

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cv.config import ARTIFACTS_ROOT
from cv.data import build_downstream_datasets, load_fixed_split_indices
from cv.models import build_downstream_model, resolve_mode_config
from cv.utils.io import ensure_parent, write_json
from cv.utils.seed import set_seed

from .evaluate import evaluate_model
from .metrics import top1_num_correct


RUN_TABLE_COLUMNS = (
    "condition",
    "training_mode",
    "recipe_id",
    "seed",
    "best_val_acc",
    "test_acc",
    "best_epoch",
    "checkpoint_path",
)

ABLATION_TRAINING_MODES = {
    "ablation_layer4",
    "ablation_mode",
    "limited_finetune",
}


@dataclass(frozen=True)
class TrainingRecipe:
    """Fixed, versioned recipe for one downstream training mode."""

    recipe_id: str
    training_mode: str
    optimizer: str
    lr: float
    weight_decay: float
    epochs: int
    batch_size: int
    loss: str = "cross_entropy"
    label_smoothing: float = 0.0
    grad_clip_norm: float | None = None
    checkpoint_selection: str = "best_val_accuracy"
    seeds: tuple[int, ...] = (0, 1, 2)
    scheduler: str = "none"
    momentum: float | None = None
    nesterov: bool | None = None
    betas: tuple[float, float] | None = None
    layer4_lr: float | None = None
    classifier_lr: float | None = None


PROBE_RECIPE_V1 = TrainingRecipe(
    recipe_id="probe_recipe_v1",
    training_mode="frozen_probe",
    optimizer="AdamW",
    lr=3e-4,
    weight_decay=1e-4,
    epochs=50,
    batch_size=64,
    loss="cross_entropy",
    label_smoothing=0.0,
    grad_clip_norm=None,
    checkpoint_selection="best_val_accuracy",
    seeds=(0, 1, 2),
    scheduler="none",
    betas=(0.9, 0.999),
)

RANDOM_INIT_RECIPE_V1 = TrainingRecipe(
    recipe_id="random_init_recipe_v1",
    training_mode="full_train_random_init",
    optimizer="SGD",
    lr=0.03,
    momentum=0.9,
    nesterov=False,
    weight_decay=1e-4,
    scheduler="cosine_decay",
    epochs=100,
    batch_size=64,
    loss="cross_entropy",
    label_smoothing=0.0,
    grad_clip_norm=None,
    checkpoint_selection="best_val_accuracy",
    seeds=(0, 1, 2),
)

ABLATION_LAYER4_RECIPE_V1 = TrainingRecipe(
    recipe_id="ablation_layer4_v1",
    training_mode="ablation_layer4",
    optimizer="AdamW",
    lr=1e-4,
    weight_decay=1e-4,
    epochs=30,
    batch_size=64,
    loss="cross_entropy",
    label_smoothing=0.0,
    grad_clip_norm=None,
    checkpoint_selection="best_val_accuracy",
    seeds=(0,),
    scheduler="none",
    betas=(0.9, 0.999),
    layer4_lr=1e-4,
    classifier_lr=1e-3,
)

TRAINING_RECIPES: dict[str, TrainingRecipe] = {
    PROBE_RECIPE_V1.recipe_id: PROBE_RECIPE_V1,
    RANDOM_INIT_RECIPE_V1.recipe_id: RANDOM_INIT_RECIPE_V1,
    ABLATION_LAYER4_RECIPE_V1.recipe_id: ABLATION_LAYER4_RECIPE_V1,
}

DEFAULT_RECIPE_BY_CONDITION = {
    "supervised": PROBE_RECIPE_V1.recipe_id,
    "moco": PROBE_RECIPE_V1.recipe_id,
    "swav": PROBE_RECIPE_V1.recipe_id,
    "random_init": RANDOM_INIT_RECIPE_V1.recipe_id,
}


@dataclass(frozen=True)
class TrainingRunConfig:
    """Single-run training configuration for one condition and seed."""

    condition: str
    seed: int
    device: str = "auto"
    recipe_id: str | None = None
    artifacts_root: str | Path | None = None
    data_root: str | Path | None = None
    download: bool = False
    num_workers: int = 0
    pin_memory: bool = True
    allow_remote_download: bool | None = None
    moco_checkpoint_path: str | None = None
    swav_checkpoint_path: str | None = None
    sanity_checks: bool = True
    verbose_batch_logging: bool = True
    log_every_n_batches: int = 1
    save_loss_history: bool = True
    save_loss_plot: bool = True


def resolve_training_recipe(*, condition: str, recipe_id: str | None) -> TrainingRecipe:
    """Resolve and validate the fixed recipe for a condition."""
    selected_recipe_id = recipe_id
    if selected_recipe_id is None:
        selected_recipe_id = DEFAULT_RECIPE_BY_CONDITION.get(condition)

    if selected_recipe_id is None:
        raise ValueError(f"No default recipe is defined for condition '{condition}'.")

    recipe = TRAINING_RECIPES.get(selected_recipe_id)
    if recipe is None:
        valid = ", ".join(sorted(TRAINING_RECIPES))
        raise ValueError(
            f"Unknown recipe_id '{selected_recipe_id}'. Valid values: {valid}"
        )

    if condition == "random_init" and recipe.training_mode != "full_train_random_init":
        raise ValueError(
            "random_init must use a full-train recipe; "
            f"got training_mode='{recipe.training_mode}'."
        )

    if condition != "random_init":
        valid_modes = {"frozen_probe"} | ABLATION_TRAINING_MODES
        if recipe.training_mode not in valid_modes:
            valid_str = ", ".join(sorted(valid_modes))
            raise ValueError(
                f"Condition '{condition}' must use one of ({valid_str}); "
                f"got training_mode='{recipe.training_mode}'."
            )

    return recipe


def build_run_table_row(run_result: dict[str, Any]) -> dict[str, Any]:
    """Project one run payload to the required summary table schema."""
    return {
        "condition": run_result["condition"],
        "training_mode": run_result["training_mode"],
        "recipe_id": run_result["recipe_id"],
        "seed": run_result["seed"],
        "best_val_acc": run_result["best_val_acc"],
        "test_acc": run_result["test_acc"],
        "best_epoch": run_result["best_epoch"],
        "checkpoint_path": run_result["checkpoint_path"],
    }


def _resolve_artifacts_root(artifacts_root: str | Path | None) -> Path:
    if artifacts_root is None:
        return ARTIFACTS_ROOT
    return Path(artifacts_root)


def _resolve_device(device: str) -> torch.device:
    normalized = device.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resolved = torch.device(device)
    if resolved.type not in {"cpu", "cuda"}:
        raise ValueError(
            f"Only CPU and CUDA devices are supported. Received device='{device}'."
        )
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(
            "CUDA device requested but CUDA is not available on this host. "
            "Use device='cpu' or device='auto'."
        )
    return resolved


def _resolve_checkpoint_path(
    *, config: TrainingRunConfig, recipe: TrainingRecipe
) -> Path:
    artifacts_root = _resolve_artifacts_root(config.artifacts_root)
    return (
        artifacts_root
        / "checkpoints"
        / config.condition
        / f"seed_{config.seed}_{recipe.recipe_id}.pt"
    )


def _resolve_run_metrics_path(
    *, config: TrainingRunConfig, recipe: TrainingRecipe
) -> Path:
    artifacts_root = _resolve_artifacts_root(config.artifacts_root)
    return (
        artifacts_root
        / "metrics"
        / "probe_runs"
        / config.condition
        / f"seed_{config.seed}_{recipe.recipe_id}.json"
    )


def _resolve_loss_artifact_paths(
    *,
    config: TrainingRunConfig,
    recipe: TrainingRecipe,
) -> dict[str, Path]:
    artifacts_root = _resolve_artifacts_root(config.artifacts_root)
    root = artifacts_root / "metrics" / "probe_runs" / config.condition
    stem = f"seed_{config.seed}_{recipe.recipe_id}"
    return {
        "batch_csv": root / f"{stem}.batch_losses.csv",
        "batch_json": root / f"{stem}.batch_losses.json",
        "epoch_csv": root / f"{stem}.epoch_losses.csv",
        "epoch_json": root / f"{stem}.epoch_losses.json",
        "curve_png": root / f"{stem}.loss_curve.png",
    }


def _write_csv_rows(
    *,
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _save_loss_curve(
    *,
    path: Path,
    epoch_rows: list[dict[str, Any]],
    condition: str,
    seed: int,
    recipe_id: str,
) -> None:
    if not epoch_rows:
        raise ValueError("Cannot save loss curve with empty epoch rows.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in epoch_rows]
    train_losses = [float(row["train_loss"]) for row in epoch_rows]
    val_losses = [float(row["val_loss"]) for row in epoch_rows]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_losses, label="train_loss", marker="o", linewidth=1.5)
    ax.plot(epochs, val_losses, label="val_loss", marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Loss Curves ({condition}, seed={seed}, recipe={recipe_id})")
    ax.grid(alpha=0.25)
    ax.legend()

    ensure_parent(path)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _resolve_condition_checkpoint_path(config: TrainingRunConfig) -> str | None:
    if config.condition == "moco":
        return config.moco_checkpoint_path
    if config.condition == "swav":
        return config.swav_checkpoint_path
    return None


def _build_layer4_ablation_param_groups(
    *,
    model: nn.Module,
    recipe: TrainingRecipe,
) -> list[dict[str, object]]:
    if recipe.layer4_lr is None:
        raise ValueError("Layer4 ablation recipe requires 'layer4_lr'.")
    if recipe.classifier_lr is None:
        raise ValueError("Layer4 ablation recipe requires 'classifier_lr'.")
    if not hasattr(model.encoder, "encoder") or not hasattr(
        model.encoder.encoder, "layer4"
    ):
        raise ValueError("Model encoder does not expose layer4 for ablation.")

    layer4_parameters = [
        parameter
        for parameter in model.encoder.encoder.layer4.parameters()
        if parameter.requires_grad
    ]
    classifier_parameters = [
        parameter
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    ]
    if not layer4_parameters:
        raise ValueError("No trainable layer4 parameters found for ablation optimizer.")
    if not classifier_parameters:
        raise ValueError(
            "No trainable classifier parameters found for ablation optimizer."
        )

    allowed_ids = {
        id(parameter) for parameter in layer4_parameters + classifier_parameters
    }
    unexpected_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in allowed_ids
    ]
    if unexpected_trainable:
        raise ValueError(
            "Found unexpected trainable parameters in ablation mode: "
            f"{unexpected_trainable}"
        )

    return [
        {
            "params": layer4_parameters,
            "lr": recipe.layer4_lr,
            "weight_decay": recipe.weight_decay,
        },
        {
            "params": classifier_parameters,
            "lr": recipe.classifier_lr,
            "weight_decay": recipe.weight_decay,
        },
    ]


def _build_optimizer(*, model: nn.Module, recipe: TrainingRecipe) -> Optimizer:
    use_layer4_groups = recipe.training_mode in ABLATION_TRAINING_MODES
    parameter_groups: list[dict[str, object]] | None = None

    if use_layer4_groups:
        parameter_groups = _build_layer4_ablation_param_groups(
            model=model, recipe=recipe
        )
    else:
        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        if not trainable_parameters:
            raise ValueError(
                "No trainable parameters found for optimizer construction."
            )

    if recipe.optimizer == "AdamW":
        if recipe.betas is None:
            raise ValueError("AdamW recipe requires 'betas'.")
        if parameter_groups is not None:
            return AdamW(
                parameter_groups,
                lr=recipe.lr,
                weight_decay=recipe.weight_decay,
                betas=recipe.betas,
            )
        return AdamW(
            trainable_parameters,
            lr=recipe.lr,
            weight_decay=recipe.weight_decay,
            betas=recipe.betas,
        )

    if recipe.optimizer == "SGD":
        if recipe.momentum is None or recipe.nesterov is None:
            raise ValueError("SGD recipe requires momentum and nesterov values.")
        if parameter_groups is not None:
            return SGD(
                parameter_groups,
                lr=recipe.lr,
                momentum=recipe.momentum,
                nesterov=recipe.nesterov,
                weight_decay=recipe.weight_decay,
            )
        return SGD(
            trainable_parameters,
            lr=recipe.lr,
            momentum=recipe.momentum,
            nesterov=recipe.nesterov,
            weight_decay=recipe.weight_decay,
        )

    raise ValueError(f"Unsupported optimizer '{recipe.optimizer}'.")


def _build_scheduler(*, optimizer: Optimizer, recipe: TrainingRecipe):
    if recipe.scheduler == "none":
        return None
    if recipe.scheduler == "cosine_decay":
        return CosineAnnealingLR(optimizer, T_max=recipe.epochs)
    raise ValueError(f"Unsupported scheduler '{recipe.scheduler}'.")


def _snapshot_encoder_bn_stats(
    model: nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    snapshots: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            snapshots.append(
                (
                    module.running_mean.detach().clone(),
                    module.running_var.detach().clone(),
                )
            )
    return snapshots


def _assert_bn_stats_unchanged(
    model: nn.Module,
    snapshot: list[tuple[torch.Tensor, torch.Tensor]],
) -> None:
    current_modules = [
        module
        for module in model.modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if len(current_modules) != len(snapshot):
        raise RuntimeError("BatchNorm module count changed during training step.")

    for module, (running_mean, running_var) in zip(current_modules, snapshot):
        if not torch.equal(module.running_mean, running_mean):
            raise RuntimeError(
                "BatchNorm running_mean changed while encoder should stay in eval mode."
            )
        if not torch.equal(module.running_var, running_var):
            raise RuntimeError(
                "BatchNorm running_var changed while encoder should stay in eval mode."
            )


def _assert_frozen_encoder_gradients(model: nn.Module) -> None:
    encoder_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.encoder.parameters()
    )
    classifier_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.classifier.parameters()
    )

    if encoder_has_grad:
        raise RuntimeError("Frozen encoder received gradients unexpectedly.")
    if not classifier_has_grad:
        raise RuntimeError("Classifier head did not receive gradients.")


def _assert_random_init_gradients(model: nn.Module) -> torch.Tensor:
    first_trainable: torch.Tensor | None = None
    encoder_has_grad = False
    for parameter in model.encoder.parameters():
        if not parameter.requires_grad:
            continue
        if first_trainable is None:
            first_trainable = parameter.detach().clone()
        if (
            parameter.grad is not None
            and torch.count_nonzero(parameter.grad).item() > 0
        ):
            encoder_has_grad = True

    if first_trainable is None:
        raise RuntimeError("random_init encoder has no trainable parameters.")
    if not encoder_has_grad:
        raise RuntimeError("random_init encoder did not receive non-zero gradients.")

    return first_trainable


def _assert_random_init_encoder_updated(model: nn.Module, before: torch.Tensor) -> None:
    first_parameter = next(
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    )
    if torch.equal(first_parameter.detach(), before):
        raise RuntimeError(
            "random_init encoder weights did not update after optimizer step."
        )


def _assert_encoder_frozen_flags(model: nn.Module) -> None:
    if any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("Frozen probe run has trainable encoder parameters.")
    if not all(parameter.requires_grad for parameter in model.classifier.parameters()):
        raise RuntimeError("Classifier head should be fully trainable.")


def _assert_random_init_flags(model: nn.Module) -> None:
    if not any(parameter.requires_grad for parameter in model.encoder.parameters()):
        raise RuntimeError("random_init run has no trainable encoder parameters.")


def _assert_layer4_ablation_flags(model: nn.Module) -> None:
    if not hasattr(model.encoder, "encoder") or not hasattr(
        model.encoder.encoder, "layer4"
    ):
        raise RuntimeError("Layer4 ablation mode requires an encoder with layer4.")

    if not all(parameter.requires_grad for parameter in model.classifier.parameters()):
        raise RuntimeError(
            "Classifier head should be fully trainable in ablation mode."
        )

    layer4_trainable = any(
        parameter.requires_grad
        for parameter in model.encoder.encoder.layer4.parameters()
    )
    if not layer4_trainable:
        raise RuntimeError("Layer4 ablation mode expects trainable layer4 parameters.")

    non_layer4_trainable = [
        name
        for name, parameter in model.encoder.encoder.named_parameters()
        if not name.startswith("layer4.") and parameter.requires_grad
    ]
    if non_layer4_trainable:
        raise RuntimeError(
            "Found non-layer4 trainable encoder parameters in ablation mode: "
            f"{non_layer4_trainable}"
        )


def _first_layer4_parameter(model: nn.Module) -> torch.Tensor:
    return next(model.encoder.encoder.layer4.parameters())


def _first_non_layer4_encoder_parameter(model: nn.Module) -> torch.Tensor | None:
    for name, parameter in model.encoder.encoder.named_parameters():
        if not name.startswith("layer4."):
            return parameter
    return None


def _assert_layer4_ablation_gradients(model: nn.Module) -> torch.Tensor:
    layer4_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.encoder.encoder.layer4.parameters()
        if parameter.requires_grad
    )
    classifier_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.classifier.parameters()
        if parameter.requires_grad
    )
    frozen_encoder_has_grad = any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for name, parameter in model.encoder.encoder.named_parameters()
        if not name.startswith("layer4.")
    )

    if not layer4_has_grad:
        raise RuntimeError(
            "Layer4 parameters did not receive gradients in ablation mode."
        )
    if not classifier_has_grad:
        raise RuntimeError("Classifier did not receive gradients in ablation mode.")
    if frozen_encoder_has_grad:
        raise RuntimeError(
            "Frozen non-layer4 encoder parameters received gradients in ablation mode."
        )

    return _first_layer4_parameter(model).detach().clone()


def _assert_layer4_updated(model: nn.Module, before: torch.Tensor) -> None:
    if torch.equal(_first_layer4_parameter(model).detach(), before):
        raise RuntimeError("Layer4 parameters did not update after optimizer step.")


def _train_one_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    grad_clip_norm: float | None,
    run_sanity_checks: bool,
    is_frozen_probe: bool,
    is_layer4_ablation: bool,
    condition: str,
    seed: int,
    epoch: int,
    global_step_start: int,
    verbose_batch_logging: bool,
    log_every_n_batches: int,
) -> dict[str, Any]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    global_step = global_step_start
    batch_history: list[dict[str, Any]] = []
    num_batches = len(dataloader)
    if num_batches <= 0:
        raise ValueError("Training dataloader has zero batches; cannot run an epoch.")

    sanity_done = False

    for batch_index, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        bn_snapshot: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        frozen_param_before: torch.Tensor | None = None
        random_init_param_before: torch.Tensor | None = None
        layer4_param_before: torch.Tensor | None = None
        frozen_non_layer4_before: torch.Tensor | None = None
        if run_sanity_checks and not sanity_done and batch_index == 0:
            if is_frozen_probe:
                bn_snapshot = _snapshot_encoder_bn_stats(model.encoder)
                frozen_param_before = next(model.encoder.parameters()).detach().clone()
            elif is_layer4_ablation:
                frozen_non_layer4 = _first_non_layer4_encoder_parameter(model)
                if frozen_non_layer4 is not None:
                    frozen_non_layer4_before = frozen_non_layer4.detach().clone()

        logits = model(images)
        loss = criterion(logits, targets)
        loss_value = float(loss.item())
        batch_size = targets.shape[0]
        batch_correct = top1_num_correct(logits, targets)

        loss.backward()

        if run_sanity_checks and not sanity_done and batch_index == 0:
            if is_frozen_probe:
                _assert_frozen_encoder_gradients(model)
                if bn_snapshot is None:
                    raise RuntimeError(
                        "Missing BatchNorm snapshot for frozen sanity check."
                    )
                _assert_bn_stats_unchanged(model.encoder, bn_snapshot)
            elif is_layer4_ablation:
                layer4_param_before = _assert_layer4_ablation_gradients(model)
            else:
                random_init_param_before = _assert_random_init_gradients(model)

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

        optimizer.step()

        if run_sanity_checks and not sanity_done and batch_index == 0:
            if is_frozen_probe:
                if frozen_param_before is None:
                    raise RuntimeError("Missing frozen encoder parameter snapshot.")
                first_encoder_param = next(model.encoder.parameters())
                if not torch.equal(first_encoder_param.detach(), frozen_param_before):
                    raise RuntimeError(
                        "Frozen encoder weights changed after optimizer step."
                    )
            elif is_layer4_ablation:
                if layer4_param_before is None:
                    raise RuntimeError("Missing layer4 parameter snapshot.")
                _assert_layer4_updated(model, layer4_param_before)
                if frozen_non_layer4_before is not None:
                    current_frozen = _first_non_layer4_encoder_parameter(model)
                    if current_frozen is None:
                        raise RuntimeError(
                            "Missing non-layer4 encoder parameter for ablation check."
                        )
                    if not torch.equal(
                        current_frozen.detach(), frozen_non_layer4_before
                    ):
                        raise RuntimeError(
                            "Frozen non-layer4 encoder parameter changed after optimizer step."
                        )
            else:
                if random_init_param_before is None:
                    raise RuntimeError("Missing random-init parameter snapshot.")
                _assert_random_init_encoder_updated(model, random_init_param_before)
            sanity_done = True

        batch_history.append(
            {
                "condition": condition,
                "seed": seed,
                "epoch": epoch,
                "batch_index": batch_index + 1,
                "num_batches": num_batches,
                "global_step": global_step,
                "train_loss": loss_value,
            }
        )
        if verbose_batch_logging and (
            (batch_index + 1) % log_every_n_batches == 0
            or batch_index + 1 == num_batches
        ):
            print(
                f"[batch] condition={condition} seed={seed} "
                f"epoch={epoch} batch={batch_index + 1}/{num_batches} "
                f"step={global_step} loss={loss_value:.6f}"
            )

        global_step += 1
        total_examples += batch_size
        total_correct += batch_correct
        total_loss += loss_value * batch_size

    if total_examples == 0:
        raise ValueError("Training dataloader is empty; cannot run an epoch.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": float(total_examples),
        "batch_history": batch_history,
        "next_global_step": global_step,
    }


def train_one_run(config: TrainingRunConfig | None = None) -> dict[str, Any]:
    """Train one downstream run and return a result payload."""
    if config is None:
        raise ValueError("train_one_run requires a TrainingRunConfig instance.")

    recipe = resolve_training_recipe(
        condition=config.condition, recipe_id=config.recipe_id
    )
    if config.seed not in recipe.seeds:
        raise ValueError(
            f"Seed {config.seed} is not valid for recipe '{recipe.recipe_id}'. "
            f"Allowed seeds: {list(recipe.seeds)}"
        )
    if config.log_every_n_batches <= 0:
        raise ValueError(
            f"log_every_n_batches must be positive, got {config.log_every_n_batches}."
        )

    mode_config = resolve_mode_config(
        condition=config.condition,
        training_mode=recipe.training_mode,
    )

    set_seed(config.seed)

    split_artifacts = load_fixed_split_indices(artifacts_root=config.artifacts_root)
    datasets = build_downstream_datasets(
        train_indices=split_artifacts.train_indices,
        val_indices=split_artifacts.val_indices,
        data_root=config.data_root,
        download=config.download,
    )

    device = _resolve_device(config.device)
    pin_memory = bool(config.pin_memory and device.type != "cpu")
    data_generator = torch.Generator()
    data_generator.manual_seed(config.seed)

    train_loader = DataLoader(
        datasets.train,
        batch_size=recipe.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        generator=data_generator,
    )
    val_loader = DataLoader(
        datasets.val,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=recipe.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
    )

    model = build_downstream_model(
        condition=config.condition,
        num_classes=10,
        freeze_encoder=mode_config.freeze_encoder,
        trainable_layer4=mode_config.trainable_layer4,
        device=device,
        allow_remote_download=config.allow_remote_download,
        checkpoint_path=_resolve_condition_checkpoint_path(config),
    )

    is_frozen_probe = recipe.training_mode == "frozen_probe"
    is_layer4_ablation = recipe.training_mode in ABLATION_TRAINING_MODES
    if is_frozen_probe:
        _assert_encoder_frozen_flags(model)
    elif is_layer4_ablation:
        _assert_layer4_ablation_flags(model)
    else:
        _assert_random_init_flags(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    optimizer = _build_optimizer(model=model, recipe=recipe)
    scheduler = _build_scheduler(optimizer=optimizer, recipe=recipe)

    best_val_acc = float("-inf")
    best_epoch = -1
    epoch_history: list[dict[str, float | int]] = []
    batch_loss_history: list[dict[str, Any]] = []
    global_step = 0
    checkpoint_path = _resolve_checkpoint_path(config=config, recipe=recipe)

    for epoch in range(1, recipe.epochs + 1):
        train_metrics = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip_norm=recipe.grad_clip_norm,
            run_sanity_checks=bool(config.sanity_checks and epoch == 1),
            is_frozen_probe=is_frozen_probe,
            is_layer4_ablation=is_layer4_ablation,
            condition=config.condition,
            seed=config.seed,
            epoch=epoch,
            global_step_start=global_step,
            verbose_batch_logging=config.verbose_batch_logging,
            log_every_n_batches=config.log_every_n_batches,
        )
        global_step = int(train_metrics["next_global_step"])
        batch_loss_history.extend(train_metrics["batch_history"])

        val_metrics = evaluate_model(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step()

        epoch_history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_acc": float(train_metrics["accuracy"]),
                "val_loss": float(val_metrics["loss"]),
                "val_acc": float(val_metrics["accuracy"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )

        print(
            f"[epoch] condition={config.condition} seed={config.seed} epoch={epoch}/{recipe.epochs} "
            f"train_loss={float(train_metrics['loss']):.6f} "
            f"train_acc={float(train_metrics['accuracy']):.4f} "
            f"val_loss={float(val_metrics['loss']):.6f} "
            f"val_acc={float(val_metrics['accuracy']):.4f}"
        )

        val_acc = float(val_metrics["accuracy"])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            payload = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "condition": config.condition,
                "seed": config.seed,
                "training_mode": recipe.training_mode,
                "recipe": asdict(recipe),
            }
            ensure_parent(checkpoint_path)
            torch.save(payload, checkpoint_path)

    if best_epoch < 1:
        raise RuntimeError("No checkpoint was selected; validation loop did not run.")

    checkpoint_payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint_payload["model_state_dict"])

    test_metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device,
    )

    run_result: dict[str, Any] = {
        "condition": config.condition,
        "training_mode": recipe.training_mode,
        "recipe_id": recipe.recipe_id,
        "seed": config.seed,
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_metrics["accuracy"]),
        "best_epoch": int(best_epoch),
        "checkpoint_path": str(checkpoint_path),
        "split_train_indices_path": str(split_artifacts.train_indices_path),
        "split_val_indices_path": str(split_artifacts.val_indices_path),
        "split_metadata_path": str(split_artifacts.metadata_path),
        "device": str(device),
        "epoch_history": epoch_history,
    }

    loss_paths = _resolve_loss_artifact_paths(config=config, recipe=recipe)
    if config.save_loss_history:
        epoch_loss_rows = [
            {
                "condition": config.condition,
                "seed": config.seed,
                "recipe_id": recipe.recipe_id,
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_acc": float(row["val_acc"]),
                "lr": float(row["lr"]),
            }
            for row in epoch_history
        ]
        _write_csv_rows(
            path=loss_paths["epoch_csv"],
            rows=epoch_loss_rows,
            fieldnames=[
                "condition",
                "seed",
                "recipe_id",
                "epoch",
                "train_loss",
                "val_loss",
                "train_acc",
                "val_acc",
                "lr",
            ],
        )
        write_json(loss_paths["epoch_json"], epoch_loss_rows)

        _write_csv_rows(
            path=loss_paths["batch_csv"],
            rows=batch_loss_history,
            fieldnames=[
                "condition",
                "seed",
                "epoch",
                "batch_index",
                "num_batches",
                "global_step",
                "train_loss",
            ],
        )
        write_json(loss_paths["batch_json"], batch_loss_history)

        run_result["batch_loss_csv_path"] = str(loss_paths["batch_csv"])
        run_result["batch_loss_json_path"] = str(loss_paths["batch_json"])
        run_result["epoch_loss_csv_path"] = str(loss_paths["epoch_csv"])
        run_result["epoch_loss_json_path"] = str(loss_paths["epoch_json"])

        print("[artifacts] saved loss history files:")
        print(f"- batch CSV: {loss_paths['batch_csv']}")
        print(f"- batch JSON: {loss_paths['batch_json']}")
        print(f"- epoch CSV: {loss_paths['epoch_csv']}")
        print(f"- epoch JSON: {loss_paths['epoch_json']}")

    if config.save_loss_plot:
        _save_loss_curve(
            path=loss_paths["curve_png"],
            epoch_rows=[dict(row) for row in epoch_history],
            condition=config.condition,
            seed=config.seed,
            recipe_id=recipe.recipe_id,
        )
        run_result["loss_curve_path"] = str(loss_paths["curve_png"])
        print(f"[artifacts] saved loss curve: {loss_paths['curve_png']}")

    metrics_path = _resolve_run_metrics_path(config=config, recipe=recipe)
    write_json(metrics_path, run_result)
    run_result["metrics_path"] = str(metrics_path)

    return run_result
