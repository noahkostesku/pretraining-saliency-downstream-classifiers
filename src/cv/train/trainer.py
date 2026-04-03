"""Shared training loop and validation checkpointing logic."""

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

TRAINING_RECIPES: dict[str, TrainingRecipe] = {
    PROBE_RECIPE_V1.recipe_id: PROBE_RECIPE_V1,
    RANDOM_INIT_RECIPE_V1.recipe_id: RANDOM_INIT_RECIPE_V1,
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
    device: str = "cpu"
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

    if condition != "random_init" and recipe.training_mode != "frozen_probe":
        raise ValueError(
            f"Condition '{condition}' must use frozen_probe; "
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


def _resolve_condition_checkpoint_path(config: TrainingRunConfig) -> str | None:
    if config.condition == "moco":
        return config.moco_checkpoint_path
    if config.condition == "swav":
        return config.swav_checkpoint_path
    return None


def _build_optimizer(*, model: nn.Module, recipe: TrainingRecipe) -> Optimizer:
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("No trainable parameters found for optimizer construction.")

    if recipe.optimizer == "AdamW":
        if recipe.betas is None:
            raise ValueError("AdamW recipe requires 'betas'.")
        return AdamW(
            trainable_parameters,
            lr=recipe.lr,
            weight_decay=recipe.weight_decay,
            betas=recipe.betas,
        )

    if recipe.optimizer == "SGD":
        if recipe.momentum is None or recipe.nesterov is None:
            raise ValueError("SGD recipe requires momentum and nesterov values.")
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
) -> dict[str, float]:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    sanity_done = False

    for batch_index, (images, targets) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        bn_snapshot: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        frozen_param_before: torch.Tensor | None = None
        random_init_param_before: torch.Tensor | None = None
        if run_sanity_checks and not sanity_done and batch_index == 0:
            if is_frozen_probe:
                bn_snapshot = _snapshot_encoder_bn_stats(model.encoder)
                frozen_param_before = next(model.encoder.parameters()).detach().clone()

        logits = model(images)
        loss = criterion(logits, targets)
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
            else:
                if random_init_param_before is None:
                    raise RuntimeError("Missing random-init parameter snapshot.")
                _assert_random_init_encoder_updated(model, random_init_param_before)
            sanity_done = True

        total_examples += batch_size
        total_correct += batch_correct
        total_loss += float(loss.item()) * batch_size

    if total_examples == 0:
        raise ValueError("Training dataloader is empty; cannot run an epoch.")

    return {
        "loss": total_loss / total_examples,
        "accuracy": total_correct / total_examples,
        "num_examples": float(total_examples),
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

    device = torch.device(config.device)
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
    if is_frozen_probe:
        _assert_encoder_frozen_flags(model)
    else:
        _assert_random_init_flags(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=recipe.label_smoothing)
    optimizer = _build_optimizer(model=model, recipe=recipe)
    scheduler = _build_scheduler(optimizer=optimizer, recipe=recipe)

    best_val_acc = float("-inf")
    best_epoch = -1
    epoch_history: list[dict[str, float | int]] = []
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
        )

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

    metrics_path = _resolve_run_metrics_path(config=config, recipe=recipe)
    write_json(metrics_path, run_result)
    run_result["metrics_path"] = str(metrics_path)

    return run_result
