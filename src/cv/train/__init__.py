"""Training, evaluation, and metrics utilities."""

from .evaluate import evaluate_model
from .metrics import (
    mean_and_std,
    summarize_test_accuracy_by_condition,
    top1_accuracy,
    top1_num_correct,
)
from .trainer import (
    RUN_TABLE_COLUMNS,
    RANDOM_INIT_RECIPE_V1,
    PROBE_RECIPE_V1,
    TrainingRecipe,
    TrainingRunConfig,
    build_run_table_row,
    resolve_training_recipe,
    train_one_run,
)

__all__ = [
    "PROBE_RECIPE_V1",
    "RANDOM_INIT_RECIPE_V1",
    "RUN_TABLE_COLUMNS",
    "TrainingRecipe",
    "TrainingRunConfig",
    "build_run_table_row",
    "evaluate_model",
    "mean_and_std",
    "resolve_training_recipe",
    "summarize_test_accuracy_by_condition",
    "top1_accuracy",
    "top1_num_correct",
    "train_one_run",
]
