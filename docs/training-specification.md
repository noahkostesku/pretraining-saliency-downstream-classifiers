# Training Specification

This doc is a summary on the training process. Please refer to `notebooks/run.ipynb` or the README in the root for the CLI version. We used the CLI for training on the GPU and the notebook for the CPU run.

The completed training study includes four STL-10 downstream conditions, each run with three seeds:

- `supervised` (ImageNet-supervised encoder, frozen probe)
- `moco` (MoCo encoder, frozen probe)
- `swav` (SwaV encoder, frozen probe)
- `random_init` (randomly initialized ResNet-50, full end-to-end training)

Totalling `12` models trained, with each 4 models across 3 seeds, [0, 1, 2]. 

## Methods

We used a fixed STL-10 split protocol persisted in `artifacts/splits/stl10_split_metadata.json`.

| Item | Value |
| --- | --- |
| Dataset | STL-10 |
| Source split for train/val | Official labeled `train` |
| Split strategy | Stratified `80/20` |
| Split seed | `42` |
| Train count | `4000` |
| Validation count | `1000` |
| Per-class train/val counts | `400 / 100` |
| Test set | Official labeled `test` |

How we trained: Due to lack of compute and time, we manually froze and determined hyperparameters based on common assumptions on the STL-10 dataset for each encoder -> model setup, and passed in those hyperparameters using the Recipe ID.

| Condition | Training mode | Recipe ID | Trainable components |
| --- | --- | --- | --- |
| `supervised` | `frozen_probe` | `probe_recipe_v1` | Classifier head only (`Linear(2048, 10)`) |
| `moco` | `frozen_probe` | `probe_recipe_v1` | Classifier head only (`Linear(2048, 10)`) |
| `swav` | `frozen_probe` | `probe_recipe_v1` | Classifier head only (`Linear(2048, 10)`) |
| `random_init` | `full_train_random_init` | `random_init_recipe_v1` | Full backbone + classifier |

For frozen-probe runs, encoder parameters are frozen and the encoder is kept in evaluation mode during training.

## Fixed Training Recipes

The core recipe definitions that define the hyperparameters for each model are in `src/cv/train/trainer.py`.

| Field | `probe_recipe_v1` | `random_init_recipe_v1` |
| --- | --- | --- |
| Training mode | `frozen_probe` | `full_train_random_init` |
| Optimizer | `AdamW` | `SGD` |
| Learning rate | `3e-4` | `0.03` |
| Betas | `(0.9, 0.999)` | N/A |
| Momentum | N/A | `0.9` |
| Nesterov | N/A | `False` |
| Weight decay | `1e-4` | `1e-4` |
| Scheduler | `none` | `cosine_decay` |
| Epoch budget | `50` | `100` |
| Batch size | `64` | `64` |
| Loss | `cross_entropy` | `cross_entropy` |
| Label smoothing | `0.0` | `0.0` |
| Gradient clipping | `None` | `None` |
| Checkpoint selection | `best_val_accuracy` | `best_val_accuracy` |
| Seeds | `0, 1, 2` | `0, 1, 2` |

## Runtime Execution Notes

From saved run metrics in `artifacts/metrics/probe_runs/*/*.json`:

| Item | Value |
| --- | --- |
| Device used | `cuda` for all 12 runs |
| Best checkpoint policy | Reload best validation-accuracy checkpoint for test evaluation |
| Run-level outputs | JSON metrics + batch/epoch loss CSV/JSON + loss curve PNG |
| Checkpoint pattern | `artifacts/checkpoints/<condition>/seed_<seed>_<recipe_id>.pt` |

## Completed Run Results (Per Seed)

| Condition | Seed | Recipe | Best val acc | Test acc | Best epoch |
| --- | --- | --- | --- | --- | --- |
| `supervised` | `0` | `probe_recipe_v1` | `0.981` | `0.977750` | `48` |
| `supervised` | `1` | `probe_recipe_v1` | `0.980` | `0.977625` | `37` |
| `supervised` | `2` | `probe_recipe_v1` | `0.980` | `0.978000` | `46` |
| `swav` | `0` | `probe_recipe_v1` | `0.976` | `0.969125` | `50` |
| `swav` | `1` | `probe_recipe_v1` | `0.976` | `0.968375` | `43` |
| `swav` | `2` | `probe_recipe_v1` | `0.976` | `0.969750` | `50` |
| `moco` | `0` | `probe_recipe_v1` | `0.953` | `0.944875` | `38` |
| `moco` | `1` | `probe_recipe_v1` | `0.953` | `0.945250` | `39` |
| `moco` | `2` | `probe_recipe_v1` | `0.953` | `0.944875` | `41` |
| `random_init` | `0` | `random_init_recipe_v1` | `0.685` | `0.675875` | `93` |
| `random_init` | `1` | `random_init_recipe_v1` | `0.684` | `0.663750` | `80` |
| `random_init` | `2` | `random_init_recipe_v1` | `0.681` | `0.675750` | `91` |

## Aggregate Accuracy Summary Across Seeds

Standard deviations below are sample standard deviation (`ddof=1`) over seeds.

| Condition | Training mode | Recipe | Mean val acc +- std | Mean test acc +- std |
| --- | --- | --- | --- | --- |
| `supervised` | `frozen_probe` | `probe_recipe_v1` | `0.980333 +- 0.000577` | `0.977792 +- 0.000191` |
| `swav` | `frozen_probe` | `probe_recipe_v1` | `0.976000 +- 0.000000` | `0.969083 +- 0.000688` |
| `moco` | `frozen_probe` | `probe_recipe_v1` | `0.953000 +- 0.000000` | `0.945000 +- 0.000217` |
| `random_init` | `full_train_random_init` | `random_init_recipe_v1` | `0.683333 +- 0.002082` | `0.671792 +- 0.006965` |

Observed test-accuracy ranking in this completed run set: `supervised > swav > moco > random_init`.
