# Training and Explainability Requirements (Canonical)

This document defines the canonical execution/configuration contract for the project pipeline used in `notebooks/run.ipynb` and corresponding CLI scripts.

Scope covered here:
- Stage 1: encoder preparation and inspection
- Stage 2: fixed STL-10 split creation
- Stage 4: downstream training runs
- Stage 5: saliency generation
- Stage 6: saliency QC checks

The canonical source for runtime behavior is:
- Notebook configs in `notebooks/run.ipynb` (`RUN_CONFIG`, `EXPLAIN_CONFIG`)
- Shared trainer implementation in `src/cv/train/trainer.py`

## Canonical Runtime Config (Notebook)

Use these exact values as the canonical baseline, including for cluster runs unless intentionally overridden.

```python
RUN_CONFIG: dict[str, object] = {
    "conditions": ["supervised", "moco", "swav", "random_init"],
    "seeds": [0, 1, 2],

    "device": "cpu",
    "num_workers": 0,
    "pin_memory": False,

    "artifacts_root": None,
    "data_root": None,
    "download_data": True,

    "prepare_checkpoints": True,
    "force_download_checkpoints": False,
    "inspect_encoders": True,
    "allow_remote_download": True,

    "create_splits": True,
    "overwrite_splits": False,
    "split_seed": 42,
    "val_ratio": 0.2,

    "run_cross_condition_sanity_check": True,
    "run_training": True,
    "sanity_checks": True,

    "probe_recipe_id": None,
    "random_init_recipe_id": None,
}

EXPLAIN_CONFIG: dict[str, object] = {
    "run_explanations": True,
    "run_qc": True,
    "conditions": list(RUN_CONFIG["conditions"]),
    "seeds": [int(seed) for seed in list(RUN_CONFIG["seeds"])],
    "methods": ["gradcam", "gradcampp", "occlusion"],
    "batch_size": 8,
    "overwrite": False,
}
```

## Stage-by-Stage Requirements

## Stage 1: Encoder Preparation + Inspection

Purpose:
- Ensure supervised/MoCo/SwaV encoders and checkpoints are available and load correctly.

CLI script:
- `scripts/prepare_encoders.py`

Key flags and canonical values:
- `--conditions supervised moco swav` (from `RUN_CONFIG["conditions"]`, excluding `random_init` for prep)
- `--force-download` maps to `RUN_CONFIG["force_download_checkpoints"]` (canonical: `False`)
- `--device` maps to `RUN_CONFIG["device"]` (canonical notebook value: `cpu`)
- `--skip-inspect` should NOT be used when `RUN_CONFIG["inspect_encoders"]` is `True`

Notes:
- MoCo/SwaV local checkpoints default under `data/external/`.
- If missing and allowed, remote fallback is controlled by `allow_remote_download` in downstream components.

## Stage 2: Fixed Data Splits

Purpose:
- Materialize deterministic train/val split indices.

CLI script:
- `scripts/make_splits.py`

Key flags and canonical values:
- `--split-seed 42` (`RUN_CONFIG["split_seed"]`)
- `--val-ratio 0.2` (`RUN_CONFIG["val_ratio"]`)
- `--download` from `RUN_CONFIG["download_data"]` (canonical: `True`)
- `--overwrite` from `RUN_CONFIG["overwrite_splits"]` (canonical: `False`)

Required outputs:
- `artifacts/splits/stl10_train_indices.json`
- `artifacts/splits/stl10_val_indices.json`
- `artifacts/splits/stl10_split_metadata.json`

## Stage 4: Downstream Training

Purpose:
- Train all downstream classifiers across configured conditions and seeds.

Primary CLI script:
- `scripts/run_probe_grid.py`

Single-run CLI script (smoke/debug):
- `scripts/train_linear_probe.py`

Canonical condition/seed grid:
- Conditions: `supervised`, `moco`, `swav`, `random_init`
- Seeds: `0,1,2`

Core training config mapping to `TrainingRunConfig`:
- `condition` <- `--condition` or grid loop over `--conditions`
- `seed` <- `--seed` or grid loop over `--seeds`
- `device` <- `--device` (notebook canonical is `cpu`; cluster override can be `cuda`)
- `num_workers` <- `--num-workers` (canonical: `0`)
- `pin_memory` <- default true unless `--no-pin-memory`; canonical notebook: `False`
- `artifacts_root` <- `--artifacts-root` or `None`
- `data_root` <- `--data-root` or `None`
- `download` <- `--download` (canonical notebook: `True`)
- `allow_remote_download` <- `--allow-remote-download` (canonical notebook: `True`)
- `sanity_checks` <- default true unless `--skip-sanity-checks` (canonical: `True`)
- `recipe_id` override:
  - pretrained conditions via `--probe-recipe-id` (canonical: `None` -> `probe_recipe_v1`)
  - random_init via `--random-init-recipe-id` (canonical: `None` -> `random_init_recipe_v1`)

Default recipe mapping (from trainer):
- `supervised` -> `probe_recipe_v1` (50 epochs)
- `moco` -> `probe_recipe_v1` (50 epochs)
- `swav` -> `probe_recipe_v1` (50 epochs)
- `random_init` -> `random_init_recipe_v1` (100 epochs)

Checkpointing behavior (required):
- Best-by-validation-accuracy checkpoint is updated during training.
- Full epoch budget is still run (no early stopping).
- Final test evaluation reloads the best checkpoint.

Checkpoint output path:
- `artifacts/checkpoints/<condition>/seed_<seed>_<recipe_id>.pt`

Run metrics + loss artifacts output path:
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.json`
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.batch_losses.csv`
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.batch_losses.json`
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.epoch_losses.csv`
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.epoch_losses.json`
- `artifacts/metrics/probe_runs/<condition>/seed_<seed>_<recipe_id>.loss_curve.png`

Important cluster execution requirement:
- For Stage 4 cluster runs, device must be explicitly set.
- The README command at line 209 (`uv run python scripts/run_probe_grid.py`) should be treated as incomplete for cluster use.
- Use explicit device selection:
  - `uv run python scripts/run_probe_grid.py --device cuda`.
- Use `--device cuda` (the flag is `--device`, not `--cuda`).

Assume that you will 100% have access to CUDA. For CLI, make sure to specify `--device cuda`.

## Stage 5: Saliency Generation

Purpose:
- Generate saliency maps and metadata from trained downstream models.

CLI script:
- `scripts/generate_explanations.py`

Canonical config mapping from `EXPLAIN_CONFIG`:
- `run_explanations=True` -> run generation
- `conditions` = `RUN_CONFIG["conditions"]`
- `seeds` = `RUN_CONFIG["seeds"]`
- `methods` = `gradcam`, `gradcampp`, `occlusion`
- `batch_size=8`
- `overwrite=False`

Expected outputs:
- `artifacts/saliency/<condition>/seed_<seed>/<method>/<image_id>.npy`
- `artifacts/saliency/<condition>/seed_<seed>/<method>/metadata.json`
- `artifacts/metrics/saliency/generation_manifest.json`

## Stage 6: Saliency QC

Purpose:
- Validate explanation coverage and artifact integrity.

CLI script:
- `scripts/qc_explanations.py`

Canonical config mapping from `EXPLAIN_CONFIG`:
- `run_qc=True` -> run QC
- Same `conditions`, `seeds`, `methods` as Stage 5

Expected output:
- `artifacts/metrics/saliency/qc_report.json`

## Notebook to CLI Correspondence

`RUN_CONFIG` -> CLI equivalence:
- `conditions` -> `run_probe_grid.py --conditions ...`
- `seeds` -> `run_probe_grid.py --seeds ...`
- `device` -> `--device ...`
- `num_workers` -> `--num-workers ...`
- `pin_memory=False` -> include `--no-pin-memory`
- `download_data=True` -> include `--download` where dataset is loaded
- `allow_remote_download=True` -> include `--allow-remote-download`
- `probe_recipe_id` -> `--probe-recipe-id ...` when not `None`
- `random_init_recipe_id` -> `--random-init-recipe-id ...` when not `None`
- `sanity_checks=True` -> do not set `--skip-sanity-checks`
- `run_cross_condition_sanity_check=True` -> do not set `--skip-cross-condition-check`

`EXPLAIN_CONFIG` -> CLI equivalence:
- `conditions` -> `generate_explanations.py --conditions ...` and `qc_explanations.py --conditions ...`
- `seeds` -> `--seeds ...`
- `methods` -> `--methods ...`
- `batch_size` -> `generate_explanations.py --batch-size ...`
- `overwrite=False` -> do not pass `--overwrite`

## Canonical Command Set (Cluster-Friendly)

These commands preserve current notebook intent while making device explicit for Stage 4.

```bash
uv run python scripts/prepare_encoders.py --conditions supervised moco swav --device cpu
uv run python scripts/make_splits.py --split-seed 42 --val-ratio 0.2 --download

# Stage 4 (EXPLICIT device required)
uv run python scripts/run_probe_grid.py --conditions supervised moco swav random_init --seeds 0 1 2 --device cuda --download --allow-remote-download --num-workers 0 --no-pin-memory

# Stage 5
uv run python scripts/generate_explanations.py --conditions supervised moco swav random_init --seeds 0 1 2 --methods gradcam gradcampp occlusion --batch-size 8 --device cpu

# Stage 6
uv run python scripts/qc_explanations.py --conditions supervised moco swav random_init --seeds 0 1 2 --methods gradcam gradcampp occlusion
```

If running fully on CPU, replace Stage 4 device flag with `--device cpu`.
