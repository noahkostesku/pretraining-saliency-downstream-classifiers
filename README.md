# CS 4452 final project 

## Research Objective 

"Do different encoders produce representations that support better downstream classification, and do the resulting end-to-end decisions rely on behaviorally important and visually plausible image regions?"

We aim to determine whether we can I compare how useful different learned representations are for downstream classification, and whether the resulting decisions are grounded in faithful and visually plausible image regions?

We evaluate whether different pretrained encoders yield representations that transfer well to STL-10 under linear probing, and whether the resulting downstream decisions are supported by faithful and visually plausible saliency regions

- We use PyTorch for models and training, Numpy for saliency arrays and sklearn for straified splitting for the validation set for downstream training and AUC helper functions for evaluating GradCAM saliency maps.

We aim to provide findings on the following:
- whether some encoders transfer better than others under the same linear-probe protocol
- whether some encoder-based pipelines produce more faithful saliency maps than others
- whether higher downstream accuracy coincides with more faithful or more object-centered explanations
- whether the learned representation appears more or less useful through the behavior of the downstream model

We carry out 8 stages in our process to complete end-to-end training and analysis of our results:

1. Load and prepare encoders 
2. Make splits 
3. Single run training for verifying architectural soundness
4. Linear probe training 
5. Generate saliency maps/explanations 
6. Saliency quality checks 
7. Explanability evaluations/analysis 
8. GradCAM++ diagnostics 

Please check the Process Overview section below for more details.

## Notes for reproducing

Training took ~20 hours on a CPU. To speed up training, a GPU setup was used and is recommended if you intend to use our training pipeline to rerproduce and train models.

For reproducing our analysis results, please unzip the `.zip` provided in `artifacts/bundles` with the following command:

```bash
mkdir -p artifacts/bundles
cd artifacts && zip -r bundles/seed0_probe_checkpoints.zip \
  checkpoints/supervised/seed_0_probe_recipe_v1.pt \
  checkpoints/moco/seed_0_probe_recipe_v1.pt \
  checkpoints/swav/seed_0_probe_recipe_v1.pt \
  checkpoints/random_init/seed_0_random_init_recipe_v1.pt
```

After running the command above, you can run the dedicated code cell to reproduce our results quickly without running the full notebook in `analysis.ipynb`. Alternatively, you can refer to `notebooks/analysis.ipynb` for reproducing instructions and running relevant analysis code.

Please refer to the setup section below for setting up dependencies and running code in this project repo.

## Setup

The [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager is needed for running this project. Run these from repo root after environment setup:

```bash
uv venv # make a virtual environment  
source .venv/bin/activate # activate the environment 
uv sync # install packages 
```

For running notebooks, please activate the environment in the project root, then run the below:

```bash
cd notebooks # navigate to the notebooks directory
jupyter notebook # run jupyter notebook to launch notebooks
```

There is notebook code for both training and analysis and CLI for only training. For running CLIs for training, please refer to below.

**IMPORTANT**: before running training, make sure to save the results (loss curves, csv and JSON for the training runs) which have been committed to GitHub for the run notebook to work before re-running or reproducing. The model weights are stored in `artifacts/checkpoints/`, which is not committed.

# Project Structure 

```text
cv/
├── README.md                             
├── pyproject.toml                         
├── uv.lock                                
├── docs/
│   ├── training-specification.md        
│   └── saliency-analysis-deep-dive.md     
├── src/
│   └── cv/
│       ├── __init__.py                    
│       ├── config/
│       │   ├── __init__.py                
│       │   ├── base.py                    # Shared path and root configs 
│       │   └── encoders.py                # encoder configs 
│       ├── transforms.py                  # transform builders for augmentation
│       ├── utils/
│       │   ├── __init__.py                
│       │   ├── seed.py                    # Seed-setting helpers
│       │   ├── io.py                      # I/O helpers 
│       │   ├── logging.py                 # Logging helpers 
│       │   └── device.py                  # CPU/CUDA routing helper 
│       ├── data/
│       │   ├── __init__.py                
│       │   ├── stl10.py                   # STL-10 dataset loading and transforms
│       │   ├── splits.py                  # make stratified splits 
│       │   └── subset.py                  # Fixed explanation subset split maker 
│       ├── encoders/
│       │   ├── __init__.py                
│       │   ├── registry.py                # Encoder routing based on encoder name 
│       │   ├── supervised.py              # Supervised ResNet-50 checkpoint loader 
│       │   ├── moco.py                    # MoCo ResNet-50 checkpoint loader
│       │   ├── swav.py                    # SwaV ResNet-50 checkpoint loader
│       │   └── wrapper.py                 # Unified encoder wrapper exposing pooled 2048-D features
│       ├── models/
│       │   ├── __init__.py                
│       │   ├── linear_probe.py            # Frozen encoder + linear head definition
│       │   └── downstream.py              # Shared downstream wrapper
│       ├── train/
│       │   ├── __init__.py                
│       │   ├── trainer.py                 # Training loop and validation checkpointing
│       │   ├── evaluate.py                # Accuracy evaluation utils 
│       │   └── metrics.py                 # Top-1 accuracy and summary aggregation helpers
│       ├── explain/
│       │   ├── __init__.py                
│       │   ├── gradcam.py                 # Grad-CAM
│       │   ├── occlusion.py               # Occlusion map 
│       │   ├── targets.py                 # target score defs for prediction class 
│       │   ├── saliency_io.py             # saliency I/O and normalization helpers
│       │   ├── pipeline.py                # saliency generation pipeline 
│       │   └── qc.py                      # saliency quality checks
│       └── analysis/
│           ├── __init__.py                
│           ├── curves.py                  # IAUC/DAUC functions 
│           ├── insertion_deletion.py      # perturbation helper 
│           ├── auc.py                     # auc plot helper 
│           ├── bootstrap.py               # bootstrap
│           └── summarize.py               # aggregate results 
├── scripts/
│   ├── make_splits.py                     # for STL-10 train/val split indices
│   ├── inspect_encoders.py                # CLI for testing encoders 
│   ├── prepare_encoders.py                # download/prepare checkpoints, then run inspection
│   ├── train_linear_probe.py              # training for linear probing 
│   ├── run_probe_grid.py                  # launch all encoder/seed probe runs
│   ├── summarize_probe_results.py         # aggregate accuracy mean +- std across seeds
│   ├── export_eval_subset.py              # sample and save the fixed explanation evaluation subset
│   ├── generate_explanations.py           # generate Grad-CAM/Grad-CAM++/Occlusion maps for saved checkpoints
│   ├── qc_explanations.py                 # run Stage-6 saliency map QC checks
│   └── sharcnet/
│       └── sbatch_gpu_probe_pipeline.sh   # Bash script for GPU run 
├── notebooks/
│   ├── run.ipynb                          # End-to-end orchestration notebook (stages 1-6 + reporting)
│   └── analysis.ipynb                     # Analysis notebook (faithfulness + Grad-CAM++ diagnostics)
├── artifacts/
│   ├── checkpoints/                       # Saved probe model checkpoints by condition and seed (gitignored)
│   ├── saliency/                          # Saved saliency maps and per-image metadata (gitignored)
│   ├── splits/                            # Split indices/metadata generated by pipeline
│   │   ├── stl10_train_indices.json
│   │   ├── stl10_val_indices.json
│   │   └── stl10_split_metadata.json
│   └── metrics/                           # Tracked output results (JSON/CSV/PNG)
│       ├── encoder_prep_report.json
│       ├── encoder_prep_report_notebook.json
│       ├── probe_runs/
│       │   ├── run_table.csv
│       │   ├── run_table.json
│       │   ├── supervised/
│       │   │   ├── seed_<0|1|2>_probe_recipe_v1.json
│       │   │   ├── seed_<0|1|2>_probe_recipe_v1.batch_losses.csv
│       │   │   ├── seed_<0|1|2>_probe_recipe_v1.epoch_losses.csv
│       │   │   └── seed_<0|1|2>_probe_recipe_v1.loss_curve.png
│       │   ├── moco/                      # same file set as supervised for seeds 0,1,2
│       │   ├── swav/                      # same file set as supervised for seeds 0,1,2
│       │   └── random_init/
│       │       ├── seed_<0|1|2>_random_init_recipe_v1.json
│       │       ├── seed_<0|1|2>_random_init_recipe_v1.batch_losses.csv
│       │       ├── seed_<0|1|2>_random_init_recipe_v1.epoch_losses.csv
│       │       └── seed_<0|1|2>_random_init_recipe_v1.loss_curve.png
│       ├── saliency/
│       │   ├── generation_manifest.json
│       │   └── qc_report.json
│       ├── faithfulness/
│       │   ├── per_image_scores.csv
│       │   ├── per_image_scores.json
│       │   ├── seed_level_scores.csv
│       │   ├── condition_summary.csv
│       │   ├── paired_stats_primary.csv
│       │   └── figures/
│       │       ├── primary_auc_by_condition_method.png
│       │       ├── accuracy_vs_faithfulness.png
│       │       └── confidence_vs_insertion_auc.png
│       ├── gradcampp_diagnostics/
│       │   ├── seed_level_method_and_delta_scores.csv
│       │   ├── seed_level_deltas.csv
│       │   ├── condition_level_deltas.csv
│       │   ├── outcome_label.json
│       │   └── diagnostics_note.json
│       └── faithfulness_repro_seed0/
│           ├── insertion_deletion_auc_primary_seed0.csv
│           ├── condition_summary.csv
│           ├── seed_level_scores.csv
│           ├── test_accuracy_seed0.csv
│           └── figures/
│               └── primary_auc_by_condition_method_seed0.png
├── data/
│   ├── raw/                               # Downloaded STL-10 files
│   ├── processed/                         # Cached processed outputs if needed
│   └── external/                          # Optional external checkpoint files
```

# Training via CLI 

### Full CLI Flow 

Training:

```bash
uv run python scripts/prepare_encoders.py
uv run python scripts/make_splits.py --download
uv run python scripts/run_probe_grid.py --device cpu # specify cuda if not on cpu
uv run python scripts/summarize_probe_results.py
```

Explainability:

```bash
uv run python scripts/export_eval_subset.py
uv run python scripts/generate_explanations.py --conditions supervised moco swav random_init --seeds 0 1 2 --methods gradcam gradcampp occlusion --device cpu # specify cuda if not on cpu
uv run python scripts/qc_explanations.py --conditions supervised moco swav random_init --seeds 0 1 2 --methods gradcam gradcampp occlusion
```

### 1. Prepare encoders

```bash
uv run python scripts/prepare_encoders.py
```

Flags:
- `--conditions supervised moco swav` to choose which encoder checkpoints to prepare/validate.
- `--force-download` to redownload MoCo/SwaV checkpoint files.
- `--skip-inspect` to skip the post-prepare encoder inspection call.
- `--device cpu|cuda` to select device for warm/inspection checks.

Optionally, inspect the encoders by running: 

```bash
uv run python scripts/inspect_encoders.py
```

Flags:
- `--conditions supervised moco swav` to select inspected encoders.
- `--moco-checkpoint <path>` and `--swav-checkpoint <path>` for explicit checkpoint paths.
- `--allow-remote-download` to allow fallback download when local files are missing.
- `--batch-size <int>`, `--device cpu|cuda`, `--output <json_path>` to control checks/output report.

### 2. Make data splits from STL-10 train and validation datasets

```bash
uv run python scripts/make_splits.py
```

Flags:
- `--download` to download STL-10 if missing.
- `--overwrite` to regenerate split artifacts.
- `--split-seed 42` and `--val-ratio 0.2` (study defaults).

Outputs:
- `artifacts/splits/stl10_train_indices.json`
- `artifacts/splits/stl10_val_indices.json`
- `artifacts/splits/stl10_split_metadata.json`

### 3. Train a single run (smoke test)

```bash
uv run python scripts/train_linear_probe.py --condition supervised --seed 0
```

Flags:
- `--condition supervised|moco|swav|random_init`
- `--seed <int>`
- `--recipe-id <id>` to override default fixed recipe mapping.
- `--device`, `--num-workers`, `--no-pin-memory`, `--download`
- `--moco-checkpoint`, `--swav-checkpoint`, `--allow-remote-download`
- `--skip-sanity-checks` to skip first-batch gradient/BN checks.
- `--no-amp` to force fp32 on GPU (only applies when not in strict reproducibility mode).
- `--no-strict-repro` to allow `cudnn.benchmark` and CUDA AMP for speed (less reproducible).

### 4. Run full training 

```bash
uv run python scripts/run_probe_grid.py --device [DEVICE]
```

You must pass in `--device` and specify what to run on.

Flags:
- `--conditions supervised moco swav random_init`
- `--seeds 0 1 2`
- `--probe-recipe-id <id>` and `--random-init-recipe-id <id>`
- `--skip-cross-condition-check` to skip one-batch shape/class consistency check.
- `--run-table-json <path>` and `--run-table-csv <path>` to set run table outputs.
- Also supports loader/checkpoint flags from single-run script (including `--no-amp`, `--no-strict-repro`).

### 5. Summarize Run Metrics 

```bash
uv run python scripts/summarize_probe_results.py
```

Key flags:
- `--run-metrics-root <dir>` (default: `artifacts/metrics/probe_runs`)
- `--run-table-json <path>`, `--run-table-csv <path>`
- `--summary-json <path>`, `--summary-csv <path>`

### 6. Export fixed explanation evaluation subset

```bash
uv run python scripts/export_eval_subset.py
```

Flags:
- `--subset-seed 42` and `--images-per-class 20` (study defaults).
- `--overwrite` to regenerate subset artifacts.
- `--download` to download STL-10 if missing.

Outputs:
- `artifacts/splits/stl10_eval_subset_indices.json`
- `artifacts/splits/stl10_eval_subset_metadata.json`

### 7. Generate explanations from trained classifiers' checkpoints

```bash
uv run python scripts/generate_explanations.py
```

Flags:
- `--conditions supervised moco swav random_init`
- `--seeds 0 1 2`
- `--methods gradcam gradcampp occlusion`
- `--batch-size <int>`, `--device cpu|cuda`
- `--overwrite` to regenerate existing saliency outputs.
- `--allow-remote-download` for MoCo/SwaV loading fallback.

Outputs:
- `artifacts/saliency/<condition>/seed_<seed>/<method>/<image_id>.npy`
- `artifacts/saliency/<condition>/seed_<seed>/<method>/metadata.json`
- `artifacts/metrics/saliency/generation_manifest.json`

### 8. Run explanation quality checks

```bash
uv run python scripts/qc_explanations.py
```

Flags:
- default behavior checks full expected coverage for `supervised|moco|swav|random_init`, seeds `0|1|2`, and methods `gradcam|gradcampp|occlusion`
- `--conditions ...`, `--seeds ...`, `--methods ...` for filtered QC.
- when generation was run with filters, run QC with the same filters to avoid intentional missing-coverage failures
- `--output <path>` to override report destination.

Output:
- `artifacts/metrics/saliency/qc_report.json`

After this, please run `notebooks/analysis.ipynb` for occlusion methods, ablation fine tuning, and GradCAM++. Alternatively, you can also run `notebooks/run.ipynb` for all of the steps above instead of using the CLI. 

# Process Overview 

This section describes the order of the processes we do for end-to-end training and analysis.

## Stage 1: Encoder preparation

- `src/cv/encoders/registry.py`:
    - Interface for loading encoders, maps encoder type loaded to the correct function for loading the models
    - use `load_encoder` from here to load an encoder wrapped in `EnocderWrapper` interface
- `src/cv/encoders/supervised.py`:
    - loads ResNet50 baseline encoder (uses torchvision weights and removes classifier head, wraps backbone and returns metadata and preprocess config), used through registry.py if condition is supervised.
- `src/cv/encoders/moco.py`:
    - loads MoCo ResNet50 checkpoint from config and preps the encoder for downstream training 
- `src/cv/encoders/swav.py`:
    - same as moco.py but for SwaV
- `src/cv/encoders/wrapper.py`:
    - interface for using encoders after loading them (freeze(), gradcam_target_layer)
- `src/cv/config/encoders.py`: 
    - defines checkpoint configs (torchvision weight enum, local MoCo checkpoint path, SwaV checkpoint path, URLs for downloading the models, remote download boolean), used for changing checkpoint, loader files read from this file for checkpointing config 
- `scripts/prepare_encoders.py`:
    - one-time setup script for fresh clones; explicitly downloads MoCo/SwaV checkpoints into `data/external/` using URLs in `src/cv/config/encoders.py`, warms supervised weights, then runs inspection
- `scripts/inspect_encoders.py`:
    - one-shot shape and loading sanity checks

## Stages 2-3: Data splits, Shared downstream setup config 

- `src/cv/data/stl10.py`:
    - Loads STL-10 splits, validates the split names, prepares for label extraction and builds the fixed index train/val/test datasets with shared transforms
- `src/cv/data/splits.py` 
    - creates and reloads the stratified splits, persist indices + metadata under `artifacts/splits`
- `src/cv/models/downstream.py` 
    - downstream wrappe rclass (encoder -> 2048D -> classifier), frozen probe config, full train random init, layer4 fine tuning ablation modes for all 4 methods 
- `src/cv/config/base.py` 
    - project configs and output paths 
- `src/cv/config/encoders.py` # stage-1 checkpoint defaults
    - default supervised/MoCo/SwaV checkpoint IDs and configs
- `src/cv/transforms.py`
    - preprocessing pipeline for STL-10 data 
- `src/cv/utils/io.py` 
    - IO helper utils 
- `src/cv/utils/seed.py`
    - seed config utils
- `scripts/make_splits.py` 
    - for re-using fixed splits 

## Stage 4: Linear Probe Training 

- `src/cv/models/linear_probe.py`
    - Linear probe model definition
- `src/cv/train/trainer.py` 
    - Train/validate loop code definition 
- `src/cv/train/evaluate.py`
    - Test-set evaluation
- `src/cv/train/metrics.py` 
    - Top 1 helpers and condition-level mean/std aggregation 
- `scripts/train_linear_probe.py` 
    - One condition + one seed run entrypoint 
- `scripts/run_probe_grid.py`
    - Runs all training recipes for each encoder + setup
- `scripts/summarize_probe_results.py` 
    - util helper for writing summary for training results

## Stages 5-6: Explainability (saliency) generation

- `src/cv/explain/targets.py`
    - predicted-class target score on original image and predcited-class target helper functions 
- `src/cv/explain/gradcam.py` 
    - Grad-CAM and Grad-CAM++ on `encoder.layer4[-1].conv3`, returns normalized saliency maps with original-image predicted-class targets/logits
- `src/cv/explain/occlusion.py` 
    - saliency using 16x16 stride 16 and a Gaussian-blur baseline
- `src/cv/explain/saliency_io.py` 
    - saliency utils and map shape/range checks 
- `src/cv/explain/pipeline.py` 
    - Loads trained checkpoints and generates saliency maps across methods 
- `src/cv/explain/qc.py` 
    - QC checks for expected condition/seed/method coverage and fixed-subset consistency; writes JSON
- `src/cv/data/subset.py` 
    - fixed explanation evaluation subset loading and creation, writes JSON
- `scripts/export_eval_subset.py` 
    - CLI for saving the reused test subset once
- `scripts/generate_explanations.py`
    - CLI for generating Grad-CAM/Grad-CAM++/Occlusion maps from Stage-4 runs
- `scripts/qc_explanations.py`
    - CLI for running Stage-6 checks and saving quality check report

## Stage 7: Explanation evaluation

- `src/cv/analysis/curves.py`:
    - builds deterministic image-space patch grids (`16x16`, `stride=16`) and perturbation x-axis fractions
    - computes patch-level saliency scores and stable patch rankings used by insertion/deletion
- `src/cv/analysis/insertion_deletion.py`:
    - runs the fixed perturbation protocol (blurred baseline + fixed saliency ranking, no saliency recompute)
    - computes per-image insertion/deletion score curves and additional diagnostics (drop/flip at top-k)
- `src/cv/analysis/auc.py`:
    - computes curve AUCs (trapezoidal integration) for insertion and deletion metrics
- `src/cv/analysis/bootstrap.py`:
    - provides paired bootstrap CI and paired permutation p-value helpers for method/condition comparisons
- `src/cv/analysis/summarize.py`:
    - computes the primary Stage-7 slice (per-seed correct-intersection across compared conditions)
    - aggregates per-image results to seed-level and condition-level summaries
- `src/cv/analysis/__init__.py`:
    - exports analysis helpers for notebook and pipeline use
- `notebooks/analysis.ipynb`:
    - orchestrates Stage-7 preflight, per-image IAUC/DAUC evaluation, paired stats, and saved figures/tables

## Stage 8: Grad-CAM++ diagnostics

- `src/cv/analysis/summarize.py`:
    - computes method deltas (`Grad-CAM++ - Grad-CAM`) on matched primary-slice image keys
    - aggregates deltas to seed/condition level and applies the decision-rule classifier (`reinforce|neutral|weaken|mixed`)
- `notebooks/analysis.ipynb`:
    - generates required core quantitative outputs:
        - `artifacts/metrics/gradcampp_diagnostics/seed_level_method_and_delta_scores.csv`
        - `artifacts/metrics/gradcampp_diagnostics/seed_level_deltas.csv`
        - `artifacts/metrics/gradcampp_diagnostics/condition_level_deltas.csv`
        - `artifacts/metrics/gradcampp_diagnostics/outcome_label.json`
        - `artifacts/metrics/gradcampp_diagnostics/diagnostics_note.json`
    - generates required qualitative side-by-side panel under `artifacts/saliency/gradcampp_diagnostics/main_panel/`
    - includes a markdown appendix plan (not yet implemented) for `per_class_deltas.csv` and `error_slice_deltas.csv`

# Notes 

* We used `.py` files for all reusable logic, dataset handling, model loading, training, and metric computation, and imported those Python modules to run training and analysis via both CLI and notebooks.
* Notebooks are used as orchestration and inspection layers (for example `notebooks/run.ipynb`), while reusable pipeline logic stays in `src/cv/` and `scripts/` for CLI.
* We saved any notebook-produced plots and images to `artifacts/saliency/` and `artifacts/metrics/` so results stay reproducible.


## Potential Next Steps

Investigate semantic meaning of encoders: “What does each encoder embedding dimension mean, semantically, and do they provide meaningful signals for downstream classification tasks?”

Using concept probing, activation maximization, nearest-neighbor retrieval in feature space, feature visualization, concept alignment or TCAV-style analysis, which is out of scope for the current project. 

