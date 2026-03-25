# setup

1. have `uv`
2. clone the repo and run `uv venv && uv sync`
3. Run `source .venv/bin/activate` for venv.

# About the Project 

We used PyTorch for models and training, Numpy for saliency arrays and sklearn for straified splitting for the validation set for downstream training and AUC helper functions for evaluating GradCAM saliency maps.


# Project Structure 

```text
cv/
├── README.md                              # Setup instructions and project overview
├── plan.md                                # High-level study design and experimental framing
├── pyproject.toml                         # Python package metadata and dependencies
├── main.py                                # Minimal entrypoint; can later dispatch CLI tasks
├── impl-stages/
│   ├── index.md                           # Stage index and concrete scaffold reference
│   ├── 01-encoder-prep.md                 # Detailed notes for encoder loading and checkpoint provenance
│   ├── 02-03-downstream-trainings-and-split.md  # STL-10 split protocol and shared wrapper design
│   ├── 04-trainprobes.md                  # Frozen linear-probe training plan
│   ├── 05-06-explainability.md            # Grad-CAM and Occlusion generation plan
│   ├── 07-eval-explain.md                 # Insertion/deletion AUC evaluation plan
│   ├── 08-opt-fine-tuning.md              # Optional limited fine-tuning ablation notes
│   └── 09-gradCam++.md                    # Optional Grad-CAM++ extension notes
├── src/
│   └── cv/
│       ├── __init__.py                    # Package marker
│       ├── config.py                      # Shared config dataclasses and path helpers
│       ├── constants.py                   # Global constants like class counts and default seeds
│       ├── transforms.py                  # Shared torchvision transform builders
│       ├── utils/
│       │   ├── __init__.py                # Utility package marker
│       │   ├── seed.py                    # Seed-setting helpers for reproducible runs
│       │   ├── io.py                      # JSON/CSV/NPY read-write helpers for artifacts
│       │   ├── logging.py                 # Lightweight experiment logging helpers
│       │   └── device.py                  # CPU/GPU device selection helpers
│       ├── data/
│       │   ├── __init__.py                # Data package marker
│       │   ├── stl10.py                   # STL-10 dataset loading and transform wiring
│       │   ├── splits.py                  # Fixed stratified split creation and loading
│       │   └── subset.py                  # Fixed explanation subset creation and reuse
│       ├── encoders/
│       │   ├── __init__.py                # Encoder registry exports
│       │   ├── registry.py                # Condition names mapped to loader functions
│       │   ├── supervised.py              # Supervised ResNet-50 checkpoint loader
│       │   ├── moco.py                    # MoCo ResNet-50 checkpoint loader
│       │   ├── swav.py                    # SwaV ResNet-50 checkpoint loader
│       │   └── wrapper.py                 # Unified encoder wrapper exposing pooled 2048-D features
│       ├── models/
│       │   ├── __init__.py                # Model package marker
│       │   ├── linear_probe.py            # Frozen encoder + linear head model definition
│       │   └── downstream.py              # Shared downstream wrapper with freeze controls
│       ├── train/
│       │   ├── __init__.py                # Training package marker
│       │   ├── trainer.py                 # Shared training loop and validation checkpointing
│       │   ├── evaluate.py                # Accuracy evaluation utilities
│       │   └── metrics.py                 # Top-1 accuracy and summary aggregation helpers
│       ├── explain/
│       │   ├── __init__.py                # Explainability package marker
│       │   ├── gradcam.py                 # Grad-CAM generation using encoder.layer4[-1]
│       │   ├── occlusion.py               # Occlusion map generation with fixed masking settings
│       │   ├── targets.py                 # Predicted-class target score definition
│       │   └── saliency_io.py             # Save/load helpers for HxW saliency maps and metadata
│       └── analysis/
│           ├── __init__.py                # Analysis package marker
│           ├── curves.py                  # Insertion/deletion curve construction helpers
│           ├── insertion_deletion.py      # Shared perturbation implementation
│           ├── auc.py                     # Curve integration helpers built on numpy/sklearn
│           ├── bootstrap.py               # Bootstrap confidence interval utilities
│           └── summarize.py               # Aggregate per-seed and per-condition result tables
├── scripts/
│   ├── make_splits.py                     # Create and save fixed STL-10 train/val split indices
│   ├── inspect_encoders.py                # Sanity-check checkpoint loading and feature shapes
│   ├── train_linear_probe.py              # Train one frozen linear probe run for one condition and seed
│   ├── run_probe_grid.py                  # Launch all encoder/seed probe runs
│   ├── summarize_probe_results.py         # Aggregate accuracy mean +- std across seeds
│   └── export_eval_subset.py              # Sample and save the fixed explanation evaluation subset
├── notebooks/
│   ├── 05_generate_explanations.ipynb     # Generate Grad-CAM and Occlusion maps for saved models
│   ├── 06_explainability_qc.ipynb         # Visual quality checks on saliency maps and metadata
│   └── 07_eval_explanations.ipynb         # Compute insertion/deletion AUC and summarize results
├── configs/
│   ├── paths.yaml                         # Root paths for data, artifacts, and checkpoints
│   ├── data.yaml                          # STL-10 preprocessing and split settings
│   ├── probes.yaml                        # Shared linear-probe hyperparameters
│   └── explainability.yaml                # Fixed evaluation subset and perturbation protocol settings
├── artifacts/
│   ├── splits/                            # Saved train/val indices and explanation subset ids
│   ├── checkpoints/                       # Saved probe model checkpoints by condition and seed
│   ├── metrics/                           # CSV/JSON summaries for accuracy and explanation scores
│   └── saliency/                          # Saved saliency maps and per-image metadata
├── data/
│   ├── raw/                               # Downloaded STL-10 files
│   ├── processed/                         # Cached processed outputs if needed
│   └── external/                          # Optional external checkpoint files
└── reports/
    ├── figures/                           # Plots and visualizations for the write-up
    └── tables/                            # Final comparison tables for the paper
```

## Stage 1 - Encoder preparation

- `src/cv/encoders/registry.py` # central mapping from condition name to loader
- `src/cv/encoders/supervised.py` # supervised checkpoint loading
- `src/cv/encoders/moco.py` # MoCo checkpoint loading
- `src/cv/encoders/swav.py` # SwaV checkpoint loading
- `src/cv/encoders/wrapper.py` # unified pooled-feature encoder interface
- `scripts/inspect_encoders.py` # one-shot shape and loading sanity checks

## Stages 2-3 - Split protocol and shared downstream setup

- `src/cv/data/stl10.py` # STL-10 dataset and transforms
- `src/cv/data/splits.py` # fixed stratified split creation/loading
- `src/cv/models/downstream.py` # shared downstream model wrapper
- `src/cv/config.py` # reusable experiment configs
- `scripts/make_splits.py` # materialize and save split indices

## Stage 4 - Frozen probe training

- `src/cv/models/linear_probe.py` # linear probe model definition
- `src/cv/train/trainer.py` # train/validate loop
- `src/cv/train/evaluate.py` # test-set evaluation
- `src/cv/train/metrics.py` # accuracy computation and summaries
- `scripts/train_linear_probe.py` # single-run training entrypoint
- `scripts/run_probe_grid.py` # batch launcher across conditions/seeds
- `scripts/summarize_probe_results.py` # mean +- std aggregation

## Stages 5-6 - Explainability generation

- `src/cv/explain/targets.py` # predicted-class target score on original image
- `src/cv/explain/gradcam.py` # Grad-CAM on `encoder.layer4[-1]`
- `src/cv/explain/occlusion.py` # occlusion maps with fixed masking settings
- `src/cv/explain/saliency_io.py` # save HxW maps and metadata
- `src/cv/data/subset.py` # fixed explanation evaluation subset
- `notebooks/05_generate_explanations.ipynb` # main saliency-generation workflow
- `notebooks/06_explainability_qc.ipynb` # visual spot checks and consistency checks
- `scripts/export_eval_subset.py` # save the reused test subset once

## Stage 7 - Explanation evaluation

- `src/cv/analysis/curves.py` # build insertion/deletion score curves
- `src/cv/analysis/insertion_deletion.py` # shared perturbation loop
- `src/cv/analysis/auc.py` # insertion/deletion AUC calculation
- `src/cv/analysis/bootstrap.py` # optional bootstrap CIs
- `src/cv/analysis/summarize.py` # aggregate per-image/per-condition outputs
- `notebooks/07_eval_explanations.ipynb` # evaluate AUCs and produce summary tables


# Notes 

* We used `.py` files for all reusable logic, dataset handling, model loading, training, and metric computation. 
* Notebooks were used only as orchestration and inspection layers for stages 5-7, and we imported Python modules into the notebook for ablation studies and explainability analysis.
* We saved any notebook-produced plots and images to `artifacts/saliency/` and `artifacts/metrics/` so results stay reproducible.
