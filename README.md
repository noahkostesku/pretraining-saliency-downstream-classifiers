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
├── docs/
│   ├── README.md                          # Stage execution order and docs guide
│   ├── 02-03-downstream-trainings-and-split.md  # STL-10 split protocol and shared wrapper design
│   ├── 04-trainprobes.md                  # Frozen linear-probe training plan
│   ├── 05-06-explainability.md            # Grad-CAM and Occlusion generation plan
│   ├── 07-eval-explain.md                 # Insertion/deletion AUC evaluation plan
│   ├── 08-opt-fine-tuning.md              # Optional limited fine-tuning ablation notes
│   ├── 09-gradCam++.md                    # Optional Grad-CAM++ extension notes
│   └── stage01/
│       ├── 01-encoder-prep.md             # Stage-1 detailed implementation checklist
│       └── results.md                     # Stage-1 notes/results log
├── src/
│   └── cv/
│       ├── __init__.py                    # Package marker
│       ├── config/
│       │   ├── __init__.py                # Centralized config exports
│       │   ├── base.py                    # Shared path and root configuration
│       │   └── encoders.py                # Stage-1 checkpoint configuration defaults
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
│       │   ├── __init__.py                
│       │   ├── linear_probe.py            # Frozen encoder + linear head model definition
│       │   └── downstream.py              # Shared downstream wrapper with freeze controls
│       ├── train/
│       │   ├── __init__.py                # Training package marker
│       │   ├── trainer.py                 # Shared training loop and validation checkpointing
│       │   ├── evaluate.py                # Accuracy evaluation utilities
│       │   └── metrics.py                 # Top-1 accuracy and summary aggregation helpers
│       ├── explain/
│       │   ├── __init__.py                
│       │   ├── gradcam.py                 # Grad-CAM
│       │   ├── occlusion.py               # Occlusion map 
│       │   ├── targets.py                 # target score defs for prediction class 
│       │   └── saliency_io.py             # helpers 
│       └── analysis/
│           ├── __init__.py                
│           ├── curves.py                  # IAUC/DAUC
│           ├── insertion_deletion.py      # perturbation helper 
│           ├── auc.py                     # auc plot helper 
│           ├── bootstrap.py               # bootstrap
│           └── summarize.py               # aggregate results 
├── scripts/
│   ├── make_splits.py                     # for STL-10 train/val split indices
│   ├── inspect_encoders.py                # test encoders 
│   ├── train_linear_probe.py              # training for linear probing 
│   ├── run_probe_grid.py                  # launch all encoder/seed probe runs
│   ├── summarize_probe_results.py         # aggregate accuracy mean +- std across seeds
│   └── export_eval_subset.py              # sample and save the fixed explanation evaluation subset
├── notebooks/
│   ├── 05_generate_explanations.ipynb     # Generate Grad-CAM and Occlusion maps for saved models
│   ├── 06_explainability_qc.ipynb         # Visual quality checks on saliency maps and metadata
│   └── 07_eval_explanations.ipynb         # Compute insertion/deletion AUC and summarize results
├── artifacts/
│   ├── splits/                            # Saved train/val indices and explanation subset ids
│   ├── checkpoints/                       # Saved probe model checkpoints by condition and seed
│   ├── metrics/                           # CSV/JSON summaries for accuracy and explanation scores
│   └── saliency/                          # Saved saliency maps and per-image metadata
├── data/
│   ├── raw/                               # Downloaded STL-10 files
│   ├── processed/                         # Cached processed outputs if needed
│   └── external/                          # Optional external checkpoint files
```

## Stage 1 - Encoder preparation

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
- `scripts/inspect_encoders.py` # one-shot shape and loading sanity checks

## Stages 2-3 - Split protocol and shared downstream setup

- `src/cv/data/stl10.py` # STL-10 dataset and transforms
- `src/cv/data/splits.py` # fixed stratified split creation/loading
- `src/cv/models/downstream.py` # shared downstream model wrapper
- `src/cv/config/base.py` # reusable project and path configs
- `src/cv/config/encoders.py` # stage-1 checkpoint defaults
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
* The active config source of truth is `src/cv/config/`; root-level `configs/*.yaml` are legacy placeholders and can be removed once no longer needed for documentation.
