# Stages 2-3 - Split Protocol and Downstream Model Setup

## Purpose

Lock the STL-10 data protocol and define the downstream training setup used for the main four-condition comparison (three pretrained conditions plus one random-init baseline) and the limited fine-tuning ablation.

## Scope

This file covers:

- fixed STL-10 train / validation / test usage
- exclusion of the unlabeled split from the main study
- shared preprocessing and augmentation rules
- common downstream model wrapper
- seed handling and run bookkeeping
- condition definitions for pretrained and random-init runs

## Split protocol

### Main dataset usage

- training data: `4000` images from the official labeled STL-10 `train` split
- validation data: `1000` images from the official labeled STL-10 `train` split
- test data: official labeled STL-10 `test`
- unlabeled STL-10 split: not used in the main study

### Fixed split decision

- split the labeled STL-10 `train` split with a stratified `80/20` rule
- use split seed `42`
- with STL-10's class balance, this yields `400` training images and `100` validation images per class

### Dataset citation requirement

- cite STL-10 using the original dataset paper: Coates, Ng, and Lee (AISTATS 2011)
- do not use catalog pages as primary scholarly citations

### Implementation requirements

- sample the validation split once and save the indices
- reuse the same train / validation / test partition for every encoder condition
- reuse the same split across seeds
- perform model selection on validation only
- report final accuracy on test only

### Recommended artifacts

- `splits/stl10_train_indices.json`
- `splits/stl10_val_indices.json`
- `splits/stl10_split_metadata.json`
- a short note describing the `80/20` stratification rule and split seed `42`

## Shared preprocessing rules

- use identical image resizing, cropping, normalization, and tensor conversion for all encoder conditions, including `random_init`
- use identical train-time augmentation for all encoder conditions
- use deterministic validation and test preprocessing
- final image size for the downstream wrapper is `224 x 224`
- train transform: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(ImageNet mean/std)`
- validation and test transform: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(ImageNet mean/std)`

## Downstream wrapper

Target interface:

```text
image -> encoder -> pooled 2048-D feature -> classifier
```

### Required behavior

- expose the encoder and classifier as separate modules
- support condition names `supervised`, `moco`, `swav`, and `random_init`
- support `freeze_encoder=True` for pretrained linear probing
- when `freeze_encoder=True`, keep the encoder in `eval()` mode for the full run
- support full-backbone training for the `random_init` condition in the main experiment
- support `trainable_layer4=True` for the fine-tuning ablation
- return logits for `10` STL-10 classes

## Seed protocol

- use exactly `3` training seeds per condition for the main downstream comparison: `0`, `1`, and `2`
- keep the split fixed across seeds
- vary initialization of the classifier head and training order via the seed
- keep the split seed and explanation-subset seed fixed at `42`
- save per-seed configs and metrics separately

## Fixed training-recipe protocol (no search)

- no hyperparameter search is used in the main downstream comparison
- declare and freeze one recipe for pretrained frozen-probe runs: `probe_recipe_v1`
- declare and freeze one separate recipe for random-init full training: `random_init_recipe_v1`
- keep both recipes fixed across seeds `0`, `1`, and `2`
- use validation only for checkpoint selection under the fixed epoch budget
- do not use the test split for recipe design or model selection

## Experiment bookkeeping

Each run should record:

- encoder condition
- training mode (`frozen_probe`, `full_train_random_init`, or ablation mode)
- seed
- split artifact ids or paths
- preprocessing config
- recipe id (`probe_recipe_v1` or `random_init_recipe_v1`)
- trainable parameter setting
- best validation epoch
- best validation accuracy
- final test accuracy

## Deliverables before moving on

- fixed split artifacts exist and are versioned in the project
- one shared downstream wrapper can instantiate all four main conditions
- one fixed preprocessing recipe exists for all four conditions
- one fixed seed policy exists for split generation, model training, and explanation subset sampling
- one config path can switch between frozen-probe, random-init full training, and limited-fine-tuning modes
- two fixed training recipes are defined and versioned by id (`probe_recipe_v1`, `random_init_recipe_v1`)
