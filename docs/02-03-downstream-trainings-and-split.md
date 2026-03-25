# Stages 2-3 - Split Protocol and Downstream Model Setup

## Purpose

Lock the STL-10 data protocol and define the downstream training setup used for both the main linear-probe experiment and the limited fine-tuning ablation.

## Scope

This file covers:

- fixed STL-10 train / validation / test usage
- exclusion of the unlabeled split from the main study
- shared preprocessing and augmentation rules
- common downstream model wrapper
- seed handling and run bookkeeping

## Split protocol

### Main dataset usage

- training data: official labeled STL-10 `train`, minus a fixed validation holdout
- validation data: fixed stratified subset from labeled STL-10 `train`
- test data: official labeled STL-10 `test`
- unlabeled STL-10 split: not used in the main study

### Implementation requirements

- sample the validation split once and save the indices
- reuse the same train / validation / test partition for every encoder condition
- reuse the same split across seeds
- perform model selection on validation only
- report final accuracy on test only

### Recommended artifacts

- `splits/stl10_train_indices.json`
- `splits/stl10_val_indices.json`
- a short note describing the stratification rule and random seed used to create the split

## Shared preprocessing rules

- use identical image resizing, cropping, normalization, and tensor conversion for all encoder conditions
- use identical train-time augmentation for all encoder conditions
- use deterministic validation and test preprocessing
- document the final image size expected by the downstream wrapper

## Downstream wrapper

Target interface:

```text
image -> encoder -> pooled 2048-D feature -> classifier
```

### Required behavior

- expose the encoder and classifier as separate modules
- support `freeze_encoder=True` for linear probing
- support `trainable_layer4=True` for the fine-tuning ablation
- return logits for `10` STL-10 classes

## Seed protocol

- use at least `3` seeds for linear probing per encoder condition
- keep the split fixed across seeds
- vary initialization of the classifier head and training order via the seed
- save per-seed configs and metrics separately

## Experiment bookkeeping

Each run should record:

- encoder condition
- seed
- split artifact ids or paths
- preprocessing config
- trainable parameter setting
- best validation epoch
- best validation accuracy
- final test accuracy

## Deliverables before moving on

- fixed split artifacts exist and are versioned in the project
- one shared downstream wrapper can instantiate all encoder conditions
- one config path can switch between frozen-probe and limited-fine-tuning modes
