# Stage 4 - Train Frozen Linear Probes

## Purpose

Train the main downstream experiment: a frozen linear probe on top of each pretrained encoder.

## Core rule

This is the primary comparison. The encoder is frozen and only the final linear classifier is trained.

## Model setting

- encoder parameters: `requires_grad=False`
- classifier head: single linear layer `2048 -> 10`
- trainable parameter count: `2048 x 10 + 10 = 20490`

## Required controls

- same optimizer across conditions
- same learning-rate schedule across conditions
- same batch size across conditions
- same augmentation and preprocessing across conditions
- same epoch budget and early-stopping rule across conditions
- same validation-based checkpoint selection rule across conditions

## Training loop requirements

### For each condition

- load the frozen encoder
- attach a randomly initialized linear head
- train on the fixed STL-10 training split
- evaluate on the fixed validation split each epoch
- save the best checkpoint by validation metric
- evaluate the selected checkpoint once on the test split

### For each seed

- set all random seeds at the start of the run
- log seed-specific metrics
- save the trained linear head or full wrapped model

## Metrics to report

- validation top-1 accuracy per seed
- test top-1 accuracy per seed
- mean +- std test accuracy across seeds for each encoder condition

## Suggested run table schema

```text
condition | seed | best_val_acc | test_acc | best_epoch | checkpoint_path
```

## Sanity checks

- confirm encoder weights do not change during training
- confirm only the classifier head receives gradients
- compare one mini-batch across conditions to verify the wrapper behavior is consistent

## Deliverables before moving on

- three probe result groups: supervised, MoCo, SwaV
- at least three seeds per condition
- mean +- std top-1 accuracy summary
- best checkpoints ready for explanation analysis
