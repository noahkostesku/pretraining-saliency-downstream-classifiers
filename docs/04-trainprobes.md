# Stage 4 - Train Main Downstream Models

## Purpose

Train the main downstream comparison with four conditions under one fixed STL-10 protocol:

- supervised pretrained `ResNet-50` with frozen linear probe
- MoCo pretrained `ResNet-50` with frozen linear probe
- SwaV pretrained `ResNet-50` with frozen linear probe
- random-init `ResNet-50` trained end-to-end as a baseline

## Core rules

- pretrained conditions use frozen linear probing only
- random-init condition must be fully trained, not frozen
- all conditions use the same fixed split and evaluation rules

## Model settings

### Pretrained conditions (`supervised`, `moco`, `swav`)

- encoder parameters: `requires_grad=False`
- classifier head: single linear layer `2048 -> 10`
- trainable parameter count: `2048 x 10 + 10 = 20490`

### Random-init condition (`random_init`)

- initialize `ResNet-50` without pretrained weights
- train encoder and classifier end-to-end on STL-10
- keep the final classifier output at `10` classes

## Required controls

- same train / validation / test split across all conditions and seeds
- same augmentation and preprocessing across all conditions
- same batch size and epoch budget across all conditions unless a deviation is documented
- same validation-based checkpoint selection rule across all conditions
- if optimizer or schedule differs for random-init stability, record and justify it explicitly

## Training loop requirements

### For pretrained conditions

- load the pretrained encoder and freeze it
- attach a randomly initialized linear head
- train on the fixed STL-10 training split
- evaluate on the fixed validation split each epoch
- save the best checkpoint by validation metric
- evaluate the selected checkpoint once on the test split

### For random-init condition

- initialize a new `ResNet-50` and set encoder parameters trainable
- train end-to-end on the same fixed STL-10 training split
- evaluate on the fixed validation split each epoch
- save the best checkpoint by validation metric
- evaluate the selected checkpoint once on the test split

### For each seed

- set all random seeds at the start of the run
- log seed-specific metrics
- save the trained model checkpoint

## Metrics to report

- validation top-1 accuracy per seed
- test top-1 accuracy per seed
- mean +- std test accuracy across seeds for each condition

## Suggested run table schema

```text
condition | training_mode | seed | best_val_acc | test_acc | best_epoch | checkpoint_path
```

## Sanity checks

- pretrained conditions: encoder weights do not change during training
- pretrained conditions: only classifier head receives gradients
- random-init condition: encoder layers receive gradients and weights update
- compare one mini-batch across conditions to verify wrapper output shape and class mapping are consistent

## Deliverables before moving on

- four result groups: supervised, MoCo, SwaV, random-init
- at least three seeds per condition
- mean +- std top-1 accuracy summary for all four conditions
- best checkpoints ready for explanation analysis
