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
- frozen probes keep the encoder in `eval()` mode for the whole run
- no hyperparameter search is performed
- training recipes are declared once and reused unchanged across seeds

## Model settings

### Pretrained conditions (`supervised`, `moco`, `swav`)

- encoder parameters: `requires_grad=False`
- encoder mode during training: `eval()`
- classifier head: single linear layer `2048 -> 10`
- trainable parameter count: `2048 x 10 + 10 = 20490`

### Random-init condition (`random_init`)

- initialize `ResNet-50` without pretrained weights
- train encoder and classifier end-to-end on STL-10
- keep the final classifier output at `10` classes

## Required controls

- same train / validation / test split across all conditions and seeds
- same augmentation and preprocessing across all conditions
- same batch size for all conditions unless a deviation is documented
- same validation-based checkpoint selection rule across all conditions
- use one fixed recipe for all pretrained probe conditions
- use one separate fixed recipe for random-init full training

## Fixed training recipes (no search)

### Policy statement

- use one identical fixed frozen-probe recipe for `supervised`, `moco`, and `swav`
- use one separate fixed end-to-end recipe for `random_init`
- no per-condition or per-seed hyperparameter tuning is allowed

### Recipe A: `probe_recipe_v1` (frozen probes only)

Applies unchanged to `supervised`, `moco`, and `swav`.

```yaml
recipe_id: probe_recipe_v1
training_mode: frozen_probe
optimizer: AdamW
lr: 3e-4
weight_decay: 1e-4
betas: [0.9, 0.999]
scheduler: none
epochs: 50
batch_size: 64
loss: cross_entropy
label_smoothing: 0.0
grad_clip_norm: null
checkpoint_selection: best_val_accuracy
seeds: [0, 1, 2]
encoder_requires_grad: false
encoder_mode_during_training: eval
classifier_head: Linear(2048, 10)
```

### Recipe B: `random_init_recipe_v1` (random-init only)

Applies unchanged to `random_init`.

```yaml
recipe_id: random_init_recipe_v1
training_mode: full_train_random_init
optimizer: SGD
lr: 0.03
momentum: 0.9
nesterov: false
weight_decay: 1e-4
scheduler: cosine_decay
epochs: 100
batch_size: 64
loss: cross_entropy
label_smoothing: 0.0
grad_clip_norm: null
checkpoint_selection: best_val_accuracy
seeds: [0, 1, 2]
encoder_requires_grad: true
encoder_mode_during_training: train
classifier_head: Linear(2048, 10)
```

### Optimization-usage rule

- run the full fixed epoch budget for each mode (`50` for frozen probes, `100` for random-init)
- do not use patience-based early stopping
- select the final checkpoint by best validation accuracy within the fixed epoch budget

## Training loop requirements

### For pretrained conditions

- load the pretrained encoder and freeze it
- keep the encoder in `eval()` mode throughout training
- attach a randomly initialized linear head
- train on the fixed STL-10 training split
- evaluate on the fixed validation split each epoch
- run the full fixed epoch budget (`50`)
- save the best checkpoint by validation metric
- evaluate the selected checkpoint once on the test split

### For random-init condition

- initialize a new `ResNet-50` and set encoder parameters trainable
- train end-to-end on the same fixed STL-10 training split
- evaluate on the fixed validation split each epoch
- run the full fixed epoch budget (`100`)
- save the best checkpoint by validation metric
- evaluate the selected checkpoint once on the test split

### For each seed

- set all random seeds at the start of the run
- use training seeds `0`, `1`, and `2`
- log seed-specific metrics
- save the trained model checkpoint

## Metrics to report

- validation top-1 accuracy per seed
- test top-1 accuracy per seed
- fixed recipe id per run (`probe_recipe_v1` or `random_init_recipe_v1`)
- mean +- std test accuracy across seeds for each condition

## Suggested run table schema

```text
condition | training_mode | recipe_id | seed | best_val_acc | test_acc | best_epoch | checkpoint_path
```

## Sanity checks

- pretrained conditions: encoder weights do not change during training
- pretrained conditions: only classifier head receives gradients
- pretrained conditions: BatchNorm running statistics do not change during training
- random-init condition: encoder layers receive gradients and weights update
- compare one mini-batch across conditions to verify wrapper output shape and class mapping are consistent

## Deliverables before moving on

- four result groups: supervised, MoCo, SwaV, random-init
- three seeds per condition: `0`, `1`, and `2`
- mean +- std top-1 accuracy summary for all four conditions
- best checkpoints ready for explanation analysis
- fixed recipe ids recorded for every run
