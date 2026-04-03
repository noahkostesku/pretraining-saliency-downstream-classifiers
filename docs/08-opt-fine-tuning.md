# Stage 8 - Optional Partial Fine-Tuning Ablation

## Purpose

Test whether limited task-specific adaptation changes the downstream ranking or meaningfully improves accuracy beyond frozen linear probing.

## Scope

This is a secondary ablation, not the main experiment.

## Trainable modules

- `model.layer4`
- `model.fc`

All earlier encoder blocks remain frozen.

## Required controls

- keep the same fixed STL-10 split protocol
- keep the same preprocessing and augmentation
- use a single seed for the ablation to keep this stage lightweight
- use seed `0` unless there is a documented reason to choose a different fixed seed
- keep the same validation-based checkpoint selection rule
- keep this ablation limited in breadth so it does not dominate the study

## Implementation tasks

### 1. Extend the downstream wrapper

- add a mode that unfreezes `layer4` and `fc`
- verify earlier layers remain frozen

### 2. Define the optimization recipe

- use one fixed fine-tuning recipe across encoder conditions rather than retuning broadly
- optimizer: `AdamW`
- epoch budget: `30`
- batch size: `64`
- weight decay: `1e-4`
- learning rate for `layer4`: `1e-4`
- learning rate for `fc`: `1e-3`
- no additional hyperparameter search unless the ablation is clearly unstable, and any deviation must be documented explicitly

### 3. Run the ablation selectively

- prioritize one best checkpoint source per encoder condition
- run one fine-tuning ablation per encoder condition with the fixed ablation seed

## Reporting goals

- compare ablation accuracy against the frozen-probe baseline
- note whether encoder ranking changes
- keep the interpretation narrow: this tests limited adaptation, not full end-to-end optimization
- keep explainability follow-up optional for this ablation unless it becomes central to the paper

## Deliverables

- one ablation result group per encoder condition
- one ablation run per encoder condition using the fixed seed
- a short comparison table against the main frozen-probe results
- a clear statement about whether the ablation changes the main story
