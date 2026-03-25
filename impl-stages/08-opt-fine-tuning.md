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
- keep the same seed count and reporting format where feasible
- keep the same validation-based checkpoint selection rule
- keep this ablation limited in breadth so it does not dominate the study

## Implementation tasks

### 1. Extend the downstream wrapper

- add a mode that unfreezes `layer4` and `fc`
- verify earlier layers remain frozen

### 2. Define the optimization recipe

- either reuse the probe optimizer recipe if stable or define a separate fine-tuning recipe and document it clearly
- if different learning rates are used for `layer4` and `fc`, record that explicitly

### 3. Run the ablation selectively

- prioritize one best checkpoint source per encoder condition
- use fewer total runs only if budget requires it, and document that reduction

## Reporting goals

- compare ablation accuracy against the frozen-probe baseline
- note whether encoder ranking changes
- keep the interpretation narrow: this tests limited adaptation, not full end-to-end optimization

## Deliverables

- one ablation result group per encoder condition
- a short comparison table against the main frozen-probe results
- a clear statement about whether the ablation changes the main story
