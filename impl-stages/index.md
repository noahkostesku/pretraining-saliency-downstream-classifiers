# Implementation Stages Index

This directory breaks the project into stage-specific implementation notes derived from `plan.md`.

## Files in this directory

### `01-encoder-prep.md`

Covers Stage 1. It defines how to select, load, normalize, freeze, and validate the supervised, MoCo, and SwaV `ResNet-50` checkpoints. It also records checkpoint provenance, shared feature dimensionality, and Grad-CAM layer compatibility.

### `02-03-downstream-trainings-and-split.md`

Covers the shared data and model setup before the main experiments. It locks the fixed STL-10 train / validation / test protocol, excludes the unlabeled split from the main study, defines the common preprocessing rules, and specifies the shared downstream wrapper used by both probing and fine-tuning.

### `04-trainprobes.md`

Covers the main experiment. It defines frozen linear-probe training for each encoder condition, seed handling, checkpoint selection, reporting requirements, and the minimal sanity checks needed before comparing accuracies.

### `05-06-explainability.md`

Covers explanation generation and explanation artifact organization. It sets the target score definition, fixes `encoder.layer4[-1]` as the Grad-CAM target layer, defines the fixed evaluation subset, and describes how Grad-CAM and Occlusion outputs should be saved consistently across conditions.

### `07-eval-explain.md`

Covers explanation faithfulness evaluation. It defines the shared perturbation protocol for insertion and deletion AUC, keeps interpretation limited to behavioral faithfulness, and specifies uncertainty reporting with bootstrap confidence intervals if feasible.

### `08-opt-fine-tuning.md`

Covers the limited fine-tuning ablation. It describes how to unfreeze only `layer4` and `fc`, what controls must remain identical to the main experiment, and what to report if the ablation changes or does not change the encoder ranking.

### `09-gradCam++.md`

Covers the optional Grad-CAM++ extension. It explains how to add the method without changing the main study framing, and how to evaluate it with the same target score, evaluation subset, and AUC pipeline used for the main explainability methods.

## Recommended execution order

1. `01-encoder-prep.md`
2. `02-03-downstream-trainings-and-split.md`
3. `04-trainprobes.md`
4. `05-06-explainability.md`
5. `07-eval-explain.md`
6. `08-opt-fine-tuning.md`
7. `09-gradCam++.md`

## How this directory maps to `plan.md`

- Stages 1-4 in `plan.md` are split into encoder prep, data/model setup, probe training, and explanation generation notes here.
- The explainability additions from `plan.md` are carried through explicitly: predicted-class target score, `encoder.layer4[-1]` for Grad-CAM, a fixed evaluation subset, and a shared perturbation protocol.
- Optional work remains separated from the core pipeline so the main transfer-learning comparison stays clear and budget-aware.
