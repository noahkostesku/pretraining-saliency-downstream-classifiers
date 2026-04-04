# Stage 1 - Encoder Preparation

## Purpose

Prepare the three encoder conditions so every downstream experiment starts from a consistent `ResNet-50` feature extractor with known checkpoint provenance.

## Scope

This stage covers:

- supervised ImageNet-pretrained `ResNet-50`
- MoCo ImageNet-pretrained `ResNet-50`
- SwaV ImageNet-pretrained `ResNet-50`

It does not train encoders from scratch.

The random-init baseline uses the same `ResNet-50` architecture but is initialized and trained in Stage 4, not prepared from a pretrained checkpoint in this stage.

## Outputs

- a common encoder-loading interface for all three conditions
- metadata for each checkpoint: architecture, source, objective, checkpoint origin, feature dimension
- a validated frozen encoder object ready for downstream probing
- a short artifact record showing exactly which checkpoint file or library entry was used

## Required controls

- keep the backbone family fixed at `ResNet-50`
- use public pretrained checkpoints only
- expose the same pooled feature dimension `2048` for all conditions
- document any checkpoint-specific preprocessing requirements, but use one shared downstream preprocessing pipeline for the main study so all conditions see the same input transform

## Implementation tasks

### 1. Select checkpoint sources

- choose one concrete public source for each encoder condition
- record the exact identifier, URL, repository, or model zoo entry
- note any known deviations in training recipe across checkpoints because those remain part of the experimental contrast

### 2. Build a common loader API

Target interface:

```text
load_encoder(condition_name) -> encoder_module, preprocess_config, metadata
```

Expected condition names:

```text
supervised
moco
swav
```

### 3. Normalize model structure

- remove or bypass the original classification head if the checkpoint includes one
- expose the encoder as `image -> pooled 2048-D feature`
- ensure the forward path is consistent across conditions
- keep access to `encoder.layer4[-1].conv3` for Grad-CAM later

### 4. Freeze behavior for probe training

- verify encoder parameters can be globally frozen with `requires_grad=False`
- ensure batch norm and evaluation mode behavior are handled consistently during probing
- probing uses `encoder.eval()` for the full run so BatchNorm running statistics remain frozen
- train only the classifier head during frozen probing

### 5. Shared preprocessing decision

- use one simple shared preprocessing pipeline for all encoder conditions in the main study
- target input size: `224 x 224`
- train transform: `RandomResizedCrop(224)`, `RandomHorizontalFlip()`, `ToTensor()`, `Normalize(ImageNet mean/std)`
- validation and test transform: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, `Normalize(ImageNet mean/std)`
- if a source checkpoint was originally trained with a slightly different recipe, record that fact in metadata, but do not switch transforms by condition in the main comparison

### 6. Sanity checks

- run a single batch through each encoder
- verify output shape is `[batch, 2048]` after pooling
- verify no missing keys or unexpected shape mismatches when loading checkpoints
- verify `layer4[-1].conv3` exists and is reachable for explanation methods
- verify a frozen probe run leaves encoder weights and BatchNorm running statistics unchanged

## Suggested metadata template

```text
condition:
architecture:
pretraining_source:
pretraining_objective:
checkpoint_origin:
checkpoint_id:
feature_dim: 2048
gradcam_target_layer: encoder.layer4[-1].conv3
notes:
```

## Deliverables before moving on

- all three encoders load successfully
- all three produce pooled `2048`-D features
- checkpoint provenance is written down
- the shared `224 x 224` preprocessing pipeline is fixed for the main study
- frozen probing behavior is fixed to `encoder.eval()`
- a shared wrapper exists for downstream training

## Risks to watch

- MoCo or SwaV checkpoints may require key remapping
- different libraries may package the encoder and `fc` layer differently
- preprocessing mismatches can quietly invalidate the comparison if not unified early
