# Stages 5-6 - Generate Explanations and Organize Comparison Outputs

## Purpose

Generate explanation maps from the trained downstream models under one fixed protocol, then organize outputs so encoder conditions can be compared fairly.

## Main method priority

- main explanation method: Grad-CAM
- secondary comparison: Occlusion
- optional extra: Grad-CAM++
- optional stability analysis: secondary only

## Fixed protocol decisions

### Target score definition

Compute explanations with respect to the model's predicted class on the original unmodified image.

### Grad-CAM target layer

Use `encoder.layer4[-1]` as the target layer.

### Explanation evaluation subset

- sample a fixed subset of STL-10 test images once
- reuse the same subset across all encoder conditions and all seeds
- optionally restrict to correctly classified images, but only if that rule is applied consistently everywhere

### Shared output format

- convert each explanation to a common `H x W` saliency map
- resize and normalize maps consistently
- save maps in a reproducible numeric format such as `.npy`

## Grad-CAM implementation tasks

- hook activations and gradients from `encoder.layer4[-1]`
- generate one heatmap per image and target class
- upsample to the common image-space resolution
- normalize with one consistent rule across all conditions

## Occlusion implementation tasks

- define patch size once for the whole study
- define masking baseline once for the whole study
- score the same predicted-class target used by Grad-CAM
- export the final map to the same `H x W` format

## Output organization

Recommended grouping:

```text
artifacts/explanations/
  supervised/
  moco/
  swav/
```

Within each condition, keep outputs separated by seed, method, and image id.

## Per-image metadata to save

- encoder condition
- seed
- explanation method
- test image id
- predicted class on original image
- true class
- correctness flag
- source checkpoint path
- saliency map path

## Deliverables before moving on

- Grad-CAM maps exist for all required models on the fixed evaluation subset
- Occlusion maps exist if included in the main implementation pass
- all maps use the same target score definition and save format
- outputs are organized so downstream AUC evaluation can read them directly
