# Stages 5-6 - Generate Explanations and Organize Comparison Outputs

## Purpose

Generate explanation maps from the trained downstream models under one fixed protocol, then organize outputs so encoder conditions can be compared fairly.

## Main method priority

- main gradient-based methods: Grad-CAM and Grad-CAM++
- secondary comparison: Occlusion
- optional stability analysis: secondary only

## Fixed protocol decisions

### Target score definition

Compute explanations with respect to the logit of the model's predicted class on the original unmodified image.

### Explanation generation rule

- generate exactly one saliency map per image and method from the original image
- keep this map fixed for downstream insertion/deletion ranking
- do not recompute saliency after each perturbation step in the main faithfulness protocol
- if explanation stability under perturbation is studied, report it as a separate secondary analysis

### Grad-CAM target layer

Use the last spatial convolution in the encoder, `encoder.layer4[-1].conv3`, as the target layer.

### Explanation evaluation subset

- sample a fixed stratified subset of `200` STL-10 test images once (`20` images per class)
- use explanation-subset seed `42`
- reuse the same subset across all encoder conditions and all seeds
- do not restrict Stage 5/6 generation to correctly classified images
- keep the exact same image ids for every condition, including random-init
- evaluate explanations on each model's predicted class for every image in the fixed subset
- for Stage 7 primary cross-condition faithfulness comparisons, compute a per-seed intersection of correctly classified images within this fixed subset
- keep all-200-image summaries as supplementary fairness diagnostics

### Shared output format

- convert each explanation to a common `224 x 224` saliency map
- resize every map into image space and min-max normalize each map to `[0, 1]`
- if a map is constant after resizing, save it as all zeros
- save maps in a reproducible numeric format such as `.npy`

## Grad-CAM implementation tasks

- hook activations and gradients from `encoder.layer4[-1].conv3`
- generate one heatmap per image and target class
- upsample to the common `224 x 224` image-space resolution
- apply the same postprocessing and `[0, 1]` normalization rule across all conditions

## Occlusion implementation tasks

- use `16 x 16` image-space patches with `stride=16`
- use a Gaussian-blurred version of the image as the masking baseline
- score the same predicted-class target used by Grad-CAM
- export the final map to the same `H x W` format

## Seed handling for explanations

- generate explanations for every seed-specific best checkpoint from Stage 4
- do not choose a single "representative" seed for the main analysis
- keep explanation artifacts separated by condition, seed, method, and image id
- summarize explanation quality at the seed level first, then across seeds in Stage 7

## Output organization

Recommended grouping:

```text
artifacts/saliency/
  supervised/
  moco/
  swav/
  random_init/
```

Within each condition, keep outputs separated by seed, method, and image id.

## Per-image metadata to save

- encoder condition
- seed
- explanation method
- test image id
- target score type
- original-image target logit
- predicted class on original image
- true class
- correctness flag
- source checkpoint path
- saliency map path

## Deliverables before moving on

- Grad-CAM maps exist for all required models on the fixed evaluation subset
- Grad-CAM++ maps exist for all required models on the fixed evaluation subset
- Occlusion maps exist for the same fixed evaluation subset
- all maps use the same target score definition and save format
- all three training seeds per condition have explanation artifacts
- outputs are organized so downstream AUC evaluation can read them directly

## Interpretation note

- Stage 5/6 artifacts intentionally cover a fixed subset that includes both correct and incorrect predictions
- this keeps generation and QC fully aligned across conditions, including random-init
- Stage 7 should treat the common correctly classified intersection as primary and the all-200 subset as supplementary
