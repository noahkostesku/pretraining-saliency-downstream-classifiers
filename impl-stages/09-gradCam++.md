# Stage 9 - Optional Grad-CAM++ Extension

## Purpose

Add Grad-CAM++ as an optional secondary explanation method without changing the main study framing.

## Priority

This stage is optional and should only happen after the main Grad-CAM and AUC pipeline is working.

## Fixed controls inherited from the main explainability pipeline

- same evaluation subset
- same target score definition based on the original-image predicted class
- same output map resolution and normalization
- same perturbation evaluation protocol
- same reporting structure used for Grad-CAM and Occlusion

## Implementation tasks

### 1. Add Grad-CAM++ generation

- use the same target layer `encoder.layer4[-1]`
- generate one map per evaluation image
- export maps to the same save format used by other methods

### 2. Evaluate with the same AUC scripts

- run insertion and deletion evaluation without changing perturbation settings
- aggregate metrics by condition and seed in the same format as other methods

### 3. Compare qualitatively and quantitatively

- check whether Grad-CAM++ supports the same broad conclusions as Grad-CAM
- note any cases where it materially changes the interpretation

## Deliverables

- Grad-CAM++ saliency maps
- AUC results under the same evaluation protocol
- a short note on whether this extension adds value beyond the main Grad-CAM analysis
