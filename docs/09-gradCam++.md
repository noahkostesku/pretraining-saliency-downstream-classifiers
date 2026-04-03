# Stage 9 - Supplemental Grad-CAM++ Diagnostics

## Purpose

Run supplemental diagnostics on top of the already-required Grad-CAM++ pipeline without changing the main study framing.

## Priority

This stage is optional and should only happen after Stages 5-7 are complete for Grad-CAM, Grad-CAM++, and Occlusion.

## Fixed controls inherited from the main explainability pipeline

- same evaluation subset
- same target score definition based on the original-image predicted class
- same output map resolution and normalization
- same perturbation evaluation protocol
- same reporting structure used for Grad-CAM and Occlusion

## Implementation tasks

### 1. Qualitative side-by-side diagnostics

- compare Grad-CAM and Grad-CAM++ maps on representative correct and incorrect predictions
- flag cases where Grad-CAM++ materially changes localization or confidence trends

### 2. Extended quantitative checks

- run per-class and error-slice summaries in addition to the main aggregate AUC table
- keep perturbation settings unchanged from the main pipeline

### 3. Document when the extension changes interpretation

- state whether supplemental diagnostics reinforce or weaken the core conclusions
- keep this section secondary to the core cross-condition comparison

## Deliverables

- supplemental Grad-CAM++ diagnostic notes
- optional extended tables or plots linked to the same fixed evaluation subset
- a short statement on whether diagnostics change the main interpretation
