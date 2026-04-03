# Stage 9 - Mandatory Grad-CAM++ Diagnostics

## Purpose

Run required Grad-CAM++ diagnostics on top of the core explainability pipeline and explicitly report whether they change interpretation.

## Priority

This stage is mandatory and should run after Stages 5-7 are complete for Grad-CAM, Grad-CAM++, and Occlusion.

- Stage 8 (limited fine-tuning) remains optional and does not replace Stage 9.
- Stage 9 does not change the core evaluation protocol from Stages 5-7.

## Fixed controls inherited from the main explainability pipeline

- same evaluation subset
- same target score definition based on the original-image predicted class
- same output map resolution and normalization
- same perturbation evaluation protocol
- same reporting structure used for Grad-CAM and Occlusion
- same condition and seed coverage (`supervised`, `moco`, `swav`, `random_init`; seeds `0`, `1`, `2`)

## Implementation tasks

### 1. Qualitative side-by-side diagnostics (required)

- compare Grad-CAM and Grad-CAM++ maps on representative correct and incorrect predictions
- main paper figure: use one compact fixed-image panel shared across all conditions (same image ids for both methods)
- recommended seed for the main paper panel: seed `0` for all conditions to avoid cherry-picking
- appendix: optional per-seed qualitative panels if additional diagnostic detail is needed
- include predicted class, true class, and correctness annotation in every displayed panel

### 2. Extended quantitative checks (required)

- required core table: report raw method scores and deltas together for interpretability
- per-row schema:
  - `condition | seed | gradcam_ins | gradcampp_ins | delta_ins | gradcam_del | gradcampp_del | delta_del`
- compute per-image deltas between methods on the same image: `delta_insertion = insertion_auc_gradcampp - insertion_auc_gradcam` and `delta_deletion = deletion_auc_gradcampp - deletion_auc_gradcam`
- report seed-level aggregate deltas by condition
- report condition-level mean delta and seed-level std
- optional appendix diagnostics: per-class and error-slice summaries (correct vs incorrect predictions)
- for per-class and error-slice appendix summaries, always report sample counts `n`
- suppress or merge appendix slices with low counts (`n < 10`) to avoid unstable comparisons
- keep perturbation settings unchanged from the main pipeline

### 3. Interpretation decision rule (required)

- classify the diagnostics outcome as `reinforce`, `neutral`, `weaken`, or `mixed`
- use `0.01` as a predeclared practical-effect threshold on normalized AUC scale
- use seed-averaged condition-level deltas to assign the label:
  - `reinforce`: Grad-CAM++ improves insertion (`>= +0.01`) and improves deletion (`<= -0.01`) in at least 3 of 4 conditions
  - `weaken`: Grad-CAM++ degrades insertion (`<= -0.01`) and degrades deletion (`>= +0.01`) in at least 3 of 4 conditions
  - `neutral`: absolute delta is `< 0.01` on both metrics in at least 3 of 4 conditions
  - otherwise `mixed`
- do not use the label alone; report uncertainty with seed-level std and optionally bootstrap intervals
- keep this interpretation section secondary to the core cross-condition comparison

## Deliverables

- mandatory Grad-CAM++ diagnostics note summarizing outcome label (`reinforce`, `neutral`, `weaken`, `mixed`)
- required core summary tables linked to the same fixed evaluation subset:
  - `artifacts/metrics/gradcampp_diagnostics/seed_level_method_and_delta_scores.csv`
  - `artifacts/metrics/gradcampp_diagnostics/seed_level_deltas.csv`
- optional appendix tables:
  - `artifacts/metrics/gradcampp_diagnostics/per_class_deltas.csv`
  - `artifacts/metrics/gradcampp_diagnostics/error_slice_deltas.csv`
- required main-paper qualitative panel under `artifacts/saliency/gradcampp_diagnostics/main_panel/`
- optional appendix qualitative panels under `artifacts/saliency/gradcampp_diagnostics/appendix/`
- a short statement on whether diagnostics change the main interpretation
