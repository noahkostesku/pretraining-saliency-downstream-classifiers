# Stage 7 - Evaluate Explanation Faithfulness

## Purpose

Score explanation maps with insertion and deletion metrics under one shared perturbation protocol.

## Core interpretation rule

These metrics evaluate faithfulness to model behavior under the chosen perturbation design. They do not establish semantic truth or full model understanding.

## Evaluation subsets and comparison policy

- Stage 5/6 artifacts are generated on one fixed stratified `200`-image subset (`20` per class)
- primary cross-condition comparison uses a per-seed intersection of correctly classified images across compared encoder conditions, computed within that fixed subset
- supplementary comparison uses all fixed `200` images for fairness diagnostics, including misclassified cases
- always report evaluated image counts for both slices (`n_primary`, `n_all`)

## Shared perturbation protocol

The following choices must be fixed and reused across all conditions:

- target score: logit of the class predicted on the original unmodified image
- evaluation image size: `224 x 224`
- masking baseline: a Gaussian-blurred version of the same resized image
- patch resolution: `16 x 16` patches with `stride=16`
- saliency ranking procedure: average saliency within each patch, ranked from highest to lowest
- insertion step schedule: restore one ranked patch at a time from the original image into the blurred baseline image
- deletion step schedule: replace one ranked patch at a time in the original image with the corresponding blurred patch
- use the original saliency ranking as fixed input for the full curve; do not recompute saliency after each perturbation step in the main protocol

## Metric definitions

- deletion AUC: lower is better
- insertion AUC: higher is better
- optional diagnostics: confidence drop after top-`k` deletion (for example `10%`, `20%`) and label-flip rate

## Evaluation procedure

### 1. Load saved saliency maps and metadata

- load map, target class, and original-image target logit metadata
- align maps to the evaluation image and convert each map into ranked `16 x 16` patches

### 2. Build analysis slices

- primary slice: compute the per-seed intersection of correctly classified image ids across compared encoder conditions
- supplementary slice: reuse all fixed `200` image ids

### 3. Run deletion curve

- start from the original image
- remove top-ranked regions according to the shared schedule
- track target logit after each step
- include the untouched original image as the first curve point

### 4. Run insertion curve

- start from the blurred baseline image
- add top-ranked regions according to the shared schedule
- track target logit after each step
- include the fully blurred baseline image as the first curve point

### 5. Compute per-image outputs

- compute insertion and deletion AUC for each image
- optionally compute confidence-drop-at-`k` and flip indicators

### 6. Aggregate reporting outputs

- primary summary: average per-image metrics over the primary intersection slice for each seed
- supplementary summary: average per-image metrics over the full fixed `200` images for each seed
- summarize each condition/method with mean +- std across seeds `0`, `1`, and `2`

## Uncertainty and significance reporting

- primary paper summary should use the primary intersection slice
- for cross-condition comparisons on the primary slice, report paired statistics over per-image AUC deltas
- recommended options: paired bootstrap confidence intervals (`1000` resamples, `95%` percentile), paired Wilcoxon signed-rank, or paired permutation tests
- if supplementary all-200 results are shown, label them as secondary diagnostics

## Comparison outputs

For each encoder condition and explanation method, report:

- seed-level mean deletion AUC and insertion AUC for the primary slice
- seed-level mean deletion AUC and insertion AUC for the all-200 supplementary slice
- condition-level mean +- std across seeds for both slices
- evaluated image counts (`n_primary` per seed and `n_all=200`)
- paired significance outputs used for primary cross-condition comparisons

## Sanity checks

- random saliency should perform worse than useful explanations
- a constant map should behave as a weak baseline
- insertion and deletion should use exactly the same ranked units and target score definition
- perturbation settings must be identical across supervised, MoCo, SwaV, and random-init
- primary slice ids must be subsets of the fixed `200` ids
- supplementary slice must use the same fixed `200` ids for every condition and seed

## Threats to validity to report

- primary intersection size can vary by seed and by which conditions are compared
- supplementary all-200 results include misclassified images by design
- insertion/deletion evaluates behavioral faithfulness, not semantic correctness

## Deliverables

- a reproducible AUC evaluation script
- per-image and aggregated explanation scores
- primary and supplementary per-seed summaries before cross-seed averaging
- comparison-ready tables for supervised, MoCo, SwaV, and random-init with `n` counts and paired primary-slice statistics
