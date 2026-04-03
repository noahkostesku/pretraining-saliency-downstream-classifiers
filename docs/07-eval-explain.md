# Stage 7 - Evaluate Explanation Faithfulness

## Purpose

Score explanation maps with insertion and deletion metrics under one shared perturbation protocol.

## Core interpretation rule

These metrics evaluate faithfulness to model behavior under the chosen perturbation design. They do not establish semantic truth or full model understanding.

## Shared perturbation protocol

The following choices must be fixed and reused across all conditions:

- target score: logit of the class predicted on the original unmodified image
- evaluation image size: `224 x 224`
- masking baseline: a Gaussian-blurred version of the same resized image
- patch resolution: non-overlapping `16 x 16` patches
- saliency ranking procedure: average saliency within each patch, ranked from highest to lowest
- insertion step schedule: restore one ranked patch at a time from the original image into the blurred baseline image
- deletion step schedule: replace one ranked patch at a time in the original image with the corresponding blurred patch
- target score definition based on the original-image predicted class

## Metric definitions

- deletion AUC: lower is better
- insertion AUC: higher is better

## Evaluation procedure

### 1. Load a saved saliency map

- align it to the evaluation image
- convert it into ranked `16 x 16` patches using the shared ranking procedure

### 2. Run deletion curve

- start from the original image
- remove top-ranked regions according to the shared schedule
- track the target score after each step
- include the untouched original image as the first curve point

### 3. Run insertion curve

- start from the masked baseline image
- add top-ranked regions according to the shared schedule
- track the target score after each step
- include the fully blurred baseline image as the first curve point

### 4. Compute summary metrics

- compute AUC for each curve
- average per-image AUCs across the fixed `200`-image subset to obtain one seed-level score
- summarize each condition and method with the mean +- std of the three seed-level scores

## Uncertainty reporting

- primary paper summary: report the three seed-level means and the condition-level mean +- std across seeds
- optional appendix summary: bootstrap images within each seed if an interval display is needed
- if bootstrap is used, use image-level resampling with `1000` resamples and `95%` percentile intervals
- record the bootstrap procedure clearly anywhere it is reported

## Comparison outputs

For each encoder condition and explanation method, report:

- seed-level mean deletion AUC for seeds `0`, `1`, and `2`
- seed-level mean insertion AUC for seeds `0`, `1`, and `2`
- mean deletion AUC
- mean insertion AUC
- mean +- std across the three seeds
- bootstrap confidence intervals if computed in the appendix
- number of evaluated images (`200` per seed)
- confirmation that evaluated image ids and counts are identical across all conditions

## Sanity checks

- random saliency should perform worse than useful explanations
- a constant map should behave as a weak baseline
- insertion and deletion should use exactly the same ranked units and target score definition
- evaluated subset size should match across supervised, MoCo, SwaV, and random-init
- the same `200` image ids must be reused for every condition and every seed

## Threats to validity to report

- explanations are evaluated on the model-predicted class for a fixed subset that includes misclassified images
- this is intentional for cross-condition fairness, especially for random-init, but some maps correspond to incorrect predictions
- interpret lower-faithfulness behavior in random-init with this limitation in mind

## Deliverables

- a reproducible AUC evaluation script
- per-image and aggregated explanation scores
- per-seed explanation summaries before any cross-seed averaging
- comparison-ready tables for supervised, MoCo, SwaV, and random-init
