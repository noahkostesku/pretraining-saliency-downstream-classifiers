# Stage 7 - Evaluate Explanation Faithfulness

## Purpose

Score explanation maps with insertion and deletion metrics under one shared perturbation protocol.

## Core interpretation rule

These metrics evaluate faithfulness to model behavior under the chosen perturbation design. They do not establish semantic truth or full model understanding.

## Shared perturbation protocol

The following choices must be fixed and reused across all conditions:

- masking baseline
- patch resolution
- saliency ranking procedure
- insertion step schedule
- deletion step schedule
- target score definition based on the original-image predicted class

## Metric definitions

- deletion AUC: lower is better
- insertion AUC: higher is better

## Evaluation procedure

### 1. Load a saved saliency map

- align it to the evaluation image
- convert it into ranked pixels or ranked patches using the shared ranking procedure

### 2. Run deletion curve

- start from the original image
- remove top-ranked regions according to the shared schedule
- track the target score after each step

### 3. Run insertion curve

- start from the masked baseline image
- add top-ranked regions according to the shared schedule
- track the target score after each step

### 4. Compute summary metrics

- compute AUC for each curve
- aggregate per-image scores into per-condition summaries

## Uncertainty reporting

- summarize explanation metrics on the fixed evaluation subset
- compute bootstrap confidence intervals for insertion AUC and deletion AUC if feasible
- record the bootstrap procedure clearly if used: sample unit, number of resamples, confidence level

## Comparison outputs

For each encoder condition and explanation method, report:

- mean deletion AUC
- mean insertion AUC
- bootstrap confidence intervals if computed
- number of evaluated images
- confirmation that evaluated image ids and counts are identical across all conditions

## Sanity checks

- random saliency should perform worse than useful explanations
- a constant map should behave as a weak baseline
- insertion and deletion should use exactly the same ranked units and target score definition
- evaluated subset size should match across supervised, MoCo, SwaV, and random-init

## Threats to validity to report

- explanations are evaluated on the model-predicted class for a fixed subset that includes misclassified images
- this is intentional for cross-condition fairness, especially for random-init, but some maps correspond to incorrect predictions
- interpret lower-faithfulness behavior in random-init with this limitation in mind

## Deliverables

- a reproducible AUC evaluation script
- per-image and aggregated explanation scores
- comparison-ready tables for supervised, MoCo, SwaV, and random-init
