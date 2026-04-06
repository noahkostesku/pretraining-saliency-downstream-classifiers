# Saliency Analysis Methods 

This doc is a summary/overview for `notebooks/analysis.ipynb` from what was done in this notebook to what we produced.

We do the following saliency analyses: 

1. Per-image insertion/deletion faithfulness evaluation
2. Seed-level and condition-level aggregation
3. Paired bootstrap confidence intervals and paired permutation tests
4. Grad-CAM vs Grad-CAM++ delta diagnostics
5. Qualitative Grad-CAM / Grad-CAM++ side-by-side panel generation


NOTE: due to similar and consistent performance between seeds 0, 1, and 2, for simplicity for analysis, seed 0 of all 4 downstream classifier setups were used.

## Inputs and Outputs 

Inputs:

| Item | Value |
| --- | --- |
| Conditions | `supervised`, `moco`, `swav`, `random_init` |
| Seeds | `0, 1, 2` |
| Methods | `gradcam`, `gradcampp`, `occlusion` |
| Evaluation subset size | `200` images |
| Target definition during map generation | Predicted-class score from original image |
| Saliency map format | `.npy`, normalized to `[0, 1]` |


We run a quality check too on basic saliency performance to ensure correctness. From `artifacts/metrics/saliency/qc_report.json`:

| QC field | Value |
| --- | --- |
| `num_rows` | `7200` |
| `num_errors` | `0` |
| `expected_subset_count` | `200` |
| Coverage entries | `36` (`4 conditions x 3 seeds x 3 methods`) |
| Complete entries | `36 / 36` |

## Faithfulness Evaluation Protocol

Faithfulness scoring uses fixed insertion/deletion perturbation logic in `src/cv/analysis/insertion_deletion.py` and `src/cv/analysis/curves.py`.

| Component | Setting |
| --- | --- |
| Image size | `224 x 224` |
| Patch size / stride | `16 / 16` |
| Patch order | Descending mean saliency per patch (stable sort) |
| Baseline for perturbation | Gaussian-blurred version of the same image |
| Blur kernel / sigma | `21 / 5.0` |
| Curve x-axis | Fraction of perturbed patches (`0.0` to `1.0`) |
| Insertion start state | Fully blurred image |
| Deletion start state | Original image |
| Target class for scoring | Predicted class on original image |
| Score tracked along curves | Predicted-class softmax probability |
| AUC computation | Trapezoidal integration (`insertion_auc`, `deletion_auc`) |

## Evaluation Slices

Two slices are reported in notebook outputs:

- `all200`: all 200 fixed subset images
- `primary`: per-seed intersection of images correctly classified by all four conditions

Please check the notebook for more details.

The primary slice size by seed from `artifacts/metrics/faithfulness/seed_level_scores.csv`:

| Seed | `n_primary` |
| --- | --- |
| `0` | `127` |
| `1` | `118` |
| `2` | `123` |

## Condition-Level Faithfulness Results from the Primary Slice

Values below come from `artifacts/metrics/faithfulness/condition_summary.csv`.

| Condition | Method | Insertion AUC (mean +- std) | Deletion AUC (mean +- std) | Mean `n_primary` |
| --- | --- | --- | --- | --- |
| `supervised` | `gradcam` | `0.938218 +- 0.000412` | `0.864973 +- 0.006411` | `122.67` |
| `supervised` | `gradcampp` | `0.927526 +- 0.001507` | `0.881597 +- 0.004646` | `122.67` |
| `supervised` | `occlusion` | `0.948614 +- 0.001734` | `0.809409 +- 0.002657` | `122.67` |
| `moco` | `gradcam` | `0.577827 +- 0.009924` | `0.328504 +- 0.010993` | `122.67` |
| `moco` | `gradcampp` | `0.570259 +- 0.009765` | `0.333071 +- 0.010884` | `122.67` |
| `moco` | `occlusion` | `0.534749 +- 0.014559` | `0.326442 +- 0.007820` | `122.67` |
| `swav` | `gradcam` | `0.849840 +- 0.007054` | `0.602174 +- 0.006081` | `122.67` |
| `swav` | `gradcampp` | `0.800310 +- 0.002329` | `0.690798 +- 0.007285` | `122.67` |
| `swav` | `occlusion` | `0.809017 +- 0.002828` | `0.599916 +- 0.003582` | `122.67` |
| `random_init` | `gradcam` | `0.800346 +- 0.006624` | `0.716450 +- 0.017334` | `122.67` |
| `random_init` | `gradcampp` | `0.791129 +- 0.003471` | `0.726847 +- 0.022622` | `122.67` |
| `random_init` | `occlusion` | `0.925075 +- 0.005614` | `0.512694 +- 0.033134` | `122.67` |

Notes for interpretation:

- Higher insertion AUC is better.
- Lower deletion AUC is better.
- For this run set, insertion/deletion rankings differ by method and metric, so results should be interpreted per metric rather than as one scalar score.

## Grad-CAM++ Diagnostics

Grad-CAM++ diagnostics are derived from primary-slice matched-image deltas (`Grad-CAM++ - Grad-CAM`) saved in `artifacts/metrics/gradcampp_diagnostics/`.

### Condition-level deltas for insertion and deletion AUC 

From `artifacts/metrics/gradcampp_diagnostics/condition_level_deltas.csv`:

| Condition | Mean delta insertion AUC | Mean delta deletion AUC | Mean `n_primary` |
| --- | --- | --- | --- |
| `moco` | `-0.007568` | `+0.004567` | `122.67` |
| `random_init` | `-0.009217` | `+0.010398` | `122.67` |
| `supervised` | `-0.010692` | `+0.016624` | `122.67` |
| `swav` | `-0.049530` | `+0.088623` | `122.67` |

Because insertion should increase and deletion should decrease, these deltas indicate that Grad-CAM++ is generally weaker than Grad-CAM in this run set, with strongest degradation on `swav`.

### Outcome labels

From `artifacts/metrics/gradcampp_diagnostics/outcome_label.json` and `artifacts/metrics/gradcampp_diagnostics/diagnostics_note.json`:

| Field | Value |
| --- | --- |
| Outcome label | `mixed` |
| Threshold | `0.01` |
| Note | `Grad-CAM++ diagnostics are mixed across conditions.` |

## Outputs

Below are the outputs written by the notebook:

| Category | Paths |
| --- | --- |
| Faithfulness tables | `artifacts/metrics/faithfulness/per_image_scores.csv`, `artifacts/metrics/faithfulness/seed_level_scores.csv`, `artifacts/metrics/faithfulness/condition_summary.csv`, `artifacts/metrics/faithfulness/paired_stats_primary.csv` |
| Faithfulness JSON | `artifacts/metrics/faithfulness/per_image_scores.json` |
| Grad-CAM++ diagnostics | `artifacts/metrics/gradcampp_diagnostics/seed_level_method_and_delta_scores.csv`, `artifacts/metrics/gradcampp_diagnostics/seed_level_deltas.csv`, `artifacts/metrics/gradcampp_diagnostics/condition_level_deltas.csv`, `artifacts/metrics/gradcampp_diagnostics/outcome_label.json`, `artifacts/metrics/gradcampp_diagnostics/diagnostics_note.json` |
| Saliency QC report | `artifacts/metrics/saliency/qc_report.json` |

These files are the canonical record of post-training explainability analysis for this project run.
