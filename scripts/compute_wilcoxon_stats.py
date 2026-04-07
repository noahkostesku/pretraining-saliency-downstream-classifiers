"""
Compute paired Wilcoxon signed-rank tests on per-image Grad-CAM insertion AUC
for all pairwise condition comparisons on the primary evaluation slice.

Output: artifacts/metrics/faithfulness/wilcoxon_paired_stats_primary.csv
"""

import csv
from collections import defaultdict
from scipy import stats
from pathlib import Path

PER_IMAGE_SCORES = Path("artifacts/metrics/faithfulness/faithfulness/per_image_scores.csv")
OUTPUT = Path("artifacts/metrics/faithfulness/wilcoxon_paired_stats_primary.csv")

COMPARISONS = [
    ("supervised", "moco"),
    ("supervised", "swav"),
    ("supervised", "random_init"),
    ("swav", "moco"),
    ("random_init", "moco"),
    ("swav", "random_init"),
]

SEEDS = ["0", "1", "2"]


def main():
    rows = []
    with open(PER_IMAGE_SCORES) as f:
        for row in csv.DictReader(f):
            rows.append(row)

    # Index: condition -> seed -> image_id -> insertion_auc
    data = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["method"] == "gradcam" and r["slice"] == "primary":
            data[r["condition"]][r["seed"]][r["test_image_id"]] = float(r["insertion_auc"])

    output = []
    for ca, cb in COMPARISONS:
        for seed in SEEDS:
            ids_a = set(data[ca][seed].keys())
            ids_b = set(data[cb][seed].keys())
            common = ids_a & ids_b
            a_vals = [data[ca][seed][i] for i in common]
            b_vals = [data[cb][seed][i] for i in common]
            w, p = stats.wilcoxon(a_vals, b_vals)
            output.append({
                "condition_a": ca,
                "condition_b": cb,
                "seed": seed,
                "n_images": len(common),
                "wilcoxon_statistic": round(w, 4),
                "wilcoxon_pvalue": round(p, 6),
            })
            print(f"{ca} vs {cb} seed {seed}: W={w:.1f}, p={p:.4f}, n={len(common)}")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    print(f"\nSaved {len(output)} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
