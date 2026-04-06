"""Aggregation helpers for Stage-7/8/9 evaluation tables."""

from collections import defaultdict
from collections.abc import Iterable

import numpy as np


def compute_primary_correct_intersection(
    metadata_rows: Iterable[dict[str, object]],
    *,
    conditions: list[str],
    seeds: list[int],
) -> dict[int, list[int]]:
    """Return per-seed image ids correct for all requested conditions.

    This implements the strict primary-slice policy: intersection of correctly
    classified images across all compared conditions for each seed.
    """
    if not conditions:
        raise ValueError("conditions must contain at least one condition.")
    if not seeds:
        raise ValueError("seeds must contain at least one seed.")

    allowed_conditions = set(conditions)
    allowed_seeds = set(seeds)

    available_ids_by_key: dict[tuple[str, int], set[int]] = defaultdict(set)
    correct_ids_by_key: dict[tuple[str, int], set[int]] = defaultdict(set)

    for row in metadata_rows:
        condition = row.get("condition")
        seed = row.get("seed")
        image_id = row.get("test_image_id")
        correct = row.get("correct")
        if not isinstance(condition, str):
            raise ValueError(f"Metadata row missing condition: {row}")
        if not isinstance(seed, int):
            raise ValueError(f"Metadata row missing seed: {row}")
        if not isinstance(image_id, int):
            raise ValueError(f"Metadata row missing test_image_id: {row}")
        if not isinstance(correct, bool):
            raise ValueError(f"Metadata row missing correct flag: {row}")

        if condition not in allowed_conditions or seed not in allowed_seeds:
            continue

        key = (condition, seed)
        available_ids_by_key[key].add(image_id)
        if correct:
            correct_ids_by_key[key].add(image_id)

    intersections: dict[int, list[int]] = {}
    for seed in seeds:
        condition_sets: list[set[int]] = []
        for condition in conditions:
            key = (condition, seed)
            if key not in available_ids_by_key:
                raise ValueError(
                    "Missing metadata rows for required condition/seed key: "
                    f"condition={condition}, seed={seed}."
                )
            condition_sets.append(correct_ids_by_key.get(key, set()))

        shared = set.intersection(*condition_sets) if condition_sets else set()
        intersections[seed] = sorted(int(image_id) for image_id in shared)

    return intersections


def summarize_seed_level_metrics(
    per_image_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-image rows into seed-level summaries."""
    grouped: dict[
        tuple[str, int, str, str],
        dict[str, list[float] | set[int] | list[bool]],
    ] = defaultdict(
        lambda: {
            "insertion_auc": [],
            "deletion_auc": [],
            "norm_insertion_auc": [],
            "norm_deletion_auc": [],
            "target_logit_original": [],
            "drop_top10": [],
            "drop_top20": [],
            "flip_top10": [],
            "flip_top20": [],
            "image_ids": set(),
        }
    )

    for row in per_image_rows:
        condition = row.get("condition")
        seed = row.get("seed")
        method = row.get("method")
        slice_name = row.get("slice")
        image_id = row.get("test_image_id")
        insertion_auc = row.get("insertion_auc")
        deletion_auc = row.get("deletion_auc")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in per-image row: {row}")
        if not isinstance(seed, int):
            raise ValueError(f"Invalid seed in per-image row: {row}")
        if not isinstance(method, str):
            raise ValueError(f"Invalid method in per-image row: {row}")
        if not isinstance(slice_name, str):
            raise ValueError(f"Invalid slice in per-image row: {row}")
        if not isinstance(image_id, int):
            raise ValueError(f"Invalid test_image_id in per-image row: {row}")
        if not isinstance(insertion_auc, (int, float)):
            raise ValueError(f"Invalid insertion_auc in per-image row: {row}")
        if not isinstance(deletion_auc, (int, float)):
            raise ValueError(f"Invalid deletion_auc in per-image row: {row}")

        key = (condition, seed, method, slice_name)
        bucket = grouped[key]
        bucket["insertion_auc"].append(float(insertion_auc))
        bucket["deletion_auc"].append(float(deletion_auc))
        bucket["image_ids"].add(image_id)

        target_logit_original = row.get("target_logit_original")
        if isinstance(target_logit_original, (int, float)) and float(target_logit_original) > 1e-8:
            t = float(target_logit_original)
            bucket["target_logit_original"].append(t)
            bucket["norm_insertion_auc"].append(float(insertion_auc) / t)
            bucket["norm_deletion_auc"].append(float(deletion_auc) / t)

        drop_top10 = row.get("drop_top10")
        drop_top20 = row.get("drop_top20")
        flip_top10 = row.get("flip_top10")
        flip_top20 = row.get("flip_top20")
        if isinstance(drop_top10, (int, float)):
            bucket["drop_top10"].append(float(drop_top10))
        if isinstance(drop_top20, (int, float)):
            bucket["drop_top20"].append(float(drop_top20))
        if isinstance(flip_top10, bool):
            bucket["flip_top10"].append(bool(flip_top10))
        if isinstance(flip_top20, bool):
            bucket["flip_top20"].append(bool(flip_top20))

    summary_rows: list[dict[str, object]] = []
    for (condition, seed, method, slice_name), bucket in sorted(grouped.items()):
        insertion_values = np.asarray(bucket["insertion_auc"], dtype=np.float64)
        deletion_values = np.asarray(bucket["deletion_auc"], dtype=np.float64)
        image_ids = bucket["image_ids"]
        summary_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "method": method,
                "slice": slice_name,
                "n_images": len(image_ids),
                "mean_insertion_auc": float(np.mean(insertion_values)),
                "mean_deletion_auc": float(np.mean(deletion_values)),
                "mean_target_logit_original": (
                    float(np.mean(bucket["target_logit_original"]))
                    if bucket["target_logit_original"]
                    else None
                ),
                "mean_norm_insertion_auc": (
                    float(np.mean(bucket["norm_insertion_auc"]))
                    if bucket["norm_insertion_auc"]
                    else None
                ),
                "mean_norm_deletion_auc": (
                    float(np.mean(bucket["norm_deletion_auc"]))
                    if bucket["norm_deletion_auc"]
                    else None
                ),
                "mean_drop_top10": (
                    float(np.mean(bucket["drop_top10"]))
                    if bucket["drop_top10"]
                    else None
                ),
                "mean_drop_top20": (
                    float(np.mean(bucket["drop_top20"]))
                    if bucket["drop_top20"]
                    else None
                ),
                "flip_rate_top10": (
                    float(np.mean(bucket["flip_top10"]))
                    if bucket["flip_top10"]
                    else None
                ),
                "flip_rate_top20": (
                    float(np.mean(bucket["flip_top20"]))
                    if bucket["flip_top20"]
                    else None
                ),
            }
        )

    return summary_rows


def _mean_and_std(values: list[float]) -> tuple[float, float]:
    values_array = np.asarray(values, dtype=np.float64)
    if values_array.size == 0:
        raise ValueError("Expected at least one value.")
    mean = float(np.mean(values_array))
    if values_array.size == 1:
        return mean, 0.0
    std = float(np.std(values_array, ddof=1))
    return mean, std


def summarize_condition_level_metrics(
    seed_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate seed-level rows into condition-level mean/std summaries."""
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in seed_rows:
        condition = row.get("condition")
        method = row.get("method")
        slice_name = row.get("slice")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in seed row: {row}")
        if not isinstance(method, str):
            raise ValueError(f"Invalid method in seed row: {row}")
        if not isinstance(slice_name, str):
            raise ValueError(f"Invalid slice in seed row: {row}")
        grouped[(condition, method, slice_name)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, method, slice_name), rows in sorted(grouped.items()):
        insertion = [float(row["mean_insertion_auc"]) for row in rows]
        deletion = [float(row["mean_deletion_auc"]) for row in rows]
        n_images = [int(row["n_images"]) for row in rows]
        ins_mean, ins_std = _mean_and_std(insertion)
        del_mean, del_std = _mean_and_std(deletion)
        summary_rows.append(
            {
                "condition": condition,
                "method": method,
                "slice": slice_name,
                "n_seeds": len(rows),
                "mean_insertion_auc": ins_mean,
                "std_insertion_auc": ins_std,
                "mean_deletion_auc": del_mean,
                "std_deletion_auc": del_std,
                "mean_n_images": float(np.mean(np.asarray(n_images, dtype=np.float64))),
                "min_n_images": int(min(n_images)),
                "max_n_images": int(max(n_images)),
            }
        )

    return summary_rows


def compute_method_deltas(
    per_image_rows: Iterable[dict[str, object]],
    *,
    method_a: str,
    method_b: str,
) -> list[dict[str, object]]:
    """Compute per-image deltas ``method_b - method_a`` for matched keys."""
    if method_a == method_b:
        raise ValueError("method_a and method_b must differ.")

    rows_by_method: dict[
        str,
        dict[tuple[str, int, str, int], dict[str, object]],
    ] = {method_a: {}, method_b: {}}

    for row in per_image_rows:
        method = row.get("method")
        if method not in rows_by_method:
            continue

        condition = row.get("condition")
        seed = row.get("seed")
        slice_name = row.get("slice")
        image_id = row.get("test_image_id")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in per-image row: {row}")
        if not isinstance(seed, int):
            raise ValueError(f"Invalid seed in per-image row: {row}")
        if not isinstance(slice_name, str):
            raise ValueError(f"Invalid slice in per-image row: {row}")
        if not isinstance(image_id, int):
            raise ValueError(f"Invalid test_image_id in per-image row: {row}")

        key = (condition, seed, slice_name, image_id)
        if key in rows_by_method[method]:
            raise ValueError(
                "Duplicate method row for key "
                f"(condition={condition}, seed={seed}, slice={slice_name}, image={image_id}, method={method})."
            )
        rows_by_method[method][key] = row

    shared_keys = sorted(
        set(rows_by_method[method_a].keys()) & set(rows_by_method[method_b].keys())
    )
    delta_rows: list[dict[str, object]] = []
    for condition, seed, slice_name, image_id in shared_keys:
        row_a = rows_by_method[method_a][(condition, seed, slice_name, image_id)]
        row_b = rows_by_method[method_b][(condition, seed, slice_name, image_id)]

        insertion_a = float(row_a["insertion_auc"])
        insertion_b = float(row_b["insertion_auc"])
        deletion_a = float(row_a["deletion_auc"])
        deletion_b = float(row_b["deletion_auc"])
        delta_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "slice": slice_name,
                "test_image_id": image_id,
                "method_a": method_a,
                "method_b": method_b,
                "insertion_auc_a": insertion_a,
                "insertion_auc_b": insertion_b,
                "deletion_auc_a": deletion_a,
                "deletion_auc_b": deletion_b,
                "delta_insertion_auc": insertion_b - insertion_a,
                "delta_deletion_auc": deletion_b - deletion_a,
            }
        )

    return delta_rows


def summarize_seed_level_deltas(
    delta_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate per-image deltas to seed-level means."""
    grouped: dict[
        tuple[str, int, str, str, str],
        list[dict[str, object]],
    ] = defaultdict(list)

    for row in delta_rows:
        condition = row.get("condition")
        seed = row.get("seed")
        slice_name = row.get("slice")
        method_a = row.get("method_a")
        method_b = row.get("method_b")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in delta row: {row}")
        if not isinstance(seed, int):
            raise ValueError(f"Invalid seed in delta row: {row}")
        if not isinstance(slice_name, str):
            raise ValueError(f"Invalid slice in delta row: {row}")
        if not isinstance(method_a, str) or not isinstance(method_b, str):
            raise ValueError(f"Invalid methods in delta row: {row}")

        grouped[(condition, seed, slice_name, method_a, method_b)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, seed, slice_name, method_a, method_b), rows in sorted(
        grouped.items()
    ):
        delta_ins = [float(row["delta_insertion_auc"]) for row in rows]
        delta_del = [float(row["delta_deletion_auc"]) for row in rows]
        summary_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "slice": slice_name,
                "method_a": method_a,
                "method_b": method_b,
                "n_images": len(rows),
                "mean_delta_insertion_auc": float(np.mean(np.asarray(delta_ins))),
                "mean_delta_deletion_auc": float(np.mean(np.asarray(delta_del))),
            }
        )

    return summary_rows


def summarize_condition_level_deltas(
    seed_delta_rows: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Aggregate seed-level delta rows into condition-level mean/std summaries."""
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in seed_delta_rows:
        condition = row.get("condition")
        slice_name = row.get("slice")
        method_a = row.get("method_a")
        method_b = row.get("method_b")
        if not isinstance(condition, str):
            raise ValueError(f"Invalid condition in seed delta row: {row}")
        if not isinstance(slice_name, str):
            raise ValueError(f"Invalid slice in seed delta row: {row}")
        if not isinstance(method_a, str) or not isinstance(method_b, str):
            raise ValueError(f"Invalid methods in seed delta row: {row}")
        grouped[(condition, slice_name, method_a, method_b)].append(row)

    summary_rows: list[dict[str, object]] = []
    for (condition, slice_name, method_a, method_b), rows in sorted(grouped.items()):
        delta_ins = [float(row["mean_delta_insertion_auc"]) for row in rows]
        delta_del = [float(row["mean_delta_deletion_auc"]) for row in rows]
        n_images = [int(row["n_images"]) for row in rows]
        ins_mean, ins_std = _mean_and_std(delta_ins)
        del_mean, del_std = _mean_and_std(delta_del)
        summary_rows.append(
            {
                "condition": condition,
                "slice": slice_name,
                "method_a": method_a,
                "method_b": method_b,
                "n_seeds": len(rows),
                "mean_delta_insertion_auc": ins_mean,
                "std_delta_insertion_auc": ins_std,
                "mean_delta_deletion_auc": del_mean,
                "std_delta_deletion_auc": del_std,
                "mean_n_images": float(np.mean(np.asarray(n_images, dtype=np.float64))),
                "min_n_images": int(min(n_images)),
                "max_n_images": int(max(n_images)),
            }
        )

    return summary_rows


def classify_gradcampp_outcome(
    condition_delta_rows: Iterable[dict[str, object]],
    *,
    threshold: float = 0.01,
    min_conditions: int = 3,
) -> str:
    """Assign Stage-9 interpretation label from condition-level deltas."""
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if min_conditions <= 0:
        raise ValueError(f"min_conditions must be positive, got {min_conditions}.")

    reinforce = 0
    weaken = 0
    neutral = 0

    for row in condition_delta_rows:
        delta_ins = row.get("mean_delta_insertion_auc")
        delta_del = row.get("mean_delta_deletion_auc")
        if not isinstance(delta_ins, (int, float)) or not isinstance(
            delta_del, (int, float)
        ):
            raise ValueError(f"Missing delta values in row: {row}")

        if float(delta_ins) >= threshold and float(delta_del) <= -threshold:
            reinforce += 1
        elif float(delta_ins) <= -threshold and float(delta_del) >= threshold:
            weaken += 1
        elif abs(float(delta_ins)) < threshold and abs(float(delta_del)) < threshold:
            neutral += 1

    if reinforce >= min_conditions:
        return "reinforce"
    if weaken >= min_conditions:
        return "weaken"
    if neutral >= min_conditions:
        return "neutral"
    return "mixed"
