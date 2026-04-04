"""Quality-control checks for saved explanation artifacts."""

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from cv.config import ARTIFACTS_ROOT
from cv.data import load_eval_subset
from cv.utils.io import write_json

from .saliency_io import load_saliency_map, read_saliency_metadata


def _resolve_artifacts_root(artifacts_root: str | Path | None) -> Path:
    if artifacts_root is None:
        return ARTIFACTS_ROOT
    return Path(artifacts_root)


def _validate_map(path: Path) -> None:
    payload = load_saliency_map(path)
    if payload.dtype != np.float32:
        raise ValueError(
            f"Expected float32 saliency map at {path}, got {payload.dtype}"
        )


def run_explanation_qc(
    *,
    artifacts_root: str | Path | None = None,
    conditions: list[str] | None = None,
    seeds: list[int] | None = None,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Run Stage-6 quality-control checks for generated saliency artifacts."""
    root = _resolve_artifacts_root(artifacts_root)
    subset = load_eval_subset(artifacts_root=artifacts_root)
    expected_ids = set(int(index) for index in subset.indices)
    expected_count = len(expected_ids)

    saliency_root = root / "saliency"
    allowed_conditions = set(conditions) if conditions is not None else None
    allowed_seeds = set(seeds) if seeds is not None else None
    allowed_methods = set(methods) if methods is not None else None

    errors: list[dict[str, str]] = []
    rows: list[dict[str, Any]] = []
    image_ids_by_key: dict[tuple[str, int, str], set[int]] = defaultdict(set)

    if not saliency_root.exists():
        raise FileNotFoundError(
            f"Saliency root does not exist: {saliency_root}. Run explanation generation first."
        )

    for condition_dir in sorted(saliency_root.glob("*")):
        if not condition_dir.is_dir():
            continue
        condition = condition_dir.name
        if allowed_conditions is not None and condition not in allowed_conditions:
            continue

        for seed_dir in sorted(condition_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            seed_label = seed_dir.name
            if not seed_label.startswith("seed_"):
                continue
            try:
                seed = int(seed_label.replace("seed_", "", 1))
            except ValueError:
                errors.append(
                    {
                        "scope": f"{condition}/{seed_label}",
                        "error": "Invalid seed directory name.",
                    }
                )
                continue

            if allowed_seeds is not None and seed not in allowed_seeds:
                continue

            for method_dir in sorted(seed_dir.glob("*")):
                if not method_dir.is_dir():
                    continue
                method = method_dir.name
                if allowed_methods is not None and method not in allowed_methods:
                    continue

                metadata_path = method_dir / "metadata.json"
                if not metadata_path.exists():
                    errors.append(
                        {
                            "scope": f"{condition}/seed_{seed}/{method}",
                            "error": "Missing metadata.json.",
                        }
                    )
                    continue

                method_rows = read_saliency_metadata(metadata_path)
                if not method_rows:
                    errors.append(
                        {
                            "scope": f"{condition}/seed_{seed}/{method}",
                            "error": "Empty metadata rows.",
                        }
                    )
                    continue

                for row in method_rows:
                    map_path_value = row.get("saliency_map_path")
                    image_id_value = row.get("test_image_id")
                    if not isinstance(map_path_value, str) or not isinstance(
                        image_id_value, int
                    ):
                        errors.append(
                            {
                                "scope": f"{condition}/seed_{seed}/{method}",
                                "error": "Row is missing saliency_map_path or test_image_id.",
                            }
                        )
                        continue

                    map_path = Path(map_path_value)
                    if not map_path.exists():
                        errors.append(
                            {
                                "scope": f"{condition}/seed_{seed}/{method}",
                                "error": f"Missing map file: {map_path}",
                            }
                        )
                        continue

                    try:
                        _validate_map(map_path)
                    except Exception as exc:
                        errors.append(
                            {
                                "scope": f"{condition}/seed_{seed}/{method}",
                                "error": f"Invalid map {map_path}: {exc}",
                            }
                        )
                        continue

                    image_ids_by_key[(condition, seed, method)].add(image_id_value)
                    rows.append(row)

    coverage_rows: list[dict[str, Any]] = []
    grouped_by_condition_seed: dict[tuple[str, int], list[set[int]]] = defaultdict(list)
    for (condition, seed, method), image_ids in sorted(image_ids_by_key.items()):
        coverage_rows.append(
            {
                "condition": condition,
                "seed": seed,
                "method": method,
                "num_images": len(image_ids),
                "expected_images": expected_count,
                "complete": len(image_ids) == expected_count,
            }
        )

        grouped_by_condition_seed[(condition, seed)].append(image_ids)
        if image_ids != expected_ids:
            errors.append(
                {
                    "scope": f"{condition}/seed_{seed}/{method}",
                    "error": "Image-id set does not match fixed eval subset.",
                }
            )

    for (condition, seed), id_sets in grouped_by_condition_seed.items():
        baseline = id_sets[0]
        for image_ids in id_sets[1:]:
            if image_ids != baseline:
                errors.append(
                    {
                        "scope": f"{condition}/seed_{seed}",
                        "error": "Method image-id sets do not match within run.",
                    }
                )
                break

    report: dict[str, Any] = {
        "num_rows": len(rows),
        "num_errors": len(errors),
        "expected_subset_count": expected_count,
        "coverage": coverage_rows,
        "errors": errors,
        "passed": len(errors) == 0,
    }
    return report


def write_explanation_qc_report(
    *,
    report: dict[str, Any],
    artifacts_root: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Persist explanation QC report JSON to disk."""
    root = _resolve_artifacts_root(artifacts_root)
    destination = (
        Path(output_path)
        if output_path is not None
        else root / "metrics" / "saliency" / "qc_report.json"
    )
    write_json(destination, report)
    return destination
