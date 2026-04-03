"""Aggregate probe accuracy results into mean +- std summaries."""

import argparse
import csv
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.config import ARTIFACTS_ROOT
from cv.train.metrics import summarize_test_accuracy_by_condition
from cv.train.trainer import RUN_TABLE_COLUMNS, build_run_table_row
from cv.utils.io import ensure_parent, read_json, write_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-metrics-root",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_runs",
        help="Directory containing per-run JSON payloads.",
    )
    parser.add_argument(
        "--run-table-json",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_run_table.json",
        help="Output path for run table JSON.",
    )
    parser.add_argument(
        "--run-table-csv",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_run_table.csv",
        help="Output path for run table CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_condition_summary.json",
        help="Output path for condition summary JSON.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=ARTIFACTS_ROOT / "metrics" / "probe_condition_summary.csv",
        help="Output path for condition summary CSV.",
    )
    return parser.parse_args()


def _load_run_rows(run_metrics_root: Path) -> list[dict[str, object]]:
    run_files = sorted(run_metrics_root.glob("*/*.json"))
    if not run_files:
        raise FileNotFoundError(
            f"No run metrics JSON files found under '{run_metrics_root}'."
        )

    run_rows: list[dict[str, object]] = []
    for run_file in run_files:
        payload = read_json(run_file)
        if not isinstance(payload, dict):
            raise ValueError(f"Run metrics file must contain a JSON object: {run_file}")
        run_rows.append(build_run_table_row(payload))

    run_rows.sort(key=lambda row: (str(row["condition"]), int(row["seed"])))
    return run_rows


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: list[str]
) -> None:
    ensure_parent(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """Entrypoint for probe result aggregation."""
    args = _parse_args()
    run_rows = _load_run_rows(args.run_metrics_root)
    summaries = summarize_test_accuracy_by_condition(run_rows)

    write_json(args.run_table_json, run_rows)
    _write_csv(args.run_table_csv, run_rows, fieldnames=list(RUN_TABLE_COLUMNS))

    write_json(args.summary_json, summaries)
    _write_csv(
        args.summary_csv,
        summaries,
        fieldnames=["condition", "num_seeds", "mean_test_acc", "std_test_acc"],
    )

    print(f"Saved run table JSON: {args.run_table_json}")
    print(f"Saved run table CSV: {args.run_table_csv}")
    print(f"Saved summary JSON: {args.summary_json}")
    print(f"Saved summary CSV: {args.summary_csv}")

    for summary in summaries:
        print(
            f"{summary['condition']}: "
            f"mean_test_acc={summary['mean_test_acc']:.4f} "
            f"std_test_acc={summary['std_test_acc']:.4f} "
            f"(n={summary['num_seeds']})"
        )


if __name__ == "__main__":
    main()
