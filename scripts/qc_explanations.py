"""Run Stage-6 QC checks for generated explanation artifacts."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cv.explain import run_explanation_qc, write_explanation_qc_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=["supervised", "moco", "swav", "random_init"],
        help="Optional subset of conditions to check.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of seeds to check.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        choices=["gradcam", "gradcampp", "occlusion"],
        help="Optional subset of methods to check.",
    )
    parser.add_argument(
        "--artifacts-root",
        type=str,
        default=None,
        help="Optional artifacts root override.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path for QC report JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Entrypoint for Stage-6 explanation QC."""
    args = _parse_args()
    report = run_explanation_qc(
        artifacts_root=args.artifacts_root,
        conditions=args.conditions,
        seeds=args.seeds,
        methods=args.methods,
    )
    output_path = write_explanation_qc_report(
        report=report,
        artifacts_root=args.artifacts_root,
        output_path=args.output,
    )

    print(f"Saved QC report: {output_path}")
    print(f"Rows checked: {report['num_rows']}")
    print(f"Errors: {report['num_errors']}")
    print(f"Passed: {report['passed']}")

    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
