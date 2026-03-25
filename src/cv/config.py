"""Shared config dataclasses and project path helpers."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
DATA_ROOT = PROJECT_ROOT / "data"
