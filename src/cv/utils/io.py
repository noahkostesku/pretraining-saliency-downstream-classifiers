"""Helpers for reading and writing JSON, CSV, and NPY artifacts."""

from pathlib import Path


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
