"""Helpers for reading and writing JSON, CSV, and NPY artifacts."""

import json
from pathlib import Path
from typing import Any


def ensure_parent(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any, *, indent: int = 2) -> None:
    """Write a JSON payload to disk."""
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=indent), encoding="utf-8")


def read_json(path: Path) -> Any:
    """Read and parse a JSON payload from disk."""
    return json.loads(path.read_text(encoding="utf-8"))
