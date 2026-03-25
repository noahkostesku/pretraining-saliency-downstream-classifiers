"""Core project configuration and resolved paths."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathsConfig:
    """Resolved repository paths used across pipeline stages."""

    project_root: Path
    src_root: Path
    data_root: Path
    artifacts_root: Path
    reports_root: Path
    external_checkpoints_root: Path


def build_paths(project_root: Path | None = None) -> PathsConfig:
    """Build path config relative to the project root."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[3]

    return PathsConfig(
        project_root=project_root,
        src_root=project_root / "src",
        data_root=project_root / "data",
        artifacts_root=project_root / "artifacts",
        reports_root=project_root / "reports",
        external_checkpoints_root=project_root / "data" / "external",
    )


DEFAULT_PATHS = build_paths()

PROJECT_ROOT = DEFAULT_PATHS.project_root
SRC_ROOT = DEFAULT_PATHS.src_root
DATA_ROOT = DEFAULT_PATHS.data_root
ARTIFACTS_ROOT = DEFAULT_PATHS.artifacts_root
REPORTS_ROOT = DEFAULT_PATHS.reports_root
EXTERNAL_CHECKPOINTS_ROOT = DEFAULT_PATHS.external_checkpoints_root
