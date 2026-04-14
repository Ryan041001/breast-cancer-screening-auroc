from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OutputPaths:
    root: Path
    checkpoints: Path
    submissions: Path
    runs: Path


def build_output_paths(output_root: str | Path) -> OutputPaths:
    root = Path(output_root)
    checkpoints = root / "checkpoints"
    submissions = root / "submissions"
    runs = root / "runs"
    for path in (root, checkpoints, submissions, runs):
        path.mkdir(parents=True, exist_ok=True)
    return OutputPaths(
        root=root,
        checkpoints=checkpoints,
        submissions=submissions,
        runs=runs,
    )
