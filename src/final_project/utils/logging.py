from __future__ import annotations

from datetime import datetime
from pathlib import Path


def log_message(output_dir: str | Path, message: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (output_path / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{line}\n")
