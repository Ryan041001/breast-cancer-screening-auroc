from __future__ import annotations

import csv
from pathlib import Path

from final_project.engine.submission import write_submission


def _write_template(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["breast_id", "pred_score"])
        writer.writeheader()
        writer.writerows(rows)


def test_submission_writer_preserves_template_order(tmp_path: Path) -> None:
    template_path = tmp_path / "template.csv"
    output_path = tmp_path / "submission.csv"
    _write_template(
        template_path,
        rows=[
            {"breast_id": "020_R", "pred_score": ""},
            {"breast_id": "010_L", "pred_score": ""},
        ],
    )

    write_submission(
        template_csv=template_path,
        predictions={"010_L": 0.25, "020_R": 0.75},
        output_csv=output_path,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"breast_id": "020_R", "pred_score": "0.75"},
        {"breast_id": "010_L", "pred_score": "0.25"},
    ]
