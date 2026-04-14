from __future__ import annotations

import csv
from pathlib import Path


def write_submission(
    template_csv: str | Path,
    predictions: dict[str, float],
    output_csv: str | Path,
) -> None:
    template_path = Path(template_csv)
    output_path = Path(output_csv)

    with template_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    serialized_rows: list[dict[str, str]] = []
    for row in rows:
        breast_id = row["breast_id"]
        if breast_id not in predictions:
            raise ValueError(f"Missing prediction for breast_id '{breast_id}'")
        serialized_rows.append(
            {
                "breast_id": breast_id,
                "pred_score": str(predictions[breast_id]),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["breast_id", "pred_score"])
        writer.writeheader()
        writer.writerows(serialized_rows)


def write_prediction_table(
    predictions: dict[str, float], output_csv: str | Path
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["breast_id", "pred_score"])
        writer.writeheader()
        for breast_id, score in predictions.items():
            writer.writerow({"breast_id": breast_id, "pred_score": str(score)})


def read_prediction_table(predictions_csv: str | Path) -> dict[str, float]:
    predictions_path = Path(predictions_csv)
    with predictions_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["breast_id"]: float(row["pred_score"]) for row in rows}
