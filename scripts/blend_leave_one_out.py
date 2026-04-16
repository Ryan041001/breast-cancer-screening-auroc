from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-out prune for a saved linear blend.")
    parser.add_argument(
        "--blend-json",
        type=Path,
        required=True,
        help="Path to outputs/runs/<blend>/blend.json",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path("train.csv"),
        help="Training manifest CSV used to rebuild binary labels.",
    )
    parser.add_argument(
        "--auto-prune",
        action="store_true",
        help="Iteratively remove members whose removal improves OOF AUC.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blend = json.loads(args.blend_json.read_text(encoding="utf-8"))
    runs_root = args.blend_json.parent.parent
    labels = (
        pd.read_csv(args.train_csv)
        .groupby("breast_id")["pathology"]
        .apply(lambda series: int((series == "M").any()))
        .reset_index(name="label")
    )
    base = labels.copy()
    for alias, run in blend["run_map"].items():
        oof_path = runs_root / run / "cv" / "oof_predictions.csv"
        df = pd.read_csv(oof_path).rename(columns={"pred_score": alias})
        base = base.merge(df[["breast_id", alias]], on="breast_id")

    def auc_for(weights: dict[str, float]) -> float:
        total_weight = sum(weights.values())
        pred = sum((weight / total_weight) * base[alias] for alias, weight in weights.items())
        return roc_auc_score(base["label"], pred)

    current_weights = dict(blend["weights"])
    reports: list[dict[str, object]] = []
    while True:
        full_auc = auc_for(current_weights)
        loo_rows: list[dict[str, object]] = []
        best_row: dict[str, object] | None = None
        for alias, weight in current_weights.items():
            remaining = {key: value for key, value in current_weights.items() if key != alias}
            auc = auc_for(remaining)
            row = {
                "alias": alias,
                "weight": weight,
                "auc_without_member": auc,
                "delta": auc - full_auc,
            }
            loo_rows.append(row)
            if best_row is None or row["delta"] > best_row["delta"]:
                best_row = row
        reports.append(
            {
                "full_auc": full_auc,
                "members": list(current_weights.keys()),
                "leave_one_out": sorted(loo_rows, key=lambda row: row["delta"]),
                "best_removal": best_row,
            }
        )
        if not args.auto_prune or best_row is None or float(best_row["delta"]) <= 0.0:
            break
        current_weights.pop(str(best_row["alias"]))

    print(json.dumps({"reports": reports, "final_weights": current_weights}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
