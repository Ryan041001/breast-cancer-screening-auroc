from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from final_project.config import load_config


ALLOWED_PATHOLOGIES = {"malignant", "benign", "normal"}
ALLOWED_LATERALITY = {"L", "R"}
ALLOWED_VIEWS = {"CC", "MLO"}


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a high-confidence paired external auxiliary CSV."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_mammonet32k_warmup_e4_lr5e4.yaml"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/data/external_paired_highconf.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/data/external_paired_highconf_summary.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.paths.external_catalog is None:
        raise ValueError("Config must resolve external catalog path")

    rows = _read_csv_rows(config.paths.external_catalog)
    grouped: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pathology = (row.get("pathology") or "").strip().lower()
        laterality = (row.get("laterality") or "").strip().upper()
        view = (row.get("view") or "").strip().upper()
        if pathology not in ALLOWED_PATHOLOGIES:
            continue
        if laterality not in ALLOWED_LATERALITY:
            continue
        if view not in ALLOWED_VIEWS:
            continue
        key = (
            row["dataset"].strip().lower(),
            row["patient_id"].strip(),
            laterality,
        )
        grouped[key].append(row)

    output_rows: list[dict[str, object]] = []
    pathology_counts = Counter()
    binary_counts = Counter()
    skipped_no_pair = 0
    skipped_conflict = 0
    for (dataset, patient_id, laterality), items in sorted(grouped.items()):
        cc_paths = sorted(
            {
                row["processed_path"]
                for row in items
                if (row.get("view") or "").strip().upper() == "CC"
            }
        )
        mlo_paths = sorted(
            {
                row["processed_path"]
                for row in items
                if (row.get("view") or "").strip().upper() == "MLO"
            }
        )
        if not cc_paths or not mlo_paths:
            skipped_no_pair += 1
            continue
        pathologies = {
            (row.get("pathology") or "").strip().lower()
            for row in items
            if (row.get("pathology") or "").strip().lower() in ALLOWED_PATHOLOGIES
        }
        if "malignant" in pathologies and "benign" in pathologies:
            skipped_conflict += 1
            continue
        if "malignant" in pathologies:
            pathology_label = "malignant"
            binary_label = 1
        elif "benign" in pathologies:
            pathology_label = "benign"
            binary_label = 0
        else:
            pathology_label = "normal"
            binary_label = 0
        pathology_counts[pathology_label] += 1
        binary_counts[binary_label] += 1
        output_rows.append(
            {
                "external_breast_id": f"{dataset}:{patient_id}:{laterality}",
                "dataset": dataset,
                "patient_id": patient_id,
                "laterality": laterality,
                "binary_label": binary_label,
                "pathology_label": pathology_label,
                "primary_cc_path": cc_paths[0],
                "primary_mlo_path": mlo_paths[0],
                "cc_paths": ";".join(cc_paths),
                "mlo_paths": ";".join(mlo_paths),
                "cc_count": len(cc_paths),
                "mlo_count": len(mlo_paths),
                "image_count": len(items),
                "source_pathologies": ";".join(sorted(pathologies)),
            }
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()) if output_rows else [
            "external_breast_id",
            "dataset",
            "patient_id",
            "laterality",
            "binary_label",
            "pathology_label",
            "primary_cc_path",
            "primary_mlo_path",
            "cc_paths",
            "mlo_paths",
            "cc_count",
            "mlo_count",
            "image_count",
            "source_pathologies",
        ])
        writer.writeheader()
        for row in output_rows:
            writer.writerow(row)

    summary = {
        "output_csv": str(args.output_csv.resolve()),
        "paired_breasts": len(output_rows),
        "pathology_counts": dict(pathology_counts),
        "binary_counts": dict(binary_counts),
        "skipped_no_pair": skipped_no_pair,
        "skipped_conflict": skipped_conflict,
        "rules": {
            "group_key": ["dataset", "patient_id", "laterality"],
            "allowed_laterality": sorted(ALLOWED_LATERALITY),
            "allowed_views": sorted(ALLOWED_VIEWS),
            "pathology_priority": ["malignant", "benign", "normal"],
            "drop_conflict": "malignant+benign",
            "usage": "train_only_auxiliary_not_mainline",
        },
    }
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"paired-highconf: csv={args.output_csv} summary={args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
