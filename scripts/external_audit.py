from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from final_project.config import load_config
from final_project.data.external import (
    EXTERNAL_DATA_CONTRACT_VERSION,
    NEGATIVE_PATHOLOGIES,
    POSITIVE_PATHOLOGIES,
    load_external_split_records,
)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _normalize_token(value: str | None) -> str:
    return (value or "").strip().lower() or "unknown"


def _sample_id_from_row(row: dict[str, str]) -> str:
    processed_path = Path(row["processed_path"])
    return (
        f"{row['dataset'].strip().lower()}:"
        f"{row['patient_id'].strip()}:"
        f"{(row.get('laterality') or 'U').strip().upper()}:"
        f"{(row.get('view') or 'U').strip().upper()}:"
        f"{processed_path.stem}"
    )


def _build_split_summary(records: list[object]) -> dict[str, object]:
    from final_project.data.external import ExternalImageRecord

    typed_records = [record for record in records if isinstance(record, ExternalImageRecord)]
    return {
        "count": len(typed_records),
        "label_counts": dict(Counter(record.label for record in typed_records)),
        "dataset_counts": dict(Counter(record.dataset for record in typed_records)),
        "patient_count": len({(record.dataset, record.patient_id) for record in typed_records}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit the external warm-up dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/baseline_mammonet32k_warmup_e4_lr5e4.yaml"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/data/external_audit_summary.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("outputs/data/external_audit_summary.md"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.paths.external_catalog is None or config.paths.external_splits_dir is None:
        raise ValueError("Config must resolve external catalog and splits paths")
    if config.paths.external_data_root is None:
        raise ValueError("Config must resolve external data root")

    catalog_rows = _read_csv_rows(config.paths.external_catalog)
    processed_root = config.paths.external_data_root / "processed"
    split_paths = {
        "train": config.paths.external_splits_dir / "train.csv",
        "val": config.paths.external_splits_dir / "val.csv",
        "test": config.paths.external_splits_dir / "test.csv",
    }
    effective_records = {
        split_name: load_external_split_records(
            catalog_csv=config.paths.external_catalog,
            processed_root=processed_root,
            split_csv=split_path,
            max_samples=None,
        )
        for split_name, split_path in split_paths.items()
    }

    raw_sample_ids = Counter(_sample_id_from_row(row) for row in catalog_rows)
    raw_processed_paths = Counter(row["processed_path"] for row in catalog_rows)
    raw_patient_sets: dict[str, set[tuple[str, str]]] = {}
    raw_path_sets: dict[str, set[str]] = {}
    split_row_counts: dict[str, int] = {}
    for split_name, split_path in split_paths.items():
        rows = _read_csv_rows(split_path)
        split_row_counts[split_name] = len(rows)
        raw_patient_sets[split_name] = {
            (row["dataset"].strip().lower(), row["patient_id"].strip())
            for row in rows
        }
        raw_path_sets[split_name] = {row["processed_path"] for row in rows}

    allowed_pathology = POSITIVE_PATHOLOGIES | NEGATIVE_PATHOLOGIES
    allowed_laterality = {"l", "r", "u"}
    allowed_view = {"cc", "mlo", "ml", "u"}

    abnormal_pathology = Counter()
    abnormal_laterality = Counter()
    abnormal_view = Counter()
    mixed_patient_map: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    missing_processed_files = 0
    for row in catalog_rows:
        pathology = _normalize_token(row.get("pathology"))
        laterality = _normalize_token(row.get("laterality"))
        view = _normalize_token(row.get("view"))
        if pathology not in allowed_pathology:
            abnormal_pathology[pathology] += 1
        if laterality not in allowed_laterality:
            abnormal_laterality[laterality] += 1
        if view not in allowed_view:
            abnormal_view[view] += 1
        mixed_patient_map[
            (row["dataset"].strip().lower(), row["patient_id"].strip())
        ].add(pathology)
        candidate = processed_root / Path(row["processed_path"])
        if not candidate.exists():
            missing_processed_files += 1

    overlap_summary = {}
    split_names = list(split_paths)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap_summary[f"{left}_{right}"] = {
                "patient_overlap": len(raw_patient_sets[left] & raw_patient_sets[right]),
                "processed_path_overlap": len(raw_path_sets[left] & raw_path_sets[right]),
            }

    mixed_combo_counts = Counter(
        tuple(sorted(pathologies))
        for pathologies in mixed_patient_map.values()
        if len(pathologies) > 1
    )
    summary = {
        "contract_version": EXTERNAL_DATA_CONTRACT_VERSION,
        "catalog_path": str(config.paths.external_catalog.resolve()),
        "processed_root": str(processed_root.resolve()),
        "total_catalog_rows": len(catalog_rows),
        "raw_dataset_counts": dict(
            Counter(row["dataset"].strip().lower() for row in catalog_rows)
        ),
        "raw_pathology_counts": dict(
            Counter(_normalize_token(row.get("pathology")) for row in catalog_rows)
        ),
        "raw_laterality_counts": dict(
            Counter(_normalize_token(row.get("laterality")) for row in catalog_rows)
        ),
        "raw_view_counts": dict(
            Counter(_normalize_token(row.get("view")) for row in catalog_rows)
        ),
        "unknown_counts": {
            "pathology": sum(
                1 for row in catalog_rows if _normalize_token(row.get("pathology")) == "unknown"
            ),
            "laterality": sum(
                1 for row in catalog_rows if _normalize_token(row.get("laterality")) == "unknown"
            ),
            "view": sum(
                1 for row in catalog_rows if _normalize_token(row.get("view")) == "unknown"
            ),
        },
        "duplicate_counts": {
            "sample_id_duplicates": sum(count - 1 for count in raw_sample_ids.values() if count > 1),
            "processed_path_duplicates": sum(
                count - 1 for count in raw_processed_paths.values() if count > 1
            ),
        },
        "missing_processed_files_in_catalog": missing_processed_files,
        "split_row_counts": split_row_counts,
        "effective_split_summary": {
            split_name: _build_split_summary(records)
            for split_name, records in effective_records.items()
        },
        "split_overlap": overlap_summary,
        "abnormal_spellings": {
            "pathology": dict(abnormal_pathology),
            "laterality": dict(abnormal_laterality),
            "view": dict(abnormal_view),
        },
        "mixed_patients": {
            "count": sum(1 for values in mixed_patient_map.values() if len(values) > 1),
            "combo_counts": {
                " / ".join(combo): count for combo, count in mixed_combo_counts.items()
            },
        },
        "cleaning_rules": {
            "unknown_pathology": "drop",
            "unknown_laterality": "keep_unknown_do_not_impute",
            "unknown_view": "keep_unknown_do_not_impute",
            "path_resolution": "processed_path_with_suffix_fallback",
            "warmup_sampler": "dataset_label_balanced_recommended",
            "cache_mode": "preprocess",
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# External Audit Summary",
        "",
        f"- Contract version: `{summary['contract_version']}`",
        f"- Total catalog rows: `{summary['total_catalog_rows']}`",
        f"- Missing processed files in catalog: `{summary['missing_processed_files_in_catalog']}`",
        "",
        "## Unknown Counts",
        "",
        f"- pathology: `{summary['unknown_counts']['pathology']}`",
        f"- laterality: `{summary['unknown_counts']['laterality']}`",
        f"- view: `{summary['unknown_counts']['view']}`",
        "",
        "## Effective Split Summary",
        "",
    ]
    for split_name, payload in summary["effective_split_summary"].items():
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- count: `{payload['count']}`",
                f"- patient_count: `{payload['patient_count']}`",
                f"- label_counts: `{payload['label_counts']}`",
                f"- dataset_counts: `{payload['dataset_counts']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Split Overlap",
            "",
        ]
    )
    for pair_name, payload in summary["split_overlap"].items():
        lines.append(
            f"- {pair_name}: patient_overlap=`{payload['patient_overlap']}`, processed_path_overlap=`{payload['processed_path_overlap']}`"
        )
    lines.extend(
        [
            "",
            "## Mixed Patients",
            "",
            f"- count: `{summary['mixed_patients']['count']}`",
            f"- combos: `{summary['mixed_patients']['combo_counts']}`",
            "",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"audit: json={args.output_json} md={args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
