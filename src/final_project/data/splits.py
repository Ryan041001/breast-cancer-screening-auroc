from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

from final_project.data.manifest import BreastManifestRecord


def assign_deterministic_folds(
    records: Sequence[BreastManifestRecord],
    num_folds: int,
    seed: int = 0,
) -> dict[str, int]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")

    patient_records: dict[str, list[BreastManifestRecord]] = defaultdict(list)
    seen_breast_ids: set[str] = set()
    for record in records:
        if record.breast_id in seen_breast_ids:
            raise ValueError(
                f"Duplicate breast_id '{record.breast_id}' found in fold assignment input"
            )
        _ = _require_training_label(record)
        seen_breast_ids.add(record.breast_id)
        patient_records[_patient_id_from_breast_id(record.breast_id)].append(record)

    patient_groups = [
        _build_patient_group(patient_id, grouped_records, seed)
        for patient_id, grouped_records in patient_records.items()
    ]
    _validate_fold_feasibility(patient_groups, num_folds)

    fold_pos = [0] * num_folds
    fold_breasts = [0] * num_folds
    fold_patients = [0] * num_folds
    assignments: dict[str, int] = {}

    ordered_groups = sorted(
        patient_groups,
        key=lambda group: (
            -group["n_pos_breasts"],
            -group["n_breasts"],
            group["order_key"],
        ),
    )

    for group in ordered_groups:
        best_fold = min(
            range(num_folds),
            key=lambda fold: (
                _projected_fold_spread(
                    fold_values=fold_pos,
                    fold=fold,
                    added_value=int(group["n_pos_breasts"]),
                ),
                _projected_fold_spread(
                    fold_values=fold_breasts,
                    fold=fold,
                    added_value=int(group["n_breasts"]),
                ),
                _projected_fold_spread(
                    fold_values=fold_patients,
                    fold=fold,
                    added_value=1,
                ),
                fold_pos[fold],
                fold_breasts[fold],
                fold_patients[fold],
                _hash_value(f"{seed}:fold:{fold}:{group['patient_id']}"),
            ),
        )
        _assign_group_to_fold(
            group,
            best_fold,
            assignments,
            fold_pos,
            fold_breasts,
            fold_patients,
        )

    return assignments


def build_fold_audit(
    records: Sequence[BreastManifestRecord],
    assignments: dict[str, int],
    num_folds: int,
) -> dict[str, object]:
    patient_ids_by_fold: dict[int, set[str]] = {fold: set() for fold in range(num_folds)}
    audit_by_fold: dict[str, dict[str, int]] = {
        str(fold): {
            "patients": 0,
            "breasts": 0,
            "positive_breasts": 0,
            "negative_breasts": 0,
        }
        for fold in range(num_folds)
    }
    for record in records:
        fold = assignments[record.breast_id]
        fold_key = str(fold)
        patient_id = _patient_id_from_breast_id(record.breast_id)
        patient_ids_by_fold[fold].add(patient_id)
        audit_by_fold[fold_key]["breasts"] += 1
        if _require_training_label(record) == 1:
            audit_by_fold[fold_key]["positive_breasts"] += 1
        else:
            audit_by_fold[fold_key]["negative_breasts"] += 1
    for fold in range(num_folds):
        audit_by_fold[str(fold)]["patients"] = len(patient_ids_by_fold[fold])

    return {
        "folds": audit_by_fold,
        "totals": {
            "patients": len({_patient_id_from_breast_id(record.breast_id) for record in records}),
            "breasts": len(records),
            "positive_breasts": sum(
                _require_training_label(record) for record in records
            ),
            "negative_breasts": sum(
                1 - _require_training_label(record) for record in records
            ),
        },
    }


def _build_patient_group(
    patient_id: str,
    grouped_records: Sequence[BreastManifestRecord],
    seed: int,
) -> dict[str, object]:
    n_pos_breasts = sum(_require_training_label(record) for record in grouped_records)
    breast_ids = sorted(record.breast_id for record in grouped_records)
    return {
        "patient_id": patient_id,
        "breast_ids": breast_ids,
        "n_breasts": len(grouped_records),
        "n_pos_breasts": n_pos_breasts,
        "patient_label": 1 if n_pos_breasts > 0 else 0,
        "order_key": _hash_value(f"{seed}:{patient_id}"),
    }


def _validate_fold_feasibility(
    patient_groups: Sequence[dict[str, object]],
    num_folds: int,
) -> None:
    label_counts: dict[int, int] = defaultdict(int)
    for group in patient_groups:
        label_counts[int(group["patient_label"])] += 1
    if not label_counts:
        return
    smallest_bucket_size = min(label_counts.values())
    if num_folds > smallest_bucket_size:
        raise ValueError("num_folds must not exceed the smallest class count")


def _projected_fold_spread(
    *,
    fold_values: Sequence[int],
    fold: int,
    added_value: int,
) -> int:
    projected = [
        value + added_value if index == fold else value
        for index, value in enumerate(fold_values)
    ]
    return max(projected) - min(projected)


def _assign_group_to_fold(
    group: dict[str, object],
    fold: int,
    assignments: dict[str, int],
    fold_pos: list[int],
    fold_breasts: list[int],
    fold_patients: list[int],
) -> None:
    fold_pos[fold] += int(group["n_pos_breasts"])
    fold_breasts[fold] += int(group["n_breasts"])
    fold_patients[fold] += 1
    for breast_id in group["breast_ids"]:
        assignments[str(breast_id)] = fold


def _hash_value(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _patient_id_from_breast_id(breast_id: str) -> str:
    patient_id, _, _ = breast_id.rpartition("_")
    return patient_id or breast_id


def _require_training_label(record: BreastManifestRecord) -> int:
    if record.label is None:
        raise ValueError(
            f"Breast '{record.breast_id}' is missing a training label for fold assignment"
        )
    return record.label
