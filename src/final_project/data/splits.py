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
        label = _require_training_label(record)
        if label is None:
            raise ValueError(
                f"Breast '{record.breast_id}' is missing a training label for fold assignment"
            )

        seen_breast_ids.add(record.breast_id)
        patient_records[_patient_id_from_breast_id(record.breast_id)].append(record)

    label_buckets: dict[int, list[str]] = defaultdict(list)
    for patient_id, grouped_records in patient_records.items():
        patient_label = max(
            _require_training_label(record) for record in grouped_records
        )
        label_buckets[patient_label].append(patient_id)

    if not label_buckets:
        return {}

    smallest_bucket_size = min(
        len(patient_ids) for patient_ids in label_buckets.values()
    )
    if num_folds > smallest_bucket_size:
        raise ValueError("num_folds must not exceed the smallest class count")

    assignments: dict[str, int] = {}
    next_fold = 0
    for label in sorted(label_buckets):
        patient_ids = sorted(
            label_buckets[label],
            key=lambda patient_id: _hash_breast_id(patient_id, seed),
        )
        for patient_id in patient_ids:
            fold = next_fold % num_folds
            for record in patient_records[patient_id]:
                assignments[record.breast_id] = fold
            next_fold += 1

    return assignments


def _hash_breast_id(breast_id: str, seed: int) -> str:
    payload = f"{seed}:{breast_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _patient_id_from_breast_id(breast_id: str) -> str:
    patient_id, _, _ = breast_id.rpartition("_")
    return patient_id or breast_id


def _require_training_label(record: BreastManifestRecord) -> int:
    if record.label is None:
        raise ValueError(
            f"Breast '{record.breast_id}' is missing a training label for fold assignment"
        )
    return record.label
