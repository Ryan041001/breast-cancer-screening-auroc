from __future__ import annotations

from pathlib import Path

import pytest

from final_project.data.manifest import BreastManifestRecord
from final_project.data.splits import assign_deterministic_folds, build_fold_audit


def _record(breast_id: str, label: int) -> BreastManifestRecord:
    return BreastManifestRecord(
        breast_id=breast_id,
        cc_path=Path(f"train_img/{breast_id}/{breast_id}_CC.jpg"),
        mlo_path=Path(f"train_img/{breast_id}/{breast_id}_MLO.jpg"),
        label=label,
    )


def test_assign_deterministic_folds_is_reproducible_and_order_independent() -> None:
    records: list[BreastManifestRecord] = []
    records.append(_record("100_L", 0))
    records.append(_record("101_L", 0))
    records.append(_record("200_R", 1))
    records.append(_record("201_R", 1))
    records.append(_record("300_L", 0))
    records.append(_record("400_R", 1))

    first = assign_deterministic_folds(records, num_folds=3, seed=17)
    second = assign_deterministic_folds(list(reversed(records)), num_folds=3, seed=17)

    assert first == second
    assert set(first) == {record.breast_id for record in records}
    assert set(first.values()) == {0, 1, 2}


def test_assign_deterministic_folds_rejects_duplicate_breast_ids() -> None:
    with pytest.raises(ValueError, match="100_L"):
        _ = assign_deterministic_folds(
            [_record("100_L", 0), _record("100_L", 1)],
            num_folds=2,
        )


def test_assign_deterministic_folds_rejects_infeasible_fold_counts() -> None:
    records = [_record("100_L", 0), _record("101_L", 0), _record("200_R", 1)]

    with pytest.raises(ValueError, match="num_folds"):
        _ = assign_deterministic_folds(records, num_folds=2, seed=17)


def test_assign_deterministic_folds_keeps_bilateral_patient_in_same_fold() -> None:
    records = [
        _record("A_L", 0),
        _record("A_R", 1),
        _record("B_L", 0),
        _record("C_L", 0),
        _record("F_L", 0),
        _record("D_R", 1),
        _record("E_R", 1),
    ]

    folds = assign_deterministic_folds(records, num_folds=3, seed=42)

    assert folds["A_L"] == folds["A_R"]


def test_build_fold_audit_counts_patients_breasts_and_labels() -> None:
    records = [
        _record("A_L", 0),
        _record("A_R", 1),
        _record("B_L", 0),
    ]
    assignments = {"A_L": 0, "A_R": 0, "B_L": 1}

    audit = build_fold_audit(records, assignments, num_folds=2)

    assert audit["folds"]["0"]["patients"] == 1
    assert audit["folds"]["0"]["breasts"] == 2
    assert audit["folds"]["0"]["positive_breasts"] == 1
    assert audit["folds"]["1"]["patients"] == 1
    assert audit["totals"]["breasts"] == 3
