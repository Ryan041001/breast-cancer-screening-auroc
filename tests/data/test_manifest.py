from __future__ import annotations

import csv
from pathlib import Path

import pytest

from final_project.data.manifest import (
    BreastManifestRecord,
    build_test_manifest,
    build_train_manifest,
)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(b"")


def test_build_train_manifest_collapses_pairs_and_derives_malignant_label(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        fieldnames=["image_path", "breast_id", "cc_mlo", "pathology"],
        rows=[
            {
                "image_path": "train_img/100_L/100_L_CC.jpg",
                "breast_id": "100_L",
                "cc_mlo": "CC",
                "pathology": "N",
            },
            {
                "image_path": "train_img/100_L/100_L_MLO.jpg",
                "breast_id": "100_L",
                "cc_mlo": "MLO",
                "pathology": "N",
            },
            {
                "image_path": "train_img/200_R/200_R_CC.jpg",
                "breast_id": "200_R",
                "cc_mlo": "CC",
                "pathology": "B",
            },
            {
                "image_path": "train_img/200_R/200_R_MLO.jpg",
                "breast_id": "200_R",
                "cc_mlo": "MLO",
                "pathology": "M",
            },
        ],
    )

    manifest = build_train_manifest(train_csv)

    assert manifest == [
        BreastManifestRecord(
            breast_id="100_L",
            cc_path=tmp_path / "train_img/100_L/100_L_CC.jpg",
            mlo_path=tmp_path / "train_img/100_L/100_L_MLO.jpg",
            label=0,
        ),
        BreastManifestRecord(
            breast_id="200_R",
            cc_path=tmp_path / "train_img/200_R/200_R_CC.jpg",
            mlo_path=tmp_path / "train_img/200_R/200_R_MLO.jpg",
            label=1,
        ),
    ]


def test_build_train_manifest_requires_exactly_one_cc_and_one_mlo(
    tmp_path: Path,
) -> None:
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        fieldnames=["image_path", "breast_id", "cc_mlo", "pathology"],
        rows=[
            {
                "image_path": "train_img/300_L/300_L_CC.jpg",
                "breast_id": "300_L",
                "cc_mlo": "CC",
                "pathology": "N",
            },
            {
                "image_path": "train_img/300_L/300_L_CC_repeat.jpg",
                "breast_id": "300_L",
                "cc_mlo": "CC",
                "pathology": "N",
            },
        ],
    )

    with pytest.raises(ValueError, match="300_L"):
        _ = build_train_manifest(train_csv)


def test_build_test_manifest_preserves_submission_order(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    _write_csv(
        submission_csv,
        fieldnames=["breast_id", "pred_score"],
        rows=[
            {"breast_id": "020_R", "pred_score": ""},
            {"breast_id": "010_L", "pred_score": ""},
        ],
    )

    test_images = tmp_path / "test_img"
    _touch(test_images / "010_L/010_L_CC.jpg")
    _touch(test_images / "010_L/010_L_MLO.jpg")
    _touch(test_images / "020_R/020_R_CC.jpg")
    _touch(test_images / "020_R/020_R_MLO.jpg")

    manifest = build_test_manifest(submission_csv, test_images)

    assert manifest == [
        BreastManifestRecord(
            breast_id="020_R",
            cc_path=test_images / "020_R/020_R_CC.jpg",
            mlo_path=test_images / "020_R/020_R_MLO.jpg",
            label=None,
        ),
        BreastManifestRecord(
            breast_id="010_L",
            cc_path=test_images / "010_L/010_L_CC.jpg",
            mlo_path=test_images / "010_L/010_L_MLO.jpg",
            label=None,
        ),
    ]


def test_build_test_manifest_rejects_extra_view_files(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    _write_csv(
        submission_csv,
        fieldnames=["breast_id", "pred_score"],
        rows=[{"breast_id": "500_L", "pred_score": ""}],
    )

    test_images = tmp_path / "test_img"
    _touch(test_images / "500_L/500_L_CC.jpg")
    _touch(test_images / "500_L/500_L_MLO.jpg")
    _touch(test_images / "500_L/500_L_CC_extra.jpg")

    with pytest.raises(ValueError, match="500_L"):
        _ = build_test_manifest(submission_csv, test_images)


def test_build_train_manifest_rejects_unknown_pathology(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        fieldnames=["image_path", "breast_id", "cc_mlo", "pathology"],
        rows=[
            {
                "image_path": "train_img/100_L/100_L_CC.jpg",
                "breast_id": "100_L",
                "cc_mlo": "CC",
                "pathology": "X",
            },
            {
                "image_path": "train_img/100_L/100_L_MLO.jpg",
                "breast_id": "100_L",
                "cc_mlo": "MLO",
                "pathology": "N",
            },
        ],
    )

    with pytest.raises(ValueError, match="pathology"):
        _ = build_train_manifest(train_csv)


def test_build_train_manifest_rejects_paths_outside_csv_root(tmp_path: Path) -> None:
    train_csv = tmp_path / "train.csv"
    _write_csv(
        train_csv,
        fieldnames=["image_path", "breast_id", "cc_mlo", "pathology"],
        rows=[
            {
                "image_path": "../escape_CC.jpg",
                "breast_id": "100_L",
                "cc_mlo": "CC",
                "pathology": "N",
            },
            {
                "image_path": "train_img/100_L/100_L_MLO.jpg",
                "breast_id": "100_L",
                "cc_mlo": "MLO",
                "pathology": "N",
            },
        ],
    )

    with pytest.raises(ValueError, match="image_path"):
        _ = build_train_manifest(train_csv)


def test_build_test_manifest_rejects_unsafe_breast_id(tmp_path: Path) -> None:
    submission_csv = tmp_path / "submission.csv"
    _write_csv(
        submission_csv,
        fieldnames=["breast_id", "pred_score"],
        rows=[{"breast_id": "../escape", "pred_score": ""}],
    )

    with pytest.raises(ValueError, match="breast_id"):
        _ = build_test_manifest(submission_csv, tmp_path / "test_img")
