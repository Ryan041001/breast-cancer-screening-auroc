from __future__ import annotations

from pathlib import Path

from PIL import Image

from final_project.data.external import load_external_split_records


def test_load_external_split_records_filters_unknown_and_resolves_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "MammoNet32k_new"
    processed_root = root / "processed"
    splits_root = root / "splits"
    processed_root.mkdir(parents=True)
    splits_root.mkdir(parents=True)

    for relative_path, pixel in [
        ("demo/patient1_cc.png", 32),
        ("demo/patient2_mlo.png", 224),
        ("demo/patient3_unknown.png", 128),
    ]:
        path = processed_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (8, 8), color=pixel).save(path)

    (root / "catalog.csv").write_text(
        "\n".join(
            [
                "dataset,patient_id,original_path,processed_path,laterality,view,pathology",
                "demo,patient1,orig1,demo/patient1_cc.png,L,CC,malignant",
                "demo,patient2,orig2,demo/patient2_mlo.png,R,MLO,normal",
                "demo,patient3,orig3,demo/patient3_unknown.png,U,U,unknown",
            ]
        ),
        encoding="utf-8",
    )
    (splits_root / "train.csv").write_text(
        "\n".join(
            [
                "dataset,patient_id,original_path,processed_path,laterality,view,pathology",
                "demo,patient1,orig1,demo/patient1_cc.png,L,CC,malignant",
                "demo,patient2,orig2,demo/patient2_mlo.png,R,MLO,normal",
                "demo,patient3,orig3,demo/patient3_unknown.png,U,U,unknown",
            ]
        ),
        encoding="utf-8",
    )

    records = load_external_split_records(
        catalog_csv=root / "catalog.csv",
        processed_root=processed_root,
        split_csv=splits_root / "train.csv",
    )

    assert len(records) == 2
    assert records[0].label == 1
    assert records[0].image_path == (processed_root / "demo/patient1_cc.png")
    assert records[1].label == 0
    assert records[1].laterality == "R"
