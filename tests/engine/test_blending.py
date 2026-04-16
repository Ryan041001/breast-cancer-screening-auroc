from __future__ import annotations

import json
from pathlib import Path

import pytest

from final_project.data.manifest import BreastManifestRecord
from final_project.engine.blending import run_blend_from_spec
from final_project.engine.submission import write_prediction_table


def _write_spec(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_run_predictions(
    runs_root: Path,
    run_name: str,
    *,
    oof_predictions: dict[str, float],
    test_predictions: dict[str, float],
) -> None:
    run_dir = runs_root / run_name / "cv"
    write_prediction_table(oof_predictions, run_dir / "oof_predictions.csv")
    write_prediction_table(test_predictions, run_dir / "test_predictions.csv")


def test_run_blend_from_spec_writes_expected_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "outputs" / "runs"
    spec_path = tmp_path / "blend_spec.json"
    monkeypatch.setattr(
        "final_project.engine.blending.build_train_manifest",
        lambda _: [
            BreastManifestRecord(
                breast_id="A",
                cc_path=tmp_path / "A_CC.jpg",
                mlo_path=tmp_path / "A_MLO.jpg",
                label=0,
            ),
            BreastManifestRecord(
                breast_id="B",
                cc_path=tmp_path / "B_CC.jpg",
                mlo_path=tmp_path / "B_MLO.jpg",
                label=1,
            ),
        ],
    )

    _write_run_predictions(
        runs_root,
        "run_a",
        oof_predictions={"A": 0.2, "B": 0.7},
        test_predictions={"T1": 0.3, "T2": 0.8},
    )
    _write_run_predictions(
        runs_root,
        "run_b",
        oof_predictions={"A": 0.4, "B": 0.9},
        test_predictions={"T1": 0.5, "T2": 0.6},
    )
    _write_spec(
        spec_path,
        {
            "output_name": "blend_demo",
            "run_map": {"a": "run_a", "b": "run_b"},
            "weights": {"a": 2.0, "b": 1.0},
            "removed_members": ["legacy_bad_shard"],
            "gain_over_prev": 0.00123,
        },
    )

    artifacts = run_blend_from_spec(spec_path, runs_root=runs_root)

    assert artifacts.output_dir == runs_root / "blend_demo"
    assert artifacts.members == ("a", "b")
    assert artifacts.weights == {"a": pytest.approx(2 / 3), "b": pytest.approx(1 / 3)}
    assert artifacts.oof_auc == pytest.approx(1.0)

    metrics = json.loads((artifacts.output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["members"] == ["a", "b"]
    assert metrics["weights"]["a"] == pytest.approx(2 / 3)
    assert metrics["weights"]["b"] == pytest.approx(1 / 3)

    blend_payload = json.loads((artifacts.output_dir / "blend.json").read_text(encoding="utf-8"))
    assert blend_payload["removed_members"] == ["legacy_bad_shard"]
    assert blend_payload["gain_over_prev"] == pytest.approx(0.00123)
    assert blend_payload["weights"]["a"] == pytest.approx(2 / 3)

    oof_csv = (artifacts.output_dir / "oof_predictions.csv").read_text(encoding="utf-8")
    test_csv = (artifacts.output_dir / "test_predictions.csv").read_text(encoding="utf-8")
    assert "A,0.26666666666666666" in oof_csv
    assert "B,0.7666666666666666" in oof_csv
    assert "T1,0.36666666666666664" in test_csv
    assert "T2,0.7333333333333333" in test_csv


def test_run_blend_from_spec_rejects_weight_alias_mismatch(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad_spec.json"
    _write_spec(
        spec_path,
        {
            "run_map": {"a": "run_a", "b": "run_b"},
            "weights": {"a": 0.5},
        },
    )

    with pytest.raises(ValueError, match="weights must match run_map aliases exactly"):
        _ = run_blend_from_spec(spec_path, runs_root=tmp_path / "runs")


def test_run_blend_from_spec_rejects_member_keyset_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs_root = tmp_path / "outputs" / "runs"
    spec_path = tmp_path / "blend_spec.json"
    monkeypatch.setattr(
        "final_project.engine.blending.build_train_manifest",
        lambda _: [
            BreastManifestRecord(
                breast_id="A",
                cc_path=tmp_path / "A_CC.jpg",
                mlo_path=tmp_path / "A_MLO.jpg",
                label=0,
            )
        ],
    )

    _write_run_predictions(
        runs_root,
        "run_a",
        oof_predictions={"A": 0.2},
        test_predictions={"T1": 0.3},
    )
    _write_run_predictions(
        runs_root,
        "run_b",
        oof_predictions={"B": 0.9},
        test_predictions={"T1": 0.5},
    )
    _write_spec(
        spec_path,
        {
            "run_map": {"a": "run_a", "b": "run_b"},
            "weights": {"a": 0.5, "b": 0.5},
        },
    )

    with pytest.raises(ValueError, match="does not match the key-set"):
        _ = run_blend_from_spec(spec_path, runs_root=runs_root)
