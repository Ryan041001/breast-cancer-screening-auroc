from __future__ import annotations

from pathlib import Path
from typing import Literal
from types import SimpleNamespace

import pytest

from final_project.config import (
    AppConfig,
    ExperimentConfig,
    PathsConfig,
    RuntimeConfig,
    TrainConfig,
)
from final_project.data.manifest import BreastManifestRecord
from final_project.engine.predict import build_prediction_loader
from final_project.engine.run_cv import run_cross_validation
from final_project.engine.run_cv import FoldRunResult, summarize_cv_results
from final_project.engine.trainer import EvaluationResult


def test_run_cv_aggregates_fold_metrics_and_oof_predictions() -> None:
    summary = summarize_cv_results(
        [
            FoldRunResult(
                fold=0,
                auc=0.8,
                predictions={"020_R": 0.75, "010_L": 0.25},
            ),
            FoldRunResult(
                fold=1,
                auc=0.6,
                predictions={"030_R": 0.55},
            ),
        ]
    )

    assert summary.mean_auc == 0.7
    assert summary.fold_metrics == {0: 0.8, 1: 0.6}
    assert summary.oof_predictions == {
        "020_R": 0.75,
        "010_L": 0.25,
        "030_R": 0.55,
    }


@pytest.mark.parametrize(
    ("transform_profile", "expected_profile"),
    [(None, "baseline"), ("normaug", "normaug")],
)
def test_build_prediction_loader_forwards_transform_profile(
    tmp_path: Path,
    monkeypatch,
    transform_profile: Literal["normaug"] | None,
    expected_profile: str,
) -> None:
    observed: dict[str, object] = {}

    class CapturingDataset:
        def __init__(
            self,
            *,
            records,
            image_size,
            training,
            transform_profile,
            cache_mode,
        ) -> None:
            observed.update(
                {
                    "records": list(records),
                    "image_size": image_size,
                    "training": training,
                    "transform_profile": transform_profile,
                    "cache_mode": cache_mode,
                }
            )

        def __len__(self) -> int:
            return 0

        def __getitem__(self, index: int) -> object:
            raise IndexError(index)

    monkeypatch.setattr(
        "final_project.engine.predict.PairedBreastDataset", CapturingDataset
    )

    records = [
        BreastManifestRecord(
            breast_id="T_001",
            cc_path=tmp_path / "T_001_CC.jpg",
            mlo_path=tmp_path / "T_001_MLO.jpg",
            label=None,
        )
    ]
    if transform_profile is None:
        build_prediction_loader(
            records,
            image_size=32,
            batch_size=2,
            num_workers=0,
        )
    else:
        build_prediction_loader(
            records,
            image_size=32,
            batch_size=2,
            num_workers=0,
            transform_profile=transform_profile,
        )

    assert observed["records"] == records
    assert observed["image_size"] == 32
    assert observed["training"] is False
    assert observed["transform_profile"] == expected_profile
    assert observed["cache_mode"] == "preprocess"


def test_run_cv_reports_startup_and_fold_progress(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    loaded_warmup: dict[str, object] = {}
    train_records = [
        BreastManifestRecord(
            breast_id="100_L",
            cc_path=tmp_path / "100_L_CC.jpg",
            mlo_path=tmp_path / "100_L_MLO.jpg",
            label=0,
        ),
        BreastManifestRecord(
            breast_id="101_R",
            cc_path=tmp_path / "101_R_CC.jpg",
            mlo_path=tmp_path / "101_R_MLO.jpg",
            label=1,
        ),
    ]
    test_records = [
        BreastManifestRecord(
            breast_id="T_001",
            cc_path=tmp_path / "T_001_CC.jpg",
            mlo_path=tmp_path / "T_001_MLO.jpg",
            label=None,
        )
    ]
    config = AppConfig(
        experiment=ExperimentConfig(name="progress-test"),
        paths=PathsConfig(
            project_root=tmp_path,
            train_csv=tmp_path / "train.csv",
            train_images=tmp_path / "train_img",
            test_images=tmp_path / "test_img",
            submission_template=tmp_path / "submission.csv",
            output_root=tmp_path / "outputs",
        ),
        runtime=RuntimeConfig(seed=42, device="cpu", fold_seed=11, train_seed=13),
        train=TrainConfig(
            folds=2,
            batch_size=2,
            image_size=16,
            epochs=1,
            num_workers=0,
            transform_profile="normaug",
            external_warmup_epochs=1,
        ),
    )

    seen_seeds: list[int] = []
    monkeypatch.setattr(
        "final_project.engine.run_cv.set_global_seed", lambda seed: seen_seeds.append(seed)
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_train_manifest",
        lambda path: train_records,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_test_manifest",
        lambda submission_csv, test_images_dir: test_records,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.assign_deterministic_folds",
        lambda manifest, num_folds, seed, group_by: {"100_L": 0, "101_R": 1}
        if seed == 11 and group_by == "patient"
        else (_ for _ in ()).throw(AssertionError("unexpected fold assignment args")),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.PairedBreastModel",
        lambda backbone_name, pretrained, fusion_head_config=None: SimpleNamespace(
            backbone_name=backbone_name,
            pretrained=pretrained,
        ),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.maybe_prepare_external_warmup",
        lambda *args, **kwargs: Path("warmup/checkpoints/best.pt"),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.load_backbone_from_warmup",
        lambda model, checkpoint_path: loaded_warmup.update(
            {"model": model, "checkpoint_path": checkpoint_path}
        ),
    )

    def fake_fit_model(
        model,
        train_records,
        val_records,
        *,
        image_size,
        batch_size,
        num_workers,
        epochs,
        device,
        output_dir,
        learning_rate=1e-3,
        weight_decay=1e-2,
        scheduler_name="none",
        min_lr=0.0,
        freeze_backbone_epochs=0,
        grad_accum_steps=1,
        cache_mode="preprocess",
        transform_profile,
    ):
        assert transform_profile == "normaug"
        assert cache_mode == "preprocess"
        fold = int(output_dir.name.rsplit("_", maxsplit=1)[-1])
        return SimpleNamespace(model=f"model-{fold}"), EvaluationResult(
            loss=0.1,
            auc=0.7 + 0.1 * fold,
            predictions={val_records[0].breast_id: 0.25 + 0.5 * fold},
        )

    monkeypatch.setattr("final_project.engine.run_cv.fit_model", fake_fit_model)
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_prediction_loader",
        lambda records, image_size, batch_size, num_workers, *, transform_profile, cache_mode: [
            (records, transform_profile, cache_mode)
        ],
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.predict_probabilities",
        lambda model, batches, device: {
            "T_001": 0.4
            if model == "model-0" and batches[0][1] == "normaug" and batches[0][2] == "preprocess"
            else 0.6
        },
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.write_prediction_table",
        lambda predictions, output_path: None,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv._write_fold_metrics",
        lambda summary, output_dir: None,
    )

    artifacts = run_cross_validation(config)

    command_output = capsys.readouterr().out
    assert "run-cv: loading train manifest" in command_output
    assert "run-cv: loading test manifest" in command_output
    assert "run-cv: assigning folds" in command_output
    assert "run-cv: fold 1/2 start" in command_output
    assert "run-cv: fold 1/2 done auc=0.700000" in command_output
    assert "run-cv: fold 2/2 done auc=0.800000" in command_output
    assert artifacts.summary.mean_auc == 0.75
    assert seen_seeds == [13, 13]
    assert str(loaded_warmup["checkpoint_path"]).endswith("best.pt")
    log_text = (tmp_path / "outputs" / "runs" / "progress-test" / "cv" / "run.log").read_text(
        encoding="utf-8"
    )
    assert "run-cv: manifests ready" in log_text
    assert "run-cv: complete mean_auc=0.750000" in log_text
    assert (tmp_path / "outputs" / "runs" / "progress-test" / "cv" / "fold_audit.json").exists()


def test_run_cv_uses_configured_fusion_eval_reference_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    train_records = [
        BreastManifestRecord(
            breast_id="100_L",
            cc_path=tmp_path / "100_L_CC.jpg",
            mlo_path=tmp_path / "100_L_MLO.jpg",
            label=0,
        ),
        BreastManifestRecord(
            breast_id="101_R",
            cc_path=tmp_path / "101_R_CC.jpg",
            mlo_path=tmp_path / "101_R_MLO.jpg",
            label=1,
        ),
    ]
    test_records = [
        BreastManifestRecord(
            breast_id="T_001",
            cc_path=tmp_path / "T_001_CC.jpg",
            mlo_path=tmp_path / "T_001_MLO.jpg",
            label=None,
        )
    ]
    captured: dict[str, object] = {}
    config = AppConfig(
        experiment=ExperimentConfig(name="fusion-ref-test"),
        paths=PathsConfig(
            project_root=tmp_path,
            train_csv=tmp_path / "train.csv",
            train_images=tmp_path / "train_img",
            test_images=tmp_path / "test_img",
            submission_template=tmp_path / "submission.csv",
            output_root=tmp_path / "outputs",
        ),
        runtime=RuntimeConfig(seed=42, device="cpu"),
        train=TrainConfig(
            folds=2,
            batch_size=2,
            image_size=16,
            epochs=1,
            num_workers=0,
            fusion_eval_reference_run="blend_best12_plus_baselinev2normaug_refined",
        ),
    )

    monkeypatch.setattr(
        "final_project.engine.run_cv.set_global_seed", lambda seed: None
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_train_manifest",
        lambda path: train_records,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_test_manifest",
        lambda submission_csv, test_images_dir: test_records,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.assign_deterministic_folds",
        lambda manifest, num_folds, seed, group_by: {"100_L": 0, "101_R": 1}
        if group_by == "patient"
        else (_ for _ in ()).throw(AssertionError("unexpected split grouping")),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.PairedBreastModel",
        lambda backbone_name, pretrained, fusion_head_config=None: SimpleNamespace(
            backbone_name=backbone_name,
            pretrained=pretrained,
        ),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.maybe_prepare_external_warmup",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.fit_model",
        lambda *args, **kwargs: (
            SimpleNamespace(model="model"),
            EvaluationResult(loss=0.1, auc=0.75, predictions={"100_L": 0.2}),
        ),
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_prediction_loader",
        lambda *args, **kwargs: ["loader"],
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.predict_probabilities",
        lambda *args, **kwargs: {"T_001": 0.5},
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.write_prediction_table",
        lambda predictions, output_path: None,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv._write_fold_metrics",
        lambda summary, output_dir: None,
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv._try_write_fusion_eval",
        lambda run_dir, output_root, summary, labels, *, reference_run: captured.update(
            {"reference_run": reference_run}
        ),
    )

    _ = run_cross_validation(config)

    assert captured["reference_run"] == "blend_best12_plus_baselinev2normaug_refined"
