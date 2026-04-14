from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from final_project.config import (
    AppConfig,
    ExperimentConfig,
    PathsConfig,
    RuntimeConfig,
    TrainConfig,
)
from final_project.data.manifest import BreastManifestRecord
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


def test_run_cv_reports_startup_and_fold_progress(
    tmp_path: Path,
    monkeypatch,
    capsys,
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
        runtime=RuntimeConfig(seed=42, device="cpu"),
        train=TrainConfig(
            folds=2,
            batch_size=2,
            image_size=16,
            epochs=1,
            num_workers=0,
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
        lambda manifest, num_folds, seed: {"100_L": 0, "101_R": 1},
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.PairedBreastModel",
        lambda backbone_name, pretrained: SimpleNamespace(
            backbone_name=backbone_name,
            pretrained=pretrained,
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
    ):
        fold = int(output_dir.name.rsplit("_", maxsplit=1)[-1])
        return SimpleNamespace(model=f"model-{fold}"), EvaluationResult(
            loss=0.1,
            auc=0.7 + 0.1 * fold,
            predictions={val_records[0].breast_id: 0.25 + 0.5 * fold},
        )

    monkeypatch.setattr("final_project.engine.run_cv.fit_model", fake_fit_model)
    monkeypatch.setattr(
        "final_project.engine.run_cv.build_prediction_loader",
        lambda records, image_size, batch_size, num_workers: [records],
    )
    monkeypatch.setattr(
        "final_project.engine.run_cv.predict_probabilities",
        lambda model, batches, device: {"T_001": 0.4 if model == "model-0" else 0.6},
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
