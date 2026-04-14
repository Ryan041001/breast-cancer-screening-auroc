from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from ..config import AppConfig
from ..data.manifest import (
    BreastManifestRecord,
    build_test_manifest,
    build_train_manifest,
)
from ..data.splits import assign_deterministic_folds
from ..model.fusion import PairedBreastModel
from ..utils.paths import build_output_paths
from ..utils.repro import set_global_seed
from .predict import (
    DEFAULT_BACKBONE_NAME,
    build_prediction_loader,
    predict_probabilities,
)
from .submission import write_prediction_table
from .trainer import EvaluationResult, fit_model


@dataclass(frozen=True, slots=True)
class FoldRunResult:
    fold: int
    auc: float
    predictions: dict[str, float]


@dataclass(frozen=True, slots=True)
class CVSummary:
    mean_auc: float
    fold_metrics: dict[int, float]
    oof_predictions: dict[str, float]


@dataclass(frozen=True, slots=True)
class CVRunArtifacts:
    summary: CVSummary
    output_dir: Path
    test_predictions: dict[str, float]


def summarize_cv_results(fold_results: list[FoldRunResult]) -> CVSummary:
    fold_metrics = {result.fold: result.auc for result in fold_results}
    oof_predictions: dict[str, float] = {}
    for result in fold_results:
        oof_predictions.update(result.predictions)
    return CVSummary(
        mean_auc=mean(fold_metrics.values()) if fold_metrics else 0.0,
        fold_metrics=fold_metrics,
        oof_predictions=oof_predictions,
    )


def run_cross_validation(config: AppConfig) -> CVRunArtifacts:
    set_global_seed(config.runtime.seed)
    print("run-cv: loading train manifest", flush=True)
    train_manifest = build_train_manifest(config.paths.train_csv)
    print("run-cv: loading test manifest", flush=True)
    test_manifest = build_test_manifest(
        config.paths.submission_template,
        config.paths.test_images,
    )
    print("run-cv: assigning folds", flush=True)
    assignments = assign_deterministic_folds(
        train_manifest,
        num_folds=config.train.folds,
        seed=config.runtime.seed,
    )

    output_paths = build_output_paths(config.paths.output_root)
    run_dir = output_paths.runs / config.experiment.name / "cv"
    run_dir.mkdir(parents=True, exist_ok=True)

    fold_results: list[FoldRunResult] = []
    test_prediction_sets: list[dict[str, float]] = []
    num_folds = config.train.folds
    for fold in range(num_folds):
        train_records = [
            record for record in train_manifest if assignments[record.breast_id] != fold
        ]
        val_records = [
            record for record in train_manifest if assignments[record.breast_id] == fold
        ]
        print(
            f"run-cv: fold {fold + 1}/{num_folds} start"
            f" train={len(train_records)} val={len(val_records)}",
            flush=True,
        )
        fold_dir = run_dir / f"fold_{fold}"
        model = PairedBreastModel(backbone_name=DEFAULT_BACKBONE_NAME, pretrained=True)
        trainer, eval_result = fit_model(
            model,
            train_records,
            val_records,
            image_size=config.train.image_size,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            epochs=config.train.epochs,
            device=config.runtime.device,
            output_dir=fold_dir,
        )
        test_predictions = predict_probabilities(
            trainer.model,
            build_prediction_loader(
                test_manifest,
                image_size=config.train.image_size,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
            ),
            device=config.runtime.device,
        )
        print(
            f"run-cv: fold {fold + 1}/{num_folds} done"
            f" auc={eval_result.auc:.6f}",
            flush=True,
        )
        fold_results.append(
            FoldRunResult(
                fold=fold,
                auc=eval_result.auc,
                predictions=eval_result.predictions,
            )
        )
        test_prediction_sets.append(test_predictions)
        write_prediction_table(
            eval_result.predictions, fold_dir / "oof_predictions.csv"
        )
        write_prediction_table(test_predictions, fold_dir / "test_predictions.csv")

    summary = summarize_cv_results(fold_results)
    averaged_test_predictions = _average_prediction_sets(test_prediction_sets)
    write_prediction_table(summary.oof_predictions, run_dir / "oof_predictions.csv")
    write_prediction_table(averaged_test_predictions, run_dir / "test_predictions.csv")
    _write_fold_metrics(summary, run_dir)
    return CVRunArtifacts(
        summary=summary,
        output_dir=run_dir,
        test_predictions=averaged_test_predictions,
    )


def _average_prediction_sets(
    prediction_sets: list[dict[str, float]],
) -> dict[str, float]:
    if not prediction_sets:
        return {}
    breast_ids = prediction_sets[0].keys()
    return {
        breast_id: sum(predictions[breast_id] for predictions in prediction_sets)
        / len(prediction_sets)
        for breast_id in breast_ids
    }


def _write_fold_metrics(summary: CVSummary, output_dir: Path) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "mean_auc": summary.mean_auc,
                "fold_metrics": summary.fold_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    csv_path = output_dir / "fold_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "auc"])
        writer.writeheader()
        for fold, auc in sorted(summary.fold_metrics.items()):
            writer.writerow({"fold": fold, "auc": auc})
