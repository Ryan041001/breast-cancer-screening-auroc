from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import cast

from ..config import AppConfig
from ..data.manifest import (
    BreastManifestRecord,
    build_test_manifest,
    build_train_manifest,
)
from ..data.splits import assign_deterministic_folds
from ..data.splits import build_fold_audit
from ..data.transforms import TransformProfile
from ..model.fusion import FusionHeadConfig, PairedBreastModel
from ..utils.paths import build_output_paths
from ..utils.repro import set_global_seed
from ..utils.logging import log_message
from .predict import (
    DEFAULT_BACKBONE_NAME,
    build_prediction_loader,
    predict_probabilities,
)
from .submission import read_prediction_table_strict, write_prediction_table
from .trainer import EvaluationResult, fit_model
from .fusion_eval import evaluate_fusion_candidate
from .external_warmup import load_backbone_from_warmup, maybe_prepare_external_warmup


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
    train_seed = config.runtime.train_seed or config.runtime.seed
    fold_seed = config.runtime.fold_seed or config.runtime.seed
    set_global_seed(train_seed)
    output_paths = build_output_paths(config.paths.output_root)
    run_dir = output_paths.runs / config.experiment.name / "cv"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_message(run_dir, "run-cv: loading train manifest")
    train_manifest = build_train_manifest(config.paths.train_csv)
    log_message(run_dir, "run-cv: loading test manifest")
    test_manifest = build_test_manifest(
        config.paths.submission_template,
        config.paths.test_images,
    )
    log_message(
        run_dir,
        f"run-cv: manifests ready train_breasts={len(train_manifest)} test_breasts={len(test_manifest)}",
    )
    log_message(run_dir, "run-cv: assigning folds")
    assignments = assign_deterministic_folds(
        train_manifest,
        num_folds=config.train.folds,
        seed=fold_seed,
    )
    _write_fold_audit(
        train_manifest=train_manifest,
        assignments=assignments,
        num_folds=config.train.folds,
        output_dir=run_dir,
        fold_seed=fold_seed,
    )
    transform_profile = cast(TransformProfile, config.train.transform_profile)
    _log_ignored_linear_fusion_settings(run_dir, config)
    warmup_checkpoint = maybe_prepare_external_warmup(
        config,
        backbone_name=DEFAULT_BACKBONE_NAME,
        output_dir=output_paths.runs / config.experiment.name / "external_warmup",
        image_size=config.train.image_size,
        transform_profile=transform_profile,
    )
    set_global_seed(train_seed)

    fusion_head_config = FusionHeadConfig.from_train_config(config.train)

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
        log_message(
            run_dir,
            f"run-cv: fold {fold + 1}/{num_folds} start"
            f" train={len(train_records)} val={len(val_records)}",
        )
        fold_dir = run_dir / f"fold_{fold}"
        model = PairedBreastModel(
            backbone_name=DEFAULT_BACKBONE_NAME,
            pretrained=True,
            fusion_head_config=fusion_head_config,
        )
        if warmup_checkpoint is not None:
            load_backbone_from_warmup(model, warmup_checkpoint)
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
            transform_profile=transform_profile,
            learning_rate=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            scheduler_name=config.train.scheduler,
            min_lr=config.train.min_lr,
            freeze_backbone_epochs=config.train.freeze_backbone_epochs,
            grad_accum_steps=config.train.grad_accum_steps,
            cache_mode=config.train.cache_mode,
        )
        test_predictions = predict_probabilities(
            trainer.model,
            build_prediction_loader(
                test_manifest,
                image_size=config.train.image_size,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
                transform_profile=transform_profile,
                cache_mode=config.train.cache_mode,
            ),
            device=config.runtime.device,
        )
        log_message(
            run_dir,
            f"run-cv: fold {fold + 1}/{num_folds} done auc={eval_result.auc:.6f}",
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

    # Attempt fusion eval against baseline if available
    labels_by_id = {
        r.breast_id: r.label
        for r in train_manifest
        if r.label is not None
    }
    _try_write_fusion_eval(
        run_dir,
        config.paths.output_root,
        summary,
        labels_by_id,
        reference_run=config.train.fusion_eval_reference_run,
    )
    log_message(
        run_dir,
        f"run-cv: complete mean_auc={summary.mean_auc:.6f} folds={len(summary.fold_metrics)}",
    )

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


def _write_fold_audit(
    *,
    train_manifest: list[BreastManifestRecord],
    assignments: dict[str, int],
    num_folds: int,
    output_dir: Path,
    fold_seed: int,
) -> None:
    audit = build_fold_audit(train_manifest, assignments, num_folds)
    audit["fold_seed"] = fold_seed
    (output_dir / "fold_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )


def _try_write_fusion_eval(
    run_dir: Path,
    output_root: Path,
    summary: CVSummary,
    labels: dict[str, int],
    *,
    reference_run: str,
) -> None:
    """If an OOF reference exists, compute and write fusion_eval.json."""
    from ..utils.paths import build_output_paths

    baseline_oof_path = (
        build_output_paths(output_root).runs / reference_run / "cv" / "oof_predictions.csv"
    )
    if not baseline_oof_path.exists():
        log_message(
            run_dir,
            f"run-cv: no fusion eval reference OOF found run={reference_run}, skipping fusion eval",
        )
        return

    try:
        baseline_oof = read_prediction_table_strict(baseline_oof_path)
    except (ValueError, FileNotFoundError) as exc:
        log_message(
            run_dir,
            f"run-cv: cannot load fusion eval reference run={reference_run}: {exc}",
        )
        return

    candidate_oof = summary.oof_predictions

    # Check key alignment
    if set(baseline_oof.keys()) != set(candidate_oof.keys()):
        log_message(
            run_dir,
            f"run-cv: fusion eval reference/candidate OOF key mismatch run={reference_run}, skipping fusion eval",
        )
        return
    if set(baseline_oof.keys()) != set(labels.keys()):
        log_message(run_dir, "run-cv: OOF/label key mismatch, skipping fusion eval")
        return

    # Load baseline fold AUCs if available
    baseline_metrics_path = baseline_oof_path.parent / "metrics.json"
    baseline_fold_aucs = None
    if baseline_metrics_path.exists():
        try:
            baseline_metrics = json.loads(
                baseline_metrics_path.read_text(encoding="utf-8")
            )
            fold_metrics = baseline_metrics.get("fold_metrics", {})
            baseline_fold_aucs = [float(v) for v in fold_metrics.values()]
        except (json.JSONDecodeError, ValueError):
            pass

    candidate_fold_aucs = list(summary.fold_metrics.values())

    report = evaluate_fusion_candidate(
        baseline_oof,
        candidate_oof,
        labels,
        baseline_fold_aucs=baseline_fold_aucs,
        candidate_fold_aucs=candidate_fold_aucs,
    )
    report.write(run_dir / "fusion_eval.json")
    log_message(
        run_dir,
        f"run-cv: fusion eval written"
        f" reference_run={reference_run}"
        f" candidate_auc={report.candidate_oof_auc:.6f}"
        f" blend_gain={report.blend_gain_over_baseline:+.6f}",
    )


def _log_ignored_linear_fusion_settings(run_dir: Path, config: AppConfig) -> None:
    if config.train.fusion_head_variant != "linear":
        return
    ignored_fields: list[str] = []
    if config.train.fusion_hidden_dim != 512:
        ignored_fields.append(f"fusion_hidden_dim={config.train.fusion_hidden_dim}")
    if config.train.fusion_dropout != 0.0:
        ignored_fields.append(f"fusion_dropout={config.train.fusion_dropout}")
    if config.train.fusion_activation != "gelu":
        ignored_fields.append(f"fusion_activation={config.train.fusion_activation}")
    if config.train.fusion_layer_norm:
        ignored_fields.append("fusion_layer_norm=true")
    if config.train.fusion_residual:
        ignored_fields.append("fusion_residual=true")
    if not ignored_fields:
        return
    log_message(
        run_dir,
        "run-cv: fusion_head_variant=linear ignores "
        + ", ".join(ignored_fields),
    )
