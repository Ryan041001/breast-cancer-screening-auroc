from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..config import AppConfig, load_config
from ..data.manifest import build_train_manifest
from ..model.metrics import binary_auroc
from ..utils.logging import log_message
from ..utils.paths import build_output_paths
from .fusion_eval import blend_test_predictions, evaluate_fusion_candidate
from .run_cv import run_cross_validation
from .submission import read_prediction_table_strict, write_prediction_table


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    config_path: str
    experiment_name: str
    status: str
    mean_auc: float | None
    oof_auc: float | None
    best_blend_auc: float | None
    blend_gain: float | None
    metrics_path: str
    fusion_eval_path: str | None


@dataclass(frozen=True, slots=True)
class TuningIterationArtifacts:
    report_dir: Path
    leaderboard_path: Path
    best_blend_dir: Path | None
    summaries: list[ExperimentSummary]


def run_tuning_iteration(
    config_paths: list[str | Path],
    *,
    output_root: str | Path,
    report_name: str,
    baseline_run: str = "baseline",
    skip_existing: bool = True,
) -> TuningIterationArtifacts:
    output_paths = build_output_paths(output_root)
    report_dir = output_paths.root / "research" / report_name
    report_dir.mkdir(parents=True, exist_ok=True)
    log_message(
        report_dir,
        f"tune-iterate: start configs={len(config_paths)} baseline_run={baseline_run} skip_existing={skip_existing}",
    )

    summaries: list[ExperimentSummary] = []
    for raw_config_path in config_paths:
        config = load_config(raw_config_path)
        experiment_dir = output_paths.runs / config.experiment.name
        metrics_path = experiment_dir / "cv" / "metrics.json"
        if skip_existing and metrics_path.exists():
            log_message(
                report_dir,
                f"tune-iterate: skip existing experiment={config.experiment.name}",
            )
            status = "skipped_existing"
        else:
            log_message(
                report_dir,
                f"tune-iterate: run experiment={config.experiment.name} config={Path(raw_config_path)}",
            )
            artifacts = run_cross_validation(config)
            write_prediction_table(
                artifacts.test_predictions,
                experiment_dir / "test_predictions.csv",
            )
            status = "completed"

        summaries.append(
            _summarize_experiment(
                config=config,
                config_path=Path(raw_config_path),
                output_paths=output_paths,
                status=status,
            )
        )

    summaries.sort(
        key=lambda item: (
            item.mean_auc is not None,
            item.mean_auc if item.mean_auc is not None else float("-inf"),
        ),
        reverse=True,
    )
    leaderboard_path = _write_leaderboard(report_dir, summaries)
    best_blend_dir = _backup_best_blend(
        report_dir=report_dir,
        output_paths=output_paths,
        baseline_run=baseline_run,
        summaries=summaries,
    )
    log_message(
        report_dir,
        f"tune-iterate: complete leaderboard={leaderboard_path} best_blend_dir={best_blend_dir}",
    )
    return TuningIterationArtifacts(
        report_dir=report_dir,
        leaderboard_path=leaderboard_path,
        best_blend_dir=best_blend_dir,
        summaries=summaries,
    )


def _summarize_experiment(
    *,
    config: AppConfig,
    config_path: Path,
    output_paths: Any,
    status: str,
) -> ExperimentSummary:
    experiment_dir = output_paths.runs / config.experiment.name
    metrics_path = experiment_dir / "cv" / "metrics.json"
    fusion_eval_path = experiment_dir / "cv" / "fusion_eval.json"
    baseline_experiment_dir = output_paths.runs / "baseline"

    mean_auc: float | None = None
    if metrics_path.exists():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        mean_auc_value = metrics_payload.get("mean_auc")
        if isinstance(mean_auc_value, (int, float)):
            mean_auc = float(mean_auc_value)

    oof_auc: float | None = None
    oof_predictions_path = experiment_dir / "cv" / "oof_predictions.csv"
    labels = {
        record.breast_id: record.label
        for record in build_train_manifest(config.paths.train_csv)
        if record.label is not None
    }
    if oof_predictions_path.exists():
        predictions = read_prediction_table_strict(oof_predictions_path)
        ordered_ids = sorted(predictions.keys())
        oof_auc = binary_auroc(
            [labels[breast_id] for breast_id in ordered_ids],
            [predictions[breast_id] for breast_id in ordered_ids],
        )

    best_blend_auc: float | None = None
    blend_gain: float | None = None
    fusion_eval_path_value: str | None = None
    if (
        config.experiment.name != "baseline"
        and not fusion_eval_path.exists()
        and oof_predictions_path.exists()
        and (baseline_experiment_dir / "cv" / "oof_predictions.csv").exists()
    ):
        _write_missing_fusion_eval(
            experiment_dir=experiment_dir,
            labels=labels,
            baseline_experiment_dir=baseline_experiment_dir,
        )
    if fusion_eval_path.exists():
        fusion_payload = json.loads(fusion_eval_path.read_text(encoding="utf-8"))
        best_blend_auc_value = fusion_payload.get("best_blend_auc")
        blend_gain_value = fusion_payload.get("blend_gain_over_baseline")
        if isinstance(best_blend_auc_value, (int, float)):
            best_blend_auc = float(best_blend_auc_value)
        if isinstance(blend_gain_value, (int, float)):
            blend_gain = float(blend_gain_value)
        fusion_eval_path_value = str(fusion_eval_path.resolve())

    return ExperimentSummary(
        config_path=str(config_path.resolve()),
        experiment_name=config.experiment.name,
        status=status,
        mean_auc=mean_auc,
        oof_auc=oof_auc,
        best_blend_auc=best_blend_auc,
        blend_gain=blend_gain,
        metrics_path=str(metrics_path.resolve()),
        fusion_eval_path=fusion_eval_path_value,
    )


def _write_missing_fusion_eval(
    *,
    experiment_dir: Path,
    labels: dict[str, int],
    baseline_experiment_dir: Path,
) -> None:
    baseline_oof_path = baseline_experiment_dir / "cv" / "oof_predictions.csv"
    candidate_oof_path = experiment_dir / "cv" / "oof_predictions.csv"
    baseline_metrics_path = baseline_experiment_dir / "cv" / "metrics.json"
    candidate_metrics_path = experiment_dir / "cv" / "metrics.json"

    baseline_oof = read_prediction_table_strict(baseline_oof_path)
    candidate_oof = read_prediction_table_strict(candidate_oof_path)

    baseline_fold_aucs = _read_fold_aucs(baseline_metrics_path)
    candidate_fold_aucs = _read_fold_aucs(candidate_metrics_path)
    report = evaluate_fusion_candidate(
        baseline_oof,
        candidate_oof,
        labels,
        baseline_fold_aucs=baseline_fold_aucs,
        candidate_fold_aucs=candidate_fold_aucs,
    )
    report.write(experiment_dir / "cv" / "fusion_eval.json")


def _read_fold_aucs(metrics_path: Path) -> list[float] | None:
    if not metrics_path.exists():
        return None
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    fold_metrics = payload.get("fold_metrics")
    if not isinstance(fold_metrics, dict):
        return None
    values: list[float] = []
    for value in fold_metrics.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values if values else None


def _write_leaderboard(
    report_dir: Path,
    summaries: list[ExperimentSummary],
) -> Path:
    json_path = report_dir / "leaderboard.json"
    csv_path = report_dir / "leaderboard.csv"
    markdown_path = report_dir / "leaderboard.md"

    json_path.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2),
        encoding="utf-8",
    )

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment_name",
                "status",
                "mean_auc",
                "oof_auc",
                "best_blend_auc",
                "blend_gain",
                "config_path",
                "metrics_path",
                "fusion_eval_path",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(asdict(summary))

    lines = [
        "# Tuning Leaderboard",
        "",
        "| Experiment | Status | mean_auc | oof_auc | best_blend_auc | blend_gain |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            "|"
            f" {summary.experiment_name} | {summary.status} |"
            f" {_format_metric(summary.mean_auc)} | {_format_metric(summary.oof_auc)} |"
            f" {_format_metric(summary.best_blend_auc)} | {_format_metric(summary.blend_gain)} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path


def _backup_best_blend(
    *,
    report_dir: Path,
    output_paths: Any,
    baseline_run: str,
    summaries: list[ExperimentSummary],
) -> Path | None:
    candidates = [
        summary
        for summary in summaries
        if summary.experiment_name != baseline_run
        and summary.blend_gain is not None
        and summary.best_blend_auc is not None
        and summary.fusion_eval_path is not None
        and summary.blend_gain > 0.0
    ]
    if not candidates:
        return None

    best_candidate = max(
        candidates,
        key=lambda summary: (
            summary.best_blend_auc if summary.best_blend_auc is not None else float("-inf")
        ),
    )
    fusion_eval_payload = json.loads(
        Path(best_candidate.fusion_eval_path).read_text(encoding="utf-8")
    )
    weight_baseline = float(fusion_eval_payload["best_blend_weight"])
    baseline_test_path = output_paths.runs / baseline_run / "cv" / "test_predictions.csv"
    candidate_test_path = (
        output_paths.runs / best_candidate.experiment_name / "cv" / "test_predictions.csv"
    )
    baseline_test = read_prediction_table_strict(baseline_test_path)
    candidate_test = read_prediction_table_strict(candidate_test_path)
    blended_test = blend_test_predictions(
        baseline_test,
        candidate_test,
        weight_a=weight_baseline,
    )

    blend_dir = output_paths.runs / f"tune_iter_{report_dir.name}_best_blend"
    blend_dir.mkdir(parents=True, exist_ok=True)
    write_prediction_table(blended_test, blend_dir / "test_predictions.csv")
    (blend_dir / "blend.json").write_text(
        json.dumps(
            {
                "baseline_run": baseline_run,
                "candidate_run": best_candidate.experiment_name,
                "weight_baseline": weight_baseline,
                "weight_candidate": 1.0 - weight_baseline,
                "best_blend_auc": best_candidate.best_blend_auc,
                "blend_gain": best_candidate.blend_gain,
                "baseline_test_predictions_path": str(baseline_test_path.resolve()),
                "candidate_test_predictions_path": str(candidate_test_path.resolve()),
                "fusion_eval_path": best_candidate.fusion_eval_path,
                "generated_test_predictions_path": str(
                    (blend_dir / "test_predictions.csv").resolve()
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log_message(
        report_dir,
        f"tune-iterate: best blend candidate={best_candidate.experiment_name} blend_dir={blend_dir}",
    )
    return blend_dir


def _format_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.6f}"
