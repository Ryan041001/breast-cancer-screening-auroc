"""Fusion evaluation: compare candidate OOF predictions against a baseline."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..model.metrics import (
    binary_auroc,
    blend_predictions,
    fold_spread,
    pairwise_blend_search,
    prediction_correlation,
)
from .submission import read_prediction_table_strict


@dataclass(frozen=True, slots=True)
class FusionEvalReport:
    """Structured evaluation report comparing candidate vs baseline."""

    baseline_oof_auc: float
    candidate_oof_auc: float
    auc_delta: float
    baseline_fold_spread: float
    candidate_fold_spread: float
    best_blend_weight: float
    best_blend_auc: float
    blend_gain_over_baseline: float
    pearson_r: float
    spearman_rho: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def write(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(), encoding="utf-8")


def evaluate_fusion_candidate(
    baseline_oof: dict[str, float],
    candidate_oof: dict[str, float],
    labels: dict[str, int],
    *,
    baseline_fold_aucs: list[float] | None = None,
    candidate_fold_aucs: list[float] | None = None,
) -> FusionEvalReport:
    """Full Phase 1 evaluation of a candidate fusion head vs baseline.

    Args:
        baseline_oof: breast_id -> prediction score (baseline OOF)
        candidate_oof: breast_id -> prediction score (candidate OOF)
        labels: breast_id -> binary label
        baseline_fold_aucs: per-fold AUCs for the baseline
        candidate_fold_aucs: per-fold AUCs for the candidate
    """
    # Verify exact key-set equality
    if set(baseline_oof.keys()) != set(candidate_oof.keys()):
        raise ValueError("Baseline and candidate OOF must have the same breast_id keys")
    if set(baseline_oof.keys()) != set(labels.keys()):
        raise ValueError("OOF predictions and labels must have the same breast_id keys")

    ordered_ids = sorted(labels.keys())
    y_true = [labels[bid] for bid in ordered_ids]

    baseline_scores = [baseline_oof[bid] for bid in ordered_ids]
    candidate_scores = [candidate_oof[bid] for bid in ordered_ids]

    baseline_auc = binary_auroc(y_true, baseline_scores)
    candidate_auc = binary_auroc(y_true, candidate_scores)

    b_spread = fold_spread(baseline_fold_aucs) if baseline_fold_aucs else 0.0
    c_spread = fold_spread(candidate_fold_aucs) if candidate_fold_aucs else 0.0

    best_weight, best_blend_auc = pairwise_blend_search(
        baseline_oof, candidate_oof, labels
    )

    pearson_r, spearman_rho = prediction_correlation(baseline_oof, candidate_oof)

    return FusionEvalReport(
        baseline_oof_auc=baseline_auc,
        candidate_oof_auc=candidate_auc,
        auc_delta=candidate_auc - baseline_auc,
        baseline_fold_spread=b_spread,
        candidate_fold_spread=c_spread,
        best_blend_weight=best_weight,
        best_blend_auc=best_blend_auc,
        blend_gain_over_baseline=best_blend_auc - baseline_auc,
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
    )


def load_and_evaluate(
    baseline_oof_csv: str | Path,
    candidate_oof_csv: str | Path,
    labels: dict[str, int],
    *,
    baseline_fold_aucs: list[float] | None = None,
    candidate_fold_aucs: list[float] | None = None,
) -> FusionEvalReport:
    """Convenience: load OOF tables from disk and evaluate."""
    baseline_oof = read_prediction_table_strict(baseline_oof_csv)
    candidate_oof = read_prediction_table_strict(candidate_oof_csv)
    return evaluate_fusion_candidate(
        baseline_oof,
        candidate_oof,
        labels,
        baseline_fold_aucs=baseline_fold_aucs,
        candidate_fold_aucs=candidate_fold_aucs,
    )


def blend_test_predictions(
    preds_a: dict[str, float],
    preds_b: dict[str, float],
    weight_a: float,
) -> dict[str, float]:
    """Blend two matching test prediction tables."""
    if set(preds_a.keys()) != set(preds_b.keys()):
        raise ValueError("Test prediction sets must have the same breast_id keys")
    return blend_predictions(preds_a, preds_b, weight_a=weight_a)
