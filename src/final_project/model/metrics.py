from __future__ import annotations

from collections.abc import Sequence
from statistics import mean

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def binary_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    return float(roc_auc_score(y_true, y_score))


def fold_spread(fold_aucs: Sequence[float]) -> float:
    """Max minus min fold AUC – measures instability across folds."""
    if len(fold_aucs) < 2:
        return 0.0
    return max(fold_aucs) - min(fold_aucs)


def blend_predictions(
    preds_a: dict[str, float],
    preds_b: dict[str, float],
    weight_a: float,
) -> dict[str, float]:
    """Weighted blend: weight_a * A + (1 - weight_a) * B."""
    weight_b = 1.0 - weight_a
    return {
        breast_id: weight_a * preds_a[breast_id] + weight_b * preds_b[breast_id]
        for breast_id in preds_a
    }


def pairwise_blend_search(
    baseline_preds: dict[str, float],
    candidate_preds: dict[str, float],
    labels: dict[str, int],
    *,
    weight_grid: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Fixed-grid search for best blend weight. Returns (best_weight, best_auc).

    weight_grid: weights for baseline. Default is 0.1..0.9 in 0.1 steps.
    """
    if weight_grid is None:
        weight_grid = [round(w * 0.1, 2) for w in range(1, 10)]

    # Verify exact key match
    if set(baseline_preds.keys()) != set(candidate_preds.keys()):
        raise ValueError("Baseline and candidate must have exactly the same breast_id keys")
    if set(baseline_preds.keys()) != set(labels.keys()):
        raise ValueError("Predictions and labels must have exactly the same breast_id keys")

    ordered_ids = sorted(baseline_preds.keys())
    y_true = [labels[bid] for bid in ordered_ids]

    best_weight = 0.5
    best_auc = -1.0

    for w in weight_grid:
        blended = blend_predictions(baseline_preds, candidate_preds, weight_a=w)
        y_score = [blended[bid] for bid in ordered_ids]
        try:
            auc = float(roc_auc_score(y_true, y_score))
        except ValueError:
            auc = 0.5
        if auc > best_auc or (auc == best_auc and w > best_weight):
            best_auc = auc
            best_weight = w

    return best_weight, best_auc


def prediction_correlation(
    preds_a: dict[str, float],
    preds_b: dict[str, float],
) -> tuple[float, float]:
    """Return (Pearson r, Spearman rho) between two aligned prediction sets."""
    if set(preds_a.keys()) != set(preds_b.keys()):
        raise ValueError("Prediction sets must have exactly the same breast_id keys")

    ordered_ids = sorted(preds_a.keys())
    a_vals = [preds_a[bid] for bid in ordered_ids]
    b_vals = [preds_b[bid] for bid in ordered_ids]

    if len(ordered_ids) < 2:
        return 0.0, 0.0

    pearson_r, _ = pearsonr(a_vals, b_vals)
    spearman_rho, _ = spearmanr(a_vals, b_vals)
    return float(pearson_r), float(spearman_rho)
