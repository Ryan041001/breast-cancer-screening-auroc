from __future__ import annotations

import pytest

from final_project.model.metrics import (
    binary_auroc,
    blend_predictions,
    fold_spread,
    pairwise_blend_search,
    prediction_correlation,
)


def test_fold_spread_uses_max_minus_min_fold_auc() -> None:
    assert fold_spread([0.8, 0.9, 0.7]) == pytest.approx(0.2)


def test_fold_spread_returns_zero_for_single_fold() -> None:
    assert fold_spread([0.85]) == 0.0


def test_fold_spread_returns_zero_for_empty() -> None:
    assert fold_spread([]) == 0.0


def test_blend_predictions_weighted() -> None:
    a = {"b1": 0.8, "b2": 0.2}
    b = {"b1": 0.4, "b2": 0.6}
    result = blend_predictions(a, b, weight_a=0.5)
    assert result["b1"] == pytest.approx(0.6)
    assert result["b2"] == pytest.approx(0.4)


def test_pairwise_blend_search_uses_fixed_weight_grid() -> None:
    # Perfect baseline, random candidate => best weight should favor baseline
    baseline = {"b1": 0.1, "b2": 0.2, "b3": 0.8, "b4": 0.9}
    candidate = {"b1": 0.5, "b2": 0.5, "b3": 0.5, "b4": 0.5}
    labels = {"b1": 0, "b2": 0, "b3": 1, "b4": 1}

    best_weight, best_auc = pairwise_blend_search(baseline, candidate, labels)

    # Should prefer mostly baseline (high weight)
    assert best_weight >= 0.5
    assert best_auc >= 0.9


def test_pairwise_blend_search_requires_exact_oof_key_match() -> None:
    baseline = {"b1": 0.5}
    candidate = {"b2": 0.5}
    labels = {"b1": 0}

    with pytest.raises(ValueError, match="same breast_id keys"):
        pairwise_blend_search(baseline, candidate, labels)


def test_prediction_correlation_reports_pearson_and_spearman() -> None:
    a = {"b1": 0.1, "b2": 0.5, "b3": 0.9}
    b = {"b1": 0.2, "b2": 0.6, "b3": 1.0}

    pearson_r, spearman_rho = prediction_correlation(a, b)

    assert pearson_r == pytest.approx(1.0)
    assert spearman_rho == pytest.approx(1.0)


def test_prediction_correlation_rejects_mismatched_keys() -> None:
    a = {"b1": 0.5}
    b = {"b2": 0.5}

    with pytest.raises(ValueError, match="same breast_id keys"):
        prediction_correlation(a, b)
