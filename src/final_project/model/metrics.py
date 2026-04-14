from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import roc_auc_score


def binary_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    return float(roc_auc_score(y_true, y_score))
