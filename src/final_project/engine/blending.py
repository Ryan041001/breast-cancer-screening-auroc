from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..data.manifest import build_train_manifest
from ..model.metrics import binary_auroc
from .submission import read_prediction_table_strict, write_prediction_table


@dataclass(frozen=True, slots=True)
class BlendArtifacts:
    output_dir: Path
    oof_auc: float
    members: tuple[str, ...]
    weights: dict[str, float]


def run_blend_from_spec(
    spec_path: str | Path,
    *,
    runs_root: str | Path,
    train_csv: str | Path = "train.csv",
) -> BlendArtifacts:
    spec_file = Path(spec_path)
    payload = json.loads(spec_file.read_text(encoding="utf-8"))
    run_map = _require_mapping(payload, "run_map")
    weights = _normalize_weights(_require_weights(payload, run_map))
    output_name = str(payload.get("output_name", spec_file.stem))
    output_dir = Path(runs_root) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = {record.breast_id: record.label for record in build_train_manifest(train_csv)}
    oof_predictions = _blend_prediction_tables(
        run_map=run_map,
        weights=weights,
        runs_root=Path(runs_root),
        relative_path=Path("cv/oof_predictions.csv"),
    )
    test_predictions = _blend_prediction_tables(
        run_map=run_map,
        weights=weights,
        runs_root=Path(runs_root),
        relative_path=Path("cv/test_predictions.csv"),
    )

    ordered_ids = sorted(labels)
    oof_auc = binary_auroc(
        [int(labels[breast_id]) for breast_id in ordered_ids],
        [oof_predictions[breast_id] for breast_id in ordered_ids],
    )

    write_prediction_table(oof_predictions, output_dir / "oof_predictions.csv")
    write_prediction_table(test_predictions, output_dir / "test_predictions.csv")

    metrics = {
        "oof_auc": oof_auc,
        "members": list(run_map.keys()),
        "weights": weights,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    blend_payload = {
        "auc": oof_auc,
        "weights": weights,
        "run_map": run_map,
        "source_spec": str(spec_file.resolve()),
    }
    if "removed_members" in payload:
        blend_payload["removed_members"] = payload["removed_members"]
    if "gain_over_prev" in payload:
        blend_payload["gain_over_prev"] = payload["gain_over_prev"]
    (output_dir / "blend.json").write_text(
        json.dumps(blend_payload, indent=2), encoding="utf-8"
    )
    return BlendArtifacts(
        output_dir=output_dir,
        oof_auc=oof_auc,
        members=tuple(run_map.keys()),
        weights=weights,
    )


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, str]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"Blend spec key '{key}' must be a non-empty mapping")
    normalized: dict[str, str] = {}
    for alias, run_name in value.items():
        if not isinstance(alias, str) or not alias:
            raise ValueError(f"Blend spec key '{key}' has an invalid alias")
        if not isinstance(run_name, str) or not run_name:
            raise ValueError(f"Blend spec key '{key}' has an invalid run name")
        normalized[alias] = run_name
    return normalized


def _require_weights(
    payload: dict[str, Any],
    run_map: dict[str, str],
) -> dict[str, float]:
    value = payload.get("weights")
    if not isinstance(value, dict) or not value:
        raise ValueError("Blend spec key 'weights' must be a non-empty mapping")
    if set(value.keys()) != set(run_map.keys()):
        raise ValueError("Blend spec weights must match run_map aliases exactly")
    normalized: dict[str, float] = {}
    for alias, weight in value.items():
        if not isinstance(weight, (int, float)):
            raise ValueError(f"Blend weight for '{alias}' must be numeric")
        normalized[alias] = float(weight)
    return normalized


def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError("Blend weights must sum to a positive value")
    return {alias: weight / total for alias, weight in weights.items()}


def _blend_prediction_tables(
    *,
    run_map: dict[str, str],
    weights: dict[str, float],
    runs_root: Path,
    relative_path: Path,
) -> dict[str, float]:
    blended: dict[str, float] = {}
    expected_keys: set[str] | None = None
    for alias, run_name in run_map.items():
        predictions = read_prediction_table_strict(runs_root / run_name / relative_path)
        if expected_keys is None:
            expected_keys = set(predictions.keys())
        elif set(predictions.keys()) != expected_keys:
            raise ValueError(
                f"Blend member '{alias}' does not match the key-set of the current pool"
            )
        for breast_id, score in predictions.items():
            blended[breast_id] = blended.get(breast_id, 0.0) + weights[alias] * score
    if expected_keys is None:
        raise ValueError("Blend spec does not contain any members")
    return blended
