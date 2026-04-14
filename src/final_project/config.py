from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


REQUIRED_SECTIONS = ("experiment", "paths", "runtime", "train")
REQUIRED_EXPERIMENT_KEYS = ("name",)
REQUIRED_PATH_KEYS = (
    "project_root",
    "train_csv",
    "train_images",
    "test_images",
    "submission_template",
    "output_root",
)
REQUIRED_RUNTIME_KEYS = ("seed", "device")
REQUIRED_TRAIN_KEYS = ("folds", "batch_size", "image_size", "epochs", "num_workers")


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    name: str


@dataclass(frozen=True, slots=True)
class PathsConfig:
    project_root: Path
    train_csv: Path
    train_images: Path
    test_images: Path
    submission_template: Path
    output_root: Path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int
    device: str


@dataclass(frozen=True, slots=True)
class TrainConfig:
    folds: int
    batch_size: int
    image_size: int
    epochs: int
    num_workers: int


@dataclass(frozen=True, slots=True)
class AppConfig:
    experiment: ExperimentConfig
    paths: PathsConfig
    runtime: RuntimeConfig
    train: TrainConfig


def _ensure_mapping(value: object, section_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{section_name}' must be a mapping")
    return cast(dict[str, object], value)


def _resolve_path(base_dir: Path, raw_value: str) -> Path:
    candidate = Path(raw_value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (base_dir / candidate).resolve()


def _missing_keys(
    payload: Mapping[str, object], required_keys: tuple[str, ...]
) -> list[str]:
    return [key for key in required_keys if key not in payload]


def _require_string(payload: Mapping[str, object], key: str) -> str:
    if key not in payload:
        raise ValueError(f"Missing required config key '{key}'")
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string")
    return value


def _require_int(payload: Mapping[str, object], key: str) -> int:
    if key not in payload:
        raise ValueError(f"Missing required config key '{key}'")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config key '{key}' must be an integer")
    return value


def _require_int_at_least(payload: Mapping[str, object], key: str, minimum: int) -> int:
    value = _require_int(payload, key)
    if value < minimum:
        raise ValueError(f"Config key '{key}' must be at least {minimum}")
    return value


def load_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path).resolve()
    raw_payload = cast(object, yaml.safe_load(config_file.read_text(encoding="utf-8")))
    if raw_payload is None:
        raw_payload = {}
    payload = _ensure_mapping(raw_payload, "root")

    missing_sections = [
        section for section in REQUIRED_SECTIONS if section not in payload
    ]
    if missing_sections:
        missing = ", ".join(missing_sections)
        raise ValueError(f"Missing required config sections: {missing}")

    experiment_payload = _ensure_mapping(payload["experiment"], "experiment")
    paths_payload = _ensure_mapping(payload["paths"], "paths")
    runtime_payload = _ensure_mapping(payload["runtime"], "runtime")
    train_payload = _ensure_mapping(payload["train"], "train")

    missing_path_keys = _missing_keys(paths_payload, REQUIRED_PATH_KEYS)
    if missing_path_keys:
        missing = ", ".join(missing_path_keys)
        raise ValueError(f"Missing required path config keys: {missing}")

    missing_experiment_keys = _missing_keys(
        experiment_payload, REQUIRED_EXPERIMENT_KEYS
    )
    if missing_experiment_keys:
        missing = ", ".join(missing_experiment_keys)
        raise ValueError(f"Missing required experiment config keys: {missing}")

    missing_runtime_keys = _missing_keys(runtime_payload, REQUIRED_RUNTIME_KEYS)
    if missing_runtime_keys:
        missing = ", ".join(missing_runtime_keys)
        raise ValueError(f"Missing required runtime config keys: {missing}")

    missing_train_keys = _missing_keys(train_payload, REQUIRED_TRAIN_KEYS)
    if missing_train_keys:
        missing = ", ".join(missing_train_keys)
        raise ValueError(f"Missing required train config keys: {missing}")

    project_root = _resolve_path(
        config_file.parent,
        _require_string(paths_payload, "project_root"),
    )

    return AppConfig(
        experiment=ExperimentConfig(name=_require_string(experiment_payload, "name")),
        paths=PathsConfig(
            project_root=project_root,
            train_images=_resolve_path(
                project_root,
                _require_string(paths_payload, "train_images"),
            ),
            train_csv=_resolve_path(
                project_root,
                _require_string(paths_payload, "train_csv"),
            ),
            test_images=_resolve_path(
                project_root,
                _require_string(paths_payload, "test_images"),
            ),
            submission_template=_resolve_path(
                project_root,
                _require_string(paths_payload, "submission_template"),
            ),
            output_root=_resolve_path(
                project_root,
                _require_string(paths_payload, "output_root"),
            ),
        ),
        runtime=RuntimeConfig(
            seed=_require_int(runtime_payload, "seed"),
            device=_require_string(runtime_payload, "device"),
        ),
        train=TrainConfig(
            folds=_require_int_at_least(train_payload, "folds", 2),
            batch_size=_require_int_at_least(train_payload, "batch_size", 1),
            image_size=_require_int_at_least(train_payload, "image_size", 1),
            epochs=_require_int_at_least(train_payload, "epochs", 1),
            num_workers=_require_int_at_least(train_payload, "num_workers", 0),
        ),
    )
