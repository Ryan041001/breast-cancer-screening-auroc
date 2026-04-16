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
ALLOWED_TRANSFORM_PROFILES = ("baseline", "normaug", "normonly")
ALLOWED_FUSION_HEAD_VARIANTS = ("linear", "mlp")
ALLOWED_FUSION_ACTIVATIONS = ("gelu", "relu")
ALLOWED_SCHEDULERS = ("none", "cosine")
ALLOWED_CACHE_MODES = ("none", "preprocess")
ALLOWED_EXTERNAL_SAMPLERS = ("none", "dataset_label_balanced")


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
    external_data_root: Path | None = None
    external_catalog: Path | None = None
    external_splits_dir: Path | None = None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    seed: int
    device: str
    fold_seed: int | None = None
    train_seed: int | None = None
    warmup_seed: int | None = None


@dataclass(frozen=True, slots=True)
class TrainConfig:
    folds: int
    batch_size: int
    image_size: int
    epochs: int
    num_workers: int
    transform_profile: str = "baseline"
    fusion_head_variant: str = "linear"
    fusion_hidden_dim: int = 512
    fusion_dropout: float = 0.0
    fusion_activation: str = "gelu"
    fusion_layer_norm: bool = False
    fusion_residual: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    scheduler: str = "none"
    min_lr: float = 0.0
    freeze_backbone_epochs: int = 0
    grad_accum_steps: int = 1
    cache_mode: str = "preprocess"
    external_warmup_epochs: int = 0
    external_warmup_batch_size: int = 32
    external_warmup_num_workers: int = 0
    external_warmup_learning_rate: float = 1e-3
    external_warmup_max_samples: int | None = None
    external_sampler: str = "none"
    fusion_eval_reference_run: str = "baseline"


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


def _resolve_project_root(config_file: Path, raw_value: str) -> Path:
    candidate = _resolve_path(config_file.parent, raw_value)
    worktrees_dir = next(
        (ancestor for ancestor in config_file.parents if ancestor.name == ".worktrees"),
        None,
    )
    if worktrees_dir is not None and candidate.is_relative_to(worktrees_dir):
        return worktrees_dir.parent
    return candidate


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


def _optional_int(payload: Mapping[str, object], key: str) -> int | None:
    if key not in payload:
        return None
    value = payload[key]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Config key '{key}' must be an integer")
    return value


def _require_int_at_least(payload: Mapping[str, object], key: str, minimum: int) -> int:
    value = _require_int(payload, key)
    if value < minimum:
        raise ValueError(f"Config key '{key}' must be at least {minimum}")
    return value


def _optional_resolved_path(
    base_dir: Path,
    payload: Mapping[str, object],
    key: str,
) -> Path | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config key '{key}' must be a string")
    return _resolve_path(base_dir, value)


def load_config(config_path: str | Path) -> AppConfig:
    config_file = Path(config_path).resolve()
    raw_payload: object = cast(
        object, yaml.safe_load(config_file.read_text(encoding="utf-8"))
    )
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

    project_root = _resolve_project_root(
        config_file,
        _require_string(paths_payload, "project_root"),
    )
    transform_profile = train_payload.get("transform_profile", "baseline")
    if not isinstance(transform_profile, str):
        raise ValueError("Config key 'transform_profile' must be a string")
    if transform_profile not in ALLOWED_TRANSFORM_PROFILES:
        allowed = ", ".join(ALLOWED_TRANSFORM_PROFILES)
        raise ValueError(f"Config key 'transform_profile' must be one of: {allowed}")

    # --- fusion head settings (backward-compatible defaults) ---
    fusion_head_variant = str(train_payload.get("fusion_head_variant", "linear"))
    if fusion_head_variant not in ALLOWED_FUSION_HEAD_VARIANTS:
        allowed = ", ".join(ALLOWED_FUSION_HEAD_VARIANTS)
        raise ValueError(
            f"Config key 'fusion_head_variant' must be one of: {allowed}"
        )

    fusion_hidden_dim_raw = train_payload.get("fusion_hidden_dim", 512)
    if not isinstance(fusion_hidden_dim_raw, int) or isinstance(fusion_hidden_dim_raw, bool):
        raise ValueError("Config key 'fusion_hidden_dim' must be an integer")
    if fusion_hidden_dim_raw < 1:
        raise ValueError("Config key 'fusion_hidden_dim' must be at least 1")
    fusion_hidden_dim = fusion_hidden_dim_raw

    fusion_dropout_raw = train_payload.get("fusion_dropout", 0.0)
    if isinstance(fusion_dropout_raw, bool):
        raise ValueError("Config key 'fusion_dropout' must be a number")
    if not isinstance(fusion_dropout_raw, (int, float)):
        raise ValueError("Config key 'fusion_dropout' must be a number")
    fusion_dropout = float(fusion_dropout_raw)
    if not (0.0 <= fusion_dropout <= 1.0):
        raise ValueError("Config key 'fusion_dropout' must be between 0.0 and 1.0")

    fusion_activation = str(train_payload.get("fusion_activation", "gelu"))
    if fusion_activation not in ALLOWED_FUSION_ACTIVATIONS:
        allowed = ", ".join(ALLOWED_FUSION_ACTIVATIONS)
        raise ValueError(
            f"Config key 'fusion_activation' must be one of: {allowed}"
        )

    fusion_layer_norm = bool(train_payload.get("fusion_layer_norm", False))
    fusion_residual = bool(train_payload.get("fusion_residual", False))
    scheduler = str(train_payload.get("scheduler", "none"))
    if scheduler not in ALLOWED_SCHEDULERS:
        allowed = ", ".join(ALLOWED_SCHEDULERS)
        raise ValueError(f"Config key 'scheduler' must be one of: {allowed}")

    cache_mode = str(train_payload.get("cache_mode", "preprocess"))
    if cache_mode not in ALLOWED_CACHE_MODES:
        allowed = ", ".join(ALLOWED_CACHE_MODES)
        raise ValueError(f"Config key 'cache_mode' must be one of: {allowed}")

    external_sampler = str(train_payload.get("external_sampler", "none"))
    if external_sampler not in ALLOWED_EXTERNAL_SAMPLERS:
        allowed = ", ".join(ALLOWED_EXTERNAL_SAMPLERS)
        raise ValueError(f"Config key 'external_sampler' must be one of: {allowed}")

    fusion_eval_reference_run = str(
        train_payload.get("fusion_eval_reference_run", "baseline")
    ).strip()
    if not fusion_eval_reference_run:
        raise ValueError("Config key 'fusion_eval_reference_run' must be a non-empty string")

    learning_rate_raw = train_payload.get("learning_rate", 1e-3)
    if isinstance(learning_rate_raw, bool) or not isinstance(
        learning_rate_raw, (int, float)
    ):
        raise ValueError("Config key 'learning_rate' must be a number")
    learning_rate = float(learning_rate_raw)
    if learning_rate <= 0.0:
        raise ValueError("Config key 'learning_rate' must be greater than 0.0")

    weight_decay_raw = train_payload.get("weight_decay", 1e-2)
    if isinstance(weight_decay_raw, bool) or not isinstance(
        weight_decay_raw, (int, float)
    ):
        raise ValueError("Config key 'weight_decay' must be a number")
    weight_decay = float(weight_decay_raw)
    if weight_decay < 0.0:
        raise ValueError("Config key 'weight_decay' must be at least 0.0")

    min_lr_raw = train_payload.get("min_lr", 0.0)
    if isinstance(min_lr_raw, bool) or not isinstance(min_lr_raw, (int, float)):
        raise ValueError("Config key 'min_lr' must be a number")
    min_lr = float(min_lr_raw)
    if min_lr < 0.0:
        raise ValueError("Config key 'min_lr' must be at least 0.0")
    if scheduler == "cosine" and min_lr > learning_rate:
        raise ValueError("Config key 'min_lr' must be less than or equal to learning_rate")

    freeze_backbone_epochs = _require_int_at_least(
        {"freeze_backbone_epochs": train_payload.get("freeze_backbone_epochs", 0)},
        "freeze_backbone_epochs",
        0,
    )
    grad_accum_steps = _require_int_at_least(
        {"grad_accum_steps": train_payload.get("grad_accum_steps", 1)},
        "grad_accum_steps",
        1,
    )

    external_warmup_epochs = _require_int_at_least(
        {"external_warmup_epochs": train_payload.get("external_warmup_epochs", 0)},
        "external_warmup_epochs",
        0,
    )
    external_warmup_batch_size = _require_int_at_least(
        {
            "external_warmup_batch_size": train_payload.get(
                "external_warmup_batch_size", 32
            )
        },
        "external_warmup_batch_size",
        1,
    )
    external_warmup_num_workers = _require_int_at_least(
        {
            "external_warmup_num_workers": train_payload.get(
                "external_warmup_num_workers", 0
            )
        },
        "external_warmup_num_workers",
        0,
    )
    external_warmup_learning_rate_raw = train_payload.get(
        "external_warmup_learning_rate", 1e-3
    )
    if isinstance(external_warmup_learning_rate_raw, bool):
        raise ValueError("Config key 'external_warmup_learning_rate' must be a number")
    if not isinstance(external_warmup_learning_rate_raw, (int, float)):
        raise ValueError("Config key 'external_warmup_learning_rate' must be a number")
    external_warmup_learning_rate = float(external_warmup_learning_rate_raw)
    if external_warmup_learning_rate <= 0.0:
        raise ValueError(
            "Config key 'external_warmup_learning_rate' must be greater than 0.0"
        )

    external_warmup_max_samples_raw = train_payload.get("external_warmup_max_samples")
    if external_warmup_max_samples_raw is None:
        external_warmup_max_samples = None
    else:
        if (
            not isinstance(external_warmup_max_samples_raw, int)
            or isinstance(external_warmup_max_samples_raw, bool)
        ):
            raise ValueError(
                "Config key 'external_warmup_max_samples' must be an integer"
            )
        if external_warmup_max_samples_raw < 1:
            raise ValueError(
                "Config key 'external_warmup_max_samples' must be at least 1"
            )
        external_warmup_max_samples = external_warmup_max_samples_raw

    external_data_root = _optional_resolved_path(project_root, paths_payload, "external_data_root")
    external_catalog = _optional_resolved_path(project_root, paths_payload, "external_catalog")
    external_splits_dir = _optional_resolved_path(
        project_root,
        paths_payload,
        "external_splits_dir",
    )
    if external_data_root is not None:
        if external_catalog is None:
            external_catalog = external_data_root / "catalog.csv"
        if external_splits_dir is None:
            external_splits_dir = external_data_root / "splits"
    if external_warmup_epochs > 0 and (
        external_data_root is None
        or external_catalog is None
        or external_splits_dir is None
    ):
        raise ValueError(
            "External warmup requires 'paths.external_data_root' or explicit "
            "'external_catalog' and 'external_splits_dir' values"
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
            external_data_root=external_data_root,
            external_catalog=external_catalog,
            external_splits_dir=external_splits_dir,
        ),
        runtime=RuntimeConfig(
            seed=_require_int(runtime_payload, "seed"),
            device=_require_string(runtime_payload, "device"),
            fold_seed=_optional_int(runtime_payload, "fold_seed"),
            train_seed=_optional_int(runtime_payload, "train_seed"),
            warmup_seed=_optional_int(runtime_payload, "warmup_seed"),
        ),
        train=TrainConfig(
            folds=_require_int_at_least(train_payload, "folds", 2),
            batch_size=_require_int_at_least(train_payload, "batch_size", 1),
            image_size=_require_int_at_least(train_payload, "image_size", 1),
            epochs=_require_int_at_least(train_payload, "epochs", 1),
            num_workers=_require_int_at_least(train_payload, "num_workers", 0),
            transform_profile=transform_profile,
            fusion_head_variant=fusion_head_variant,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_dropout=fusion_dropout,
            fusion_activation=fusion_activation,
            fusion_layer_norm=fusion_layer_norm,
            fusion_residual=fusion_residual,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            scheduler=scheduler,
            min_lr=min_lr,
            freeze_backbone_epochs=freeze_backbone_epochs,
            grad_accum_steps=grad_accum_steps,
            cache_mode=cache_mode,
            external_warmup_epochs=external_warmup_epochs,
            external_warmup_batch_size=external_warmup_batch_size,
            external_warmup_num_workers=external_warmup_num_workers,
            external_warmup_learning_rate=external_warmup_learning_rate,
            external_warmup_max_samples=external_warmup_max_samples,
            external_sampler=external_sampler,
            fusion_eval_reference_run=fusion_eval_reference_run,
        ),
    )
