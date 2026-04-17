from pathlib import Path

import pytest
import yaml

from final_project.config import AppConfig, load_config


def _write_config(path: Path, payload: dict[str, object]) -> None:
    _ = path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def _base_payload() -> dict[str, object]:
    return {
        "experiment": {"name": "demo"},
        "paths": {
            "project_root": ".",
            "train_csv": "train.csv",
            "train_images": "train_img",
            "test_images": "test_img",
            "submission_template": "name_sid_submission.csv",
            "output_root": "outputs",
        },
        "runtime": {"seed": 7, "device": "cpu"},
        "train": {
            "folds": 5,
            "batch_size": 4,
            "image_size": 384,
            "epochs": 1,
            "num_workers": 0,
        },
    }


def test_load_config_returns_expected_fields() -> None:
    config = load_config(Path("configs/smoke.yaml"))
    repo_root = Path(__file__).resolve().parents[1]

    assert isinstance(config, AppConfig)
    assert config.experiment.name == "smoke"
    assert config.paths.project_root == repo_root
    assert config.paths.train_csv == repo_root / "train.csv"
    assert config.paths.train_images == repo_root / "train_img"
    assert config.paths.test_images == repo_root / "test_img"
    assert config.paths.output_root == repo_root / "outputs"
    assert config.runtime.seed == 7
    assert config.runtime.device == "cuda"
    assert config.runtime.fold_seed is None
    assert config.runtime.train_seed is None
    assert config.runtime.warmup_seed is None
    assert config.train.batch_size == 2
    assert config.train.image_size == 384
    assert config.train.transform_profile == "baseline"
    assert config.paths.external_data_root is None
    assert config.train.external_warmup_epochs == 0
    assert config.train.learning_rate == pytest.approx(1e-3)
    assert config.train.weight_decay == pytest.approx(1e-2)
    assert config.train.scheduler == "none"
    assert config.train.cache_mode == "preprocess"
    assert config.train.external_sampler == "none"
    assert config.train.fusion_eval_reference_run == "baseline"


def test_load_config_loads_default_linear_fusion_head_values() -> None:
    config = load_config(Path("configs/smoke.yaml"))

    assert config.train.fusion_head_variant == "linear"
    assert config.train.fusion_hidden_dim == 512
    assert config.train.fusion_dropout == 0.0
    assert config.train.fusion_activation == "gelu"
    assert config.train.fusion_layer_norm is False
    assert config.train.fusion_residual is False
    assert config.train.freeze_backbone_epochs == 0
    assert config.train.grad_accum_steps == 1
    assert config.train.external_warmup_batch_size == 16
    assert config.train.external_warmup_learning_rate == 0.001


def test_load_config_resolves_external_warmup_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "external.yaml"
    payload = _base_payload()
    payload["paths"] = {
        **payload["paths"],
        "external_data_root": "MammoNet32k_new/MammoNet32k_new",
    }
    payload["train"] = {
        **payload["train"],
        "external_warmup_epochs": 2,
        "external_warmup_batch_size": 32,
        "external_warmup_learning_rate": 5e-4,
        "external_warmup_max_samples": 128,
    }
    _write_config(config_path, payload)

    config = load_config(config_path)

    assert config.paths.external_data_root == (
        tmp_path / "MammoNet32k_new" / "MammoNet32k_new"
    )
    assert config.paths.external_catalog == (
        tmp_path / "MammoNet32k_new" / "MammoNet32k_new" / "catalog.csv"
    )
    assert config.paths.external_splits_dir == (
        tmp_path / "MammoNet32k_new" / "MammoNet32k_new" / "splits"
    )
    assert config.train.external_warmup_epochs == 2
    assert config.train.external_warmup_batch_size == 32
    assert config.train.external_warmup_learning_rate == pytest.approx(5e-4)
    assert config.train.external_warmup_max_samples == 128


def test_load_config_rejects_external_warmup_without_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_external.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        "external_warmup_epochs": 1,
    }
    _write_config(config_path, payload)

    with pytest.raises(ValueError, match="External warmup requires"):
        _ = load_config(config_path)


def test_load_config_resolves_project_root_from_sibling_worktree(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "Final_Project"
    config_path = (
        repo_root
        / ".worktrees"
        / "fusion-head-tuning-20260415"
        / "configs"
        / "smoke.yaml"
    )
    config_path.parent.mkdir(parents=True)
    _write_config(
        config_path,
        {
            "experiment": {"name": "smoke"},
            "paths": {
                "project_root": "..",
                "train_csv": "train.csv",
                "train_images": "train_img",
                "test_images": "test_img",
                "submission_template": "name_sid_submission.csv",
                "output_root": "outputs",
            },
            "runtime": {"seed": 7, "device": "cuda"},
            "train": {
                "folds": 5,
                "batch_size": 2,
                "image_size": 384,
                "epochs": 1,
                "num_workers": 0,
            },
        },
    )

    config = load_config(config_path)

    assert config.paths.project_root == repo_root
    assert config.paths.train_csv == repo_root / "train.csv"


@pytest.mark.parametrize("fusion_head_variant", ["linear", "mlp", "transformer"])
def test_load_config_accepts_supported_fusion_head_variants(
    tmp_path: Path,
    fusion_head_variant: str,
) -> None:
    config_path = tmp_path / f"{fusion_head_variant}.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        "fusion_head_variant": fusion_head_variant,
    }
    _write_config(config_path, payload)

    config = load_config(config_path)

    assert config.train.fusion_head_variant == fusion_head_variant


def test_load_config_accepts_transformer_fusion_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "transformer.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        "fusion_head_variant": "transformer",
        "fusion_hidden_dim": 256,
        "fusion_dropout": 0.1,
        "fusion_layer_norm": True,
        "fusion_transformer_layers": 2,
        "fusion_transformer_heads": 4,
    }
    _write_config(config_path, payload)

    config = load_config(config_path)

    assert config.train.fusion_head_variant == "transformer"
    assert config.train.fusion_transformer_layers == 2
    assert config.train.fusion_transformer_heads == 4


def test_load_config_rejects_unknown_fusion_head_variant(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        "fusion_head_variant": "transformer_x",
    }
    _write_config(bad_config, payload)

    with pytest.raises(ValueError, match="fusion_head_variant"):
        _ = load_config(bad_config)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("fusion_hidden_dim", 0, "fusion_hidden_dim"),
        ("fusion_hidden_dim", -1, "fusion_hidden_dim"),
        ("fusion_dropout", -0.1, "fusion_dropout"),
        ("fusion_dropout", 1.1, "fusion_dropout"),
        ("fusion_transformer_layers", 0, "fusion_transformer_layers"),
        ("fusion_transformer_heads", 0, "fusion_transformer_heads"),
        ("external_warmup_batch_size", 0, "external_warmup_batch_size"),
        ("external_warmup_num_workers", -1, "external_warmup_num_workers"),
        ("external_warmup_learning_rate", 0.0, "external_warmup_learning_rate"),
        ("external_warmup_max_samples", 0, "external_warmup_max_samples"),
        ("learning_rate", 0.0, "learning_rate"),
        ("weight_decay", -0.1, "weight_decay"),
        ("min_lr", -0.1, "min_lr"),
        ("freeze_backbone_epochs", -1, "freeze_backbone_epochs"),
        ("grad_accum_steps", 0, "grad_accum_steps"),
    ],
)
def test_load_config_rejects_invalid_fusion_head_numeric_values(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    bad_config = tmp_path / f"{field}.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        field: value,
    }
    _write_config(bad_config, payload)

    with pytest.raises(ValueError, match=match):
        _ = load_config(bad_config)


def test_load_config_rejects_transformer_head_with_non_divisible_hidden_dim(
    tmp_path: Path,
) -> None:
    bad_config = tmp_path / "bad_transformer.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        "fusion_head_variant": "transformer",
        "fusion_hidden_dim": 250,
        "fusion_transformer_heads": 4,
    }
    _write_config(bad_config, payload)

    with pytest.raises(ValueError, match="fusion_hidden_dim"):
        _ = load_config(bad_config)


@pytest.mark.parametrize("transform_profile", ["baseline", "normaug", "normonly"])
def test_load_config_accepts_supported_transform_profiles(
    tmp_path: Path,
    transform_profile: str,
) -> None:
    config_path = tmp_path / f"{transform_profile}.yaml"
    _write_config(
        config_path,
        {
            "experiment": {"name": "demo"},
            "paths": {
                "project_root": ".",
                "train_csv": "train.csv",
                "train_images": "train_img",
                "test_images": "test_img",
                "submission_template": "name_sid_submission.csv",
                "output_root": "outputs",
            },
            "runtime": {"seed": 7, "device": "cpu"},
            "train": {
                "folds": 5,
                "batch_size": 4,
                "image_size": 384,
                "epochs": 1,
                "num_workers": 0,
                "transform_profile": transform_profile,
            },
        },
    )

    config = load_config(config_path)

    assert config.train.transform_profile == transform_profile


def test_load_config_rejects_unknown_transform_profile(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    _write_config(
        bad_config,
        {
            "experiment": {"name": "demo"},
            "paths": {
                "project_root": ".",
                "train_csv": "train.csv",
                "train_images": "train_img",
                "test_images": "test_img",
                "submission_template": "name_sid_submission.csv",
                "output_root": "outputs",
            },
            "runtime": {"seed": 7, "device": "cpu"},
            "train": {
                "folds": 5,
                "batch_size": 4,
                "image_size": 384,
                "epochs": 1,
                "num_workers": 0,
                "transform_profile": "unknown",
            },
        },
    )

    with pytest.raises(ValueError, match="transform_profile"):
        _ = load_config(bad_config)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheduler", "cosine"),
        ("cache_mode", "preprocess"),
        ("external_sampler", "dataset_label_balanced"),
        ("fusion_eval_reference_run", "blend_best12_plus_baselinev2normaug_refined"),
    ],
)
def test_load_config_accepts_supported_training_controls(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    config_path = tmp_path / f"{field}.yaml"
    payload = _base_payload()
    payload["train"] = {
        **payload["train"],
        field: value,
    }
    _write_config(config_path, payload)

    config = load_config(config_path)

    assert getattr(config.train, field) == value


def test_load_config_rejects_missing_required_sections(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.yaml"
    _ = bad_config.write_text("experiment:\n  name: broken\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required config sections"):
        _ = load_config(bad_config)


def test_load_config_rejects_missing_runtime_seed_with_validation_error(
    tmp_path: Path,
) -> None:
    bad_config = tmp_path / "bad.yaml"
    _write_config(
        bad_config,
        {
            "experiment": {"name": "demo"},
            "paths": {
                "project_root": ".",
                "train_csv": "train.csv",
                "train_images": "train_img",
                "test_images": "test_img",
                "submission_template": "name_sid_submission.csv",
                "output_root": "outputs",
            },
            "runtime": {"device": "cpu"},
            "train": {
                "folds": 5,
                "batch_size": 4,
                "image_size": 384,
                "epochs": 1,
                "num_workers": 0,
            },
        },
    )

    with pytest.raises(ValueError, match="runtime"):
        _ = load_config(bad_config)


def test_load_config_accepts_optional_runtime_seed_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "runtime_seeds.yaml"
    payload = _base_payload()
    payload["runtime"] = {
        "seed": 7,
        "device": "cpu",
        "fold_seed": 11,
        "train_seed": 13,
        "warmup_seed": 17,
    }
    _write_config(config_path, payload)

    config = load_config(config_path)

    assert config.runtime.fold_seed == 11
    assert config.runtime.train_seed == 13
    assert config.runtime.warmup_seed == 17


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("folds", 1, "folds"),
        ("batch_size", 0, "batch_size"),
        ("image_size", 0, "image_size"),
        ("epochs", 0, "epochs"),
        ("num_workers", -1, "num_workers"),
        ("seed", True, "seed"),
    ],
)
def test_load_config_rejects_invalid_runtime_and_train_values(
    tmp_path: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    runtime_payload: dict[str, object] = {"seed": 7, "device": "cpu"}
    train_payload: dict[str, object] = {
        "folds": 5,
        "batch_size": 4,
        "image_size": 384,
        "epochs": 1,
        "num_workers": 0,
    }
    if field == "seed":
        runtime_payload[field] = value
    else:
        train_payload[field] = value

    bad_config = tmp_path / f"{field}.yaml"
    _write_config(
        bad_config,
        {
            "experiment": {"name": "demo"},
            "paths": {
                "project_root": ".",
                "train_csv": "train.csv",
                "train_images": "train_img",
                "test_images": "test_img",
                "submission_template": "name_sid_submission.csv",
                "output_root": "outputs",
            },
            "runtime": runtime_payload,
            "train": train_payload,
        },
    )

    with pytest.raises(ValueError, match=match):
        _ = load_config(bad_config)
