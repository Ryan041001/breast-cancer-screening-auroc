from pathlib import Path

import pytest
import yaml

from final_project.config import AppConfig, load_config


def _write_config(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


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
    assert config.train.batch_size == 2
    assert config.train.image_size == 384


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
