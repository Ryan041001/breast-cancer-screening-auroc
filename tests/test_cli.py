import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import final_project.cli as cli
from final_project.cli import main


EXPECTED_COMMANDS = [
    "build-manifest",
    "warmup-external",
    "train",
    "predict",
    "submit",
    "run-cv",
    "tune-iterate",
    "blend",
]


def _make_config(*, transform_profile: str = "baseline") -> SimpleNamespace:
    return SimpleNamespace(
        experiment=SimpleNamespace(name="smoke"),
        paths=SimpleNamespace(
            train_csv=Path("train.csv"),
            submission_template=Path("submission.csv"),
            test_images=Path("test_images"),
            output_root=Path("outputs"),
        ),
        train=SimpleNamespace(
            image_size=224,
            batch_size=2,
            num_workers=0,
            epochs=1,
            transform_profile=transform_profile,
            cache_mode="preprocess",
            learning_rate=1e-3,
            weight_decay=1e-2,
            scheduler="none",
            min_lr=0.0,
            freeze_backbone_epochs=0,
            grad_accum_steps=1,
            external_warmup_epochs=0,
        ),
        runtime=SimpleNamespace(seed=7, device="cpu"),
    )


def test_cli_lists_expected_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["--help"])

    assert exc_info.value.code == 0

    help_output = capsys.readouterr().out
    for command in EXPECTED_COMMANDS:
        assert command in help_output


def test_build_manifest_command_reports_manifest_counts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["build-manifest", "--config", str(Path("configs/smoke.yaml"))])

    assert exit_code == 0

    command_output = capsys.readouterr().out
    assert "train_breasts=650" in command_output
    assert "test_breasts=650" in command_output


def test_train_command_supports_dry_run_loader(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["train", "--config", str(Path("configs/smoke.yaml")), "--dry-run-loader"]
    )

    assert exit_code == 0

    command_output = capsys.readouterr().out
    assert "cc_shape" in command_output
    assert "mlo_shape" in command_output


def test_train_command_supports_dry_run_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        ["train", "--config", str(Path("configs/smoke.yaml")), "--dry-run-model"]
    )

    assert exit_code == 0

    command_output = capsys.readouterr().out
    assert "logit_shape" in command_output


def test_train_dry_run_loader_passes_transform_profile_to_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeDataset:
        def __init__(
            self,
            *,
            records: list[object],
            image_size: int,
            training: bool,
            transform_profile: str = "baseline",
            cache_mode: str = "preprocess",
        ) -> None:
            captured["transform_profile"] = transform_profile

        def __getitem__(self, index: int) -> dict[str, object]:
            tensor = SimpleNamespace(shape=(1, 2, 3))
            return {"cc_image": tensor, "mlo_image": tensor}

    monkeypatch.setattr(
        cli, "load_config", lambda _: _make_config(transform_profile="normaug")
    )
    monkeypatch.setattr(cli, "set_global_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_train_manifest", lambda _: [object()])
    monkeypatch.setattr(cli, "PairedBreastDataset", FakeDataset)

    exit_code = cli.main(
        ["train", "--config", str(Path("configs/smoke.yaml")), "--dry-run-loader"]
    )

    assert exit_code == 0
    assert captured["transform_profile"] == "normaug"


def test_train_full_run_passes_transform_profile_to_fit_full_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    class FakeDataset:
        def __init__(self, **_: object) -> None:
            pass

        def __getitem__(self, index: int) -> dict[str, object]:
            tensor = SimpleNamespace(shape=(1, 2, 3))
            return {"cc_image": tensor, "mlo_image": tensor}

    class FakeModel:
        def __init__(self, *, backbone_name: str, pretrained: bool, **_: object) -> None:
            self.backbone_name = backbone_name
            self.pretrained = pretrained

    def fake_fit_full_model(
        model: object,
        train_manifest: list[object],
        *,
        image_size: int,
        batch_size: int,
        num_workers: int,
        epochs: int,
        device: str,
        output_dir: Path,
        transform_profile: str = "baseline",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        scheduler_name: str = "none",
        min_lr: float = 0.0,
        freeze_backbone_epochs: int = 0,
        grad_accum_steps: int = 1,
        cache_mode: str = "preprocess",
    ) -> SimpleNamespace:
        captured["transform_profile"] = transform_profile
        return SimpleNamespace(checkpoints_dir=Path("outputs/checkpoints"))

    monkeypatch.setattr(
        cli, "load_config", lambda _: _make_config(transform_profile="normaug")
    )
    monkeypatch.setattr(cli, "set_global_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_train_manifest", lambda _: [object()])
    monkeypatch.setattr(cli, "PairedBreastDataset", FakeDataset)
    monkeypatch.setattr(
        cli, "build_output_paths", lambda _: SimpleNamespace(runs=Path("outputs/runs"))
    )
    monkeypatch.setattr(
        cli, "import_module", lambda _: SimpleNamespace(PairedBreastModel=FakeModel)
    )
    monkeypatch.setattr(cli, "fit_full_model", fake_fit_full_model)

    exit_code = cli.main(["train", "--config", str(Path("configs/smoke.yaml"))])

    assert exit_code == 0
    assert captured["transform_profile"] == "normaug"


def test_warmup_external_command_invokes_warmup_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    config = _make_config()
    config.train.external_warmup_epochs = 1

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli, "build_output_paths", lambda _: SimpleNamespace(runs=Path("outputs/runs"))
    )

    def fake_prepare(
        config_obj: object,
        *,
        backbone_name: str,
        output_dir: Path,
        image_size: int,
        transform_profile: str,
    ) -> Path:
        captured["backbone_name"] = backbone_name
        captured["output_dir"] = output_dir
        captured["image_size"] = image_size
        captured["transform_profile"] = transform_profile
        return output_dir / "checkpoints" / "best.pt"

    monkeypatch.setattr(cli, "maybe_prepare_external_warmup", fake_prepare)

    exit_code = cli.main(
        ["warmup-external", "--config", str(Path("configs/smoke.yaml"))]
    )

    assert exit_code == 0
    assert captured["backbone_name"] == cli.DEFAULT_BACKBONE_NAME
    assert captured["image_size"] == 224
    assert captured["transform_profile"] == "baseline"


def test_train_full_run_loads_external_warmup_checkpoint_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: dict[str, object] = {}

    config = _make_config()
    config.train.external_warmup_epochs = 1

    class FakeDataset:
        def __init__(self, **_: object) -> None:
            pass

        def __getitem__(self, index: int) -> dict[str, object]:
            tensor = SimpleNamespace(shape=(1, 2, 3))
            return {"cc_image": tensor, "mlo_image": tensor}

    class FakeModel:
        def __init__(self, *, backbone_name: str, pretrained: bool, **_: object) -> None:
            self.backbone_name = backbone_name
            self.pretrained = pretrained

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "set_global_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_train_manifest", lambda _: [object()])
    monkeypatch.setattr(cli, "PairedBreastDataset", FakeDataset)
    monkeypatch.setattr(
        cli, "build_output_paths", lambda _: SimpleNamespace(runs=Path("outputs/runs"))
    )
    monkeypatch.setattr(
        cli, "import_module", lambda _: SimpleNamespace(PairedBreastModel=FakeModel)
    )
    monkeypatch.setattr(
        cli,
        "maybe_prepare_external_warmup",
        lambda *args, **kwargs: Path("outputs/runs/smoke/external_warmup/checkpoints/best.pt"),
    )
    monkeypatch.setattr(
        cli,
        "load_backbone_from_warmup",
        lambda model, checkpoint_path: loaded.update(
            {"model": model, "checkpoint_path": checkpoint_path}
        ),
    )
    monkeypatch.setattr(
        cli,
        "fit_full_model",
        lambda *args, **kwargs: SimpleNamespace(checkpoints_dir=Path("outputs/checkpoints")),
    )

    exit_code = cli.main(["train", "--config", str(Path("configs/smoke.yaml"))])

    assert exit_code == 0
    assert str(loaded["checkpoint_path"]).endswith("best.pt")


def test_tune_iterate_runs_batch_tuning_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_config()
    config.paths.output_root = Path("outputs")

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(
        cli,
        "run_tuning_iteration",
        lambda config_paths, *, output_root, report_name, baseline_run, skip_existing: SimpleNamespace(
            report_dir=Path("outputs/research/demo"),
            leaderboard_path=Path("outputs/research/demo/leaderboard.json"),
            best_blend_dir=Path("outputs/runs/tune_iter_demo_best_blend"),
            summaries=[SimpleNamespace(experiment_name="baseline")],
        ),
    )

    exit_code = cli.main(
        [
            "tune-iterate",
            "--configs",
            "configs/baseline.yaml",
            "configs/baseline_fusion_mlp_gelu_d0.yaml",
            "--report-name",
            "demo",
        ]
    )

    assert exit_code == 0
    command_output = capsys.readouterr().out
    assert "leaderboard=" in command_output
    assert "best_blend_dir=" in command_output


def test_blend_runs_from_json_spec(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def fake_run_blend_from_spec(
        spec_path: Path,
        *,
        runs_root: Path,
    ) -> SimpleNamespace:
        captured["spec_path"] = spec_path
        captured["runs_root"] = runs_root
        return SimpleNamespace(
            oof_auc=0.9755,
            members=("a", "b"),
            output_dir=Path("outputs/runs/blend_demo"),
        )

    monkeypatch.setattr(cli, "run_blend_from_spec", fake_run_blend_from_spec)

    exit_code = cli.main(
        ["blend", "--spec", str(Path("outputs/research/demo.json"))]
    )

    assert exit_code == 0
    assert captured["spec_path"] == Path("outputs/research/demo.json")
    assert captured["runs_root"] == Path("outputs") / "runs"
    command_output = capsys.readouterr().out
    assert "oof_auc=0.975500" in command_output
    assert "members=2" in command_output


@pytest.mark.parametrize(
    ("cv_checkpoints", "expected_loader_calls"),
    [([], 1), ([Path("outputs/runs/smoke/cv/fold_0/checkpoints/best.pt")], 2)],
)
def test_predict_passes_transform_profile_to_prediction_loader(
    monkeypatch: pytest.MonkeyPatch,
    cv_checkpoints: list[Path],
    expected_loader_calls: int,
) -> None:
    captured_profiles: list[str] = []
    predict_call_count = 0

    def fake_build_prediction_loader(
        records: list[object],
        image_size: int,
        batch_size: int,
        num_workers: int,
        *,
        transform_profile: str = "baseline",
        cache_mode: str = "preprocess",
    ) -> object:
        captured_profiles.append(transform_profile)
        return object()

    def fake_predict_probabilities(
        model: object,
        prediction_loader: object,
        *,
        device: str,
    ) -> dict[str, float]:
        nonlocal predict_call_count
        predict_call_count += 1
        return {"breast-1": float(predict_call_count)}

    monkeypatch.setattr(
        cli, "load_config", lambda _: _make_config(transform_profile="normaug")
    )
    monkeypatch.setattr(
        cli, "build_output_paths", lambda _: SimpleNamespace(runs=Path("outputs/runs"))
    )
    monkeypatch.setattr(cli, "build_test_manifest", lambda *_: [object()])
    monkeypatch.setattr(cli, "glob", lambda _: [str(path) for path in cv_checkpoints])
    monkeypatch.setattr(cli, "build_prediction_loader", fake_build_prediction_loader)
    monkeypatch.setattr(
        cli, "load_model_from_checkpoint", lambda *args, **kwargs: object()
    )
    monkeypatch.setattr(cli, "predict_probabilities", fake_predict_probabilities)
    monkeypatch.setattr(cli, "write_prediction_table", lambda *args, **kwargs: None)

    exit_code = cli.main(["predict", "--config", str(Path("configs/smoke.yaml"))])

    assert exit_code == 0
    assert captured_profiles == ["normaug"] * expected_loader_calls


def test_cli_missing_config_path_exits_cleanly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _ = main(["build-manifest", "--config", "configs/does-not-exist.yaml"])

    assert exc_info.value.code == 2

    error_output = capsys.readouterr().err
    assert "does-not-exist.yaml" in error_output
    assert "error" in error_output.lower()


def test_python_module_entrypoint_lists_expected_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "final_project", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    for command in EXPECTED_COMMANDS:
        assert command in completed.stdout


def test_train_warns_when_linear_head_ignores_fusion_fields(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = _make_config()
    config.train.fusion_head_variant = "linear"
    config.train.fusion_hidden_dim = 256

    class FakeDataset:
        def __init__(self, **_: object) -> None:
            pass

        def __getitem__(self, index: int) -> dict[str, object]:
            tensor = SimpleNamespace(shape=(1, 2, 3))
            return {"cc_image": tensor, "mlo_image": tensor}

    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "set_global_seed", lambda _: None)
    monkeypatch.setattr(cli, "build_train_manifest", lambda _: [object()])
    monkeypatch.setattr(cli, "PairedBreastDataset", FakeDataset)

    exit_code = cli.main(
        ["train", "--config", str(Path("configs/smoke.yaml")), "--dry-run-loader"]
    )

    assert exit_code == 0
    assert "fusion_head_variant=linear ignores fusion_hidden_dim=256" in capsys.readouterr().out
