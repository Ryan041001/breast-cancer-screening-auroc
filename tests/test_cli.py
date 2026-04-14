import subprocess
import sys
from pathlib import Path

import pytest

from final_project.cli import main


EXPECTED_COMMANDS = [
    "build-manifest",
    "train",
    "predict",
    "submit",
    "run-cv",
]


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
