from __future__ import annotations

import argparse
from importlib import import_module
from glob import glob
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import load_config
from .data.dataset import PairedBreastDataset
from .data.manifest import build_test_manifest, build_train_manifest
from .engine.predict import (
    DEFAULT_BACKBONE_NAME,
    build_prediction_loader,
    load_model_from_checkpoint,
    predict_probabilities,
)
from .engine.run_cv import run_cross_validation
from .engine.submission import (
    read_prediction_table,
    write_prediction_table,
    write_submission,
)
from .engine.trainer import fit_full_model
from .utils.paths import build_output_paths
from .utils.repro import set_global_seed


@dataclass(frozen=True, slots=True)
class CommandArgs:
    command: str
    config: Path
    dry_run_loader: bool
    dry_run_model: bool


COMMAND_DESCRIPTIONS = {
    "build-manifest": "Build the breast-level manifest.",
    "train": "Train the paired-view model.",
    "predict": "Generate predictions from saved checkpoints.",
    "submit": "Create a submission CSV from predictions.",
    "run-cv": "Run cross-validation end to end.",
}


def _run_stub(args: CommandArgs) -> int:
    config = load_config(args.config)
    print(
        f"{args.command}: loaded config '{config.experiment.name}' - not implemented yet.",
    )
    return 0


def _run_build_manifest(args: CommandArgs) -> int:
    config = load_config(args.config)
    train_manifest = build_train_manifest(config.paths.train_csv)
    test_manifest = build_test_manifest(
        config.paths.submission_template,
        config.paths.test_images,
    )
    print(
        "build-manifest:",
        f"train_breasts={len(train_manifest)}",
        f"test_breasts={len(test_manifest)}",
    )
    return 0


def _run_train(args: CommandArgs) -> int:
    config = load_config(args.config)
    set_global_seed(config.runtime.seed)
    train_manifest = build_train_manifest(config.paths.train_csv)
    dataset = PairedBreastDataset(
        records=train_manifest,
        image_size=config.train.image_size,
        training=False,
    )
    sample = dataset[0]

    if args.dry_run_loader:
        print(
            "train dry-run-loader:",
            f"cc_shape={tuple(sample['cc_image'].shape)}",
            f"mlo_shape={tuple(sample['mlo_image'].shape)}",
        )
        return 0

    if args.dry_run_model:
        model_module = import_module("final_project.model.fusion")
        model_class = getattr(model_module, "PairedBreastModel")
        model = model_class(backbone_name=DEFAULT_BACKBONE_NAME, pretrained=False)
        logits = model(
            sample["cc_image"].unsqueeze(0),
            sample["mlo_image"].unsqueeze(0),
        )
        print("train dry-run-model:", f"logit_shape={tuple(logits.shape)}")
        return 0

    output_paths = build_output_paths(config.paths.output_root)
    run_dir = output_paths.runs / config.experiment.name / "full_train"
    model_module = import_module("final_project.model.fusion")
    model_class = getattr(model_module, "PairedBreastModel")
    model = model_class(backbone_name=DEFAULT_BACKBONE_NAME, pretrained=True)
    trainer = fit_full_model(
        model,
        train_manifest,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        epochs=config.train.epochs,
        device=config.runtime.device,
        output_dir=run_dir,
    )
    checkpoint_path = trainer.checkpoints_dir / "best.pt"
    print("train:", f"checkpoint={checkpoint_path}")
    return 0


def _run_predict(args: CommandArgs) -> int:
    config = load_config(args.config)
    output_paths = build_output_paths(config.paths.output_root)
    experiment_dir = output_paths.runs / config.experiment.name
    test_manifest = build_test_manifest(
        config.paths.submission_template,
        config.paths.test_images,
    )

    cv_checkpoints = sorted(
        Path(path)
        for path in glob(
            str(experiment_dir / "cv" / "fold_*" / "checkpoints" / "best.pt")
        )
    )
    prediction_loader = build_prediction_loader(
        test_manifest,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
    )
    if cv_checkpoints:
        prediction_sets: list[dict[str, float]] = []
        for checkpoint_path in cv_checkpoints:
            model = load_model_from_checkpoint(
                checkpoint_path,
                device=config.runtime.device,
                backbone_name=DEFAULT_BACKBONE_NAME,
            )
            prediction_sets.append(
                predict_probabilities(
                    model, prediction_loader, device=config.runtime.device
                )
            )
            prediction_loader = build_prediction_loader(
                test_manifest,
                image_size=config.train.image_size,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
            )
        predictions = {
            breast_id: sum(pred[breast_id] for pred in prediction_sets)
            / len(prediction_sets)
            for breast_id in prediction_sets[0]
        }
    else:
        checkpoint_path = experiment_dir / "full_train" / "checkpoints" / "best.pt"
        model = load_model_from_checkpoint(
            checkpoint_path,
            device=config.runtime.device,
            backbone_name=DEFAULT_BACKBONE_NAME,
        )
        predictions = predict_probabilities(
            model,
            prediction_loader,
            device=config.runtime.device,
        )

    predictions_path = experiment_dir / "test_predictions.csv"
    write_prediction_table(predictions, predictions_path)
    print("predict:", f"predictions={predictions_path}", f"count={len(predictions)}")
    return 0


def _run_submit(args: CommandArgs) -> int:
    config = load_config(args.config)
    output_paths = build_output_paths(config.paths.output_root)
    experiment_dir = output_paths.runs / config.experiment.name
    predictions_path = experiment_dir / "test_predictions.csv"
    predictions = read_prediction_table(predictions_path)
    submission_path = (
        output_paths.submissions / f"{config.experiment.name}_submission.csv"
    )
    write_submission(config.paths.submission_template, predictions, submission_path)
    print("submit:", f"submission={submission_path}")
    return 0


def _run_cv(args: CommandArgs) -> int:
    config = load_config(args.config)
    artifacts = run_cross_validation(config)
    experiment_dir = (
        build_output_paths(config.paths.output_root).runs / config.experiment.name
    )
    write_prediction_table(
        artifacts.test_predictions, experiment_dir / "test_predictions.csv"
    )
    print(
        "run-cv:",
        f"mean_auc={artifacts.summary.mean_auc:.6f}",
        f"output_dir={artifacts.output_dir}",
    )
    return 0


def _run_command(args: CommandArgs) -> int:
    if args.command == "build-manifest":
        return _run_build_manifest(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "submit":
        return _run_submit(args)
    if args.command == "run-cv":
        return _run_cv(args)
    return _run_stub(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mammography submission pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, description in COMMAND_DESCRIPTIONS.items():
        subparser = subparsers.add_parser(
            command, help=description, description=description
        )
        _ = subparser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Path to a YAML configuration file.",
        )
        if command == "train":
            _ = subparser.add_argument(
                "--dry-run-loader",
                action="store_true",
                help="Load one paired sample and report tensor shapes.",
            )
            _ = subparser.add_argument(
                "--dry-run-model",
                action="store_true",
                help="Run one forward pass and report logit shape.",
            )
        subparser.set_defaults(command=command)

    return parser


def _parse_args(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None = None,
) -> CommandArgs:
    namespace = parser.parse_args(argv)
    command_obj: object = getattr(namespace, "command", None)
    config_obj: object = getattr(namespace, "config", None)

    if not isinstance(command_obj, str):
        parser.error("A command is required.")

    if not isinstance(config_obj, Path):
        parser.error("--config must be a valid path.")

    dry_run_loader = bool(getattr(namespace, "dry_run_loader", False))
    dry_run_model = bool(getattr(namespace, "dry_run_model", False))

    return CommandArgs(
        command=command_obj,
        config=config_obj,
        dry_run_loader=dry_run_loader,
        dry_run_model=dry_run_model,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _parse_args(parser, argv)
    try:
        return _run_command(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
