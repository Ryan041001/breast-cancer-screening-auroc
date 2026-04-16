from __future__ import annotations

import argparse
import inspect
from importlib import import_module
from glob import glob
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

from .config import load_config
from .data.dataset import PairedBreastDataset
from .data.manifest import build_test_manifest, build_train_manifest
from .data.transforms import TransformProfile
from .model.fusion import FusionHeadConfig
from .engine.predict import (
    DEFAULT_BACKBONE_NAME,
    build_prediction_loader,
    load_model_from_checkpoint,
    predict_probabilities,
)
from .engine.run_cv import run_cross_validation
from .engine.external_warmup import (
    load_backbone_from_warmup,
    maybe_prepare_external_warmup,
)
from .engine.tuning import run_tuning_iteration
from .engine.blending import run_blend_from_spec
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
    config: Path | None
    configs: tuple[Path, ...]
    spec: Path | None
    dry_run_loader: bool
    dry_run_model: bool
    report_name: str
    baseline_run: str
    rerun_completed: bool


COMMAND_DESCRIPTIONS = {
    "build-manifest": "Build the breast-level manifest.",
    "warmup-external": "Warm up the backbone on MammoNet32k-style external data.",
    "train": "Train the paired-view model.",
    "predict": "Generate predictions from saved checkpoints.",
    "submit": "Create a submission CSV from predictions.",
    "run-cv": "Run cross-validation end to end.",
    "tune-iterate": "Run a batch of configs and summarize tuning results.",
    "blend": "Blend existing run outputs from a JSON spec.",
}


_ReturnT = TypeVar("_ReturnT")


def _get_transform_profile(config: object) -> TransformProfile:
    train_config = getattr(config, "train", None)
    transform_profile = getattr(train_config, "transform_profile", "baseline")
    if transform_profile == "normaug":
        return "normaug"
    if transform_profile == "normonly":
        return "normonly"
    return "baseline"


def _call_with_transform_profile(
    func: Callable[..., _ReturnT],
    transform_profile: TransformProfile,
    *args: object,
    **kwargs: object,
) -> _ReturnT:
    try:
        parameters = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        or parameter.name == "transform_profile"
        for parameter in parameters
    ):
        kwargs["transform_profile"] = transform_profile
    return func(*args, **kwargs)


def _run_stub(args: CommandArgs) -> int:
    if args.config is None:
        raise ValueError("This command requires --config")
    config = load_config(args.config)
    print(
        f"{args.command}: loaded config '{config.experiment.name}' - not implemented yet.",
    )
    return 0


def _run_build_manifest(args: CommandArgs) -> int:
    if args.config is None:
        raise ValueError("build-manifest requires --config")
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


def _run_warmup_external(args: CommandArgs) -> int:
    if args.config is None:
        raise ValueError("warmup-external requires --config")
    config = load_config(args.config)
    transform_profile = _get_transform_profile(config)
    output_paths = build_output_paths(config.paths.output_root)
    output_dir = output_paths.runs / config.experiment.name / "external_warmup"
    checkpoint_path = maybe_prepare_external_warmup(
        config,
        backbone_name=DEFAULT_BACKBONE_NAME,
        output_dir=output_dir,
        image_size=config.train.image_size,
        transform_profile=transform_profile,
    )
    if checkpoint_path is None:
        raise ValueError(
            "External warmup is disabled. Set train.external_warmup_epochs > 0."
        )
    print("warmup-external:", f"checkpoint={checkpoint_path}")
    return 0


def _run_train(args: CommandArgs) -> int:
    if args.config is None:
        raise ValueError("train requires --config")
    config = load_config(args.config)
    transform_profile = _get_transform_profile(config)
    fusion_head_config = FusionHeadConfig.from_train_config(config.train)
    _warn_if_linear_fusion_ignores_fields(config)
    set_global_seed(config.runtime.seed)
    train_manifest = build_train_manifest(config.paths.train_csv)
    dataset = PairedBreastDataset(
        records=train_manifest,
        image_size=config.train.image_size,
        training=False,
        transform_profile=transform_profile,
        cache_mode=config.train.cache_mode,
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
        model = model_class(
            backbone_name=DEFAULT_BACKBONE_NAME,
            pretrained=False,
            fusion_head_config=fusion_head_config,
        )
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
    model = model_class(
        backbone_name=DEFAULT_BACKBONE_NAME,
        pretrained=True,
        fusion_head_config=fusion_head_config,
    )
    warmup_checkpoint = maybe_prepare_external_warmup(
        config,
        backbone_name=DEFAULT_BACKBONE_NAME,
        output_dir=output_paths.runs / config.experiment.name / "external_warmup",
        image_size=config.train.image_size,
        transform_profile=transform_profile,
    )
    if warmup_checkpoint is not None:
        load_backbone_from_warmup(model, warmup_checkpoint)
    trainer = _call_with_transform_profile(
        fit_full_model,
        transform_profile,
        model,
        train_manifest,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        epochs=config.train.epochs,
        device=config.runtime.device,
        output_dir=run_dir,
        learning_rate=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
        scheduler_name=config.train.scheduler,
        min_lr=config.train.min_lr,
        freeze_backbone_epochs=config.train.freeze_backbone_epochs,
        grad_accum_steps=config.train.grad_accum_steps,
        cache_mode=config.train.cache_mode,
    )
    checkpoint_path = trainer.checkpoints_dir / "best.pt"
    print("train:", f"checkpoint={checkpoint_path}")
    return 0


def _run_predict(args: CommandArgs) -> int:
    if args.config is None:
        raise ValueError("predict requires --config")
    config = load_config(args.config)
    transform_profile = _get_transform_profile(config)
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
    prediction_loader = _call_with_transform_profile(
        build_prediction_loader,
        transform_profile,
        test_manifest,
        image_size=config.train.image_size,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        cache_mode=config.train.cache_mode,
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
            prediction_loader = _call_with_transform_profile(
                build_prediction_loader,
                transform_profile,
                test_manifest,
                image_size=config.train.image_size,
                batch_size=config.train.batch_size,
                num_workers=config.train.num_workers,
                cache_mode=config.train.cache_mode,
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
    if args.config is None:
        raise ValueError("submit requires --config")
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
    if args.config is None:
        raise ValueError("run-cv requires --config")
    config = load_config(args.config)
    _warn_if_linear_fusion_ignores_fields(config)
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


def _run_tune_iterate(args: CommandArgs) -> int:
    if not args.configs:
        raise ValueError("tune-iterate requires --configs")
    first_config = load_config(args.configs[0])
    artifacts = run_tuning_iteration(
        list(args.configs),
        output_root=first_config.paths.output_root,
        report_name=args.report_name,
        baseline_run=args.baseline_run,
        skip_existing=not args.rerun_completed,
    )
    print(
        "tune-iterate:",
        f"report_dir={artifacts.report_dir}",
        f"leaderboard={artifacts.leaderboard_path}",
        f"experiments={len(artifacts.summaries)}",
        f"best_blend_dir={artifacts.best_blend_dir}",
    )
    return 0


def _run_blend(args: CommandArgs) -> int:
    if args.spec is None:
        raise ValueError("blend requires --spec")
    artifacts = run_blend_from_spec(
        args.spec,
        runs_root=Path("outputs") / "runs",
    )
    print(
        "blend:",
        f"oof_auc={artifacts.oof_auc:.6f}",
        f"members={len(artifacts.members)}",
        f"output_dir={artifacts.output_dir}",
    )
    return 0


def _run_command(args: CommandArgs) -> int:
    if args.command == "build-manifest":
        return _run_build_manifest(args)
    if args.command == "warmup-external":
        return _run_warmup_external(args)
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "submit":
        return _run_submit(args)
    if args.command == "run-cv":
        return _run_cv(args)
    if args.command == "tune-iterate":
        return _run_tune_iterate(args)
    if args.command == "blend":
        return _run_blend(args)
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
        if command == "tune-iterate":
            _ = subparser.add_argument(
                "--configs",
                type=Path,
                nargs="+",
                required=True,
                help="One or more YAML configuration files to evaluate.",
            )
            _ = subparser.add_argument(
                "--report-name",
                default="latest",
                help="Directory name for the tuning report under outputs/research/.",
            )
            _ = subparser.add_argument(
                "--baseline-run",
                default="baseline",
                help="Existing baseline run used for fusion/blend comparison.",
            )
            _ = subparser.add_argument(
                "--rerun-completed",
                action="store_true",
                help="Re-run experiments even if outputs/runs/<experiment>/cv/metrics.json already exists.",
            )
        elif command == "blend":
            _ = subparser.add_argument(
                "--spec",
                type=Path,
                required=True,
                help="Path to a blend JSON spec with run_map and weights.",
            )
        else:
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
    configs_obj: object = getattr(namespace, "configs", ())
    spec_obj: object = getattr(namespace, "spec", None)

    if not isinstance(command_obj, str):
        parser.error("A command is required.")

    if command_obj not in {"tune-iterate", "blend"} and not isinstance(config_obj, Path):
        parser.error("--config must be a valid path.")
    if command_obj == "tune-iterate":
        if not isinstance(configs_obj, list) or not all(
            isinstance(item, Path) for item in configs_obj
        ):
            parser.error("--configs must be one or more valid paths.")
    if command_obj == "blend" and not isinstance(spec_obj, Path):
        parser.error("--spec must be a valid path.")

    dry_run_loader = bool(getattr(namespace, "dry_run_loader", False))
    dry_run_model = bool(getattr(namespace, "dry_run_model", False))
    report_name_obj: object = getattr(namespace, "report_name", "latest")
    baseline_run_obj: object = getattr(namespace, "baseline_run", "baseline")

    return CommandArgs(
        command=command_obj,
        config=config_obj if isinstance(config_obj, Path) else None,
        configs=tuple(configs_obj) if isinstance(configs_obj, list) else (),
        spec=spec_obj if isinstance(spec_obj, Path) else None,
        dry_run_loader=dry_run_loader,
        dry_run_model=dry_run_model,
        report_name=str(report_name_obj),
        baseline_run=str(baseline_run_obj),
        rerun_completed=bool(getattr(namespace, "rerun_completed", False)),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = _parse_args(parser, argv)
    try:
        return _run_command(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _warn_if_linear_fusion_ignores_fields(config: object) -> None:
    train_config = getattr(config, "train", None)
    if getattr(train_config, "fusion_head_variant", "linear") != "linear":
        return
    ignored_fields: list[str] = []
    if getattr(train_config, "fusion_hidden_dim", 512) != 512:
        ignored_fields.append(
            f"fusion_hidden_dim={getattr(train_config, 'fusion_hidden_dim')}"
        )
    if getattr(train_config, "fusion_dropout", 0.0) != 0.0:
        ignored_fields.append(
            f"fusion_dropout={getattr(train_config, 'fusion_dropout')}"
        )
    if getattr(train_config, "fusion_activation", "gelu") != "gelu":
        ignored_fields.append(
            f"fusion_activation={getattr(train_config, 'fusion_activation')}"
        )
    if getattr(train_config, "fusion_layer_norm", False):
        ignored_fields.append("fusion_layer_norm=true")
    if getattr(train_config, "fusion_residual", False):
        ignored_fields.append("fusion_residual=true")
    if not ignored_fields:
        return
    print(
        "warning: fusion_head_variant=linear ignores "
        + ", ".join(ignored_fields),
        flush=True,
    )
