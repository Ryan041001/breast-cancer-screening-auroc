from __future__ import annotations

import json
from pathlib import Path

import yaml

from final_project.engine.tuning import run_tuning_iteration


def test_run_tuning_iteration_writes_leaderboard_and_best_blend(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "outputs"
    runs_root = output_root / "runs"
    research_root = output_root / "research"
    runs_root.mkdir(parents=True)
    research_root.mkdir(parents=True)

    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "\n".join(
            [
                "breast_id,patient_id,cc_mlo,pathology,image_path",
                "A_L,p1,CC,N,train_img/A_L_CC.jpg",
                "A_L,p1,MLO,N,train_img/A_L_MLO.jpg",
                "B_R,p2,CC,N,train_img/B_R_CC.jpg",
                "B_R,p2,MLO,N,train_img/B_R_MLO.jpg",
                "C_L,p3,CC,M,train_img/C_L_CC.jpg",
                "C_L,p3,MLO,M,train_img/C_L_MLO.jpg",
                "D_R,p4,CC,M,train_img/D_R_CC.jpg",
                "D_R,p4,MLO,M,train_img/D_R_MLO.jpg",
            ]
        ),
        encoding="utf-8",
    )
    for relative_path in [
        "train_img/A_L_CC.jpg",
        "train_img/A_L_MLO.jpg",
        "train_img/B_R_CC.jpg",
        "train_img/B_R_MLO.jpg",
        "train_img/C_L_CC.jpg",
        "train_img/C_L_MLO.jpg",
        "train_img/D_R_CC.jpg",
        "train_img/D_R_MLO.jpg",
    ]:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")

    baseline_dir = runs_root / "baseline" / "cv"
    candidate_dir = runs_root / "candidate" / "cv"
    baseline_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)

    (baseline_dir / "metrics.json").write_text(
        json.dumps({"mean_auc": 0.95, "fold_metrics": {"0": 0.95}}),
        encoding="utf-8",
    )
    (candidate_dir / "metrics.json").write_text(
        json.dumps({"mean_auc": 0.94, "fold_metrics": {"0": 0.94}}),
        encoding="utf-8",
    )
    (baseline_dir / "oof_predictions.csv").write_text(
        "breast_id,pred_score\nA_L,0.1\nB_R,0.8\nC_L,0.4\nD_R,0.9\n",
        encoding="utf-8",
    )
    (candidate_dir / "oof_predictions.csv").write_text(
        "breast_id,pred_score\nA_L,0.2\nB_R,0.3\nC_L,0.7\nD_R,0.6\n",
        encoding="utf-8",
    )
    (baseline_dir / "test_predictions.csv").write_text(
        "breast_id,pred_score\nT1,0.2\nT2,0.8\n",
        encoding="utf-8",
    )
    (candidate_dir / "test_predictions.csv").write_text(
        "breast_id,pred_score\nT1,0.4\nT2,0.6\n",
        encoding="utf-8",
    )
    baseline_config = tmp_path / "baseline.yaml"
    candidate_config = tmp_path / "candidate.yaml"
    for path, name in [
        (baseline_config, "baseline"),
        (candidate_config, "candidate"),
    ]:
        path.write_text(
            yaml.safe_dump(
                {
                    "experiment": {"name": name},
                    "paths": {
                        "project_root": str(tmp_path),
                        "train_csv": "train.csv",
                        "train_images": "train_img",
                        "test_images": "test_img",
                        "submission_template": "name_sid_submission.csv",
                        "output_root": "outputs",
                    },
                    "runtime": {"seed": 7, "device": "cpu"},
                    "train": {
                        "folds": 2,
                        "batch_size": 2,
                        "image_size": 32,
                        "epochs": 1,
                        "num_workers": 0,
                    },
                }
            ),
            encoding="utf-8",
        )

    artifacts = run_tuning_iteration(
        [baseline_config, candidate_config],
        output_root=output_root,
        report_name="demo",
        baseline_run="baseline",
        skip_existing=True,
    )

    assert artifacts.leaderboard_path.exists()
    leaderboard = json.loads(artifacts.leaderboard_path.read_text(encoding="utf-8"))
    assert leaderboard[0]["experiment_name"] == "baseline"
    assert artifacts.best_blend_dir is not None
    assert (candidate_dir / "fusion_eval.json").exists()
    assert (artifacts.best_blend_dir / "blend.json").exists()
    assert (artifacts.best_blend_dir / "test_predictions.csv").exists()
