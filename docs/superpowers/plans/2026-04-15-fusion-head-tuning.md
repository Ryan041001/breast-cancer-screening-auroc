# Fusion Head Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the paired-view fusion head in a controlled way, verify whether the new architecture improves standalone AUROC or ensemble value, and preserve a clean evidence trail for final submission selection.

**Architecture:** Keep the current paired `CC`/`MLO` backbone path, preprocessing, folds, and baseline training settings fixed while adding a configurable fusion-head family. Start with a lightweight MLP head that consumes the existing fused feature contract, then tune only the head and closely related training knobs before considering broader architectural changes.

**Tech Stack:** `uv`, Python 3.12, `torch`, `timm`, `pandas`, `numpy`, `scikit-learn`, `PyYAML`, `pytest`

---

## File Layout

- Modify: `src/final_project/model/fusion.py`
- Modify: `src/final_project/config.py`
- Modify: `src/final_project/engine/trainer.py`
- Modify: `src/final_project/engine/run_cv.py`
- Modify: `README.md`
- Create: `configs/baseline_fusion_mlp.yaml`
- Create: `configs/baseline_fusion_mlp_tune.yaml`
- Create: `tests/model/test_pair_model.py`
- Create: `tests/test_config.py`
- Create: `docs/superpowers/specs/2026-04-15-fusion-head-tuning-design.md`

## Task 1: Write the focused design note for the fusion-head experiment

**Files:**
- Create: `docs/superpowers/specs/2026-04-15-fusion-head-tuning-design.md`
- Reference: `docs/superpowers/specs/2026-04-14-mammography-auroc-design.md`

- [ ] **Step 1: Write a short design note**

Document the scope boundary: fusion head only first, no input-pipeline changes in this branch, and success measured by both CV AUROC and OOF ensemble complementarity.

- [ ] **Step 2: Save the design note under the spec directory**

Path: `docs/superpowers/specs/2026-04-15-fusion-head-tuning-design.md`

- [ ] **Step 3: Sanity-read the note for alignment**

Check that it explicitly says baseline preprocessing, folds, and backbone remain fixed for the first stage.

## Task 2: Add configurable fusion-head settings to the config contract

**Files:**
- Modify: `src/final_project/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
def test_load_config_supports_default_linear_fusion_head():
    config = load_config("configs/baseline.yaml")
    assert config.train.fusion_head_type == "linear"


def test_load_config_supports_mlp_fusion_head_options(tmp_path: Path):
    config_file = tmp_path / "fusion.yaml"
    config_file.write_text(
        """
experiment:
  name: demo
paths:
  project_root: ..
  train_csv: train.csv
  train_images: train_img
  test_images: test_img
  submission_template: name_sid_submission.csv
  output_root: outputs
runtime:
  seed: 42
  device: cuda
train:
  folds: 5
  batch_size: 16
  image_size: 384
  epochs: 10
  num_workers: 0
  fusion_head_type: mlp
  fusion_hidden_dim: 1024
  fusion_dropout: 0.2
  fusion_activation: gelu
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.train.fusion_head_type == "mlp"
    assert config.train.fusion_hidden_dim == 1024
    assert config.train.fusion_dropout == 0.2
    assert config.train.fusion_activation == "gelu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL because the fusion-head config fields are not defined yet.

- [ ] **Step 3: Implement the minimal config extension**

Add optional train-level fields for:
- `fusion_head_type` with default `linear`
- `fusion_hidden_dim` with a safe default for MLP mode
- `fusion_dropout` with a validated range
- `fusion_activation` constrained to known values

Preserve backward compatibility so `configs/baseline.yaml` still loads unchanged.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

## Task 3: Implement a configurable fusion head without breaking the baseline path

**Files:**
- Modify: `src/final_project/model/fusion.py`
- Modify: `tests/model/test_pair_model.py`

- [ ] **Step 1: Write the failing model tests**

```python
def test_pair_model_supports_linear_fusion_head():
    model = PairedBreastModel(backbone_name="resnet18", pretrained=False)
    cc = torch.randn(2, 3, 64, 64)
    mlo = torch.randn(2, 3, 64, 64)
    assert tuple(model(cc, mlo).shape) == (2,)


def test_pair_model_supports_mlp_fusion_head():
    model = PairedBreastModel(
        backbone_name="resnet18",
        pretrained=False,
        fusion_head_type="mlp",
        fusion_hidden_dim=128,
        fusion_dropout=0.1,
        fusion_activation="gelu",
    )
    cc = torch.randn(2, 3, 64, 64)
    mlo = torch.randn(2, 3, 64, 64)
    assert tuple(model(cc, mlo).shape) == (2,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_pair_model.py -q`
Expected: FAIL because `PairedBreastModel` does not accept fusion-head options yet.

- [ ] **Step 3: Implement the minimal architecture change**

Keep the fused feature contract unchanged:

```python
torch.cat([cc, mlo, torch.abs(cc - mlo), cc * mlo], dim=1)
```

Then add:
- a baseline `linear` head that reproduces current behavior
- an `mlp` head with `Linear -> Activation -> Dropout -> Linear`
- a small internal helper to build the requested head cleanly

Do not add attention, bilinear pooling, or extra branches in this task.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/model/test_pair_model.py -q`
Expected: PASS.

- [ ] **Step 5: Manually verify one forward pass for both head types**

Run: `uv run python main.py train --config configs/smoke.yaml --dry-run-model`

Run: `uv run python main.py train --config configs/baseline_fusion_mlp.yaml --dry-run-model`

Expected: both complete and report one logit per breast.

## Task 4: Thread the new head settings through training and CV artifact outputs

**Files:**
- Modify: `src/final_project/engine/trainer.py`
- Modify: `src/final_project/engine/run_cv.py`
- Modify: `README.md`

- [ ] **Step 1: Add the failing integration assertion**

If there is an existing trainer or run-CV test file that can cheaply assert config propagation, add a narrow assertion there; otherwise, use a manual smoke check in this task.

- [ ] **Step 2: Pass fusion-head settings into model construction**

Ensure the trainer and CV runner build `PairedBreastModel` with the selected head type and hyperparameters, and store enough config/checkpoint metadata to reconstruct the model for prediction.

- [ ] **Step 3: Update documentation for experiment knobs**

Document the new config fields in `README.md`, including the reason for keeping the first-stage architecture experiment isolated from transform changes.

- [ ] **Step 4: Run focused verification**

Run: `uv run pytest tests/test_config.py tests/model/test_pair_model.py tests/engine/test_trainer.py tests/engine/test_predict.py -q`
Expected: PASS.

## Task 5: Run the controlled fusion-head experiments

**Files:**
- Create: `configs/baseline_fusion_mlp.yaml`
- Create: `configs/baseline_fusion_mlp_tune.yaml`
- Output: `outputs/runs/baseline_fusion_mlp/`
- Output: `outputs/runs/baseline_fusion_mlp_tune/`

- [ ] **Step 1: Create the first controlled config**

`configs/baseline_fusion_mlp.yaml` should match `configs/baseline.yaml` except for:
- `experiment.name: baseline_fusion_mlp`
- `train.fusion_head_type: mlp`
- selected default head knobs such as hidden dim, dropout, activation

- [ ] **Step 2: Run the first full CV experiment**

Run: `uv run python main.py run-cv --config configs/baseline_fusion_mlp.yaml`
Expected: fold checkpoints, `oof_predictions.csv`, `test_predictions.csv`, and `metrics.json` are created.

- [ ] **Step 3: Create the follow-up tuning config**

`configs/baseline_fusion_mlp_tune.yaml` should keep the same backbone and preprocessing but adjust only a small number of training knobs around the best first-stage head.

- [ ] **Step 4: Run the second full CV experiment**

Run: `uv run python main.py run-cv --config configs/baseline_fusion_mlp_tune.yaml`
Expected: a second fully logged experiment exists for comparison.

- [ ] **Step 5: Compare against the current baseline using explicit evidence**

Read and record:
- `outputs/runs/baseline/cv/metrics.json`
- `outputs/runs/baseline_fusion_mlp/cv/metrics.json`
- `outputs/runs/baseline_fusion_mlp_tune/cv/metrics.json`

Summarize which run has the best mean AUROC and whether fold stability improved.

## Task 6: Measure ensemble complementarity before accepting or rejecting the new architecture

**Files:**
- Read: `outputs/runs/*/cv/oof_predictions.csv`
- Create: `outputs/runs/<best-ensemble-name>/blend.json`
- Create: `outputs/runs/<best-ensemble-name>/test_predictions.csv`

- [ ] **Step 1: Compute standalone OOF AUROC for all candidate runs**

Candidates must include at least:
- `baseline`
- `baseline_fusion_mlp`
- `baseline_fusion_mlp_tune`

Include other retained candidates only if they are already available and relevant.

- [ ] **Step 2: Search a small blend grid**

Run a simple weight sweep between the best new-architecture run and the current strongest retained baseline-like run.

- [ ] **Step 3: Export the best blended test predictions if the blend wins**

Save:
- `blend.json` with weights and OOF evidence
- `test_predictions.csv` for the same blend

- [ ] **Step 4: Make the keep/drop decision with two criteria**

Keep the new architecture if either:
- it wins on standalone CV mean AUROC, or
- it materially improves OOF blend AUROC with the baseline

Drop it only if it loses on both.

## Task 7: Final verification and handoff

**Files:**
- Read: `outputs/runs/<selected-experiment>/...`
- Read: `outputs/submissions/`

- [ ] **Step 1: Run regression tests after the architecture work**

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 2: Confirm the chosen experiment artifacts are complete**

Verify the selected run or blend directory includes metrics, OOF predictions, test predictions, and enough metadata to reproduce the choice.

- [ ] **Step 3: Generate or refresh the submission artifact if this run becomes the best**

Run the submission path for the chosen final predictions.

- [ ] **Step 4: Record the final result summary**

State:
- best standalone CV mean AUROC
- best OOF blend AUROC
- selected experiment or blend name
- whether the fusion-head architecture should continue to a broader second-generation design
