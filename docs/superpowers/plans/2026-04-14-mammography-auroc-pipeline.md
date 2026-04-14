# Mammography AUROC Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible paired-view mammography classifier that trains on local data, validates with patient-safe folds over breast-level samples, generates submission files, and supports disciplined AUROC-focused tuning.

**Primary Deliverable:** Fill `name_sid_submission.csv` with the strongest realistic `pred_score` values. All implementation choices should be judged by whether they improve the quality of the final submission.

**Architecture:** Collapse `train.csv` to one breast-level sample before splitting, load paired `CC` and `MLO` images, encode both views with a shared pretrained backbone, fuse the embeddings into one breast-level logit, and run deterministic fold-based evaluation with artifact logging.

**Tech Stack:** `uv`, Python 3.12, `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `scikit-learn`, `Pillow`, `PyYAML`, `tqdm`, `pytest`

---

## File Layout

- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `main.py`
- Create: `configs/smoke.yaml`
- Create: `configs/baseline.yaml`
- Create: `configs/tune.yaml`
- Create: `src/final_project/__init__.py`
- Create: `src/final_project/cli.py`
- Create: `src/final_project/config.py`
- Create: `src/final_project/data/manifest.py`
- Create: `src/final_project/data/splits.py`
- Create: `src/final_project/data/preprocess.py`
- Create: `src/final_project/data/transforms.py`
- Create: `src/final_project/data/dataset.py`
- Create: `src/final_project/model/backbone.py`
- Create: `src/final_project/model/fusion.py`
- Create: `src/final_project/model/losses.py`
- Create: `src/final_project/model/metrics.py`
- Create: `src/final_project/engine/trainer.py`
- Create: `src/final_project/engine/predict.py`
- Create: `src/final_project/engine/submission.py`
- Create: `src/final_project/engine/run_cv.py`
- Create: `src/final_project/utils/repro.py`
- Create: `src/final_project/utils/paths.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_config.py`
- Create: `tests/data/test_manifest.py`
- Create: `tests/data/test_splits.py`
- Create: `tests/data/test_dataset.py`
- Create: `tests/model/test_pair_model.py`
- Create: `tests/engine/test_trainer.py`
- Create: `tests/engine/test_submission.py`
- Create: `tests/engine/test_run_cv.py`

## Task 1: Scaffold the project contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `main.py`
- Create: `src/final_project/__init__.py`
- Create: `src/final_project/cli.py`
- Create: `src/final_project/config.py`
- Create: `configs/smoke.yaml`
- Create: `configs/baseline.yaml`
- Create: `configs/tune.yaml`
- Test: `tests/test_cli.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing CLI/config tests**

```python
def test_cli_lists_expected_commands():
    ...

def test_load_config_returns_expected_fields():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/test_config.py -q`
Expected: command/config imports fail because the package does not exist yet.

- [ ] **Step 3: Add the dependency contract and package skeleton**

Declare runtime dependencies in `pyproject.toml` and create the package, CLI, and config loader skeleton.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/test_config.py -q`
Expected: PASS.

- [ ] **Step 5: Verify the CLI manually**

Run: `uv run python main.py --help`
Expected: help output includes `build-manifest`, `train`, `predict`, `submit`, and `run-cv`.

## Task 2: Build the breast-level manifest and deterministic folds

**Files:**
- Create: `src/final_project/data/manifest.py`
- Create: `src/final_project/data/splits.py`
- Test: `tests/data/test_manifest.py`
- Test: `tests/data/test_splits.py`

- [ ] **Step 1: Write the failing manifest/split tests**

```python
def test_manifest_collapses_two_rows_into_one_breast_record():
    ...

def test_split_is_patient_safe_and_deterministic():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_manifest.py tests/data/test_splits.py -q`
Expected: FAIL because manifest and split builders do not exist yet.

- [ ] **Step 3: Implement manifest construction and fold assignment**

Build one row per `breast_id`, ensure exactly one `CC` and one `MLO`, derive `label`, and assign deterministic patient-safe folds from the breast-level table.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/data/test_manifest.py tests/data/test_splits.py -q`
Expected: PASS.

- [ ] **Step 5: Manually sanity-check the generated manifest**

Run: `uv run python main.py build-manifest --config configs/smoke.yaml`
Expected: artifacts show 650 train breasts and 650 test breasts with no missing view pair.

## Task 3: Implement preprocessing and the paired dataset

**Files:**
- Create: `src/final_project/data/preprocess.py`
- Create: `src/final_project/data/transforms.py`
- Create: `src/final_project/data/dataset.py`
- Test: `tests/data/test_dataset.py`

- [ ] **Step 1: Write the failing paired-dataset tests**

```python
def test_dataset_returns_cc_and_mlo_in_fixed_order():
    ...

def test_preprocess_canonicalizes_sample_shape():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/data/test_dataset.py -q`
Expected: FAIL because dataset and preprocessing code are missing.

- [ ] **Step 3: Implement crop, channel conversion, laterality handling, and paired dataset loading**

Keep augmentation light and deterministic in eval mode.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/data/test_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Manually inspect one loader batch**

Run: `uv run python main.py train --config configs/smoke.yaml --dry-run-loader`
Expected: one batch loads with two views per breast and consistent tensor shapes.

## Task 4: Implement the paired-view model, loss, and metrics

**Files:**
- Create: `src/final_project/model/backbone.py`
- Create: `src/final_project/model/fusion.py`
- Create: `src/final_project/model/losses.py`
- Create: `src/final_project/model/metrics.py`
- Test: `tests/model/test_pair_model.py`

- [ ] **Step 1: Write the failing model tests**

```python
def test_pair_model_returns_one_logit_per_breast():
    ...

def test_auc_metric_accepts_breast_level_predictions():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/model/test_pair_model.py -q`
Expected: FAIL because the model stack does not exist yet.

- [ ] **Step 3: Implement the shared encoder, fusion head, loss, and AUROC helper**

Start with a configurable `timm` backbone and a cheap fusion head.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/model/test_pair_model.py -q`
Expected: PASS.

- [ ] **Step 5: Manually verify one forward pass**

Run: `uv run python main.py train --config configs/smoke.yaml --dry-run-model`
Expected: one forward pass completes and reports one logit per breast.

## Task 5: Build the trainer, checkpointing, and prediction pipeline

**Files:**
- Create: `src/final_project/utils/repro.py`
- Create: `src/final_project/utils/paths.py`
- Create: `src/final_project/engine/trainer.py`
- Create: `src/final_project/engine/predict.py`
- Create: `src/final_project/engine/submission.py`
- Test: `tests/engine/test_trainer.py`
- Test: `tests/engine/test_submission.py`

- [ ] **Step 1: Write the failing trainer/submission tests**

```python
def test_trainer_saves_and_reloads_best_checkpoint():
    ...

def test_submission_writer_preserves_template_order():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_trainer.py tests/engine/test_submission.py -q`
Expected: FAIL because trainer and submission components are missing.

- [ ] **Step 3: Implement training loop, checkpointing, inference, and submission writing**

Save configs, metrics, and predictions under a clean `outputs/` hierarchy.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/engine/test_trainer.py tests/engine/test_submission.py -q`
Expected: PASS.

- [ ] **Step 5: Manually run a smoke training + prediction cycle**

Run: `uv run python main.py train --config configs/smoke.yaml`

Run: `uv run python main.py predict --config configs/smoke.yaml`

Run: `uv run python main.py submit --config configs/smoke.yaml`

Expected: smoke artifacts exist, a checkpoint is saved, and a submission-shaped CSV is produced.

## Task 6: Add CV execution, tuning harness, and final documentation

**Files:**
- Create: `src/final_project/engine/run_cv.py`
- Test: `tests/engine/test_run_cv.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing CV test**

```python
def test_run_cv_aggregates_fold_metrics_and_oof_predictions():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/engine/test_run_cv.py -q`
Expected: FAIL because the CV runner does not exist yet.

- [ ] **Step 3: Implement CV orchestration and tuning outputs**

Support smoke, dev, and full-CV modes using the same command surface.

- [ ] **Step 4: Re-run tests to verify they pass**

Run: `uv run pytest tests/engine/test_run_cv.py -q`
Expected: PASS.

- [ ] **Step 5: Update `README.md` with structure, commands, and attribution**

Document the directory layout, training commands, outputs, and every external library/model/data source used.

- [ ] **Step 6: Manually run the full smoke workflow**

Run: `uv run pytest -q`

Run: `uv run python main.py run-cv --config configs/smoke.yaml`

Expected: tests pass, smoke CV completes, OOF-like outputs are generated, and the README is sufficient for a fresh user.

## Tuning Order

1. Smoke pipeline correctness
2. Resolution: `384` vs `512`
3. Backbone: `efficientnetv2_s` vs `convnext_tiny`
4. Crop and laterality canonicalization
5. Loss weighting and batch size
6. TTA and fold averaging
7. Optional second strong config for ensembling

## Documentation Rule

`README.md` must include a section that lists:

- third-party libraries used
- pretrained models or weights used
- public datasets or extra data used
- a short note on what each one contributes to the final pipeline

## Guardrail

Do not spend time on work that does not help produce better `pred_score` values in `name_sid_submission.csv`. Repository cleanliness, tests, and documentation are required because they reduce mistakes and support tuning, not because they are independent goals.
