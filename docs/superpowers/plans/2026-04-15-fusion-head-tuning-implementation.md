# Fusion Head Tuning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configurable Phase 1 fusion-head experiments, strict OOF/blend evaluation, and the experiment configs needed to decide whether lightweight nonlinear fusion is worth deeper tuning.

**Architecture:** Keep the existing paired-view backbone and input pipeline unchanged, extend the model/config/checkpoint path to support a baseline head plus lightweight MLP fusion-head variants, and add deterministic evaluation/reporting that compares each candidate against the existing baseline with strict OOF alignment. Keep Phase 1 small, then use the resulting reports to decide whether to stop, run the tie-band retest, or continue into Phase 2.

**Tech Stack:** `uv`, Python 3.12, `torch`, `timm`, `scikit-learn`, `PyYAML`, `pytest`

**Constraint:** Do not commit unless the user explicitly asks for a commit.

---

## File Layout

- Modify: `configs/smoke.yaml`
- Create: `configs/smoke_fusion_mlp_gelu_d0.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d0.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d0_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d0_epochs12.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10_epochs12.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10_epochs12.yaml`
- Modify: `src/final_project/config.py`
- Modify: `src/final_project/model/fusion.py`
- Modify: `src/final_project/cli.py`
- Modify: `src/final_project/engine/run_cv.py`
- Modify: `src/final_project/engine/trainer.py`
- Modify: `src/final_project/engine/predict.py`
- Modify: `src/final_project/engine/submission.py`
- Modify: `src/final_project/model/metrics.py`
- Create: `src/final_project/engine/fusion_eval.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/model/test_pair_model.py`
- Create: `tests/model/test_metrics.py`
- Modify: `tests/engine/test_predict.py`
- Modify: `tests/engine/test_run_cv.py`
- Modify: `tests/engine/test_trainer.py`
- Modify: `tests/engine/test_submission.py`

## Task 1: Add a backward-compatible fusion-head config contract

**Files:**
- Modify: `src/final_project/config.py`
- Modify: `configs/smoke.yaml`
- Create: `configs/smoke_fusion_mlp_gelu_d0.yaml`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write the failing config tests**

Add tests in `tests/test_config.py` that pin all of the following:

```python
def test_load_config_defaults_to_linear_fusion_head():
    ...

def test_load_config_accepts_supported_fusion_head_variants():
    ...

def test_load_config_rejects_unknown_fusion_head_variant():
    ...

def test_load_config_rejects_invalid_fusion_head_numbers():
    ...
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run: `uv run pytest tests/test_config.py -q`
Expected: FAIL because the config schema does not yet accept or validate fusion-head settings.

- [ ] **Step 3: Extend the config schema with minimal Phase 1 head fields**

Implement the smallest backward-compatible config surface in `src/final_project/config.py`.

Use baseline-safe defaults so every existing config still loads unchanged:

- `fusion_head_variant: linear`
- `fusion_hidden_dim: 512`
- `fusion_dropout: 0.0`
- `fusion_activation: gelu`
- `fusion_layer_norm: false`
- `fusion_residual: false`

Keep these values under `TrainConfig` to minimize plumbing for Phase 1.

- [ ] **Step 4: Add smoke configs for baseline and one nonlinear head**

Update `configs/smoke.yaml` so it is explicit about baseline fusion defaults.

Create `configs/smoke_fusion_mlp_gelu_d0.yaml` as the first end-to-end nonlinear smoke config:

- same as `configs/smoke.yaml`
- `experiment.name: smoke_fusion_mlp_gelu_d0`
- `fusion_head_variant: mlp`
- `fusion_hidden_dim: 512`
- `fusion_dropout: 0.0`
- `fusion_activation: gelu`

- [ ] **Step 5: Re-run the config tests to verify they pass**

Run: `uv run pytest tests/test_config.py -q`
Expected: PASS.

## Task 2: Implement configurable fusion heads and preserve checkpoint reloads

**Files:**
- Modify: `src/final_project/model/fusion.py`
- Modify: `src/final_project/cli.py`
- Modify: `src/final_project/engine/run_cv.py`
- Modify: `src/final_project/engine/trainer.py`
- Modify: `src/final_project/engine/predict.py`
- Modify: `tests/model/test_pair_model.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/engine/test_predict.py`
- Modify: `tests/engine/test_run_cv.py`
- Modify: `tests/engine/test_trainer.py`

- [ ] **Step 1: Write the failing fusion-head and reload tests**

Add focused tests for:

```python
def test_pair_model_supports_linear_and_mlp_fusion_heads():
    ...

def test_train_path_forwards_fusion_head_config():
    ...

def test_run_cv_forwards_fusion_head_config():
    ...

def test_checkpoint_saves_fusion_head_metadata():
    ...

def test_load_model_from_checkpoint_rebuilds_saved_fusion_head():
    ...

def test_load_model_from_old_checkpoint_falls_back_to_linear_head():
    ...
```

- [ ] **Step 2: Run the focused tests to verify they fail**

Run: `uv run pytest tests/model/test_pair_model.py tests/test_cli.py tests/engine/test_predict.py tests/engine/test_run_cv.py tests/engine/test_trainer.py -q`
Expected: FAIL because the model constructor, call sites, and checkpoint payload do not yet understand configurable heads.

- [ ] **Step 3: Refactor `fusion.py` into an explicit head contract**

Implement the following in `src/final_project/model/fusion.py`:

- keep the current concat contract: `cc`, `mlo`, `abs(cc - mlo)`, `cc * mlo`
- retain the current linear head as the `linear` baseline variant
- add an `mlp` variant with:
  - optional `LayerNorm`
  - `Linear(input_dim, hidden_dim)`
  - activation (`GELU` or `ReLU`)
  - optional `Dropout`
  - `Linear(hidden_dim, 1)`
- if `fusion_residual` is enabled for a later experiment, implement it only as a lightweight residual on the fused feature path; do not introduce new interaction families

Make `PairedBreastModel` store a normalized, serializable `fusion_head_config` attribute for checkpointing.

- [ ] **Step 4: Thread the config through all model construction sites**

Update `src/final_project/cli.py` and `src/final_project/engine/run_cv.py` so the selected head config is passed everywhere `PairedBreastModel(...)` is built, including dry-run model paths.

- [ ] **Step 5: Make checkpoints self-describing without breaking old ones**

Update `src/final_project/engine/trainer.py` to save `fusion_head_config` in `best.pt`.

Update `src/final_project/engine/predict.py` so `load_model_from_checkpoint()`:

- prefers checkpoint-saved `fusion_head_config`
- falls back to the baseline linear head when older checkpoints do not have that metadata

- [ ] **Step 6: Re-run the focused tests to verify they pass**

Run: `uv run pytest tests/model/test_pair_model.py tests/test_cli.py tests/engine/test_predict.py tests/engine/test_run_cv.py tests/engine/test_trainer.py -q`
Expected: PASS.

- [ ] **Step 7: Manually smoke the nonlinear head constructor path**

Run: `uv run python main.py train --config configs/smoke_fusion_mlp_gelu_d0.yaml --dry-run-model`
Expected: output includes `logit_shape`, and the run completes without constructor or checkpoint-related errors.

## Task 3: Add strict OOF evaluation and report serialization

**Files:**
- Modify: `src/final_project/model/metrics.py`
- Create: `src/final_project/engine/fusion_eval.py`
- Modify: `src/final_project/engine/submission.py`
- Modify: `src/final_project/engine/run_cv.py`
- Create: `tests/model/test_metrics.py`
- Modify: `tests/engine/test_submission.py`
- Modify: `tests/engine/test_run_cv.py`

- [ ] **Step 1: Write the failing evaluation tests**

Add tests that pin the Phase 1 evaluation contract:

```python
def test_fold_spread_uses_max_minus_min_fold_auc():
    ...

def test_strict_prediction_table_reader_rejects_duplicate_ids():
    ...

def test_pairwise_blend_search_uses_fixed_weight_grid():
    ...

def test_pairwise_blend_search_requires_exact_oof_key_match():
    ...

def test_prediction_correlation_reports_pearson_and_spearman():
    ...

def test_run_cv_writes_fusion_eval_report_when_baseline_reference_exists():
    ...
```

- [ ] **Step 2: Run the evaluation tests to verify they fail**

Run: `uv run pytest tests/model/test_metrics.py tests/engine/test_submission.py tests/engine/test_run_cv.py -q`
Expected: FAIL because no strict reader, blend search, or fusion-eval reporting exists yet.

- [ ] **Step 3: Expand the pure metric helpers**

In `src/final_project/model/metrics.py`, add deterministic helpers for:

- pooled OOF AUROC
- fold spread
- fixed-grid blend scoring
- prediction correlation metrics

Keep these helpers pure; no filesystem access here.

- [ ] **Step 4: Add strict prediction-table loading and alignment helpers**

In `src/final_project/engine/submission.py`, either harden `read_prediction_table()` or add a strict sibling reader that:

- rejects duplicate `breast_id`
- preserves exact one-to-one mapping expectations
- can support exact key-set checks before blending

- [ ] **Step 5: Add a focused fusion evaluation module**

Create `src/final_project/engine/fusion_eval.py` to orchestrate the spec-defined evaluation flow:

- consume baseline OOF and candidate OOF
- consume a `labels_by_breast_id` mapping derived from `build_train_manifest()` / `BreastManifestRecord.label`
- verify exact `breast_id` key-set equality
- join by `breast_id`
- compute:
  - candidate pooled OOF AUROC
  - fold spread
  - best pairwise blend weight over `0.1..0.9`
  - best pairwise blend AUROC
  - absolute blend gain over baseline OOF
  - Pearson and Spearman correlation
  - fold-wise positive-gain summary if fold-level candidate data is available
- expose a reusable weighted-table helper for blending test prediction tables from two matching `breast_id -> pred_score` mappings
- emit a serializable report structure

- [ ] **Step 6: Wire the report into `run_cv.py`**

Update `src/final_project/engine/run_cv.py` so Phase 1 candidate runs can optionally load the fixed baseline OOF reference, derive `labels_by_breast_id` from `train_manifest`, and write a structured evaluation artifact such as `cv/fusion_eval.json` without changing the existing `oof_predictions.csv` or `test_predictions.csv` schema.

Extend `metrics.json` only if it improves readability without duplicating the whole report.

- [ ] **Step 7: Re-run the evaluation tests to verify they pass**

Run: `uv run pytest tests/model/test_metrics.py tests/engine/test_submission.py tests/engine/test_run_cv.py -q`
Expected: PASS.

- [ ] **Step 8: Manually smoke the strict evaluation path**

Run: `uv run python main.py run-cv --config configs/smoke_fusion_mlp_gelu_d0.yaml`
Expected:

- `outputs/runs/smoke_fusion_mlp_gelu_d0/cv/oof_predictions.csv` exists
- `outputs/runs/smoke_fusion_mlp_gelu_d0/cv/metrics.json` exists
- `outputs/runs/smoke_fusion_mlp_gelu_d0/cv/fusion_eval.json` exists if the baseline OOF reference is available

## Task 4: Create the Phase 1 experiment configs

**Files:**
- Create: `configs/baseline_fusion_mlp_gelu_d0.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d0_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10_seed123.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d0_epochs12.yaml`
- Create: `configs/baseline_fusion_mlp_gelu_d10_epochs12.yaml`
- Create: `configs/baseline_fusion_mlp_relu_d10_epochs12.yaml`

- [ ] **Step 1: Add the three approved Phase 1 configs**

Each config should clone the baseline contract:

- `folds: 5`
- `batch_size: 16`
- `image_size: 384`
- `epochs: 10`
- `seed: 42`
- `transform_profile: baseline`
- backbone remains the default `efficientnet_b0`

Only the fusion-head settings and `experiment.name` should differ.

- [ ] **Step 2: Add the exact tie-band rerun configs up front**

Create deterministic rerun configs now so the tie-band path is executable without ad hoc file editing later:

- `configs/baseline_fusion_mlp_gelu_d0_seed123.yaml`
- `configs/baseline_fusion_mlp_gelu_d10_seed123.yaml`
- `configs/baseline_fusion_mlp_relu_d10_seed123.yaml`

These should match their Phase 1 parents exactly except for:

- `runtime.seed: 123`
- `experiment.name` suffix `_seed123`

Create the single allowed Phase 1b configs now as well:

- `configs/baseline_fusion_mlp_gelu_d0_epochs12.yaml`
- `configs/baseline_fusion_mlp_gelu_d10_epochs12.yaml`
- `configs/baseline_fusion_mlp_relu_d10_epochs12.yaml`

These should match their Phase 1 parents exactly except for:

- `train.epochs: 12`
- `experiment.name` suffix `_epochs12`

- [ ] **Step 3: Sanity-check all configs load**

Run:

```bash
uv run python -c "from final_project.config import load_config; [load_config(path) for path in ['configs/baseline_fusion_mlp_gelu_d0.yaml', 'configs/baseline_fusion_mlp_gelu_d10.yaml', 'configs/baseline_fusion_mlp_relu_d10.yaml', 'configs/baseline_fusion_mlp_gelu_d0_seed123.yaml', 'configs/baseline_fusion_mlp_gelu_d10_seed123.yaml', 'configs/baseline_fusion_mlp_relu_d10_seed123.yaml', 'configs/baseline_fusion_mlp_gelu_d0_epochs12.yaml', 'configs/baseline_fusion_mlp_gelu_d10_epochs12.yaml', 'configs/baseline_fusion_mlp_relu_d10_epochs12.yaml']]"
```

Expected: no exception.

## Task 5: Run the fixed-budget Phase 1 experiments and record the decision

**Files:**
- Read/Write: `outputs/runs/baseline_fusion_mlp_gelu_d0/cv/*`
- Read/Write: `outputs/runs/baseline_fusion_mlp_gelu_d10/cv/*`
- Read/Write: `outputs/runs/baseline_fusion_mlp_relu_d10/cv/*`
- Read: `outputs/runs/baseline/cv/oof_predictions.csv`
- Read: `outputs/runs/baseline/cv/metrics.json`

- [ ] **Step 1: Run the three Phase 1 CV experiments**

Run in sequence or one at a time, depending on GPU availability:

```bash
uv run python main.py run-cv --config configs/baseline_fusion_mlp_gelu_d0.yaml
uv run python main.py run-cv --config configs/baseline_fusion_mlp_gelu_d10.yaml
uv run python main.py run-cv --config configs/baseline_fusion_mlp_relu_d10.yaml
```

- [ ] **Step 2: Read the generated reports**

For each run, inspect:

- `cv/metrics.json`
- `cv/fusion_eval.json`
- `cv/oof_predictions.csv`

Record the best candidate according to the revised spec:

- `CV mean_auc >= baseline_cv - 0.003` and `fold_spread <= 0.03`
- or best pairwise blend AUROC `>= baseline_oof + 0.0025`

- [ ] **Step 3: Apply the tie-band rule if needed**

If the best candidate lands within `+-0.002` of the current baseline CV score:

1. rerun the matching pre-created `*_seed123.yaml` config for that exact candidate
2. average the seed-42 and seed-123 CV mean AUC values
3. keep using the same fixed pairwise blend result from the seed-42 Phase 1 run unless you explicitly add a second-seed blend comparison helper

- [ ] **Step 4: Apply the single allowed Phase 1b retest if still needed**

Only if the two-seed average still remains inside the tie-band without a decisive answer:

1. rerun the matching pre-created `*_epochs12.yaml` config for that exact candidate
2. keep every other setting unchanged from the Phase 1 parent config
3. treat that run as the one allowed Phase 1b retest from the spec

- [ ] **Step 5: Stop or promote according to the spec**

Use the rules from `docs/superpowers/specs/2026-04-15-fusion-head-tuning-design.md` exactly.

If the best candidate fails both thresholds, stop the fusion-head path.

If it exceeds `baseline_cv + 0.002`, promote it immediately to final-model and ensemble candidacy.

If it remains below that immediate-promotion bar but clears either promotion threshold, move on to Phase 2 head-only tuning.

## Task 6: Run the smallest justified Phase 2 follow-up and back up the best ensemble artifact

**Files:**
- Create as needed: `configs/baseline_fusion_mlp_*_phase2*.yaml`
- Read/Write as needed: `outputs/runs/<phase2-experiment>/cv/*`
- Read: `outputs/runs/baseline/cv/test_predictions.csv`
- Read: `outputs/runs/<promoted-candidate>/cv/test_predictions.csv`
- Create as needed: `outputs/runs/<best-ensemble>/blend.json`
- Create as needed: `outputs/runs/<best-ensemble>/test_predictions.csv`

- [ ] **Step 1: Only if Phase 1 promoted a candidate, run a tiny Phase 2 sweep**

Allowed knobs:

- `hidden_dim` in `{256, 512, 768}`
- `dropout`
- activation choice
- `seed`
- optionally `LayerNorm` or lightweight residual only if the promoted MLP family still looks unstable

Do not reopen transforms or backbone changes here.

- [ ] **Step 2: Compare each promoted Phase 2 run against the same baseline artifacts**

Use the same fixed OOF/blend protocol and the same report fields so comparisons stay apples-to-apples.

- [ ] **Step 3: Back up the best justified blend**

If a new head meaningfully improves the final ensemble story, create a dedicated backup directory under `outputs/runs/<blend-name>/` containing at minimum:

- `blend.json`
- `test_predictions.csv`
- any summary notes needed to reproduce the weights from `fusion_eval.json`

Generate that blended `test_predictions.csv` by:

1. reading `outputs/runs/baseline/cv/test_predictions.csv`
2. reading `outputs/runs/<promoted-candidate>/cv/test_predictions.csv`
3. validating exact `breast_id` key-set equality with the same strict loader used for OOF tables
4. applying the saved best blend weight from `outputs/runs/<promoted-candidate>/cv/fusion_eval.json`
5. writing the blended table with `write_prediction_table()`

Write `blend.json` with at least:

- `baseline_test_predictions_path`
- `candidate_test_predictions_path`
- `candidate_fusion_eval_path`
- `blend_weight`
- `blend_formula`
- `generated_test_predictions_path`

- [ ] **Step 4: Run the full verification pass before claiming success**

Run:

```bash
uv run pytest -q
uv run python main.py train --config configs/smoke_fusion_mlp_gelu_d0.yaml --dry-run-model
uv run python main.py run-cv --config configs/smoke_fusion_mlp_gelu_d0.yaml
```

Expected:

- tests pass
- dry-run model path passes
- smoke CV path writes `metrics.json`, `oof_predictions.csv`, and `fusion_eval.json` as designed

## Exit Criteria

- Baseline-compatible fusion-head configuration support exists end-to-end.
- Checkpoints can rebuild the saved head config and still load older baseline checkpoints.
- The repo can compute strict, reproducible baseline-vs-candidate blend diagnostics.
- The three fixed-budget Phase 1 experiments are runnable from config files.
- The experiment decision is made using the revised spec, not ad hoc interpretation.
