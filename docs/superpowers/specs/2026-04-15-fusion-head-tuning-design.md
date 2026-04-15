# Fusion Head Tuning Design

## Goal

Evaluate whether a stronger paired-view fusion architecture can outperform or complement the current mammography baseline without changing the backbone or the input pipeline.

The near-term objective is not to redesign the whole model stack. It is to isolate the fusion head as the only architectural variable, measure whether it improves cross-validation AUROC or ensemble diversity, and then decide whether it deserves deeper investment.

## Current State

- The strongest current single-model result is `outputs/runs/baseline/cv/metrics.json` with `mean_auc = 0.957409590180158`.
- The current paired-view head in `src/final_project/model/fusion.py` is a single linear layer over `concat([cc, mlo, abs(cc - mlo), cc * mlo])`.
- A previous transform-focused experiment (`baseline_v2_normaug`) underperformed the baseline under its tested recipe, so transforms are temporarily frozen here to isolate fusion-head effects. That result does not rule out future preprocessing or augmentation work.
- Several weaker single models have still shown useful out-of-fold complementarity, which means a new architecture should be judged by both standalone quality and ensemble value.

## Problem Statement

The current head is extremely cheap and stable, but it may underuse the paired feature interaction signal from `CC` and `MLO`. The project now needs a controlled way to test whether a slightly richer fusion head can capture more cross-view structure without destabilizing training on a small dataset.

The key constraint is experimental clarity. If backbone choice, transforms, and fusion design all change together, then a good or bad result is hard to interpret. This design therefore treats fusion architecture as the only structural change in the first round.

## Scope

### In Scope

- Modify `src/final_project/model/fusion.py` to support one or more configurable fusion-head variants.
- Add any minimal config plumbing required to select the fusion-head variant and its small set of hyperparameters.
- Run controlled CV experiments that keep the current backbone, image size, fold logic, and preprocessing contract fixed.
- Compare new fusion variants against the current baseline using both CV metrics and OOF blending behavior.
- Save each experiment under a distinct output directory without overwriting existing baseline artifacts.

### Out of Scope

- Backbone replacement or backbone-specific tuning.
- New transform pipelines, normalization policies, or aggressive augmentation changes.
- Label-definition changes.
- Large architectural additions such as attention blocks, bilinear pooling, or multi-stage cross-view interaction modules in the first pass.

## Design Principles

- Keep the first architecture change small enough that training remains cheap and debuggable.
- Keep the comparison fair by changing one main variable at a time.
- Freeze the Phase 1 training contract to the current baseline settings so any early movement is attributable to the head, not to silent training-side drift.
- Prefer configurations that can be expressed cleanly in the current repository layout.
- Optimize for final submission quality, which includes ensemble usefulness, not just single-run vanity metrics.

## Fixed Comparison Contract

Phase 1 must keep the following baseline contract unchanged:

- backbone: `efficientnet_b0`
- folds: `5`
- image size: `384`
- batch size: `16`
- epochs: `10`
- seed: `42`
- transform profile: `baseline`
- patient-safe fold assignment and the current breast-level label definition

No epoch, batch-size, transform, or backbone changes are allowed in Phase 1.

## Proposed Architecture Path

### Phase 1: Controlled Head Upgrade

Replace the fixed linear head with a configurable fusion head that still starts from the same fused feature vector:

`concat([cc, mlo, abs(cc - mlo), cc * mlo])`

The first new family should be a lightweight MLP head. A typical candidate shape is:

- `Linear(input_dim, 512)`
- activation (`GELU` or `ReLU`)
- optional `Dropout`
- `Linear(512, 1)`

This is the recommended first experiment because it increases nonlinearity and capacity without introducing hard-to-debug interaction mechanics.

### Phase 2: Head-Only Tuning

If a Phase 1 head is competitive, tune only the head-related or immediately adjacent training parameters, such as:

- `hidden_dim` in `{256, 512, 768}`
- `dropout`
- activation choice
- training seed
- only after a Phase 1 promotion, a small epoch adjustment such as `10 -> 12`

This phase exists to distinguish between two cases:

1. the architecture is not helpful, or
2. the architecture is helpful but was initially under-tuned.

Phase 1 is intentionally a fixed-budget comparison and may underestimate slower-converging heads. That is why a narrowly bounded follow-up step is allowed for near-baseline candidates before this path is rejected.

### Deferred Path

Only if the lightweight MLP family shows promise should the project consider modest stabilizing extensions, such as `LayerNorm` before the MLP or a light residual MLP variant. More expressive fusion variants, such as gated projection or wider residual multi-branch heads, remain deferred until there is evidence that the simple nonlinear upgrade is worth pursuing.

## Experiment Matrix

The first pass should stay compact.

Recommended initial variants:

1. `baseline_fusion_mlp_gelu_d0`
   - hidden dim `512`
   - `GELU`
   - no dropout
2. `baseline_fusion_mlp_gelu_d10`
   - hidden dim `512`
   - `GELU`
   - `dropout=0.10`
3. `baseline_fusion_mlp_relu_d10`
   - hidden dim `512`
   - `ReLU`
   - `dropout=0.10`

The backbone, image size, folds, seed, and preprocessing remain exactly aligned with `configs/baseline.yaml` in Phase 1.

## Evaluation Contract

Each new experiment must be judged on two axes.

### Single-Model Value

- Primary metric: CV `mean_auc`
- Secondary check: fold spread, defined as `max(fold_auc) - min(fold_auc)`, with `<= 0.03` treated as stable enough to trust

### Ensemble Value

- Use `outputs/runs/baseline/cv/oof_predictions.csv` as the fixed reference OOF artifact for all pairwise blend checks in this experiment family
- Use baseline OOF AUROC `0.940317308561905` as the fixed reference score
- Measure pairwise blends only between baseline OOF and the candidate head OOF
- Require `breast_id` to be unique in both OOF tables and require the exact same `breast_id` key set in both files before blending
- Join the two OOF tables on `breast_id` with a one-to-one merge and blend the `pred_score` column only
- Use the exact formula `blend_pred = (1 - w) * baseline_pred_score + w * candidate_pred_score`
- Use a fixed candidate-weight grid of `0.1, 0.2, ..., 0.9` and record the best pooled OOF AUROC from that grid
- Compute that pooled OOF AUROC against the breast-level `label` derived from the training manifest rule (`1` if any row for the breast has `pathology == 'M'`, else `0`)
- Record prediction-correlation diagnostics between baseline and candidate `pred_score`, at minimum Pearson correlation and Spearman rank correlation
- Record the best blend weight, the absolute OOF AUROC gain over the baseline OOF score, and whether the gain is positive in most folds
- Keep a model even if it loses single-model AUROC, provided that this fixed pairwise blend protocol clears the promotion threshold below

This matters because the project objective is the strongest final submission, not necessarily the prettiest standalone checkpoint.

## Success Criteria

Let `baseline_cv` be the current baseline CV mean AUC (`0.957409590180158`) and let `baseline_oof` be the current baseline pooled OOF AUROC (`0.940317308561905`).

At least one of the following must happen for the experiment family to be considered successful:

- A new fusion-head variant reaches `CV mean_auc >= baseline_cv - 0.003` while also keeping fold spread `<= 0.03`.
- A new fusion-head variant produces a fixed-protocol pairwise blend score of at least `baseline_oof + 0.0025` against the baseline OOF.
- The experiments produce clear evidence that the controlled nonlinear head family is exhausted, allowing the project to move on without ambiguity.

If the best Phase 1 single-model result lands within `+-0.002` of `baseline_cv`, rerun that exact head once with seed `123` before calling it a win or loss. Use the average of the seed `42` and seed `123` CV mean AUC values as the final single-model decision score for tie-band cases. If that two-seed average still lands within `+-0.002` of `baseline_cv` but the model has not clearly separated itself, allow exactly one Phase 1b retest at `12` epochs with all other settings unchanged. That tie-band exists to reduce the chance of mistaking seed noise or slower convergence for architecture signal.

## Risks

- Overfitting because the dataset is small and the head gains capacity faster than the data can support.
- Mistaking random seed variance for architectural improvement.
- Polluting the comparison by quietly changing transforms or inference behavior at the same time.
- Spending too much budget on exotic heads before a simple MLP variant is properly tuned.

## Required Outputs

- Fusion-head implementation changes in `src/final_project/model/fusion.py` and any minimal supporting config changes.
- New experiment config files for each controlled head variant.
- CV artifacts under unique directories in `outputs/runs/`.
- OOF comparison notes including correlation diagnostics and, if warranted, a backed-up blended prediction set using the best new head.

## Decision Rule After Phase 1

After the first controlled batch of runs:

- If the best new head fails both promotion thresholds (`CV mean_auc < baseline_cv - 0.003` and best pairwise blend `< baseline_oof + 0.0025`), stop this path.
- If the best new head exceeds the current baseline CV score by more than `0.002` (`CV mean_auc > baseline_cv + 0.002`), promote it immediately to final-model and ensemble candidacy.
- If the best new head lands inside the `+-0.002` tie-band around `baseline_cv`, run the seed-`123` confirmation and then apply the promotion thresholds using the two-seed average CV score plus the fixed pairwise blend result. If it still remains inside that tie-band without a decisive answer, use the single allowed `12`-epoch Phase 1b retest before stopping or promoting.
- In all other cases, if it clears either promotion threshold, continue with Phase 2 head-only tuning.
