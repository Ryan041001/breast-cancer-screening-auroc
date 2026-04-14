# Mammography AUROC Design

## Goal

Build a clear, reproducible deep learning pipeline that predicts one malignant probability per `breast_id` and generates a valid submission file for the course project.

The only hard deliverable is to fill `name_sid_submission.csv` with high-quality `pred_score` predictions. Everything else in the repository exists only to improve that final submission quality and make the process repeatable.

## Success Criteria

- Predict breast-level malignancy from paired `CC` and `MLO` mammography views.
- Validate with patient-safe folds so the same patient's left/right breasts never cross validation boundaries.
- Optimize for strong AUROC on a single GPU within a few hours per training run.
- Keep the repository structure clean and intentional.
- Record all external libraries, pretrained models, and public datasets used in `README.md`.

## Non-Goals

- Full lesion detection or segmentation as the primary task.
- Complicated research-only architectures that are hard to train or debug locally.
- Using train-only metadata such as `pathology`, `birads`, `lesion_type`, `difficult`, or `annotations` as direct model inputs at inference time.
- Building side features that do not materially help submission quality.

## Data Contract

- Training metadata comes from `train.csv`.
- Each `breast_id` should map to exactly two images: one `CC` and one `MLO` view.
- The training target is breast-level malignancy:
  - `label = 1` if any row for the breast has `pathology == 'M'`
  - `label = 0` otherwise
- Test-time output must preserve the row order from `name_sid_submission.csv` and fill `pred_score` with probabilities.

## Modeling Strategy

### Baseline

Use a shared pretrained image encoder on each view separately, then fuse the two view embeddings into one breast-level prediction.

Recommended default:

- Pretrained `timm` encoder
- Input resolution around `384` or `512`
- Shared weights for `CC` and `MLO`
- Fusion head on `concat([cc, mlo, abs(cc - mlo), cc * mlo])`
- BCE-with-logits with positive-class weighting
- Early stopping on validation AUROC

### Why this route

This is the strongest low-risk path for the assignment because it matches the breast-level task, uses both views explicitly, and keeps the model simple enough to tune reliably on a small dataset.

## Validation Strategy

- Collapse data to one row per `breast_id` before any split.
- Use deterministic grouped stratified folds at the patient level, then fan assignments back out to each `breast_id`.
- Track out-of-fold predictions and fold-level AUROC.
- Select configurations by mean OOF AUROC and fold stability, not by a single lucky run.

## Preprocessing Strategy

- Crop obvious black borders.
- Convert grayscale images into the model's expected channel layout.
- Canonicalize laterality so left/right anatomy is aligned consistently when useful.
- Use mild augmentation only; avoid aggressive transformations that distort medical signal.

## Repository Structure

Keep the root directory focused on entrypoints, data, and project metadata. Put implementation code under a single package, keep configs together, and keep outputs separate from source.

Proposed structure:

```text
Final_Project/
├─ configs/
├─ docs/
│  └─ superpowers/
│     ├─ plans/
│     └─ specs/
├─ outputs/
│  ├─ checkpoints/
│  ├─ oof/
│  ├─ submissions/
│  └─ runs/
├─ src/
│  └─ final_project/
│     ├─ data/
│     ├─ engine/
│     ├─ model/
│     └─ utils/
├─ tests/
└─ main.py
```

## External Resource Policy

If the implementation uses any of the following, they must be called out in `README.md`:

- Third-party libraries
- Pretrained models or weights
- Public datasets or external pretraining sources

Each entry should include what it is, where it came from, and how it is used in this project.

## Tuning Priorities

Tune in this order:

1. Leakage-safe validation and reproducibility
2. Resolution and crop strategy
3. Backbone choice
4. Loss weighting and batch configuration
5. TTA and fold ensembling
6. Optional annotation-aware enhancements

## High-Upside Enhancements

Only after the baseline is stable:

- Test-time augmentation
- Two strong backbones with light ensembling
- Annotation-aware cropping or ROI-guided sampling
- Optional external public data integration, if it can be documented and integrated cleanly

## Risks

- Data leakage from image-level or breast-only splitting instead of patient-safe grouping
- Overfitting due to small sample size
- Overcomplicated architecture that hurts tuning speed
- Missing attribution for external resources in `README.md`

## Deliverables

- Reproducible training pipeline
- Breast-level inference pipeline
- Submission generator
- Tuning harness with OOF metrics
- Clean `README.md` with usage and attribution
