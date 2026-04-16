# Current State

## Production Line

- best single:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
  - `mean_auc = 0.967337`
- best blend:
  `blend_best14_pruned_refined`
  - `oof_auc = 0.97799770028811`

## Research Line

- completed:
  - `best13` leave-one-out prune
  - `splitter-v2`
  - `fold_seed / train_seed / warmup_seed` decoupling
  - splitv2 minimal comparison
- verified splitv2 runs:
  - `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
  - `baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2`

## Important Rule

splitv2 is a new CV universe.
Do not compare splitv2 OOF metrics directly against the old best14 universe.
Only compare models inside the same splitv2 universe.
