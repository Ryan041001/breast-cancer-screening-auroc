# Current State

## Submission Line

- best single in current `splitv2` universe:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
  - `mean_auc = 0.9699796059675092`
- recommended submission blend:
  `blend_terminal_splitv2_plus_pairedlr8e4_refined`
  - `oof_auc = 0.9783852921796876`
- old-universe fallback:
  `blend_terminal_old_universe_refined`
  - `oof_auc = 0.9781914962338988`
- absolute mixed-pool ceiling:
  `blend_terminal_all_singles_from_splitv2_plus_refined`
  - `oof_auc = 0.9810338367721347`

主提交推荐已经切到 `splitv2_plus_pairedlr8e4`。mixed pool 只做研究上限，不做默认提交版本。

## Research Line

已完成并验证:

- `best13` leave-one-out prune
- `splitter-v2`
- `fold_seed / train_seed / warmup_seed` 解耦
- splitv2 最小对照
- splitv2 5-fold diversity pool rebuild
- 正式 `blend` CLI

已验证的 splitv2 runs:

- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2`
- `baseline_mammonet32k_warmup_splitv2`
- `baseline_mammonet32k_warmup_normaug_e4_splitv2`
- `baseline_mammonet32k_mainline_v1_f5_splitv2`
- `baseline_v2_normaug_splitv2`

当前 splitv2 最佳池:

- `xfuse_e6_splitv2`
- `mainline_v1_f5_splitv2`
- `normaug_e4_splitv2`
- `xfuse_e5_splitv2`
- `f5_e5_cosine_splitv2_trainseed123`
- `f5_e5_cosine_pairedlr8e4_splitv2`

当前 splitv2 最佳 blend:

- `blend_terminal_splitv2_plus_pairedlr8e4_refined`
  - `oof_auc = 0.9783852921796876`

## Important Rule

splitv2 是新的 CV 宇宙。
不要把 mixed ceiling 直接当成主提交版本。
只把 `splitv2` 单宇宙高分 blend 当默认提交候选。

## Blend CLI

正式命令:

```bash
uv run python main.py blend --spec outputs/research/terminal_splitv2_plus_pairedlr8e4_blend_spec.json
```

输出内容:

- `blend.json`
- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`
