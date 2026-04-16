# Current State

## Production Line

- best single:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
  - `mean_auc = 0.967337`
- best blend:
  `blend_best14_pruned_refined`
  - `oof_auc = 0.97799770028811`

Production 已冻结在旧宇宙，不再继续往 `best14` 中加入新成员。

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

- `new_champion_splitv2`
- `f5_e5_cosine_splitv2`
- `old_champion_splitv2`
- `normaug_e4_splitv2`
- `mainline_v1_f5_splitv2`

当前 splitv2 最佳 blend:

- `blend_splitv2_stageD_refined`
  - `oof_auc = 0.9755429516414516`
- CLI 复现:
  - `blend_splitv2_stageD_cli`
  - `oof_auc = 0.9755429516414516`

## Important Rule

splitv2 是新的 CV 宇宙。
不要把 splitv2 OOF 直接和旧宇宙 `best14` 的 OOF 做数值对比。
只在 splitv2 内部比较模型和 blend。

## Blend CLI

正式命令:

```bash
uv run python main.py blend --spec outputs/research/splitv2_stageD_cli_spec.json
```

输出内容:

- `blend.json`
- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`
