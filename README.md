# Mammography Final Project

当前主任务仍然是 `internal` breast-level paired `CC/MLO` 二分类。`MammoNet32k_new` 只用于 `external single-image warm-up`，不直接替代 paired 主训练集。

## Current State

- 当前最高单模 `splitv2`:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
  - `mean_auc = 0.9699796059675092`
  - `oof_auc = 0.9602718311132931`
  - [metrics.json](outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2/cv/metrics.json)
- 当前推荐提交:
  `blend_terminal_splitv2_plus_pairedlr8e4_refined`
  - `oof_auc = 0.9783852921796876`
  - [blend.json](outputs/runs/blend_terminal_splitv2_plus_pairedlr8e4_refined/blend.json)
  - [submission.csv](outputs/submissions/blend_terminal_splitv2_plus_pairedlr8e4_refined_submission.csv)
  - [root copy](submission_recommended_splitv2_plus_pairedlr8e4.csv)
- 当前旧宇宙 fallback:
  `blend_terminal_old_universe_refined`
  - `oof_auc = 0.9781914962338988`
  - [blend.json](outputs/runs/blend_terminal_old_universe_refined/blend.json)
- 当前绝对数值最高:
  `blend_terminal_all_singles_from_splitv2_plus_refined`
  - `oof_auc = 0.9810338367721347`
  - mixed old-universe + splitv2 ceiling, not the default submission choice

## Submission Logic

- 推荐主提交使用 `splitv2` 单宇宙高分 blend:
  `blend_terminal_splitv2_plus_pairedlr8e4_refined`
- `blend_terminal_old_universe_refined` 仍然是干净、稳妥的旧宇宙 fallback。
- `blend_terminal_all_singles_from_splitv2_plus_refined` 只作为研究上限，因为它混合了 old-universe 和 splitv2。

## Splitv2 Status

已验证的 splitv2 5-fold 运行:

- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
  - `mean_auc = 0.9699796059675092`
  - `oof_auc = 0.9602718311132931`
- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2`
  - `mean_auc = 0.9604119079205485`
  - `oof_auc = 0.9574294905750572`
- `baseline_mammonet32k_warmup_splitv2`
  - `mean_auc = 0.9662724284484078`
- `baseline_mammonet32k_warmup_normaug_e4_splitv2`
  - `mean_auc = 0.9567673837619115`
- `baseline_mammonet32k_mainline_v1_f5_splitv2`
  - `mean_auc = 0.9454734141946123`
- `baseline_v2_normaug_splitv2`
  - 未进入当前 splitv2 最佳池

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
  - relative to `blend_terminal_splitv2_refined`: `+0.00020671567550811787`

对应文件:
- [splitv2 candidate scan](outputs/research/2026-04-17_splitv2_candidate_scan.json)
- [splitv2 plus spec](outputs/research/terminal_splitv2_plus_pairedlr8e4_blend_spec.json)
- [splitv2 plus blend](outputs/runs/blend_terminal_splitv2_plus_pairedlr8e4_refined/blend.json)

## Warm-up Reuse

shared warm-up 目录:

- `outputs/runs/_shared_external_warmup/7159fb51f1f5fd90896bc62b8a70745cc59d5491c7e6301aa4edf0a5190e5eb3/`

复用条件:

- external 数据 contract 相同
- warm-up 超参相同
- `warmup_seed` 相同
- 数据签名相同

因此:

- 只改 paired finetune / CV / blend，可以复用同一个 warm-up
- 只改 `train_seed` 或 `fold_seed`，不会强制开新的 warm-up family
- 改 `warmup_seed` 才会生成新的 warm-up family

## CLI

常用命令:

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2.yaml
uv run python main.py blend --spec outputs/research/terminal_splitv2_plus_pairedlr8e4_blend_spec.json
```

`blend` 命令会输出:

- `blend.json`
- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`

## Guardrails

- 不把 mixed old-universe + splitv2 ceiling 直接当主提交版本
- 不把 splitv2 的 OOF 和旧宇宙早期 frozen 结果做粗暴一对一对位
- 不在 `fusion_head_variant: linear` 下继续扫 `fusion_hidden_dim`
- 不把 `mainline_v1_f5` 升成主线
- 不把 `external_paired_highconf` 提前升成主研究线
- 不恢复 3-fold 家族到 splitv2 主线
