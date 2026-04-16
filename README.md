# Mammography Final Project

当前主任务仍然是 `internal` breast-level paired `CC/MLO` 二分类。`MammoNet32k_new` 只用于 `external single-image warm-up`，不直接替代 paired 主训练集。

## Current State

- 当前第一单模:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
  - `mean_auc = 0.967337`
  - [metrics.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4/cv/metrics.json)
- 当前 production blend:
  `blend_best14_pruned_refined`
  - `oof_auc = 0.97799770028811`
  - [blend.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/blend_best14_pruned_refined/blend.json)
  - [submission.csv](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/submissions/blend_best14_pruned_refined_submission.csv)

`best14` 来自对 `best13` 的迭代 leave-one-out prune，确认移除:
- `f5_cosine`
- `cv3_control_seed123`

对应报告:
- [blend_best13_leave_one_out.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/blend_best13_leave_one_out.json)
- [2026-04-16_model_bottleneck_report.md](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/2026-04-16_model_bottleneck_report.md)

## Production vs Research

- old universe:
  production only
  - 固定使用 `blend_best14_pruned_refined`
  - 不再继续往旧宇宙里塞新成员、扫 LR、扫 epoch
- splitv2 universe:
  research only
  - 使用新的 deterministic greedy splitter
  - `fold_seed / train_seed / warmup_seed` 已解耦
  - 只能在 splitv2 内部比较，不直接和旧宇宙 OOF 做数值对比

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

- `new_champion_splitv2`
- `f5_e5_cosine_splitv2`
- `old_champion_splitv2`
- `normaug_e4_splitv2`
- `mainline_v1_f5_splitv2`

当前 splitv2 最佳 blend:

- `blend_splitv2_stageD_refined`
  - `oof_auc = 0.9755429516414516`
- CLI 复现产物:
  - `blend_splitv2_stageD_cli`
  - `oof_auc = 0.9755429516414516`

对应文件:
- [splitv2_stage_build.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_stage_build.json)
- [splitv2 pair eval](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_pair_blend_eval.json)
- [splitv2 stage-D blend](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/blend_splitv2_stageD_refined/blend.json)

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
uv run python main.py blend --spec outputs/research/splitv2_stageD_cli_spec.json
```

`blend` 命令会输出:

- `blend.json`
- `metrics.json`
- `oof_predictions.csv`
- `test_predictions.csv`

## Guardrails

- 不把 splitv2 的 OOF 直接和旧宇宙 `best14` 做表面对比
- 不在 `fusion_head_variant: linear` 下继续扫 `fusion_hidden_dim`
- 不把 `mainline_v1_f5` 升成主线
- 不把 `external_paired_highconf` 提前升成主研究线
- 不恢复 3-fold 家族到 splitv2 主线
