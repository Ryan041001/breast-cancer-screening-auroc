# Mammography Final Project

当前主任务仍然是 internal breast-level paired CC/MLO 二分类。
`MammoNet32k_new` 只用于 external single-image warm-up，不直接替代 paired 主训练集。

## Current State

- 当前第一单模:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
  - `mean_auc = 0.967337`
  - [metrics.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4/cv/metrics.json)
- 当前第一 blend:
  `blend_best14_pruned_refined`
  - `oof_auc = 0.97799770028811`
  - [blend.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/blend_best14_pruned_refined/blend.json)
  - [submission.csv](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/submissions/blend_best14_pruned_refined_submission.csv)

`best14` 是从 `best13` 做 leave-one-out prune 得到的。
确认移除的两个成员是:
- `f5_cosine`
- `cv3_control_seed123`

对应报告:
- [blend_best13_leave_one_out.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/blend_best13_leave_one_out.json)
- [2026-04-16_model_bottleneck_report.md](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/2026-04-16_model_bottleneck_report.md)

## Verified Protocol

当前已经完成并验证的协议内工作:

1. `best13` leave-one-out prune
2. `splitter-v2 + fold_seed/train_seed/warmup_seed` 解耦
3. splitv2 下的两条最小对照线

splitv2 两条协议内结果:

- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2`
  - `mean_auc = 0.9699796059675092`
  - `oof_auc = 0.9602718311132931`
- `baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2`
  - `mean_auc = 0.9604119079205485`
  - `oof_auc = 0.9574294905750572`
- pair blend in splitv2 universe only:
  - best weight = `0.464 * new_champion_splitv2 + 0.536 * f5_e5_cosine_splitv2`
  - `oof_auc = 0.9635146832728259`
  - gain over splitv2 baseline = `+0.003242852159532794`

对应文件:
- [fold_audit.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2/cv/fold_audit.json)
- [splitv2_pair_blend_eval.json](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_pair_blend_eval.json)

重要规则:
- splitv2 是新的 CV 宇宙
- 不把 splitv2 的 OOF AUC 直接和旧 `best14` 宇宙一一比较
- 只在 splitv2 内部比较 `new_champion_splitv2`、`f5_e5_cosine_splitv2` 及其组合

## Warm-up Reuse

当前 shared warm-up 复用目录:

- `outputs/runs/_shared_external_warmup/7159fb51f1f5fd90896bc62b8a70745cc59d5491c7e6301aa4edf0a5190e5eb3/`

当前复用规则:

- external 数据 contract 相同
- warm-up 超参相同
- `warmup_seed` 相同
- 数据签名相同

因此:
- 只改 paired finetune / CV / blend，可以复用同一个 warm-up
- 只改 `train_seed` 或 `fold_seed`，不会强制开新 warm-up family
- 改 `warmup_seed` 才会生成新的 warm-up family

## Current Bottleneck

当前瓶颈不是工程缺项，而是 residual diversity 的边际递减。

明确不要做:

- 不要继续把 `mainline_v1_f5` 当主线推进
- 不要在 `fusion_head_variant: linear` 下继续扫 `fusion_hidden_dim`
- 不要把 external paired 子集提前升级成主研究线
- 不要把 splitv2 结果直接混进旧宇宙的 blend 排名

## Layout

```text
configs/
docs/
outputs/
scripts/
src/final_project/
tests/
main.py
```
