# Next Steps Guide

## Fixed Conclusions

- 第一单模固定为:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
- 第一 production blend 固定为:
  `blend_best14_pruned_refined`

对应产物:

- [best14 blend](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/blend_best14_pruned_refined/blend.json)
- [LOO report](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/blend_best13_leave_one_out.json)
- [splitv2 pair eval](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_pair_blend_eval.json)
- [splitv2 stage build](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_stage_build.json)

## Protocol That Was Actually Followed

1. 对 `best13` 做 leave-one-out prune
2. 实现 `splitter-v2`
3. 把 `seed` 解耦为 `fold_seed / train_seed / warmup_seed`
4. 在 splitv2 下重跑最小 5-fold 对照
5. 用 staged refined blend 重建 splitv2 小池

协议内结论:

- splitv2 fold 已平衡:
  - 每 fold `130 breasts`
  - 正例分布 `31 / 31 / 32 / 32 / 31`
  - patient 数在 `123 / 124` 间波动
- `f5_e5_cosine_splitv2` 仍然是有效 diversity shard
- splitv2 当前最优池为 5 个成员
- `baseline_v2_normaug_splitv2` 没有达到 `gain > 1e-4` 的保留阈值

## Current Splitv2 Baseline

当前 splitv2 best pool:

- `new_champion_splitv2`
- `f5_e5_cosine_splitv2`
- `old_champion_splitv2`
- `normaug_e4_splitv2`
- `mainline_v1_f5_splitv2`

当前 splitv2 best blend:

- `blend_splitv2_stageD_refined`
  - `oof_auc = 0.9755429516414516`

当前 stopping rule:

- 新成员只有在对当前 splitv2 best pool 的增益 `> 1e-4` 时才保留
- 连续两步不过阈值就停止扩池

## Evaluation Rules

- split 改了，就是新的 CV 宇宙
- 不把 splitv2 OOF 直接和旧宇宙 `best14` 做数值对位
- 只看 splitv2 内部相对比较
- 优先保留可解释、可复现、5-fold 的 diversity shard

## Do Not Do

- 不把 `fusion_eval.json` 当最终主判据
- 不在 `fusion_head_variant: linear` 下继续扫 hidden dim
- 不把 `mainline_v1_f5` 升成生产主线
- 不在 paired-side 已进入边际递减后重新做大范围 LR/epoch 扫描
- 不恢复 3-fold 家族到 splitv2 主线
- 不把 `external_paired_highconf` 提前升级成当前主研究方向

## Immediate Follow-up

优先级顺序:

1. 维持旧宇宙 production 冻结
2. 在 splitv2 内继续小池式增量研究
3. 只接受满足 `gain > 1e-4` 的新 5-fold 成员
4. 优先增强 blend 工具链和测试，而不是重新大扫超参

## Commands

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2.yaml
uv run python main.py blend --spec outputs/research/splitv2_stageD_cli_spec.json
```
