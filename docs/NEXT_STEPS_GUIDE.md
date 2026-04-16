# Next Steps Guide

## Fixed Conclusions

- 第一单模固定为:
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
- 第一 blend 固定为:
  `blend_best14_pruned_refined`

当前公开可复查的真实状态应以 README 和以下产物为准:
- [best14 blend](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/runs/blend_best14_pruned_refined/blend.json)
- [LOO report](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/blend_best13_leave_one_out.json)
- [splitv2 pair eval](/D:/A_ZJGSU/CODE/school/deep_learning/Final_Project/outputs/research/splitv2_pair_blend_eval.json)

## Protocol That Was Actually Followed

1. 对 `best13` 做 leave-one-out prune
2. 实现 `splitter-v2`
3. 把 `seed` 解耦为 `fold_seed / train_seed / warmup_seed`
4. 在 splitv2 下只重跑两条最小对照线:
   - `new_champion_splitv2`
   - `f5_e5_cosine_splitv2`

协议内结论:

- splitv2 折分已经平衡:
  - 每 fold `130 breasts`
  - 正例分布 `31 / 31 / 32 / 32 / 31`
- `f5_e5_cosine_splitv2` 仍然是有效 diversity shard
- 两条线在 splitv2 内部的 pair blend 有明确正增益

## Evaluation Rules

- split 改了，就是新的 CV 宇宙
- 不把 splitv2 OOF AUC 直接和旧 `best14` 宇宙做数值对位
- 只看 splitv2 内部的相对比较
- 如果后续继续 splitv2 线，优先保留最小对照和可解释增益

## Do Not Do

- 不要把 `fusion_eval.json` 当最终主判据
- 不要在 `fusion_head_variant: linear` 下继续扫 hidden dim
- 不要把 `mainline_v1_f5` 升成主线
- 不要在 paired-side 还没彻底停滞前，把 `external_paired_highconf` 提前升格为主研究方向

## Notes On Strictness

协议外实验如果发生，必须单独标记，不能直接并入主结论。
当前主结论只基于:

- `best14` prune
- `splitter-v2`
- `new_champion_splitv2`
- `f5_e5_cosine_splitv2`

## Commands

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2.yaml
```
