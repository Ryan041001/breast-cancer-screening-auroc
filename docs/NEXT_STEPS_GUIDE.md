# Next Steps Guide

## Fixed Conclusions

- 当前推荐提交固定为:
  `blend_terminal_splitv2_plus_pairedlr8e4_refined`
- 旧宇宙 fallback 为:
  `blend_terminal_old_universe_refined`
- 绝对数值最高但不作为默认提交的是:
  `blend_terminal_all_singles_from_splitv2_plus_refined`

对应产物:

- [recommended splitv2 plus blend](../outputs/runs/blend_terminal_splitv2_plus_pairedlr8e4_refined/blend.json)
- [splitv2 plus scan](../outputs/research/2026-04-17_splitv2_candidate_scan.json)
- [splitv2 plus spec](../outputs/research/terminal_splitv2_plus_pairedlr8e4_blend_spec.json)
- [mixed ceiling blend](../outputs/runs/blend_terminal_all_singles_from_splitv2_plus_refined/blend.json)

## Protocol That Was Actually Followed

1. 在 `splitv2` 下重跑主单模与最小对照
2. 提升 `xfuse` shard 进入 splitv2 终局池
3. 用 terminal search 得到 `blend_terminal_splitv2_refined`
4. 再加入 `pairedlr8e4_splitv2` shard
5. 形成 `blend_terminal_splitv2_plus_pairedlr8e4_refined`

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

- `xfuse_e6_splitv2`
- `mainline_v1_f5_splitv2`
- `normaug_e4_splitv2`
- `xfuse_e5_splitv2`
- `f5_e5_cosine_splitv2_trainseed123`
- `f5_e5_cosine_pairedlr8e4_splitv2`

当前 splitv2 best blend:

- `blend_terminal_splitv2_plus_pairedlr8e4_refined`
  - `oof_auc = 0.9783852921796876`

当前 stopping rule:

- 新成员只有在对当前 splitv2 best pool 的增益 `> 1e-4` 时才保留
- 连续两步不过阈值就停止扩池

## Evaluation Rules

- split 改了，就是新的 CV 宇宙
- 默认提交优先选择单宇宙高分 blend
- mixed pool 只作为 ceiling，不直接当主提交
- splitv2 内部比较优先，其次再看 old-universe fallback
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

1. 默认提交使用 `blend_terminal_splitv2_plus_pairedlr8e4_refined`
2. 保留 `blend_terminal_old_universe_refined` 作为 fallback
3. mixed ceiling 只在需要研究上限时引用
4. 后续只接受满足 `gain > 1e-4` 的新 5-fold 成员

## Commands

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4_splitv2.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine_splitv2.yaml
uv run python main.py blend --spec outputs/research/terminal_splitv2_plus_pairedlr8e4_blend_spec.json
```
