# Next Steps Guide

## Fixed conclusions

- 第一单模固定为
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
- 第一 blend 固定为
  `blend_best12_plus_baselinev2normaug_refined`

最新指标：
- 单模 `mean_auc = 0.967337`
- blend `oof_auc = 0.977119`

## Why the old plan changed

- `mainline_v1_f5` 作为单模较弱，只适合小权重互补
- `linear` fusion head 下，`fusion_hidden_dim` 一类参数不会生效
- 改 `seed` 会同时改变 warm-up metadata hash，不是便宜的纯 seed ablation
- 当前真正的瓶颈是高互补 family 如何做非平凡 5-fold promotion

## Next experiments

第一发：

```yaml
experiment:
  name: baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine

train:
  epochs: 5
```

第二发：

```yaml
experiment:
  name: baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr8e4

train:
  learning_rate: 0.0008
```

只有当前两发任一配置加入 `blend_best12_plus_baselinev2normaug_refined` 后带来 `> 1e-4` 的 OOF 增益，才做第三发组合版。

## Evaluation rules

- `mean_auc` 不能明显塌
- 加入当前 best12 pool 后必须 `+OOF AUC > 1e-4`
- 最优权重必须保持在小权重区且有解释性

不要再把自动生成的 `fusion_eval.json` 当主判据。它只适合粗筛，不适合当前这种弱但互补 shard 的决策。

## Do not do

- 不要优先跑 `seed123`
- 不要在 `fusion_head_variant: linear` 下继续测 hidden dim
- 不要继续推进 `mainline_v1_f5`
- 不要提前把 `external_paired_highconf` 升级成主研究方向

## Commands

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr8e4.yaml
```
