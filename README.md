# Mammography Final Project

当前主任务是 internal breast-level paired CC/MLO 二分类。`MammoNet32k_new` 只用于 single-image warm-up，不直接替代 paired 主训练集。

## Current Leaders

- 第一单模：
  `baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4`
  - `mean_auc = 0.967337`
- 第一 blend：
  `blend_best12_plus_baselinev2normaug_refined`
  - `oof_auc = 0.977119`

对应产物：
- `outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr5e4/cv/metrics.json`
- `outputs/runs/blend_best12_plus_baselinev2normaug_refined/blend.json`
- `outputs/submissions/blend_best12_plus_baselinev2normaug_refined_submission.csv`

## Current Bottleneck

工程基础已经补齐，当前瓶颈是 residual diversity 的边际递减，不再是缺训练开关。

当前不建议：
- 继续把 `baseline_mammonet32k_mainline_v1` / `mainline_v1_f5` 当主线推进
- 继续堆只能带来 `1e-5` 到 `5e-5` 增益的弱 blend 成员
- 在 `fusion_head_variant: linear` 下测试 `fusion_hidden_dim` 一类参数并把它当有效实验

## Recommended Next Experiments

优先顺序固定为两发：

1. `configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine.yaml`
2. `configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr8e4.yaml`

只有当其中任一配置加入当前 best12 blend 后带来 `> 1e-4` 的 OOF 增益时，才继续做组合版。

## Warm-up Reuse

shared warm-up 目录：
- `outputs/runs/_shared_external_warmup/<metadata_hash>/`

复用条件包括：
- external 数据 contract 相同
- 数据签名相同
- warm-up 超参相同
- `seed` 相同

所以只改 paired finetune / CV / blend 的 config，可以共用同一个 warm-up；改 seed 则会形成新的 warm-up family。

## Practical Notes

- `fusion_head_variant=linear` 时，`fusion_hidden_dim` / `fusion_dropout` / `fusion_activation` / `fusion_layer_norm` / `fusion_residual` 都会被忽略。CLI 和 `run-cv` 现在会显式告警。
- `run-cv` 的 fusion eval reference 现在应该显式配置，而不是默认只锚定 `baseline`。默认值仍是 `baseline`，但可以通过 `train.fusion_eval_reference_run` 修改。

## Commands

安装依赖：

```bash
uv sync --extra train --group dev
```

运行 CV：

```bash
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e5_freeze1_cosine.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup_e4_lr5e4_f5_e6_freeze1_cosine_pairedlr8e4.yaml
```

external 数据审计：

```bash
uv run python scripts/external_audit.py
uv run python scripts/build_external_paired_highconf.py
```

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
