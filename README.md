# Mammography Final Project

基于双视图乳腺 X 光图像的恶性概率预测项目。当前实现以课程数据的 `train.csv` / `train_img` / `test_img` 为主任务，支持：

- breast-level manifest 构建
- patient-safe cross validation
- paired `CC` / `MLO` 共享 backbone + fusion head
- OOF / test 预测与提交文件生成
- 基于 `MammoNet32k_new` 的外部单视图 warmup，再回到课程 paired-view 任务微调

## Project Layout

```text
Final_Project/
├─ configs/
├─ docs/superpowers/
├─ MammoNet32k_new/
├─ outputs/
│  ├─ runs/
│  └─ submissions/
├─ src/final_project/
│  ├─ data/
│  ├─ engine/
│  ├─ model/
│  └─ utils/
├─ tests/
└─ main.py
```

## Environment

使用 `uv` 管理环境：

```bash
uv sync --extra train --group dev
```

确认 GPU：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Main Commands

基础流程：

```bash
uv run python main.py build-manifest --config configs/smoke.yaml
uv run python main.py train --config configs/smoke.yaml
uv run python main.py predict --config configs/smoke.yaml
uv run python main.py submit --config configs/smoke.yaml
uv run python main.py run-cv --config configs/smoke.yaml
uv run python main.py tune-iterate --configs configs/baseline.yaml configs/retune_b8_e12.yaml --report-name round_1
```

调试入口：

```bash
uv run python main.py train --config configs/smoke.yaml --dry-run-loader
uv run python main.py train --config configs/smoke.yaml --dry-run-model
```

外部数据 warmup：

```bash
uv run python main.py warmup-external --config configs/smoke_mammonet32k_warmup.yaml
uv run python main.py train --config configs/smoke_mammonet32k_warmup.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup.yaml
```

## Configs

- `configs/smoke.yaml`: 基础 smoke 流程
- `configs/baseline.yaml`: 课程数据 baseline
- `configs/smoke_mammonet32k_warmup.yaml`: 128 张外部图像 warmup + 课程数据 smoke
- `configs/baseline_mammonet32k_warmup.yaml`: 课程数据 baseline，前置 2 epoch MammoNet32k warmup
- `configs/baseline_mammonet32k_warmup_normaug.yaml`: warmup + `normaug` 版本，用于下一轮对比

外部 warmup 相关字段都放在 `train` 下：

- `external_warmup_epochs`
- `external_warmup_batch_size`
- `external_warmup_num_workers`
- `external_warmup_learning_rate`
- `external_warmup_max_samples`

数据路径相关字段：

- `paths.external_data_root`
- `paths.external_catalog`
- `paths.external_splits_dir`

只提供 `external_data_root` 时，会自动解析 `catalog.csv` 和 `splits/`。

## Outputs

- `outputs/runs/<experiment>/external_warmup/checkpoints/best.pt`: 外部 warmup backbone checkpoint
- `outputs/runs/<experiment>/external_warmup/run.log`: 外部 warmup 过程日志
- `outputs/runs/<experiment>/full_train/checkpoints/best.pt`: 全量训练 checkpoint
- `outputs/runs/<experiment>/full_train/run.log`: 全量训练过程日志
- `outputs/runs/<experiment>/cv/oof_predictions.csv`: OOF 预测
- `outputs/runs/<experiment>/cv/test_predictions.csv`: CV 平均后的测试集预测
- `outputs/runs/<experiment>/cv/run.log`: CV 过程日志
- `outputs/runs/<experiment>/test_predictions.csv`: 当前实验主测试预测
- `outputs/runs/tune_iter_<report_name>_best_blend/`: 调优迭代自动备份的最佳 blend 预测
- `outputs/submissions/<experiment>_submission.csv`: 提交文件
- `outputs/research/<report_name>/leaderboard.json|csv|md`: 调优迭代汇总排行榜

## Current Modeling Contract

- 每个 `breast_id` 对应一对 `CC` / `MLO`
- 标签规则：同一乳房任一行 `pathology == 'M'` 即记为 1，否则为 0
- 切分规则：patient-safe folds，同一病人的左右乳房不会跨 fold
- 预处理：裁掉黑边，并将右乳统一翻转到同一朝向
- 外部 warmup：将 `MammoNet32k_new` 中有标签的单张图像映射为恶性/非恶性二分类，先训练 backbone，再加载到 paired model

## Tuning Order

建议按这个顺序继续迭代：

1. `baseline.yaml`
2. `baseline_mammonet32k_warmup.yaml`
3. `baseline_mammonet32k_warmup_normaug.yaml`
4. `baseline_fusion_mlp_gelu_d0.yaml`
5. `baseline_fusion_mlp_gelu_d10.yaml`
6. 对最佳候选做 seed / epoch retest

优先看：

- `cv/metrics.json` 的 `mean_auc`
- `cv/fusion_eval.json` 的 blend gain
- OOF 与 baseline 的互补性，而不是只看单次 fold 波动

## External Libraries / Models / Data Attribution

### Libraries

- `torch` / `torchvision`: 训练、推理、数据加载和图像变换
- `timm`: backbone 构建与 ImageNet 预训练权重加载
- `scikit-learn`: AUROC 计算
- `Pillow`: 图像读取与基础预处理
- `PyYAML`: 配置加载
- `numpy` / `pandas` / `tqdm`: 实验辅助

### Pretrained Models / Weights

- `timm` pretrained backbones: 当前默认通过 `timm.create_model(...)` 使用 ImageNet 预训练权重初始化视觉骨干

### Data Sources

- 课程数据 `train.csv` / `train_img` / `test_img` / `name_sid_submission.csv`: 最终提交目标数据
- `MammoNet32k_new`: 外部公开乳腺影像数据，用于 backbone warmup，提升课程任务的初始化质量
