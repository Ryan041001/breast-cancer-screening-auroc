# Mammography Final Project

基于双视图乳腺 X 光图像的恶性概率预测项目。当前仓库的主任务是 internal breast-level paired classification，external `MammoNet32k_new` 只作为 single-image warm-up 数据源使用。

## 当前主线

当前不要把 `baseline_mammonet32k_mainline_v1` 当成新主线继续推。

现阶段固定结论：

- 第一单模：`baseline_mammonet32k_warmup_e4_lr5e4`
- 第一 blend：`baseline 0.2 + baseline_mammonet32k_warmup_e4_lr5e4 0.8`

当前最重要的下一步不是换 backbone，而是对冠军线做 paired finetune 控制项的拆解式移植和筛选。

详细执行方向见 [docs/NEXT_STEPS_GUIDE.md](docs/NEXT_STEPS_GUIDE.md)。

## Project Layout

```text
Final_Project/
├─ configs/
├─ docs/
│  ├─ NEXT_STEPS_GUIDE.md
│  └─ superpowers/
│     ├─ plans/
│     └─ specs/
├─ MammoNet32k_new/
├─ outputs/
│  ├─ research/
│  ├─ runs/
│  └─ submissions/
├─ scripts/
├─ src/final_project/
│  ├─ data/
│  ├─ engine/
│  ├─ model/
│  └─ utils/
├─ tests/
└─ main.py
```

## Environment

安装依赖：

```bash
uv sync --extra train --group dev
```

检查 GPU：

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

external warm-up：

```bash
uv run python main.py warmup-external --config configs/smoke_mammonet32k_warmup.yaml
uv run python main.py train --config configs/smoke_mammonet32k_warmup.yaml
uv run python main.py run-cv --config configs/baseline_mammonet32k_warmup.yaml
```

数据审计与辅助集生成：

```bash
uv run python scripts/external_audit.py
uv run python scripts/build_external_paired_highconf.py
```

## Modeling Contract

### Internal paired task

- 每个 `breast_id` 必须恰好对应一对 `CC + MLO`
- 标签规则：同一乳房任一视图 `pathology == 'M'` 记为 `1`，否则为 `0`
- 切分规则：patient-safe folds，同一病人的左右乳房不会跨 fold
- 预处理：裁剪黑边，并将右乳规范到统一朝向

### External warm-up

- `MammoNet32k_new` 仅用于 single-image warm-up
- `malignant -> 1`
- `benign / normal -> 0`
- `unknown pathology` 直接过滤
- `unknown laterality / unknown view` 不做补全

## Warm-up Reuse

当前已支持 shared warm-up 复用。不同 config 只要 warm-up 数据 contract、数据签名、seed 和 warm-up 超参一致，就会复用同一份 warm-up。

共享目录：

- `outputs/runs/_shared_external_warmup/<metadata_hash>/`

已验证的 clean-v1 shared warm-up：

- metadata hash:
  `7159fb51f1f5fd90896bc62b8a70745cc59d5491c7e6301aa4edf0a5190e5eb3`

注意：

- 旧的无 metadata warm-up checkpoint 会被保留
- 但不会自动当成当前 clean contract 的可复用缓存
- `transform_profile` 目前同时影响 warm-up 和 paired；`normonly` 会形成新的 warm-up family

## Outputs

- `outputs/runs/<experiment>/external_warmup/checkpoints/best.pt`
- `outputs/runs/<experiment>/external_warmup/run.log`
- `outputs/runs/<experiment>/full_train/checkpoints/best.pt`
- `outputs/runs/<experiment>/full_train/run.log`
- `outputs/runs/<experiment>/cv/oof_predictions.csv`
- `outputs/runs/<experiment>/cv/test_predictions.csv`
- `outputs/runs/<experiment>/cv/run.log`
- `outputs/runs/<experiment>/cv/metrics.json`
- `outputs/runs/<experiment>/cv/fusion_eval.json`
- `outputs/research/<report_name>/leaderboard.json`
- `outputs/research/<report_name>/leaderboard.csv`
- `outputs/research/<report_name>/leaderboard.md`
- `outputs/submissions/<experiment>_submission.csv`

## Tuning Priority

当前推荐顺序：

1. 固定冠军生产线与冠军 blend
2. 做 paired finetune 控制项的 `3-fold` quick screen
3. 只把 quick screen 前 1 名升到 `5-fold`
4. paired retune 收敛后，再做轻量 fusion head
5. external paired high-confidence 子集只作为 train-only auxiliary 试验

不要直接把 external 数据升格成新的 paired 主训练集。

## External Libraries / Models / Data Attribution

### Libraries

- `torch` / `torchvision`: 训练、推理、数据加载与图像变换
- `timm`: backbone 构建与 ImageNet 预训练权重加载
- `scikit-learn`: AUROC 计算
- `Pillow`: 图像读取与基础预处理
- `PyYAML`: 配置加载
- `numpy` / `pandas` / `tqdm`: 实验与数据处理辅助

### Pretrained models / weights

- `timm` pretrained backbones: 当前默认通过 `timm.create_model(...)` 使用 ImageNet 预训练权重初始化视觉 backbone

### Data sources

- 课程数据：`train.csv` / `train_img` / `test_img` / `name_sid_submission.csv`
- `MammoNet32k_new`：external public mammography dataset，用于 backbone warm-up，不直接替代主任务 paired 训练集
