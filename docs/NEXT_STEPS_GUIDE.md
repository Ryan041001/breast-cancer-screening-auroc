# 项目向导：当前主线与下一步

## 1. 当前结论

当前项目不要再把 `baseline_mammonet32k_mainline_v1` 当成新主线继续推进。

现阶段应固定两条生产结论：

- 第一单模：`baseline_mammonet32k_warmup_e4_lr5e4`
- 第一 blend：`baseline 0.2 + baseline_mammonet32k_warmup_e4_lr5e4 0.8`

已知核心指标：

- 单模 `mean_auc = 0.959462`
- 单模 `candidate_oof_auc = 0.951144`
- 最优 blend `best_blend_auc = 0.965052`

对应产物：

- `outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4/cv/metrics.json`
- `outputs/runs/baseline_mammonet32k_warmup_e4_lr5e4/cv/fusion_eval.json`

## 2. 当前任务定义

### internal 主任务

- 任务单位：`breast_id`
- 每个样本必须恰好一对 `CC + MLO`
- patient-safe CV
- 这是最终提交和模型选择的主依据

规模：

- `train_breasts = 650`
- `test_breasts = 650`
- 训练标签：`negative = 493`，`positive = 157`

### external warm-up

- 数据源：`MammoNet32k_new`
- 当前用途：single-image warm-up
- 目标：先训练 backbone，再回到 internal paired 主任务微调

有效 warm-up 样本：

- `train = 22448`
- `val = 4827`
- `test = 4846`

为什么不能直接升格成 paired 主训练集：

- `unknown view = 27212 / 32191`
- `unknown laterality = 29403 / 32191`
- `mixed-label patients = 911`

结论：

- external 很适合做表征学习
- 当前 metadata 下不适合直接替代 internal paired 主训练集

## 3. 之后的方向

### 生产线

保持不动：

- 生产冠军单模继续使用 `baseline_mammonet32k_warmup_e4_lr5e4`
- 生产冠军 blend 继续使用 `baseline 0.2 + champion 0.8`

### 提分线

优先级最高，按最小改动推进：

1. 先做 paired finetune retune，不改 backbone，不重构主数据集。
2. 只把新加入的训练控制项一项一项移植到冠军线上。
3. 先做 `3-fold` quick screen，再把前 1 名升到 `5-fold`。

当前推荐实验顺序：

1. `baseline_mammonet32k_warmup_e4_lr5e4_cv3_control`
2. `baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6`
3. `baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1`
4. `baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1_cosine`
5. `baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1_cosine_normonly`

判断规则：

- 先和 `cv3_control` 比，不直接和 `5-fold` 冠军比
- 优先看 `mean_auc`、`candidate_oof_auc`、`best_blend_auc`
- 只有在 `3-fold` 中赢下控制组，才值得升到 `5-fold`

### 研究线

暂缓，等 paired finetune 收敛后再做：

- 轻量 fusion head
- `external_paired_highconf.csv` 辅助实验
- external auxiliary 混入比例实验

## 4. 数据清洗策略

### internal

只做完整性核查，不重构。

### external warm-up

采用保守清洗，不做激进修复：

- 丢弃 `unknown pathology`
- `unknown laterality / unknown view` 不补全
- 不再把 unknown laterality 默认映射成左侧
- 保留 `cache_mode: preprocess`
- 保留 warm-up metadata 校验
- 保留 `dataset + label balanced` 采样能力

辅助脚本：

- 审计：`scripts/external_audit.py`
- 高置信 paired 辅助集生成：`scripts/build_external_paired_highconf.py`

### external paired 子集

只允许作为 train-only auxiliary：

- 不替代 internal paired train
- 不参与最终模型选择
- 不拿 external val/test 做最终优选依据

## 5. warm-up 复用规则

当前代码已经支持 shared warm-up 复用。

共享目录：

- `outputs/runs/_shared_external_warmup/<metadata_hash>/`

复用条件：

- external 数据 contract 相同
- catalog 和 split 数据签名相同
- warm-up 超参相同
- seed 相同

这意味着：

- 如果不同 config 只改 paired finetune / CV / blend，可以共用同一个 warm-up
- 清洗前和清洗后的 warm-up 会自动分开保存，不会互相覆盖

当前已经验证可复用的 warm-up family：

- metadata hash:
  `7159fb51f1f5fd90896bc62b8a70745cc59d5491c7e6301aa4edf0a5190e5eb3`

对应 shared checkpoint：

- `outputs/runs/_shared_external_warmup/7159fb51f1f5fd90896bc62b8a70745cc59d5491c7e6301aa4edf0a5190e5eb3/checkpoints/best.pt`

注意：

- `normonly` 目前会同时影响 warm-up 和 paired
- 所以 `..._cosine_normonly.yaml` 不会和 baseline 那组共用同一个 warm-up family

## 6. 运行与排查

### 推荐命令

建立 shared warm-up 并启动 quick screen：

```bash
uv run python main.py warmup-external --config configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_control.yaml

uv run python main.py tune-iterate --configs \
  configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_control.yaml \
  configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6.yaml \
  configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1.yaml \
  configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1_cosine.yaml \
  configs/baseline_mammonet32k_warmup_e4_lr5e4_cv3_e6_freeze1_cosine_normonly.yaml \
  --report-name round_4_paired_retune_seed42
```

做 external 数据审计：

```bash
uv run python scripts/external_audit.py
```

生成 high-confidence auxiliary：

```bash
uv run python scripts/build_external_paired_highconf.py
```

### 进度条与 GPU

如果出现“进度条停在 0，但 GPU 占用很高”：

1. 优先检查是否有旧训练进程残留。
2. 再检查当前运行是否复用了已有 warm-up，导致很快跳过 warm-up 阶段。
3. 终端现在已强制输出 `tqdm`，fresh run 应该能看到 epoch 级和 batch 级进度。

## 7. 不要做的事

- 不要把 `MammoNet32k_new` 直接升格成 paired 主训练集
- 不要按 patient 级别直接聚合 external 标签
- 不要在 view 和 laterality 大量缺失时强行补 paired
- 不要优先跑 fusion MLP，再回头补 paired 控制
- 不要因为 old checkpoint 存在，就直接复用无 metadata 的旧 warm-up

## 8. 文档说明

当前仓库文档分成两类：

- 本文档：当前执行向导
- `docs/superpowers/specs/`：保留的历史设计说明

旧的执行计划文档已从主入口清理，因为它们不再代表当前优先级。
