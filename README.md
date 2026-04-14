# Mammography Final Project

基于双视角乳腺 X 光图像的恶性概率预测项目。当前实现会把 `train.csv` 折叠成 breast-level 样本，使用 paired `CC` / `MLO` 视图训练共享编码器模型，并生成提交所需的 `pred_score`。

## Project layout

```text
Final_Project/
├─ configs/
├─ docs/superpowers/
├─ outputs/
│  ├─ checkpoints/
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

本项目使用 `uv` 管理 Python 环境。

```bash
uv sync --extra train --group dev
```

在当前 Windows + NVIDIA GPU 环境下，`pyproject.toml` 已经把 `torch` / `torchvision` 固定到官方 PyTorch CUDA 12.8 索引，因此上面的 `uv sync` 会安装 CUDA 版而不是 CPU 版。可用下面命令确认：

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## Main commands

```bash
uv run python main.py build-manifest --config configs/smoke.yaml
uv run python main.py train --config configs/smoke.yaml
uv run python main.py predict --config configs/smoke.yaml
uv run python main.py submit --config configs/smoke.yaml
uv run python main.py run-cv --config configs/smoke.yaml
```

调试入口：

```bash
uv run python main.py train --config configs/smoke.yaml --dry-run-loader
uv run python main.py train --config configs/smoke.yaml --dry-run-model
```

## Outputs

- `outputs/runs/<experiment>/full_train/checkpoints/best.pt`：全量训练 checkpoint
- `outputs/runs/<experiment>/test_predictions.csv`：当前实验的测试集预测
- `outputs/runs/<experiment>/cv/`：fold checkpoints、OOF、fold metrics、CV 测试集均值预测
- `outputs/submissions/<experiment>_submission.csv`：提交文件

## Current modeling contract

- 每个 `breast_id` 对应一对 `CC` / `MLO` 图像
- 训练标签：只要该乳房任一行 `pathology == 'M'`，则标签为 1，否则为 0
- 验证切分采用 **patient-safe folds**，同一病人的左右乳房不会跨 fold
- 输入预处理会裁掉明显黑边，并将右侧乳房翻转到统一方向

## External libraries / models / data attribution

### Libraries

- `torch` / `torchvision`：训练、张量运算、图像变换与数据加载
- `timm`：预训练视觉骨干网络创建接口
- `scikit-learn`：AUROC 计算
- `Pillow`：JPG 读取与基础图像处理
- `PyYAML`：配置加载
- `numpy` / `pandas` / `tqdm`：训练与实验辅助依赖

### Pretrained models / weights

- `timm` pretrained image backbones：当前训练/推理流程通过 `timm.create_model(...)` 使用 ImageNet 预训练权重来初始化分类骨干，以提升小数据集收敛与泛化能力。

### Data sources

- 课程提供的 `train.csv`、`train_img/`、`test_img/`、`name_sid_submission.csv`：本项目的核心训练/测试数据
- 当前实现 **没有额外引入公开外部数据集**；如果后续为了提升最终分数接入额外公开数据，应在这里继续补充来源与用途说明。
