# AuroraCP

**AuroraCP** 是一个基于 Aurora 预训练模型的多变量天气微调方法。它在 Aurora
原有的地表变量和多层大气变量之外，进一步学习三个与陆面能量和水分过程相关的
变量：

- `sshf`：地表感热通量（surface sensible heat flux）
- `slhf`：地表潜热通量（surface latent heat flux）
- `vswl`：土壤体积含水量（volumetric soil water layer）

AuroraCP 使用 LoRA、变量 token embeddings 和输出 heads 适配预训练模型，并通过
单步监督、两步 rollout 监督和 replay-buffer rollout 三个阶段逐步提高长期预测
稳定性。训练入口支持单卡以及单机多卡 DDP。

## 1. 数据准备

### 1.1 环境配置

推荐环境为 Linux、Python 3.10、PyTorch 2.7.1、CUDA 12.8：

```bash
conda env create -f environment.yml
conda activate aurora
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

### 1.2 预训练权重和静态变量

训练前需要准备 Aurora 预训练权重以及静态地理变量：

```text
ckpt/aurora-0.25-pretrained.ckpt
ckpt/aurora-0.25-static.pickle
```

这些大文件不包含在 Git 仓库中。默认训练脚本从上述位置加载，也可以通过
`PRETRAINED` 指定其他预训练权重。静态文件包含 `lsm`、`z` 和 `slt`，其中 `lsm`
还可用于仅在陆地区域计算 `sshf/slhf/vswl` 的损失。

### 1.3 数据目录与文件格式

数据按年份存放，每个 `.pt` 文件对应一个时刻：

```text
weatherbench2_73var/
├── 1979/
│   ├── 1979-0000.pt
│   ├── 1979-0006.pt
│   └── ...
├── 1980/
└── ...
```

每个文件应为 `float32 [73, 120, 240]` tensor。文件名采用 `YEAR-HOURS.pt`，
`HOURS` 表示从当年 1 月 1 日 00:00 开始累计的小时数。数据原始采样间隔为 6
小时，训练时可以设置 6 小时或 12 小时预测步长。

AuroraCP 使用的通道映射如下：

| 类型 | 变量 | 数据通道 |
| --- | --- | --- |
| 原始地表变量 | `2t`, `10u`, `10v`, `msl` | 0, 1, 2, 3 |
| 位势高度 | `z`，13 个气压层 | 4:17 |
| 纬向风 | `u`，13 个气压层 | 17:30 |
| 经向风 | `v`，13 个气压层 | 30:43 |
| 温度 | `t`，13 个气压层 | 43:56 |
| 比湿 | `q`，13 个气压层 | 56:69 |
| AuroraCP 陆面变量 | `sshf`, `slhf`, `vswl` | 69, 70, 71 |

默认气压层为：

```text
1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa
```

### 1.4 启用 AuroraCP 的三个陆面变量

训练 `sshf/slhf/vswl` 前，需要确认 [args.py](args.py) 中使用的是 7 个 surface
变量配置：

```python
DEFAULT_MAIN_SURF_VARS = ["2t", "10u", "10v", "msl"]
DEFAULT_LAND_VARS = ["sshf", "slhf", "vswl"]

DEFAULT_SURF_LOSS_WEIGHTS = [3.5, 0.77, 0.66, 1.6, 0.5, 0.5, 0.5]

args.surf_vars = args.main_surf_vars + args.land_vars
args.surf_channels = {**args.main_surf_channels, **args.land_channels}
```

`surf_loss_weights` 的顺序必须与 `main_surf_vars + land_vars` 完全一致。当前代码
如果使用仅包含 `main_surf_vars` 的配置，则不会训练、输出或评估
`sshf/slhf/vswl`。

## 2. 训练

### 2.1 三阶段训练流程

AuroraCP 使用累计 epoch 编号衔接三个阶段：

| 阶段 | 训练入口 | 累计 epoch | 默认训练年份 | 训练方式 | 默认学习率 |
| --- | --- | ---: | --- | --- | --- |
| Stage 1 | `main.py` | 0 -> 20 | 1979-2017 | 单步监督 | LoRA `1e-3`，embeddings/heads `1e-4` |
| Stage 2 | `main.py` | 20 -> 30 | 2010-2017 | 两步 rollout 联合监督 | 两组均为 `5e-5` |
| Stage 3 | `rollout_finetune.py` | 30 -> 100 | 2010-2017 | replay-buffer rollout | 两组均为 `5e-5` |

年份范围采用左闭右开区间，例如 `1979 2018` 表示使用 1979 至 2017 年。

Stage 3 从 Stage 2 的最终 checkpoint 初始化。默认在全局 Epoch 40 开始评估；
生成 Epoch 56 时，rollout curriculum 从最大 12 个 lead steps 切换到 20 个 lead
steps。

### 2.2 检查启动命令

正式训练前建议先执行 dry run。该命令只打印三个阶段的完整命令，不加载模型和
数据：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DRY_RUN=1 \
./train_three_stage.sh
```

### 2.3 单卡与多卡训练

单卡训练：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0 \
DATA_FOLDER=/path/to/weatherbench2_73var \
RUN_ROOT=weights/auroracp \
LOG_DIR=log/auroracp \
./train_three_stage.sh
```

双卡 DDP 训练：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DATA_FOLDER=/path/to/weatherbench2_73var \
RUN_ROOT=weights/auroracp \
LOG_DIR=log/auroracp \
./train_three_stage.sh
```

`STAGE1_BATCH_SIZE`、`STAGE2_BATCH_SIZE` 和 `STAGE3_BATCH_SIZE` 均表示每张 GPU
的 batch size：

```text
effective batch size = per-GPU batch size x GPU 数量 x accumulation_steps
```

### 2.4 陆面变量损失区域

默认情况下，`sshf/slhf/vswl` 与其他变量一样在整个网格上计算损失：

```bash
MASK_OCEAN_FOR_LAND_VARS=false GPU_IDS=0,1 ./train_three_stage.sh
```

如果只希望在 `lsm >= 0.5` 的陆地像素上监督这三个变量：

```bash
MASK_OCEAN_FOR_LAND_VARS=true GPU_IDS=0,1 ./train_three_stage.sh
```

该 mask 只作用于 `sshf/slhf/vswl`，不会改变 `2t/10u/10v/msl` 和大气变量的
损失区域。

### 2.5 Checkpoint 与恢复

`train_three_stage.sh` 会自动查找每个阶段目录中编号最大的 `epoch_NNN.pt`：

- Stage 1/2 恢复模型、optimizer、scheduler 和 AMP scaler。
- Stage 2 第一次从 Stage 1 初始化时，会应用 Stage 2 学习率并重新开始 StepLR。
- Stage 3 额外保存每个 DDP rank 的 replay buffer 和随机数状态。
- 严格恢复 Stage 3 时，GPU 数量必须与保存 checkpoint 时一致。

如果修改了变量集合或模型结构，应使用新的 `RUN_ROOT`，不要恢复参数形状不同的
旧 checkpoint。更完整的参数说明见 [TRAINING_GUIDE.md](TRAINING_GUIDE.md)。

## 3. 评估

### 3.1 独立评估

以下命令对 Stage 3 最终 checkpoint 执行 20-step rollout 评估：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --device cuda \
  --ckpt weights/auroracp/stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt \
  --pretrained ckpt/aurora-0.25-pretrained.ckpt \
  --data_folder /path/to/weatherbench2_73var \
  --timestep 12 \
  --test_year 2020 2021 \
  --train_roll_step 1 \
  --roll_step 20
```

当 7 个 surface 变量配置正确时，评估会同时报告：

- 原始地表变量：`2t`、`10u`、`10v`、`msl`
- AuroraCP 陆面变量：`sshf`、`slhf`、`vswl`
- 13 层大气变量：`z`、`u`、`v`、`t`、`q`

评估指标为按变量累计的空间加权 RMSE。`--roll_step` 控制自回归预测步数。

### 3.2 保存预测结果

加入 `--save_pt` 可以保存每个样本的 rollout 输出：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --device cuda \
  --ckpt weights/auroracp/stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt \
  --pretrained ckpt/aurora-0.25-pretrained.ckpt \
  --data_folder /path/to/weatherbench2_73var \
  --timestep 12 \
  --test_year 2020 2021 \
  --train_roll_step 1 \
  --roll_step 20 \
  --save_pt \
  --save_pt_dir results/auroracp_epoch100_roll20
```

模型权重、训练 checkpoint、日志和预测结果均被 `.gitignore` 排除，不会提交到
Git 仓库。
