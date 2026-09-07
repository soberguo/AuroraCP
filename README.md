# Finetune Aurora

本项目基于 Microsoft Aurora 0.25-degree 预训练模型，实现面向 WeatherBench2
张量数据的三阶段微调流程：单步监督、两步 rollout 监督，以及带 replay buffer
的长时序 rollout 微调。训练入口支持单机单卡和单机多卡 DDP。

## 当前训练配置

当前代码直接加载 Aurora 预训练权重进行 LoRA 微调。`CustomAurora.forward()` 将
`hyp_x` 固定为 `None`，因此保留在代码中的 land hypernetwork 不参与实际前向。

默认参与模型输入、损失和评估的变量为：

- Surface：`2t`、`10u`、`10v`、`msl`
- Atmospheric：`z`、`u`、`v`、`t`、`q`，每个变量 13 个气压层

数据张量中的 `sshf`、`slhf`、`vswl` 当前不会送入 Aurora，也不会计算 loss。
Aurora 主干参数默认冻结，训练 LoRA、token embeddings 和输出 heads。代码仍保留
部分未参与前向的 hypernetwork/LoRA dense 参数，因此日志中的“可训练参数量”会
高于真正获得梯度的参数量，详见“已知限制”。

## 项目结构

```text
.
├── aurora/                    # 修改后的 Aurora 模型实现
├── data/weather_dataset.py    # WeatherBench2 tensor dataset
├── rollout_finetune/          # replay buffer 与严格恢复逻辑
├── small_model/               # 当前模型构造仍需导入的兼容代码
├── scripts/data_tools/        # 数据统计工具
├── args.py                    # 参数及变量配置
├── distributed_utils.py       # DDP 初始化与同步工具
├── main.py                    # Stage 1/2 训练入口
├── rollout_finetune.py        # Stage 3 训练入口
├── eval.py                    # 独立评估入口
├── model.py                   # Aurora 训练包装器
├── train_three_stage.sh       # 三阶段串联脚本
├── environment.yml            # Conda 环境
└── TRAINING_GUIDE.md          # 完整训练与恢复说明
```

模型权重、数据、checkpoint、日志和预测结果不提交到 Git。完整排除规则见
`.gitignore`。

## 环境安装

建议使用 Linux、NVIDIA GPU、CUDA 12.8 和 Python 3.10：

```bash
conda env create -f environment.yml
conda activate aurora
```

检查 PyTorch 和 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

`environment.yml` 是当前已验证环境的完整快照。如果目标机器无法解析其中的
CUDA wheel，可参考 `TRAINING_GUIDE.md` 分步安装 PyTorch 和其余依赖。

## 外部数据与权重

以下大文件不会包含在仓库中，需要在训练前自行准备：

```text
ckpt/aurora-0.25-pretrained.ckpt
ckpt/aurora-0.25-static.pickle
<DATA_FOLDER>/<YEAR>/*.pt
```

Aurora 权重应从 Microsoft Aurora 官方发布渠道获取。默认数据目录按年份组织：

```text
weatherbench2_73var/
├── 1979/
│   ├── 1979-0000.pt
│   ├── 1979-0006.pt
│   └── ...
├── 1980/
└── ...
```

每个 `.pt` 文件应为 `float32 [73, 120, 240]` tensor。文件名格式必须为
`YEAR-HOURS.pt`，其中 `HOURS` 是从该年 1 月 1 日 00:00 开始累计的小时数；原始
数据时间间隔为 6 小时。

代码当前使用的通道映射：

| 变量 | 通道 |
| --- | --- |
| `2t`, `10u`, `10v`, `msl` | 0, 1, 2, 3 |
| `z` | 4:17 |
| `u` | 17:30 |
| `v` | 30:43 |
| `t` | 43:56 |
| `q` | 56:69 |

张量的其他通道可以保留在数据文件中，但当前训练不会读取它们构造 Aurora Batch。

## 三阶段流程

| 阶段 | 入口 | 累计 epoch | 默认数据年份 | 学习率 |
| --- | --- | ---: | --- | --- |
| Stage 1 | `main.py` | 0 -> 20 | 1979-2017 | LoRA `1e-3`，其他 `1e-4` |
| Stage 2 | `main.py` | 20 -> 30 | 2010-2017 | 两组均为 `5e-5` |
| Stage 3 | `rollout_finetune.py` | 30 -> 100 | 2010-2017 | 两组均为 `5e-5` |

年份参数使用左闭右开区间，例如 `1979 2018` 表示读取 1979 至 2017 年。
Stage 3 默认在全局 Epoch 40 开始评估；保存 Epoch 56 时，curriculum 从最大
12 个 lead steps 切换为 20 个 lead steps。

### Dry run

先确认最终命令、数据路径和阶段衔接，不加载模型：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DRY_RUN=1 \
./train_three_stage.sh
```

### 单卡训练

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0 \
./train_three_stage.sh
```

### 双卡训练

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
./train_three_stage.sh
```

`STAGE*_BATCH_SIZE` 表示每张 GPU 的 batch size。有效 batch size 为：

```text
per-GPU batch size x GPU 数量 x accumulation_steps
```

### 自定义路径

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DATA_FOLDER=/path/to/weatherbench2_73var \
PRETRAINED=ckpt/aurora-0.25-pretrained.ckpt \
RUN_ROOT=weights/three_stage_run \
LOG_DIR=log/three_stage_run \
./train_three_stage.sh
```

不要复用模型结构或 optimizer 参数组不同的旧 checkpoint。开始新实验时应指定新的
`RUN_ROOT`。

## Checkpoint 与恢复

串联脚本会自动查找每个阶段目录中编号最大的 `epoch_NNN.pt`：

- Stage 1/2 恢复 model、optimizer、scheduler 和 AMP scaler。
- Stage 2 首次由 Stage 1 初始化时会按 Stage 2 配置重置学习率和 StepLR。
- 新版 Stage 3 还会为每个 rank 保存 replay buffer 和 RNG sidecar，实现严格恢复。
- Stage 3 严格恢复要求 GPU 数量（`world_size`）与保存时相同。

默认输出位于 `weights/`，日志位于 `log/`，二者均被 Git 忽略。更完整的恢复规则
见 `TRAINING_GUIDE.md`。

## 独立评估

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --device cuda \
  --ckpt weights/three_stage_run/stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt \
  --pretrained ckpt/aurora-0.25-pretrained.ckpt \
  --data_folder /path/to/weatherbench2_73var \
  --timestep 12 \
  --test_year 2020 2021 \
  --train_roll_step 1 \
  --roll_step 20
```

加入 `--save_pt --save_pt_dir <OUTPUT_DIR>` 可保存每个样本的 rollout 预测。

## 已知限制

- `--full_finetune` 和 `--train_aux` 已声明，但当前训练入口尚未实现相应分支。
- land hypernetwork 已退出前向，但其模块仍会被构造，部分参数仍会被加入 optimizer。
- 每个 LoRA 模块仍包含未使用的 dense `proj`/`fc_gate`，导致名义可训练参数量和
  checkpoint 体积偏大。
- Stage 1/2 仅由 rank 0 评估，其他 rank 在 NCCL barrier 等待；年度长评估可能超过
  NCCL watchdog timeout。Stage 3 已使用文件标记规避该问题。
- DDP 是数据并行，每张 GPU 都持有完整模型，不会降低单卡模型显存占用。

## 上游项目

Aurora 模型实现来源于 Microsoft Aurora，并在本项目中针对 LoRA、rollout 和训练
流程做了修改。使用或再分发前请同时遵守 Microsoft Aurora 上游项目的许可证与
模型权重条款。
