# Aurora_weather2 三阶段训练指南

本文档对应当前项目中的三阶段训练入口 `train_three_stage.sh`。默认训练流程为：

1. 阶段 1：单步监督训练，累计 epoch `0 -> 20`。
2. 阶段 2：两步 rollout 监督训练，累计 epoch `20 -> 30`。
3. 阶段 3：replay-buffer rollout 微调，累计 epoch `30 -> 100`。

三阶段均支持单卡和单机多卡 DDP。串联脚本会传递上一阶段的最终 checkpoint，并自动从已有的阶段内 checkpoint 继续训练。

## 1. Conda 环境配置

### 1.1 已验证环境

当前环境已验证的主要版本如下：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10.13 |
| PyTorch | 2.7.1+cu128 |
| CUDA runtime | 12.8 |
| NVIDIA driver | 580.173.02 |
| NumPy | 2.2.6 |
| timm | 0.6.13 |
| einops | 0.8.1 |
| SciPy | 1.15.3 |
| tqdm | 4.67.1 |

项目根目录已经提供完整的 `environment.yml`。优先使用该文件创建环境：

```bash
cd /path/to/Finetune_aurora
conda env create -n aurora -f environment.yml
conda activate aurora
```

如果环境已经存在，可更新已有环境：

```bash
conda env update -n aurora -f environment.yml
conda activate aurora
```

`environment.yml` 是完整环境快照，包含训练、数据处理和 WeatherBench2 工具依赖。如果在新机器上解析其中的 CUDA wheel 失败，可以先创建基础环境，再从 PyTorch 官方 CUDA 12.8 索引安装 PyTorch：

```bash
conda create -n aurora python=3.10.13 pip -y
conda activate aurora

python -m pip install \
  torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128

python -m pip install \
  numpy==2.2.6 scipy==1.15.3 pandas==2.3.0 xarray==2025.6.1 \
  timm==0.6.13 einops==0.8.1 tqdm==4.67.1 matplotlib==3.10.3 \
  huggingface-hub==0.33.0
```

训练前建议检查环境和 GPU：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

串联脚本默认使用服务器上的绝对 Python 路径。为了确保它使用当前激活的 Conda 环境，启动时建议始终传入：

```bash
export PYTHON_BIN="$(which python)"
```

## 2. 数据与预训练权重

### 2.1 数据目录

默认数据目录是：

```text
/sharefiles1/guoyixin/datasets/weatherbench2_73var
```

目录应按年份组织，每个文件是一个时刻的 PyTorch tensor：

```text
weatherbench2_73var/
├── 2010/
│   ├── 2010-0000.pt
│   ├── 2010-0006.pt
│   └── ...
├── 2011/
│   └── ...
└── 2020/
    └── ...
```

当前数据文件的 tensor 格式为 `float32 [73, 120, 240]`，文件名格式必须是 `YEAR-HOURS.pt`。代码使用文件名中的累计小时数恢复真实日期，并假设原始数据间隔为 6 小时。

年份参数采用左闭右开区间：

- 阶段 1 默认使用 `--train_year 1979 2018`，实际读取 1979 至 2017 年。
- 阶段 2 和阶段 3 默认使用 `--train_year 2010 2018`，实际读取 2010 至 2017 年。
- `--test_year 2020 2021` 实际只读取 2020 年。

如果阶段 1 还需包含 2018 年，应设置 `STAGE1_TRAIN_END_YEAR=2019`。

### 2.2 必需权重和静态变量

默认使用以下文件：

```text
ckpt/aurora-0.25-pretrained.ckpt
ckpt/aurora-0.25-static.pickle
```

训练前检查：

```bash
test -f ckpt/aurora-0.25-pretrained.ckpt
test -f ckpt/aurora-0.25-static.pickle
```

如需使用其他 Aurora 预训练权重，通过 `PRETRAINED` 指定；静态变量路径需要直接运行 Python 入口并传入 `--static_path`，或者在 `train_three_stage.sh` 的公共参数中配置。

## 3. 三阶段训练逻辑

| 阶段 | 入口 | epoch 范围 | 核心设置 | 输入 checkpoint | 默认输出 |
| --- | --- | --- | --- | --- | --- |
| 1 | `main.py` | 0-20 | `train_roll_step=1`，单步监督 | Aurora pretrained | `stage1_single_step/epoch_020.pt` |
| 2 | `main.py` | 20-30 | `train_roll_step=2`，两步 rollout 联合损失 | 阶段 1 epoch 20 | `stage2_two_step/epoch_030.pt` |
| 3 | `rollout_finetune.py` | 30-100 | replay buffer、动态目标、预测状态回灌 | 阶段 2 epoch 30 | `stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt` |

这里的 `--epochs` 是累计结束 epoch，不是该阶段额外训练的 epoch 数。例如阶段 2 从 checkpoint 中读取 `epoch=20`，然后运行至 `--epochs 30`，因此实际训练 10 个 epoch。

阶段 3 默认 curriculum 为：

- 生成 `epoch_031.pt` 至 `epoch_055.pt` 时，最大 lead step 为 12。
- 从 `epoch_056.pt` 开始，最大 lead step 为 20。
- 每个 DDP rank 维护独立 replay buffer，模型梯度由 DDP 同步。

默认可训练参数包括 Aurora LoRA、surface/atmos token embeddings、输出 heads 和 hypernetwork，其余 Aurora 参数被冻结。

默认 loss 是加权 MAE。当前配置仅将 `2t`、`10u`、`10v`、`msl` 送入 Aurora；`sshf`、`slhf`、`vswl` 不参与模型输入、loss 或评估。因此 `MASK_OCEAN_FOR_LAND_VARS` 当前为兼容参数，不会改变实际 loss。

## 4. 启动训练

所有命令均应在项目根目录执行：

```bash
cd /path/to/Finetune_aurora
conda activate aurora
```

### 4.1 先做 dry-run

dry-run 只打印三阶段完整命令，不加载模型和数据：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DRY_RUN=1 \
./train_three_stage.sh
```

建议每次修改数据路径、阶段 epoch 或输出目录后先执行一次 dry-run。

### 4.2 单卡训练

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0 \
./train_three_stage.sh
```

旧的单卡写法 `GPU_ID=0` 仍兼容，但新命令建议统一使用 `GPU_IDS`。

### 4.3 多卡 DDP 训练

双卡：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
./train_three_stage.sh
```

四卡：

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1,2,3 \
./train_three_stage.sh
```

当 `GPU_IDS` 包含多个设备时，脚本自动使用：

```text
python -m torch.distributed.run --standalone --nproc_per_node=<GPU 数量>
```

训练数据由 `DistributedSampler` 分片。checkpoint、日志和训练内评估只由 global rank 0 写入，其他 rank 在同步点等待。

### 4.4 开启陆地变量海洋 mask

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
MASK_OCEAN_FOR_LAND_VARS=true \
./train_three_stage.sh
```

### 4.5 完整自定义示例

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DATA_FOLDER=/sharefiles1/guoyixin/datasets/weatherbench2_73var \
PRETRAINED=ckpt/aurora-0.25-pretrained.ckpt \
RUN_ROOT=weights/three_stage_73var_12h \
LOG_DIR=log/three_stage_73var_12h \
TIMESTEP=12 \
TRAIN_START_YEAR=2010 \
TRAIN_END_YEAR=2018 \
TEST_START_YEAR=2020 \
TEST_END_YEAR=2021 \
STAGE1_END_EPOCH=20 \
STAGE2_END_EPOCH=30 \
STAGE3_END_EPOCH=100 \
STAGE1_BATCH_SIZE=4 \
STAGE2_BATCH_SIZE=4 \
STAGE3_BATCH_SIZE=4 \
MASK_OCEAN_FOR_LAND_VARS=true \
SAVE_EVERY=1 \
EVAL_EVERY=1 \
./train_three_stage.sh
```

## 5. Batch size 与显存

`STAGE*_BATCH_SIZE` 均表示每个 GPU 的 batch size。默认值为：

| 阶段 | 每卡 batch size | 双卡全局 batch size | 四卡全局 batch size |
| --- | ---: | ---: | ---: |
| 1 | 4 | 8 | 16 |
| 2 | 4 | 8 | 16 |
| 3 | 4 | 8 | 16 |

计算方式为：

```text
effective batch size = per-GPU batch size x GPU 数量 x accumulation_steps
```

多卡训练若希望保持与单卡相同的全局 batch size，应相应减小每卡 batch size。显存不足时，优先降低对应阶段的 `STAGE*_BATCH_SIZE`。阶段 3 还要求 `replay_buffer_capacity >= batchsize`。

## 6. Checkpoint、续训与跳过规则

默认输出结构：

```text
weights/three_stage_73var_12h/
├── stage1_single_step/
│   ├── epoch_001.pt
│   └── epoch_020.pt
├── stage2_two_step/
│   ├── epoch_021.pt
│   └── epoch_030.pt
└── stage3_replay_rollout_from_epoch030_eval040/
    ├── epoch_031.pt
    ├── rank_states/
    │   ├── epoch_099_rank_000.pt
    │   ├── epoch_099_rank_001.pt
    │   ├── epoch_100_rank_000.pt
    │   └── epoch_100_rank_001.pt
    └── epoch_100.pt

log/three_stage_73var_12h/
├── stage1_single_step.log
├── stage2_two_step.log
└── stage3_replay_rollout_from_epoch030_eval040.log
```

串联脚本的恢复规则如下：

1. 如果某阶段的最终 checkpoint 已存在，该阶段直接跳过。
2. 如果最终 checkpoint 不存在，但阶段目录内存在 `epoch_NNN.pt`，从编号最大的 checkpoint 继续。
3. 阶段 2 没有自身 checkpoint 时，从阶段 1 的最终 checkpoint 初始化。
4. 阶段 3 没有自身 checkpoint 时，从阶段 2 的最终 checkpoint 初始化。
5. 所有 checkpoint 仅由 rank 0 保存，因此多卡不会重复写文件。

阶段 1 和阶段 2 在各自阶段内续训时恢复模型、optimizer、scheduler 和 AMP scaler。首次从阶段 1 进入阶段 2 时会保留 optimizer 动量，但将两个参数组的学习率重置为阶段 2 配置，并重新开始 StepLR 计数，防止阶段 1 的学习率和 scheduler 状态泄漏到阶段 2。

新版阶段 3 checkpoint 会恢复模型、optimizer、scheduler、AMP scaler、`global_step`，以及每个 rank 独立的 replay buffer 和 Python/NumPy/PyTorch/CUDA RNG 状态。恢复时必须保持相同的 `world_size`、rank 编号和 replay buffer capacity；缺失或不匹配的 sidecar 会直接报错，不会静默退化为非连续续训。rank state 采用原子写入，默认每个 rank 保留最近 2 个 epoch。

`epoch_056.pt` 及更早的旧格式 checkpoint 没有 replay/RNG sidecar。从此类 checkpoint 恢复时，已有 optimizer、scheduler、AMP scaler 和 `global_step` 会正常恢复，但 replay buffer 会重新冷启动，因此第一次恢复仍无法做到严格数值连续；产生首个新版 checkpoint 后，后续中断即可恢复完整训练状态。

当前 buffer capacity 为 400 时，每个 rank 的 sidecar 约为 6.3 GiB。两卡且每个 rank 保留 2 份时，额外占用约 25 GiB。

如需完全重新开始实验，推荐指定一个新的 `RUN_ROOT`，避免误用旧 checkpoint：

```bash
RUN_ROOT=weights/three_stage_new_run GPU_IDS=0,1 ./train_three_stage.sh
```

## 7. 训练内评估与独立评估

`EVAL_EVERY=1` 表示阶段 1 和阶段 2 每个 epoch 评估一次；设为 `0` 可关闭：

```bash
EVAL_EVERY=0 GPU_IDS=0,1 ./train_three_stage.sh
```

阶段 3 默认从全局 Epoch 40 开始、每个 epoch 评估一次，即 `STAGE3_EVAL_START_EPOCH=40`、`STAGE3_EVAL_EVERY=1`。完整的 20-step 年度评估只在 rank 0 执行；其他 rank 通过文件标记等待，不会在评估期间挂起 NCCL collective，因此不再触发默认 10 分钟的 NCCL watchdog 超时。评估前后的 RNG 状态保持不变，不会扰动后续训练随机序列。

最终 checkpoint 可单独评估：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --device cuda \
  --ckpt weights/three_stage_73var_12h/stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt \
  --pretrained ckpt/aurora-0.25-pretrained.ckpt \
  --data_folder /sharefiles1/guoyixin/datasets/weatherbench2_73var \
  --timestep 12 \
  --test_year 2020 2021 \
  --train_roll_step 1 \
  --roll_step 20
```

如需保存每个样本的 rollout 结果：

```bash
CUDA_VISIBLE_DEVICES=0 python eval.py \
  --device cuda \
  --ckpt weights/three_stage_73var_12h/stage3_replay_rollout_from_epoch030_eval040/epoch_100.pt \
  --pretrained ckpt/aurora-0.25-pretrained.ckpt \
  --data_folder /sharefiles1/guoyixin/datasets/weatherbench2_73var \
  --timestep 12 \
  --test_year 2020 2021 \
  --train_roll_step 1 \
  --roll_step 20 \
  --save_pt \
  --save_pt_dir results/stage3_epoch100_roll20
```

当 `--save_pt` 已开启且目标 `.pt` 已存在时，`eval.py` 会读取已有预测并跳过对应模型 rollout，不会覆盖该结果文件。

## 8. 常用配置项

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PYTHON_BIN` | `/home/guoyixin/miniconda3/envs/aurora/bin/python` | Python 可执行文件 |
| `GPU_IDS` | `0,1` | 逗号分隔的 GPU 编号 |
| `DATA_FOLDER` | `/sharefiles1/guoyixin/datasets/weatherbench2_73var` | 数据根目录 |
| `PRETRAINED` | `ckpt/aurora-0.25-pretrained.ckpt` | Aurora 预训练权重 |
| `RUN_ROOT` | `weights/three_stage_73var_12h` | 三阶段 checkpoint 根目录 |
| `LOG_DIR` | `log/three_stage_73var_12h` | 日志目录 |
| `TIMESTEP` | `12` | 时间步，支持 6 或 12 小时 |
| `STAGE1_TRAIN_START_YEAR` | `1979` | 阶段 1 训练起始年份（包含） |
| `STAGE1_TRAIN_END_YEAR` | `2018` | 阶段 1 训练结束边界（不包含，因此最后一年为 2017） |
| `STAGE2_TRAIN_START_YEAR` | `2010` | 阶段 2 训练起始年份（包含） |
| `STAGE2_TRAIN_END_YEAR` | `2018` | 阶段 2 训练结束边界（不包含） |
| `STAGE1_LR1` | `1e-3` | 阶段 1 的 LoRA 参数学习率 |
| `STAGE1_LR2` | `1e-4` | 阶段 1 的 embeddings、heads、hypernetwork 学习率 |
| `STAGE2_LR1` | `5e-5` | 阶段 2 的 LoRA 参数学习率 |
| `STAGE2_LR2` | `5e-5` | 阶段 2 的 embeddings、heads、hypernetwork 学习率 |
| `LR1` | `5e-5` | 阶段 2 学习率的兼容默认值，以及阶段 3 的 LoRA 参数学习率 |
| `LR2` | `5e-5` | 阶段 2 学习率的兼容默认值，以及阶段 3 的其他可训练参数学习率 |
| `MASK_OCEAN_FOR_LAND_VARS` | `false` | land vars 重新加入训练时，用于仅在陆地计算对应 loss；当前配置无实际影响 |
| `SAVE_EVERY` | `1` | checkpoint 保存间隔；串联训练必须设为正整数 |
| `EVAL_EVERY` | `1` | 评估间隔，0 表示关闭训练内评估 |
| `STAGE3_EVAL_EVERY` | `1` | 阶段 3 从评估起始 epoch 后的评估间隔 |
| `STAGE3_EVAL_START_EPOCH` | `40` | 阶段 3 首次评估的全局 epoch（包含） |
| `STAGE3_DIR` | `weights/three_stage_73var_12h/stage3_replay_rollout_from_epoch030_eval040` | 新 Stage 3 checkpoint 目录 |
| `STAGE3_LOG_FILE` | `stage3_replay_rollout_from_epoch030_eval040.log` | 新 Stage 3 日志文件名 |
| `REPLAY_STATE_KEEP` | `2` | 每个 rank 保留的 replay/RNG sidecar 数量 |
| `DRY_RUN` | `0` | 设为 1 时仅打印命令 |

不要在三阶段串联训练中设置 `SAVE_EVERY=0`。Python 入口会停止保存 checkpoint，而串联脚本要求每个阶段生成最终 checkpoint，因而会在阶段结束后报错。

## 9. 常见问题

### 找不到 Python

设置当前 Conda 环境的解释器：

```bash
PYTHON_BIN="$(which python)" GPU_IDS=0 ./train_three_stage.sh
```

### 数据集显示 0 个文件

检查 `DATA_FOLDER` 下是否存在年份子目录，并注意结束年份是开区间。文件必须以 `.pt` 结尾。

### CUDA out of memory

降低对应阶段每卡 batch size，例如：

```bash
STAGE1_BATCH_SIZE=1 STAGE2_BATCH_SIZE=1 STAGE3_BATCH_SIZE=1 GPU_IDS=0,1 ./train_three_stage.sh
```

### 多卡 NCCL 初始化失败或卡住

确认以下条件：

1. `GPU_IDS` 中设备编号有效且不重复。
2. `nvidia-smi` 能看到所有目标 GPU。
3. 所有 rank 均可读取数据目录、预训练权重和静态变量。
4. 没有手动设置冲突的 `RANK`、`LOCAL_RANK` 或 `WORLD_SIZE`。

需要更多 NCCL 日志时可临时设置：

```bash
NCCL_DEBUG=INFO GPU_IDS=0,1 ./train_three_stage.sh
```

### 只想检查最终命令

```bash
DRY_RUN=1 GPU_IDS=0,1 ./train_three_stage.sh
```

该命令也可以确认阶段边界、checkpoint 传递、数据路径和 DDP 进程数是否符合预期。
