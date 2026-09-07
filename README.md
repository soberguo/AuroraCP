# AuroraCP

**AuroraCP** 是一个基于 Aurora 预训练模型的多变量天气微调方法。它在 Aurora
原有的地表变量和多层大气变量之外，进一步学习三个与陆面能量和水分过程相关的
变量：

- `sshf`：地表感热通量（surface sensible heat flux）
- `slhf`：地表潜热通量（surface latent heat flux）
- `vswl`：土壤体积含水量（volumetric soil water layer）

AuroraCP 通过单步监督、两步 rollout 监督和 replay-buffer rollout 三个阶段逐步提高长期预测
稳定性。

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
`PRETRAINED` 指定其他预训练权重。

### 1.3 数据目录与文件格式

数据按年份存放，每个 `.pt` 文件对应一个时刻：

```text
weatherbench2_72var/
├── 1979/
│   ├── 1979-0000.pt
│   ├── 1979-0006.pt
│   └── ...
├── 1980/
└── ...
```

每个文件应为 `float32 [72, 120, 240]` tensor。文件名采用 `YEAR-HOURS.pt`，
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



## 2. 训练

### 2.1 三阶段训练流程



| 阶段 | 训练入口 | 累计 epoch | 默认训练年份 | 训练方式 | 默认学习率 |
| --- | --- | ---: | --- | --- | --- |
| Stage 1 | `main.py` | 0 -> 20 | 1979-2017 | 单步监督 | LoRA `1e-3`，embeddings/heads `1e-4` |
| Stage 2 | `main.py` | 20 -> 30 | 2010-2017 | 两步 rollout 联合监督 | 两组均为 `5e-5` |
| Stage 3 | `rollout_finetune.py` | 30 -> 100 | 2010-2017 | replay-buffer rollout | 两组均为 `5e-5` |





### 2.2 训练



```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DATA_FOLDER=/path/to/weatherbench2_72var \
RUN_ROOT=weights/auroracp \
LOG_DIR=log/auroracp \
./train_three_stage.sh
```




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

评估指标为按变量累计的空间加权 RMSE。

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


