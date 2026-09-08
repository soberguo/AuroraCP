# AuroraCP

**AuroraCP** is a multivariable weather fine-tuning method based on the pretrained Aurora model. In addition to Aurora's original surface variables and multilevel atmospheric variables, it learns three variables related to land-surface energy and water processes:

- `sshf`: surface sensible heat flux
- `slhf`: surface latent heat flux
- `vswl`: volumetric soil water layer

AuroraCP progressively improves long-range forecast stability through three stages: single-step supervision, two-step rollout supervision, and replay-buffer rollout.

## 1. Data Preparation

### 1.1 Environment Setup

The recommended environment is Linux, Python 3.10, PyTorch 2.7.1, and CUDA 12.8:

```bash
conda env create -f environment.yml
conda activate aurora
```

Check PyTorch and GPU availability:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi
```

### 1.2 Pretrained Weights and Static Variables

Prepare the pretrained Aurora weights and static geographic variables before training:

```text
ckpt/aurora-0.25-pretrained.ckpt
ckpt/aurora-0.25-static.pickle
```

These large files are not included in the Git repository. By default, the training scripts load them from the locations above. You can also specify a different pretrained checkpoint with `PRETRAINED`.

### 1.3 Data Directory and File Format

Data are organized by year, with each `.pt` file corresponding to one time step:

```text
weatherbench2_72var/
├── 1979/
│   ├── 1979-0000.pt
│   ├── 1979-0006.pt
│   └── ...
├── 1980/
└── ...
```

Each file must be a `float32 [72, 120, 240]` tensor. File names follow the `YEAR-HOURS.pt` convention, where `HOURS` is the number of hours elapsed since 00:00 on January 1 of that year. The source data have a 6-hour sampling interval. Training can use either a 6-hour or 12-hour forecast step.

AuroraCP uses the following channel mapping:

| Type | Variables | Data channels |
| --- | --- | --- |
| Original surface variables | `2t`, `10u`, `10v`, `msl` | 0, 1, 2, 3 |
| Geopotential | `z`, 13 pressure levels | 4:17 |
| Zonal wind | `u`, 13 pressure levels | 17:30 |
| Meridional wind | `v`, 13 pressure levels | 30:43 |
| Temperature | `t`, 13 pressure levels | 43:56 |
| Specific humidity | `q`, 13 pressure levels | 56:69 |
| AuroraCP land variables | `sshf`, `slhf`, `vswl` | 69, 70, 71 |

The default pressure levels are:

```text
1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50 hPa
```

### 1.4 Enabling the Three AuroraCP Land Variables

Before training `sshf/slhf/vswl`, verify that [args.py](args.py) uses the seven-surface-variable configuration:

```python
DEFAULT_MAIN_SURF_VARS = ["2t", "10u", "10v", "msl"]
DEFAULT_LAND_VARS = ["sshf", "slhf", "vswl"]

DEFAULT_SURF_LOSS_WEIGHTS = [3.5, 0.77, 0.66, 1.6, 0.5, 0.5, 0.5]

args.surf_vars = args.main_surf_vars + args.land_vars
args.surf_channels = {**args.main_surf_channels, **args.land_channels}
```

## 2. Training

### 2.1 Three-Stage Training Workflow

| Stage | Entry point | Epoch range | Default training years | Training method | Default learning rate |
| --- | --- | ---: | --- | --- | --- |
| Stage 1 | `main.py` | 0 -> 20 | 1979-2017 | Single-step supervision | LoRA `1e-3`, embeddings/heads `1e-4` |
| Stage 2 | `main.py` | 20 -> 30 | 2010-2017 | Joint two-step rollout supervision | `5e-5` for both groups |
| Stage 3 | `rollout_finetune.py` | 30 -> 100 | 2010-2017 | Replay-buffer rollout | `5e-5` for both groups |

### 2.2 Training

```bash
PYTHON_BIN="$(which python)" \
GPU_IDS=0,1 \
DATA_FOLDER=/path/to/weatherbench2_72var \
RUN_ROOT=weights/auroracp \
LOG_DIR=log/auroracp \
./train_three_stage.sh
```

## 3. Evaluation

### 3.1 Standalone Evaluation

The following command performs a 20-step rollout evaluation on the final Stage 3 checkpoint:

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

When the seven-surface-variable configuration is correct, the evaluation reports results for:

- Original surface variables: `2t`, `10u`, `10v`, `msl`
- AuroraCP land variables: `sshf`, `slhf`, `vswl`
- Atmospheric variables at 13 levels: `z`, `u`, `v`, `t`, `q`

The evaluation metric is spatially weighted RMSE aggregated by variable.

### 3.2 Saving Prediction Results

Add `--save_pt` to save the rollout output for each sample:

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
