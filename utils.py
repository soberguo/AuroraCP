from datetime import datetime
import torch
import pickle
import torch.nn.functional as F
import json
import numpy as np
from aurora import AuroraPretrained, Batch, Metadata,rollout
import torch.nn as nn
import math

def poincare_distance(x, y, c=1.0):
    """
    计算庞加莱球模型中两点间的双曲距离
    Args:
        x, y: 输入张量，形状相同
        c: 曲率参数，默认 c=1 (曲率 K=-c)
    Returns:
        双曲距离
    """
    # ||x - y||^2
    diff_norm_sq = torch.sum((x - y) ** 2, dim=list(range(1, x.dim())))
    # 1 - c*||x||^2
    x_norm_sq = torch.sum(x ** 2, dim=list(range(1, x.dim())))
    y_norm_sq = torch.sum(y ** 2, dim=list(range(1, y.dim())))

    denom = (1 - c * x_norm_sq) * (1 - c * y_norm_sq)
    # arcosh formula: arcosh(1 + 2c*||x-y||^2 / ((1-c||x||^2)(1-c||y||^2)))
    # 使用 acosh 代替 arcosh (PyTorch 1.9+ 支持)
    inner = 1 + 2 * c * diff_norm_sq / (denom + 1e-8)  # 添加 small epsilon 防止除零
    inner = torch.clamp(inner, min=1.0)  # acosh 要求输入 >= 1
    dist = (1 / math.sqrt(c)) * torch.acosh(inner)
    return dist


def exp_map_0(v, c=1.0):
    """
    指数映射：将切空间原点的向量映射到庞加莱球
    exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||)

    Args:
        v: 输入向量 (..., D)
        c: 曲率参数
    Returns:
        庞加莱球上的点 (..., D)
    """
    sqrt_c = math.sqrt(c)
    norm = torch.norm(v, p=2, dim=-1, keepdim=True).clamp(min=1e-10)
    return torch.tanh(sqrt_c * norm / 2) * v / (sqrt_c * norm)


class HyperbolicEmbeddingLoss(nn.Module):
    """
    双曲 embedding 损失模块：
    1. 每个变量独立 embedding 到隐空间
    2. 通过 exp_0 映射到庞加莱球
    3. 每个变量单独计算双曲距离，再加权求和
    """

    def __init__(self, embed_dim=64, hyperbolic_c=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hyperbolic_c = hyperbolic_c

        # Surface variables: 每个变量独立 embedding
        # 2t, 10u, 10v, msl: shape (B, 1, H, W) -> unsqueeze后 (B, 1, H, W)
        # sshf, slhf, vswl: shape (B, 1, H, W) -> unsqueeze后 (B, 1, H, W)
        self.surf_embeds = nn.ModuleDict({
            '2t': nn.Linear(1, embed_dim),
            '10u': nn.Linear(1, embed_dim),
            '10v': nn.Linear(1, embed_dim),
            'msl': nn.Linear(1, embed_dim),
            'sshf': nn.Linear(1, embed_dim),
            'slhf': nn.Linear(1, embed_dim),
            'vswl': nn.Linear(1, embed_dim),
        })

        # Atmospheric variables: z, u, v, t, q 各有 13 个 level
        self.atmos_embeds = nn.ModuleDict({
            'z': nn.Linear(13, embed_dim),
            'u': nn.Linear(13, embed_dim),
            'v': nn.Linear(13, embed_dim),
            't': nn.Linear(13, embed_dim),
            'q': nn.Linear(13, embed_dim),
        })

    def forward(self, pred_batch, tgt_batch):
        """
        Args:
            pred_batch: 预测的 Aurora Batch
            tgt_batch: 目标的 Aurora Batch
        Returns:
            surf_dist, atmos_dist: 分别为表面和大气的双曲距离
        """
        total_surf_dist = 0.0
        total_atmos_dist = 0.0
        surf_weight_sum = 0.0
        atmos_weight_sum = 0.0

        # ===== Surface Variables =====
        surf_weights = {
            '2t': 3.5, '10u': 0.77, '10v': 0.66, 'msl': 1.6,
            'sshf': 0.5, 'slhf': 0.5, 'vswl': 0.5
        }

        for name, weight in surf_weights.items():
            # pred: (B, 1, H, W) -> (B, H, W, 1)
            pred = pred_batch.surf_vars[name][:, 0].unsqueeze(-1)
            # tgt: (B, H, W) -> (B, H, W, 1)
            tgt = tgt_batch.surf_vars[name].unsqueeze(-1)

            # Embedding: (B, H, W, 1) -> (B, H, W, embed_dim)
            pred_emb = self.surf_embeds[name](pred)
            tgt_emb = self.surf_embeds[name](tgt)

            # exp_0 mapping to Poincaré ball
            pred_hyp = exp_map_0(pred_emb, self.hyperbolic_c)
            tgt_hyp = exp_map_0(tgt_emb, self.hyperbolic_c)

            # 双曲距离
            dist = poincare_distance(pred_hyp, tgt_hyp, self.hyperbolic_c)
            total_surf_dist += weight * dist
            surf_weight_sum += weight

        # ===== Atmospheric Variables =====
        atmos_weights = {'z': 3.5, 'u': 0.87, 'v': 0.6, 't': 1.7, 'q': 0.8}

        for name, weight in atmos_weights.items():
            # pred: (B, 13, H, W) -> (B, H, W, 13)
            pred = pred_batch.atmos_vars[name][:, 0].permute(0, 2, 3, 1)
            # tgt: (B, 13, H, W) -> (B, H, W, 13)
            tgt = tgt_batch.atmos_vars[name].permute(0, 2, 3, 1)

            # Embedding: (B, H, W, 13) -> (B, H, W, embed_dim)
            pred_emb = self.atmos_embeds[name](pred)
            tgt_emb = self.atmos_embeds[name](tgt)

            # exp_0 mapping to Poincaré ball
            pred_hyp = exp_map_0(pred_emb, self.hyperbolic_c)
            tgt_hyp = exp_map_0(tgt_emb, self.hyperbolic_c)

            # 双曲距离
            dist = poincare_distance(pred_hyp, tgt_hyp, self.hyperbolic_c)
            total_atmos_dist += weight * dist
            atmos_weight_sum += weight

        # 加权平均
        surf_dist = total_surf_dist / surf_weight_sum
        atmos_dist = total_atmos_dist / atmos_weight_sum

        return surf_dist, atmos_dist


# 全局实例，在 compute_loss_hyperbolic_embedding 时使用
_hyperbolic_loss_module = None


def get_hyperbolic_loss_module(embed_dim=64, hyperbolic_c=1.0):
    """获取或创建全局双曲 embedding 损失模块"""
    global _hyperbolic_loss_module
    if _hyperbolic_loss_module is None or _hyperbolic_loss_module.hyperbolic_c != hyperbolic_c:
        _hyperbolic_loss_module = HyperbolicEmbeddingLoss(embed_dim, hyperbolic_c)
    return _hyperbolic_loss_module


def _slice_tensor_by_channel(tensor, channel):
    if isinstance(channel, int):
        return tensor[:, channel]
    start, end = channel
    return tensor[:, start:end]


def build_batch_from_tensor(tensor, time, args, history_tensor=None):
    surf_vars = {}
    for name in args.surf_vars:
        current = _slice_tensor_by_channel(tensor, args.surf_channels[name])
        if history_tensor is None:
            surf_vars[name] = current
        else:
            previous = _slice_tensor_by_channel(history_tensor, args.surf_channels[name])
            surf_vars[name] = torch.stack([previous, current], dim=1)

    atmos_vars = {}
    for name in args.atmos_vars:
        current = _slice_tensor_by_channel(tensor, args.atmos_channels[name])
        if history_tensor is None:
            atmos_vars[name] = current
        else:
            previous = _slice_tensor_by_channel(history_tensor, args.atmos_channels[name])
            atmos_vars[name] = torch.stack([previous, current], dim=1)

    return Batch(
        surf_vars=surf_vars,
        static_vars={"lsm": args.static_vars_lsm, "z": args.static_vars_z, "slt": args.static_vars_slt},
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=args.lat_tensor,
            lon=args.lon_tensor,
            time=time,
            atmos_levels=args.atmos_levels,
        ),
    )


def _masked_spatial_mean(elementwise_loss, spatial_mask):
    if spatial_mask.ndim != 2:
        raise ValueError(f"land-sea mask 必须是 [H, W]，当前 shape={tuple(spatial_mask.shape)}")
    if tuple(elementwise_loss.shape[-2:]) != tuple(spatial_mask.shape):
        raise ValueError(
            f"land-sea mask 与 loss 空间尺寸不匹配: "
            f"mask={tuple(spatial_mask.shape)}, loss={tuple(elementwise_loss.shape)}"
        )

    mask = (spatial_mask >= 0.5).to(device=elementwise_loss.device, dtype=torch.float32)
    valid_pixels = mask.sum()
    mask = mask.view(*([1] * (elementwise_loss.ndim - 2)), *mask.shape)
    leading_size = elementwise_loss.numel() // spatial_mask.numel()
    return (elementwise_loss.float() * mask).sum() / (valid_pixels * leading_size)


def compute_loss(prediction, new_target, args=None):
    """
    计算损失函数
    Args:
        prediction: 预测值 (Aurora Batch)
        new_target: 目标值 (Aurora Batch)
        use_hyperbolic_embedding: 是否使用双曲 embedding 损失，默认 False (使用MAE)
        hyperbolic_c: 庞加莱球曲率参数，默认 c=1
        embed_dim: embedding 维度，默认 64
    """

    if args is None:
        loss_fn = nn.L1Loss(reduction="none")
        surf_vars = ["2t", "10u", "10v", "msl", "sshf", "slhf", "vswl"]
        land_vars = {"sshf", "slhf", "vswl"}
        atmos_vars = ["z", "u", "v", "t", "q"]
        surf_weights = [3.5, 0.77, 0.66, 1.6, 0.5, 0.5, 0.5]
        atmos_weights = [3.5, 0.87, 0.6, 1.7, 0.8]
        surf_scale = 0.25
        mask_ocean_for_land_vars = False
        land_sea_mask = None
    else:
        loss_fn = nn.MSELoss(reduction="none") if args.loss_type == "mse" else nn.L1Loss(reduction="none")
        surf_vars = args.surf_vars
        land_vars = set(args.land_vars)
        atmos_vars = args.atmos_vars
        surf_weights = args.surf_loss_weights
        atmos_weights = args.atmos_loss_weights
        surf_scale = args.surf_loss_scale
        mask_ocean_for_land_vars = getattr(args, "mask_ocean_for_land_vars", False)
        land_sea_mask = getattr(args, "static_vars_lsm", None)
        if mask_ocean_for_land_vars and land_sea_mask is None:
            raise ValueError("--mask_ocean_for_land_vars=true 时 args.static_vars_lsm 不能为空")

    surf_loss = 0.0
    for name, weight in zip(surf_vars, surf_weights):
        if name not in prediction.surf_vars:
            raise KeyError(f"prediction.surf_vars 缺少变量: {name}")
        if name not in new_target.surf_vars:
            raise KeyError(f"new_target.surf_vars 缺少变量: {name}")
        elementwise_loss = loss_fn(prediction.surf_vars[name][:, 0], new_target.surf_vars[name])
        if mask_ocean_for_land_vars and name in land_vars:
            variable_loss = _masked_spatial_mean(elementwise_loss, land_sea_mask)
        else:
            variable_loss = elementwise_loss.mean()
        surf_loss = surf_loss + weight * variable_loss

    atmos_loss = 0.0
    for name, weight in zip(atmos_vars, atmos_weights):
        if name not in prediction.atmos_vars:
            raise KeyError(f"prediction.atmos_vars 缺少变量: {name}")
        if name not in new_target.atmos_vars:
            raise KeyError(f"new_target.atmos_vars 缺少变量: {name}")
        atmos_loss = atmos_loss + weight * loss_fn(
            prediction.atmos_vars[name][:, 0], new_target.atmos_vars[name]
        ).mean()

    return surf_scale * surf_loss + atmos_loss


def calculate_batch_rmse_sum(pred, target):
    """
    自动处理 Surface (B, Lat, Lon) 和 Atmos (B, Level, Lat, Lon)
    """
    num_lat = pred.shape[-2]
    lat_degrees = torch.linspace(90, -90, num_lat + 1, device=pred.device)[:-1]
    weights = torch.cos(torch.deg2rad(lat_degrees))
    weights = weights / weights.mean()  # 归一化权重
    diff_sq = (pred - target.to(pred.device)) ** 2

    # 维度广播权重
    if pred.ndim == 3: # Surface: [B, H, W]
        w = weights.view(1, num_lat, 1)
        # 严格按照公式：
        # 1. 先在 H, W 维度上进行加权平均 (Inside the square root)
        # sum(w * diff^2) / (H*W)  <-- 注意公式里 w(i) 求和是 H，所以除以 H*W 等价于 mean
        # 实际上，因为 weights.mean()=1, 所以 mean(w * diff^2) 就是加权平均
        weighted_mse = torch.mean(diff_sq * w, dim=(1, 2)) # 结果 shape: [Batch]

    elif pred.ndim == 4: # Atmos: [B, L, H, W]
        w = weights.view(1, 1, num_lat, 1)
        # 大气变量通常对 H, W 求平均。Level 维度怎么处理？
        # Aurora 论文中对于大气变量通常是报告 specific level (如 z500) 或 level-averaged。
        # 这里我们假设是对 H, W 求 RMSE，保留 Level 或对 Level 也平均。
        # 如果要严格对标 "Spatial RMSE"，通常是把 Level 视为独立变量或一起平均。
        # 这里采取对 (2, 3) 维度求平均，结果 shape [Batch, Level] -> mean -> [Batch]
        weighted_mse = torch.mean(diff_sq * w, dim=(2, 3))
        # 如果你要计算所有层的平均 RMSE，再对 dim=1 平均
        weighted_mse = torch.mean(weighted_mse, dim=1) # shape: [Batch]

    else:
        return 0.0

    # 2. 开根号 (Square root)
    batch_rmse = torch.sqrt(weighted_mse) # shape: [Batch]

    # 3. 返回 Batch 内 RMSE 的总和 (Sum over t in current batch)
    return batch_rmse.sum().item()

def fix_tensor(t):
    """内部函数：将 NaN/Inf 替换为该通道的均值"""
    # 1. 先把 Inf 转为 NaN，统一处理
    t[torch.isinf(t)] = float('nan')

    # 2. 找到所有 NaN 的位置
    if torch.isnan(t).any():
        nan_mask = torch.isnan(t)
        # 遍历 Batch 维度
        for b in range(t.shape[0]):
            # 遍历 Channel 维度
            for c in range(t.shape[1]):
                # 取出当前通道的 Mask
                ch_mask = nan_mask[b, c]
                if ch_mask.any():
                    # 取出该通道数据
                    ch_data = t[b, c]
                    # 计算非 NaN 值的均值
                    valid_mean = torch.nanmean(ch_data)

                    # 如果整个通道全是 NaN，valid_mean 也会是 NaN，这时候填 0
                    if torch.isnan(valid_mean):
                        valid_mean = 0.0

                    # 执行替换
                    t[b, c][ch_mask] = valid_mean
    return t


def construct_batch(target,time,args):
    return build_batch_from_tensor(target, time, args)


def hours_to_datetime(filename):
    year= int(filename.split('-')[0])  # 从文件名中提取年份
    hours = int(filename.split('-')[1])  # 从文件名中提取小时数
    # 检查年份是否是闰年
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

    # 每个月的天数（平年）
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap:
        month_days[1] = 29  # 闰年2月有29天

    # 计算天数和剩余的小时
    total_days = hours // 24  # 从0开始（0 = 1月1日）
    remaining_hour = hours % 24  # 0~23

    # 计算月份和日期
    current_day = total_days + 1  # 转换为1-based（1月1日=第1天）
    month = 1
    for days_in_month in month_days:
        if current_day <= days_in_month:
            break
        current_day -= days_in_month
        month += 1

    return datetime(year, month, current_day, remaining_hour, 0)  # 返回 datetime 对象


def static_var(static_path='ckpt/aurora-0.25-static.pickle', target_size=(120, 240)):
    with open(static_path, 'rb') as f:
        data = pickle.load(f)
    static_vars_z = torch.from_numpy(data['z']).unsqueeze(0).unsqueeze(0)  # [1, 128, 256]
    static_vars_lsm = torch.from_numpy(data['lsm']).unsqueeze(0).unsqueeze(0)
    static_vars_slt = torch.from_numpy(data['slt']).unsqueeze(0).unsqueeze(0)
    static_vars_z =F.interpolate(static_vars_z, size=target_size, mode='bilinear', align_corners=False)
    static_vars_lsm =F.interpolate(static_vars_lsm, size=target_size, mode='bilinear', align_corners=False)
    static_vars_slt =F.interpolate(static_vars_slt, size=target_size, mode='bilinear', align_corners=False)
    static_vars_z = static_vars_z.squeeze()
    static_vars_lsm = static_vars_lsm.squeeze()
    static_vars_slt = static_vars_slt.squeeze()
    return static_vars_z, static_vars_lsm, static_vars_slt

def mean_std_1d():
    with open('ckpt/mean_std_info.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    surf_stats={}

    for key, value in data.items():
        surf_stats[key]=tuple([value['mean'][0],value['std'][0]])
    return surf_stats

def mean_std_2d():
    folder_path = './mean_std'
    # 存储结果的字典
    json_data_dict = {}

    # 遍历文件夹下所有 .json 文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 使用去除扩展名的文件名作为 key
                key = os.path.splitext(filename)[0][:-9]
                json_data_dict[key] = tuple([torch.tensor(np.array(data['mean_temp'])).to(torch.float32).cuda(),
                                             torch.tensor(np.array(data['std_temp'])).to(torch.float32).cuda()])
    return json_data_dict



import os
import re
def get_latest_checkpoint(ckpt_dir):
    pt_files = [f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]
    if not pt_files:
        return None

    # 匹配 epoch_XXX.pt 文件名中的数字
    epoch_ckpts = []
    for f in pt_files:
        match = re.search(r'epoch_(\d+)\.pt', f)
        if match:
            epoch_ckpts.append((int(match.group(1)), f))

    if not epoch_ckpts:
        return None

    # 找到最大 epoch 的 ckpt 文件
    latest_epoch, latest_ckpt = max(epoch_ckpts, key=lambda x: x[0])
    return os.path.join(ckpt_dir, latest_ckpt), latest_epoch



from matplotlib import pyplot as plt
from matplotlib import colors
def save_visualization(pred, target, save_path, time,var):
    time = hours_to_datetime(time)
    """保存可视化图片"""
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 预测结果（t2m）
    a0 = axs[0].imshow(pred.detach().cpu().flip(0).numpy())
    axs[0].set_title(f'Prediction {var} - {time.strftime("%Y%m%d%H")}')
    axs[0].axis('off')
    fig.colorbar(a0, ax=axs[0], orientation='horizontal', shrink=0.8, aspect=16, extend='both')

    # 真实值（t2m）
    a1 = axs[1].imshow(target.detach().cpu().flip(0).numpy())
    axs[1].set_title(f'Ground Truth {var} - {time.strftime("%Y%m%d%H")}')
    axs[1].axis('off')
    fig.colorbar(a1, ax=axs[1], orientation='horizontal', shrink=0.8, aspect=16, extend='both')
    # 误差
    error = pred - target
    mean_error = error.mean().item()
    error_data = error.detach().cpu().flip(0).numpy()
    # a2 = axs[2].imshow(error_data, cmap='RdBu_r', norm=colors.Normalize(np.floor(error_data.min()), np.ceil(error_data.max())))
    if var=='2t':
        a2 = axs[2].imshow(error_data, cmap='RdBu_r', norm=colors.Normalize(-10, 10))
    else:
        a2 = axs[2].imshow(error_data, cmap='RdBu_r', norm=colors.Normalize(-500, 500))
    axs[2].set_title(f'Prediction Error {var} - {time.strftime("%Y%m%d%H")}\nMean Error: {mean_error:.3f} $m^2/s^2$')
    axs[2].axis('off')
    fig.colorbar(a2, ax=axs[2], orientation='horizontal', shrink=0.8, aspect=16, extend='both')

    plt.tight_layout()
    path=os.path.join(save_path, f'{var}_{time.strftime("%Y%m%d%H")}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
