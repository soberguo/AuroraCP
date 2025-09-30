from datetime import datetime
import torch
import pickle
import torch.nn.functional as F
import json
import numpy as np



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


def static_var():
    with open('ckpt/aurora-0.25-static.pickle', 'rb') as f:
        data = pickle.load(f)
    static_vars_z = torch.from_numpy(data['z']).unsqueeze(0).unsqueeze(0)  # [1, 128, 256]
    static_vars_lsm = torch.from_numpy(data['lsm']).unsqueeze(0).unsqueeze(0)
    static_vars_slt = torch.from_numpy(data['slt']).unsqueeze(0).unsqueeze(0)
    static_vars_z =F.interpolate(static_vars_z, size=(128, 256), mode='bilinear', align_corners=False)
    static_vars_lsm =F.interpolate(static_vars_lsm, size=(128, 256), mode='bilinear', align_corners=False)
    static_vars_slt =F.interpolate(static_vars_slt, size=(128, 256), mode='bilinear', align_corners=False)
    static_vars_z = static_vars_z.squeeze()
    static_vars_lsm = static_vars_lsm.squeeze()
    static_vars_slt = static_vars_slt.squeeze()
    return static_vars_z, static_vars_lsm, static_vars_slt

def mean_std_1d():
    with open('/sharefiles2/guoyixin/datasets/weather_tensors/mean_std_info.json', 'r', encoding='utf-8') as f:
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
        a2 = axs[2].imshow(error_data, cmap='RdBu_r', norm=colors.Normalize(-1000, 1000))
    axs[2].set_title(f'Prediction Error {var} - {time.strftime("%Y%m%d%H")}\nMean Error: {mean_error:.3f}K')
    axs[2].axis('off')
    fig.colorbar(a2, ax=axs[2], orientation='horizontal', shrink=0.8, aspect=16, extend='both')

    plt.tight_layout()
    path=os.path.join(save_path, f'{var}_{time.strftime("%Y%m%d%H")}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()