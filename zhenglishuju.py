import os
import numpy as np
import torch
from tqdm import tqdm

root_dir = '/sharefiles4/zhaodan'
output_dir = '/sharefiles2/guoyixin/datasets/weather_tensors'
os.makedirs(output_dir, exist_ok=True)

simple_vars = ['t2m_npy', 'u10_npy', 'v10_npy', 'tp_npy']#['2t', '10u', '10v', 'msl']
level_vars = ['z_npy', 'u_npy', 'v_npy', 't_npy', 'r_npy']
levels = [f'level_{i:02d}' for i in range(13)]
years = [str(y) for y in range(1980, 2019)]  # 1979 to 2018

# 假设所有变量都拥有相同的文件名列表，以 '2t/1979' 为参考
# ref_var = 't2m_npy'
# ref_year = '1979'
# ref_dir = os.path.join(root_dir, ref_var, ref_year)
# file_list = sorted([f for f in os.listdir(ref_dir) if f.endswith('.npy')])

# print(f"🔍 发现 {len(file_list)} 个 npy 文件")

for year in tqdm(years, desc="Processing years"):
    ref_var = 't2m_npy'
    ref_year = year
    ref_dir = os.path.join(root_dir, ref_var, ref_year)
    file_list = sorted([f for f in os.listdir(ref_dir) if f.endswith('.npy')])

    print(f"🔍 发现 {len(file_list)} 个 npy 文件")
    for fname in tqdm(file_list, desc=f"Year {year}", leave=False):
        all_arrays = []

        # 1. simple vars
        for var in simple_vars:
            path = os.path.join(root_dir, var, year, fname)
            if not os.path.exists(path):
                print(f"⚠️ 缺失: {path}")
                continue
            arr = np.load(path)
            all_arrays.append(arr)

        # 2. level vars
        for var in level_vars:
            for level in levels:
                path = os.path.join(root_dir, var, level, year, fname)
                if not os.path.exists(path):
                    print(f"⚠️ 缺失: {path}")
                    continue
                arr = np.load(path)
                all_arrays.append(arr)

        # 确保维度匹配
        if len(all_arrays) != 69:
            print(f"⛔ 数据不完整：{year}/{fname} 只找到 {len(all_arrays)} 个通道，跳过")
            continue

        stacked = np.stack(all_arrays, axis=0)  # [69, 128, 256]
        tensor = torch.from_numpy(stacked.astype(np.float32))

        save_path = os.path.join(output_dir,year, f"{fname.replace('.npy', '.pt')}")
        os.makedirs(os.path.join(output_dir,year), exist_ok=True)
        torch.save(tensor, save_path)
