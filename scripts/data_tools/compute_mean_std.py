import xarray as xr
import os
import glob
import numpy as np

# 1. 设置路径
data_dir = '/sharefiles2/guoyixin/datasets/vswl/regrid'
nc_files = sorted(glob.glob(os.path.join(data_dir, '*.nc')))

print(f"检测到 {len(nc_files)} 个文件。")

if len(nc_files) > 0:
    try:
        # 2. 读取文件 (使用 chunks 避免内存溢出)
        print("正在懒加载数据集...")
        # chunks='auto' 会让 dask 自动选择合适的块大小
        ds = xr.open_mfdataset(nc_files, combine='by_coords', parallel=True, chunks='auto')

        print(f"包含变量: {list(ds.data_vars)}")
        print("-" * 30)

        # 3. 循环计算每个数据变量的全局均值和标准差
        for var_name in ds.data_vars:
            print(f"正在计算变量: [{var_name}] ...")

            # 计算全局均值 (Scalar)
            # .item() 将 numpy/dask 0-d array 转为纯 Python float
            mean_val = ds[var_name].mean().compute().item()

            # 计算全局标准差 (Scalar)
            std_val = ds[var_name].std().compute().item()

            print(f"  >>> 均值 (Mean): {mean_val:.6f}")
            print(f"  >>> 标差 (Std) : {std_val:.6f}")
            print("-" * 30)

    except Exception as e:
        print(f"发生错误: {e}")
else:
    print("未找到文件。")