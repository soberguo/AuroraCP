import xarray as xr
import json

# 打开文件
ds_merged = xr.open_mfdataset('/sharefiles4/zhaodan/1.40625/v_component_of_wind/*.nc', combine='by_coords')

# 对 time 维度求均值和方差，得到 [lat, lon]
# mean_temp = ds_merged['z'].mean(dim='time').compute()
# std_temp  = ds_merged['z'].std(dim='time').compute()
for i in range(13):
    mean_temp =ds_merged['v'][:,i]
    mean_temp =mean_temp.mean(dim='time').compute()
    std_temp  = ds_merged['v'][:,i]
    std_temp  = std_temp.std(dim='time').compute()
    # 转成 list
    mean_list = mean_temp.values.tolist()
    std_list  = std_temp.values.tolist()
    atmos_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    # 也可以一起保存到一个 dict
    result = {
        'mean_temp': mean_list,
        'std_temp': std_list
    }

    # 保存成 JSON 文件
    with open(f'v_{atmos_levels[i]}_mean_std.json', 'w') as f:
        json.dump(result, f)