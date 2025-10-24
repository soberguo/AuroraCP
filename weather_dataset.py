
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch.nn.functional as F
import xarray as xr
import torch



class WeatherBench128(Dataset):
    def __init__(self, data_folder,n=6,train=True,roll_step=0):  # 添加保存路径参数
        self.n = n  # 时间步间隔
        self.roll_step=roll_step
        self.data_folder = data_folder

        self.level=['level_00','level_01','level_02','level_03',
                    'level_04','level_05','level_06','level_07',
                    'level_08','level_09','level_10','level_11',
                    'level_12']
        if train:
            self.year=[str(y) for y in range(1979, 2015)]
        else:
            self.year=[str(y) for y in range(2017, 2019)]
        self.surf_var=['2t', '10u', '10v', 'msl']
        self.atmos_var=['z', 'u', 'v', 't', 'q']
        # self.sshf = xr.open_mfdataset('/sharefiles4/zhaodan/1.40625/rigrid_heat_fllux/*.nc', combine='by_coords')
        # sshf = xr.open_dataset('/sharefiles4/zhaodan/1.40625/rigrid_heat_fllux/surface_sensible_heat_flux_1979_1.40625deg.nc')
        
        #        收集所有文件
        all_files = []
        for y in self.year:
            year_dir = os.path.join(self.data_folder, y)
            files = [f for f in os.listdir(year_dir) if f.endswith('.pt')]
            year_files = [os.path.join(y, f) for f in sorted(files)]
            all_files.extend(year_files)

        all_files = sorted(all_files)

        # 每隔 n 个保留一个
        self.file_list = all_files[::self.n]
        print(f"🔍 发现 {len(self.file_list)} 个文件")
    def __len__(self):
        return len(self.file_list) - 2 

    def __getitem__(self, index):#输入t-1和t，输出t+1
        target={}
        idx1 = index
        idx2 = index + 1
        idx_tgt = index + 2 +(self.roll_step-1)

        file_path1 = self.file_list[idx1]
        file_path2 = self.file_list[idx2]
        target_path = self.file_list[idx_tgt]

        sample_x1 = torch.load(os.path.join(self.data_folder, file_path1))
        sample_x2 = torch.load(os.path.join(self.data_folder, file_path2))


        target['tgt'] = torch.load(os.path.join(self.data_folder, target_path))
        #训练用到的t时刻文件名
        target['filename']= file_path2.split('/')[-1].split(".")[0]
        #roll_step用到的t+1时刻文件名
        target['tgt_filename']= target_path.split('/')[-1].split(".")[0]

        return (sample_x1, sample_x2), target
    
def custom_collate(batch):
    images = []
    targets = []
    # images_clip = []
    
    for im, tar in batch:
        images.append(im)
        targets.append(tar)
        
        # images_clip.append(im_clip)
    return images, targets
 






    

    