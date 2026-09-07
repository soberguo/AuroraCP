from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import torch
import torch.nn.functional as F


class WeatherBench2(Dataset):
    """
    统一的 WeatherBench2 数据集，支持 6h 和 12h 时间步长

    Args:
        data_folder: 数据文件夹路径
        train: 是否为训练集
        roll_step: rollout 步数
        timestep: 时间步长（6 或 12 小时）
    """
    def __init__(self, data_folder, roll_step=0, timestep=6,years=(1979,2018),args=None,
                 num_rollout_targets=None, return_train_aux_target=None):
        self.roll_step = roll_step
        self.data_folder = data_folder
        self.timestep = timestep
        self.train_roll_step = args.train_roll_step if args is not None else 1
        self.num_rollout_targets = num_rollout_targets if num_rollout_targets is not None else 0
        self.return_train_aux_target = (
            return_train_aux_target
            if return_train_aux_target is not None
            else self.train_roll_step != 1
        )

        # 计算步长倍数：原始数据是6小时间隔
        # 6h: step_multiplier = 1
        # 12h: step_multiplier = 2
        self.step_multiplier = timestep // 6


        self.year = [str(y) for y in range(years[0], years[1])]


        # 获取所有文件列表
        all_files = []
        for y in self.year:
            year_dir = os.path.join(self.data_folder, y)
            if not os.path.exists(year_dir):
                continue
            # 确保按文件名排序，保证时间连续性
            files = [f for f in os.listdir(year_dir) if f.endswith('.pt')]
            year_files = [os.path.join(y, f) for f in sorted(files)]
            all_files.extend(year_files)

        self.file_list = sorted(all_files)
        print(f"🔍 发现 {len(self.file_list)} 个文件 (timestep={timestep}h)")

        # [新增] 建立 文件名 -> 文件路径 的映射，方便快速查找
        # 同时也建立 文件名 -> 索引 的映射，方便找"下一个"
        self.name_to_idx = {}
        for idx, f_path in enumerate(self.file_list):
            # 假设文件名格式是 "..../2018/0101_00.pt"，提取 "0101_00" 这种唯一标识
            # 你原来的代码逻辑: target['filename'] = file_path2.split('/')[-1].split(".")[0]
            fname = f_path.split('/')[-1].split(".")[0]
            self.name_to_idx[fname] = idx


    def __len__(self):
        # 根据 timestep 计算有效长度
        # 需要 2 个输入时刻 + roll_step 个输出时刻
        m = self.step_multiplier

        max_target_offset = max(2 + self.roll_step, 1 + self.num_rollout_targets)
        if self.return_train_aux_target:
            max_target_offset = max(max_target_offset, 3)
        return max(0, len(self.file_list) - max_target_offset * m)



    def __getitem__(self, index):
        """
        输入 t-1 和 t 时刻，输出 t+1 时刻
        """
        target = {}
        m = self.step_multiplier

        idx1 = index
        idx2 = index + 1 * m
        idx_tgt = index + 2 * m + self.roll_step * m  #12h
        if self.return_train_aux_target:
            idx_tgt2 = index + 3 * m
            target_path2 = self.file_list[idx_tgt2]
            full_path_tgt2 = os.path.join(self.data_folder, target_path2)
            target_tensor2 = torch.load(full_path_tgt2)
            target['tgt2'] = target_tensor2
        file_path1 = self.file_list[idx1]
        file_path2 = self.file_list[idx2]
        target_path = self.file_list[idx_tgt]


        # 加载数据
        full_path1 = os.path.join(self.data_folder, file_path1)
        full_path2 = os.path.join(self.data_folder, file_path2)
        full_path_tgt = os.path.join(self.data_folder, target_path)
        #

        sample_x1 = torch.load(full_path1)
        sample_x2 = torch.load(full_path2)
        target_tensor = torch.load(full_path_tgt)
        #
        target['tgt'] = target_tensor
        if self.num_rollout_targets > 0:
            target['rollout_tgts'] = [
                torch.load(os.path.join(self.data_folder, self.file_list[index + (2 + lead) * m]))
                for lead in range(self.num_rollout_targets)
            ]
        #
        # 文件名信息 (去除后缀)
        target['filename'] = os.path.basename(file_path2).split(".")[0]
        target['tgt_filename'] = os.path.basename(target_path).split(".")[0]

        return (sample_x1, sample_x2), target

    # [新增] 根据当前文件名，加载下一个时刻的真值
    def get_next_target(self, current_filename):
        """
        用于 Buffer Replay：给定当前输入的时间(filename)，找到下一个时刻的真值。
        """
        if current_filename not in self.name_to_idx:
            return None, None

        curr_idx = self.name_to_idx[current_filename]
        next_idx = curr_idx + 1*self.step_multiplier

        # 检查是否越界
        if next_idx >= len(self.file_list):
            return None, None

        next_file_path = self.file_list[next_idx]
        next_filename = next_file_path.split('/')[-1].split(".")[0]

        # 加载数据
        full_path = os.path.join(self.data_folder, next_file_path)
        try:
            target_tensor = torch.load(full_path)
            return target_tensor, next_filename
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, None



def custom_collate(batch):
    """自定义 collate 函数"""
    images = []
    targets = []

    for im, tar in batch:
        images.append(im)
        targets.append(tar)

    return images, targets


if __name__ == "__main__":
    from tqdm import tqdm
    from types import SimpleNamespace
    test_args = SimpleNamespace(train_roll_step=1)

    # 测试 6h 数据集
    print("Testing 6h dataset...")
    dataset_6h = WeatherBench2(
        data_folder='/sharefiles3/guoyixin/datasets/weatherbench2_71var',
        timestep=6,
        args=test_args,
    )
    print(f"6h dataset length: {len(dataset_6h)}")

    # 测试 12h 数据集
    print("\nTesting 12h dataset...")
    dataset_12h = WeatherBench2(
        data_folder='/sharefiles3/guoyixin/datasets/weatherbench2_71var',
        timestep=12,
        args=test_args,
    )
    print(f"12h dataset length: {len(dataset_12h)}")
