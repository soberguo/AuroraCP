# buffer.py
import torch
import random
from collections import deque

class AuroraReplayBuffer:
    def __init__(self, capacity=4000): # 论文中 20 GPUs * 200 = 4000
        self.buffer = deque(maxlen=capacity)

    def push(self, prev_states, curr_states, current_filenames, lead_times):
        """
        prev_states: (B, C, H, W) - T-1 时刻
        curr_states: (B, C, H, W) - T 时刻
        current_filenames: list[str] - T 时刻的文件名 (用于找 T+1 的真值)
        lead_times: tensor(B) - 当前已经 rollout 了多少步
        """
        # 我们把 batch 拆开存，或者存整个 batch 也可以。
        # 为了灵活性，通常拆开存，取的时候随机组 batch。
        # 为了显存效率，这里存 CPU tensor。

        batch_size = prev_states.shape[0]
        for i in range(batch_size):
            item = {
                'prev': prev_states[i].detach().cpu(),
                'curr': curr_states[i].detach().cpu(),
                'fname': current_filenames[i],
                'lead': lead_times[i].item()
            }
            self.buffer.append(item)

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None

        samples = random.sample(self.buffer, batch_size)

        # 拼装回 Tensor
        batch = {
            'prev': torch.stack([s['prev'] for s in samples]),
            'curr': torch.stack([s['curr'] for s in samples]),
            'fname': [s['fname'] for s in samples],
            'lead': torch.tensor([s['lead'] for s in samples])
        }
        return batch

    def __len__(self):
        return len(self.buffer)

    def state_dict(self):
        """Return the exact CPU state needed to continue replay sampling."""
        return {
            'version': 1,
            'capacity': self.buffer.maxlen,
            'items': list(self.buffer),
        }

    def load_state_dict(self, state_dict):
        if state_dict.get('version') != 1:
            raise ValueError(f"不支持的 replay buffer state 版本: {state_dict.get('version')}")

        saved_capacity = state_dict.get('capacity')
        if saved_capacity != self.buffer.maxlen:
            raise ValueError(
                "replay buffer capacity 不一致: "
                f"checkpoint={saved_capacity}, current={self.buffer.maxlen}"
            )

        items = state_dict.get('items')
        if not isinstance(items, list):
            raise TypeError("replay buffer state 中的 items 必须是 list")
        if len(items) > self.buffer.maxlen:
            raise ValueError(
                f"replay buffer state 包含 {len(items)} 条记录，超过容量 {self.buffer.maxlen}"
            )

        required_keys = {'prev', 'curr', 'fname', 'lead'}
        restored_items = []
        for index, item in enumerate(items):
            if not isinstance(item, dict) or set(item) != required_keys:
                raise ValueError(f"replay buffer 第 {index} 条记录格式无效")
            restored_items.append({
                'prev': item['prev'].detach().cpu(),
                'curr': item['curr'].detach().cpu(),
                'fname': item['fname'],
                'lead': int(item['lead']),
            })

        self.buffer.clear()
        self.buffer.extend(restored_items)
