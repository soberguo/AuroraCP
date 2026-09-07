import os
import random
import time
from pathlib import Path

import numpy as np
import torch


RANK_STATE_VERSION = 1
STAGE3_CHECKPOINT_VERSION = 2
STAGE3_NAME = 'stage3_replay_rollout'


def capture_rng_state(device):
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': None,
    }
    if device.type == 'cuda':
        state['torch_cuda'] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'])
    if device.type == 'cuda':
        cuda_state = state.get('torch_cuda')
        if cuda_state is None:
            raise ValueError('rank state 缺少 CUDA RNG 状态')
        torch.cuda.set_rng_state(cuda_state, device)


def atomic_torch_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def rank_state_path(save_dir, epoch, rank):
    return Path(save_dir) / 'rank_states' / f'epoch_{epoch:03d}_rank_{rank:03d}.pt'


def save_rank_state(save_dir, epoch, rank, world_size, replay_buffer, device):
    path = rank_state_path(save_dir, epoch, rank)
    atomic_torch_save({
        'version': RANK_STATE_VERSION,
        'epoch': epoch,
        'rank': rank,
        'world_size': world_size,
        'replay_buffer_state': replay_buffer.state_dict(),
        'rng_state': capture_rng_state(device),
    }, path)
    return path


def load_rank_state(save_dir, epoch, rank, world_size, replay_buffer):
    path = rank_state_path(save_dir, epoch, rank)
    if not path.is_file():
        raise FileNotFoundError(f'缺少 rank state: {path}')

    state = torch.load(path, map_location='cpu', weights_only=False)
    expected = {
        'version': RANK_STATE_VERSION,
        'epoch': epoch,
        'rank': rank,
        'world_size': world_size,
    }
    for key, expected_value in expected.items():
        if state.get(key) != expected_value:
            raise ValueError(
                f'rank state 元数据不匹配 ({path}): '
                f'{key}={state.get(key)!r}, expected={expected_value!r}'
            )
    replay_buffer.load_state_dict(state['replay_buffer_state'])
    return state['rng_state'], path


def prune_rank_states(save_dir, rank, keep):
    state_dir = Path(save_dir) / 'rank_states'
    paths = sorted(state_dir.glob(f'epoch_*_rank_{rank:03d}.pt'))
    for path in paths[:-keep]:
        path.unlink()


def wait_for_file(path, poll_interval_seconds=1.0):
    path = Path(path)
    while not path.is_file():
        time.sleep(poll_interval_seconds)
