import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True)
class DistributedContext:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self):
        return self.rank == 0


def initialize_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    requested = getattr(args, "distributed", False)

    if requested and not distributed:
        raise RuntimeError(
            "--distributed=true requires torchrun with more than one process, for example: "
            "torchrun --standalone --nproc_per_node=2 main.py ..."
        )

    if distributed:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = args.dist_backend
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL distributed training requires CUDA GPUs")
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but only {torch.cuda.device_count()} CUDA device(s) are visible"
                )
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device(args.device if args.device == "cpu" else "cpu")
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank = 0
        local_rank = 0
        if args.device.startswith("cuda") and torch.cuda.is_available():
            device = torch.device(args.device)
        else:
            device = torch.device("cpu")

    args.distributed = distributed
    args.rank = rank
    args.local_rank = local_rank
    args.world_size = world_size
    return DistributedContext(distributed, rank, local_rank, world_size, device)


def seed_everything(seed, rank=0):
    process_seed = seed + rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(process_seed)


def wrap_ddp(model, context, find_unused_parameters=True):
    if not context.distributed:
        return model
    kwargs = {
        "broadcast_buffers": False,
        "find_unused_parameters": find_unused_parameters,
        "gradient_as_bucket_view": True,
    }
    if context.device.type == "cuda":
        kwargs.update(device_ids=[context.local_rank], output_device=context.local_rank)
    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


def distributed_all_true(value, context):
    if not context.distributed:
        return bool(value)
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=context.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def distributed_any_true(value, context):
    if not context.distributed:
        return bool(value)
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=context.device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def distributed_mean(value, context):
    if not context.distributed:
        return float(value)
    result = torch.tensor(float(value), dtype=torch.float64, device=context.device)
    dist.all_reduce(result, op=dist.ReduceOp.SUM)
    return (result / context.world_size).item()


def barrier(context):
    if context.distributed:
        dist.barrier()


def cleanup_distributed(context):
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()
