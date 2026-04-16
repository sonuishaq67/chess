import os

import torch
import torch.distributed as dist

import habana_frameworks.torch.core as htcore  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401 registers HCCL backend


def setup_dist():
    """Initialize HCCL process group and return (device, local_rank, rank, world_size)."""
    dist.init_process_group(backend="hccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("hpu")
    return device, local_rank, dist.get_rank(), dist.get_world_size()


def cleanup_dist():
    dist.destroy_process_group()
