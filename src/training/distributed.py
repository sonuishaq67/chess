import os

import torch
import torch.distributed as dist

import habana_frameworks.torch.core as htcore  # noqa: F401
import habana_frameworks.torch.distributed.hccl  # noqa: F401 registers HCCL backend
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu


def setup_dist():
    """Initialize HCCL process group and return (device, local_rank, rank, world_size)."""
    world_size, rank, local_rank = initialize_distributed_hpu()
    dist.init_process_group(backend="hccl")
    device = torch.device("hpu")

    if hasattr(torch, "hpu") and hasattr(torch.hpu, "set_device"):
        torch.hpu.set_device(local_rank)

    try:
        # Fail fast on device mapping/acquisition issues before the dataset preload.
        torch.empty(1, device=device)
        htcore.mark_step()
    except Exception as exc:
        raise RuntimeError(
            "Failed to acquire HPU during distributed setup "
            f"(rank={rank}, local_rank={local_rank}, world_size={world_size}, "
            f"HLS_MODULE_ID={os.environ.get('HLS_MODULE_ID')}, "
            f"HABANA_VISIBLE_MODULES={os.environ.get('HABANA_VISIBLE_MODULES')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')})."
        ) from exc

    return device, local_rank, rank, world_size


def cleanup_dist():
    dist.destroy_process_group()
