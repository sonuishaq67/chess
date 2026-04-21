import os

import torch
import torch.distributed as dist

# habana_frameworks is imported inside setup_dist() on purpose. Top-level
# habana imports register HPU device FDs and the HCCL backend with torch
# eagerly; if this module is imported before DataLoader workers are forked,
# the forked children inherit that registration and hl-smi attributes AIP
# memory to every `pt_data_worker`. train_fast.py prestarts workers, then
# calls setup_dist(), so all habana state is created after fork.


def setup_dist():
    """Initialize HCCL process group and return (device, local_rank, rank, world_size)."""
    import habana_frameworks.torch.core as htcore  # noqa: F401
    import habana_frameworks.torch.distributed.hccl  # noqa: F401 registers HCCL backend
    from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu

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
