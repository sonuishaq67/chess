import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, random_split

import yaml

import habana_frameworks.torch.core as htcore

from src.model import Transformer, TransformerConfig
from src.training.dataset import ChessDataset, PAD_ID
from src.training.distributed import setup_dist, cleanup_dist


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_model(model_cfg: dict, device: torch.device) -> Transformer:
    config = TransformerConfig(**model_cfg)
    return Transformer(config).to(device)


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Linear warmup then cosine decay to 10% of max_lr."""
    min_lr = max_lr * 0.1
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yml")
    parser.add_argument("--model-config", default="configs/model.yml")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="one forward pass then exit")
    args = parser.parse_args()

    device, local_rank, rank, world_size = setup_dist()
    is_main = rank == 0

    # Habana v1.20 release notes: PyTorch >=2.5.1 SDPA can upcast to fp32 and
    # trip synStatus graph-compile failures under bf16 autocast. Opt in to
    # bf16 reductions inside math SDPA to keep it compilable on Gaudi.
    torch._C._set_math_sdp_allow_fp16_bf16_reduction(True)

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)

    if is_main:
        print(f"Device: {device}  |  world_size: {world_size}")

    # --- Data ---
    base_dir = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
    packing_dir = os.path.join(base_dir, cfg.get("packing_dir", "dataset/packing"))

    dataset = ChessDataset(packing_dir, max_seq_len=model_cfg.get("max_seq_len", 256))
    if is_main:
        print(f"Total chunks: {len(dataset):,}")

    val_frac = cfg.get("val_split", 0.05)
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    if is_main:
        print(f"Train chunks: {train_size:,}  Val chunks: {val_size:,}")

    batch_size = cfg.get("batch_size", 128)
    num_workers = min((os.cpu_count() or 8) // world_size, 8)
    if is_main:
        print(f"DataLoader workers per rank: {num_workers}")

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # --- Model ---
    model = build_model(model_cfg, device)
    if is_main:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Resume: load into bare model BEFORE DDP wrap ---
    resume_ckpt = None
    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"])
        start_epoch = resume_ckpt["epoch"]
        global_step = resume_ckpt["step"]
        if is_main:
            print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    # Flush lazy graph from .to(hpu) / state-dict load before first HCCL collective.
    htcore.mark_step()

    # --- DDP wrap ---
    # broadcast_buffers=False: skip per-step buffer sync (rope cos/sin are identical on every rank).
    # gradient_as_bucket_view left at default (False): on HPU lazy mode, bucket-view
    # gradients (non-zero storage_offset) trip ValidateSyncInputTensors at clip_grad_norm_.
    model = torch.nn.parallel.DistributedDataParallel(
        model, broadcast_buffers=False
    )

    # Flush DDP's init-broadcast recipe before the first forward — otherwise it
    # fuses with forward ops into one oversized lazy graph that fails to compile.
    htcore.mark_step()

    # --- Dry run ---
    if args.dry_run:
        use_amp = cfg.get("use_amp", True)
        if is_main:
            import json
            vocab_path = os.path.join(base_dir, "dataset", "vocab.json")
            id2tok = {v: k for k, v in json.loads(open(vocab_path).read()).items()}
            print("\n--- Sample chunk (first 3) ---")
            for i in range(min(3, len(dataset))):
                inp, tgt = dataset[i]
                tokens = [id2tok.get(t.item(), "?") for t in inp]
                visible = [t for t in tokens if t != "PAD"][:20]
                print(f"  chunk {i}: {len(visible)} non-pad tokens | {' '.join(visible)} ...")
            print("\n--- One forward pass ---")

        inp, tgt = dataset[0]
        inp = inp.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        with torch.no_grad():
            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(inp)
                loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)(
                    logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
                )
        htcore.mark_step()
        if is_main:
            print(f"  input shape:  {inp.shape}")
            print(f"  logits shape: {logits.shape}")
            print(
                f"  loss: {loss.item():.4f} "
                f"(random init ~{math.log(model_cfg.get('vocab_size', 1972)):.2f})"
            )
            print("\nDry run complete.")
        cleanup_dist()
        return

    # --- Optimizer ---
    max_lr = cfg.get("learning_rate", 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    max_epochs = cfg.get("max_epochs", 10)
    max_steps = max_epochs * (len(train_loader) // grad_accum_steps)
    warmup_steps = cfg.get("warmup_steps", 500)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    use_amp = cfg.get("use_amp", True)

    ckpt_dir = os.path.join(base_dir, cfg.get("checkpoint_dir", "checkpoints"))
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()  # all ranks wait until rank 0 creates the dir

    log_every = cfg.get("log_every", 100)
    save_every = cfg.get("save_every", 2000)
    grad_clip = cfg.get("grad_clip", 1.0)

    if is_main:
        print(
            f"\nStarting training: {max_epochs} epochs, {len(train_loader)} micro-batches/epoch, "
            f"{max_steps} optimizer steps "
            f"(grad_accum={grad_accum_steps}, world_size={world_size}, "
            f"global_batch={batch_size * world_size * grad_accum_steps})\n"
        )

    # --- Training loop ---
    for epoch in range(start_epoch, max_epochs):
        model.train()
        train_sampler.set_epoch(epoch)  # ensures different shuffles per epoch across ranks
        # Accumulators on device — avoids per-micro-batch .item() syncs that were
        # flushing the HPU lazy graph and causing ~4s-busy / ~18s-idle cycles.
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        token_sum = torch.zeros((), device=device, dtype=torch.float32)
        tokens_since_log = torch.zeros((), device=device, dtype=torch.float32)
        t0 = time.time()
        batch_idx = -1

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(input_ids)
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )
                loss = loss / grad_accum_steps

            loss.backward()

            with torch.no_grad():
                n_tokens = (labels != PAD_ID).sum().to(torch.float32)
                loss_sum += loss.detach() * grad_accum_steps * n_tokens
                token_sum += n_tokens
                tokens_since_log += n_tokens

            if (batch_idx + 1) % grad_accum_steps == 0:
                lr = get_lr(global_step, warmup_steps, max_steps, max_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                htcore.mark_step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % log_every == 0:
                    # One host sync per log interval, not per micro-batch.
                    loss_val = loss_sum.item()
                    tokens_val = token_sum.item()
                    avg = loss_val / tokens_val if tokens_val > 0 else 0.0
                    elapsed = time.time() - t0
                    tok_per_sec = (
                        tokens_since_log.item() * world_size / elapsed if elapsed > 0 else 0.0
                    )
                    cur_loss = (loss.detach() * grad_accum_steps).item()
                    print(
                        f"  step {global_step:>6d} | loss {cur_loss:.4f} | "
                        f"avg {avg:.4f} | lr {lr:.2e} | {tok_per_sec:.0f} tok/s (global)"
                    )
                    tokens_since_log.zero_()
                    t0 = time.time()

                if is_main and global_step % save_every == 0:
                    path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "step": global_step,
                            "config": model_cfg,
                        },
                        path,
                    )
                    print(f"  checkpoint saved: {path}")

        # Flush leftover accumulated gradients at end of epoch
        if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
            lr = get_lr(global_step, warmup_steps, max_steps, max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            htcore.mark_step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        # Epoch summary — all-reduce device accumulators, sync once.
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_sum, op=dist.ReduceOp.SUM)
        train_avg = (loss_sum / token_sum).item() if token_sum.item() > 0 else 0.0
        if is_main:
            print(f"\nEpoch {epoch + 1}/{max_epochs} done | train loss: {train_avg:.4f}")

        # --- Validation ---
        model.eval()
        val_loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        val_token_sum = torch.zeros((), device=device, dtype=torch.float32)
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                n_tokens = (labels != PAD_ID).sum().to(torch.float32)
                val_loss_sum += loss * n_tokens
                val_token_sum += n_tokens
                htcore.mark_step()

        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_sum, op=dist.ReduceOp.SUM)
        val_avg = (val_loss_sum / val_token_sum).item()

        if is_main:
            print(f"  val loss: {val_avg:.4f}\n")

        # Save end-of-epoch checkpoint (rank 0 only)
        if is_main:
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "step": global_step,
                    "config": model_cfg,
                },
                path,
            )
            print(f"  epoch checkpoint saved: {path}")

    if is_main:
        print("Training complete.")
    cleanup_dist()


if __name__ == "__main__":
    train()
