import argparse
import contextlib
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, random_split

import yaml

import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore
from habana_frameworks.torch.hpex.optimizers import FusedAdamW
from habana_frameworks.torch.hpex.kernels import FusedSDPA

from src.model import Transformer, TransformerConfig
from src.model.transformer import MultiHeadAttention, apply_rope
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


def _fused_attn_forward(
    self,
    x: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    causal_mask: torch.Tensor,
) -> torch.Tensor:
    # Monkey-patched forward — uses FusedSDPA with is_causal=True so the
    # [seq,seq] mask is never materialized. causal_mask arg is accepted for
    # signature compatibility with the original forward but unused.
    batch, seq_len, _ = x.shape

    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)

    q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    q = apply_rope(q, rope_cos, rope_sin)
    k = apply_rope(k, rope_cos, rope_sin)

    dropout_p = self.attn_dropout.p if self.training else 0.0
    out = FusedSDPA.apply(q, k, v, None, dropout_p, True, None, "fast", False)

    out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    return self.w_o(out)


def _device_clip_grad_norm(parameters, max_norm: float) -> torch.Tensor:
    # Device-only grad clip — no host sync. torch.nn.utils.clip_grad_norm_
    # internally does a .item() on the total norm, which stalls the lazy graph.
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return None
    device = params[0].grad.device
    norms = torch.stack(
        [torch.linalg.vector_norm(p.grad.detach()) for p in params]
    )
    total_norm = torch.linalg.vector_norm(norms)
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for p in params:
        p.grad.detach().mul_(clip_coef)
    return total_norm


def _unwrap(model: nn.Module) -> nn.Module:
    """Peel DDP / HPU-graph wrappers to reach the bare Transformer for state_dict."""
    inner = model
    while hasattr(inner, "module"):
        inner = inner.module
    return inner


def _set_graph_iteration_count(model: nn.Module, iteration: int) -> None:
    """Tell ModuleCacher about grad-accum phase when that mode is enabled."""
    inner = model
    while inner is not None:
        setter = getattr(inner, "set_iteration_count", None)
        if setter is not None:
            setter(iteration)
            return
        inner = getattr(inner, "module", None)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training.yml")
    parser.add_argument("--model-config", default="configs/model.yml")
    parser.add_argument("--resume", default=None, help="path to checkpoint to resume from")
    parser.add_argument("--dry-run", action="store_true", help="one forward pass then exit")
    parser.add_argument(
        "--max-steps-override",
        type=int,
        default=None,
        help="cap number of optimizer steps (for short benchmark runs)",
    )
    parser.add_argument(
        "--no-hpu-graphs",
        action="store_true",
        help="disable graph wrapping and run pure lazy mode",
    )
    parser.add_argument(
        "--graph-mode",
        choices=("modulecacher", "compile", "none"),
        default="modulecacher",
        help="training graph backend to use; default is Habana ModuleCacher",
    )
    parser.add_argument(
        "--graph-max-graphs",
        type=int,
        default=1,
        help="max cached training graphs; static-shape training should use 1",
    )
    parser.add_argument(
        "--graph-grad-accum",
        action="store_true",
        help="let ModuleCacher keep separate graphs for grad-accum phases",
    )
    parser.add_argument(
        "--no-fused-sdpa",
        action="store_true",
        help="fallback: keep the manual attention forward from transformer.py",
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="time dl/fwd/bwd/opt/mark on first 20 micro-batches then exit",
    )
    args = parser.parse_args()

    device, local_rank, rank, world_size = setup_dist()
    is_main = rank == 0

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)
    grad_accum_steps = cfg.get("grad_accum_steps", 1)
    graph_mode = "none" if args.no_hpu_graphs else args.graph_mode
    use_graph_grad_accum = args.graph_grad_accum and grad_accum_steps > 1

    if is_main:
        print(f"Device: {device}  |  world_size: {world_size}")

    # Patch MultiHeadAttention.forward -> FusedSDPA for the training process
    # only. Leaves src/model/transformer.py untouched so ONNX export for the
    # Lichess bot keeps its portable manual-attention forward.
    if not args.no_fused_sdpa:
        MultiHeadAttention.forward = _fused_attn_forward
        if is_main:
            print("Patched MultiHeadAttention.forward -> FusedSDPA (is_causal=True)")
    elif is_main:
        print("Keeping manual attention (--no-fused-sdpa)")

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
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    # --- Model ---
    model = build_model(model_cfg, device)
    if is_main:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # --- Resume: load into bare model BEFORE DDP/graph wrap ---
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

    htcore.mark_step()
    torch.hpu.synchronize()
    if is_main:
        print("Model on HPU.", flush=True)

    if graph_mode == "compile":
        if is_main:
            print("Graph mode compile selected; wrapping after DDP.", flush=True)
    elif graph_mode == "none":
        if is_main:
            print("Graph wrapping disabled; pure lazy mode.", flush=True)
    else:
        if is_main:
            print("Graph mode modulecacher selected; wrapping after DDP.", flush=True)

    # --- DDP wrap (outer) ---
    # DDP MUST be outermost wrapper for model.no_sync() to be callable.
    model = torch.nn.parallel.DistributedDataParallel(
        model, broadcast_buffers=False
    )

    htcore.mark_step()
    torch.hpu.synchronize()
    if is_main:
        print("DDP init broadcast complete.", flush=True)

    if graph_mode == "modulecacher":
        # Keep DDP outermost so model.no_sync() remains available, but wrap the
        # inner module so the cached training graph sees the post-DDP module path.
        ht.hpu.ModuleCacher(max_graphs=args.graph_max_graphs)(
            model=model.module,
            inplace=True,
            have_grad_accumulation=use_graph_grad_accum,
            log_frequency=200,
        )
        htcore.mark_step()
        torch.hpu.synchronize()
        if is_main:
            print(
                "ModuleCacher ready "
                f"(training HPU graphs, max_graphs={args.graph_max_graphs}, "
                f"grad_accum_graphs={use_graph_grad_accum}).",
                flush=True,
            )

    # Graph compilation path is kept as an opt-in fallback only. Training HPU
    # graphs should use ModuleCacher in lazy mode; torch.compile is more
    # environment-sensitive and has hit recipe-compiler ABI issues on Sol.
    if graph_mode == "compile":
        model = torch.compile(model, backend="hpu_backend")
        htcore.mark_step()
        torch.hpu.synchronize()
        if is_main:
            print("torch.compile ready (hpu_backend).", flush=True)
    elif graph_mode == "none":
        htcore.mark_step()
        torch.hpu.synchronize()
        if is_main:
            print("Lazy mode ready (--no-hpu-graphs).", flush=True)

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
        inp = inp.unsqueeze(0).to(device, non_blocking=True)
        tgt = tgt.unsqueeze(0).to(device, non_blocking=True)
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

    # --- Optimizer: FusedAdamW ---
    # Drop-in numerical match to torch.optim.AdamW but runs as a single fused
    # HPU kernel instead of many small per-tensor kernels.
    max_lr = cfg.get("learning_rate", 3e-4)
    optimizer = FusedAdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )
    if resume_ckpt is not None:
        optimizer.load_state_dict(resume_ckpt["optimizer"])

    max_epochs = cfg.get("max_epochs", 10)
    max_steps = max_epochs * (len(train_loader) // grad_accum_steps)
    if args.max_steps_override is not None:
        max_steps = min(max_steps, args.max_steps_override)
    warmup_steps = cfg.get("warmup_steps", 500)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    use_amp = cfg.get("use_amp", True)

    ckpt_dir = os.path.join(base_dir, cfg.get("checkpoint_dir", "checkpoints"))
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    log_every = cfg.get("log_every", 100)
    save_every = cfg.get("save_every", 2000)
    grad_clip = cfg.get("grad_clip", 1.0)
    val_mark_every = 16  # one htcore.mark_step() per N validation micro-batches

    if is_main:
        print(
            f"\nStarting training: {max_epochs} epochs, {len(train_loader)} micro-batches/epoch, "
            f"{max_steps} optimizer steps "
            f"(grad_accum={grad_accum_steps}, world_size={world_size}, "
            f"global_batch={batch_size * world_size * grad_accum_steps})\n"
        )

    if args.profile_phases:
        if is_main:
            print("--profile-phases: timing dl/fwd/bwd/opt/mark over 25 micro-batches\n", flush=True)
        model.train()
        train_sampler.set_epoch(0)
        # warm-up: 5 micro-batches so graph cache is hot before timing
        warmup_n = 5
        profile_n = 20
        t_dl = t_fwd = t_bwd = t_opt = t_mark = 0.0
        it = iter(train_loader)
        for i in range(warmup_n + profile_n):
            measure = i >= warmup_n
            is_last_accum = ((i + 1) % grad_accum_steps) == 0
            sync_ctx = contextlib.nullcontext() if is_last_accum else model.no_sync()
            if use_graph_grad_accum:
                _set_graph_iteration_count(model, 0 if (i % grad_accum_steps) == 0 else 1)

            if measure:
                torch.hpu.synchronize()
            t0 = time.time()
            input_ids, labels = next(it)
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if measure:
                torch.hpu.synchronize()
                t_dl += time.time() - t0

            with sync_ctx:
                if measure:
                    t0 = time.time()
                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                    loss = loss / grad_accum_steps
                if measure:
                    torch.hpu.synchronize()
                    t_fwd += time.time() - t0
                    t0 = time.time()
                loss.backward()
                if measure:
                    torch.hpu.synchronize()
                    t_bwd += time.time() - t0

            if measure:
                t0 = time.time()
            htcore.mark_step()
            if measure:
                torch.hpu.synchronize()
                t_mark += time.time() - t0

            if is_last_accum:
                if measure:
                    t0 = time.time()
                _device_clip_grad_norm(model.parameters(), grad_clip)
                optimizer.step()
                htcore.mark_step()
                optimizer.zero_grad(set_to_none=True)
                if measure:
                    torch.hpu.synchronize()
                    t_opt += time.time() - t0

        if is_main:
            total = t_dl + t_fwd + t_bwd + t_opt + t_mark
            print(
                f"profile over {profile_n} micro-batches (after {warmup_n} warm-up):\n"
                f"  dl={t_dl:.2f}s  fwd={t_fwd:.2f}s  bwd={t_bwd:.2f}s  "
                f"opt={t_opt:.2f}s  mark={t_mark:.2f}s  total={total:.2f}s\n"
                f"  avg per micro-batch: {total/profile_n*1000:.0f} ms",
                flush=True,
            )
        cleanup_dist()
        return

    stop = False
    for epoch in range(start_epoch, max_epochs):
        if stop:
            break
        model.train()
        train_sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device, dtype=torch.float32)
        token_sum = torch.zeros((), device=device, dtype=torch.float32)
        tokens_since_log = torch.zeros((), device=device, dtype=torch.float32)
        t0 = time.time()
        batch_idx = -1
        first_step_t0 = time.time() if epoch == start_epoch else None

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            is_last_accum = (batch_idx + 1) % grad_accum_steps == 0
            if use_graph_grad_accum:
                _set_graph_iteration_count(
                    model, 0 if (batch_idx % grad_accum_steps) == 0 else 1
                )
            # Skip DDP all-reduce on the first (grad_accum-1) micro-batches.
            # With eager HCCL (PT_HPU_ENABLE_LAZY_COLLECTIVES=0), this eliminates
            # 3 of every 4 blocking collective launches per optimizer step.
            sync_ctx = contextlib.nullcontext() if is_last_accum else model.no_sync()

            with sync_ctx:
                with torch.autocast(device_type="hpu", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                    loss = loss / grad_accum_steps
                loss.backward()
            # Flush the micro-batch graph. Without this, no_sync() removes DDP's
            # per-step all-reduce (which used to implicitly dispatch), so all 8
            # accum micro-batches queue up in lazy mode — peak activation memory
            # becomes 8× larger and OOMs (slurm.51590531: 192 MB attention-scores
            # tensor, 18 layers × 8 micro-batches = 27 GB).
            htcore.mark_step()

            if first_step_t0 is not None and batch_idx == 0:
                torch.hpu.synchronize()
                first_step_dt = time.time() - first_step_t0
                if is_main:
                    print(f"  first-step compile/warmup: {first_step_dt:.1f}s", flush=True)
                first_step_t0 = None
                t0 = time.time()

            with torch.no_grad():
                n_tokens = (labels != PAD_ID).sum().to(torch.float32)
                loss_sum += loss.detach() * grad_accum_steps * n_tokens
                token_sum += n_tokens
                tokens_since_log += n_tokens

            if is_last_accum:
                lr = get_lr(global_step, warmup_steps, max_steps, max_lr)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                _device_clip_grad_norm(model.parameters(), grad_clip)
                optimizer.step()
                htcore.mark_step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if is_main and global_step % log_every == 0:
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
                            "model": _unwrap(model).state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                            "step": global_step,
                            "config": model_cfg,
                        },
                        path,
                    )
                    print(f"  checkpoint saved: {path}")

                if args.max_steps_override is not None and global_step >= args.max_steps_override:
                    if is_main:
                        print(f"Reached max_steps_override={args.max_steps_override}, stopping.")
                    stop = True
                    break

        if stop:
            break

        if batch_idx >= 0 and (batch_idx + 1) % grad_accum_steps != 0:
            lr = get_lr(global_step, warmup_steps, max_steps, max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            _device_clip_grad_norm(model.parameters(), grad_clip)
            optimizer.step()
            htcore.mark_step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

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
            for i, (input_ids, labels) in enumerate(val_loader):
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
                if (i + 1) % val_mark_every == 0:
                    htcore.mark_step()
        htcore.mark_step()

        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_sum, op=dist.ReduceOp.SUM)
        val_avg = (val_loss_sum / val_token_sum).item()

        if is_main:
            print(f"  val loss: {val_avg:.4f}\n")

        if is_main:
            path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
            torch.save(
                {
                    "model": _unwrap(model).state_dict(),
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
