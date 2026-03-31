import argparse
import math
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import yaml

from src.model import Transformer, TransformerConfig
from src.training.dataset import ChessDataset, PAD_ID


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_model(model_cfg: dict, device: torch.device) -> Transformer:
    config = TransformerConfig(**model_cfg)
    model = Transformer(config).to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    return model


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
    parser.add_argument(
        "--resume", default=None, help="path to checkpoint to resume from"
    )
    parser.add_argument(
        "--max-files", type=int, default=None, help="limit number of .bin/.idx files to load"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="load data, print sample, run one batch, then exit"
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = load_yaml(args.model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Data ---
    base_dir = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
    bin_dir = os.path.join(base_dir, cfg.get("bin_dir", "dataset/bin"))

    dataset = ChessDataset(
        bin_dir, max_seq_len=model_cfg.get("max_seq_len", 256), max_files=args.max_files
    )
    print(f"Total chunks: {len(dataset):,}")

    # Train/val split
    val_frac = cfg.get("val_split", 0.05)
    val_size = int(len(dataset) * val_frac)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Train chunks: {train_size:,}  Val chunks: {val_size:,}")

    batch_size = cfg.get("batch_size", 64)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # --- Model ---
    model = build_model(model_cfg, device)

    # --- Dry run ---
    if args.dry_run:
        import json

        vocab_path = os.path.join(base_dir, "dataset", "vocab.json")
        id2tok = {v: k for k, v in json.loads(open(vocab_path).read()).items()}

        print("\n--- Sample chunk (first 3) ---")
        for i in range(min(3, len(dataset))):
            inp, tgt = dataset[i]
            tokens = [id2tok.get(t.item(), "?") for t in inp]
            # Show first 20 tokens, skip PAD at the end
            visible = [t for t in tokens if t != "PAD"][:20]
            print(f"  chunk {i}: {len(visible)} non-pad tokens | {' '.join(visible)} ...")

        print("\n--- One forward pass ---")
        inp, tgt = dataset[0]
        inp = inp.unsqueeze(0).to(device)
        tgt = tgt.unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            loss = nn.CrossEntropyLoss(ignore_index=PAD_ID)(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1)
            )
        print(f"  input shape:  {inp.shape}")
        print(f"  logits shape: {logits.shape}")
        print(f"  loss: {loss.item():.4f} (random init ~{math.log(model_cfg.get('vocab_size', 1972)):.2f})")
        print("\nDry run complete.")
        return

    # --- Optimizer ---
    max_lr = cfg.get("learning_rate", 3e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=cfg.get("weight_decay", 0.1),
        betas=(0.9, 0.95),
    )

    max_epochs = cfg.get("max_epochs", 10)
    max_steps = max_epochs * len(train_loader)
    warmup_steps = cfg.get("warmup_steps", 500)

    # --- Loss ---
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # --- Mixed precision ---
    use_amp = cfg.get("use_amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # --- Checkpointing ---
    ckpt_dir = os.path.join(base_dir, cfg.get("checkpoint_dir", "checkpoints"))
    os.makedirs(ckpt_dir, exist_ok=True)
    log_every = cfg.get("log_every", 100)
    save_every = cfg.get("save_every", 2000)
    grad_clip = cfg.get("grad_clip", 1.0)

    # --- Resume ---
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"]
        global_step = ckpt["step"]
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    # --- Training loop ---
    print(
        f"\nStarting training: {max_epochs} epochs, {len(train_loader)} steps/epoch, "
        f"{max_steps} total steps\n"
    )

    for epoch in range(start_epoch, max_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_tokens = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Update LR
            lr = get_lr(global_step, warmup_steps, max_steps, max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids)  # [B, seq_len, vocab]
                loss = criterion(
                    logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                )

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            # Track
            n_tokens = (labels != PAD_ID).sum().item()
            epoch_loss += loss.item() * n_tokens
            epoch_tokens += n_tokens
            global_step += 1

            if global_step % log_every == 0:
                avg = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
                elapsed = time.time() - t0
                tok_per_sec = epoch_tokens / elapsed if elapsed > 0 else 0
                print(
                    f"  step {global_step:>6d} | loss {loss.item():.4f} | "
                    f"avg {avg:.4f} | lr {lr:.2e} | "
                    f"{tok_per_sec:.0f} tok/s"
                )

            if global_step % save_every == 0:
                path = os.path.join(ckpt_dir, f"step_{global_step}.pt")
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "step": global_step,
                        "config": model_cfg,
                    },
                    path,
                )
                print(f"  checkpoint saved: {path}")

        # Epoch summary
        train_avg = epoch_loss / epoch_tokens if epoch_tokens > 0 else 0
        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch + 1}/{max_epochs} done in {elapsed:.1f}s | "
            f"train loss: {train_avg:.4f}"
        )

        # Validation
        model.eval()
        val_loss = 0.0
        val_tokens = 0
        with torch.no_grad():
            for input_ids, labels in val_loader:
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(input_ids)
                    loss = criterion(
                        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
                    )
                n_tokens = (labels != PAD_ID).sum().item()
                val_loss += loss.item() * n_tokens
                val_tokens += n_tokens

        val_avg = val_loss / val_tokens if val_tokens > 0 else 0
        print(f"  val loss: {val_avg:.4f}\n")

        # Save end-of-epoch checkpoint
        path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pt")
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch + 1,
                "step": global_step,
                "config": model_cfg,
            },
            path,
        )
        print(f"  epoch checkpoint saved: {path}")

    print("Training complete.")


if __name__ == "__main__":
    train()
