# Chess Transformer Project (sonuishaq67/chess)

Project-specific context for the chess transformer training on 8 Gaudi2 HPUs on the ASU Sol supercomputer. Read this file whenever the user references `~/chess`, `src/model/transformer.py`, `src/training/train.py`, `src/training/train_fast.py`, `src/data/`, the chess configs, or the Gaudi SLURM scripts.

## What this project is

Decoder-only transformer (GPT-style) trained on Lichess games for next-move prediction. Treats chess as autoregressive language modeling over UCI move sequences. Full pipeline from HuggingFace parquet ingest through SLURM training to ONNX export and Lichess bot deployment.

GitHub: https://github.com/sonuishaq67/chess

## Hardware and environment

- Cluster: ASU Sol supercomputer.
- Node shape: 1 node, 8 Gaudi2 HPUs (configured via SLURM `-N 1 -G 8`).
- SLURM partition `gaudi`, QOS `class_gaudi`.
- Python env: `/packages/envs/pytorch-2.9.0-gaudi` (ASU prebuilt, `habana-torch-plugin` matched to `torch 2.9.0`).
- Launcher: `torchrun --nproc_per_node=8 -m src.training.train[_fast] --config ... --model-config ...`.
- `habanalabs.sh` is auto-sourced by the `#!/bin/bash -l` login shell, so `GC_KERNEL_PATH` is set automatically. Do not try to set it manually.

## Already-configured env vars (do not re-set unless changing)

From `train_gaudi.sbatch` and `train_gaudi_fast.sbatch`:

```bash
export PT_HPU_LAZY_MODE=1                 # Lazy mode, required for HPU graph APIs
export PT_HPU_ENABLE_LAZY_COLLECTIVES=0   # Correct for DDP. Only flip to 1 if switching to DeepSpeed inference.
export PT_HPU_RECIPE_CACHE_CONFIG=/tmp/recipe_cache_${SLURM_JOB_ID},true,1024,false
```

**Recommended change**: The recipe cache is in `/tmp` which is job-scoped and wipes between jobs. Move it to a persistent scratch path so second and later runs skip compile time:

```bash
export PT_HPU_RECIPE_CACHE_CONFIG=/scratch/$USER/gaudi_recipe_cache,true,4096,false
```

First run populates the cache (slow warmup). Every subsequent run hits the cache and boots fast.

## Codebase layout

```
chess/
├── configs/
│   ├── model.yml          # d_model, n_heads, n_layers, dropout, vocab, context
│   └── training.yml       # batch_size, lr, schedule, checkpointing
├── dataset/
│   ├── bin/               # Memmap .bin + .idx (tokenized chunks)
│   └── vocab.json         # 1972 tokens = 1968 UCI moves + PAD/BOS/EOS/MASK
├── src/
│   ├── data/              # extract_zst.py, uci_tokenizer.py, build_packing.py
│   ├── model/             # transformer.py, embeddings.py
│   ├── training/
│   │   ├── train.py       # Baseline DDP training on Gaudi
│   │   ├── train_fast.py  # FusedSDPA + FusedAdamW + HPU graphs + DDP no_sync
│   │   ├── dataset.py     # Chunk-packed memmap loader
│   │   └── distributed.py # DDP setup
│   └── serving/           # ONNX export, Lichess bot
├── train_gaudi.sbatch     # Baseline 16h job
└── train_gaudi_fast.sbatch # Fast path 3-day job
```

## Model specifics that matter for HPU optimization

From `configs/model.yml` and the README:

| Setting | Value | HPU implication |
|---|---|---|
| Vocab size | 1972 | Tiny output projection (~0.5M params). Cross-entropy is cheap. |
| Context length | 256 moves | Static. Perfect for HPU graphs. |
| Positional encoding | RoPE | Check RoPE impl for CPU fallback (see below). |
| Attention | `F.scaled_dot_product_attention` | **Replace with FusedSDPA**. See below. |
| FFN activation | SiLU | HPU-native. No issue. |
| Norm | Pre-norm LayerNorm | HPU-native. No issue. |
| Dropout | 0.1 | HPU-native. No issue. |
| d_model sizes | 128 / 256 / 768 | Full size (768) is where graphs pay off most. |
| Head count | 4 / 8 / 12 | Ensure d_model / heads is multiple of 8 for best FusedSDPA performance. |
| Layer count | 4 / 8 / 12 | More layers = more host dispatch, so more benefit from HPU graphs. |

All input shapes are static because of chunk packing (every batch is (B, 256) input_ids). This is the ideal case for `wrap_in_hpu_graph` or `ModuleCacher`.

## Critical change 1: FusedSDPA

The README's Phase 4 uses PyTorch's `scaled_dot_product_attention`. On HPU that runs but does not always route to the tiled flash attention kernel. Replace it explicitly.

In `src/model/transformer.py`, find the attention block:

```python
# BEFORE
out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)

# AFTER
from habana_frameworks.torch.hpex.kernels import FusedSDPA

out = FusedSDPA.apply(
    q, k, v,
    None,                 # attn_mask, None because is_causal=True covers it
    self.dropout if self.training else 0.0,
    True,                 # is_causal
    None,                 # scale, None uses default 1/sqrt(d_head)
    'fast',               # softmax_mode. 'fast' for bf16 causal, 'None' for default
)
```

Gate the import so the model still runs on CUDA:

```python
try:
    from habana_frameworks.torch.hpex.kernels import FusedSDPA
    _HAS_FUSED_SDPA = True
except ImportError:
    _HAS_FUSED_SDPA = False

def attention(q, k, v, dropout_p, training):
    if _HAS_FUSED_SDPA and q.device.type == "hpu":
        return FusedSDPA.apply(q, k, v, None, dropout_p if training else 0.0, True, None, 'fast')
    return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True)
```

Expected gain: 1.3x to 2x throughput on the transformer block at seq_len=256, more at longer contexts. Memory drops significantly.

Constraint: FusedSDPA requires q/k/v with the same batch size (no batch broadcasting). With standard multi-head attention this is always the case.

## Critical change 2: RoPE CPU fallback audit

RoPE is often implemented one of two ways:

1. **Multiplicative with sin/cos tables** (typical `llama`-style). Uses `torch.cos`, `torch.sin`, element-wise multiply. All HPU-native. This is fine.
2. **Complex number rotation** via `torch.polar`, `torch.view_as_complex`, `torch.view_as_real`. `torch.polar` can fall back to CPU on some Habana versions.

Check `src/model/embeddings.py` or wherever RoPE is defined. If you see `torch.polar` or `complex` tensor ops, rewrite to the sin/cos multiplication form. Reference implementation that is known HPU-safe:

```python
def apply_rope(x, cos, sin):
    # x shape: (B, H, T, D). cos/sin shape: (T, D).
    x1, x2 = x[..., ::2], x[..., 1::2]
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return x * cos + rotated * sin
```

Turn on CPU fallback logging once to verify:

```bash
LOG_LEVEL_PT_FALLBACK=1 ENABLE_CONSOLE=true torchrun --nproc_per_node=1 -m src.training.train 2>&1 | grep -i fallback | head
```

If `grep` is empty, all ops are on HPU. Turn logging back off for normal runs.

## Critical change 3: HPU graph wrapping in train_fast.py

`train_fast.py` says it uses HPU graphs. The correct place to wrap the model is after DDP wrapping but before the training loop:

```python
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

model = build_model(model_cfg).to("hpu")
model = torch.nn.parallel.DistributedDataParallel(
    model,
    device_ids=None,              # HPU does not use device_ids
    gradient_as_bucket_view=True, # Memory optimization
    broadcast_buffers=False,      # Small win if model has no buffers that drift
)

# Wrap AFTER DDP so the graph includes the collective.
# ModuleCacher handles any residual shape variety.
ht.hpu.ModuleCacher()(model=model.module, inplace=True)
```

If inputs are guaranteed static shape (they are with chunk packing), you can use `wrap_in_hpu_graph` instead for lower overhead:

```python
model.module = ht.hpu.wrap_in_hpu_graph(model.module)
```

Pick one. Do not wrap with both.

## Critical change 4: Gradient accumulation with `no_sync`

The `train_gaudi_fast.sbatch` comment mentions "no_sync". For gradient accumulation over K micro-batches, the pattern is:

```python
for step, batch in enumerate(loader):
    is_last_microbatch = (step + 1) % accum_steps == 0

    sync_ctx = model.no_sync() if not is_last_microbatch else nullcontext()
    with sync_ctx:
        with torch.autocast("hpu", dtype=torch.bfloat16):
            logits = model(batch["inputs"])
            loss = loss_fn(logits, batch["labels"]) / accum_steps
        loss.backward()
        htcore.mark_step()

    if is_last_microbatch:
        optimizer.step()
        htcore.mark_step()
        optimizer.zero_grad(set_to_none=True)
```

`no_sync()` skips the gradient all-reduce for K-1 micro-batches and does one big all-reduce on the last one. Standard DDP gradient accumulation. The two `mark_step()` calls are what give Gaudi the sync points to flush ops.

## Training loop checklist for this project

- [ ] `torch.autocast("hpu", dtype=torch.bfloat16)` around the forward. Not fp16.
- [ ] `FusedAdamW` from `habana_frameworks.torch.hpex.optimizers`, not `torch.optim.AdamW`. Remember epsilon defaults to 1e-6 not 1e-8.
- [ ] `FusedClipNorm` from `habana_frameworks.torch.hpex.normalization` for gradient clipping.
- [ ] FusedSDPA replaces `F.scaled_dot_product_attention` in the model.
- [ ] `mark_step()` after `loss.backward()`.
- [ ] `mark_step()` after `optimizer.step()`.
- [ ] `ModuleCacher` or `wrap_in_hpu_graph` wrapping `model.module` after DDP.
- [ ] DataLoader with `num_workers >= 4`, `pin_memory=True`, `persistent_workers=True`.
- [ ] No `loss.item()` every step. Accumulate in a tensor and call `.item()` every N steps only.
- [ ] Validation runs in its own wrapped module so training captures stay warm (or wrap eval forward separately).

## Things to NOT do in this repo

- **Do not add `torch.compile`** alongside `wrap_in_hpu_graph`. They are mutually exclusive. Stay on Lazy mode + HPU graphs.
- **Do not use Python `if tensor > 0` checks** in the model. Keep control flow shape-only. The README's causal mask is a pure tensor op, which is correct.
- **Do not call `.cpu()` or `.numpy()`** inside the training step. Move logging out of the hot path.
- **Do not use `fp16`** in autocast on Gaudi2. Use `bfloat16`. fp16 works but has fewer optimized kernels.
- **Do not call `all_gather` on huge logits** for validation. Compute top-k on device, all_gather the small top-k result.

## Expected tensor shapes

For context length 256 and batch size B per rank:

| Tensor | Shape |
|---|---|
| `input_ids` | `(B, 256)` int64 |
| `attention_mask` | not used, causal handled by FusedSDPA's `is_causal=True` |
| Token embedding | `(B, 256, d_model)` |
| Per-head q, k, v | `(B, n_heads, 256, d_head)` |
| Attention output | `(B, n_heads, 256, d_head)` |
| Logits | `(B, 256, 1972)` |
| Labels | `(B, 256)` int64 |
| Loss | scalar |

All shapes static. Every tensor on HPU. No bucketing needed.

## Measuring throughput and diagnosing

The metric that matters is tokens/second/HPU and total tokens/second (sum across 8 HPUs). Standard loop:

```python
import time, habana_frameworks.torch.core as htcore, torch

torch.hpu.synchronize()
t0 = time.perf_counter()
tokens_seen = 0

for step, batch in enumerate(loader):
    ...training step...
    tokens_seen += batch["inputs"].numel()
    if step > 0 and step % 100 == 0:
        torch.hpu.synchronize()
        dt = time.perf_counter() - t0
        tok_s = tokens_seen / dt
        if rank == 0:
            print(f"step {step} | {tok_s/1e3:.1f} ktok/s/HPU | {tok_s*world_size/1e3:.1f} ktok/s total")
```

Target for a small-ish transformer (d_model=768, 12 layers, seq=256, bs=64/HPU) on Gaudi2 is roughly 40-80 ktok/s/HPU after warmup. If significantly below that, open `references/troubleshooting.md` and walk the checklist.

## SLURM-specific notes

- `SLURM_GPUS_ON_NODE=8` is set by SLURM, `NPROC_PER_NODE` already picks this up.
- `HABANA_VISIBLE_MODULES` can override for partial-node runs. The sbatch scripts handle this.
- Use `squeue -u $USER` to check queue, `scancel <job_id>` to kill, `sacct -j <job_id>` for history.
- Fast path (`train_gaudi_fast.sbatch`) requests 3 days, baseline requests 16 hours. Run the fast version after confirming the baseline converges.

## Likely next optimizations in priority order

1. Verify FusedSDPA is actually wired in and benchmark before/after.
2. Move recipe cache to scratch for cross-job warm starts.
3. Increase `per_rank_batch_size` to the point just before OOM. HPU graphs pin memory so watch `hl-smi` output.
4. Enable gradient accumulation with `no_sync` if you want a bigger effective batch than fits in memory.
5. FP8 via Intel Neural Compressor for inference (production bot), not training. Most practical for when you want lower latency on the Lichess bot.

## Related files to check when editing

When editing one of these, consider the corresponding file too:

| Editing | Also check |
|---|---|
| `src/model/transformer.py` (attention) | FusedSDPA import, device guard |
| `src/model/embeddings.py` (RoPE) | CPU fallback, no complex ops |
| `src/training/train.py` or `train_fast.py` | mark_step placement, autocast, fused optimizer |
| `configs/model.yml` | d_model / n_heads divisibility by 8 |
| `configs/training.yml` | batch_size, accum_steps, eval_freq (to reduce sync overhead) |
| `train_gaudi*.sbatch` | env vars, recipe cache path |

## Useful one-liners

```bash
# Check HPU status.
hl-smi

# Check habana_frameworks version matches torch.
python -c "import habana_frameworks.torch as ht, torch; print(ht.__version__, torch.__version__)"

# CPU fallback detection pass.
LOG_LEVEL_PT_FALLBACK=1 ENABLE_CONSOLE=true python -m src.training.train --dry-run 2>&1 | grep -i fallback

# Interactive single-GPU run for debugging on a compute node.
srun -p gaudi -q class_gaudi -N 1 -G 1 -t 1:00:00 --pty bash

# Profile a few steps to tensorboard.
# Set the profiler inside the train script, then:
tensorboard --logdir logs/ --bind_all
```
