---
name: gaudi2-optimization
description: Optimize PyTorch code for Intel Gaudi / Gaudi2 / Gaudi3 HPU accelerators. Use this skill whenever the user mentions Gaudi, HPU, Habana, habana_frameworks, Synapse, running models on Intel AI accelerators, or when editing code that imports habana_frameworks.torch. Also use when the user says their Gaudi code is slow, host-bound, has dynamic shape recompiles, or when they ask about HPU graphs, wrap_in_hpu_graph, ModuleCacher, mark_step, lazy mode, FusedSDPA, or bf16/fp8 on HPU. Treat GPU-style PyTorch code targeting HPU as a red flag that warrants applying this skill even if the user does not ask for optimization explicitly. This skill also includes project-specific context for the chess transformer repo at sonuishaq67/chess (ASU Sol, 8 Gaudi2 HPUs, SLURM) - when working in that repo or on files under ~/chess, src/model/transformer.py, src/training/train*.py, configs/model.yml, or the train_gaudi*.sbatch scripts, read references/chess-transformer.md first.
---

# Gaudi2 Optimization

Intel Gaudi / Gaudi2 / Gaudi3 performs best when the workload runs as a static graph. Naive PyTorch code (written as if for CUDA) will often be host-bound on Gaudi because the host has to dispatch every op and recompile when shapes change. This skill covers the patterns that matter in practice.

## Project context: chess transformer

If the working directory is `~/chess`, or the user references `src/model/transformer.py`, `src/training/train.py`, `src/training/train_fast.py`, `configs/model.yml`, `configs/training.yml`, or `train_gaudi*.sbatch`, read `references/chess-transformer.md` before editing. That file documents the specific setup (ASU Sol, 8 Gaudi2 HPUs, SLURM partition `gaudi`, torch 2.9.0 with habana-torch-plugin, chunk-packed static shapes of `(B, 256)`, vocab 1972, RoPE, DDP with `no_sync` gradient accumulation) and lists the exact changes that would most help performance on that codebase.

## Core mental model

Gaudi has two execution modes. Pick one and commit to its rules.

1. **Lazy mode** (`PT_HPU_LAZY_MODE=1`). Ops are accumulated and flushed on `mark_step()` or on a CPU read. Works with `wrap_in_hpu_graph`, `make_graphed_callables`, `ModuleCacher`. This is the mode that all the older Habana tutorials assume. It is being deprecated but is still the path with the most features and the one you should use for Gaudi2 optimization work today.
2. **Eager + torch.compile**. Standard PyTorch 2.x path. No HPU graph wrapper needed, `torch.compile` handles the graph. Fewer manual knobs but less mature on Gaudi.

Rule of thumb for existing Gaudi2 code: **if you see `import habana_frameworks.torch` and `mark_step`, you are in Lazy mode. Stay in Lazy mode and apply HPU graphs. Do not mix torch.compile and wrap_in_hpu_graph.**

## When the user says "it is slow"

Walk through this checklist before touching the model.

1. Is the model host-bound? Look for a profile trace where HPU device time is small relative to wall time, or where the device has visible idle gaps between ops. Host-bound is the condition HPU graphs fix. If the model is already device-bound, HPU graphs will not help much. Look at FusedSDPA, bf16, batch size, and FP8 instead.
2. Are there dynamic shapes? Print shapes of tensors entering the compiled region across iterations. If shapes change every step, you will recompile every step and HPU graphs will fail to capture. Fix dynamicity first (padding, bucketing, `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1`).
3. Is anything falling back to CPU? Set `PT_HPU_PT_COMPILE_ONLY_MODE=False` and `PT_HPU_LOG_MOD_MASK=0x08` plus enable CPU fallback logs to see which ops are running on CPU. Any CPU op inside the capture region kills HPU graphs.
4. Is `mark_step()` in the right places? In a training loop it goes after `loss.backward()` and after `optimizer.step()`. With `wrap_in_hpu_graph` for inference you do not need a `mark_step` after `model()` because it is implicit.

## Quick wins that apply to almost every model

Apply these before doing anything fancy.

- **bf16 autocast**: wrap forward in `torch.autocast("hpu", dtype=torch.bfloat16)`. This is the default precision for Gaudi2 and often doubles throughput with no accuracy loss for modern architectures.
- **FusedSDPA**: replace `torch.nn.functional.scaled_dot_product_attention` with `from habana_frameworks.torch.hpex.kernels import FusedSDPA` and `FusedSDPA.apply(q, k, v, mask, dropout_p, is_causal, scale, softmax_mode)`. This is the Gaudi flash attention kernel and is a large win for transformer workloads.
- **Fused optimizer**: use `from habana_frameworks.torch.hpex.optimizers import FusedAdamW` instead of `torch.optim.AdamW`. Note epsilon default is `1e-6`, not `1e-8`.
- **Fused clip norm**: use `from habana_frameworks.torch.hpex.normalization import FusedClipNorm` when gradient clipping.
- **Non-blocking H2D copies**: use `tensor.to("hpu", non_blocking=True)` followed by `htcore.mark_step()` so the host can keep building the next graph while the copy runs.

## HPU Graphs for inference

This is the highest-leverage change for latency-sensitive inference. One line wraps the whole model.

```python
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

model = GetModel().eval().to("hpu")
model = ht.hpu.wrap_in_hpu_graph(model)

# Warmup so the graph is captured on the expected input shape.
# Do this for every distinct input shape you will see at runtime.
with torch.no_grad():
    for _ in range(3):
        _ = model(torch.randn(1, 3, 224, 224, device="hpu"))

# Inference loop.
with torch.no_grad():
    for batch in loader:
        batch = batch.to("hpu", non_blocking=True)
        htcore.mark_step()
        out = model(batch)  # No mark_step needed after, it is implicit.
```

Rules that will bite if violated.

- Input shapes must be static across calls. If batch size or sequence length varies, either pad to a fixed shape or call `wrap_in_hpu_graph` with `disable_tensor_cache=True` and expect one capture per distinct shape.
- No data-dependent control flow inside the model. `if tensor.item() > 0` and `while cond.any()` will break capture.
- No CPU ops inside the captured region. Everything must be on HPU. Look for stray `.cpu()` calls, numpy conversions, or `print(tensor)` calls.
- Multi-card DeepSpeed inference with HPU Graphs needs `PT_HPU_ENABLE_LAZY_COLLECTIVES=true`.

For more detail including shape-varying inference and DeepSpeed see `references/hpu-graphs-inference.md`.

## HPU Graphs for training

Three APIs, pick the highest-level one that fits.

1. **`ModuleCacher`** - handles dynamic inputs by caching a separate graph per shape and per control-flow path. Accepts non-tensor and non-positional args. Start here for training.
2. **`make_graphed_callables`** - wraps callables. Similar to the CUDA API of the same name. Good when you want tighter control over what is captured.
3. **`capture_begin` / `capture_end` / `replay`** - low-level. Use only when the above are not flexible enough.

`ModuleCacher` example:

```python
import torch
import habana_frameworks.torch as ht
import habana_frameworks.torch.core as htcore

model = MyModel().to("hpu")
optimizer = FusedAdamW(model.parameters(), lr=1e-4)

# Wrap for automatic graph caching across shapes and branches.
model = ht.hpu.ModuleCacher()(model=model, inplace=True)

for batch in loader:
    inputs = batch["inputs"].to("hpu", non_blocking=True)
    labels = batch["labels"].to("hpu", non_blocking=True)

    with torch.autocast("hpu", dtype=torch.bfloat16):
        logits = model(inputs)
        loss = loss_fn(logits, labels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    htcore.mark_step()
    optimizer.step()
    htcore.mark_step()
```

If you are using `optimum-habana` with a Hugging Face `Trainer`, pass `--use_hpu_graphs_for_training True` and `--distribution_strategy fast_ddp`. These flags wire up `ModuleCacher` and replace `DistributedDataParallel` with `optimum.habana.distributed.all_reduce_gradients`, which is usually faster.

For the `make_graphed_callables` and low-level capture / replay APIs see `references/hpu-graphs-training.md`.

## Environment variables worth knowing

Set these before `import habana_frameworks.torch` or `import torch` (with autoload).

| Variable | Effect |
|---|---|
| `PT_HPU_LAZY_MODE=1` | Enable Lazy mode. Required for HPU graph APIs. |
| `PT_HPU_ENABLE_REFINE_DYNAMIC_SHAPES=1` | Auto bucket and pad dynamic shapes. Reduces recompiles when input shapes vary. |
| `PT_HPU_RECIPE_CACHE_CONFIG` | Disk recipe cache. Essential for multi-card so cards share compiled graphs. |
| `PT_HPU_ENABLE_LAZY_COLLECTIVES=true` | Needed for HPU Graphs with DeepSpeed multi-card inference. |
| `PT_HPU_EAGER_ENABLE_GRADIENT_VIEW_LAYOUT_OPT=1` | Speeds up vision models in Eager mode. |
| `PT_HPU_HUGE_PAGES_LIMIT_MB` | Override per-process huge page allocation. |
| `RUNTIME_SCALE_PATCHING=1` | Cuts FP8 warmup time by a lot. Trades 5-20% throughput. |

## Common pitfalls and how to fix them

- **"AttributeError: module 'habana_frameworks.torch.hpu' has no attribute 'wrap_in_hpu_graph'"** - The code imported wrong or the Habana installation is broken. Use `import habana_frameworks.torch as ht` then `ht.hpu.wrap_in_hpu_graph(model)`. If that fails, the Docker image version does not match the expected API. Check `hl-smi` and the installed `habana_frameworks` version.
- **"... is not supported during HPU Graph capturing"** - An op in the model is not capturable. Usually a CPU op or a data-dependent op. Find and replace it.
- **Training looks correct but is slower with HPU graphs** - Likely shape-varying inputs causing a capture per step. Check shapes or switch to `ModuleCacher`.
- **OOM after wrapping in HPU graph** - HPU graphs hold fixed memory for captures. If you cache many shapes you will run out. Reduce the shape variety or lower batch size.
- **First iterations are very slow, then fine** - This is graph compilation. Expected. Warm up with representative inputs before timing.

## Diagnosing host-bound workloads

Before reaching for HPU graphs, confirm the model is actually host-bound.

```bash
# Gaudi TensorBoard plugin, then open http://localhost:6006
pip install habana-torch-plugin
# or use Perfetto on a captured trace
```

In code, a quick smoke test:

```python
import time, torch, habana_frameworks.torch.core as htcore
torch.hpu.synchronize()
t0 = time.perf_counter()
for _ in range(100):
    out = model(x)
htcore.mark_step()
torch.hpu.synchronize()
print((time.perf_counter() - t0) / 100 * 1000, "ms/iter")
```

Compare with and without `wrap_in_hpu_graph`. If the wrapped version is 2x or more faster, the model was host-bound.

## Reference files

Go deeper when the situation calls for it.

- `references/chess-transformer.md` - **read first when working in the `~/chess` repo.** Project-specific layout, env vars already set in SLURM, FusedSDPA wiring for the model, RoPE CPU fallback audit, DDP `no_sync` pattern, training loop checklist, expected tensor shapes, throughput targets.
- `references/hpu-graphs-inference.md` - shape-varying inference, DeepSpeed multi-card, async copy patterns, and the Habana `Gaudi_inference_ex2` / `ex3` tutorial walkthroughs.
- `references/hpu-graphs-training.md` - `make_graphed_callables`, low-level `capture_begin` / `capture_end` / `replay`, dynamicity handling with `ModuleCacher`.
- `references/performance-optimization.md` - FusedSDPA modes and flags, FP8 with Intel Neural Compressor, CPU fallback detection, `pipelining_fwd_bwd`, `fast_ddp`, recipe caching.
- `references/troubleshooting.md` - common errors, shape bucketing, diagnosing recompiles, and profiling workflows.

## Source

Material is adapted from the Habana Gaudi tutorials at https://github.com/HabanaAI/Gaudi-tutorials (archived by Intel in Sep 2025 but still the canonical examples) and the Intel Gaudi docs at https://docs.habana.ai. When in doubt, link the user to the specific tutorial notebook that matches their workload (inference, training, profiling, DeepSpeed, vLLM, TGI).
