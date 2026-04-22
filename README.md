# Chess Transformer

A decoder-only transformer trained on Lichess games for next-move prediction. Chess as autoregressive language modeling over UCI move sequences — no board state, no search, just the move stream.

The bot is live on Lichess.

## At a glance

| | |
|---|---|
| **Task**         | Next-move prediction on UCI sequences |
| **Architecture** | Decoder-only transformer, RoPE, pre-norm, SiLU FFN, causal SDPA |
| **Vocabulary**   | 1,972 tokens (1,968 legal UCI moves + PAD/BOS/EOS/MASK) |
| **Context**      | 256 tokens |
| **Trained model**| 205M params (d_model=1024, 16 layers, 16 heads), 1 epoch on 1× A100 |
| **Scaling run**  | 512M config (d_model=1536, 18 layers, 24 heads) on 8× Intel Gaudi2 |
| **Dataset**      | HuggingFace `Lichess/standard-chess-games`, filtered: both Elo > 2200, base time ≥ 180 s, ≥ 20 moves |
| **Filtered set** | ~37.3M games, ~3.1B tokens, 9,635 shard pairs |
| **Deployment**   | ONNX INT8 on AWS EC2 c6i.xlarge (eu-west-3), ~50 ms CPU inference |
| **Bot status**   | Live on Lichess, systemd-managed, per-game logs |

## Pipeline

```
HuggingFace parquet  ──►  DuckDB filter  ──►  python-chess (PGN→UCI)
                                                     │
                                                     ▼
                                      Tokenize → .bin/.idx memmap
                                                     │
                                                     ▼
       ┌──────────────────────────────────────────────┴──────────────────────────────────┐
       │                                                                                 │
       ▼                                                                                 ▼
  Train on Sol A100 (CUDA)                                              Train on Sol Gaudi2 (HPU, Apptainer)
       │                                                                                 │
       └────────────────────────────► Checkpoint ◄───────────────────────────────────────┘
                                           │
                                           ▼
                              ONNX export + INT8 quantization
                                           │
                                           ▼
                                AWS EC2 CPU ──► Lichess Bot API
```

## Repo layout

```
chess/
├── src/
│   ├── data/         # extract_zst.py, uci_tokenizer.py, build_packing.py, query_parquet.py
│   ├── model/        # transformer.py (RoPE, pre-norm, SDPA), embeddings.py
│   ├── training/     # train.py (CUDA), train_fast.py (Gaudi/HPU), dataset.py, distributed.py
│   └── serving/      # export.py (ONNX + INT8), inference.py, bot.py (Lichess API client)
├── configs/          # model.yml, training.yml
├── deploy/           # systemd unit files (chess-bot.service, crash-alert notifier)
├── models/           # model.onnx, model_opt.onnx, model_int8.onnx
├── chess-gaudi.def   # Apptainer def for Habana container (host driver 1.23.0)
├── train.sbatch                   # single A100 on Sol
├── train_gaudi.sbatch             # bare-metal Gaudi (lazy mode only, fallback)
├── train_gaudi_fast.sbatch        # bare-metal Gaudi with graph compile (broken on current Sol host libs)
├── train_gaudi_apptainer.sbatch   # containerized Gaudi (current primary path)
├── cli.sh            # setup / check / data / tokenize / train / deploy wrappers
└── roadmap/README.md # original task roadmap (kept for history)
```

## Commands

```bash
./cli.sh setup                          # conda env + torch CUDA 12.4 + pip deps
./cli.sh check                          # verify imports
./cli.sh data                           # download HF parquet → filter → UCI
./cli.sh tokenize                       # UCI text → uint16 .bin + .idx memmap
./cli.sh train                          # single-GPU CUDA
./cli.sh train '' '' --resume checkpoints/step_2000.pt
./cli.sh train '' '' --dry-run

# Gaudi / Apptainer on Sol
apptainer build /scratch/$USER/sif/chess-gaudi.sif chess-gaudi.def
sbatch train_gaudi_apptainer.sbatch

# Deployment
./cli.sh deploy-install                 # provision EC2
./cli.sh deploy-setup
./cli.sh bot-start | bot-stop | bot-status | bot-logs
```

## Notes on what's in here

- **RoPE** is implemented two ways — as complex-exponential buffer for CUDA training, and rewritten in real arithmetic for the ONNX export (complex ops don't survive the converter).
- **Data pipeline** uses DuckDB predicate pushdown to scan parquet without materializing full columns, then hands filtered PGN movetext to `python-chess` for SAN→UCI conversion. Processing runs in a `ProcessPoolExecutor` (threads were GIL-bound).
- **DataLoader** stores chunk index as flat numpy arrays so forked workers share it via copy-on-write. Python lists of tuples fork-duplicated ~14 GB per worker.
- **Gaudi training** targets 8× HPUs with DDP. The bare-metal path hits a `GraphStorage::get` ABI mismatch in the Sol host userspace; the containerized path (Apptainer + Habana's official 1.23.0 PyTorch image) is the working route.
- **ONNX model** ships INT8-quantized and runs on 4 vCPUs with enough headroom for realtime Lichess play.

## Gotchas I hit

Battle scars from actually running this end-to-end. Writing them down so the next person (or the next me) doesn't re-lose a weekend to the same footguns.

1. **`ThreadPoolExecutor` for PGN parsing does almost nothing.** `python-chess`'s `read_game` is pure Python → GIL-bound. `--workers 32` gave roughly one core of real parsing while the Slurm accounting looked busy. Switched the data pipeline to `ProcessPoolExecutor`.

2. **DataLoader workers fork-duplicate Python containers.** With 16 workers and ~14M chunk entries stored as a Python list of tuples, each forked worker copied ~14 GB because CPython refcounts touch every object page and defeat copy-on-write. Flattened the index into two numpy arrays (`_packed`, `_bounds`) — now actually shared across workers.

3. **`PYTHONUNBUFFERED=1` is not optional on SLURM.** Without it, Python block-buffers stdout into the SLURM log and the job looks frozen for hours while the GPU sits at 100%. Diagnosed by `nvidia-smi` showing activity with zero visible log progress.

4. **Sol's cgroup accounting lies on threaded Python jobs.** A job that actually processed thousands of files came back with a completion email reporting `Max Memory Used: 0 B`, `% User: 0.00%`, `Max Disk Write: 0 B`. The cgroups sampler sometimes misses threaded Python entirely on certain nodes. Trust `slurm.out` and the scratch contents, not the email.

5. **Gaudi sbatch scripts must use `#!/bin/bash -l` (login shell).** Non-login shells don't source `/etc/profile.d/habanalabs.sh`, so `GC_KERNEL_PATH` stays unset and every graph compile dies with the unhelpful `synStatus 26 [Generic failure]`. This is the #1 silent footgun for Gaudi on Sol.

6. **Apptainer + Habana device binds are subtle.** Sol's Gaudi2 uses the newer `accel` kernel subsystem (`/dev/accel/accel0..7`, no `/dev/hl*`). Three traps: (a) `--bind /dev/accel:/dev/accel` (directory) lets `hl-smi` see 8 HPUs but breaks `synDeviceAcquire` with `synStatus=8 [Device not found]` — bind each `/dev/accel/*` file individually instead; (b) `--bind /dev:/dev` (whole /dev) breaks `/dev/null` and `/dev/urandom` under SLURM's cgroup view; (c) `bash -lc` inside `apptainer exec` retriggers the same `/dev/null: Permission denied` via `/etc/profile.d/modules.sh`. Use `bash -c` and explicitly source `/etc/profile.d/habanalabs*.sh` inside the container for `GC_KERNEL_PATH` and friends.

7. **Bare-metal Gaudi graph compile is currently busted on Sol.** `torch.compile(backend="hpu_backend")` against `/packages/envs/pytorch-2.9.0-gaudi` fails with `_recipe_compiler_C.so: undefined symbol: _ZN6habana5graph12GraphStorage3getEv` — host userspace ABI mismatch. Lazy mode still works bare-metal as a fallback. The container path (host driver 1.23.0 matched to `pytorch-installer-2.9.0` image) is the real fix.

8. **RoPE as complex exponentials does not survive ONNX export.** The PyTorch version precomputes `exp(i·mθ)` into a complex buffer, which is fine for CUDA training but the ONNX converter has no complex-tensor support. Export path uses a second RoPE implementation in real arithmetic, plus manual attention instead of SDPA.

See `roadmap/README.md` for the original phase-by-phase task breakdown used while building this out.
