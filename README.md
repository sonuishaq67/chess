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

See `roadmap/README.md` for the original phase-by-phase task breakdown used while building this out.
