# Chess Transformer

A decoder-only transformer trained on Lichess game data for next-move prediction, treating chess as an autoregressive language modeling problem over UCI move sequences.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  HuggingFace Parquet ──► Filter ──► UCI Sequences               │
│  (Lichess/standard-chess-games)  (Elo,TC)   (python-chess)      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TOKENIZATION & PREP                         │
│  UCI Moves ──► Vocabulary (~1800 + special) ──► Integer Tensors │
│  e2e4 g1f3...    PAD BOS EOS MASK               memory-mapped   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (ASU Sol)                            │
│  Decoder-Only Transformer (GPT-style)                           │
│  Distributed training (DDP/FSDP) · Mixed precision · SLURM      │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPORT & DEPLOYMENT                           │
│  PyTorch ──► ONNX ──► ONNX Runtime ──► AWS EC2 (eu-west-3)     │
│                        optimized          Lichess Bot API        │
└─────────────────────────────────────────────────────────────────┘
```

## Current Status

| Phase | Status |
|-------|--------|
| 0. Environment & Setup | Done |
| 1. Data Download & Exploration | Done |
| 2. Data Filtering & Conversion | Done |
| 3. Tokenization & Tensor Prep | ~90% — train/val/test split remaining |
| 4. Transformer Architecture | ~80% — implemented, sanity checks remaining |
| 5. Training on Sol | ~60% — train loop + SLURM script done, distributed training remaining |
| 6. Evaluation & Analysis | Not started |
| 7. ONNX Export & Optimization | Not started (stub files) |
| 8. Deployment as Lichess Bot | Not started (stub files) |

## Data Decisions

| Parameter         | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Source            | [HuggingFace](https://huggingface.co/datasets/Lichess/standard-chess-games) parquet |
| Elo filter        | Both players > 2200                                                  |
| Time control      | Base time >= 180s (rapid/classical)                                  |
| Move format       | UCI (e.g. `e2e4`, `g1f3`)                                           |
| Vocab size        | ~1800 unique move tokens + special tokens                            |
| Min moves         | 20 per game (filters short resignations/aborts)                      |
| Termination       | Checkmate, resignation, draws (agreement/stalemate/repetition). Exclude abandoned/timeout. |

## Infrastructure

- **Training**: ASU Sol supercomputer (SLURM, multi-GPU)
- **Export**: ONNX format
- **Deployment**: AWS EC2 in eu-west-3 (France), Lichess Bot API

---

## Task Roadmap

### Phase 0: Environment & Setup

- [x] Set up Python environment (venv or conda) with core dependencies
  - [x] `python-chess` — PGN parsing and move conversion to UCI
  - [x] `zstandard` — streaming decompression of .zst files
  - [x] `torch` — model development and training
  - [x] `tokenizers` (HuggingFace) — vocabulary building
  - [x] `onnx` / `onnxruntime` — model export and inference
- [x] Get Sol supercomputer access and test job submission
- [x] Learn SLURM basics: `sbatch`, `srun`, `squeue`, `scancel`, resource requests
- [x] Set up project directory structure:
  ```
  chess/
  ├── dataset/
  │   ├── parquet/       # downloaded HF parquet files
  │   ├── uci/           # filtered games as UCI text (one game per line)
  │   ├── bin/           # tokenized binary memmap (.bin + .idx)
  │   └── vocab.json     # token↔id mapping (1972 tokens)
  ├── src/
  │   ├── data/          # extract_zst.py, query_parquet.py, uci_tokenizer.py
  │   ├── model/         # transformer.py, embeddings.py
  │   ├── training/      # train.py, dataset.py, distributed.py, slurm_job.sh
  │   └── serving/       # export.py, inference.py, bot.py
  ├── configs/           # model.yml, training.yml
  ├── cli.sh             # project CLI (setup, data, tokenize, etc.)
  └── README.md
  ```
- [x] Set up version control and `.gitignore` (exclude data, checkpoints, .zst files)

### Phase 1: Data Download & Exploration

- [x] Download parquet files from HuggingFace (Lichess/standard-chess-games)
  - [x] Use `huggingface_hub` for downloads
  - [x] Parquet format provides structured columns for efficient filtering
- [x] Explore a single sample file to understand structure
  - [x] Parse PGN headers: `WhiteElo`, `BlackElo`, `TimeControl`, `Termination`, `Result`
  - [x] Understand move representation in PGN (SAN) vs UCI
  - [x] Check metadata availability and consistency across months/years
- [x] Estimate total game count across all files
- [x] Estimate filtered game count (both players > 1900, base time >= 180s)
- [x] Analyze termination reasons and their distribution
- [x] Decide final filter thresholds based on exploration findings
  - [x] Minimum move count for resignations
  - [x] Any additional filters worth applying

### Phase 2: Data Filtering & Conversion

- [x] Build filtering pipeline from HuggingFace parquet files
  - [x] Filter by Elo: both `WhiteElo` and `BlackElo` > 1900
  - [x] Filter by time control: base time >= 180 seconds
  - [x] Filter by termination: keep checkmate, resignation, draws (agreement/stalemate/repetition); exclude abandoned/timeout
  - [x] Apply minimum move threshold for resignations
- [x] Convert filtered PGN movetext (SAN) to UCI using `python-chess`
  - [x] Handle edge cases: promotions (`e7e8q`), castling, en passant
- [x] Output format: one game per line, UCI moves space-separated
  - [x] Include metadata line or separate file (Elo, result, date) for later analysis
- [x] Run pipeline across all downloaded files
  - [x] Parallelize across parquet files
- [x] Compute dataset statistics:
  - [x] Total filtered games
  - [x] Average / median game length (in moves)
  - [x] Move frequency distribution
  - [x] Elo distribution of filtered games
  - [x] Result distribution (white win / black win / draw)

### Phase 3: Tokenization & Tensor Preparation

- [x] Build UCI move vocabulary
  - [x] Enumerate all unique UCI moves seen in filtered data
  - [x] Add special tokens: `PAD`, `BOS`, `EOS`, `MASK`
  - [x] Save vocabulary mapping (token → id, id → token)
  - [x] Verify vocab size (~1800 + specials)
- [x] Tokenize all games: UCI move strings → integer sequences
  - [x] Prepend `BOS`, append `EOS` to each game
- [ ] Create train / validation / test split
  - [ ] Split by date (preferred) or random
  - [ ] Typical split: 95% / 2.5% / 2.5% or similar
- [x] Pack tokenized sequences into efficient storage
  - [x] Memory-mapped tensors (numpy `.npy` or PyTorch `.pt`) or Arrow/Parquet
  - [x] Design DataLoader-friendly format (batching, padding, attention masks)
- [x] Transfer prepared data to Sol's fast storage (scratch filesystem)
- [x] Verify data loading speed with a dummy training loop

### Phase 4: Transformer Architecture

- [x] Determine model hyperparameters (based on dataset size):
  - [x] Embedding dimension (128 tiny / 256 small / 768 full)
  - [x] Number of attention heads (4 tiny / 8 small / 12 full)
  - [x] Number of transformer layers (4 tiny / 8 small / 12 full)
  - [x] Context length: 256 moves max
  - [x] Dropout rate: 0.1
  - [x] Vocabulary size: 1972 (1968 UCI moves + 4 special tokens)
- [x] Choose positional encoding: **RoPE** (Rotary Position Embeddings)
- [x] Implement decoder-only transformer
  - [x] Token embedding + √d_model scaling
  - [x] Causal multi-head self-attention (PyTorch `scaled_dot_product_attention`)
  - [x] Feed-forward network with SiLU activation
  - [x] Pre-norm layer normalization
  - [x] Output projection to vocabulary logits
- [ ] Sanity checks:
  - [ ] Verify causal mask is correct (no future leakage)
  - [ ] Overfit on a tiny batch (loss → ~0)
  - [ ] Check parameter count matches expectations
  - [ ] Verify output shape: `(batch, seq_len, vocab_size)`

### Phase 5: Training on Sol

- [x] Write SLURM job scripts
  - [x] Resource requests: GPUs, memory, time limits
  - [x] Module loads and environment activation
  - [ ] Multi-node setup if needed
- [ ] Set up distributed training
  - [ ] DDP (DistributedDataParallel) for multi-GPU
  - [ ] FSDP (FullyShardedDataParallel) if model doesn't fit in single GPU memory
- [x] Configure mixed precision training (bf16 or fp16)
- [x] Implement training loop
  - [x] Cross-entropy loss on next-move prediction
  - [x] Learning rate schedule: warmup + cosine decay
  - [x] Gradient clipping
  - [x] Optimizer: AdamW with weight decay
- [x] Set up logging
  - [x] Stdout logging: loss, avg loss, learning rate, throughput (tok/s)
  - [ ] Weights & Biases or TensorBoard integration
- [x] Implement checkpointing strategy
  - [x] Save every N steps and at end of each epoch
  - [ ] Keep best-K checkpoints by validation loss
  - [x] Support resuming from checkpoint
- [x] Evaluate during training
  - [x] Validation loss every epoch
  - [ ] Top-1 and top-5 move prediction accuracy on validation set

### Phase 6: Evaluation & Analysis

- [ ] Evaluate move prediction accuracy by game phase
  - [ ] Opening (moves 1–10)
  - [ ] Middlegame (moves 11–25)
  - [ ] Endgame (moves 26+)
- [ ] Compare model predictions against known opening theory
  - [ ] Feed standard opening positions and check if model follows book lines
- [ ] Analyze failure modes
  - [ ] Positions where model confidence is low
  - [ ] Illegal move rate (if generating outside legal move set)
- [ ] Estimate approximate play strength (if feasible)
- [ ] Generate sample full games via autoregressive generation
  - [ ] Check for coherence and legal move sequences

### Phase 7: ONNX Export & Optimization

- [ ] Export trained PyTorch model to ONNX format
  - [ ] Define input/output signatures
  - [ ] Handle dynamic sequence lengths
  - [ ] Verify ONNX model outputs match PyTorch outputs (numerical parity)
- [ ] Optimize with ONNX Runtime
  - [ ] Graph optimizations (constant folding, operator fusion)
  - [ ] Quantization if needed (INT8 dynamic or static)
- [ ] Benchmark inference latency
  - [ ] Measure on target hardware (CPU and/or GPU)
  - [ ] Ensure latency is acceptable for Lichess bot (< 1–2 seconds per move)

### Phase 8: Deployment as Lichess Bot

- [ ] Set up AWS EC2 instance in eu-west-3 (France)
  - [ ] Choose instance type (CPU or GPU based on latency requirements)
  - [ ] Install ONNX Runtime and dependencies
- [ ] Create Lichess bot account
  - [ ] Register account and request bot flag via Lichess API
  - [ ] Generate API token with bot scopes
- [ ] Implement Lichess Bot API integration
  - [ ] Connect to game stream (SSE / WebSocket)
  - [ ] Parse incoming game state to UCI move history
  - [ ] Feed move history to ONNX model, get next move prediction
  - [ ] Submit moves via Lichess API
- [ ] Implement move selection strategy
  - [ ] Argmax (strongest single prediction)
  - [ ] Temperature sampling (for variety)
  - [ ] Optionally filter to legal moves only before selection
- [ ] Test bot in casual games
- [ ] Monitor bot performance and resource usage in production
