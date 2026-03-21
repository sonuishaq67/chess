# Chess Transformer

A decoder-only transformer trained on Lichess game data for next-move prediction, treating chess as an autoregressive language modeling problem over UCI move sequences.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                            │
│  HuggingFace Parquet ──► Filter ──► UCI Sequences               │
│  (Lichess/standard-chess-games)  (Elo,TC)   (python-chess)      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                     TOKENIZATION & PREP                         │
│  UCI Moves ──► Vocabulary (~1800 + special) ──► Integer Tensors │
│  e2e4 g1f3...    PAD BOS EOS MASK               memory-mapped   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING (ASU Sol)                            │
│  Decoder-Only Transformer (GPT-style)                           │
│  Distributed training (DDP/FSDP) · Mixed precision · SLURM      │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                   EXPORT & DEPLOYMENT                           │
│  PyTorch ──► ONNX ──► ONNX Runtime ──► AWS EC2 (eu-west-3)     │
│                        optimized          Lichess Bot API        │
└─────────────────────────────────────────────────────────────────┘
```

## Data Decisions

| Parameter         | Value                                                                 |
|-------------------|-----------------------------------------------------------------------|
| Source            | [HuggingFace](https://huggingface.co/datasets/Lichess/standard-chess-games) parquet |
| Elo filter        | Both players > 1900                                                  |
| Time control      | Base time >= 180s (rapid/classical)                                  |
| Move format       | UCI (e.g. `e2e4`, `g1f3`)                                           |
| Vocab size        | ~1800 unique move tokens + special tokens                            |
| Termination       | Checkmate, resignation, draws (agreement/stalemate/repetition). Exclude abandoned/timeout. Exact thresholds TBD after data exploration. |

## Infrastructure

- **Training**: ASU Sol supercomputer (SLURM, multi-GPU)
- **Export**: ONNX format
- **Deployment**: AWS EC2 in eu-west-3 (France), Lichess Bot API

---

## Codebase Structure

```
chess/
├── dataset/              # raw and processed data
│   ├── pgn/              # PGN game files
│   ├── uci/              # UCI move sequences
│   ├── parquet/          # HuggingFace parquet files
│   └── tensors/          # tokenized memory-mapped tensors
├── src/
│   ├── data/             # download, filter, tokenize scripts
│   │   ├── uci_tokenizer.py   # UCI move vocabulary builder
│   │   └── extract_zst.py     # .zst decompression utility
│   ├── model/            # transformer architecture (planned)
│   ├── training/         # training loop, SLURM scripts (planned)
│   └── serving/          # ONNX export, bot integration (planned)
├── configs/              # hyperparams, data filter configs
├── tasks/                # detailed task roadmap (this file)
├── requirements.txt      # pip dependencies
├── cli.sh                # CLI helper script
└── README.md             # project overview
```

## Dependencies

- `python-chess` — PGN parsing and move conversion to UCI
- `huggingface_hub` — dataset download
- `pyarrow` / `pandas` — parquet file reading
- `tokenizers` — vocabulary building
- `onnx` / `onnxruntime` — model export and inference
- `torch` — model training (installed via conda for CUDA support)

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
  ├── dataset/          # raw and processed data
  ├── src/
  │   ├── data/         # download, filter, tokenize scripts
  │   ├── model/        # transformer architecture
  │   ├── training/     # training loop, SLURM scripts
  │   └── serving/      # ONNX export, bot integration
  ├── configs/          # hyperparams, data filters
  ├── notebooks/        # exploration and analysis
  ├── checkpoints/      # saved model weights
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
- [ ] Estimate total game count across all files
- [ ] Estimate filtered game count (both players > 1900, base time >= 180s)
- [ ] Analyze termination reasons and their distribution
- [ ] Decide final filter thresholds based on exploration findings
  - [ ] Minimum move count for resignations
  - [ ] Any additional filters worth applying

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
- [ ] Compute dataset statistics:
  - [ ] Total filtered games
  - [ ] Average / median game length (in moves)
  - [ ] Move frequency distribution
  - [ ] Elo distribution of filtered games
  - [ ] Result distribution (white win / black win / draw)

### Phase 3: Tokenization & Tensor Preparation

- [ ] Build UCI move vocabulary
  - [ ] Enumerate all unique UCI moves seen in filtered data
  - [ ] Add special tokens: `PAD`, `BOS`, `EOS`, `MASK`
  - [ ] Save vocabulary mapping (token → id, id → token)
  - [ ] Verify vocab size (~1800 + specials)
- [ ] Tokenize all games: UCI move strings → integer sequences
  - [ ] Prepend `BOS`, append `EOS` to each game
- [ ] Create train / validation / test split
  - [ ] Split by date (preferred) or random
  - [ ] Typical split: 95% / 2.5% / 2.5% or similar
- [ ] Pack tokenized sequences into efficient storage
  - [ ] Memory-mapped tensors (numpy `.npy` or PyTorch `.pt`) or Arrow/Parquet
  - [ ] Design DataLoader-friendly format (batching, padding, attention masks)
- [ ] Transfer prepared data to Sol's fast storage (scratch filesystem)
- [ ] Verify data loading speed with a dummy training loop

### Phase 4: Transformer Architecture

- [ ] Determine model hyperparameters (based on dataset size):
  - [ ] Embedding dimension
  - [ ] Number of attention heads
  - [ ] Number of transformer layers
  - [ ] Context length (max game length in moves, from Phase 2 stats)
  - [ ] Dropout rate
  - [ ] Vocabulary size (from Phase 3)
- [ ] Choose positional encoding: learned vs sinusoidal vs RoPE
- [ ] Implement decoder-only transformer
  - [ ] Token embedding + positional encoding
  - [ ] Causal (masked) multi-head self-attention
  - [ ] Feed-forward network (with GELU or SiLU)
  - [ ] Layer normalization (pre-norm preferred)
  - [ ] Output projection to vocabulary logits
- [ ] Sanity checks:
  - [ ] Verify causal mask is correct (no future leakage)
  - [ ] Overfit on a tiny batch (loss → ~0)
  - [ ] Check parameter count matches expectations
  - [ ] Verify output shape: `(batch, seq_len, vocab_size)`

### Phase 5: Training on Sol

- [ ] Write SLURM job scripts
  - [ ] Resource requests: GPUs, memory, time limits
  - [ ] Module loads and environment activation
  - [ ] Multi-node setup if needed
- [ ] Set up distributed training
  - [ ] DDP (DistributedDataParallel) for multi-GPU
  - [ ] FSDP (FullyShardedDataParallel) if model doesn't fit in single GPU memory
- [ ] Configure mixed precision training (bf16 or fp16)
- [ ] Implement training loop
  - [ ] Cross-entropy loss on next-move prediction
  - [ ] Learning rate schedule: warmup + cosine decay
  - [ ] Gradient clipping
  - [ ] Optimizer: AdamW with weight decay
- [ ] Set up logging
  - [ ] Weights & Biases or TensorBoard
  - [ ] Track: loss, perplexity, learning rate, gradient norms, throughput
- [ ] Implement checkpointing strategy
  - [ ] Save every N steps and at end of each epoch
  - [ ] Keep best-K checkpoints by validation loss
  - [ ] Support resuming from checkpoint
- [ ] Evaluate during training
  - [ ] Validation loss and perplexity every N steps
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

### Phase 9: Hyperparameter Autotuning (Karpathy AutoResearch)

Inspired by Andrej Karpathy's AutoResearch methodology — using LLM-driven agents to systematically propose, run, and evaluate hyperparameter experiments, closing the loop between training signal and configuration choices.

**Goal**: Automatically discover optimal transformer hyperparameters for chess move prediction without manual grid search.

#### Approach

The autotuning loop follows a research-agent pattern:
1. **Propose** — an LLM agent reads current training logs/metrics and proposes a new hyperparameter configuration (or experiment hypothesis)
2. **Run** — the proposed config is launched as a SLURM job on Sol
3. **Evaluate** — validation loss, move accuracy, and throughput are collected
4. **Reflect** — the agent updates a running experiment log and decides the next step

#### Tasks

- [ ] Set up experiment tracking database
  - [ ] Log all hyperparameter configs and their validation metrics (loss, top-1 accuracy, perplexity)
  - [ ] Use a simple JSON/CSV or SQLite store per experiment run
  - [ ] Each run: `{run_id, config, val_loss, top1_acc, perplexity, tokens/sec, notes}`
- [ ] Define the hyperparameter search space
  - [ ] `n_layers`: [4, 6, 8, 12]
  - [ ] `d_model`: [256, 512, 768]
  - [ ] `n_heads`: [4, 8, 12]
  - [ ] `d_ff` (feedforward dim): [1024, 2048, 3072]
  - [ ] `dropout`: [0.0, 0.1, 0.2]
  - [ ] `learning_rate`: [1e-4, 3e-4, 1e-3]
  - [ ] `batch_size`: [64, 128, 256]
  - [ ] `context_length`: [128, 256, 512]
  - [ ] `positional_encoding`: [learned, sinusoidal, RoPE]
- [ ] Implement LLM-driven proposal agent
  - [ ] Prompt includes: current best config, experiment history, training curves, hypotheses
  - [ ] Agent outputs: next config to try + rationale
  - [ ] Use Claude API (or local model) to generate proposals
  - [ ] Parse structured JSON output from agent
- [ ] Build SLURM job launcher
  - [ ] Accept config dict → write config YAML → submit `sbatch` job
  - [ ] Poll job status and collect metrics on completion
  - [ ] Handle failures/timeouts gracefully
- [ ] Implement reflection step
  - [ ] After each run, feed results back to agent with updated history
  - [ ] Agent identifies trends (e.g., "larger d_model consistently helps, dropout hurts")
  - [ ] Maintain a `research_log.md` with findings and decisions
- [ ] Run initial baseline experiments
  - [ ] Small model (6L, 256d) as sanity check
  - [ ] Medium model (8L, 512d) as primary target
  - [ ] Log results and feed into autotuning loop
- [ ] Autotuning loop: run N=20 agent-proposed experiments
  - [ ] Compare against manual grid search baseline
  - [ ] Report best config and performance
- [ ] Produce final hyperparameter report
  - [ ] Best config found
  - [ ] Ablation insights from agent's reasoning
  - [ ] Comparison: autotuned vs manually chosen hyperparameters
