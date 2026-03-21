# Chess Transformer — Task Roadmap

A decoder-only transformer trained on Lichess game data for next-move prediction, treating chess as an autoregressive language modeling problem over UCI move sequences.

## Codebase Structure

```
chess/
├── README.md               # project overview
├── tasks/
│   └── README.md           # this file — full roadmap
├── requirements.txt        # Python dependencies
├── cli.sh                  # helper CLI commands
├── configs/                # hyperparameter and data filter configs
├── dataset/                # raw and processed data
│   └── readme.md
└── src/
    ├── data/
    │   ├── uci_tokenizer.py    # build vocabulary and tokenize games
    │   └── (download/filter scripts)
    ├── model/              # transformer architecture
    ├── training/           # training loop, SLURM scripts
    └── serving/            # ONNX export, bot integration
```

## Dependencies

- `python-chess` — PGN parsing and move conversion to UCI
- `zstandard` — streaming decompression of .zst files
- `torch` — model development and training
- `tokenizers` (HuggingFace) — vocabulary building
- `onnx` / `onnxruntime` — model export and inference
- `huggingface_hub` — dataset download

---

## Phase 0: Environment & Setup

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

## Phase 1: Data Download & Exploration

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

## Phase 2: Data Filtering & Conversion

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

## Phase 3: Tokenization & Tensor Preparation

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

## Phase 4: Transformer Architecture

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

## Phase 5: Training on Sol

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

## Phase 6: Evaluation & Analysis

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

## Phase 7: ONNX Export & Optimization

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

## Phase 8: Deployment as Lichess Bot

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

## Phase 9: Hyperparameter Autotuning (Karpathy AutoResearch)

- [ ] Define hyperparameter search space
  - [ ] Learning rate, batch size, model depth (layers) and width (embedding dim)
  - [ ] Dropout rate, weight decay, warmup steps
  - [ ] Positional encoding type: RoPE vs sinusoidal vs learned
  - [ ] Optimizer betas and epsilon
- [ ] Implement LLM proposal agent (Claude API)
  - [ ] Feed full experiment history (configs + results) as context to Claude
  - [ ] Prompt agent to propose next hyperparameter config with reasoning
  - [ ] Parse structured config from LLM response
  - [ ] Log proposals and agent reasoning to research log
- [ ] Build automated SLURM job launcher
  - [ ] Template SLURM job script with configurable hyperparameter injection
  - [ ] Submit jobs programmatically via `subprocess` / `sbatch`
  - [ ] Poll for job completion and collect validation loss and move accuracy
- [ ] Maintain reflection / research log
  - [ ] Record each trial: config, validation loss, perplexity, wall-clock time
  - [ ] Append LLM reasoning and proposed next experiment after each trial
  - [ ] Version-control the log for reproducibility
- [ ] Run autotuning loop
  - [ ] Start with baseline config from Phase 5
  - [ ] Iterate: propose → launch → evaluate → reflect
  - [ ] Stop after N trials or when validation loss converges
- [ ] Generate final ablation report
  - [ ] Compare all trials by validation loss and top-1 move accuracy
  - [ ] Identify best-performing hyperparameter config
  - [ ] Summarize insights: what the LLM agent learned across iterations
  - [ ] Document any surprising findings or emergent search strategies
