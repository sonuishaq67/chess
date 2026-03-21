# Chess Transformer

A decoder-only transformer trained on Lichess game data for chess move prediction.

## What it does

Trains a GPT-style transformer to predict the next move in a chess game, treating chess as an autoregressive sequence modeling problem over UCI move notation (e.g. `e2e4`, `g1f3`).

## Stack

- **Data**: Lichess games via HuggingFace parquet (Elo > 1900, rapid/classical)
- **Model**: Decoder-only transformer (PyTorch)
- **Training**: ASU Sol supercomputer (SLURM, multi-GPU, DDP/FSDP)
- **Export**: ONNX → ONNX Runtime
- **Deployment**: AWS EC2 (eu-west-3) + Lichess Bot API

## Project Layout

```
chess/
├── src/
│   ├── data/       # filtering, tokenization scripts
│   ├── model/      # transformer architecture
│   ├── training/   # training loop + SLURM scripts
│   └── serving/    # ONNX export, Lichess bot
├── dataset/        # raw and processed data (gitignored)
├── configs/        # hyperparameter configs
├── tasks/          # detailed task roadmap
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
# PyTorch: install separately via conda for CUDA support
```

See [`tasks/README.md`](tasks/README.md) for the full task roadmap, including hyperparameter autotuning with the Karpathy AutoResearch approach.
