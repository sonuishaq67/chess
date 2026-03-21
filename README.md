# Chess Transformer

A decoder-only transformer trained on Lichess game data for next-move prediction, treating chess as an autoregressive language modeling problem over UCI move sequences.

## Stack

- **Model**: GPT-style decoder-only transformer (PyTorch)
- **Data**: HuggingFace Lichess parquet files, filtered by Elo (>1900) and time control (≥180s)
- **Tokenization**: UCI move vocabulary (~1800 tokens + specials)
- **Training**: ASU Sol supercomputer (SLURM, DDP/FSDP, mixed precision)
- **Export**: ONNX + ONNX Runtime
- **Deployment**: AWS EC2 eu-west-3, Lichess Bot API

## Layout

```
chess/
├── src/
│   ├── data/       # download, filter, tokenize
│   ├── model/      # transformer architecture
│   ├── training/   # training loop, SLURM scripts
│   └── serving/    # ONNX export, bot integration
├── configs/        # hyperparameter and data filter configs
├── dataset/        # raw and processed data
├── tasks/          # full project roadmap
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
```

See [`tasks/README.md`](tasks/README.md) for the full phased roadmap (Phases 0–9).
