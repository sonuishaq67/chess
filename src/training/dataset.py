import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


PAD_ID = 0


class ChessDataset(Dataset):
    def __init__(self, bin_dir: str, max_seq_len: int = 256, max_files: int | None = None):
        self.max_seq_len = max_seq_len
        self.chunk_len = max_seq_len + 1

        bin_dir = Path(bin_dir)
        idx_files = sorted(bin_dir.glob("*.idx"))
        if max_files is not None:
            idx_files = idx_files[:max_files]

        # Phase 1: scan .idx files to collect game metadata (no token data loaded)
        games = []  # list of (bin_path_index, start, end)
        self.bin_paths = []  # deduplicated bin paths

        for idx_path in idx_files:
            bin_path = idx_path.with_suffix(".bin")
            if not bin_path.exists():
                continue

            bin_idx = len(self.bin_paths)
            self.bin_paths.append(str(bin_path))

            with open(idx_path, "rb") as f:
                n_games = struct.unpack("<Q", f.read(8))[0]
                offsets = np.frombuffer(f.read(), dtype=np.uint64)

            for i in range(n_games):
                start, end = int(offsets[i]), int(offsets[i + 1])
                game_len = end - start
                if game_len <= self.chunk_len:
                    games.append((bin_idx, start, end, game_len))

        print(f"  Scanned {len(idx_files)} files, {len(games):,} games")

        # Phase 2: compute packing assignments from lengths only
        rng = np.random.default_rng(42)
        order = rng.permutation(len(games))

        self.chunks = []  # list of [(bin_idx, start, end), ...]
        cur_chunk = []
        cur_len = 0

        for i in order:
            bin_idx, start, end, game_len = games[i]
            if cur_len + game_len <= self.chunk_len:
                cur_chunk.append((bin_idx, start, end))
                cur_len += game_len
            else:
                if cur_chunk:
                    self.chunks.append(cur_chunk)
                cur_chunk = [(bin_idx, start, end)]
                cur_len = game_len

        if cur_chunk:
            self.chunks.append(cur_chunk)

        # Open memmaps once (shared across all __getitem__ calls)
        self._memmaps = {}

    def _get_memmap(self, bin_idx: int) -> np.ndarray:
        if bin_idx not in self._memmaps:
            self._memmaps[bin_idx] = np.memmap(
                self.bin_paths[bin_idx], dtype=np.uint16, mode="r"
            )
        return self._memmaps[bin_idx]

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_games = self.chunks[idx]

        # Read token slices from memmaps and concatenate
        parts = []
        total = 0
        for bin_idx, start, end in chunk_games:
            mmap = self._get_memmap(bin_idx)
            parts.append(mmap[start:end])
            total += end - start

        tokens = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        # Pad if needed
        if total < self.chunk_len:
            padding = np.zeros(self.chunk_len - total, dtype=np.uint16)
            tokens = np.concatenate([tokens, padding])

        tokens = torch.from_numpy(tokens.astype(np.int64))
        return tokens[:-1], tokens[1:]
