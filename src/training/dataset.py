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
        # Store as flat numpy arrays so forked workers share memory (no copy)
        games_arr = np.array(games, dtype=np.int64)  # [N, 4] = bin_idx, start, end, game_len
        rng = np.random.default_rng(42)
        order = rng.permutation(len(games_arr))

        # Build flat game list + chunk boundary offsets
        packed_games = []  # flat list of (bin_idx, start, end)
        chunk_bounds = [0]  # chunk i spans packed_games[chunk_bounds[i]:chunk_bounds[i+1]]
        cur_len = 0

        for i in order:
            bin_idx, start, end, game_len = games_arr[i]
            if cur_len + game_len <= self.chunk_len:
                packed_games.append((bin_idx, start, end))
                cur_len += game_len
            else:
                if packed_games and cur_len > 0:
                    chunk_bounds.append(len(packed_games))
                packed_games.append((bin_idx, start, end))
                cur_len = int(game_len)

        if len(packed_games) > chunk_bounds[-1]:
            chunk_bounds.append(len(packed_games))

        # Store as numpy arrays — shared across forked workers, not copied
        self._packed = np.array(packed_games, dtype=np.int64)  # [M, 3]
        self._bounds = np.array(chunk_bounds, dtype=np.int64)  # [num_chunks + 1]
        self._num_chunks = len(chunk_bounds) - 1

        # Open memmaps once (shared across all __getitem__ calls)
        self._memmaps = {}

    def _get_memmap(self, bin_idx: int) -> np.ndarray:
        if bin_idx not in self._memmaps:
            self._memmaps[bin_idx] = np.memmap(
                self.bin_paths[bin_idx], dtype=np.uint16, mode="r"
            )
        return self._memmaps[bin_idx]

    def __len__(self) -> int:
        return self._num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = self._bounds[idx], self._bounds[idx + 1]

        # Read token slices from memmaps and concatenate
        parts = []
        total = 0
        for j in range(lo, hi):
            bin_idx, start, end = self._packed[j]
            mmap = self._get_memmap(int(bin_idx))
            parts.append(mmap[int(start):int(end)])
            total += int(end) - int(start)

        tokens = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        # Pad if needed
        if total < self.chunk_len:
            padding = np.zeros(self.chunk_len - total, dtype=np.uint16)
            tokens = np.concatenate([tokens, padding])

        tokens = torch.from_numpy(tokens.astype(np.int64))
        return tokens[:-1], tokens[1:]
