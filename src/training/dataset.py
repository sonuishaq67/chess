import struct
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


PAD_ID = 0


class ChessDataset(Dataset):
    def __init__(self, bin_dir: str, max_seq_len: int = 256, max_files: int | None = None):
        self.max_seq_len = max_seq_len
        self.chunk_len = max_seq_len + 1  # +1 so input[:-1] and label[1:] are both max_seq_len

        games = []
        bin_dir = Path(bin_dir)
        idx_files = sorted(bin_dir.glob("*.idx"))
        if max_files is not None:
            idx_files = idx_files[:max_files]

        for idx_path in idx_files:
            bin_path = idx_path.with_suffix(".bin")
            if not bin_path.exists():
                continue

            with open(idx_path, "rb") as f:
                n_games = struct.unpack("<Q", f.read(8))[0]
                offsets = np.frombuffer(f.read(), dtype=np.uint64)

            total_tokens = int(offsets[-1])
            tokens = np.memmap(
                bin_path, dtype=np.uint16, mode="r", shape=(total_tokens,)
            )

            for i in range(n_games):
                start, end = int(offsets[i]), int(offsets[i + 1])
                game_len = end - start
                if game_len <= self.chunk_len:
                    games.append(tokens[start:end].copy())

        self.chunks = []
        self._pack_games(games)

    def _pack_games(self, games: list[np.ndarray]):
        rng = np.random.default_rng(42)
        rng.shuffle(games)

        chunk = []
        chunk_len = 0

        for game in games:
            game_len = len(game)
            if chunk_len + game_len <= self.chunk_len:
                chunk.append(game)
                chunk_len += game_len
            else:
                if chunk:
                    self.chunks.append(self._pad_chunk(chunk, chunk_len))
                chunk = [game]
                chunk_len = game_len

        if chunk:
            self.chunks.append(self._pad_chunk(chunk, chunk_len))

    def _pad_chunk(self, games: list[np.ndarray], used: int) -> np.ndarray:
        arr = np.concatenate(games)
        if used < self.chunk_len:
            padding = np.full(self.chunk_len - used, PAD_ID, dtype=np.uint16)
            arr = np.concatenate([arr, padding])
        return arr

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.chunks[idx]
        tokens = torch.from_numpy(tokens.astype(np.int64))
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return input_ids, labels
