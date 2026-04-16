import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


PAD_ID = 0


class ChessDataset(Dataset):
    def __init__(self, packing_dir: str, max_seq_len: int = 256):
        """Load pre-built packing arrays produced by `src/data/build_packing.py`.

        Each rank just mmaps the two .npy files — no Python lists, no per-rank rebuild.
        To change data subset or seed, rerun build_packing.py.
        """
        self.chunk_len = max_seq_len + 1

        packing_dir = Path(packing_dir)
        meta = json.loads((packing_dir / "meta.json").read_text())

        if meta["chunk_len"] != self.chunk_len:
            raise ValueError(
                f"packing.chunk_len={meta['chunk_len']} but training expects "
                f"chunk_len={self.chunk_len} (max_seq_len={max_seq_len}). Rebuild with: "
                f"python -m src.data.build_packing --max-seq-len {max_seq_len}"
            )

        self.bin_paths = meta["bin_paths"]
        # mmap_mode='r' — zero RAM load, shared across forked DataLoader workers
        self._packed = np.load(packing_dir / "packed.npy", mmap_mode="r")
        self._bounds = np.load(packing_dir / "bounds.npy", mmap_mode="r")
        self._num_chunks = len(self._bounds) - 1

        # Opened lazily per worker; numpy memmap file handles don't survive fork
        self._memmaps: dict[int, np.ndarray] = {}

    def _get_memmap(self, bin_idx: int) -> np.ndarray:
        if bin_idx not in self._memmaps:
            self._memmaps[bin_idx] = np.memmap(
                self.bin_paths[bin_idx], dtype=np.uint16, mode="r"
            )
        return self._memmaps[bin_idx]

    def __len__(self) -> int:
        return self._num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = int(self._bounds[idx]), int(self._bounds[idx + 1])

        parts = []
        total = 0
        for j in range(lo, hi):
            bin_idx, start, end = self._packed[j]
            mmap = self._get_memmap(int(bin_idx))
            parts.append(mmap[int(start):int(end)])
            total += int(end) - int(start)

        tokens = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        if total < self.chunk_len:
            padding = np.zeros(self.chunk_len - total, dtype=np.uint16)
            tokens = np.concatenate([tokens, padding])

        tokens = torch.from_numpy(tokens.astype(np.int64))
        return tokens[:-1], tokens[1:]
