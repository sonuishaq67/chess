import json
import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


PAD_ID = 0


class ChessDataset(Dataset):
    def __init__(self, packing_dir: str, max_seq_len: int = 256):
        """Load pre-built packing arrays produced by `src/data/build_packing.py`.

        Bin files are preloaded into RAM at init time. This avoids per-access
        page faults on scratch FS that were starving the HPUs. DataLoader workers
        share the loaded arrays via fork COW — numpy's data buffer is separate
        from the Python object header, so access from children doesn't break COW.
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
        # Small index metadata — load fully into RAM (tiny).
        self._packed = np.load(packing_dir / "packed.npy")
        self._bounds = np.load(packing_dir / "bounds.npy")
        self._num_chunks = len(self._bounds) - 1

        # Preload every bin file into RAM. One copy per rank; workers inherit
        # via fork COW so there is no per-worker duplication of the data buffer.
        is_rank0 = os.environ.get("RANK", "0") == "0"
        if is_rank0:
            print(f"Preloading {len(self.bin_paths)} bin files into RAM...")
        t0 = time.time()
        self._bins: list[np.ndarray] = [
            np.fromfile(p, dtype=np.uint16) for p in self.bin_paths
        ]
        if is_rank0:
            total_gb = sum(b.nbytes for b in self._bins) / 1e9
            print(
                f"Loaded {total_gb:.2f} GB of token data in "
                f"{time.time() - t0:.1f}s"
            )

    def __len__(self) -> int:
        return self._num_chunks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        lo, hi = int(self._bounds[idx]), int(self._bounds[idx + 1])

        parts = []
        total = 0
        for j in range(lo, hi):
            bin_idx, start, end = self._packed[j]
            arr = self._bins[int(bin_idx)]
            parts.append(arr[int(start):int(end)])
            total += int(end) - int(start)

        tokens = np.concatenate(parts) if len(parts) > 1 else parts[0].copy()

        if total < self.chunk_len:
            padding = np.zeros(self.chunk_len - total, dtype=np.uint16)
            tokens = np.concatenate([tokens, padding])

        tokens = torch.from_numpy(tokens.astype(np.int64))
        return tokens[:-1], tokens[1:]
