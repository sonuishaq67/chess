"""Build chunk-packing arrays once so distributed training doesn't rebuild on every rank.

Reads dataset/bin/*.idx files, computes deterministic first-fit packing into chunks of
(max_seq_len + 1) tokens, writes:

    dataset/packing/packed.npy  - shape (M, 3) int64: (bin_idx, start, end)
    dataset/packing/bounds.npy  - shape (num_chunks+1,) int64: chunk boundary offsets
    dataset/packing/meta.json   - {chunk_len, seed, bin_paths: [str]}

ChessDataset.__init__ then mmaps these instead of rebuilding, keeping rank startup flat
in RAM (matters because DDP runs __init__ once per rank concurrently).
"""

import argparse
import json
import struct
import time
from pathlib import Path

import numpy as np


def build_packing(
    bin_dir: Path, out_dir: Path, max_seq_len: int, seed: int, max_files: int | None
) -> None:
    chunk_len = max_seq_len + 1

    idx_files = sorted(bin_dir.glob("*.idx"))
    if max_files is not None:
        idx_files = idx_files[:max_files]
    print(f"Scanning {len(idx_files):,} .idx files...")

    # Phase 1: load per-file metadata straight into numpy (no Python list-of-tuples)
    bin_parts: list[np.ndarray] = []
    start_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    len_parts: list[np.ndarray] = []
    bin_paths: list[str] = []
    dropped_zero_len = 0
    dropped_too_long = 0
    t0 = time.time()

    for idx_path in idx_files:
        bin_path = idx_path.with_suffix(".bin")
        if not bin_path.exists():
            continue
        bin_idx = len(bin_paths)
        bin_paths.append(str(bin_path))

        with open(idx_path, "rb") as f:
            struct.unpack("<Q", f.read(8))[0]  # n_games header (unused - use offsets shape)
            offsets = np.frombuffer(f.read(), dtype=np.uint64).astype(np.int64)

        starts = offsets[:-1]
        ends = offsets[1:]
        lens = ends - starts
        dropped_zero_len += int((lens <= 0).sum())
        dropped_too_long += int((lens > chunk_len).sum())
        mask = (lens > 0) & (lens <= chunk_len)

        n_kept = int(mask.sum())
        bin_parts.append(np.full(n_kept, bin_idx, dtype=np.int64))
        start_parts.append(starts[mask])
        end_parts.append(ends[mask])
        len_parts.append(lens[mask])

    if not len_parts or sum(part.size for part in len_parts) == 0:
        raise ValueError(f"No packable games found in {bin_dir}")

    bin_idxs = np.concatenate(bin_parts)
    starts = np.concatenate(start_parts)
    ends = np.concatenate(end_parts)
    lens = np.concatenate(len_parts)
    del bin_parts, start_parts, end_parts, len_parts

    N = len(lens)
    print(
        f"  {N:,} games (<= {chunk_len} tokens) across {len(bin_paths):,} .bin files "
        f"in {time.time() - t0:.1f}s"
    )
    if dropped_zero_len or dropped_too_long:
        print(
            f"  Dropped {dropped_zero_len:,} zero-length and "
            f"{dropped_too_long:,} overlong games before packing"
        )

    # Phase 2: first-fit packing, writing into preallocated numpy arrays
    rng = np.random.default_rng(seed)
    order = rng.permutation(N)
    lens_ord = lens[order]
    bin_ord = bin_idxs[order]
    start_ord = starts[order]
    end_ord = ends[order]
    del bin_idxs, starts, ends, lens, order

    # Upper bound: worst case every game is its own chunk
    packed = np.empty((N, 3), dtype=np.int64)
    bounds = np.empty(N + 1, dtype=np.int64)
    bounds[0] = 0
    n_packed = 0
    n_chunks = 0
    cur_len = 0
    t0 = time.time()

    for i in range(N):
        l = int(lens_ord[i])
        if cur_len + l > chunk_len and n_packed > bounds[n_chunks]:
            n_chunks += 1
            bounds[n_chunks] = n_packed
            cur_len = 0
        packed[n_packed, 0] = bin_ord[i]
        packed[n_packed, 1] = start_ord[i]
        packed[n_packed, 2] = end_ord[i]
        cur_len += l
        n_packed += 1

    if n_packed > bounds[n_chunks]:
        n_chunks += 1
        bounds[n_chunks] = n_packed

    print(f"  Packed into {n_chunks:,} chunks in {time.time() - t0:.1f}s")

    # Trim + save
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "packed.npy", packed[:n_packed])
    np.save(out_dir / "bounds.npy", bounds[: n_chunks + 1])
    (out_dir / "meta.json").write_text(
        json.dumps(
            {"chunk_len": chunk_len, "seed": seed, "bin_paths": bin_paths},
            indent=2,
        )
    )
    print(f"Wrote {out_dir}/{{packed,bounds}}.npy + meta.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", default="dataset/bin")
    parser.add_argument("--out-dir", default="dataset/packing")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-files", type=int, default=None)
    args = parser.parse_args()
    build_packing(
        Path(args.bin_dir),
        Path(args.out_dir),
        args.max_seq_len,
        args.seed,
        args.max_files,
    )


if __name__ == "__main__":
    main()
