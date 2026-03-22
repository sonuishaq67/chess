import json
import mmap
import os
import struct
import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from pathlib import Path

FILES = "abcdefgh"
RANKS = "12345678"
SQUARES = [f + r for f in FILES for r in RANKS]  # a1..h8

SQ_TO_IDX = {sq: i for i, sq in enumerate(SQUARES)}


def file_idx(sq: str) -> int:
    return ord(sq[0]) - ord("a")


def rank_idx(sq: str) -> int:
    return int(sq[1]) - 1


def generate_all_uci_moves() -> list[str]:
    """
    Enumerate every UCI move string that can appear in a legal chess game.

    Piece reachability:
      King   : 1 step in 8 directions
      Knight : L-shape (8 targets)
      Rook   : up to 7 steps along rank or file (4 directions)
      Bishop : up to 7 steps diagonally (4 directions)
      Queen  : rook + bishop combined
      Pawn   : forward 1, forward 2 (from start rank), diagonal capture ±1,
               all with promotion suffixes (q/r/b/n) when reaching rank 1 or 8

    Returns a sorted list of unique UCI strings.
    """
    moves = set()

    for sq in SQUARES:
        f, r = file_idx(sq), rank_idx(sq)

        # ── Sliding pieces: Queen covers Rook + Bishop directions ──
        # Rook directions (straight)
        for df, dr in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for dist in range(1, 8):
                nf, nr = f + df * dist, r + dr * dist
                if 0 <= nf < 8 and 0 <= nr < 8:
                    moves.add(sq + FILES[nf] + RANKS[nr])

        # Bishop directions (diagonal)
        for df, dr in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for dist in range(1, 8):
                nf, nr = f + df * dist, r + dr * dist
                if 0 <= nf < 8 and 0 <= nr < 8:
                    moves.add(sq + FILES[nf] + RANKS[nr])

        # ── Knight ──
        for df, dr in [
            (1, 2),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ]:
            nf, nr = f + df, r + dr
            if 0 <= nf < 8 and 0 <= nr < 8:
                moves.add(sq + FILES[nf] + RANKS[nr])

        # ── King (1-step in all 8 directions) ──
        # Already covered by the sliding loops at dist=1

        # ── Pawn (white) ──
        if r == 1:  # rank 2 — white pawn start
            moves.add(sq + FILES[f] + RANKS[r + 1])  # forward 1
            moves.add(sq + FILES[f] + RANKS[r + 2])  # forward 2
            for df in [-1, 1]:  # diagonal captures
                if 0 <= f + df < 8:
                    moves.add(sq + FILES[f + df] + RANKS[r + 1])

        elif 2 <= r <= 5:  # ranks 3-6 — white pawn mid-board
            moves.add(sq + FILES[f] + RANKS[r + 1])
            for df in [-1, 1]:
                if 0 <= f + df < 8:
                    moves.add(sq + FILES[f + df] + RANKS[r + 1])

        elif r == 6:  # rank 7 — white pawn about to promote
            for promo in ["q", "r", "b", "n"]:
                moves.add(sq + FILES[f] + "8" + promo)  # push to 8th
                for df in [-1, 1]:  # capture-promote
                    if 0 <= f + df < 8:
                        moves.add(sq + FILES[f + df] + "8" + promo)

        # ── Pawn (black) ──
        if r == 6:  # rank 7 — black pawn start
            moves.add(sq + FILES[f] + RANKS[r - 1])
            moves.add(sq + FILES[f] + RANKS[r - 2])
            for df in [-1, 1]:
                if 0 <= f + df < 8:
                    moves.add(sq + FILES[f + df] + RANKS[r - 1])

        elif 2 <= r <= 5:  # ranks 3-6 — black pawn mid-board
            moves.add(sq + FILES[f] + RANKS[r - 1])
            for df in [-1, 1]:
                if 0 <= f + df < 8:
                    moves.add(sq + FILES[f + df] + RANKS[r - 1])

        elif r == 1:  # rank 2 — black pawn about to promote
            for promo in ["q", "r", "b", "n"]:
                moves.add(sq + FILES[f] + "1" + promo)
                for df in [-1, 1]:
                    if 0 <= f + df < 8:
                        moves.add(sq + FILES[f + df] + "1" + promo)

    return sorted(moves)


def build_vocab(moves: list[str]) -> dict:
    """
    Build the full token↔id mapping with special tokens.

    Layout:
      0: PAD
      1: BOS
      2: EOS
      3: MASK
      4+: UCI moves (sorted alphabetically)
    """
    special = {"PAD": 0, "BOS": 1, "EOS": 2, "MASK": 3}
    token2id = dict(special)
    for i, move in enumerate(moves):
        token2id[move] = len(special) + i
    return token2id


BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
UCI_DIR = os.path.join(BASE_DIR, "dataset", "uci")
BIN_DIR = os.path.join(BASE_DIR, "dataset", "bin")
VOCAB_PATH = os.path.join(BASE_DIR, "dataset", "vocab.json")

# BOS + ~200 moves + EOS, padded — uint16 is plenty for vocab < 65k
DTYPE = np.uint16


def tokenize_game(line: str, token2id: dict) -> np.ndarray | None:
    """Tokenize a single UCI game line into a numpy array: [BOS, move_ids..., EOS]."""
    moves = line.strip().split()
    if not moves:
        return None
    ids = [token2id["BOS"]]
    for m in moves:
        tid = token2id.get(m)
        if tid is None:
            return None  # skip games with unknown moves
        ids.append(tid)
    ids.append(token2id["EOS"])
    return np.array(ids, dtype=DTYPE)


def tokenize_file(args: tuple) -> tuple[str, int, int]:
    """
    Convert one UCI text file into a memmap-backed .bin + .idx pair.

    .bin  — flat uint16 array of all tokenized games concatenated
    .idx  — binary index: N (uint64), then N+1 uint64 offsets into .bin
            (offsets[i]:offsets[i+1] gives game i's token slice)

    Returns (filename, num_games_written, num_lines_skipped).
    """
    uci_path, vocab_path = args
    token2id = json.loads(Path(vocab_path).read_text())

    stem = Path(uci_path).stem
    bin_path = os.path.join(BIN_DIR, stem + ".bin")
    idx_path = os.path.join(BIN_DIR, stem + ".idx")

    # First pass: mmap the input file and split on newlines
    # Avoids Python's per-line buffered I/O overhead
    game_arrays = []
    skipped = 0
    with open(uci_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            # Kernel hint: we'll read sequentially
            mm.madvise(mmap.MADV_SEQUENTIAL)
            data = mm[:]  # single read into bytes
    for line in data.split(b"\n"):
        if not line:
            continue
        arr = tokenize_game(line.decode("ascii"), token2id)
        if arr is not None:
            game_arrays.append(arr)
        else:
            skipped += 1

    if not game_arrays:
        return stem, 0, skipped

    # Build offset index
    offsets = np.zeros(len(game_arrays) + 1, dtype=np.uint64)
    for i, arr in enumerate(game_arrays):
        offsets[i + 1] = offsets[i] + len(arr)

    total_tokens = int(offsets[-1])

    # Write .bin as memmap
    mm = np.memmap(bin_path, dtype=DTYPE, mode="w+", shape=(total_tokens,))
    pos = 0
    for arr in game_arrays:
        mm[pos : pos + len(arr)] = arr
        pos += len(arr)
    mm.flush()
    del mm

    # Write .idx: N (uint64) then N+1 offsets (uint64)
    with open(idx_path, "wb") as f:
        f.write(struct.pack("<Q", len(game_arrays)))
        f.write(offsets.tobytes())

    return stem, len(game_arrays), skipped


def load_bin_memmap(stem: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a tokenized .bin/.idx pair as read-only memmaps.

    Returns (tokens_mmap, offsets) where:
      tokens_mmap[offsets[i]:offsets[i+1]] gives game i's token ids.
    """
    bin_path = os.path.join(BIN_DIR, stem + ".bin")
    idx_path = os.path.join(BIN_DIR, stem + ".idx")

    with open(idx_path, "rb") as f:
        n_games = struct.unpack("<Q", f.read(8))[0]
        offsets = np.frombuffer(f.read(), dtype=np.uint64)

    total_tokens = int(offsets[-1])
    tokens = np.memmap(bin_path, dtype=DTYPE, mode="r", shape=(total_tokens,))
    return tokens, offsets


def main():
    moves = generate_all_uci_moves()
    vocab = build_vocab(moves)

    # Save vocab
    os.makedirs(BIN_DIR, exist_ok=True)
    Path(VOCAB_PATH).write_text(json.dumps(vocab, indent=2))
    print(f"Vocab size: {len(vocab)} tokens -> {VOCAB_PATH}")

    # Gather UCI files
    uci_files = sorted(Path(UCI_DIR).glob("*.txt"))
    if not uci_files:
        print(f"No UCI files found in {UCI_DIR}")
        return

    print(f"Found {len(uci_files)} UCI files to tokenize")

    # Tokenize in parallel
    workers = min(cpu_count(), len(uci_files))
    tasks = [(str(f), VOCAB_PATH) for f in uci_files]

    total_games = 0
    total_skipped = 0
    files_done = 0
    n_files = len(uci_files)
    bar_width = 40

    with Pool(processes=workers) as pool:
        for stem, n_games, n_skipped in pool.imap_unordered(tokenize_file, tasks):
            total_games += n_games
            total_skipped += n_skipped
            files_done += 1
            pct = files_done / n_files
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(
                f"\r  [{bar}] {files_done}/{n_files} files | "
                f"{total_games:,} games tokenized, {total_skipped:,} skipped"
            )
            sys.stdout.flush()

    print(f"\n\nDone. {total_games:,} games tokenized across {n_files} files "
          f"({total_skipped:,} skipped). Output: {BIN_DIR}/")


if __name__ == "__main__":
    main()
