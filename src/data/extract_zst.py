import zstandard
import os
import glob
import argparse
from multiprocessing import Pool, cpu_count
from pathlib import Path
import chess.pgn

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
ZST_DIR = os.path.join(BASE_DIR, "dataset", "zst")
PGN_DIR = os.path.join(BASE_DIR, "dataset", "pgn")
UCI_DIR = os.path.join(BASE_DIR, "dataset", "uci")
MIN_ELO = 1900
MIN_TIME_CONTROL = 180
MIN_MOVES = 20


def download_from_file():
    """Download .zst files from dataset/links using aria2c."""
    links_file = os.path.join(BASE_DIR, "dataset", "links")
    ret = os.system(f"aria2c -x 16 -j 16 -d {ZST_DIR} -i {links_file}")
    if ret != 0:
        raise RuntimeError(f"aria2c failed with exit code {ret}")


def extract_zst(filename):
    """Decompress a single .zst file and write the output to PGN_DIR."""
    out_name = Path(filename).stem  # e.g. "file.pgn" from "file.pgn.zst"
    out_path = os.path.join(PGN_DIR, out_name)

    print(f"Extracting {filename} -> {out_path}")
    dctx = zstandard.ZstdDecompressor()
    with open(filename, "rb") as ifh, open(out_path, "wb") as ofh:
        dctx.copy_stream(ifh, ofh)
    os.remove(filename)
    print(f"Done: {out_path} (deleted {filename})")


def extract_all_parallel(workers=None):
    """Decompress all .zst files in ZST_DIR in parallel."""
    os.makedirs(PGN_DIR, exist_ok=True)

    zst_files = sorted(glob.glob(os.path.join(ZST_DIR, "*.zst")))
    if not zst_files:
        print(f"No .zst files found in {ZST_DIR}")
        return

    num_workers = workers or min(cpu_count(), len(zst_files))
    print(f"Decompressing {len(zst_files)} files with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        pool.map(extract_zst, zst_files)

    print("All files decompressed.")


def passes_filters(game):
    """Return True if the game meets Elo and time control requirements."""
    headers = game.headers

    # Both players must be rated > 1900
    try:
        white_elo = int(headers.get("WhiteElo", "0"))
        black_elo = int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if white_elo <= MIN_ELO or black_elo <= MIN_ELO:
        return False

    # Base time control must be >= 180s  (format: "seconds+increment" or "seconds")
    tc = headers.get("TimeControl", "-")
    if tc == "-":
        return False
    try:
        base_time = int(tc.split("+")[0])
    except ValueError:
        return False
    if base_time < MIN_TIME_CONTROL:
        return False

    return True


def game_to_uci(game):
    """Convert a chess.pgn.Game to a UCI move string."""
    board = game.board()
    uci_moves = []
    for move in game.mainline_moves():
        uci_moves.append(move.uci())
        board.push(move)
    return " ".join(uci_moves)


def process_pgn_file(pgn_file):
    """Read all games from a PGN file, filter, and write UCI lines to UCI_DIR."""
    out_name = Path(pgn_file).stem + ".txt"
    out_path = os.path.join(UCI_DIR, out_name)

    print(f"Processing {pgn_file} -> {out_path}")
    total = 0
    kept = 0
    with open(pgn_file, "r") as f, open(out_path, "w") as out:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total += 1
            if not passes_filters(game):
                continue
            uci = game_to_uci(game)
            if uci and len(uci.split()) >= MIN_MOVES:
                out.write(uci + "\n")
                kept += 1

    os.remove(pgn_file)
    print(f"Done: {out_path} ({kept}/{total} games kept, deleted {pgn_file})")
    return total, kept


def process_all_pgn_parallel(workers=None):
    """Convert all PGN files in PGN_DIR to UCI format in parallel."""
    os.makedirs(UCI_DIR, exist_ok=True)

    pgn_files = sorted(glob.glob(os.path.join(PGN_DIR, "*.pgn")))
    if not pgn_files:
        print(f"No .pgn files found in {PGN_DIR}")
        return

    num_workers = workers or min(cpu_count(), len(pgn_files))
    print(f"Processing {len(pgn_files)} PGN files with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_pgn_file, pgn_files)

    grand_total = sum(t for t, _ in results)
    grand_kept = sum(k for _, k in results)
    print(f"All PGN files converted to UCI. {grand_kept}/{grand_total} games kept.")


def run_all():
    """Run the full pipeline: download -> extract -> process."""
    download_from_file()
    extract_all_parallel()
    process_all_pgn_parallel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess data pipeline")
    parser.add_argument(
        "command",
        choices=["all", "download", "extract", "process"],
        help="all: full pipeline | download: fetch .zst files | extract: decompress .zst to .pgn | process: convert .pgn to UCI",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    args = parser.parse_args()

    if args.command == "all":
        run_all()
    elif args.command == "download":
        download_from_file()
    elif args.command == "extract":
        extract_all_parallel(workers=args.workers)
    elif args.command == "process":
        process_all_pgn_parallel(workers=args.workers)
