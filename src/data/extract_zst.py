import os
import glob
import argparse
import io
from multiprocessing import Pool, cpu_count
from pathlib import Path
import chess.pgn
import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
PARQUET_DIR = os.path.join(BASE_DIR, "dataset", "parquet")
UCI_DIR = os.path.join(BASE_DIR, "dataset", "uci")
MIN_ELO = 1900
MIN_TIME_CONTROL = 180
MIN_MOVES = 20

HF_REPO = "Lichess/standard-chess-games"


def download_dataset():
    """Download parquet files from HuggingFace using huggingface_hub."""
    from huggingface_hub import HfApi, hf_hub_download

    os.makedirs(PARQUET_DIR, exist_ok=True)

    api = HfApi()
    # List all parquet files in the dataset repo (converted branch)
    files = api.list_repo_tree(
        HF_REPO, repo_type="dataset", revision="refs/convert/parquet",
        path_in_repo="default", recursive=True,
    )
    parquet_files = [f.rfilename for f in files if hasattr(f, "rfilename") and f.rfilename.endswith(".parquet")]
    print(f"Found {len(parquet_files)} parquet files to download.")

    for i, rfilename in enumerate(parquet_files, 1):
        local_path = os.path.join(PARQUET_DIR, rfilename.replace("/", "_"))
        if os.path.exists(local_path):
            print(f"[{i}/{len(parquet_files)}] Already exists: {local_path}")
            continue

        print(f"[{i}/{len(parquet_files)}] Downloading {rfilename}...")
        downloaded = hf_hub_download(
            HF_REPO, filename=rfilename, repo_type="dataset",
            revision="refs/convert/parquet", local_dir=PARQUET_DIR,
        )
        # Move from nested HF cache structure to flat dir
        if downloaded != local_path:
            os.rename(downloaded, local_path)

    print("All parquet files downloaded.")


def movetext_to_uci(movetext):
    """Convert PGN movetext string to UCI move string using python-chess."""
    try:
        game = chess.pgn.read_game(io.StringIO(movetext))
        if game is None:
            return None
        board = game.board()
        uci_moves = []
        for move in game.mainline_moves():
            uci_moves.append(move.uci())
            board.push(move)
        return " ".join(uci_moves)
    except Exception:
        return None


def passes_filters(row):
    """Return True if the row meets Elo and time control requirements."""
    white_elo = row.get("WhiteElo")
    black_elo = row.get("BlackElo")
    if white_elo is None or black_elo is None:
        return False
    if white_elo <= MIN_ELO or black_elo <= MIN_ELO:
        return False

    tc = row.get("TimeControl", "-")
    if tc is None or tc == "-":
        return False
    try:
        base_time = int(str(tc).split("+")[0])
    except ValueError:
        return False
    if base_time < MIN_TIME_CONTROL:
        return False

    return True


def process_parquet_file(parquet_file):
    """Read a parquet file, filter rows, convert movetext to UCI, write output."""
    out_name = Path(parquet_file).stem + ".txt"
    out_path = os.path.join(UCI_DIR, out_name)

    print(f"Processing {parquet_file} -> {out_path}")

    table = pq.read_table(
        parquet_file,
        columns=["WhiteElo", "BlackElo", "TimeControl", "movetext"],
    )
    df = table.to_pandas()

    total = len(df)
    kept = 0

    with open(out_path, "w") as out:
        for _, row in df.iterrows():
            if not passes_filters(row):
                continue
            uci = movetext_to_uci(row["movetext"])
            if uci and len(uci.split()) >= MIN_MOVES:
                out.write(uci + "\n")
                kept += 1

    print(f"Done: {out_path} ({kept}/{total} games kept)")
    return total, kept


def process_all_parquet_parallel(workers=None):
    """Convert all parquet files in PARQUET_DIR to UCI format in parallel."""
    os.makedirs(UCI_DIR, exist_ok=True)

    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "**/*.parquet"), recursive=True))
    if not parquet_files:
        # Also check flat structure
        parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*.parquet")))
    if not parquet_files:
        print(f"No parquet files found in {PARQUET_DIR}")
        return

    num_workers = workers or min(cpu_count(), len(parquet_files))
    print(f"Processing {len(parquet_files)} parquet files with {num_workers} workers...")

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_parquet_file, parquet_files)

    grand_total = sum(t for t, _ in results)
    grand_kept = sum(k for _, k in results)
    print(f"All parquet files converted to UCI. {grand_kept}/{grand_total} games kept.")


def run_all():
    """Run the full pipeline: download -> process."""
    download_dataset()
    process_all_parquet_parallel()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess data pipeline")
    parser.add_argument(
        "command",
        choices=["all", "download", "process"],
        help="all: full pipeline | download: fetch parquet files from HuggingFace | process: convert parquet to UCI",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    args = parser.parse_args()

    if args.command == "all":
        run_all()
    elif args.command == "download":
        download_dataset()
    elif args.command == "process":
        process_all_parquet_parallel(workers=args.workers)
