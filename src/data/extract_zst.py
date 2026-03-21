import os
import glob
import argparse
import io
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed
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


DOWNLOAD_WORKERS = 3


def _download_one(args):
    """Download a single parquet file from HuggingFace. Returns (rfilename, status)."""
    from huggingface_hub import hf_hub_download

    idx, total, rfilename = args
    local_path = os.path.join(PARQUET_DIR, rfilename.replace("/", "_"))
    if os.path.exists(local_path):
        print(f"[{idx}/{total}] Already exists: {local_path}")
        return rfilename, "skipped"

    print(f"[{idx}/{total}] Downloading {rfilename}...")
    downloaded = hf_hub_download(
        HF_REPO, filename=rfilename, repo_type="dataset",
        revision="refs/convert/parquet", local_dir=PARQUET_DIR,
    )
    # Move from nested HF cache structure to flat dir
    if downloaded != local_path:
        os.rename(downloaded, local_path)

    print(f"[{idx}/{total}] Done: {rfilename}")
    return rfilename, "downloaded"


def download_dataset(workers=DOWNLOAD_WORKERS):
    """Download parquet files from HuggingFace using huggingface_hub."""
    from huggingface_hub import HfApi

    os.makedirs(PARQUET_DIR, exist_ok=True)

    api = HfApi()
    # List all parquet files in the dataset repo (converted branch)
    files = api.list_repo_tree(
        HF_REPO, repo_type="dataset", revision="refs/convert/parquet",
        path_in_repo="default", recursive=True,
    )
    parquet_files = [f.rfilename for f in files if hasattr(f, "rfilename") and f.rfilename.endswith(".parquet")]
    total = len(parquet_files)
    print(f"Found {total} parquet files to download ({workers} parallel workers).")

    tasks = [(i, total, rf) for i, rf in enumerate(parquet_files, 1)]

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_one, t): t for t in tasks}
        for future in as_completed(futures):
            future.result()  # raise any exceptions

    print("All parquet files downloaded.")


def movetext_to_uci(movetext):
    """Convert PGN movetext string to UCI move string using python-chess."""
    try:
        game = chess.pgn.read_game(io.StringIO(movetext))
        if game is None:
            return None
        # read_game already validates moves and generates chess.Move objects.
        # Calling board.push() for every move is redundant overhead.
        return " ".join(move.uci() for move in game.mainline_moves())
    except Exception:
        return None


def process_parquet_file(parquet_file):
    """Read a parquet file, filter rows, convert movetext to UCI, write output."""
    out_name = Path(parquet_file).stem + ".txt"
    out_path = os.path.join(UCI_DIR, out_name)

    print(f"Processing {parquet_file} -> {out_path}")

    # Read specific columns
    table = pq.read_table(
        parquet_file,
        columns=["WhiteElo", "BlackElo", "TimeControl", "movetext"],
    )
    df = table.to_pandas()

    total = len(df)

    # Fast vectorized filtering (avoids iterating through non-matching rows)
    # 1. Elo > MIN_ELO
    df["WhiteElo"] = df["WhiteElo"].fillna(0).astype(int)
    df["BlackElo"] = df["BlackElo"].fillna(0).astype(int)
    mask = (df["WhiteElo"] > MIN_ELO) & (df["BlackElo"] > MIN_ELO)
    
    # 2. Extract base time from TimeControl
    # Fill missing values and convert to string so string operations don't fail
    tc_series = df["TimeControl"].fillna("-").astype(str)
    # Extract the part before '+' using pandas string methods
    base_time_str = tc_series.str.split("+", n=1).str[0]
    
    # Safely convert extracted base time strings to numeric (coercing errors to NaN and then filling with 0)
    import pandas as pd
    base_time = pd.to_numeric(base_time_str, errors="coerce").fillna(0)
    
    # 3. Combine masks
    mask &= (base_time >= MIN_TIME_CONTROL)
    
    # Apply filters
    filtered_df = df[mask]

    kept = 0

    # Itertuples is significantly faster than iterrows
    with open(out_path, "w") as out:
        for row in filtered_df.itertuples(index=False):
            uci = movetext_to_uci(row.movetext)
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
        download_dataset(workers=args.workers or DOWNLOAD_WORKERS)
    elif args.command == "process":
        process_all_parquet_parallel(workers=args.workers)
