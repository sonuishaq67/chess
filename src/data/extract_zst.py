"""Chess data pipeline: download HuggingFace parquet files and convert to UCI.

Default mode downloads parquet files locally via hf_hub_download (with built-in
caching/resume), then queries them with DuckDB locally — no 429 rate limits.
Use --remote for direct httpfs queries (only practical for small batches).
"""

import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess.pgn
import duckdb

BASE_DIR = os.path.dirname(os.path.abspath(__file__)).rsplit("/src", 1)[0]
UCI_DIR = os.path.join(BASE_DIR, "dataset", "uci")
PARQUET_DIR = os.path.join(BASE_DIR, "dataset", "parquet")

MIN_ELO = 2200
MIN_TIME_CONTROL = 180
MIN_MOVES = 20

HF_REPO = "Lichess/standard-chess-games"
HF_REVISION = "refs/convert/parquet"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION.replace('/', '%2F')}"

DEFAULT_WORKERS = 8   # processing workers (PGN→UCI is CPU-bound)
DOWNLOAD_WORKERS = 5  # download workers (conservative for HF rate limits)
MAX_RETRIES = 7
RETRY_BASE_DELAY = 2  # seconds, doubles each retry


def _list_parquet_files() -> list[str]:
    """List all parquet file paths in the HuggingFace dataset repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_tree(
        HF_REPO, repo_type="dataset", revision=HF_REVISION,
        path_in_repo="default", recursive=True,
    )
    return [
        f.rfilename for f in files
        if hasattr(f, "rfilename") and f.rfilename.endswith(".parquet")
    ]


def _make_connection() -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with httpfs and HuggingFace auth."""
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    token = os.environ.get("HF_TOKEN", "")
    con.execute(f"""
        CREATE SECRET hf (
            TYPE HUGGINGFACE,
            TOKEN '{token}'
        );
    """)
    return con


def _execute_with_retry(con, query: str, max_retries: int = MAX_RETRIES):
    """Execute a DuckDB query with exponential backoff on HTTP 429."""
    for attempt in range(max_retries + 1):
        try:
            return con.execute(query)
        except duckdb.HTTPException as e:
            if "429" in str(e) and attempt < max_retries:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  429 rate limited, retrying in {delay}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                raise


def movetext_to_uci(movetext: str) -> str | None:
    """Convert PGN movetext string to UCI move string using python-chess."""
    try:
        game = chess.pgn.read_game(io.StringIO(movetext))
        if game is None:
            return None
        return " ".join(move.uci() for move in game.mainline_moves())
    except Exception:
        return None


def _build_query(source: str) -> str:
    """Build the SQL query for filtering chess games from a parquet source."""
    return f"""
        SELECT movetext
        FROM read_parquet('{source}')
        WHERE
            CAST(WhiteElo AS INTEGER) > {MIN_ELO}
            AND CAST(BlackElo AS INTEGER) > {MIN_ELO}
            AND TRY_CAST(split_part(COALESCE(TimeControl, '-'), '+', 1) AS INTEGER) >= {MIN_TIME_CONTROL}
    """


def _write_uci(rows: list, out_path: str) -> int:
    """Convert rows to UCI and write to file. Returns number of games kept."""
    kept = 0
    with open(out_path, "w") as out:
        for (movetext,) in rows:
            uci = movetext_to_uci(movetext)
            if uci and len(uci.split()) >= MIN_MOVES:
                out.write(uci + "\n")
                kept += 1
    if kept == 0 and os.path.exists(out_path):
        os.remove(out_path)
    return kept


def _process_one(idx: int, total_files: int, rfilename: str) -> tuple[str, int, int, int]:
    """Query a single remote parquet file with DuckDB, filter, and write UCI output.

    Returns (filename, total_rows, filtered_rows, kept_after_uci).
    """
    url = f"{HF_BASE_URL}/{rfilename}"
    stem = Path(rfilename).stem
    out_path = os.path.join(UCI_DIR, stem + ".txt")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            existing = sum(1 for _ in f)
        print(f"[{idx}/{total_files}] Skipping {rfilename} (already have {existing} games)")
        return rfilename, 0, 0, existing

    con = _make_connection()
    rows = _execute_with_retry(con, _build_query(url)).fetchall()
    filtered = len(rows)
    con.close()

    kept = _write_uci(rows, out_path)

    print(f"[{idx}/{total_files}] {rfilename}: {kept} games kept ({filtered} passed SQL filters)")
    return rfilename, 0, filtered, kept


def _process_one_local(
    idx: int, total_files: int, rfilename: str, local_path: str,
) -> tuple[str, int, int, int]:
    """Query a local parquet file with DuckDB, filter, and write UCI output."""
    stem = Path(rfilename).stem
    out_path = os.path.join(UCI_DIR, stem + ".txt")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            existing = sum(1 for _ in f)
        print(f"[{idx}/{total_files}] Skipping {rfilename} (already have {existing} games)")
        return rfilename, 0, 0, existing

    con = duckdb.connect()
    rows = con.execute(_build_query(local_path)).fetchall()
    filtered = len(rows)
    con.close()

    kept = _write_uci(rows, out_path)

    print(f"[{idx}/{total_files}] {rfilename}: {kept} games kept ({filtered} passed SQL filters)")
    return rfilename, 0, filtered, kept


# ---------------------------------------------------------------------------
# Download helpers (opt-in via --download-first)
# ---------------------------------------------------------------------------

def _download_one(idx: int, total: int, rfilename: str, local_dir: str) -> str:
    """Download a single parquet file from HuggingFace. Returns local path."""
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        HF_REPO, filename=rfilename,
        repo_type="dataset", revision=HF_REVISION,
        local_dir=local_dir,
    )
    print(f"[{idx}/{total}] Downloaded {rfilename}")
    return local_path


def _download_parquet_files(
    parquet_files: list[str], workers: int = DOWNLOAD_WORKERS,
    local_dir: str | None = None,
) -> dict[str, str]:
    """Download parquet files from HuggingFace. Returns {rfilename: local_path}."""
    dest = local_dir or PARQUET_DIR
    os.makedirs(dest, exist_ok=True)
    n = len(parquet_files)
    print(f"Downloading {n} parquet files to {dest} ({workers} download workers)...")

    paths: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, i, n, rf, dest): rf
            for i, rf in enumerate(parquet_files, 1)
        }
        for future in as_completed(futures):
            rf = futures[future]
            paths[rf] = future.result()

    print(f"Download complete: {len(paths)} files\n")
    return paths


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def process_all(
    workers: int | None = None, limit: int | None = None,
    remote: bool = False, download_workers: int = DOWNLOAD_WORKERS,
    cache_dir: str | None = None,
):
    """Download and process all parquet files from HuggingFace."""
    os.makedirs(UCI_DIR, exist_ok=True)

    print("Listing parquet files on HuggingFace...")
    parquet_files = _list_parquet_files()
    if limit:
        parquet_files = parquet_files[:limit]

    n_files = len(parquet_files)
    num_workers = workers or DEFAULT_WORKERS
    mode = "remote httpfs" if remote else "download-first"
    print(f"Found {n_files} parquet files to process ({num_workers} workers, {mode})\n")

    if remote:
        _run_processing(
            parquet_files, n_files, num_workers,
            process_fn=_process_one,
        )
    else:
        local_paths = _download_parquet_files(
            parquet_files, workers=download_workers, local_dir=cache_dir,
        )
        _run_processing(
            parquet_files, n_files, num_workers,
            process_fn=lambda i, n, rf: _process_one_local(i, n, rf, local_paths[rf]),
        )


def _run_processing(parquet_files, n_files, num_workers, process_fn):
    """Run processing with a progress bar."""
    grand_filtered = 0
    grand_kept = 0
    files_done = 0
    bar_width = 40

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_fn, i, n_files, rf): rf
            for i, rf in enumerate(parquet_files, 1)
        }
        for future in as_completed(futures):
            _, _, filtered, kept = future.result()
            grand_filtered += filtered
            grand_kept += kept
            files_done += 1
            pct = files_done / n_files
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            sys.stdout.write(
                f"\r  [{bar}] {files_done}/{n_files} files | "
                f"{grand_kept:,} games kept"
            )
            sys.stdout.flush()

    print(f"\n\nDone. {grand_kept:,} games kept "
          f"({grand_filtered:,} passed SQL filters). Output: {UCI_DIR}/")


def download_only(
    limit: int | None = None,
    download_workers: int = DOWNLOAD_WORKERS, cache_dir: str | None = None,
):
    """Download parquet files without processing (for HPC split workflows)."""
    print("Listing parquet files on HuggingFace...")
    parquet_files = _list_parquet_files()
    if limit:
        parquet_files = parquet_files[:limit]

    _download_parquet_files(
        parquet_files, workers=download_workers, local_dir=cache_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess data pipeline (DuckDB + httpfs)")
    parser.add_argument(
        "command",
        choices=["all", "process", "download"],
        nargs="?",
        default="all",
        help="all/process: query + convert to UCI; download: download parquets only (default: all)",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help=f"Number of parallel processing workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process first N parquet files (useful for testing)",
    )
    parser.add_argument(
        "--remote", action="store_true",
        help="Query parquet files remotely via httpfs (only practical for small batches)",
    )
    parser.add_argument(
        "--download-workers", type=int, default=DOWNLOAD_WORKERS,
        help=f"Number of parallel download workers (default: {DOWNLOAD_WORKERS})",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=f"Directory for downloaded parquet files (default: {PARQUET_DIR})",
    )
    args = parser.parse_args()

    if args.command == "download":
        download_only(
            limit=args.limit,
            download_workers=args.download_workers, cache_dir=args.cache_dir,
        )
    else:
        process_all(
            workers=args.workers, limit=args.limit,
            remote=args.remote,
            download_workers=args.download_workers,
            cache_dir=args.cache_dir,
        )
