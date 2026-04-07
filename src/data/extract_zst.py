"""Chess data pipeline: download HuggingFace parquet files and convert to UCI.

Default mode downloads parquet files locally via hf_hub_download (with built-in
caching/resume), then queries them with DuckDB locally — no 429 rate limits.
Use --remote for direct httpfs queries (only practical for small batches).
"""

import argparse
import io
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
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

DEFAULT_WORKERS = 16  # processing workers (PGN→UCI is CPU-bound)
DOWNLOAD_WORKERS = 3  # download workers (conservative for HF rate limits)
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


def _unique_stem(rfilename: str) -> str:
    """Derive a unique stem from a parquet path like 'default/train-part0/0000.parquet'."""
    parts = Path(rfilename).parts
    # e.g. "train-part0_0000"
    return f"{parts[-2]}_{Path(rfilename).stem}"


def _process_one(idx: int, total_files: int, rfilename: str) -> tuple[str, int, int, int]:
    """Query a single remote parquet file with DuckDB, filter, and write UCI output.

    Returns (filename, total_rows, filtered_rows, kept_after_uci).
    """
    url = f"{HF_BASE_URL}/{rfilename}"
    stem = _unique_stem(rfilename)
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
    stem = _unique_stem(rfilename)
    out_path = os.path.join(UCI_DIR, stem + ".txt")

    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        with open(out_path) as f:
            existing = sum(1 for _ in f)
        print(f"[{idx}/{total_files}] Skipping {rfilename} (already have {existing} games)")
        return rfilename, 0, 0, existing

    con = duckdb.connect()
    con.execute("PRAGMA threads=1")
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
    """Download and process all parquet files from HuggingFace.

    In local mode, downloads and processing run concurrently: download workers
    feed a queue that processing workers consume from immediately, so processing
    starts as soon as the first file is downloaded rather than waiting for all
    4.4 TB to arrive.
    """
    os.makedirs(UCI_DIR, exist_ok=True)

    print("Listing parquet files on HuggingFace...")
    parquet_files = _list_parquet_files()
    if limit:
        parquet_files = parquet_files[:limit]

    n_files = len(parquet_files)
    num_workers = workers or DEFAULT_WORKERS
    mode = "remote httpfs" if remote else "streaming download+process"
    print(f"Found {n_files} parquet files to process "
          f"({download_workers} download workers, {num_workers} process workers, {mode})\n")

    if remote:
        _run_processing(
            parquet_files, n_files, num_workers,
            process_fn=_process_one,
        )
    else:
        _run_streaming_pipeline(
            parquet_files, n_files,
            download_workers=download_workers,
            process_workers=num_workers,
            cache_dir=cache_dir,
        )


def _run_streaming_pipeline(
    parquet_files: list[str],
    n_files: int,
    download_workers: int,
    process_workers: int,
    cache_dir: str | None = None,
):
    """Stream download→process: files are processed as soon as they're downloaded."""
    dest = cache_dir or PARQUET_DIR
    os.makedirs(dest, exist_ok=True)

    # Queue holds (index, rfilename, local_path) tuples; None is the sentinel.
    work_queue: queue.Queue[tuple[int, str, str] | None] = queue.Queue(
        maxsize=download_workers * 2,
    )

    # Shared progress state
    lock = threading.Lock()
    stats = {"files_done": 0, "grand_filtered": 0, "grand_kept": 0}
    bar_width = 40
    download_errors: list[str] = []

    def _update_progress(filtered: int, kept: int):
        with lock:
            stats["grand_filtered"] += filtered
            stats["grand_kept"] += kept
            stats["files_done"] += 1
            done = stats["files_done"]
            total_kept = stats["grand_kept"]
        pct = done / n_files
        filled = int(bar_width * pct)
        bar = "█" * filled + "░" * (bar_width - filled)
        sys.stdout.write(
            f"\r  [{bar}] {done}/{n_files} files | {total_kept:,} games kept"
        )
        sys.stdout.flush()

    def _downloader():
        """Download files and put them on the queue for processing."""
        with ThreadPoolExecutor(max_workers=download_workers) as dl_pool:
            futures = {
                dl_pool.submit(_download_one, i, n_files, rf, dest): (i, rf)
                for i, rf in enumerate(parquet_files, 1)
            }
            for future in as_completed(futures):
                idx, rf = futures[future]
                try:
                    local_path = future.result()
                    work_queue.put((idx, rf, local_path))
                except Exception as e:
                    print(f"\n  [DOWNLOAD ERROR] {rf}: {e}")
                    download_errors.append(rf)
        # Signal processing workers to shut down
        for _ in range(process_workers):
            work_queue.put(None)

    def _processor():
        """Pull downloaded files from the queue and process them."""
        while True:
            item = work_queue.get()
            if item is None:
                return
            idx, rf, local_path = item
            try:
                _, _, filtered, kept = _process_one_local(idx, n_files, rf, local_path)
                _update_progress(filtered, kept)
            except Exception as e:
                print(f"\n  [PROCESS ERROR] {rf}: {e}")
                _update_progress(0, 0)

    # Start the downloader thread (it manages its own pool internally)
    dl_thread = threading.Thread(target=_downloader, daemon=True)
    dl_thread.start()

    # Start processing workers
    proc_threads = [
        threading.Thread(target=_processor, daemon=True)
        for _ in range(process_workers)
    ]
    for t in proc_threads:
        t.start()

    # Wait for everything to finish
    dl_thread.join()
    for t in proc_threads:
        t.join()

    print(f"\n\nDone. {stats['grand_kept']:,} games kept "
          f"({stats['grand_filtered']:,} passed SQL filters). Output: {UCI_DIR}/")
    if download_errors:
        print(f"  {len(download_errors)} files failed to download: {download_errors[:5]}{'...' if len(download_errors) > 5 else ''}")


def _run_processing(parquet_files, n_files, num_workers, process_fn):
    """Run processing with a progress bar (used for remote httpfs mode)."""
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


def _process_one_local_wrapper(args: tuple) -> tuple[int, int]:
    """Top-level wrapper for ProcessPoolExecutor (picklable)."""
    idx, total, rfilename, local_path = args
    try:
        _, _, filtered, kept = _process_one_local(idx, total, rfilename, local_path)
        return filtered, kept
    except Exception as e:
        print(f"\n  [PROCESS ERROR] {rfilename}: {e}", flush=True)
        return 0, 0


def process_local_only(workers: int | None = None, cache_dir: str | None = None):
    """Process parquet files already on disk. No downloads, no HF API calls.

    Walks the local parquet directory, skips files whose UCI output already
    exists, and parallelizes across processes (CPU-bound, GIL-bound work).
    """
    os.makedirs(UCI_DIR, exist_ok=True)
    dest = cache_dir or PARQUET_DIR

    if not os.path.isdir(dest):
        print(f"ERROR: parquet directory not found: {dest}")
        sys.exit(1)

    # Walk local parquet dir; reconstruct rfilename-style relative paths
    # so _unique_stem and log lines stay consistent with the HF layout.
    print(f"Scanning local parquet files in {dest}...")
    all_local: list[tuple[str, str]] = []  # (rfilename, local_path)
    for root, _, files in os.walk(dest):
        for f in files:
            if not f.endswith(".parquet"):
                continue
            local_path = os.path.join(root, f)
            rfilename = os.path.relpath(local_path, dest)
            all_local.append((rfilename, local_path))
    all_local.sort()

    # Filter out files whose UCI output already exists and is non-empty
    todo: list[tuple[str, str]] = []
    skipped = 0
    for rf, lp in all_local:
        out_path = os.path.join(UCI_DIR, _unique_stem(rf) + ".txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            skipped += 1
            continue
        todo.append((rf, lp))

    n_total = len(all_local)
    n_todo = len(todo)
    num_workers = workers or DEFAULT_WORKERS
    print(f"Found {n_total} local parquet files "
          f"({skipped} already processed, {n_todo} to do)")
    print(f"Processing with {num_workers} processes (multiprocessing, bypasses GIL)\n")

    if n_todo == 0:
        print("Nothing to do.")
        return

    grand_filtered = 0
    grand_kept = 0
    files_done = 0
    bar_width = 40
    t0 = time.time()

    tasks = [(i, n_todo, rf, lp) for i, (rf, lp) in enumerate(todo, 1)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for filtered, kept in executor.map(
            _process_one_local_wrapper, tasks, chunksize=1,
        ):
            grand_filtered += filtered
            grand_kept += kept
            files_done += 1
            pct = files_done / n_todo
            filled = int(bar_width * pct)
            bar = "█" * filled + "░" * (bar_width - filled)
            elapsed = time.time() - t0
            rate = files_done / elapsed if elapsed > 0 else 0
            eta = (n_todo - files_done) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  [{bar}] {files_done}/{n_todo} | "
                f"{grand_kept:,} games kept | "
                f"{rate:.1f} files/s | ETA {eta/60:.0f}m"
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
        help="all: download+process; process: process local parquets only; download: download only",
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
    elif args.command == "process":
        process_local_only(workers=args.workers, cache_dir=args.cache_dir)
    else:
        process_all(
            workers=args.workers, limit=args.limit,
            remote=args.remote,
            download_workers=args.download_workers,
            cache_dir=args.cache_dir,
        )
