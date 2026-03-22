"""Query a single HuggingFace parquet file using DuckDB with predicate pushdown.

Usage:
    python src/data/query_parquet.py default/train-part1/7157.parquet
    python src/data/query_parquet.py default/train-part1/7157.parquet --save data/sample.txt
"""

import argparse

from extract_zst import (
    HF_BASE_URL,
    MIN_ELO,
    MIN_MOVES,
    MIN_TIME_CONTROL,
    _execute_with_retry,
    _make_connection,
    movetext_to_uci,
)


def query_parquet(filename: str, save_path: str | None = None):
    url = f"{HF_BASE_URL}/{filename}"
    con = _make_connection()

    total = _execute_with_retry(con, f"SELECT COUNT(*) FROM read_parquet('{url}')").fetchone()[0]
    print(f"Total games in file: {total}")

    query = f"""
        SELECT movetext
        FROM read_parquet('{url}')
        WHERE
            CAST(WhiteElo AS INTEGER) > {MIN_ELO}
            AND CAST(BlackElo AS INTEGER) > {MIN_ELO}
            AND TRY_CAST(split_part(COALESCE(TimeControl, '-'), '+', 1) AS INTEGER) >= {MIN_TIME_CONTROL}
    """
    rows = _execute_with_retry(con, query).fetchall()
    filtered = len(rows)
    print(f"After SQL filters (Elo>{MIN_ELO}, TC>={MIN_TIME_CONTROL}s): {filtered}")
    con.close()

    out_file = open(save_path, "w") if save_path else None
    kept = 0

    for i, (movetext,) in enumerate(rows):
        uci = movetext_to_uci(movetext)
        if uci and len(uci.split()) >= MIN_MOVES:
            kept += 1
            if out_file:
                out_file.write(uci + "\n")
        if (i + 1) % 5000 == 0:
            print(f"  processed {i + 1}/{filtered}...")

    if out_file:
        out_file.close()
        print(f"Saved {kept} games to {save_path}")

    print(f"After >= {MIN_MOVES} moves: {kept}")
    print(f"Final: {kept}/{total} games match all filters")
    return kept, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query a single HuggingFace parquet file with DuckDB")
    parser.add_argument("filename", help="Parquet file path in the HF repo (e.g. default/train-part1/7157.parquet)")
    parser.add_argument("--save", default=None, help="Save matching UCI games to this file")
    args = parser.parse_args()

    query_parquet(args.filename, save_path=args.save)
