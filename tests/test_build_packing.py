import struct

import numpy as np

from src.data.build_packing import build_packing
from src.training.dataset import ChessDataset, PAD_ID


def _write_idx(path, offsets):
    offsets = np.asarray(offsets, dtype=np.uint64)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(offsets) - 1))
        f.write(offsets.tobytes())


def test_build_packing_drops_zero_length_and_overlong_games(tmp_path):
    bin_dir = tmp_path / "bin"
    out_dir = tmp_path / "packing"
    bin_dir.mkdir()

    # Offsets encode four games with lengths [0, 3, 7, 3].
    tokens = np.array([11, 12, 13, 21, 22, 23, 24, 25, 26, 27, 31, 32, 33], dtype=np.uint16)
    (bin_dir / "sample.bin").write_bytes(tokens.tobytes())
    _write_idx(bin_dir / "sample.idx", [0, 0, 3, 10, 13])

    build_packing(bin_dir, out_dir, max_seq_len=4, seed=0, max_files=None)

    dataset = ChessDataset(str(out_dir), max_seq_len=4)

    assert len(dataset) == 2

    for i in range(len(dataset)):
        input_ids, labels = dataset[i]
        assert input_ids.shape == (4,)
        assert labels.shape == (4,)
        assert int((labels != PAD_ID).sum().item()) > 0


def test_build_packing_errors_when_no_games_are_packable(tmp_path):
    bin_dir = tmp_path / "bin"
    out_dir = tmp_path / "packing"
    bin_dir.mkdir()

    tokens = np.array([11, 12, 13, 21, 22, 23], dtype=np.uint16)
    (bin_dir / "sample.bin").write_bytes(tokens.tobytes())
    _write_idx(bin_dir / "sample.idx", [0, 0, 6])

    try:
        build_packing(bin_dir, out_dir, max_seq_len=4, seed=0, max_files=None)
    except ValueError as exc:
        assert "No packable games found" in str(exc)
    else:
        raise AssertionError("expected build_packing to reject all-invalid input")
