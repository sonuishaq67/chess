import json

import chess
import numpy as np
import onnxruntime as ort


class ChessOnnxInference:
    PAD, BOS, EOS, MASK = 0, 1, 2, 3
    NUM_SPECIAL = 4

    def __init__(self, model_path: str, vocab_path: str, num_threads: int = 4):
        with open(vocab_path) as f:
            self.token2id = json.load(f)
        self.id2token = {v: k for k, v in self.token2id.items()}

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path, opts, providers=["CPUExecutionProvider"]
        )

    def encode(self, uci_moves: list[str]) -> np.ndarray:
        ids = [self.BOS]
        for move in uci_moves:
            tid = self.token2id.get(move)
            if tid is None:
                raise ValueError(f"Unknown UCI move: {move}")
            ids.append(tid)
        return np.array([ids], dtype=np.int64)

    def _get_last_logits(self, uci_moves: list[str]) -> np.ndarray:
        tokens = self.encode(uci_moves)
        logits = self.session.run(None, {"tokens": tokens})[0]  # [1, seq, vocab]
        last_logits = logits[0, -1, :]  # [vocab]

        # Mask out special tokens
        last_logits[: self.NUM_SPECIAL] = -np.inf
        return last_logits

    @staticmethod
    def _board_from_moves(uci_moves: list[str]) -> chess.Board:
        board = chess.Board()
        for m in uci_moves:
            board.push_uci(m)
        return board

    def predict_move(
        self, uci_moves: list[str], temperature: float = 0.0
    ) -> str | None:
        board = self._board_from_moves(uci_moves)
        if board.is_game_over():
            return None

        legal_uci = {m.uci() for m in board.legal_moves}
        logits = self._get_last_logits(uci_moves)

        if temperature <= 0.0:
            for idx in np.argsort(logits)[::-1]:
                token = self.id2token[int(idx)]
                if token in legal_uci:
                    return token
        else:
            logits = logits / temperature
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            # Zero out illegal moves, renormalize, sample
            legal_mask = np.zeros_like(probs)
            for move_uci in legal_uci:
                tid = self.token2id.get(move_uci)
                if tid is not None:
                    legal_mask[tid] = 1.0
            probs *= legal_mask
            probs /= probs.sum()
            idx = int(np.random.choice(len(probs), p=probs))
            return self.id2token[idx]

        return None

    def predict_top_k(
        self, uci_moves: list[str], k: int = 5, temperature: float = 1.0
    ) -> list[tuple[str, float]]:
        board = self._board_from_moves(uci_moves)
        legal_uci = {m.uci() for m in board.legal_moves}

        logits = self._get_last_logits(uci_moves)
        logits = logits / max(temperature, 1e-8)
        logits -= np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))

        results = []
        for idx in np.argsort(probs)[::-1]:
            token = self.id2token[int(idx)]
            if token in legal_uci:
                results.append((token, float(probs[idx])))
                if len(results) >= k:
                    break
        return results
