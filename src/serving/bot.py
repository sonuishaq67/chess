import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import chess
import httpx

from src.serving.inference import ChessOnnxInference

log = logging.getLogger("chess-bot")

LICHESS_API = "https://lichess.org/api"
LOG_DIR = Path.home() / "logs"
LOG_FMT = "%(asctime)s %(name)s %(levelname)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _setup_main_logger():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / "main.log")
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    # Also log to stderr so systemd journal captures it
    stderr = logging.StreamHandler()
    stderr.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    log.addHandler(stderr)


def _make_game_logger(game_id: str) -> logging.Logger:
    game_log = logging.getLogger(f"chess-bot.game.{game_id}")
    game_log.setLevel(logging.DEBUG)
    game_log.propagate = True  # also goes to main.log via parent

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(LOG_DIR / f"game_{game_id}.log")
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATEFMT))
    game_log.addHandler(handler)
    return game_log


# ---------------------------------------------------------------------------
# Move selection strategies
# ---------------------------------------------------------------------------

class MoveStrategy:
    def __init__(
        self,
        engine: ChessOnnxInference,
        mode: str = "greedy",
        temperature: float = 0.6,
        top_k: int = 3,
    ):
        self.engine = engine
        self.mode = mode
        self.temperature = temperature
        self.top_k = top_k

    def select_move(self, uci_moves: list[str]) -> str | None:
        if self.mode == "greedy":
            return self.engine.predict_move(uci_moves, temperature=0.0)

        if self.mode == "sample":
            return self.engine.predict_move(uci_moves, temperature=self.temperature)

        if self.mode == "top_k":
            candidates = self.engine.predict_top_k(
                uci_moves, k=self.top_k, temperature=self.temperature
            )
            if not candidates:
                return None
            # Weighted sample from top-k by their probabilities
            import numpy as np
            moves, probs = zip(*candidates)
            probs = np.array(probs)
            probs /= probs.sum()
            return str(np.random.choice(moves, p=probs))

        raise ValueError(f"Unknown move strategy: {self.mode}")

class LichessBot:
    def __init__(
        self,
        token: str,
        strategy: MoveStrategy,
        accept_variants: set[str] | None = None,
        accept_rated: bool = True,
        accept_casual: bool = True,
        max_concurrent_games: int = 1,
    ):
        self.token = token
        self.strategy = strategy
        self.accept_variants = accept_variants or {"standard"}
        self.accept_rated = accept_rated
        self.accept_casual = accept_casual
        self.max_concurrent_games = max_concurrent_games

        self._headers = {"Authorization": f"Bearer {token}"}
        self._active_games: set[str] = set()
        self._running = True

    def _get(self, path: str, **kwargs) -> httpx.Response:
        return httpx.get(
            f"{LICHESS_API}{path}", headers=self._headers, timeout=10, **kwargs
        )

    def _post(self, path: str, **kwargs) -> httpx.Response:
        return httpx.post(
            f"{LICHESS_API}{path}", headers=self._headers, timeout=10, **kwargs
        )

    def get_account(self) -> dict:
        resp = self._get("/account")
        resp.raise_for_status()
        return resp.json()

    def accept_challenge(self, challenge_id: str):
        resp = self._post(f"/challenge/{challenge_id}/accept")
        if resp.status_code == 200:
            log.info("Accepted challenge %s", challenge_id)
        else:
            log.warning("Failed to accept challenge %s: %s", challenge_id, resp.text)

    def decline_challenge(self, challenge_id: str, reason: str = "generic"):
        resp = self._post(
            f"/challenge/{challenge_id}/decline",
            json={"reason": reason},
        )
        log.info("Declined challenge %s (%s): %s", challenge_id, reason, resp.status_code)

    def make_move(self, game_id: str, move: str):
        resp = self._post(f"/bot/game/{game_id}/move/{move}")
        if resp.status_code == 200:
            log.debug("Played %s in game %s", move, game_id)
        else:
            log.warning("Move %s failed in game %s: %s", move, game_id, resp.text)

    def send_chat(self, game_id: str, text: str, room: str = "player"):
        self._post(
            f"/bot/game/{game_id}/chat",
            json={"room": room, "text": text},
        )

    def resign_game(self, game_id: str):
        self._post(f"/bot/game/{game_id}/resign")
        log.info("Resigned game %s", game_id)

# ---------------------------------------------------------------------------

    def _should_accept(self, challenge: dict) -> tuple[bool, str]:
        """Decide whether to accept an incoming challenge."""
        variant = challenge.get("variant", {}).get("key", "standard")
        if variant not in self.accept_variants:
            return False, "variant"

        rated = challenge.get("rated", False)
        if rated and not self.accept_rated:
            return False, "casual"
        if not rated and not self.accept_casual:
            return False, "rated"

        if len(self._active_games) >= self.max_concurrent_games:
            return False, "later"

        return True, ""

    def _handle_challenge(self, event: dict):
        challenge = event["challenge"]
        cid = challenge["id"]
        challenger = challenge.get("challenger", {}).get("name", "?")
        variant = challenge.get("variant", {}).get("key", "?")
        log.info("Challenge from %s (variant=%s, id=%s)", challenger, variant, cid)

        accept, reason = self._should_accept(challenge)
        if accept:
            self.accept_challenge(cid)
        else:
            self.decline_challenge(cid, reason)

    # -- Game play ----------------------------------------------------------

    def _parse_moves(self, moves_str: str) -> list[str]:
        if not moves_str or not moves_str.strip():
            return []
        return moves_str.strip().split()

    def _is_my_turn(self, moves: list[str], my_color: str) -> bool:
        if my_color == "white":
            return len(moves) % 2 == 0
        return len(moves) % 2 == 1

    def play_game(self, game_id: str):
        glog = _make_game_logger(game_id)
        glog.info("Starting game %s", game_id)
        self._active_games.add(game_id)
        my_color = None

        try:
            with httpx.stream(
                "GET",
                f"{LICHESS_API}/bot/game/stream/{game_id}",
                headers=self._headers,
                timeout=None,
            ) as stream:
                for line in stream.iter_lines():
                    if not self._running:
                        break
                    if not line:
                        continue

                    event = json.loads(line)
                    event_type = event.get("type")

                    if event_type == "gameFull":
                        # Initial full game state
                        white_id = event.get("white", {}).get("id", "")
                        account = self.get_account()
                        my_id = account.get("id", "")
                        my_color = "white" if white_id == my_id else "black"
                        glog.info("Playing as %s in game %s", my_color, game_id)

                        self.send_chat(
                            game_id,
                            "Good luck! I'm a transformer-based chess bot.",
                        )

                        moves = self._parse_moves(event.get("state", {}).get("moves", ""))
                        if self._is_my_turn(moves, my_color):
                            self._play_turn(game_id, moves, glog)

                    elif event_type == "gameState":
                        status = event.get("status", "started")
                        if status != "started":
                            glog.info("Game %s ended: %s", game_id, status)
                            break

                        moves = self._parse_moves(event.get("moves", ""))
                        if my_color and self._is_my_turn(moves, my_color):
                            self._play_turn(game_id, moves, glog)

                    elif event_type == "gameFinish":
                        glog.info("Game %s finished", game_id)
                        break

        except httpx.HTTPStatusError as e:
            glog.error("HTTP error in game %s: %s", game_id, e)
        except Exception as e:
            glog.error("Error in game %s: %s", game_id, e, exc_info=True)
        finally:
            self._active_games.discard(game_id)
            glog.info("Game %s loop ended", game_id)
            # Clean up per-game file handler
            for h in glog.handlers[:]:
                if isinstance(h, logging.FileHandler):
                    h.close()
                    glog.removeHandler(h)

    def _play_turn(self, game_id: str, moves: list[str], glog: logging.Logger):
        try:
            move = self.strategy.select_move(moves)
            if move is None:
                glog.info("No legal move found in game %s, game must be over", game_id)
                return
            glog.info("Move %d: %s", len(moves) + 1, move)
            self.make_move(game_id, move)
        except Exception as e:
            glog.error("Error selecting move in game %s: %s", game_id, e, exc_info=True)
            # Fall back to first legal move
            board = chess.Board()
            for m in moves:
                board.push_uci(m)
            if not board.is_game_over():
                fallback = list(board.legal_moves)[0].uci()
                glog.warning("Falling back to %s in game %s", fallback, game_id)
                self.make_move(game_id, fallback)

    def run(self):
        """Main event loop: stream events and dispatch."""
        account = self.get_account()
        username = account.get("username", "?")
        log.info("Logged in as %s", username)

        # Handle graceful shutdown
        def _shutdown(signum, frame):
            log.info("Received signal %s, shutting down...", signum)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        while self._running:
            try:
                log.info("Connecting to event stream...")
                with httpx.stream(
                    "GET",
                    f"{LICHESS_API}/stream/event",
                    headers=self._headers,
                    timeout=None,
                ) as stream:
                    for line in stream.iter_lines():
                        if not self._running:
                            break
                        if not line:
                            continue

                        event = json.loads(line)
                        event_type = event.get("type")

                        if event_type == "challenge":
                            self._handle_challenge(event)

                        elif event_type == "gameStart":
                            game_id = event["game"]["gameId"]
                            log.info("Game started: %s", game_id)
                            # Play game synchronously (single-game bot)
                            self.play_game(game_id)

                        elif event_type == "gameFinish":
                            game_id = event["game"]["gameId"]
                            log.info("Game finished event: %s", game_id)

            except httpx.ReadTimeout:
                log.warning("Event stream timed out, reconnecting...")
            except httpx.HTTPStatusError as e:
                log.error("HTTP error on event stream: %s", e)
                time.sleep(5)
            except Exception as e:
                if self._running:
                    log.error("Event stream error: %s", e, exc_info=True)
                    time.sleep(5)

        log.info("Bot shut down.")


def main():
    parser = argparse.ArgumentParser(description="Lichess Bot powered by chess transformer")
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--token", default=None, help="Lichess API token (or set LICHESS_BOT_TOKEN)")
    parser.add_argument(
        "--strategy",
        choices=["greedy", "sample", "top_k"],
        default="greedy",
        help="Move selection strategy (default: greedy)",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k candidates for top_k strategy")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime inference threads")
    parser.add_argument("--max-games", type=int, default=1, help="Max concurrent games")
    args = parser.parse_args()

    token = args.token or os.environ.get("LICHESS_BOT_TOKEN")
    if not token:
        print("Error: provide --token or set LICHESS_BOT_TOKEN env var", file=sys.stderr)
        sys.exit(1)

    _setup_main_logger()

    log.info("Loading ONNX model from %s", args.model)
    engine = ChessOnnxInference(args.model, args.vocab, num_threads=args.threads)

    strategy = MoveStrategy(
        engine,
        mode=args.strategy,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    log.info("Strategy: %s (temp=%.2f, top_k=%d)", args.strategy, args.temperature, args.top_k)

    bot = LichessBot(
        token=token,
        strategy=strategy,
        max_concurrent_games=args.max_games,
    )
    bot.run()


if __name__ == "__main__":
    main()
