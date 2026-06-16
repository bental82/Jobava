"""Optional Stockfish integration.

Everything here degrades gracefully: if no engine path is given, the binary is
missing, or the engine misbehaves, the wrapper reports ``available == False``
and the analysis helpers return ``None``. The rest of the app keeps working.

The engine is treated strictly as *supporting evidence*. Human/heuristic
explanation is always produced regardless of what the engine says.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import List, Optional

import chess

try:  # python-chess ships the engine module, but guard just in case.
    import chess.engine

    _ENGINE_IMPORT_OK = True
except Exception:  # pragma: no cover - defensive
    _ENGINE_IMPORT_OK = False


@dataclass
class EngineLine:
    """One candidate move from the engine's multipv analysis."""

    rank: int
    san: str
    uci: str
    score_cp: Optional[int]      # centipawns from the side-to-move's view, or None for mate
    mate: Optional[int]          # mate-in-N from side-to-move's view, or None
    pv: List[str]                # principal variation in SAN

    def score_text(self) -> str:
        """Format the score from White's perspective is handled by caller;
        here we present it from the moving side's point of view."""
        if self.mate is not None:
            return f"#{self.mate}" if self.mate > 0 else f"#-{abs(self.mate)}"
        if self.score_cp is None:
            return "?"
        return f"{self.score_cp / 100:+.2f}"


def find_stockfish(explicit: Optional[str] = None) -> Optional[str]:
    """Best-effort discovery of a Stockfish binary.

    Order: explicit path -> ``STOCKFISH_PATH`` env var -> ``stockfish`` on PATH.
    Returns ``None`` if nothing usable is found.
    """
    candidates = [
        explicit,
        os.environ.get("STOCKFISH_PATH"),
        shutil.which("stockfish"),
    ]
    for cand in candidates:
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    # ``shutil.which`` already checks executability; accept it directly.
    if candidates[2]:
        return candidates[2]
    return None


def _score_to_parts(pov_score: "chess.engine.PovScore", color: chess.Color):
    """Return ``(score_cp, mate)`` from ``color``'s point of view."""
    score = pov_score.pov(color)
    if score.is_mate():
        return None, score.mate()
    return score.score(), None


class Engine:
    """A thin, crash-safe wrapper around a UCI engine (Stockfish)."""

    def __init__(self, path: Optional[str] = None):
        self.path = find_stockfish(path)
        self._engine = None
        self.available = False
        self.error: Optional[str] = None

        if not _ENGINE_IMPORT_OK:
            self.error = "python-chess engine support is unavailable."
            return
        if not self.path:
            self.error = "No Stockfish binary found (set a path or STOCKFISH_PATH)."
            return
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self.available = True
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = f"Could not start engine at {self.path}: {exc}"
            self._engine = None
            self.available = False

    # -- analysis ----------------------------------------------------------
    def analyse(self, fen: str, depth: int = 12, multipv: int = 3) -> Optional[List[EngineLine]]:
        """Return up to ``multipv`` candidate lines for ``fen`` (or ``None``)."""
        if not self.available or self._engine is None:
            return None
        try:
            board = chess.Board(fen)
        except Exception:
            return None
        if board.is_game_over():
            return []
        try:
            infos = self._engine.analyse(
                board,
                chess.engine.Limit(depth=depth),
                multipv=max(1, multipv),
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            self.error = f"Engine analysis failed: {exc}"
            return None

        if isinstance(infos, dict):  # multipv=1 may return a single dict
            infos = [infos]

        lines: List[EngineLine] = []
        mover = board.turn
        for idx, info in enumerate(infos, start=1):
            pv_moves = info.get("pv") or []
            if not pv_moves:
                continue
            score_cp, mate = _score_to_parts(info["score"], mover)
            san_pv = self._pv_to_san(board, pv_moves[:6])
            best = pv_moves[0]
            try:
                best_san = board.san(best)
            except Exception:
                best_san = best.uci()
            lines.append(
                EngineLine(
                    rank=idx,
                    san=best_san,
                    uci=best.uci(),
                    score_cp=score_cp,
                    mate=mate,
                    pv=san_pv,
                )
            )
        return lines

    @staticmethod
    def _pv_to_san(board: chess.Board, moves) -> List[str]:
        out: List[str] = []
        tmp = board.copy(stack=False)
        for mv in moves:
            try:
                out.append(tmp.san(mv))
                tmp.push(mv)
            except Exception:
                break
        return out

    def evaluate(self, fen: str, depth: int = 12) -> Optional[EngineLine]:
        """Convenience: the single best line for a position."""
        lines = self.analyse(fen, depth=depth, multipv=1)
        if lines:
            return lines[0]
        return None

    def move_assessment(
        self, fen_before: str, move_uci: str, depth: int = 12, multipv: int = 3
    ):
        """Compare a repertoire move against the engine's preferences.

        Returns a dict with the candidate lines, whether the repertoire move is
        among them and at what rank, and the before/after evaluations (both
        normalised to White's point of view so they are directly comparable).
        Returns ``None`` if the engine is unavailable.
        """
        if not self.available:
            return None
        candidates = self.analyse(fen_before, depth=depth, multipv=multipv)
        if candidates is None:
            return None

        rank = None
        for line in candidates:
            if line.uci == move_uci:
                rank = line.rank
                break

        eval_before = self._white_pov_eval(fen_before, depth)
        eval_after = None
        try:
            board = chess.Board(fen_before)
            board.push(chess.Move.from_uci(move_uci))
            eval_after = self._white_pov_eval(board.fen(), depth)
        except Exception:
            eval_after = None

        return {
            "candidates": candidates,
            "rep_move_rank": rank,
            "rep_move_in_top": rank is not None,
            "eval_before_white": eval_before,
            "eval_after_white": eval_after,
            "depth": depth,
        }

    def _white_pov_eval(self, fen: str, depth: int):
        """Single number from White's perspective: ``(score_cp, mate)``."""
        if not self.available or self._engine is None:
            return None
        try:
            board = chess.Board(fen)
            if board.is_game_over():
                return None
            info = self._engine.analyse(board, chess.engine.Limit(depth=depth))
            return _score_to_parts(info["score"], chess.WHITE)
        except Exception:
            return None

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None
            self.available = False

    def __enter__(self) -> "Engine":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def white_pov_text(parts) -> str:
    """Format a ``(score_cp, mate)`` tuple from White's perspective."""
    if parts is None:
        return "n/a"
    score_cp, mate = parts
    if mate is not None:
        return f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    if score_cp is None:
        return "n/a"
    return f"{score_cp / 100:+.2f}"


__all__ = ["Engine", "EngineLine", "find_stockfish", "white_pov_text"]
