"""Rationale generation for repertoire positions.

Two engines produce the *same* structured ``RationaleCard``:

* :class:`HeuristicExplainer` — always available, no API key required. It reads
  the board with python-chess and combines concrete facts (material, threats,
  loose pieces, checks, pins, forks) with an opening knowledge base picked to
  match the loaded repertoire (see :mod:`openings`): Jobava London for 1.d4
  files, a Scotch-centred book for 1.e4 files, and a neutral principle-based
  book otherwise. This is what makes the tool useful out of the box, since real
  repertoire PGNs are often sparsely annotated.

* :class:`LLMExplainer` — optional. If ``ANTHROPIC_API_KEY`` is set and the
  ``anthropic`` package is installed, it asks Claude to write a concise,
  club-level explanation (with a persona matched to the detected opening). It
  is told to hedge rather than invent when a position's rationale is unclear.
  On any failure it falls back to the heuristic card.

The explanations target a beginner-to-intermediate club player: plans, threats,
piece activity, pawn breaks, king safety and memorable patterns rather than
deep engine theory.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess

from openings import OpeningBook, detect_book
from pgn_parser import RepNode, RepertoireTree

# Default Claude model. Configurable via the JOBAVA_LLM_MODEL environment
# variable or the Streamlit sidebar. Opus 4.8 gives the most reliable
# chess explanations; users may switch to a cheaper model if they prefer.
DEFAULT_LLM_MODEL = os.environ.get("JOBAVA_LLM_MODEL", "claude-opus-4-8")

PIECE_VALUE = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Alternative:
    move: str
    idea: str
    why_not_preferred: str
    tag: str  # tactical / positional / risky / theoretical

    def to_dict(self) -> Dict[str, str]:
        return {
            "move": self.move,
            "idea": self.idea,
            "why_not_preferred": self.why_not_preferred,
            "tag": self.tag,
        }


@dataclass
class RationaleCard:
    move_label: str
    fen: str
    side_to_move: str
    position_summary: str = ""
    why_this_move: str = ""
    opening_logic: str = ""
    logic_label: str = "Opening logic"
    prevents: str = ""
    alternatives: List[Alternative] = field(default_factory=list)
    tactical_checks: str = ""
    plans: List[str] = field(default_factory=list)
    memory_hook: str = ""
    difficulty: int = 3
    mistake_warning: str = ""
    pgn_comment: str = ""
    source: str = "heuristic"  # heuristic / llm / placeholder

    def to_dict(self) -> Dict:
        data = self.__dict__.copy()
        data["alternatives"] = [a.to_dict() for a in self.alternatives]
        return data


# --------------------------------------------------------------------------- #
# Board analysis helpers
# --------------------------------------------------------------------------- #
def _material(board: chess.Board):
    white = sum(
        PIECE_VALUE[p.piece_type]
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black = sum(
        PIECE_VALUE[p.piece_type]
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )
    return white, black


def _developed_minors(board: chess.Board, color: chess.Color) -> int:
    """Count knights/bishops that have left their starting rank."""
    back_rank = 0 if color == chess.WHITE else 7
    count = 0
    for sq, piece in board.piece_map().items():
        if piece.color != color:
            continue
        if piece.piece_type in (chess.KNIGHT, chess.BISHOP):
            if chess.square_rank(sq) != back_rank:
                count += 1
    return count


def _king_status(board: chess.Board, color: chess.Color) -> str:
    king_sq = board.king(color)
    if king_sq is None:
        return "no king (?)"
    file_letter = chess.square_name(king_sq)
    castled_kingside = file_letter in ("g1", "g8")
    castled_queenside = file_letter in ("c1", "c8")
    if castled_kingside:
        return "castled kingside"
    if castled_queenside:
        return "castled queenside"
    if board.has_castling_rights(color):
        return "still in the centre (can still castle)"
    return "in the centre, castling rights lost"


def _pawn_structure_notes(board: chess.Board) -> List[str]:
    notes: List[str] = []
    for color, name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        files: Dict[int, int] = {}
        for sq, piece in board.piece_map().items():
            if piece.color == color and piece.piece_type == chess.PAWN:
                files[chess.square_file(sq)] = files.get(chess.square_file(sq), 0) + 1
        doubled = [chess.FILE_NAMES[f] for f, c in files.items() if c > 1]
        isolated = []
        for f in files:
            if (f - 1) not in files and (f + 1) not in files:
                isolated.append(chess.FILE_NAMES[f])
        if doubled:
            notes.append(f"{name} has doubled pawns on the {', '.join(sorted(doubled))}-file")
        if isolated:
            notes.append(f"{name} has an isolated pawn on the {', '.join(sorted(isolated))}-file")
    return notes


def _center_notes(board: chess.Board) -> str:
    spots = {chess.D4: "d4", chess.E4: "e4", chess.D5: "d5", chess.E5: "e5"}
    occupied = []
    for sq, name in spots.items():
        piece = board.piece_at(sq)
        if piece and piece.piece_type == chess.PAWN:
            who = "White" if piece.color == chess.WHITE else "Black"
            occupied.append(f"{who} pawn on {name}")
    if not occupied:
        return "Neither side has a pawn on the four central squares yet"
    return "Central pawns: " + ", ".join(occupied)


def _attacked_undefended(board: chess.Board, color: chess.Color):
    """Pieces of ``color`` that are loose: attacked and under-defended."""
    loose = []
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        attackers = board.attackers(not color, sq)
        if not attackers:
            continue
        defenders = board.attackers(color, sq)
        if not defenders:
            loose.append((sq, piece, "undefended"))
        else:
            min_att = min(PIECE_VALUE[board.piece_at(a).piece_type] for a in attackers)
            if min_att < PIECE_VALUE[piece.piece_type]:
                loose.append((sq, piece, "attacked by a cheaper piece"))
    return loose


def _pins(board: chess.Board, color: chess.Color):
    pinned = []
    for sq, piece in board.piece_map().items():
        if piece.color != color or piece.piece_type == chess.KING:
            continue
        try:
            if board.is_pinned(color, sq):
                pinned.append((sq, piece))
        except Exception:
            continue
    return pinned


def _forks(board: chess.Board, color: chess.Color):
    """Pieces of ``color`` hitting two or more valuable enemy targets."""
    found = []
    for sq, piece in board.piece_map().items():
        if piece.color != color:
            continue
        targets = []
        for t in board.attacks(sq):
            tp = board.piece_at(t)
            if tp and tp.color != color and (
                PIECE_VALUE[tp.piece_type] >= 3 or tp.piece_type == chess.KING
            ):
                targets.append(t)
        if len(targets) >= 2:
            found.append((sq, piece, targets))
    return found


def _legal_checks(board: chess.Board) -> List[str]:
    out = []
    for mv in board.legal_moves:
        if board.gives_check(mv):
            try:
                out.append(board.san(mv))
            except Exception:
                pass
    return out


def _winning_captures(board: chess.Board) -> List[str]:
    """Captures that win material outright (target loose or higher value)."""
    out = []
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue
        victim = board.piece_at(mv.to_square)
        attacker = board.piece_at(mv.from_square)
        if victim is None or attacker is None:  # en passant etc.
            continue
        defenders = board.attackers(victim.color, mv.to_square)
        gains = PIECE_VALUE[victim.piece_type] - PIECE_VALUE[attacker.piece_type]
        if not defenders or gains > 0:
            try:
                out.append(board.san(mv))
            except Exception:
                pass
    return out


def _mate_in_one(board: chess.Board) -> List[str]:
    out = []
    for mv in board.legal_moves:
        board.push(mv)
        mate = board.is_checkmate()
        board.pop()
        if mate:
            try:
                out.append(board.san(mv))
            except Exception:
                pass
    return out


def _tag_for_move(san: str) -> str:
    if san in ("g4", "h4", "h5", "f3"):
        return "risky / aggressive"
    if "x" in san or "+" in san:
        return "tactical"
    if san in ("O-O", "O-O-O", "0-0", "0-0-0"):
        return "positional"
    if san.startswith(("N", "B", "Q", "R")):
        return "positional"
    return "theoretical"


# --------------------------------------------------------------------------- #
# Heuristic explainer
# --------------------------------------------------------------------------- #
class HeuristicExplainer:
    """Rule-based, always-available explanation generator.

    The opening knowledge base is chosen per-tree via
    :func:`openings.detect_book`, so the same code explains a Jobava London
    file, a Scotch file, or any other repertoire.
    """

    def explain(
        self,
        tree: RepertoireTree,
        node: RepNode,
        engine_info: Optional[dict] = None,
    ) -> RationaleCard:
        book = detect_book(tree)
        board_after = chess.Board(node.fen_after)
        board_before = (
            chess.Board(node.fen_before) if node.fen_before else chess.Board()
        )
        theme = book.classify(node, board_before)

        card = RationaleCard(
            move_label=node.label,
            fen=node.fen_after,
            side_to_move=node.side_to_move,
            pgn_comment=node.comment,
            logic_label=book.logic_label,
            source="heuristic",
        )
        card.position_summary = self._position_summary(board_after, node)
        card.why_this_move = self._why_this_move(board_before, node, engine_info, book, theme)
        card.opening_logic = self._opening_logic(tree, node, book, theme)
        card.prevents = self._prevents(node, board_after, book, theme)
        card.alternatives = self._alternatives(tree, node, board_before, engine_info, book)
        card.tactical_checks = self._tactical_checks(board_after, node)
        card.plans = self._plans(book, theme)
        card.memory_hook = book.hook.get(theme) or (
            f"Remember {node.label} as a natural building move in the {book.name} setup."
        )
        card.difficulty = self._difficulty(node, book, theme)
        card.mistake_warning = book.mistake.get(theme) or (
            "Don't drift: keep developing and castle before starting an attack, and "
            "don't push pawns in front of your own king without a reason."
        )
        return card

    # -- sections ----------------------------------------------------------
    def _position_summary(self, board: chess.Board, node: RepNode) -> str:
        w, b = _material(board)
        bits: List[str] = []
        if w == b:
            bits.append("Material is level.")
        else:
            leader = "White" if w > b else "Black"
            bits.append(f"Material favours {leader} by {abs(w - b)} point(s).")
        bits.append(_center_notes(board) + ".")
        wd, bd = _developed_minors(board, chess.WHITE), _developed_minors(board, chess.BLACK)
        bits.append(
            f"Development: White has {wd} minor piece(s) out, Black {bd}."
        )
        bits.append(
            f"Kings — White is {_king_status(board, chess.WHITE)}; "
            f"Black is {_king_status(board, chess.BLACK)}."
        )
        struct = _pawn_structure_notes(board)
        if struct:
            bits.append("Pawn structure: " + "; ".join(struct) + ".")
        else:
            bits.append("Pawn structure is still healthy and symmetrical-ish.")

        loose_w = _attacked_undefended(board, chess.WHITE)
        loose_b = _attacked_undefended(board, chess.BLACK)
        if loose_w or loose_b:
            who = []
            if loose_w:
                who.append("White: " + ", ".join(
                    f"{PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
                    for sq, p, _ in loose_w[:3]
                ))
            if loose_b:
                who.append("Black: " + ", ".join(
                    f"{PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
                    for sq, p, _ in loose_b[:3]
                ))
            bits.append("Loose/attacked pieces — " + "; ".join(who) + ".")
        bits.append(f"It is {node.side_to_move}'s move.")
        return " ".join(bits)

    def _move_basics(self, board_before: chess.Board, node: RepNode) -> str:
        """A plain-language description of the mechanical effect of the move."""
        try:
            move = chess.Move.from_uci(node.move_uci)
        except Exception:
            return f"{node.move_san} is played."
        piece = board_before.piece_at(move.from_square)
        desc = []
        if board_before.is_castling(move):
            return "castles, tucking the king to safety and connecting the rooks"
        if piece:
            pname = PIECE_NAME[piece.piece_type]
            if board_before.is_capture(move):
                victim = board_before.piece_at(move.to_square)
                vname = PIECE_NAME[victim.piece_type] if victim else "pawn"
                desc.append(f"the {pname} captures the {vname} on {chess.square_name(move.to_square)}")
            elif piece.piece_type == chess.PAWN:
                desc.append(f"advances a pawn to {chess.square_name(move.to_square)}")
            else:
                desc.append(f"develops/repositions the {pname} to {chess.square_name(move.to_square)}")
        board_after = board_before.copy()
        try:
            board_after.push(move)
            if board_after.is_check():
                desc.append("with check")
        except Exception:
            pass
        return ", ".join(desc) if desc else f"{node.move_san} is played"

    def _why_this_move(self, board_before, node, engine_info, book: OpeningBook, theme: str) -> str:
        basics = self._move_basics(board_before, node)
        text = f"{node.label} {basics}. "
        if theme in book.why:
            text += book.why[theme] + " "
        else:
            text += (
                "It is a natural, healthy move that improves a piece or the pawn "
                "structure and keeps the position sound and flexible. "
            )
        if engine_info and engine_info.get("rep_move_in_top"):
            rank = engine_info.get("rep_move_rank")
            text += f"The engine agrees: this is among its top choices (rank {rank}). "
        elif engine_info and not engine_info.get("rep_move_in_top"):
            text += (
                "The engine's first choice differs, but the repertoire move is "
                "chosen for clarity and ease of play, not raw evaluation. "
            )
        return text.strip()

    def _opening_logic(self, tree, node, book: OpeningBook, theme: str) -> str:
        base = book.logic.get(theme, book.default_logic)
        # If this is an opponent move, point out how the repertoire reacts.
        if node.mover == "Black":
            kids = tree.children(node.node_id)
            if kids:
                base += f" Against {node.move_san}, the repertoire continues {kids[0].label}."
        return base

    def _prevents(self, node, board_after, book: OpeningBook, theme: str) -> str:
        if theme in book.prevents:
            return book.prevents[theme]
        # Generic: did the move create a concrete worry for the opponent?
        opp = chess.WHITE if node.mover == "Black" else chess.BLACK
        loose = _attacked_undefended(board_after, opp)
        if loose:
            sq, piece, _ = loose[0]
            return (
                f"It keeps the initiative — the opponent now has to worry about the "
                f"{PIECE_NAME[piece.piece_type]} on {chess.square_name(sq)} rather than "
                "carrying out their own plan."
            )
        return (
            "More a constructive move than a preventive one: it denies the opponent "
            "easy targets and keeps the position free of weaknesses."
        )

    def _alternatives(self, tree, node, board_before, engine_info, book: OpeningBook) -> List[Alternative]:
        alts: List[Alternative] = []
        seen = {node.move_san}

        # 1) Sibling moves listed in the repertoire itself (most relevant).
        for sib in tree.siblings(node.node_id):
            if sib.move_san in seen:
                continue
            seen.add(sib.move_san)
            kids = tree.children(sib.node_id)
            response = f" The repertoire answers {kids[0].label}." if kids else ""
            # Use the knowledge base for a meaningful one-liner when we recognise
            # the sibling move; otherwise fall back to a neutral description.
            sib_board = chess.Board(sib.fen_before) if sib.fen_before else chess.Board()
            sib_theme = book.classify(sib, sib_board)
            core = book.why.get(sib_theme, "Another move available to "
                                + ("Black" if sib.mover == "Black" else "White")
                                + " in this position.")
            idea = core.split(". ")[0].rstrip(".") + "."
            alts.append(
                Alternative(
                    move=sib.move_san,
                    idea=f"{idea}{response}",
                    why_not_preferred=(
                        "Covered by the repertoire as a side line; "
                        f"the main move {node.move_san} is the primary recommendation."
                        if node.is_mainline
                        else "Another branch the repertoire prepares for."
                    ),
                    tag=_tag_for_move(sib.move_san),
                )
            )

        # 2) Engine candidates (real, sound moves) not already listed.
        if engine_info:
            for line in engine_info.get("candidates", []) or []:
                if line.san in seen:
                    continue
                seen.add(line.san)
                alts.append(
                    Alternative(
                        move=line.san,
                        idea=f"Engine suggestion (eval {line.score_text()}).",
                        why_not_preferred=(
                            "Engine-approved, but the repertoire move is easier to "
                            "play and remember for a club player."
                        ),
                        tag="engine",
                    )
                )
                if len(alts) >= 4:
                    break

        # 3) If still empty, offer one conservative natural alternative.
        if not alts:
            nat = self._natural_alternative(board_before, node)
            if nat:
                alts.append(nat)

        return alts[:4]

    def _natural_alternative(self, board_before, node) -> Optional[Alternative]:
        """Suggest castling as a sane alternative, if legal."""
        for mv in board_before.legal_moves:
            if board_before.is_castling(mv):
                try:
                    san = board_before.san(mv)
                except Exception:
                    continue
                if san != node.move_san:
                    return Alternative(
                        move=san,
                        idea="Castle the king into safety before continuing.",
                        why_not_preferred=(
                            "Sensible, but the repertoire move addresses a more "
                            "concrete need first."
                        ),
                        tag="positional",
                    )
        return None

    def _tactical_checks(self, board_after, node) -> str:
        stm = board_after.turn
        stm_name = "White" if stm == chess.WHITE else "Black"
        bits: List[str] = []

        mates = _mate_in_one(board_after)
        if mates:
            bits.append(f"⚠️ Mate in one available for {stm_name}: {', '.join(mates[:2])}.")

        checks = _legal_checks(board_after)
        if checks:
            bits.append(f"Checks for {stm_name}: {', '.join(checks[:4])}.")

        caps = _winning_captures(board_after)
        if caps:
            bits.append(f"Tempting captures for {stm_name}: {', '.join(caps[:4])}.")

        for color, cname in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
            loose = _attacked_undefended(board_after, color)
            if loose:
                items = ", ".join(
                    f"{PIECE_NAME[p.piece_type]} on {chess.square_name(sq)} ({why})"
                    for sq, p, why in loose[:3]
                )
                bits.append(f"Loose {cname} pieces: {items}.")
            pins = _pins(board_after, color)
            if pins:
                items = ", ".join(
                    f"{PIECE_NAME[p.piece_type]} on {chess.square_name(sq)}"
                    for sq, p in pins[:2]
                )
                bits.append(f"Pinned {cname} pieces: {items}.")

        forks = _forks(board_after, stm)
        if forks:
            sq, piece, targets = forks[0]
            tnames = ", ".join(chess.square_name(t) for t in targets[:3])
            bits.append(
                f"Forking idea: {stm_name}'s {PIECE_NAME[piece.piece_type]} on "
                f"{chess.square_name(sq)} hits {tnames}."
            )

        if not bits:
            return (
                "No immediate forcing tactics: no checks, no winning captures and no "
                "loose pieces. The position is calm and about plans, not tricks."
            )
        return " ".join(bits)

    def _plans(self, book: OpeningBook, theme: str) -> List[str]:
        if theme in book.plans and book.plans[theme]:
            return list(book.plans[theme])
        return list(book.default_plans)

    def _difficulty(self, node, book: OpeningBook, theme: str) -> int:
        base = book.difficulty.get(theme, 2)
        # Slightly harder the deeper into a line you are.
        if node.ply >= 8:
            base += 1
        if node.ply >= 12:
            base += 1
        return max(1, min(5, base))


# --------------------------------------------------------------------------- #
# LLM explainer (optional)
# --------------------------------------------------------------------------- #
LLM_SYSTEM_CORE = (
    "You are explaining a repertoire to a beginner-to-intermediate club "
    "player. Be concrete and practical: talk about plans, threats, piece "
    "activity, pawn breaks, king safety and memorable patterns rather than "
    "deep engine theory. Do NOT assume the reader already knows why a "
    "standard theory move is played. Never invent forced lines or "
    "evaluations you are unsure of — if the rationale for a move is unclear, "
    "say what is *likely* rather than pretending certainty. Keep every "
    "section short and useful. Respond with ONLY a single JSON object, no "
    "markdown fences, no commentary."
)


def llm_system_prompt(book: OpeningBook) -> str:
    return f"{book.llm_persona} {LLM_SYSTEM_CORE}"


LLM_SCHEMA_HINT = (
    '{\n'
    '  "position_summary": "material, king safety, centre, development, pawn structure, immediate threats",\n'
    '  "why_this_move": "the strategic and tactical reason for the repertoire move",\n'
    '  "opening_logic": "how it connects to this opening\'s typical themes and plans",\n'
    '  "prevents": "what Black/White was threatening or what plan it reduces",\n'
    '  "alternatives": [{"move": "SAN", "idea": "...", "why_not_preferred": "...", "tag": "tactical|positional|risky|theoretical"}],\n'
    '  "tactical_checks": "checks, captures, threats, pins, forks, loose pieces, mating ideas",\n'
    '  "plans": ["plan 1", "plan 2", "plan 3"],\n'
    '  "memory_hook": "one short sentence to remember the move",\n'
    '  "difficulty": 3,\n'
    '  "mistake_warning": "the most likely beginner/intermediate misunderstanding here"\n'
    '}'
)


def build_llm_prompt(
    tree: RepertoireTree,
    node: RepNode,
    engine_info: Optional[dict] = None,
) -> str:
    """Assemble the per-position user prompt (reused for export and the UI)."""
    book = detect_book(tree)
    line = tree.line_san(node.parent_id) if node.parent_id is not None else ""
    children = tree.children(node.node_id)
    siblings = tree.siblings(node.node_id)

    sib_lines = []
    for sib in siblings:
        kids = tree.children(sib.node_id)
        resp = f" -> repertoire plays {kids[0].label}" if kids else ""
        sib_lines.append(f"{sib.move_san}{resp}")

    engine_text = "Engine not available."
    if engine_info:
        cands = engine_info.get("candidates") or []
        parts = [f"{c.rank}. {c.san} ({c.score_text()})" for c in cands]
        engine_text = "Engine top moves (side to move): " + "; ".join(parts) if parts else "Engine returned no lines."
        if engine_info.get("rep_move_in_top"):
            engine_text += f" | Repertoire move IS in the engine top {len(cands)} (rank {engine_info.get('rep_move_rank')})."
        else:
            engine_text += " | Repertoire move is NOT in the engine's top list."

    opening = tree.headers.get("Event", f"{book.name} repertoire")

    return (
        f"Opening / file: {opening}\n"
        f"Repertoire family: {book.name}\n"
        f"Moves so far: {line or '(starting position)'}\n"
        f"Repertoire move just played: {node.label} (SAN {node.move_san}, "
        f"played by {node.mover})\n"
        f"Side to move now: {node.side_to_move}\n"
        f"FEN before the move: {node.fen_before}\n"
        f"FEN after the move: {node.fen_after}\n"
        f"Existing PGN comment: {node.comment or '(none)'}\n"
        f"Existing NAGs: {', '.join(node.nag_texts) or '(none)'}\n"
        f"Next repertoire move(s) from here: "
        f"{', '.join(c.label for c in children) or '(end of line)'}\n"
        f"Sibling variations at this point: {'; '.join(sib_lines) or '(none)'}\n"
        f"{engine_text}\n\n"
        f"Write the rationale card as JSON with exactly these keys:\n"
        f"{LLM_SCHEMA_HINT}\n"
        f"'difficulty' is an integer 1-5 (how hard the move is to understand and "
        f"remember). Provide 2-4 alternatives. Keep each text field to a few "
        f"sentences at most."
    )


class LLMExplainer:
    """Optional Anthropic-backed explainer. Safe to construct without a key."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.model = model or DEFAULT_LLM_MODEL
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        self.available = False
        self.error: Optional[str] = None

        if not self.api_key:
            self.error = "No ANTHROPIC_API_KEY set."
            return
        try:
            import anthropic  # imported lazily so the app runs without it

            self._client = anthropic.Anthropic(api_key=self.api_key)
            self._anthropic = anthropic
            self.available = True
        except Exception as exc:
            self.error = f"anthropic SDK not available: {exc}"
            self.available = False

    def explain(
        self,
        tree: RepertoireTree,
        node: RepNode,
        engine_info: Optional[dict] = None,
        fallback: Optional[RationaleCard] = None,
    ) -> RationaleCard:
        base = fallback or HeuristicExplainer().explain(tree, node, engine_info)
        if not self.available or self._client is None:
            return base

        prompt = build_llm_prompt(tree, node, engine_info)
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=2048,
                system=llm_system_prompt(detect_book(tree)),
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            data = _extract_json(text)
        except Exception as exc:
            base.mistake_warning = (
                base.mistake_warning
                + f"  (AI explanation unavailable: {exc})"
            ).strip()
            return base

        if not data:
            return base
        return _merge_llm_into_card(base, data, node)


def _extract_json(text: str) -> Optional[dict]:
    """Pull the first JSON object out of an LLM response, tolerantly."""
    text = text.strip()
    if not text:
        return None
    # Strip code fences if present.
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fall back to the substring between the first { and the last }.
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


def _merge_llm_into_card(card: RationaleCard, data: dict, node: RepNode) -> RationaleCard:
    """Overlay LLM JSON onto a heuristic card, keeping heuristic facts as backup."""
    card.source = "llm"
    # Accept the legacy key name from older prompts/caches.
    if "opening_logic" not in data and "jobava_logic" in data:
        data["opening_logic"] = data["jobava_logic"]
    for key in (
        "position_summary",
        "why_this_move",
        "opening_logic",
        "prevents",
        "tactical_checks",
        "memory_hook",
        "mistake_warning",
    ):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            setattr(card, key, val.strip())

    plans = data.get("plans")
    if isinstance(plans, list) and plans:
        card.plans = [str(p) for p in plans if str(p).strip()]

    alts = data.get("alternatives")
    if isinstance(alts, list) and alts:
        parsed = []
        for a in alts:
            if not isinstance(a, dict):
                continue
            parsed.append(
                Alternative(
                    move=str(a.get("move", "?")),
                    idea=str(a.get("idea", "")),
                    why_not_preferred=str(a.get("why_not_preferred", "")),
                    tag=str(a.get("tag", "")),
                )
            )
        if parsed:
            card.alternatives = parsed[:4]

    diff = data.get("difficulty")
    try:
        card.difficulty = max(1, min(5, int(diff)))
    except Exception:
        pass
    return card


def placeholder_card(tree: RepertoireTree, node: RepNode) -> RationaleCard:
    """Bare template used only if even the heuristic path fails."""
    return RationaleCard(
        move_label=node.label,
        fen=node.fen_after,
        side_to_move=node.side_to_move,
        pgn_comment=node.comment,
        position_summary="(Position summary unavailable.)",
        why_this_move="(Explanation template — enable the heuristic or AI explainer.)",
        opening_logic="(Opening logic placeholder.)",
        prevents="(What this move prevents — placeholder.)",
        tactical_checks="(Tactical checks placeholder.)",
        plans=["(Plan placeholder.)"],
        memory_hook="(Memory hook placeholder.)",
        difficulty=3,
        mistake_warning="(Mistake warning placeholder.)",
        source="placeholder",
    )


def explain_position(
    tree: RepertoireTree,
    node: RepNode,
    engine_info: Optional[dict] = None,
    use_llm: bool = False,
    llm: Optional[LLMExplainer] = None,
) -> RationaleCard:
    """High-level dispatcher used by the app and the exporter."""
    if node.is_root:
        book = detect_book(tree)
        return RationaleCard(
            move_label="Starting position",
            fen=node.fen_after,
            side_to_move=node.side_to_move,
            position_summary=book.root_summary,
            why_this_move=book.root_why,
            opening_logic=book.root_logic,
            logic_label=book.logic_label,
            plans=list(book.root_plans),
            memory_hook=book.root_hook,
            difficulty=1,
            source="heuristic",
        )

    try:
        base = HeuristicExplainer().explain(tree, node, engine_info)
    except Exception:
        base = placeholder_card(tree, node)

    if use_llm and llm is not None and llm.available:
        try:
            return llm.explain(tree, node, engine_info, fallback=base)
        except Exception:
            return base
    return base


__all__ = [
    "Alternative",
    "RationaleCard",
    "HeuristicExplainer",
    "LLMExplainer",
    "explain_position",
    "build_llm_prompt",
    "llm_system_prompt",
    "placeholder_card",
    "DEFAULT_LLM_MODEL",
]
