"""Rationale generation for repertoire positions.

Two engines produce the *same* structured ``RationaleCard``:

* :class:`HeuristicExplainer` — always available, no API key required. It reads
  the board with python-chess and combines concrete facts (material, threats,
  loose pieces, checks, pins, forks) with a small Jobava-London knowledge base.
  This is what makes the tool useful out of the box, since real repertoire PGNs
  are often sparsely annotated.

* :class:`LLMExplainer` — optional. If ``ANTHROPIC_API_KEY`` is set and the
  ``anthropic`` package is installed, it asks Claude to write a concise,
  club-level explanation. It is told to hedge rather than invent when a
  position's rationale is unclear. On any failure it falls back to the
  heuristic card.

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
    jobava_logic: str = ""
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


# --------------------------------------------------------------------------- #
# Heuristic explainer
# --------------------------------------------------------------------------- #
class HeuristicExplainer:
    """Rule-based, always-available explanation generator."""

    def explain(
        self,
        tree: RepertoireTree,
        node: RepNode,
        engine_info: Optional[dict] = None,
    ) -> RationaleCard:
        board_after = chess.Board(node.fen_after)
        board_before = (
            chess.Board(node.fen_before) if node.fen_before else chess.Board()
        )

        card = RationaleCard(
            move_label=node.label,
            fen=node.fen_after,
            side_to_move=node.side_to_move,
            pgn_comment=node.comment,
            source="heuristic",
        )
        card.position_summary = self._position_summary(board_after, node)
        card.why_this_move = self._why_this_move(board_before, board_after, node, engine_info)
        card.jobava_logic = self._jobava_logic(tree, node, board_before, board_after)
        card.prevents = self._prevents(tree, node, board_before, board_after)
        card.alternatives = self._alternatives(tree, node, board_before, engine_info)
        card.tactical_checks = self._tactical_checks(board_after, node)
        card.plans = self._plans(tree, node, board_after)
        card.memory_hook = self._memory_hook(node)
        card.difficulty = self._difficulty(tree, node)
        card.mistake_warning = self._mistake_warning(node)
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

    def _why_this_move(self, board_before, board_after, node, engine_info) -> str:
        basics = self._move_basics(board_before, node)
        text = f"{node.label} {basics}. "
        theme = _classify(node, board_before)
        if theme in THEME_WHY:
            text += THEME_WHY[theme] + " "
        else:
            text += (
                "It is a natural, healthy move that improves a piece or the pawn "
                "structure and keeps White's setup flexible. "
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

    def _jobava_logic(self, tree, node, board_before, board_after) -> str:
        theme = _classify(node, board_before)
        if theme in THEME_JOBAVA:
            base = THEME_JOBAVA[theme]
        else:
            base = (
                "This move supports the core Jobava London plan: rapid development "
                "with Nc3 and Bf4, pressure on c7/d5, and the option to break with "
                "e4 or launch the h-pawn depending on where Black puts the king."
            )
        # If this is a Black move, point out how the repertoire intends to react.
        if node.mover == "Black":
            kids = tree.children(node.node_id)
            if kids:
                base += f" Against {node.move_san}, the repertoire continues {kids[0].label}."
        return base

    def _prevents(self, tree, node, board_before, board_after) -> str:
        theme = _classify(node, board_before)
        if theme in THEME_PREVENTS:
            return THEME_PREVENTS[theme]
        # Generic: did the move stop an enemy capture or a strong central push?
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
            "easy targets and keeps White's structure free of weaknesses."
        )

    def _alternatives(self, tree, node, board_before, engine_info) -> List[Alternative]:
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
            sib_theme = _classify(sib, sib_board)
            core = THEME_WHY.get(sib_theme, "Another move available to "
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
        """Suggest castling or developing an undeveloped minor, if legal."""
        color = chess.WHITE if node.mover == "White" else chess.BLACK
        # Prefer castling.
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
        other_name = "Black" if stm == chess.WHITE else "White"
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

    def _plans(self, tree, node, board_after) -> List[str]:
        theme = _classify(node, chess.Board(node.fen_before) if node.fen_before else chess.Board())
        if theme in THEME_PLANS:
            return list(THEME_PLANS[theme])
        # Generic Jobava plans depending on whose move logic we describe (White).
        return [
            "Finish developing: e3, a knight to f3 or e2, the light-squared bishop, then castle.",
            "Look for the e4 break to open the centre once your pieces are ready.",
            "Watch for Nb5 ideas hitting c7 and d6 if Black neglects that square.",
        ]

    def _memory_hook(self, node: RepNode) -> str:
        theme = _classify(node, chess.Board(node.fen_before) if node.fen_before else chess.Board())
        if theme in THEME_HOOK:
            return THEME_HOOK[theme]
        return f"Remember {node.label} as a natural building move in the Jobava setup."

    def _difficulty(self, tree, node) -> int:
        theme = _classify(node, chess.Board(node.fen_before) if node.fen_before else chess.Board())
        base = THEME_DIFFICULTY.get(theme, 2)
        # Slightly harder the deeper into a line you are.
        if node.ply >= 8:
            base += 1
        if node.ply >= 12:
            base += 1
        return max(1, min(5, base))

    def _mistake_warning(self, node: RepNode) -> str:
        theme = _classify(node, chess.Board(node.fen_before) if node.fen_before else chess.Board())
        if theme in THEME_MISTAKE:
            return THEME_MISTAKE[theme]
        return (
            "Don't drift: keep developing and castle before starting an attack, and "
            "don't push pawns in front of your own king without a reason."
        )


# --------------------------------------------------------------------------- #
# Jobava-London knowledge base
# --------------------------------------------------------------------------- #
def _classify(node: RepNode, board_before: chess.Board) -> str:
    """Map a move to a Jobava-London theme key used by the templates below."""
    san = node.move_san or ""
    try:
        move = chess.Move.from_uci(node.move_uci) if node.move_uci else None
    except Exception:
        move = None
    to_name = chess.square_name(move.to_square) if move else ""
    piece = board_before.piece_at(move.from_square) if move else None
    ptype = piece.piece_type if piece else None

    # White's defining moves
    if san == "d4" and node.ply == 1:
        return "d4"
    if san == "Nc3":
        return "Nc3"
    if san == "Bf4":
        return "Bf4"
    if san == "e3":
        return "e3"
    if san in ("Nf3", "Ngf3"):
        return "Nf3"
    if san in ("Ne2", "Nge2"):
        return "Ne2"
    if ptype == chess.KNIGHT and to_name == "b5":
        return "Nb5"
    if san.startswith("Nxd6") or (ptype == chess.KNIGHT and to_name == "d6" and "x" in san):
        return "Nxd6"
    if san.startswith("Bxc7"):
        return "Bxc7"
    if san == "dxe5" or (ptype == chess.PAWN and to_name == "e5" and "x" in san and node.mover == "White"):
        return "dxe5"
    if san.startswith("Qxd5"):
        return "Qxd5"
    if san.startswith("Nxc7"):
        return "Nxc7"
    if san == "f3":
        return "f3"
    if san == "g4":
        return "g4"
    if san in ("h4", "h5"):
        return "h_storm"
    if san == "Qd2":
        return "Qd2"
    if san in ("O-O-O", "0-0-0"):
        return "OOO"
    if san in ("O-O", "0-0"):
        return "OO"
    if san == "a3":
        return "a3"
    if san == "Bd3":
        return "Bd3"
    if san == "e4" and node.mover == "White":
        return "e4"
    if san in ("c4",) and node.mover == "White":
        return "c4"

    # Black's typical tries
    if node.mover == "Black":
        if san in ("Bf5", "Bg4"):
            return "blk_bishop_out"
        if san in ("Bb4",):
            return "blk_Bb4"
        if san in ("Bd6",):
            return "blk_Bd6"
        if san == "e5":
            return "blk_e5"
        if san in ("g6",):
            return "blk_g6"
        if san in ("c5",):
            return "blk_c5"
        if san in ("Ne4",):
            return "blk_Ne4"
        if san in ("a6",):
            return "blk_a6"
    return "generic"


THEME_WHY: Dict[str, str] = {
    "d4": "It grabs the centre and opens lines for the queen and dark-squared bishop, the standard first step of a queen's-pawn opening.",
    "Nc3": "This is the move that defines the Jobava London. The knight eyes d5, e4 and b5, and (unlike the slow c3 London) keeps White ready for a quick e4 break.",
    "Bf4": "The bishop comes out before e3, so it is active outside the pawn chain. From f4 it pressures c7 and supports the whole queenside plan.",
    "e3": "A small but important move: it makes luft for the f1-bishop, supports d4, and keeps everything solid before White commits to a plan.",
    "Nf3": "Natural development that controls e5 and prepares to castle. White finishes the setup before looking for breaks.",
    "Ne2": "A flexible square: the knight avoids any ...Bb4 pin, keeps the f3-pawn or g4 thrust available, and can later reroute to g3 or f4.",
    "Nb5": "The thematic jump. The knight hits c7 (forking ideas with the f4-bishop) and d6, often forcing Black to make a concession or allow Nxc7/Nxd6.",
    "Nxd6": "Trading off Black's good bishop (or winning the bishop pair) and damaging Black's structure when the pawn recaptures.",
    "Bxc7": "A concrete pawn grab made possible because the b5-knight and f4-bishop both eye c7 — punishing Black for leaving the square loose.",
    "dxe5": "Opening the centre at the right moment so White's better-developed pieces (especially Bf4) get to work.",
    "Qxd5": "Grabbing the centre pawn with tempo; it works because of the tactical follow-up against c7.",
    "Nxc7": "The point of the whole line: a knight fork on c7 hitting the rook and exploiting the loose queenside.",
    "f3": "Restrains Black's ...Ne4 jump and prepares either e4 in the centre or a g4 pawn storm against a bishop on f5/g6.",
    "g4": "An aggressive space-gaining thrust. It kicks a Black bishop on f5/g6 and gains kingside space, often as a prelude to h4-h5.",
    "h_storm": "A direct pawn-storm move. With Black committed on the kingside, White rolls the h-pawn to pry open lines in front of the enemy king.",
    "Qd2": "Connects White's plan: it eyes Bh6 to trade Black's fianchetto bishop and prepares queenside castling for a pawn-storm race.",
    "OOO": "Castling queenside commits to opposite-side castling, after which both sides race their pawns at each other's king — and White is usually faster here.",
    "OO": "Castling short for a calmer, safer game when the position does not call for a pawn storm.",
    "a3": "A useful prophylactic move stopping ...Bb4 (pinning the c3-knight) and ...Nb4 ideas before they become annoying.",
    "Bd3": "The bishop aims at the h7-square and supports a future e4 break — a classic attacking diagonal.",
    "e4": "The key Jobava break. Backed by Nc3 and Bf4, White strikes in the centre to open lines while ahead in development.",
    "c4": "A space-gaining central move that can transpose toward a more classical queen's-pawn structure.",
    "blk_bishop_out": "Black develops the light-squared bishop actively before playing ...e6, a common and sensible idea.",
    "blk_Bb4": "Black pins the c3-knight to add pressure and slow White's e4 break.",
    "blk_Bd6": "Black challenges the f4-bishop and fights for the key dark squares.",
    "blk_e5": "Black strikes in the centre, the most principled response, but it opens lines while behind in development.",
    "blk_g6": "Black heads for a King's-Indian-style fianchetto, inviting White's h4-h5 plan.",
    "blk_c5": "Black hits the d4-pawn to open the position and create queenside play.",
    "blk_Ne4": "Black jumps into e4 to trade pieces and ease the cramped position.",
    "blk_a6": "Black spends a move to prevent Nb5, accepting a slightly slower setup.",
}

THEME_JOBAVA: Dict[str, str] = {
    "d4": "Every Jobava London starts here: claim the centre, then develop with Nc3 and Bf4 rather than the slow c3 London.",
    "Nc3": "Nc3 + Bf4 is the signature. The knight on c3 (not pinned by ...Bb4 yet) keeps e4 in the air and enables the Nb5 jab at c7/d6 — the whole point of the system.",
    "Bf4": "Bf4 before e3 is the London bishop done right: active, outside the pawn chain, and aimed at c7. Combined with Nc3 it sets up Nb5 and e4 ideas.",
    "e3": "A quieter, more positional moment: White solidifies and prepares to complete development. The aggression (Nb5, e4, h4) comes a little later.",
    "Nf3": "Standard development toward a safe, flexible Jobava setup; White keeps both the e4 break and short castling on the table.",
    "Ne2": "Choosing Ne2 over Nf3 keeps the f-pawn free for f3/g4 and sidesteps pins — a more aggressive, kingside-storming flavour of the system.",
    "Nb5": "Nb5 is the tactical heartbeat of the Jobava London: it attacks c7 (with the f4-bishop) and d6, forcing concessions. Often it wins the bishop pair or a pawn.",
    "Nxd6": "Eliminating Black's dark-squared bishop hands White the bishop pair and leaves Black with structural damage after the pawn recapture.",
    "Bxc7": "This is the reward for the Nc3/Bf4/Nb5 setup: when Black is careless, c7 simply falls, because two White pieces converge on it.",
    "dxe5": "Opening the centre when ahead in development is a core idea — the f4-bishop and c3-knight spring to life.",
    "Qxd5": "A concrete tactical sequence in the 3...Nc6 4.Nb5 e5 line; it relies on the c7-fork to justify grabbing the pawn.",
    "Nxc7": "The knight fork on c7 is the tactical payoff of Nb5 — hitting the rook on a8 and exploiting the loose queenside.",
    "f3": "f3 supports the g4-h4 expansion and an eventual e4, the aggressive 'pawn-storm' interpretation of the Jobava London, especially against ...Bf5 setups.",
    "g4": "g4 is part of White's kingside pawn storm. Against ...Bf5/...Bg6 it gains space and chases the bishop, frequently followed by h4-h5.",
    "h_storm": "h4-h5 is the attacking soul of the aggressive Jobava: roll the rook's pawn at Black's king, especially after ...g6 or opposite-side castling.",
    "Qd2": "Qd2 is the prelude to O-O-O and Bh6: White castles long and storms the kingside while keeping the centre closed.",
    "OOO": "Queenside castling turns the game into a pawn-storm race. With the Bf4/Nc3 setup, White's attack usually arrives first.",
    "OO": "When the position is calmer, short castling is perfectly fine — the Jobava is flexible, not always an all-out attack.",
    "a3": "Small but typical: removing the ...Bb4 pin keeps the c3-knight free for e4 and Nb5 ideas.",
    "Bd3": "Bd3 lines up the bishop on the b1-h7 diagonal toward h7 and supports e4 — a standard attacking deployment.",
    "e4": "The e4 break is what separates the Jobava from the dull London: with Nc3 and Bf4 already developed, White opens the centre to attack.",
    "c4": "A more classical, space-based handling that can transpose to mainstream queen's-pawn structures.",
    "blk_bishop_out": "Black's ...Bf5/...Bg4 invites exactly White's f3 + g4 plan — note how the repertoire chases the bishop and grabs kingside space.",
    "blk_Bb4": "...Bb4 pins the knight, so the repertoire often meets it with Ne2 and a3, untangling and keeping e4 alive.",
    "blk_Bd6": "When Black offers a trade with ...Bd6, White keeps the tension or develops naturally with e3 and Nf3.",
    "blk_e5": "...e5 is the critical central try; the repertoire answers concretely (dxe5 and tactics on c7/d5) because Black is behind in development.",
    "blk_g6": "Against a fianchetto, the repertoire leans into h4-h5 and Qd2/Bh6 — Black's king home becomes a target.",
    "blk_c5": "Against ...c5 White keeps the centre solid and continues developing; the d4-tension usually favours the better-developed side.",
    "blk_Ne4": "...Ne4 invites a3 and piece play; White is happy to keep more pieces on with a space edge.",
    "blk_a6": "...a6 stops Nb5 but costs time, so White just builds with e3, Nf3 and a comfortable game.",
}

THEME_PREVENTS: Dict[str, str] = {
    "Nc3": "It keeps e4 and Nb5 available, denying Black an easy, comfortable equalising setup.",
    "Bf4": "Developing the bishop now stops it getting locked behind an e3-pawn (the classic 'bad London bishop' problem).",
    "a3": "It prevents ...Bb4 pinning the c3-knight and ...Nb4 hitting d3/c2.",
    "f3": "It takes the e4-square away from a Black knight (...Ne4) and supports White's own central/kingside expansion.",
    "Nb5": "It punishes Black for leaving c7 and d6 loose and prevents a comfortable ...Bd6 or ...e5 setup.",
    "g4": "It denies the f5/g6 squares to Black's bishop and gains space before Black can complete a solid setup.",
    "Ne2": "Developing to e2 sidesteps the ...Bb4 pin entirely and keeps f3/g4 available.",
    "blk_e5": "By striking early Black tries to free the position before White's pieces dominate — the repertoire reacts before that happens.",
    "Qd2": "It prepares to trade Black's defensive fianchetto bishop with Bh6 before Black can use it.",
}

THEME_PLANS: Dict[str, List[str]] = {
    "d4": [
        "Develop quickly with Nc3 and Bf4 (the Jobava move order).",
        "Keep the e4 break in mind once pieces are out.",
        "Decide later between a kingside storm (h4-h5) and quiet short castling.",
    ],
    "Nc3": [
        "Bring the bishop to f4 next, before playing e3.",
        "Keep Nb5 in reserve to hit c7/d6 if Black is careless.",
        "Prepare e4 to blow open the centre while ahead in development.",
    ],
    "Bf4": [
        "Play e3 to free the f1-bishop, then develop the knight (f3 or e2) and castle.",
        "Eye Nb5 — if Black allows it, the knight jumps in with tempo.",
        "Against ...Bf5/...g6, switch to the f3 + g4 + h4 expansion.",
    ],
    "e3": [
        "Complete development: knight to f3 or e2, bishop to d3/e2, then castle.",
        "Choose your plan based on Black: pawn storm vs a fianchetto, calm play otherwise.",
        "Keep the e4 break ready as your central lever.",
    ],
    "Nb5": [
        "If Black defends c7, consider Nxd6 to grab the bishop pair, or retreat with a gain.",
        "Combine the knight with Bf4 and the queen to keep hitting c7.",
        "Open the position with dxe5 / e4 while your pieces are the more active.",
    ],
    "f3": [
        "Expand with g4 to chase the bishop and grab kingside space.",
        "Follow up with h4-h5 to open lines against the king.",
        "Keep e4 available as the central break once the kingside is rolling.",
    ],
    "g4": [
        "Continue with h4-h5 to pry open the h-file.",
        "Castle queenside (Qd2, O-O-O) so your king is safe during the storm.",
        "Target the chased bishop and the weakened light squares.",
    ],
    "Qd2": [
        "Castle queenside next and start the kingside pawn storm.",
        "Consider Bh6 to trade Black's fianchettoed bishop.",
        "Push h4-h5 to open lines toward the Black king.",
    ],
    "e4": [
        "Open the centre and use your lead in development.",
        "Bring rooks to the central files (d- and e-files).",
        "Look for tactics against Black's slightly exposed pieces.",
    ],
}

THEME_HOOK: Dict[str, str] = {
    "d4": "1.d4 then Nc3+Bf4 — that's the Jobava, not the slow c3 London.",
    "Nc3": "Knight to c3 first: it's the move that makes the London bite.",
    "Bf4": "Bishop OUT to f4 before e3 — never bury the London bishop.",
    "e3": "e3 = open the door for the f1-bishop, then castle.",
    "Nf3": "Develop, castle, then break — Nf3 keeps it simple.",
    "Ne2": "Ne2 dodges ...Bb4 and keeps g4 in the holster.",
    "Nb5": "See c7 loose? Nb5! — the Jobava's signature jab.",
    "Nxd6": "Nxd6 = win the bishop pair and wreck Black's pawns.",
    "Bxc7": "Two pieces on c7 means the pawn just drops — take it.",
    "dxe5": "Open the centre when you're the developed side.",
    "Qxd5": "Qxd5 works only because c7 will fall next — remember the combo.",
    "Nxc7": "Nxc7 forks the rook — the payoff of the Nb5 jump.",
    "f3": "f3 says 'no ...Ne4' and 'g4 next'.",
    "g4": "g4 kicks the bishop and screams 'pawn storm!'",
    "h_storm": "h4-h5: roll the rook's pawn at the enemy king.",
    "Qd2": "Qd2 → O-O-O → storm. Memorise the trio.",
    "OOO": "Castle long, then race pawns — you're usually faster.",
    "a3": "a3 = no annoying ...Bb4 pin.",
    "Bd3": "Bd3 aims at h7 and backs up e4.",
    "e4": "e4 is the Jobava punch — the London that actually attacks.",
}

THEME_DIFFICULTY: Dict[str, int] = {
    "d4": 1,
    "Nc3": 1,
    "Bf4": 1,
    "e3": 1,
    "Nf3": 1,
    "OO": 1,
    "Ne2": 3,
    "Nb5": 3,
    "Nxd6": 3,
    "Bxc7": 3,
    "dxe5": 3,
    "Qxd5": 4,
    "Nxc7": 4,
    "f3": 3,
    "g4": 3,
    "h_storm": 3,
    "Qd2": 3,
    "OOO": 3,
    "a3": 2,
    "Bd3": 2,
    "e4": 3,
}

THEME_MISTAKE: Dict[str, str] = {
    "Nc3": "Don't reflexively block the c-pawn worries — in the Jobava, Nc3 is correct precisely because you want e4 and Nb5, not c2-c4.",
    "Bf4": "Don't play e3 before Bf4, or the bishop gets locked in. Bishop first!",
    "Nb5": "Don't play Nb5 just for show — check that c7/d6 are genuinely loose; otherwise the knight gets chased by ...a6 with tempo.",
    "Bxc7": "Before grabbing c7, make sure the bishop isn't trapped afterwards (watch for ...Rc8 or ...b6 ideas).",
    "Qxd5": "This pawn grab is only safe because of the c7 tactic — don't try it without that concrete follow-up.",
    "g4": "Don't launch g4/h4 with your own king still in the centre — make sure you can castle queenside or your king is safe first.",
    "h_storm": "A pawn storm needs your king tucked away on the other wing; storming with a king in the centre often backfires.",
    "OOO": "Count the attack before castling into a storm — in an opposite-castling race, tempo is everything.",
    "e4": "Don't rush e4 before you're developed; premature in the centre, it just opens lines for the opponent.",
    "blk_e5": "As White, meet ...e5 concretely (dxe5 and the c7/d5 tactics) — passive play hands Black free equality.",
    "f3": "Remember f3 slightly weakens the king's diagonal; only commit to it when you intend g4/e4, not as a default.",
}


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
# LLM explainer (optional)
# --------------------------------------------------------------------------- #
LLM_SYSTEM_PROMPT = (
    "You are a friendly, practical chess coach who specialises in the Jobava "
    "London System (1.d4 followed by Nc3 and Bf4). You are explaining a "
    "repertoire to a beginner-to-intermediate club player. Be concrete and "
    "practical: talk about plans, threats, piece activity, pawn breaks, king "
    "safety and memorable patterns rather than deep engine theory. Do NOT "
    "assume the reader already knows why a standard Jobava move is played. "
    "Never invent forced lines or evaluations you are unsure of — if the "
    "rationale for a move is unclear, say what is *likely* rather than "
    "pretending certainty. Keep every section short and useful. Respond with "
    "ONLY a single JSON object, no markdown fences, no commentary."
)

LLM_SCHEMA_HINT = (
    '{\n'
    '  "position_summary": "material, king safety, centre, development, pawn structure, immediate threats",\n'
    '  "why_this_move": "the strategic and tactical reason for the repertoire move",\n'
    '  "jobava_logic": "how it connects to typical Jobava London themes",\n'
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

    opening = tree.headers.get("Event", "Jobava London repertoire")

    return (
        f"Opening / file: {opening}\n"
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
                system=LLM_SYSTEM_PROMPT,
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
    for key in (
        "position_summary",
        "why_this_move",
        "jobava_logic",
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
        jobava_logic="(Jobava London logic placeholder.)",
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
        card = RationaleCard(
            move_label="Starting position",
            fen=node.fen_after,
            side_to_move=node.side_to_move,
            position_summary="The initial position. White is about to begin the Jobava London.",
            why_this_move="Select a move in the tree to see its rationale.",
            jobava_logic="The plan: 1.d4, then Nc3 and Bf4 — active, attacking London chess.",
            plans=["Play 1.d4 and aim for the Nc3 + Bf4 setup."],
            memory_hook="d4, Nc3, Bf4 — the Jobava skeleton.",
            difficulty=1,
            source="heuristic",
        )
        return card

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
    "placeholder_card",
    "DEFAULT_LLM_MODEL",
]
