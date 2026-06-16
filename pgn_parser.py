"""PGN parsing for the Jobava London rationale explorer.

This module turns a PGN file (a repertoire, not a single game) into a
*repertoire tree*: every move is a node, mainlines and side variations are
preserved, PGN comments and NAGs are kept, and positions are deduplicated by
FEN so transpositions can be recognised.

The parser is deliberately defensive: a malformed game should never crash the
whole load. Anything that cannot be parsed is skipped and reported in
``RepertoireTree.errors`` instead.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import chess
import chess.pgn


# Human-readable text for the most common NAGs (Numeric Annotation Glyphs).
# We only map the ones a club player is likely to meet in a repertoire file.
NAG_TEXT: Dict[int, str] = {
    1: "good move (!)",
    2: "mistake (?)",
    3: "brilliant move (!!)",
    4: "blunder (??)",
    5: "interesting move (!?)",
    6: "dubious move (?!)",
    7: "forced move",
    10: "equal position (=)",
    11: "equal chances",
    13: "unclear position (∞)",
    14: "White is slightly better (⩲)",
    15: "Black is slightly better (⩱)",
    16: "White is clearly better (±)",
    17: "Black is clearly better (∓)",
    18: "White is winning (+−)",
    19: "Black is winning (−+)",
    22: "White is in zugzwang",
    23: "Black is in zugzwang",
    36: "White has the initiative",
    40: "White has the attack",
    132: "counterplay",
    146: "novelty",
}


def nag_to_text(nag: int) -> str:
    """Return a readable label for a NAG code (falls back to ``$<n>``)."""
    return NAG_TEXT.get(nag, f"${nag}")


@dataclass
class RepNode:
    """A single move (or the synthetic root) in the repertoire tree."""

    node_id: int
    parent_id: Optional[int]
    move_uci: Optional[str]          # None only for the root
    move_san: Optional[str]          # None only for the root
    move_number: int                 # full-move number of the move (0 for root)
    mover: Optional[str]             # "White" / "Black" (who made the move)
    side_to_move: str                # "White" / "Black" to move in fen_after
    fen_before: Optional[str]
    fen_after: str
    comment: str = ""
    nags: List[int] = field(default_factory=list)
    is_mainline: bool = True
    depth: int = 0                   # variation nesting depth (0 = main line)
    ply: int = 0                     # half-moves from the start (0 for root)
    children: List[int] = field(default_factory=list)
    san_path: List[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        """Compact move label such as ``3.Bf4`` or ``3...Nc6`` (``start`` for root)."""
        if self.move_san is None:
            return "start"
        if self.mover == "White":
            return f"{self.move_number}.{self.move_san}"
        return f"{self.move_number}...{self.move_san}"

    @property
    def nag_texts(self) -> List[str]:
        return [nag_to_text(n) for n in self.nags]

    @property
    def is_root(self) -> bool:
        return self.move_san is None


class RepertoireTree:
    """An indexed, deduplicated view over one or more parsed PGN games."""

    def __init__(self) -> None:
        self.headers: Dict[str, str] = {}
        self.nodes: Dict[int, RepNode] = {}
        self.root_id: int = 0
        self.by_fen: Dict[str, List[int]] = {}
        self.game_count: int = 0
        self.errors: List[str] = []

    # -- basic accessors ---------------------------------------------------
    @property
    def root(self) -> RepNode:
        return self.nodes[self.root_id]

    def get(self, node_id: int) -> RepNode:
        return self.nodes[node_id]

    def children(self, node_id: int) -> List[RepNode]:
        return [self.nodes[c] for c in self.nodes[node_id].children]

    def parent(self, node_id: int) -> Optional[RepNode]:
        pid = self.nodes[node_id].parent_id
        return self.nodes[pid] if pid is not None else None

    def siblings(self, node_id: int) -> List[RepNode]:
        """Other moves available at the same position (same parent)."""
        node = self.nodes[node_id]
        if node.parent_id is None:
            return []
        return [
            self.nodes[c]
            for c in self.nodes[node.parent_id].children
            if c != node_id
        ]

    def line_to(self, node_id: int) -> List[RepNode]:
        """The list of moves (excluding the root) from the start to ``node_id``."""
        chain: List[RepNode] = []
        cur: Optional[int] = node_id
        while cur is not None:
            node = self.nodes[cur]
            if not node.is_root:
                chain.append(node)
            cur = node.parent_id
        chain.reverse()
        return chain

    def line_san(self, node_id: int) -> str:
        """Readable move sequence, e.g. ``1.d4 d5 2.Nc3 Nf6``."""
        parts: List[str] = []
        for n in self.line_to(node_id):
            if n.mover == "White":
                parts.append(f"{n.move_number}.{n.move_san}")
            else:
                parts.append(n.move_san)
        return " ".join(parts)

    def leaves(self) -> List[int]:
        return [nid for nid, n in self.nodes.items() if not n.children]

    def transpositions(self, node_id: int) -> List[int]:
        """Other nodes that reach the exact same position (same FEN)."""
        fen = self.nodes[node_id].fen_after
        return [nid for nid in self.by_fen.get(fen, []) if nid != node_id]

    def unique_position_count(self) -> int:
        return len(self.by_fen)

    def move_count(self) -> int:
        # All nodes except the synthetic root represent a move.
        return len(self.nodes) - 1

    def iter_dfs(self) -> List[int]:
        """Depth-first ordering of node ids (mainline-first at each branch)."""
        order: List[int] = []
        stack = [self.root_id]
        while stack:
            nid = stack.pop()
            order.append(nid)
            # push children reversed so the first child is processed first
            stack.extend(reversed(self.nodes[nid].children))
        return order


def _fen_key(fen: str) -> str:
    """Key a position by board + side + castling + en-passant (ignore clocks)."""
    return " ".join(fen.split(" ")[:4])


class _Builder:
    """Internal helper that walks python-chess game trees into a RepertoireTree."""

    def __init__(self) -> None:
        self.tree = RepertoireTree()
        self._next_id = 0
        # Map (parent_id, move_uci) -> child node_id so identical move-sequences
        # from different games merge instead of duplicating.
        self._child_index: Dict[tuple, int] = {}

    def _new_id(self) -> int:
        nid = self._next_id
        self._next_id += 1
        return nid

    def _register_fen(self, node: RepNode) -> None:
        key = _fen_key(node.fen_after)
        self.tree.by_fen.setdefault(key, []).append(node.node_id)

    def create_root(self) -> None:
        board = chess.Board()
        root = RepNode(
            node_id=self._new_id(),
            parent_id=None,
            move_uci=None,
            move_san=None,
            move_number=0,
            mover=None,
            side_to_move="White",
            fen_before=None,
            fen_after=board.fen(),
            is_mainline=True,
            depth=0,
            ply=0,
        )
        self.tree.nodes[root.node_id] = root
        self.tree.root_id = root.node_id

    def walk(self, game_node, parent_id: int, depth: int, parent_is_main: bool) -> None:
        """Recursively attach the variations of ``game_node`` under ``parent_id``."""
        for index, child in enumerate(game_node.variations):
            board = game_node.board()  # position before the move
            move = child.move
            try:
                san = board.san(move)
            except Exception:
                san = move.uci() if move is not None else "?"

            move_uci = move.uci() if move is not None else "?"
            is_main = parent_is_main and index == 0
            key = (parent_id, move_uci)

            existing = self._child_index.get(key)
            if existing is not None:
                # Same move already present from another game/line: merge.
                node = self.tree.nodes[existing]
                self._merge_into(node, child)
                node_id = existing
            else:
                after_board = child.board()
                node = RepNode(
                    node_id=self._new_id(),
                    parent_id=parent_id,
                    move_uci=move_uci,
                    move_san=san,
                    move_number=board.fullmove_number,
                    mover="White" if board.turn == chess.WHITE else "Black",
                    side_to_move="White" if after_board.turn == chess.WHITE else "Black",
                    fen_before=board.fen(),
                    fen_after=after_board.fen(),
                    comment=(child.comment or "").strip(),
                    nags=sorted(child.nags),
                    is_mainline=is_main,
                    depth=depth if is_main else max(depth, 1),
                    ply=self.tree.nodes[parent_id].ply + 1,
                    san_path=self.tree.nodes[parent_id].san_path + [san],
                )
                self.tree.nodes[node_id := node.node_id] = node
                self.tree.nodes[parent_id].children.append(node_id)
                self._child_index[key] = node_id
                self._register_fen(node)

            # Recurse. A child keeps mainline status only if this move was mainline.
            self.walk(child, node_id, depth if is_main else depth + 1, is_main)

    def _merge_into(self, node: RepNode, game_child) -> None:
        """Fold a duplicate move's comment / NAGs into the existing node."""
        comment = (game_child.comment or "").strip()
        if comment and comment not in node.comment:
            node.comment = (node.comment + "  " + comment).strip()
        for nag in game_child.nags:
            if nag not in node.nags:
                node.nags.append(nag)
        node.nags.sort()


def load_pgn_text(text: str) -> RepertoireTree:
    """Parse PGN text (one or more games) into a :class:`RepertoireTree`.

    Malformed games are skipped and recorded in ``tree.errors`` rather than
    raising, so a single bad game never takes down the whole repertoire.
    """
    builder = _Builder()
    builder.create_root()
    tree = builder.tree

    stream = io.StringIO(text)
    game_index = 0
    while True:
        try:
            game = chess.pgn.read_game(stream)
        except Exception as exc:  # pragma: no cover - extremely malformed input
            tree.errors.append(f"Could not read game #{game_index + 1}: {exc}")
            break
        if game is None:
            break
        game_index += 1

        # Record header info from the first game.
        if not tree.headers:
            tree.headers = {k: v for k, v in game.headers.items()}

        # Surface python-chess's own per-game parse errors but keep going.
        for err in getattr(game, "errors", []) or []:
            tree.errors.append(f"Game #{game_index}: {err}")

        try:
            builder.walk(game, tree.root_id, depth=0, parent_is_main=(game_index == 1))
        except Exception as exc:  # pragma: no cover - defensive
            tree.errors.append(f"Game #{game_index} could not be fully parsed: {exc}")

    tree.game_count = game_index
    if game_index == 0:
        tree.errors.append("No valid games were found in the PGN input.")
    elif tree.move_count() == 0:
        tree.errors.append(
            "The PGN was read but contained no moves — is this a valid PGN file?"
        )
    return tree


def load_pgn_file(path: str) -> RepertoireTree:
    """Load a PGN file from disk (UTF-8, tolerant of decoding issues)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return load_pgn_text(handle.read())


__all__ = [
    "RepNode",
    "RepertoireTree",
    "load_pgn_text",
    "load_pgn_file",
    "nag_to_text",
    "NAG_TEXT",
]
