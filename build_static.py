"""Build the static (Vercel-deployable) version of the rationale explorer.

Everything the Streamlit app computes on the fly is pre-generated here into
plain JSON + Markdown under ``public/``:

* ``public/data/<name>.json`` — the repertoire tree, one rationale card per
  node (heuristic explainer), train-mode distractors, and transpositions.
* ``public/guides/<name>_study_guide.md`` — the printable study guide.

``public/index.html`` (hand-written, committed) renders it all client-side:
board, tree, rationale cards and a multiple-choice Train mode. No server, no
Python at runtime — exactly what static hosts like Vercel can serve.

Run after changing a PGN or the explainer:

    python build_static.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import chess

import export as export_mod
from explainer import explain_position
from openings import detect_book
from pgn_parser import RepertoireTree, load_pgn_file

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"

REPERTOIRES = {
    "jobava": ROOT / "sample_data" / "jobava_london_repertoire.pgn",
    "scotch": ROOT / "sample_data" / "scotch_repertoire.pgn",
}

TITLES = {
    "jobava": "Jobava London (1.d4)",
    "scotch": "Scotch Game (1.e4)",
}


def _distractors(tree: RepertoireTree, node_id: int, count: int = 3) -> list:
    """Plausible-but-wrong answer choices for a train question at ``node_id``.

    Deterministic (seeded by node id) so rebuilds don't churn the JSON.
    """
    node = tree.nodes[node_id]
    try:
        board = chess.Board(node.fen_after)
    except Exception:
        return []
    child_sans = {c.move_san for c in tree.children(node_id)}
    legal = []
    for mv in board.legal_moves:
        try:
            san = board.san(mv)
        except Exception:
            continue
        if san not in child_sans:
            legal.append(san)
    rng = random.Random(node_id * 7919 + 13)
    rng.shuffle(legal)
    # Prefer piece moves / captures over random pawn nudges for plausibility.
    legal.sort(key=lambda s: (0 if (s[0] in "NBQRKO" or "x" in s) else 1))
    picked, seen = [], set()
    for san in legal:
        if san in seen:
            continue
        seen.add(san)
        picked.append(san)
        if len(picked) >= count:
            break
    return picked


def build_repertoire_json(name: str, path: Path) -> dict:
    tree = load_pgn_file(str(path))
    book = detect_book(tree)

    nodes = {}
    cards = {}
    distractors = {}

    for nid, node in tree.nodes.items():
        nodes[str(nid)] = {
            "id": nid,
            "parent": node.parent_id,
            "children": list(node.children),
            "san": node.move_san,
            "label": node.label if not node.is_root else "Start",
            "uci": node.move_uci,
            "fen": node.fen_after,
            "fenBefore": node.fen_before,
            "mover": node.mover,
            "stm": node.side_to_move,
            "main": node.is_mainline,
            "ply": node.ply,
            "depth": node.depth,
            "comment": node.comment,
            "nags": node.nag_texts,
            "line": tree.line_san(nid) if not node.is_root else "",
            "trans": [tree.line_san(t) for t in tree.transpositions(nid)[:4]],
        }
        card = explain_position(tree, node)
        cards[str(nid)] = {
            "summary": card.position_summary,
            "why": card.why_this_move,
            "logic": card.opening_logic,
            "logicLabel": card.logic_label,
            "prevents": card.prevents,
            "alts": [a.to_dict() for a in card.alternatives],
            "tactics": card.tactical_checks,
            "plans": card.plans,
            "hook": card.memory_hook,
            "difficulty": card.difficulty,
            "mistake": card.mistake_warning,
        }
        # Train questions live on nodes where White is to move and the
        # repertoire has at least one answer.
        if node.side_to_move == "White" and node.children:
            distractors[str(nid)] = _distractors(tree, nid)

    return {
        "name": name,
        "title": TITLES.get(name, name),
        "family": book.key,
        "logicLabel": book.logic_label,
        "stats": {
            "moves": tree.move_count(),
            "unique": tree.unique_position_count(),
            "lines": len(tree.leaves()),
        },
        "root": tree.root_id,
        "nodes": nodes,
        "cards": cards,
        "distractors": distractors,
    }


def build_guide(name: str, path: Path) -> str:
    tree = load_pgn_file(str(path))
    studies = export_mod.generate_study(tree)
    return export_mod.to_markdown(tree, studies)


def main() -> None:
    (PUBLIC / "data").mkdir(parents=True, exist_ok=True)
    (PUBLIC / "guides").mkdir(parents=True, exist_ok=True)

    manifest = []
    for name, path in REPERTOIRES.items():
        print(f"Building {name} …")
        data = build_repertoire_json(name, path)
        out = PUBLIC / "data" / f"{name}.json"
        out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"  {out.relative_to(ROOT)}  ({out.stat().st_size // 1024} KB, "
              f"{data['stats']['moves']} moves)")

        guide = build_guide(name, path)
        gout = PUBLIC / "guides" / f"{name}_study_guide.md"
        gout.write_text(guide, encoding="utf-8")
        print(f"  {gout.relative_to(ROOT)}  ({gout.stat().st_size // 1024} KB)")

        manifest.append({"name": name, "title": data["title"], "stats": data["stats"]})

    (PUBLIC / "data" / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Done. Deploy the public/ directory (vercel.json already points at it).")


if __name__ == "__main__":
    main()
