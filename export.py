"""Study-guide export: turn the repertoire + rationale cards into Markdown / HTML.

The exporter walks the repertoire tree, deduplicates positions by FEN (so a
transposition is explained once), and emits:

* an opening title and summary,
* the full repertoire tree as an indented outline (with PGN comments / NAGs),
* one rationale card per unique position,
* FEN strings everywhere, plus inline SVG board diagrams in the HTML output.

It reuses :func:`explainer.explain_position`, so Markdown/HTML stay in sync with
what the app shows on screen.
"""

from __future__ import annotations

import html
from typing import Callable, Dict, List, Optional, Tuple

import chess
import chess.svg

from engine import Engine, white_pov_text
from explainer import LLMExplainer, RationaleCard, explain_position
from pgn_parser import RepNode, RepertoireTree


def generate_study(
    tree: RepertoireTree,
    engine: Optional[Engine] = None,
    depth: int = 12,
    use_llm: bool = False,
    llm: Optional[LLMExplainer] = None,
    progress: Optional[Callable[[float, str], None]] = None,
) -> List[Tuple[RepNode, RationaleCard, Optional[dict]]]:
    """Build (node, card, engine_info) triples for each unique position."""
    order = [nid for nid in tree.iter_dfs() if not tree.nodes[nid].is_root]
    seen_fen = set()
    results: List[Tuple[RepNode, RationaleCard, Optional[dict]]] = []
    total = len(order)

    for i, nid in enumerate(order):
        node = tree.nodes[nid]
        fen_key = " ".join(node.fen_after.split(" ")[:4])
        if fen_key in seen_fen:
            continue
        seen_fen.add(fen_key)

        engine_info = None
        if engine is not None and engine.available and node.fen_before:
            engine_info = engine.move_assessment(
                node.fen_before, node.move_uci, depth=depth
            )

        card = explain_position(tree, node, engine_info, use_llm=use_llm, llm=llm)
        results.append((node, card, engine_info))

        if progress:
            progress((i + 1) / max(1, total), node.label)

    return results


# --------------------------------------------------------------------------- #
# Tree outline
# --------------------------------------------------------------------------- #
def _outline_lines(tree: RepertoireTree) -> List[Tuple[int, RepNode]]:
    """Return (indent, node) pairs in DFS order, excluding the root."""
    lines: List[Tuple[int, RepNode]] = []

    def walk(nid: int) -> None:
        node = tree.nodes[nid]
        if not node.is_root:
            lines.append((node.depth, node))
        for child in node.children:
            walk(child)

    walk(tree.root_id)
    return lines


def tree_outline_markdown(tree: RepertoireTree) -> str:
    out: List[str] = []
    for indent, node in _outline_lines(tree):
        prefix = "  " * indent + "- "
        tag = "" if node.is_mainline else " _(variation)_"
        comment = f" — {node.comment}" if node.comment else ""
        nags = f" [{', '.join(node.nag_texts)}]" if node.nags else ""
        out.append(f"{prefix}**{node.label}**{tag}{nags}{comment}")
    return "\n".join(out)


def tree_outline_html(tree: RepertoireTree) -> str:
    out: List[str] = ['<ul class="tree">']
    for indent, node in _outline_lines(tree):
        cls = "main" if node.is_mainline else "var"
        comment = (
            f' <span class="cmt">— {html.escape(node.comment)}</span>'
            if node.comment
            else ""
        )
        nags = (
            f' <span class="nag">[{html.escape(", ".join(node.nag_texts))}]</span>'
            if node.nags
            else ""
        )
        out.append(
            f'<li style="margin-left:{indent * 18}px" class="{cls}">'
            f'<span class="mv">{html.escape(node.label)}</span>{nags}{comment}</li>'
        )
    out.append("</ul>")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# Card rendering
# --------------------------------------------------------------------------- #
def _engine_summary(engine_info: Optional[dict]) -> Optional[str]:
    if not engine_info:
        return None
    parts = []
    before = white_pov_text(engine_info.get("eval_before_white"))
    after = white_pov_text(engine_info.get("eval_after_white"))
    parts.append(f"Eval before: {before} (White) → after: {after} (White).")
    cands = engine_info.get("candidates") or []
    if cands:
        listed = "; ".join(f"{c.rank}. {c.san} ({c.score_text()})" for c in cands)
        parts.append(f"Engine top moves: {listed}.")
    if engine_info.get("rep_move_in_top"):
        parts.append(f"Repertoire move is in the engine's top list (rank {engine_info.get('rep_move_rank')}).")
    else:
        parts.append("Repertoire move is not the engine's top pick, but is chosen for practical repertoire reasons.")
    return " ".join(parts)


def card_markdown(node: RepNode, card: RationaleCard, engine_info: Optional[dict]) -> str:
    out: List[str] = []
    out.append(f"### {card.move_label}")
    out.append("")
    out.append(f"- **Line:** `{node.san_path and ' '.join(node.san_path) or card.move_label}`")
    out.append(f"- **Side to move:** {card.side_to_move}")
    out.append(f"- **FEN:** `{card.fen}`")
    out.append(f"- **Difficulty:** {'★' * card.difficulty}{'☆' * (5 - card.difficulty)} ({card.difficulty}/5)")
    out.append(f"- **Explanation source:** {card.source}")
    if card.pgn_comment:
        out.append(f"- **PGN annotation:** {card.pgn_comment}")
    out.append("")
    out.append(f"**Position summary.** {card.position_summary}")
    out.append("")
    out.append(f"**Why this move.** {card.why_this_move}")
    out.append("")
    out.append(f"**Jobava London logic.** {card.jobava_logic}")
    out.append("")
    out.append(f"**What it prevents.** {card.prevents}")
    out.append("")
    if card.alternatives:
        out.append("**Candidate alternatives.**")
        for alt in card.alternatives:
            out.append(f"- `{alt.move}` _({alt.tag})_ — {alt.idea} {alt.why_not_preferred}".rstrip())
        out.append("")
    out.append(f"**Tactical checks.** {card.tactical_checks}")
    out.append("")
    if card.plans:
        out.append("**Plans after this move.**")
        for p in card.plans:
            out.append(f"- {p}")
        out.append("")
    out.append(f"**Memory hook.** {card.memory_hook}")
    out.append("")
    out.append(f"**Mistake warning.** {card.mistake_warning}")
    eng = _engine_summary(engine_info)
    if eng:
        out.append("")
        out.append(f"**Engine note.** {eng}")
    out.append("")
    return "\n".join(out)


def to_markdown(
    tree: RepertoireTree,
    studies: List[Tuple[RepNode, RationaleCard, Optional[dict]]],
) -> str:
    title = tree.headers.get("Event", "Jobava London Repertoire")
    out: List[str] = []
    out.append(f"# {title} — Study Guide")
    out.append("")
    out.append(
        f"_Generated by the Jobava London Rationale Explorer. "
        f"{tree.move_count()} moves, {tree.unique_position_count()} unique positions, "
        f"{len(tree.leaves())} lines._"
    )
    out.append("")
    if tree.headers:
        out.append("**PGN headers:** " + ", ".join(f"{k}={v}" for k, v in tree.headers.items() if v))
        out.append("")
    out.append("## Repertoire tree")
    out.append("")
    out.append(tree_outline_markdown(tree))
    out.append("")
    out.append("## Move-by-move rationale")
    out.append("")
    out.append(f"_{len(studies)} unique positions explained (transpositions shown once)._")
    out.append("")
    for node, card, engine_info in studies:
        out.append(card_markdown(node, card, engine_info))
        out.append("---")
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
HTML_STYLE = """
body { font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
       max-width: 1000px; margin: 0 auto; padding: 24px; color: #1c1c1c; line-height: 1.5; }
h1 { border-bottom: 3px solid #5b8def; padding-bottom: 8px; }
h3 { margin-top: 4px; color: #2b4a8b; }
.card { border: 1px solid #ddd; border-radius: 10px; padding: 16px 20px; margin: 18px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.card .layout { display: flex; gap: 20px; flex-wrap: wrap; }
.card .board { flex: 0 0 280px; }
.card .body { flex: 1 1 380px; min-width: 320px; }
.meta { font-size: 0.85em; color: #555; margin-bottom: 8px; }
.meta code { background: #f3f3f3; padding: 1px 5px; border-radius: 4px; }
.section { margin: 8px 0; }
.section b { color: #2b4a8b; }
.tree { list-style: none; padding-left: 0; font-family: 'SF Mono', Consolas, monospace; font-size: 0.92em; }
.tree li { padding: 1px 0; }
.tree .main .mv { font-weight: 700; }
.tree .var .mv { color: #777; }
.cmt { color: #2a7d2a; font-style: italic; }
.nag { color: #b8860b; }
.alt { background: #f7f9ff; border-left: 3px solid #5b8def; padding: 4px 10px; margin: 4px 0; border-radius: 4px; }
.tag { font-size: 0.8em; color: #fff; background: #5b8def; border-radius: 10px; padding: 1px 8px; }
.engine { background: #fff7ec; border-left: 3px solid #e0a106; padding: 6px 10px; border-radius: 4px; }
.diff { color: #d08700; letter-spacing: 2px; }
"""


def _card_html(node: RepNode, card: RationaleCard, engine_info: Optional[dict]) -> str:
    try:
        board = chess.Board(card.fen)
        last = chess.Move.from_uci(node.move_uci) if node.move_uci else None
        svg = chess.svg.board(board, size=280, lastmove=last)
    except Exception:
        svg = ""

    alts = ""
    if card.alternatives:
        rows = "".join(
            f'<div class="alt"><span class="tag">{html.escape(a.tag)}</span> '
            f"<code>{html.escape(a.move)}</code> — {html.escape(a.idea)} "
            f"{html.escape(a.why_not_preferred)}</div>"
            for a in card.alternatives
        )
        alts = f'<div class="section"><b>Candidate alternatives</b>{rows}</div>'

    plans = ""
    if card.plans:
        items = "".join(f"<li>{html.escape(p)}</li>" for p in card.plans)
        plans = f'<div class="section"><b>Plans after this move</b><ul>{items}</ul></div>'

    eng = _engine_summary(engine_info)
    eng_html = f'<div class="engine"><b>Engine note.</b> {html.escape(eng)}</div>' if eng else ""

    comment = (
        f'<div class="section"><b>PGN annotation.</b> {html.escape(card.pgn_comment)}</div>'
        if card.pgn_comment
        else ""
    )

    diff_stars = "★" * card.difficulty + "☆" * (5 - card.difficulty)
    line = " ".join(node.san_path) if node.san_path else card.move_label

    return f"""
<div class="card">
  <h3>{html.escape(card.move_label)}</h3>
  <div class="meta">
    <code>{html.escape(line)}</code> &nbsp;|&nbsp; {html.escape(card.side_to_move)} to move
    &nbsp;|&nbsp; difficulty <span class="diff">{diff_stars}</span> ({card.difficulty}/5)
    &nbsp;|&nbsp; source: {html.escape(card.source)}<br>
    FEN: <code>{html.escape(card.fen)}</code>
  </div>
  <div class="layout">
    <div class="board">{svg}</div>
    <div class="body">
      {comment}
      <div class="section"><b>Position summary.</b> {html.escape(card.position_summary)}</div>
      <div class="section"><b>Why this move.</b> {html.escape(card.why_this_move)}</div>
      <div class="section"><b>Jobava London logic.</b> {html.escape(card.jobava_logic)}</div>
      <div class="section"><b>What it prevents.</b> {html.escape(card.prevents)}</div>
      {alts}
      <div class="section"><b>Tactical checks.</b> {html.escape(card.tactical_checks)}</div>
      {plans}
      <div class="section"><b>Memory hook.</b> {html.escape(card.memory_hook)}</div>
      <div class="section"><b>Mistake warning.</b> {html.escape(card.mistake_warning)}</div>
      {eng_html}
    </div>
  </div>
</div>
"""


def to_html(
    tree: RepertoireTree,
    studies: List[Tuple[RepNode, RationaleCard, Optional[dict]]],
) -> str:
    title = tree.headers.get("Event", "Jobava London Repertoire")
    cards = "\n".join(_card_html(n, c, e) for n, c, e in studies)
    header_bits = ", ".join(
        f"{html.escape(k)}={html.escape(v)}" for k, v in tree.headers.items() if v
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — Study Guide</title>
<style>{HTML_STYLE}</style>
</head>
<body>
<h1>{html.escape(title)} — Study Guide</h1>
<p><em>Generated by the Jobava London Rationale Explorer.
{tree.move_count()} moves, {tree.unique_position_count()} unique positions,
{len(tree.leaves())} lines.</em></p>
<p><strong>PGN headers:</strong> {header_bits or '(none)'}</p>
<h2>Repertoire tree</h2>
{tree_outline_html(tree)}
<h2>Move-by-move rationale</h2>
<p><em>{len(studies)} unique positions explained (transpositions shown once).</em></p>
{cards}
</body>
</html>
"""


__all__ = [
    "generate_study",
    "to_markdown",
    "to_html",
    "tree_outline_markdown",
    "tree_outline_html",
    "card_markdown",
]
