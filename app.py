"""Jobava London — Opening Rationale Explorer (Streamlit app).

Run with:  streamlit run app.py

Two modes:

* **Explore** — left: the repertoire as a collapsible tree (with search);
  main: board with next-move arrows + tabs (Rationale / Engine / PGN context /
  Export) and start/back/next navigation.
* **Train** — guess-the-move drilling: the app plays the opponent's repertoire
  replies at random and asks you for your side's move, tracking score & streak.

Everything degrades gracefully: no Stockfish and no API key still gives you the
parsed tree, PGN comments, board diagrams and the built-in heuristic rationale.
"""

from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import List, Optional

import chess
import chess.svg
import streamlit as st
import streamlit.components.v1 as components

from engine import Engine, white_pov_text
from explainer import (
    DEFAULT_LLM_MODEL,
    HeuristicExplainer,
    LLMExplainer,
    explain_position,
)
from export import generate_study, to_html, to_markdown
from pgn_parser import RepertoireTree, load_pgn_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"

MODE_EXPLORE = "📖 Explore"
MODE_TRAIN = "🎯 Train"

ARROW_MAIN = "#15781BB0"   # green — mainline continuation
ARROW_ALT = "#E69F00A0"    # amber — other repertoire branches

st.set_page_config(
    page_title="Jobava London Rationale Explorer",
    page_icon="♞",
    layout="wide",
)


def _show_svg(svg: str) -> None:
    """Render inline SVG, preferring the modern ``st.html`` over the deprecated
    ``components.html`` (which uses an iframe and is being removed)."""
    wrapped = f"<div style='display:flex;justify-content:center'>{svg}</div>"
    html_fn = getattr(st, "html", None)
    if callable(html_fn):
        html_fn(wrapped)
    else:  # Streamlit < 1.33 fallback
        components.html(wrapped, height=420)


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("tree", None)
    ss.setdefault("source_name", "")
    ss.setdefault("selected_id", None)
    ss.setdefault("expanded", set())
    ss.setdefault("engine", None)
    ss.setdefault("engine_path", os.environ.get("STOCKFISH_PATH", ""))
    ss.setdefault("engine_cache", {})
    ss.setdefault("llm_card_cache", {})
    ss.setdefault("depth", 12)
    ss.setdefault("use_engine", False)
    ss.setdefault("use_llm", False)
    ss.setdefault("llm_model", DEFAULT_LLM_MODEL)
    ss.setdefault("api_key_override", "")
    ss.setdefault("board_size", 380)
    ss.setdefault("show_arrows", True)
    ss.setdefault("train", None)


def load_tree(text: str, name: str) -> None:
    tree = load_pgn_text(text)
    st.session_state.tree = tree
    st.session_state.source_name = name
    st.session_state.selected_id = tree.root_id
    # Expand the root and the mainline so something useful shows immediately.
    expanded = {tree.root_id}
    cur = tree.root_id
    while tree.nodes[cur].children:
        expanded.add(cur)
        cur = tree.nodes[cur].children[0]
    st.session_state.expanded = expanded
    st.session_state.engine_cache = {}
    st.session_state.llm_card_cache = {}
    st.session_state.train = None


def _select_node(tree: RepertoireTree, node_id: int) -> None:
    """Select a node and make sure all its ancestors are expanded."""
    ss = st.session_state
    ss.selected_id = node_id
    cur: Optional[int] = node_id
    while cur is not None:
        ss.expanded.add(cur)
        cur = tree.nodes[cur].parent_id


def get_engine() -> Optional[Engine]:
    """Connect (or reuse) the engine for the current path; None if unavailable."""
    ss = st.session_state
    path = ss.engine_path or None
    eng: Optional[Engine] = ss.engine
    # Reuse the already-connected engine unless the requested path changed.
    if eng is not None and eng.available and (path is None or eng.path == path):
        return eng
    # (Re)connect.
    if eng is not None:
        eng.close()
    eng = Engine(path)
    ss.engine = eng
    return eng


def get_llm() -> LLMExplainer:
    key = st.session_state.api_key_override or os.environ.get("ANTHROPIC_API_KEY")
    return LLMExplainer(api_key=key, model=st.session_state.llm_model)


def engine_info_for(node) -> Optional[dict]:
    """Cached engine assessment for a node (or None)."""
    ss = st.session_state
    if not ss.use_engine:
        return None
    eng = get_engine()
    if eng is None or not eng.available or not node.fen_before:
        return None
    cache_key = (node.fen_before, node.move_uci, ss.depth)
    if cache_key in ss.engine_cache:
        return ss.engine_cache[cache_key]
    info = eng.move_assessment(node.fen_before, node.move_uci, depth=ss.depth)
    ss.engine_cache[cache_key] = info
    return info


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    ss = st.session_state
    st.sidebar.title("♞ Jobava London")
    st.sidebar.caption("Opening Rationale Explorer")

    with st.sidebar.expander("1 · Load a PGN", expanded=ss.tree is None):
        uploaded = st.file_uploader("Upload a PGN file", type=["pgn"])
        if uploaded is not None:
            if st.button("Load uploaded PGN", width="stretch"):
                text = uploaded.read().decode("utf-8", errors="replace")
                load_tree(text, uploaded.name)
                st.rerun()

        # Bundled samples.
        samples = sorted(SAMPLE_DIR.glob("*.pgn")) if SAMPLE_DIR.exists() else []
        if samples:
            names = [s.name for s in samples]
            choice = st.selectbox("…or pick a bundled sample", names)
            if st.button("Load sample", width="stretch"):
                text = (SAMPLE_DIR / choice).read_text(encoding="utf-8", errors="replace")
                load_tree(text, choice)
                st.rerun()

        path = st.text_input("…or a path on disk", value="")
        if path and st.button("Load from path", width="stretch"):
            try:
                text = Path(path).read_text(encoding="utf-8", errors="replace")
                load_tree(text, Path(path).name)
                st.rerun()
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

    with st.sidebar.expander("2 · Stockfish engine", expanded=False):
        ss.engine_path = st.text_input(
            "Stockfish binary path",
            value=ss.engine_path,
            help="Leave blank to auto-detect 'stockfish' on PATH, $STOCKFISH_PATH, "
                 "or common install locations (/usr/games, Homebrew).",
        )
        ss.depth = st.slider("Analysis depth", 6, 22, ss.depth)
        if st.button("Connect / test engine", width="stretch"):
            if ss.engine is not None:
                ss.engine.close()
                ss.engine = None
            eng = get_engine()
            if eng and eng.available:
                st.success(f"Engine ready: {eng.path}")
            else:
                st.warning(eng.error if eng else "Could not connect.")
        ss.use_engine = st.checkbox("Use engine for analysis", value=ss.use_engine)
        eng = ss.engine
        if eng and eng.available:
            st.caption(f"🟢 Connected: {os.path.basename(eng.path or '')}")
        else:
            st.caption("⚪ Engine not connected (optional).")

    with st.sidebar.expander("3 · AI explanations", expanded=False):
        has_env_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        st.caption(
            "🟢 ANTHROPIC_API_KEY detected." if has_env_key
            else "⚪ No ANTHROPIC_API_KEY in environment (optional)."
        )
        ss.api_key_override = st.text_input(
            "API key override (optional)",
            value=ss.api_key_override,
            type="password",
            help="Stored only in this session. Leave blank to use ANTHROPIC_API_KEY.",
        )
        ss.llm_model = st.text_input("Claude model", value=ss.llm_model)
        ss.use_llm = st.checkbox("Use AI explanations", value=ss.use_llm)
        if ss.use_llm:
            llm = get_llm()
            if llm.available:
                st.caption(f"🟢 AI ready ({ss.llm_model}).")
            else:
                st.caption(f"🔴 AI unavailable: {llm.error}")

    with st.sidebar.expander("4 · Display", expanded=False):
        ss.board_size = st.slider("Board size", 300, 560, ss.board_size, step=20)
        ss.show_arrows = st.checkbox(
            "Show next-move arrows",
            value=ss.show_arrows,
            help="Green = mainline continuation, amber = other repertoire branches.",
        )

    if ss.tree is not None:
        t = ss.tree
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**Loaded:** `{ss.source_name}`")
        st.sidebar.caption(
            f"{t.move_count()} moves · {t.unique_position_count()} unique positions · "
            f"{len(t.leaves())} lines · {t.game_count} game(s)"
        )
        if t.errors:
            with st.sidebar.expander(f"⚠️ {len(t.errors)} parser note(s)"):
                for e in t.errors[:20]:
                    st.caption(e)


# --------------------------------------------------------------------------- #
# Tree view (Explore mode, left column)
# --------------------------------------------------------------------------- #
def _visible_rows(tree: RepertoireTree):
    rows = []

    def rec(nid: int, depth: int) -> None:
        rows.append((nid, depth))
        if nid in st.session_state.expanded:
            for c in tree.nodes[nid].children:
                rec(c, depth + 1)

    rec(tree.root_id, 0)
    return rows


def render_tree(tree: RepertoireTree) -> None:
    ss = st.session_state
    st.subheader("Repertoire tree")

    query = st.text_input(
        "🔎 Find a move",
        placeholder="e.g. Nb5, h4, O-O-O",
        label_visibility="collapsed",
    )
    if query:
        q = query.strip().lower()
        matches = [
            nid for nid, n in tree.nodes.items()
            if not n.is_root and q in (n.move_san or "").lower()
        ][:10]
        if matches:
            st.caption(f"{len(matches)} match(es) — click to jump:")
            for m in matches:
                line = tree.line_san(m)
                label = line if len(line) <= 58 else "… " + line[-56:]
                if st.button(label, key=f"srch_{m}", width="stretch"):
                    _select_node(tree, m)
                    st.rerun()
        else:
            st.caption("No matching move found.")
        st.markdown("---")

    c1, c2, c3 = st.columns(3)
    if c1.button("Expand all", width="stretch"):
        ss.expanded = set(tree.nodes.keys())
        st.rerun()
    if c2.button("Collapse", width="stretch"):
        ss.expanded = {tree.root_id}
        st.rerun()
    if c3.button("Mainline", width="stretch"):
        expanded = {tree.root_id}
        cur = tree.root_id
        while tree.nodes[cur].children:
            expanded.add(cur)
            cur = tree.nodes[cur].children[0]
        ss.expanded = expanded
        st.rerun()

    # Quick jump to a full line (leaf).
    leaves = tree.leaves()
    if leaves:
        leaf_labels = {tree.line_san(nid): nid for nid in leaves}
        pick = st.selectbox(
            "Jump to a full line",
            ["—"] + sorted(leaf_labels.keys()),
            index=0,
        )
        if pick != "—":
            target = leaf_labels[pick]
            if st.button("Go to this line", width="stretch"):
                _select_node(tree, target)
                st.rerun()

    st.markdown(
        "<div style='max-height:60vh; overflow-y:auto; padding-right:6px'>",
        unsafe_allow_html=True,
    )
    for nid, depth in _visible_rows(tree):
        node = tree.nodes[nid]
        has_kids = bool(node.children)
        is_exp = nid in ss.expanded
        col_toggle, col_label = st.columns([0.13, 0.87])

        with col_toggle:
            if has_kids:
                sym = "▾" if is_exp else "▸"
                if st.button(sym, key=f"tog_{nid}"):
                    if is_exp:
                        ss.expanded.discard(nid)
                    else:
                        ss.expanded.add(nid)
                    st.rerun()
            else:
                st.write("")

        with col_label:
            indent = " " * max(0, depth - 1)
            if node.is_root:
                text = "Starting position"
            else:
                marker = "" if node.is_mainline else "° "
                flag = " 💬" if node.comment else ""
                nag = " ‼" if node.nags else ""
                text = f"{indent}{marker}{node.label}{nag}{flag}"
            kind = "primary" if nid == ss.selected_id else "secondary"
            if st.button(text, key=f"sel_{nid}", width="stretch", type=kind):
                ss.selected_id = nid
                ss.expanded.add(nid)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("° = side variation · 💬 = has PGN comment · ‼ = has NAG")


# --------------------------------------------------------------------------- #
# Board + navigation (Explore mode)
# --------------------------------------------------------------------------- #
def _board_svg(
    fen: str,
    last_uci: Optional[str],
    size: int,
    flipped: bool,
    arrows: Optional[List[chess.svg.Arrow]] = None,
) -> Optional[str]:
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    last = None
    if last_uci:
        try:
            last = chess.Move.from_uci(last_uci)
        except Exception:
            last = None
    check_sq = board.king(board.turn) if board.is_check() else None
    return chess.svg.board(
        board,
        size=size,
        lastmove=last,
        check=check_sq,
        flipped=flipped,
        coordinates=True,
        arrows=arrows or [],
    )


def render_nav(tree: RepertoireTree, node) -> None:
    """Start / back / next buttons plus a switcher for sibling branches."""
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2.4])
    if c1.button("⏮ Start", width="stretch", disabled=node.is_root):
        _select_node(tree, tree.root_id)
        st.rerun()
    if c2.button("◀ Back", width="stretch", disabled=node.parent_id is None):
        _select_node(tree, node.parent_id)
        st.rerun()
    kids = tree.children(node.node_id)
    if c3.button("Next ▶", width="stretch", disabled=not kids):
        _select_node(tree, kids[0].node_id)
        st.rerun()
    with c4:
        siblings = tree.siblings(node.node_id)
        if siblings and not node.is_root:
            options = [node.label] + [s.label for s in siblings]
            pills_fn = getattr(st, "pills", None)
            if callable(pills_fn):
                choice = pills_fn(
                    "Branches here",
                    options,
                    default=node.label,
                    key=f"sib_{node.node_id}",
                    label_visibility="collapsed",
                )
            else:  # very old Streamlit fallback
                choice = st.selectbox(
                    "Branches here", options, index=0, key=f"sib_{node.node_id}"
                )
            if choice and choice != node.label:
                target = next(s for s in siblings if s.label == choice)
                _select_node(tree, target.node_id)
                st.rerun()


def render_board(tree: RepertoireTree, node) -> None:
    ss = st.session_state
    arrows: List[chess.svg.Arrow] = []
    kids = tree.children(node.node_id)
    if ss.show_arrows and kids:
        for i, child in enumerate(kids):
            try:
                mv = chess.Move.from_uci(child.move_uci)
            except Exception:
                continue
            color = ARROW_MAIN if i == 0 else ARROW_ALT
            arrows.append(chess.svg.Arrow(mv.from_square, mv.to_square, color=color))

    svg = _board_svg(
        node.fen_after,
        None if node.is_root else node.move_uci,
        ss.board_size,
        st.session_state.get("flip", False),
        arrows,
    )
    if svg is None:
        st.error("Could not render this position.")
        return
    _show_svg(svg)
    cols = st.columns([1, 1, 2])
    cols[0].checkbox("Flip board", key="flip")
    cols[1].caption(f"{node.side_to_move} to move")
    if ss.show_arrows and kids:
        legend = "🟩 next mainline move" + (" · 🟧 other branches" if len(kids) > 1 else "")
        cols[2].caption(legend)


def _difficulty_stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


# --------------------------------------------------------------------------- #
# Tabs (Explore mode)
# --------------------------------------------------------------------------- #
def render_rationale_tab(tree: RepertoireTree, node, engine_info) -> None:
    ss = st.session_state

    use_llm = ss.use_llm
    llm = get_llm() if use_llm else None
    cache_key = (node.fen_after, ss.llm_model)
    cached = ss.llm_card_cache.get(cache_key) if use_llm else None

    if use_llm and llm and llm.available and cached is None:
        st.info("AI explanations are enabled. Generate one for this position:")
        if st.button("✨ Generate AI explanation", type="primary"):
            with st.spinner("Asking Claude…"):
                card = explain_position(tree, node, engine_info, use_llm=True, llm=llm)
                ss.llm_card_cache[cache_key] = card
            st.rerun()

    card = cached or explain_position(tree, node, engine_info, use_llm=False)

    badge = {"llm": "🤖 AI", "heuristic": "🧮 heuristic", "placeholder": "📄 template"}.get(
        card.source, card.source
    )
    head, regen = st.columns([4, 1])
    head.markdown(
        f"#### {card.move_label} &nbsp; "
        f"<span style='font-size:0.7em;color:#888'>({badge}) · "
        f"difficulty {_difficulty_stars(card.difficulty)} {card.difficulty}/5</span>",
        unsafe_allow_html=True,
    )
    if cached is not None and llm and llm.available:
        if regen.button("🔁 Regenerate", help="Ask Claude again for this position."):
            with st.spinner("Asking Claude…"):
                fresh = explain_position(tree, node, engine_info, use_llm=True, llm=llm)
                ss.llm_card_cache[cache_key] = fresh
            st.rerun()

    if card.pgn_comment:
        st.success(f"**PGN annotation:** {card.pgn_comment}")

    st.markdown(f"**📋 Position summary**\n\n{card.position_summary}")
    st.markdown(f"**🎯 Why this move**\n\n{card.why_this_move}")
    st.markdown(f"**♘ Jobava London logic**\n\n{card.jobava_logic}")
    st.markdown(f"**🛡️ What it prevents**\n\n{card.prevents}")

    if card.alternatives:
        st.markdown("**🔀 Candidate alternatives**")
        for alt in card.alternatives:
            st.markdown(
                f"- **`{alt.move}`** _({alt.tag})_ — {alt.idea} {alt.why_not_preferred}".rstrip()
            )

    st.markdown(f"**⚡ Tactical checks**\n\n{card.tactical_checks}")

    if card.plans:
        st.markdown("**🗺️ Plans after this move**")
        for p in card.plans:
            st.markdown(f"- {p}")

    st.markdown(f"**🧠 Memory hook**\n\n> {card.memory_hook}")
    st.warning(f"**Mistake warning:** {card.mistake_warning}")


def _eval_bar_html(parts) -> Optional[str]:
    """A horizontal white/black bar from a White-POV ``(cp, mate)`` tuple."""
    if parts is None:
        return None
    cp, mate = parts
    if mate is not None:
        pct = 100.0 if mate > 0 else 0.0
    elif cp is None:
        return None
    else:
        pct = 100.0 / (1.0 + math.exp(-0.004 * float(cp)))
    return (
        "<div style='height:16px;border:1px solid #888;border-radius:8px;"
        "overflow:hidden;background:#3a3a3a;max-width:420px'>"
        f"<div style='height:100%;width:{pct:.1f}%;background:#f2f2f2'></div></div>"
        f"<div style='font-size:0.8em;color:#777'>White's share of winning chances: {pct:.0f}%</div>"
    )


def render_engine_tab(tree: RepertoireTree, node, engine_info) -> None:
    ss = st.session_state
    if not ss.use_engine:
        st.info("Engine analysis is off. Enable it in the sidebar (Stockfish required).")
        return
    eng = get_engine()
    if eng is None or not eng.available:
        st.warning(
            (eng.error if eng else "No engine.")
            + "  Provide a Stockfish path in the sidebar and click *Connect / test engine*."
        )
        return
    if node.is_root:
        st.caption("Select a move to analyse.")
        return
    if engine_info is None:
        engine_info = engine_info_for(node)
    if engine_info is None:
        st.warning("Engine did not return analysis for this position.")
        return

    before = white_pov_text(engine_info.get("eval_before_white"))
    after = white_pov_text(engine_info.get("eval_after_white"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Eval before (White)", before)
    c2.metric("Eval after (White)", after)
    c3.metric("Depth", engine_info.get("depth"))

    bar = _eval_bar_html(
        engine_info.get("eval_after_white") or engine_info.get("eval_before_white")
    )
    if bar and callable(getattr(st, "html", None)):
        st.html(bar)

    cands = engine_info.get("candidates") or []
    if cands:
        st.markdown("**Top engine moves** (from the side to move's view):")
        rows = []
        for c in cands:
            mark = "⬅️ repertoire move" if c.uci == node.move_uci else ""
            rows.append(
                {"#": c.rank, "Move": c.san, "Eval": c.score_text(),
                 "Line": " ".join(c.pv[:5]), "": mark}
            )
        st.dataframe(rows, hide_index=True, width="stretch")

    if engine_info.get("rep_move_in_top"):
        st.success(
            f"✅ The repertoire move **{node.move_san}** is among the engine's top "
            f"choices (rank {engine_info.get('rep_move_rank')})."
        )
    else:
        st.info(
            f"ℹ️ The repertoire move **{node.move_san}** is not the engine's top pick. "
            "That is fine — repertoire moves are chosen for clear plans and ease of "
            "play, not just raw evaluation. Use the engine as supporting evidence, not "
            "as the final word."
        )


def render_pgn_tab(tree: RepertoireTree, node) -> None:
    if node.is_root:
        st.caption("The starting position.")
        return
    st.markdown("**Line (copy with the button in the corner):**")
    st.code(tree.line_san(node.node_id), language=None)
    st.markdown(f"**Move:** {node.label}  ({'mainline' if node.is_mainline else 'side variation'})")
    st.markdown(f"**Played by:** {node.mover} · **Now to move:** {node.side_to_move} · **Ply:** {node.ply}")
    if node.comment:
        st.markdown(f"**PGN comment:** {node.comment}")
    if node.nags:
        st.markdown(f"**NAGs:** {', '.join(node.nag_texts)}")
    st.markdown(f"**FEN before:** `{node.fen_before}`")
    st.markdown(f"**FEN after:** `{node.fen_after}`")

    children = tree.children(node.node_id)
    if children:
        st.markdown("**Next repertoire move(s):** " + ", ".join(c.label for c in children))
    siblings = tree.siblings(node.node_id)
    if siblings:
        st.markdown("**Sibling variations (other tries here):**")
        for s in siblings:
            kids = tree.children(s.node_id)
            resp = f" → repertoire plays {kids[0].label}" if kids else ""
            st.markdown(f"- {s.label}{resp}")
    trans = tree.transpositions(node.node_id)
    if trans:
        st.markdown("**Transposes to/from:**")
        for t in trans[:6]:
            st.markdown(f"- {tree.line_san(t)}")


def render_export_tab(tree: RepertoireTree) -> None:
    ss = st.session_state
    st.markdown(
        "Build a complete study guide: the full repertoire tree plus a rationale "
        "card for every unique position, exported as Markdown or HTML."
    )
    col1, col2 = st.columns(2)
    inc_engine = col1.checkbox(
        "Include engine notes", value=False,
        help="Requires a connected engine. Adds analysis to every position (slower).",
    )
    inc_llm = col2.checkbox(
        "Use AI for every card", value=False,
        help="Requires an API key. One Claude call per unique position — can be slow/costly.",
    )

    if inc_llm:
        st.warning(
            f"AI mode will make ~{tree.unique_position_count()} Claude calls. "
            "This can take a while and consume API credits."
        )

    if st.button("📦 Build study guide", type="primary"):
        eng = get_engine() if inc_engine else None
        if inc_engine and (eng is None or not eng.available):
            st.error("Engine not available — connect Stockfish first or untick engine notes.")
            return
        llm = get_llm() if inc_llm else None
        if inc_llm and (llm is None or not llm.available):
            st.error("AI not available — set ANTHROPIC_API_KEY or untick AI mode.")
            return

        bar = st.progress(0.0, text="Generating…")

        def _progress(frac, label):
            bar.progress(min(1.0, frac), text=f"Explaining {label} …")

        studies = generate_study(
            tree, engine=eng, depth=ss.depth, use_llm=inc_llm, llm=llm, progress=_progress
        )
        bar.progress(1.0, text="Done.")
        md = to_markdown(tree, studies)
        html_doc = to_html(tree, studies)
        st.session_state["_export_md"] = md
        st.session_state["_export_html"] = html_doc
        st.success(f"Built {len(studies)} rationale cards.")

    if "_export_md" in st.session_state:
        base = Path(ss.source_name or "repertoire").stem
        st.download_button(
            "⬇️ Download Markdown",
            data=st.session_state["_export_md"],
            file_name=f"{base}_study_guide.md",
            mime="text/markdown",
            width="stretch",
        )
        st.download_button(
            "⬇️ Download HTML",
            data=st.session_state["_export_html"],
            file_name=f"{base}_study_guide.html",
            mime="text/html",
            width="stretch",
        )
        with st.expander("Preview Markdown"):
            st.code(st.session_state["_export_md"][:6000], language="markdown")


# --------------------------------------------------------------------------- #
# Train mode (guess the move)
# --------------------------------------------------------------------------- #
def _advance_to_question(tree: RepertoireTree, node_id: int, side: str):
    """Auto-play the opponent's repertoire replies (randomly) starting from
    ``node_id`` until it's ``side``'s turn with at least one repertoire answer.

    Returns ``(position_node_id, done)`` where ``done`` means the line ended
    before another question could be asked.
    """
    cur = tree.nodes[node_id]
    while True:
        kids = tree.children(cur.node_id)
        if not kids:
            return cur.node_id, True
        if cur.side_to_move == side:
            return cur.node_id, False
        cur = random.choice(kids)


def _train_new_line(tree: RepertoireTree, side: str) -> dict:
    pos, done = _advance_to_question(tree, tree.root_id, side)
    return {"pos": pos, "done": done, "answered": None, "hint": False}


def _start_training(tree: RepertoireTree, side: str) -> None:
    st.session_state.train = {
        "side": side,
        "score": 0,
        "total": 0,
        "streak": 0,
        "best_streak": 0,
        "lines": 0,
        "source": st.session_state.source_name,
        **_train_new_line(tree, side),
    }


def _parse_user_move(board: chess.Board, raw: str) -> Optional[chess.Move]:
    """Accept SAN (with 0-0 style castling) or UCI. None if not legal here."""
    txt = (raw or "").strip()
    if not txt:
        return None
    txt = txt.replace("0-0-0", "O-O-O").replace("0-0", "O-O")
    try:
        return board.parse_san(txt)
    except Exception:
        pass
    try:
        mv = chess.Move.from_uci(txt.lower())
        if mv in board.legal_moves:
            return mv
    except Exception:
        pass
    return None


def _hint_for(board: chess.Board, child) -> str:
    try:
        mv = chess.Move.from_uci(child.move_uci)
    except Exception:
        return "Think about the most active piece."
    if board.is_castling(mv):
        return "Hint: the king wants safety…"
    piece = board.piece_at(mv.from_square)
    names = {
        chess.PAWN: "a pawn move", chess.KNIGHT: "a knight move",
        chess.BISHOP: "a bishop move", chess.ROOK: "a rook move",
        chess.QUEEN: "a queen move", chess.KING: "a king move",
    }
    kind = names.get(piece.piece_type, "a piece move") if piece else "a move"
    extra = " It's a capture!" if board.is_capture(mv) else ""
    return f"Hint: it's {kind}.{extra}"


def render_train(tree: RepertoireTree) -> None:
    ss = st.session_state
    tr = ss.train

    st.subheader("🎯 Train: guess the repertoire move")

    # Setup panel.
    if tr is None or tr.get("source") != ss.source_name:
        st.markdown(
            "The app walks a random line of your repertoire, playing the "
            "**opponent's** moves for you. At every one of *your* moves, type "
            "what the repertoire recommends. Opponent replies are chosen at "
            "random, so every run drills a different line."
        )
        side = st.selectbox("Train as", ["White", "Black"], index=0,
                            help="This repertoire is for White — but the option is here.")
        if st.button("▶ Start training", type="primary"):
            _start_training(tree, side)
            st.rerun()
        return

    side = tr["side"]
    node = tree.nodes[tr["pos"]]

    # Scoreboard.
    acc = f"{(100 * tr['score'] / tr['total']):.0f}%" if tr["total"] else "—"
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Score", f"{tr['score']} / {tr['total']}")
    m2.metric("Accuracy", acc)
    m3.metric("Streak", tr["streak"])
    m4.metric("Best streak", tr["best_streak"])
    m5.metric("Lines finished", tr["lines"])

    left, right = st.columns([1.1, 1])
    with left:
        svg = _board_svg(
            node.fen_after,
            None if node.is_root else node.move_uci,
            min(ss.board_size, 440),
            flipped=(side == "Black"),
        )
        if svg:
            _show_svg(svg)
        line = tree.line_san(node.node_id)
        st.caption(f"Line so far: `{line or '(start position)'}`")

    with right:
        kids = tree.children(node.node_id)
        board = chess.Board(node.fen_after)

        if tr["done"]:
            st.success("🏁 End of this repertoire line — well done!")
            if st.button("▶ Next line", type="primary"):
                tr.update(_train_new_line(tree, side))
                st.rerun()
        elif tr["answered"] is None:
            st.markdown(f"**You are {side}. What does the repertoire play here?**")
            with st.form("train_form", clear_on_submit=True):
                raw = st.text_input(
                    "Your move",
                    placeholder="SAN like Nb5, e4, O-O — or UCI like g1f3",
                )
                submitted = st.form_submit_button("✔ Check", type="primary")
            if submitted:
                mv = _parse_user_move(board, raw)
                if mv is None:
                    st.warning(
                        f"'{raw}' is not a legal move here — try again "
                        "(no penalty). Example format: Nf3, exd5, O-O."
                    )
                else:
                    given_san = board.san(mv)
                    match = next((k for k in kids if k.move_uci == mv.uci()), None)
                    tr["total"] += 1
                    if match is not None:
                        tr["score"] += 1
                        tr["streak"] += 1
                        tr["best_streak"] = max(tr["best_streak"], tr["streak"])
                    else:
                        tr["streak"] = 0
                    tr["answered"] = {
                        "given": given_san,
                        "correct": match is not None,
                        "child": (match or kids[0]).node_id,
                        "revealed": False,
                    }
                    st.rerun()

            hc1, hc2 = st.columns(2)
            if hc1.button("💡 Hint"):
                tr["hint"] = True
                st.rerun()
            if hc2.button("👀 Reveal"):
                tr["total"] += 1
                tr["streak"] = 0
                tr["answered"] = {
                    "given": None, "correct": False,
                    "child": kids[0].node_id, "revealed": True,
                }
                st.rerun()
            if tr.get("hint"):
                st.info(_hint_for(board, kids[0]))
        else:
            ans = tr["answered"]
            expected = tree.nodes[ans["child"]]
            if ans["revealed"]:
                st.info(f"👀 The repertoire move is **{expected.label}**.")
            elif ans["correct"]:
                st.success(f"✅ **{ans['given']}** — that's the repertoire move!")
            else:
                st.error(
                    f"❌ You played **{ans['given']}** — the repertoire plays "
                    f"**{expected.label}**."
                )
            # Short teaching moment: hook + why for the expected move.
            try:
                card = HeuristicExplainer().explain(tree, expected)
                st.markdown(f"> 🧠 {card.memory_hook}")
                with st.expander("Why this move?"):
                    st.markdown(card.why_this_move)
                    st.markdown(f"_{card.jobava_logic}_")
            except Exception:
                pass

            n1, n2 = st.columns(2)
            if n1.button("Next ▶", type="primary"):
                nxt, done = _advance_to_question(tree, expected.node_id, side)
                if done:
                    tr["lines"] += 1
                tr.update({"pos": nxt, "done": done, "answered": None, "hint": False})
                st.rerun()
            if n2.button("🔍 Open in Explorer"):
                _select_node(tree, expected.node_id)
                # The 'mode' radio is already instantiated this run, so stage the
                # switch and apply it at the top of the next run (see main()).
                ss["_pending_mode"] = MODE_EXPLORE
                st.rerun()

    st.markdown("---")
    if st.button("⏹ End training session"):
        ss.train = None
        st.rerun()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_welcome() -> None:
    st.title("♞ Jobava London — Opening Rationale Explorer")
    st.markdown(
        """
Turn a PGN repertoire into an interactive study tool. For every move you'll get
a structured **rationale card**: what's happening, *why* the move is played, the
Jobava London idea behind it, what it prevents, alternatives, tactics, plans, a
memory hook, a difficulty rating and a common-mistake warning. When you know the
lines, switch to **🎯 Train** and drill them guess-the-move style.

This tool works fully offline. Stockfish and an Anthropic API key are optional
extras — without them you still get the parsed tree, PGN comments, board
diagrams and the built-in heuristic explanations.
        """
    )
    samples = sorted(SAMPLE_DIR.glob("*.pgn")) if SAMPLE_DIR.exists() else []
    if samples:
        st.markdown("#### Quick start")
        cols = st.columns(min(3, len(samples)))
        descriptions = {
            "jobava_london_repertoire.pgn": "Full Jobava London white repertoire (260 moves, 70 lines).",
            "jobava_sample.pgn": "Tiny annotated sample — good for a first look.",
        }
        for col, sample in zip(cols, samples):
            with col:
                st.markdown(f"**{sample.name}**")
                st.caption(descriptions.get(sample.name, "Bundled sample PGN."))
                if st.button(f"Load {sample.stem}", key=f"qs_{sample.name}",
                             type="primary", width="stretch"):
                    text = sample.read_text(encoding="utf-8", errors="replace")
                    load_tree(text, sample.name)
                    st.rerun()
        st.caption("…or upload your own PGN from the sidebar (1 · Load a PGN).")


def main() -> None:
    _init_state()
    render_sidebar()
    tree: Optional[RepertoireTree] = st.session_state.tree

    if tree is None:
        render_welcome()
        return

    # Apply a mode switch staged by a button handler on the previous run
    # (a widget-bound key can only be written before its widget exists).
    pending = st.session_state.pop("_pending_mode", None)
    if pending is not None:
        st.session_state["mode"] = pending

    mode = st.radio(
        "Mode",
        [MODE_EXPLORE, MODE_TRAIN],
        horizontal=True,
        key="mode",
        label_visibility="collapsed",
    )

    if mode == MODE_TRAIN:
        render_train(tree)
        return

    left, right = st.columns([1, 1.9], gap="large")
    with left:
        render_tree(tree)

    with right:
        sel = st.session_state.selected_id
        if sel is None or sel not in tree.nodes:
            sel = tree.root_id
            st.session_state.selected_id = sel
        node = tree.nodes[sel]

        st.subheader(node.label if not node.is_root else "Starting position")
        render_nav(tree, node)
        render_board(tree, node)

        engine_info = engine_info_for(node) if not node.is_root else None

        tab_r, tab_e, tab_p, tab_x = st.tabs(
            ["📖 Rationale", "🔧 Engine", "🗂️ PGN context", "📦 Export"]
        )
        with tab_r:
            render_rationale_tab(tree, node, engine_info)
        with tab_e:
            render_engine_tab(tree, node, engine_info)
        with tab_p:
            render_pgn_tab(tree, node)
        with tab_x:
            render_export_tab(tree)


if __name__ == "__main__":
    main()
