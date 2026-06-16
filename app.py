"""Jobava London — Opening Rationale Explorer (Streamlit app).

Run with:  streamlit run app.py

Left:  the repertoire as a collapsible tree.
Main:  the board for the selected position + tabs
       (Rationale / Engine / PGN context / Export).

Everything degrades gracefully: no Stockfish and no API key still gives you the
parsed tree, PGN comments, board diagrams and the built-in heuristic rationale.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import chess
import chess.svg
import streamlit as st
import streamlit.components.v1 as components


def _show_svg(svg: str) -> None:
    """Render inline SVG, preferring the modern ``st.html`` over the deprecated
    ``components.html`` (which uses an iframe and is being removed)."""
    wrapped = f"<div style='display:flex;justify-content:center'>{svg}</div>"
    html_fn = getattr(st, "html", None)
    if callable(html_fn):
        html_fn(wrapped)
    else:  # Streamlit < 1.33 fallback
        components.html(wrapped, height=420)

from engine import Engine, white_pov_text
from explainer import (
    DEFAULT_LLM_MODEL,
    LLMExplainer,
    explain_position,
)
from export import generate_study, to_html, to_markdown
from pgn_parser import RepertoireTree, load_pgn_text

APP_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = APP_DIR / "sample_data"

st.set_page_config(page_title="Jobava London Rationale Explorer", layout="wide")


# --------------------------------------------------------------------------- #
# Session state helpers
# --------------------------------------------------------------------------- #
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("tree", None)
    ss.setdefault("source_name", "")
    ss.setdefault("selected_id", None)
    ss.setdefault("expanded", set())
    ss.setdefault("flip", False)
    ss.setdefault("engine", None)
    ss.setdefault("engine_path", os.environ.get("STOCKFISH_PATH", ""))
    ss.setdefault("engine_cache", {})
    ss.setdefault("llm_card_cache", {})
    ss.setdefault("depth", 12)
    ss.setdefault("use_engine", False)
    ss.setdefault("use_llm", False)
    ss.setdefault("llm_model", DEFAULT_LLM_MODEL)
    ss.setdefault("api_key_override", "")


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
    st.sidebar.title("♘ Jobava London")
    st.sidebar.caption("Opening Rationale Explorer")

    with st.sidebar.expander("1 · Load a PGN", expanded=ss.tree is None):
        uploaded = st.file_uploader("Upload a PGN file", type=["pgn"])
        if uploaded is not None:
            if st.button("Load uploaded PGN", use_container_width=True):
                text = uploaded.read().decode("utf-8", errors="replace")
                load_tree(text, uploaded.name)
                st.rerun()

        # Bundled samples.
        samples = sorted(SAMPLE_DIR.glob("*.pgn")) if SAMPLE_DIR.exists() else []
        if samples:
            names = [s.name for s in samples]
            choice = st.selectbox("…or pick a bundled sample", names)
            if st.button("Load sample", use_container_width=True):
                text = (SAMPLE_DIR / choice).read_text(encoding="utf-8", errors="replace")
                load_tree(text, choice)
                st.rerun()

        path = st.text_input("…or a path on disk", value="")
        if path and st.button("Load from path", use_container_width=True):
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
            help="Leave blank to auto-detect 'stockfish' on PATH or $STOCKFISH_PATH.",
        )
        ss.depth = st.slider("Analysis depth", 6, 22, ss.depth)
        if st.button("Connect / test engine", use_container_width=True):
            if ss.engine is not None:
                ss.engine.close()
                ss.engine = None
            eng = get_engine()
            if eng and eng.available:
                st.success(f"Engine ready: {eng.path}")
            else:
                st.warning(eng.error if eng else "Could not connect.")
        ss.use_engine = st.checkbox("Use engine for analysis", value=ss.use_engine)
        # Status line.
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
# Tree view
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

    c1, c2, c3 = st.columns(3)
    if c1.button("Expand all", use_container_width=True):
        ss.expanded = set(tree.nodes.keys())
        st.rerun()
    if c2.button("Collapse", use_container_width=True):
        ss.expanded = {tree.root_id}
        st.rerun()
    if c3.button("Mainline", use_container_width=True):
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
            if st.button("Go to this line", use_container_width=True):
                ss.selected_id = target
                # expand all ancestors
                cur = target
                while cur is not None:
                    ss.expanded.add(cur)
                    cur = tree.nodes[cur].parent_id
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
            indent = " " * max(0, depth - 1)
            if node.is_root:
                text = "Starting position"
            else:
                marker = "" if node.is_mainline else "° "
                flag = " 💬" if node.comment else ""
                nag = " ‼" if node.nags else ""
                text = f"{indent}{marker}{node.label}{nag}{flag}"
            kind = "primary" if nid == ss.selected_id else "secondary"
            if st.button(text, key=f"sel_{nid}", use_container_width=True, type=kind):
                ss.selected_id = nid
                ss.expanded.add(nid)
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("° = side variation · 💬 = has PGN comment · ‼ = has NAG")


# --------------------------------------------------------------------------- #
# Board + detail panel
# --------------------------------------------------------------------------- #
def render_board(tree: RepertoireTree, node) -> None:
    try:
        board = chess.Board(node.fen_after)
    except Exception:
        st.error("Could not render this position.")
        return
    last = None
    if not node.is_root and node.move_uci:
        try:
            last = chess.Move.from_uci(node.move_uci)
        except Exception:
            last = None
    check_sq = board.king(board.turn) if board.is_check() else None
    svg = chess.svg.board(
        board,
        size=380,
        lastmove=last,
        check=check_sq,
        flipped=st.session_state.flip,
        coordinates=True,
    )
    _show_svg(svg)
    cols = st.columns([1, 1, 2])
    cols[0].checkbox("Flip board", key="flip")
    cols[1].caption(f"{node.side_to_move} to move")
    cols[2].caption(f"`{node.fen_after}`")


def _difficulty_stars(n: int) -> str:
    return "★" * n + "☆" * (5 - n)


def render_rationale_tab(tree: RepertoireTree, node, engine_info) -> None:
    ss = st.session_state

    use_llm = ss.use_llm
    llm = get_llm() if use_llm else None
    cache_key = (node.fen_after, ss.llm_model)

    if use_llm and llm and llm.available:
        cached = ss.llm_card_cache.get(cache_key)
        if cached is not None:
            card = cached
        else:
            st.info("AI explanations are enabled. Generate one for this position:")
            if st.button("✨ Generate AI explanation", type="primary"):
                with st.spinner("Asking Claude…"):
                    card = explain_position(tree, node, engine_info, use_llm=True, llm=llm)
                    ss.llm_card_cache[cache_key] = card
                st.rerun()
            card = explain_position(tree, node, engine_info, use_llm=False)
    else:
        card = explain_position(tree, node, engine_info, use_llm=False)

    badge = {"llm": "🤖 AI", "heuristic": "🧮 heuristic", "placeholder": "📄 template"}.get(
        card.source, card.source
    )
    st.markdown(
        f"#### {card.move_label} &nbsp; "
        f"<span style='font-size:0.7em;color:#888'>({badge}) · "
        f"difficulty {_difficulty_stars(card.difficulty)} {card.difficulty}/5</span>",
        unsafe_allow_html=True,
    )
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
        st.caption("Computing…")
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
        st.dataframe(rows, hide_index=True, use_container_width=True)

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
    st.markdown(f"**Line:** `{tree.line_san(node.node_id)}`")
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
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download HTML",
            data=st.session_state["_export_html"],
            file_name=f"{base}_study_guide.html",
            mime="text/html",
            use_container_width=True,
        )
        with st.expander("Preview Markdown"):
            st.code(st.session_state["_export_md"][:6000], language="markdown")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def render_welcome() -> None:
    st.title("♘ Jobava London — Opening Rationale Explorer")
    st.markdown(
        """
Turn a PGN repertoire into an interactive study tool. For every move you'll get
a structured **rationale card**: what's happening, *why* the move is played, the
Jobava London idea behind it, what it prevents, alternatives, tactics, plans, a
memory hook, a difficulty rating and a common-mistake warning.

**To start:** use the sidebar to upload a PGN, pick a bundled sample, or load a
file from disk.

This tool works fully offline. Stockfish and an Anthropic API key are optional
extras — without them you still get the parsed tree, PGN comments, board
diagrams and the built-in heuristic explanations.
        """
    )


def main() -> None:
    _init_state()
    render_sidebar()
    tree: Optional[RepertoireTree] = st.session_state.tree

    if tree is None:
        render_welcome()
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
