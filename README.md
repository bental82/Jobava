# ♘ Opening Rationale Explorer — Jobava London & Scotch

A small, local study tool that turns a **PGN repertoire** into an interactive
*opening rationale explorer*. For every move in your repertoire it shows a
structured **rationale card** that answers the questions that actually matter
when you're learning an opening:

1. What is happening in the position?
2. Why is the repertoire move recommended?
3. What plan does the move support?
4. What tactical or strategic idea is hidden behind it?
5. What danger / opponent idea does it prevent?
6. What should you remember as a club-level player?
7. What are the common alternatives, and why are they less relevant here?

It ships with **two complete White repertoires** — the **Jobava London**
(1.d4 + Nc3 + Bf4) and a **Scotch-centred 1.e4 repertoire** (Scotch Game with
the Qxd4 centralisation lines, 2.c3 Alapin vs the Sicilian, Advance Caro-Kann,
Exchange French, the critical 3.Nxe5 Petroff with the Qe2 tricks, Stafford
Gambit antidote, and refutations of the early gambits) — and works with any
PGN repertoire. The explainer **auto-detects the opening family** from the
loaded PGN (1.d4 → Jobava book, 1.e4 → Scotch book, anything else → a neutral
principles book) and adapts every rationale card to it.

> The explanations are written for a **beginner-to-intermediate** player who
> wants practical understanding — plans, threats, piece activity, pawn breaks,
> king safety and memorable patterns — not grandmaster-level engine precision.

---

## What it does

- **Imports a PGN repertoire** (not just a single game): mainlines, side
  variations, comments and NAGs are all parsed into an expandable
  *repertoire tree*. Positions are deduplicated by **FEN**, so transpositions
  are recognised.
- **Explains every position** with a rationale card containing: position
  summary · why this move · opening logic (Jobava- or Scotch-aware) · what it
  prevents · candidate alternatives · tactical checks · plans after the move ·
  a memory hook · a difficulty rating (1–5) · a common-mistake warning.
- **Heuristic explanations work offline** with no API key and no engine. They
  combine concrete board analysis (material, loose pieces, checks, pins, forks,
  threats) with built-in, opening-aware knowledge bases (Jobava London and
  Scotch/1.e4), selected automatically per PGN.
- **Optional Stockfish support**: evaluation before/after the move, the top 3
  engine candidates, and whether your repertoire move is among them — used as
  *supporting evidence only*, never as a replacement for the human explanation.
- **Optional AI explanations** via the Anthropic API (Claude) for richer,
  prose explanations. Disabled gracefully if no key is present.
- **Board diagrams** (python-chess SVG) that update as you click through the
  tree — with **arrows showing the next repertoire move(s)** (green = mainline,
  amber = other branches).
- **Fast navigation**: start/back/next buttons, a branch switcher for sibling
  variations, a move search box, and jump-to-line.
- **🎯 Train mode**: guess-the-move drilling. The app plays the opponent's
  repertoire replies at random and asks for *your* move, with score, streak,
  hints and instant "why" feedback — spaced-repetition style studying.
- **Study-guide export** to **Markdown** and **HTML** (with board diagrams and
  FENs).

## What it does *not* do

- It is **not** a full chess GUI, engine match runner, or game database.
- It does **not** require a database — everything is parsed in memory.
- It will **not** crash if Stockfish or an API key is missing; those features
  simply switch off.
- The engine is used for *evidence*, not to overrule the repertoire. AI
  explanations are instructed to **hedge rather than hallucinate** when a
  position's rationale is unclear.

---

## Install

Requires **Python 3.9+**.

```bash
# from the project directory
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **Note on the chess library:** the PyPI package is named `chess`
> (this *is* "python-chess"). If `pip install chess` fails to build on your
> system because of an old `setuptools`, upgrade your build tools first:
> `pip install --upgrade pip setuptools wheel`, then reinstall. Using a fresh
> virtual environment (as above) avoids this on most systems.

## Run

```bash
streamlit run app.py
```

Your browser opens the app. Then:

1. **Load a PGN** — use the quick-start buttons on the welcome screen, or the
   sidebar (upload a file, pick a bundled sample, or load from a path on disk).
2. **Explore**: click any move in the repertoire tree on the left (or use
   ⏮/◀/▶ navigation and the search box). Read the **Rationale** tab on the
   right; check **Engine** / **PGN context** as needed; build a study guide
   from the **Export** tab.
3. **Train**: switch to **🎯 Train** at the top and drill the repertoire
   guess-the-move style. Opponent replies are randomised from the repertoire,
   so every run covers different lines.

## Provide the PGN file

Three ways, all in the sidebar under *“1 · Load a PGN”*:

- **Upload** a `.pgn` file.
- **Pick a bundled sample** from `sample_data/`:
  - `jobava_sample.pgn` — a tiny example for testing (has a comment, a NAG and
    variations).
  - `jobava_london_repertoire.pgn` — a full Jobava London white repertoire.
  - `scotch_repertoire.pgn` — a complete 1.e4 white repertoire built around
    the Scotch Game (317 moves, 67 lines, answers to all main defences).
- **Load from a path** on disk.

Multiple games in one PGN are merged into a single repertoire tree, and
identical move-sequences from different games are de-duplicated automatically.

## Connect Stockfish (optional)

1. Download Stockfish from <https://stockfishchess.org/download/> (or install
   via your package manager, e.g. `brew install stockfish`,
   `apt install stockfish`).
2. In the sidebar under *“2 · Stockfish engine”*, paste the path to the binary
   (or leave blank to auto-detect `stockfish` on your `PATH` or the
   `STOCKFISH_PATH` environment variable), then click **Connect / test engine**.
3. Tick **Use engine for analysis**. Choose a depth (default 12 — shallow and
   fast is recommended for study).

```bash
# alternative: tell the app where Stockfish is via the environment
export STOCKFISH_PATH=/usr/local/bin/stockfish
streamlit run app.py
```

## Enable Anthropic (Claude) explanations (optional)

1. Set your API key before launching:

   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...
   streamlit run app.py
   ```

   (Or paste it into the sidebar under *“3 · AI explanations”* — it is kept only
   for the current session.)
2. Tick **Use AI explanations**. On the **Rationale** tab, click
   **✨ Generate AI explanation** for the selected position.
3. Optionally change the **Claude model** (default `claude-opus-4-8`; the
   `JOBAVA_LLM_MODEL` env var sets the default).

If `anthropic` isn't installed or no key is found, the app silently falls back
to the heuristic explanations.

---

## Deploy it (optional)

The app is a normal Streamlit app, so any host that runs a persistent Python
process works.

### Streamlit Community Cloud (free, easiest)

1. Go to <https://share.streamlit.io> and sign in with GitHub.
2. *Create app* → repo `bental82/Jobava`, branch **`production`**, main file
   **`app.py`** → Deploy.
3. The included `packages.txt` installs **Stockfish** automatically (the app
   auto-detects it at `/usr/games/stockfish`).
4. For AI explanations: app *Settings → Secrets* → add
   `ANTHROPIC_API_KEY = "sk-ant-..."`.

### Docker (Render, Railway, Fly.io, Cloud Run, your own box)

```bash
docker build -t jobava-explorer .
docker run -p 8501:8501 -e ANTHROPIC_API_KEY=sk-ant-... jobava-explorer
```

The image bundles Stockfish and respects the host's `$PORT`, so it deploys
unmodified to the usual container platforms.

### A note on Vercel

Vercel hosts static sites and short-lived serverless functions; it cannot run a
persistent WebSocket server like Streamlit, so this app will always show
*"No Production Deployment"* there. Use Streamlit Community Cloud or a container
host instead.

---

## Project structure

```
.
├── app.py             # Streamlit UI (Explore + Train modes, tree, board, tabs)
├── pgn_parser.py      # PGN -> repertoire tree (variations, comments, NAGs, FEN dedup)
├── engine.py          # optional Stockfish wrapper (safe when missing)
├── explainer.py       # heuristic + optional LLM rationale cards
├── openings.py        # opening knowledge bases (Jobava London, Scotch/1.e4) + auto-detection
├── export.py          # Markdown / HTML study-guide export
├── requirements.txt
├── packages.txt       # apt deps for Streamlit Community Cloud (Stockfish)
├── Dockerfile         # container image (bundles Stockfish)
├── .streamlit/
│   └── config.toml    # theme
├── README.md
└── sample_data/
    ├── jobava_sample.pgn                 # tiny test PGN
    ├── jobava_london_repertoire.pgn      # full Jobava London repertoire
    └── scotch_repertoire.pgn             # full Scotch-centred 1.e4 repertoire
```

## Tips for study

- Start with the **mainline** (the tree opens it for you) and read each card's
  *memory hook* and *mistake warning*.
- Use the *difficulty* rating to spot the moves worth extra repetition.
- Turn the engine on only when you want a sanity check — the **why** matters
  more than the centipawns.
- Export a Markdown/HTML guide once you've explored, and review it away from the
  board.
- Once the ideas feel familiar, drill them in **🎯 Train** mode — a few random
  lines a day beats one long session. Use the hint before revealing!

## Troubleshooting

- **“No Stockfish binary found.”** Provide a full path in the sidebar, or set
  `STOCKFISH_PATH`. The path must point at the executable.
- **AI button does nothing / errors.** Check that `ANTHROPIC_API_KEY` is set and
  `pip install anthropic` succeeded. Errors are surfaced in the sidebar status.
- **A move/variation didn't import.** Open the **⚠️ parser note(s)** expander in
  the sidebar — malformed games are skipped, not fatal.
