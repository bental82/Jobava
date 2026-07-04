"""Opening knowledge bases for the rationale explorer.

Each :class:`OpeningBook` bundles a move classifier plus the hand-written,
club-level explanation templates for one repertoire family. The explainer picks
a book automatically from the loaded PGN (:func:`detect_book`): 1.d4 loads the
Jobava London book, 1.e4 loads the Scotch-centred book, anything else falls
back to a neutral, principle-based book.

The texts aim to be *thorough and general*: they explain plans, threats, piece
activity, pawn breaks and king safety in language a beginner-to-intermediate
player can reuse across move orders — not memorised engine lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List

import chess

from pgn_parser import RepNode, RepertoireTree


@dataclass
class OpeningBook:
    """A move classifier + explanation templates for one repertoire family."""

    key: str
    name: str
    logic_label: str
    classify: Callable[[RepNode, chess.Board], str]
    why: Dict[str, str] = field(default_factory=dict)
    logic: Dict[str, str] = field(default_factory=dict)
    prevents: Dict[str, str] = field(default_factory=dict)
    plans: Dict[str, List[str]] = field(default_factory=dict)
    hook: Dict[str, str] = field(default_factory=dict)
    difficulty: Dict[str, int] = field(default_factory=dict)
    mistake: Dict[str, str] = field(default_factory=dict)
    default_logic: str = ""
    default_plans: List[str] = field(default_factory=list)
    root_summary: str = ""
    root_why: str = "Select a move in the tree to see its rationale."
    root_logic: str = ""
    root_plans: List[str] = field(default_factory=list)
    root_hook: str = ""
    llm_persona: str = "You are a friendly, practical chess coach."


# --------------------------------------------------------------------------- #
# Jobava-London knowledge base
# --------------------------------------------------------------------------- #
def _classify_jobava(node: RepNode, board_before: chess.Board) -> str:
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
    if san == "Bxe5":
        return "Bxe5"
    if san == "dxe5" and node.mover == "White" and node.ply <= 3:
        return "take_englund"
    if san == "dxe5" or (ptype == chess.PAWN and to_name == "e5" and "x" in san and node.mover == "White"):
        return "dxe5"
    if san == "exd6" and node.mover == "White":
        return "exd6"
    if san == "d5" and node.mover == "White" and ptype == chess.PAWN:
        return "d5_push"
    if san == "e5" and node.mover == "White" and ptype == chess.PAWN:
        return "e5_push"
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
        if san == "e5" and node.ply == 2:
            return "blk_englund"
        if san == "cxd6":
            return "blk_cxd6"
        if san == "Nh5":
            return "blk_Nh5"
        if san == "Rb8":
            return "blk_Rb8"
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
    "Bxe5": "A clean capture in the centre: the bishop removes the recapturing knight, keeping White's extra pawn or superior activity from the earlier exchange.",
    "take_englund": "Simply accept the gambit. 1...e5 (the Englund Gambit) gives up a pawn for vague activity — White takes it and develops calmly, asking Black to prove compensation that isn't there.",
    "exd6": "Keeping the extra pawn and the initiative: the capture on d6 disrupts Black's development and leaves White simply a pawn up if Black regains it slowly.",
    "d5_push": "Against ...c5 White pushes past: d5 grabs space, fixes the centre and leaves Black's c5-pawn biting on granite while White develops comfortably.",
    "e5_push": "The pawn advances to e5 with tempo, kicking the f6-knight away from the defence of the king and gaining a big space advantage.",
    "blk_englund": "The Englund Gambit: Black sacrifices a pawn immediately for open lines and tricks (like ...Qb4+). Objectively unsound, but you must know the safe path.",
    "blk_cxd6": "Black is forced to recapture, accepting doubled d-pawns and an open c-file that can later become a target.",
    "blk_Nh5": "Black lunges at the f4-bishop to win the bishop pair — at the cost of three tempi with one knight and a misplaced piece on the rim.",
    "blk_Rb8": "Black side-steps the Nxc7+ fork by moving the rook off a8 — but this loses time and c7 is still fatally weak.",
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
    "Bxe5": "The tactical dust settles with White on top — a typical outcome of the Nb5/c7 pressure: Black's ...e5 counter-strike fails to concrete moves.",
    "take_englund": "Not strictly a Jobava position, but part of the repertoire's philosophy: meet sidelines with simple, healthy moves and keep the material.",
    "exd6": "Same repertoire philosophy as the main lines: when Black plays loosely, take the material and develop — no need for heroics.",
    "d5_push": "With the centre fixed by d5, White gets a comfortable space edge and can still develop the Bf4/Nc3 way or switch to a mainline setup.",
    "e5_push": "The e4–e5 space grab is the punishment for slow fianchetto setups: Black's knight is kicked before it settles, and White attacks on the kingside next.",
    "blk_englund": "The repertoire's answer is pure pragmatism: take on e5, develop with Nf3/Nc3, give nothing back.",
    "blk_cxd6": "This structure is a long-term win for White: the bishop pair plus Black's doubled d-pawns give you something to play against all game.",
    "blk_Nh5": "Don't fear ...Nh5 — if the knight takes on f4, exf4 opens the e-file for White and the f4-pawn clamps e5. Black has spent tempi for very little.",
    "blk_Rb8": "When Nb5 forces awkward moves like ...Rb8, the system has already succeeded — now White cashes in on c7 with Bxc7.",
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
    "Bxe5": "Take back on e5 with the bishop — the tactics already worked.",
    "take_englund": "Englund? Take the pawn, keep the pawn.",
    "exd6": "Grabbed a pawn? Keep it and just develop.",
    "d5_push": "Meet ...c5 with d5: space now, plans later.",
    "e5_push": "Kick the knight with e5 — space plus attack.",
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
    "Bxe5": 3,
    "take_englund": 2,
    "exd6": 2,
    "d5_push": 2,
    "e5_push": 2,
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
    "take_englund": "Against the Englund, don't get greedy or fancy — beware the ...Qb4+ trick if you defend e5 carelessly with Bf4 too early.",
    "exd6": "Don't rush to protect the extra pawn with awkward moves — development first; the pawn often holds itself.",
    "e5_push": "Only push e5 when it gains a tempo or space that matters — otherwise it can leave d5 and the light squares weak.",
    "d5_push": "After d5, don't let the position drift — Black wants ...e6/...b5 breaks; keep developing actively.",
}


# --------------------------------------------------------------------------- #
# Scotch-centred 1.e4 knowledge base
# --------------------------------------------------------------------------- #
def _classify_scotch(node: RepNode, board_before: chess.Board) -> str:
    """Map a move to a theme key for a Scotch-centred 1.e4 repertoire."""
    san = node.move_san or ""
    path = node.san_path
    ply = node.ply
    first_black = path[1] if len(path) >= 2 else None
    fred_line = first_black == "f5"
    prev_san = path[-2] if len(path) >= 2 else ""

    if node.mover == "White":
        if san == "e4" and ply == 1:
            return "e4"
        if san == "exf5":
            return "accept_gambit"
        if fred_line and ply >= 3:
            # Inside the forced 1...f5 refutation: forcing moves vs build-up moves.
            return "king_hunt" if ("+" in san or "x" in san) else "fred_build"
        if san == "c3":
            if ply == 3:
                return "alapin_c3"
            if prev_san == "Bb4+":
                return "c3_tempo"
            return "c3_solid"
        if san == "cxd4":
            return "cxd4_rebuild"
        if san in ("Nf3", "Ngf3"):
            return "N_retreat" if "Nxe5" in path[:-1] else "Nf3_dev"
        if san == "d4":
            return "d4_strike" if "e5" in path[:4] else "d4_center"
        if san == "d5":
            return "d5_clamp"
        if san == "e5":
            return "e5_space"
        if san == "exd5":
            return "exd5_simplify"
        if san == "dxe5":
            return "dxe5_clear"
        if san == "Nxd4":
            return "Nxd4_recapture"
        if san.startswith("Nxe4"):
            return "Nxe4_recapture"
        if san.startswith("Qxd4"):
            return "Qxd4_central"
        if san.startswith("Qxd8"):
            return "queen_trade_edge"
        if san == "Qd1":
            return "Q_retreat"
        if san == "Qe2":
            return "Qe2_pin"
        if san.startswith("Qxf3"):
            return "Qxf3_pair"
        if san == "Nxe5":
            if prev_san == "Nf6" and ply == 5:
                return "petroff_Nxe5"
            return "Nxe5_tactic"
        if san.startswith("Nxc6"):
            return "Nxc6_keep_pawn" if "Nxe5" in path else "Nxc6_structure"
        if san == "Nc3":
            return "Nc3_tempo" if any(p.startswith("Qxd5") for p in path) else "Nc3_dev"
        if san in ("Ne2", "Nge2"):
            return "N_reroute"
        if san == "Ng3":
            return "Ng3_kick"
        if san == "Bg5":
            return "Bg5_pin"
        if san == "Bc4":
            return "Bc4_f7"
        if san == "Be3":
            return "Be3_challenge"
        if san == "Be2":
            return "Be2_calm"
        if san.startswith("Bb5"):
            return "Bb5_check"
        if san == "d3":
            return "d3_solid"
        if san == "h3" and prev_san in ("Bg4", "h5"):
            return "h3_question"
        if san in ("O-O", "0-0"):
            return "OO_safe"
        if san in ("O-O-O", "0-0-0"):
            return "OOO_pressure"
        return "generic"

    # ---- Black's moves ----
    if fred_line and ply > 2:
        return "blk_fred_defence"
    if ply == 2:
        first_moves = {
            "e5": "blk_e5_open", "d5": "blk_scandi", "e6": "blk_french",
            "c6": "blk_caro", "c5": "blk_sicilian", "d6": "blk_pirc",
            "Nf6": "blk_alekhine", "Nc6": "blk_nimzo", "g6": "blk_modern",
            "b6": "blk_owen", "f5": "blk_fred",
        }
        if san in first_moves:
            return first_moves[san]
    if ply == 4 and first_black == "e5":
        second_moves = {
            "Nc6": "blk_Nc6_dev", "Nf6": "blk_petroff", "d6": "blk_philidor",
            "Bc5": "blk_Bc5_early", "f6": "blk_damiano", "Qf6": "blk_early_queen",
            "d5": "blk_elephant",
        }
        if san in second_moves:
            return second_moves[san]
    if san == "e5" and first_black in ("Nc6", "Nf6"):
        return "blk_e5_open"
    if san == "Nc6" and "Nxe5" in path:
        return "blk_stafford"
    if san == "d6" and "Nxe5" in path and ply == 6:
        return "blk_petroff_main"
    if san == "Qe7" and "Nxe5" in path:
        return "blk_Qe7_petroff"
    if san in ("Kxd8", "Nxd8"):
        return "blk_recapture_d8"
    if san in ("bxc6", "dxc6"):
        return "blk_recapture_c6"
    if san == "exd4":
        return "blk_exd4"
    if san.startswith("Nxd4"):
        return "blk_Nxd4_trade"
    if san.startswith("Nxe5"):
        return "blk_Nxe5_recapture"
    if san.startswith("Nxe4"):
        return "blk_Nxe4_grab"
    if san == "Bc5":
        return "blk_Bc5_early"
    if san == "Bb4+":
        return "blk_check_b4"
    if san == "Qf6":
        return "blk_early_queen"
    if san == "c5":
        return "blk_c5_kick" if any(p.startswith("Qxd4") for p in path) else "blk_c5_break"
    if san == "cxd4":
        return "blk_cxd4_release"
    if san == "Nf6" and first_black == "e5":
        return "blk_Nf6_hit"
    if san == "d6":
        return "blk_d6_solid"
    if san == "f6":
        return "blk_f6_weak"
    if san.startswith("Qxd5") or (san in ("Qe6+", "Qe5+", "Qd8", "Qa5") and first_black == "d5"):
        return "blk_scandi_queen"
    if san == "h5" and "Nxe5" in path:
        return "blk_h5_lash"
    if san == "d5" and first_black in ("c6", "e6", "c5"):
        return "blk_d5_stake"
    if san == "d4":
        return "blk_d4_push"
    if san == "dxe4":
        return "blk_release"
    if san == "exd5":
        return "blk_recapture_symm"
    if san == "Bf5":
        return "blk_bishop_active"
    if san == "Bg4":
        return "blk_bishop_pin"
    if san == "Bxf3":
        return "blk_trade_pair"
    if san == "Be7":
        return "blk_Be7_solid"
    if san == "Nd5":
        return "blk_N_dance"
    if san == "Nd4":
        return "blk_Nd4_lunge"
    return "generic"


SCOTCH_WHY: Dict[str, str] = {
    "e4": "Stakes a claim in the centre and frees the queen and king's bishop in one move — the classical way to fight for the initiative from move one.",
    "Nf3_dev": "Develops with a threat (the e5-pawn). Every developing move that also creates a threat gains time — the opponent must react instead of developing.",
    "d4_strike": "The Scotch strike: open the centre immediately, before Black has finished developing. After ...exd4 White recaptures with a piece that lands actively in the centre.",
    "d4_center": "Builds the ideal pawn pair. With pawns on e4 and d4 White controls the four key central squares and both bishops breathe freely.",
    "Nxd4_recapture": "Recaptures toward the centre: the knight is superbly placed on d4, and White enjoys freer development and open lines.",
    "Nxe4_recapture": "Recaptures toward the centre — the knight stands proudly on e4 and Black must spend more time challenging it.",
    "Qxd4_central": "Normally the queen shouldn't come out early — but the c6-knight that would harass her has just been traded. She sits on d4 like a monarch: central, safe, and controlling both wings. This is the repertoire's core idea.",
    "Nc3_dev": "Simple development toward the centre: it supports e4, covers the d5-square, and keeps queenside castling available.",
    "Nc3_tempo": "Develops and attacks the early queen at the same time — Black must move her again, handing White free development.",
    "Bg5_pin": "Develops with pressure on the f6-knight. The pin kills ...Nxe4-style tricks and softens Black's kingside dark squares.",
    "OOO_pressure": "Castling long completes the plan: the king reaches safety while the rook lands on d1 in the same move, staring straight down the d-file at Black's position.",
    "OO_safe": "Tucks the king away and connects the rooks — with the centre under control, king safety comes before adventures.",
    "queen_trade_edge": "This queen trade is a *gain*, not a concession: Black must recapture awkwardly — ...Kxd8 loses castling forever, ...Nxd8 buries the knight — and White keeps a risk-free endgame pull.",
    "Nxe5_tactic": "Simply wins (or wins back) the e5-pawn — the preceding exchanges left it short of defenders. Count the captures first; here the arithmetic works.",
    "petroff_Nxe5": "The critical test of the Petroff: take the pawn. Black usually regains it, but White chooses the moment, keeps a small initiative, and gets to set the well-known e-file traps.",
    "N_retreat": "The knight returns with the job done: material is safe, the position is sound, and Black still has to prove the time spent was worth anything.",
    "Nxc6_keep_pawn": "Trade pieces while a pawn up — every exchange makes the extra pawn more valuable and drains the gambiteer's attacking dreams.",
    "Nxc6_structure": "Removes the key defender and damages Black's queenside: after ...bxc6 the doubled pawns are a permanent target, and White's next central push comes with tempo.",
    "Q_retreat": "The queen has done her work in the centre; she steps back before Black gains any more time. Good habit: centralise while it's free, retreat the moment it starts costing tempi.",
    "Qe2_pin": "A modest-looking move loaded with venom: the queen takes the e-file, and careless natural replies run into discovered checks — in some lines even an immediate mate on d6. Learn the *point*, not just the move.",
    "Qxf3_pair": "Recaptures with the queen, keeping the pawn structure intact and banking the bishop pair for the long game.",
    "Bc4_f7": "The Italian bishop: it aims at f7, the one square only the king defends — especially painful when Black's king is stuck in the centre.",
    "Be3_challenge": "Reinforces the strong d4-point and asks Black's dark-squared bishop a question — any exchange strengthens White's centre or opens the f-file for the rook.",
    "Be2_calm": "Unpretentious and strong: the bishop blocks checks and pins without creating a single weakness, and White simply castles next.",
    "Bb5_check": "An in-between check that disrupts Black's coordination before White carries on with the main plan — checks are free tempo when they force a concession.",
    "d3_solid": "Modest but poisonous: it props up e4, blunts the counterplay, and quietly asks the advanced Black pieces what they are actually doing.",
    "d5_clamp": "Pushes past with gain of space and tempo — Black's knight is driven offside or into tactics that favour the better-developed side.",
    "e5_space": "Gains space and time by kicking Black's pieces. The cramping pawn on e5 cuts Black's position in two and fixes a long-term advantage.",
    "exd5_simplify": "The clean solution: capture, simplify, and let Black spend time recapturing while White develops.",
    "dxe5_clear": "Resolves the central tension at exactly the right moment: Black's natural recaptures walk into the queen-trade endgame that this repertoire is built to reach.",
    "accept_gambit": "1...f5 gives up a pawn for almost nothing and fatally weakens the king. Take it — the refutation is forcing, and it starts with keeping the material.",
    "king_hunt": "Part of a forced refutation: a check, a capture or a direct threat that drags the Black king further up the board. Move order matters — follow the sequence exactly.",
    "fred_build": "A building move inside the refutation: it keeps the extra material, brings another piece toward the attack, and tightens the net before the next blow.",
    "alapin_c3": "The Alapin answer to the Sicilian: prepare d4 so that after ...cxd4 White recaptures with the c-pawn and keeps the full e4+d4 centre — the very thing 1...c5 was meant to prevent.",
    "cxd4_rebuild": "Recaptures with the c-pawn — the whole point of 2.c3. White owns the unchallenged pawn duo and Black's opening promise is gone.",
    "c3_tempo": "Blocks the check and puts the question to the bishop with tempo — it must retreat (losing time) or trade itself off (conceding the bishop pair).",
    "c3_solid": "A quiet strongpoint move: it takes b4 and d4 away from Black's pieces, supports the centre, and prepares to push the enemy pieces back at leisure.",
    "h3_question": "Puts the question to the annoying bishop or pawn immediately: retreat (and lose the sting) or trade (and hand over the bishop pair).",
    "Ng3_kick": "Gains a tempo on the bishop while rerouting the knight toward the kingside squares (f5, h5) it dreams about.",
    "N_reroute": "A flexible retreat: the knight avoids being traded or shut out and will re-emerge on a better square — losing a little time now to stand better later.",
    # ---- Black ----
    "blk_e5_open": "The classical reply: Black stakes an equal claim in the centre. Both sides now fight for the initiative in an open game — exactly where the Scotch strike works best.",
    "blk_scandi": "The Scandinavian: Black challenges e4 at once, accepting an early queen excursion after the capture on d5 — time White will collect with developing moves.",
    "blk_french": "The French: solid but a little passive. Black concedes space, plans ...d5 counterplay, and lives with the problem c8-bishop. The exchange line keeps things simple and slightly nicer for White.",
    "blk_caro": "The Caro-Kann: like the French but with the light-squared bishop's diagonal kept open. Solid and low-risk — and slow, which lets White grab space.",
    "blk_sicilian": "The Sicilian: Black fights for d4 with a wing pawn and aims for unbalanced play. The repertoire's 2.c3 sidesteps the mountain of open-Sicilian theory.",
    "blk_pirc": "A Pirc/Philidor setup: Black concedes the centre for now, intending to chip at it with pieces and pawn breaks later. White simply takes the centre offered.",
    "blk_alekhine": "Alekhine's Defence: Black invites the pawns forward hoping they overextend. White gains space and time with e5 and d4 — and stays sensible rather than greedy.",
    "blk_nimzo": "The Nimzowitsch Defence — a rare bird. After 2.Nf3 e5 3.d4 the game usually transposes straight back into the Scotch, so nothing new to learn.",
    "blk_modern": "The Modern: Black fianchettoes and lets White build. The answer is principled — take the full centre and develop naturally.",
    "blk_owen": "Owen's Defence: ...b6 lets White build the complete pawn centre for free. Develop, castle, and enjoy more of everything.",
    "blk_fred": "The Fred (1...f5?!): a gambit with far more surprise value than substance — it opens the very diagonal the Black king lives on.",
    "blk_Nc6_dev": "The main move: develop and defend e5. Now comes the repertoire's signature strike, 3.d4.",
    "blk_petroff": "The Petroff: instead of defending e5, Black counter-attacks e4. Symmetric and solid — the repertoire meets it with the critical 3.Nxe5.",
    "blk_petroff_main": "The correct move-order move: Black must kick the knight *before* grabbing e4 (the immediate 3...Nxe4 runs into the Qe2 tricks).",
    "blk_philidor": "Philidor-style: ...d6 defends e5 but blocks the f8-bishop. White exchanges in the centre and steers for the pleasant queen-trade endgame.",
    "blk_Bc5_early": "Actively posted on the a7–g1 diagonal, eyeing d4 and f2 — the most natural square for the bishop, and the repertoire meets it with concrete play in the centre.",
    "blk_damiano": "Defending the pawn with 2...f6? is Damiano's old mistake: the f-pawn is the worst defender because it opens the e8–h5 diagonal toward the king. A sharp refutation exists; the repertoire punishes with simple strong development.",
    "blk_early_queen": "The queen comes out early, eyeing f2/d4 and quick mates. The cure is always the same: develop with threats and collect the tempi she leaves behind.",
    "blk_elephant": "The Elephant Gambit: a central lunge that concrete play simply refutes — take the pawn and develop.",
    "blk_exd4": "Black releases the central tension — the main line. White recaptures with a piece and keeps lasting central superiority.",
    "blk_Nxd4_trade": "Trading knights looks natural but is exactly what White wants: it removes the only piece that could harass the queen, justifying the strong Qxd4 centralisation.",
    "blk_Nxe5_recapture": "Recaptures centrally, hoping to keep the queens on — but White trades on e5 anyway and reaches the same favourable endgame.",
    "blk_c5_kick": "Kicks the centralised queen and gains queenside space — at a real price: the d5-square and the d-pawn are weak forever.",
    "blk_c5_break": "The thematic pawn break: Black chips at White's centre from the wing. White meets it calmly, keeping the chain intact rather than grabbing pawns.",
    "blk_cxd4_release": "Trades in the centre — but this is the Alapin's point: after cxd4 White owns the pawn duo the Sicilian was designed to prevent.",
    "blk_Nf6_hit": "Develops and hits e4 — the most natural move on the board. White must mind the e4-pawn while continuing the plan.",
    "blk_d6_solid": "Solid: it shores up the centre and frees the c8-bishop, at the cost of a passive bishop on f8 and no immediate counterplay.",
    "blk_f6_weak": "Propping up the centre with the f-pawn — structurally the worst defender: it weakens the king's diagonal and develops nothing.",
    "blk_check_b4": "A disruptive check — but after the calm pawn block the bishop must decide what it's doing, and White gains the time back with interest.",
    "blk_scandi_queen": "Early queen wandering: every check and retreat costs a tempo, and White cashes each one in as a developing move.",
    "blk_Nxe4_grab": "Grabs the pawn back Petroff-style — but the e-file is exactly where White's tricks live. Precision required from Black, not from White.",
    "blk_stafford": "The Stafford Gambit — an internet favourite bristling with traps (...Bc5, ...Ng4, ...Qd4, ...h5). Objectively unsound: White consolidates with a few precise, quiet moves and keeps the pawn.",
    "blk_Qe7_petroff": "Pins the knight against White's king — met calmly: White breaks the pin with a developing move and keeps the sounder structure.",
    "blk_h5_lash": "The trademark Stafford lunge: the rook's pawn charges to threaten ...Ng4 mating tricks. It looks terrifying and isn't — with the right setup White defuses it and stays a pawn up.",
    "blk_recapture_d8": "A forced recapture, and the choice matters: ...Kxd8 keeps pieces active but loses castling forever; ...Nxd8 keeps the king flexible but buries the knight. White is comfortable either way.",
    "blk_recapture_c6": "The recapture fixes Black's structure for the whole game: doubled c-pawns that can be blockaded and attacked — and in the gambit lines White is a pawn up on top.",
    "blk_recapture_symm": "The symmetric recapture: the structure is level, so the game will be decided purely by piece activity — which is precisely White's plan.",
    "blk_d5_stake": "Black stakes a claim in the centre — the defining move of this defence, and the moment White's setup choice matters.",
    "blk_d4_push": "Black pushes past to shut pieces out and grab space — but every advanced pawn needs defenders, and White reroutes to attack it later.",
    "blk_release": "Releases the central tension and hands White the freer game — the recapture centralises a White piece with gain of time.",
    "blk_bishop_active": "The bishop gets out *before* ...e6 closes the door — the whole point of Black's move order, and the piece White's plan takes aim at.",
    "blk_bishop_pin": "Pins the knight to fight indirectly for the central squares — pressure White should question sooner rather than later.",
    "blk_trade_pair": "Concedes the bishop pair to avoid losing time with the bishop — a small but permanent concession White banks for the endgame.",
    "blk_Be7_solid": "Breaks the pin and completes development — necessary, but it confirms Black is reacting to White's ideas rather than executing their own.",
    "blk_N_dance": "The knight dances to stay alive — each hop lets White build the centre or develop with tempo. Two knight moves for one developing move is a bad trade.",
    "blk_Nd4_lunge": "The knight lunges into d4 looking active — but the tactics favour the better-developed side, and White has a concrete answer ready.",
    "blk_fred_defence": "Black flails for compensation that never existed — each try meets a forcing reply, and the king walks further into the storm.",
}

SCOTCH_LOGIC: Dict[str, str] = {
    "e4": "The repertoire's identity: open games, fast development, and an early central strike. 1.e4 aims for positions where knowing *why* beats memorising *what*.",
    "Nf3_dev": "Step two of the plan: develop with a threat, provoke a committal defence of e5, and only then strike with d4.",
    "d4_strike": "This is the Scotch. Instead of slow Italian manoeuvring, White forces matters in the centre on move three, taking the initiative while Black's pieces are still at home.",
    "d4_center": "Against defences that hand over the centre, the repertoire takes it — two pawns abreast, pieces behind them, and space to attack later.",
    "Nxd4_recapture": "The Scotch tabiya: a centralised knight, open c1- and d1-diagonals... every White piece gets a good square out of this structure.",
    "Qxd4_central": "The repertoire's pet line. After the knight trade the classic punishment for an early queen (a tempo-gaining ...Nc6) no longer exists, so White gets a dream setup for free: queen on d4, Nc3, Bg5 and O-O-O.",
    "Nc3_dev": "Fits the castling-long plan: the knight guards e4, watches d5, and clears the way for Qd4/Bg5/O-O-O to land in quick succession.",
    "Nc3_tempo": "The Scandinavian's tax: White develops the knight *and* Black must move the queen again — a free move, every time.",
    "Bg5_pin": "The third piece of the Qxd4 machine: the pin ties Black's best defender to the board while White finishes the plan with O-O-O.",
    "OOO_pressure": "The payoff move of the whole setup: king safe, rook active on d1 the very same tempo — no other castling gains this much time.",
    "queen_trade_edge": "A signature structure of this repertoire: against ...d6 systems, trade queens on d8 and grind — Black's misplaced king and the weak e5-pawn outlast the symmetry.",
    "Nxe5_tactic": "The concrete justification of the queen-trade line: the e5-pawn falls because Black's recapture decentralised its defender.",
    "petroff_Nxe5": "The repertoire's anti-Petroff weapon: accept the symmetry challenge, take e5, and rely on the e-file tricks (Qe2!) that Black must know precisely — most club players don't.",
    "N_retreat": "Part of the anti-Petroff package: with the pawn count restored, White retreats, develops normally, and keeps a comfortable, slightly freer game.",
    "Nxc6_keep_pawn": "The anti-Stafford recipe in action: trade down, stay a pawn up, and let the attacker's compensation evaporate piece by piece.",
    "Nxc6_structure": "The Mieses idea: fix the doubled c-pawns first, then push e5 with tempo — Black's structure and knight are both worse afterwards.",
    "Q_retreat": "Discipline is part of the system: the centralised queen already earned its keep (development lead, structural gains); retreating on your own terms keeps every gain and gives back nothing.",
    "Qe2_pin": "One of the repertoire's teeth: in both the Petroff and the two-knights Caro-Kann this quiet queen move wins games outright against natural-looking replies (Nc6+ forking king and queen, or even Nd6 mate).",
    "Qxf3_pair": "Consistent with the repertoire's style: accept small permanent pluses (bishop pair, better structure) and let them grow.",
    "Bc4_f7": "The classical attacking diagonal — in this repertoire it appears exactly when Black's king has lost the right (or the time) to castle.",
    "Be3_challenge": "The standard Scotch answer to ...Bc5: defend the d4-point in a way that welcomes exchanges — each trade leaves White's centre and files better.",
    "Be2_calm": "The repertoire's answer to early aggression: no weakening, no drama — a developing move that says 'your threats are spent, now I play'.",
    "Bb5_check": "A little tactical seasoning: the check forces a block or a king move before Black is ready, improving White's version of the coming structure.",
    "d3_solid": "The anti-trick move: it takes e4-squares away from Black's pieces and stabilises everything — the quiet backbone of the anti-Petroff and anti-Stafford lines.",
    "d5_clamp": "When Black delays ...exd4, the repertoire punishes with the space-gaining push — the same lesson as everywhere: central tension resolves in White's favour.",
    "e5_space": "The advance structure: fix the pawn chain, claim the kingside, and manoeuvre behind it (the Short-System treatment against the Caro-Kann).",
    "exd5_simplify": "Repertoire philosophy: against counter-gambits and symmetry attempts, take what's offered, simplify, and out-develop.",
    "dxe5_clear": "The gateway to the repertoire's favourite endgame: every recapture leads either to Qxd8+ (stuck king) or to Nxe5 (won pawn).",
    "accept_gambit": "Same philosophy as everywhere in this repertoire: refute by development and material, not by counter-speculation.",
    "king_hunt": "The one truly forced line in the repertoire: the Fred refutation is a sequence of checks and threats worth learning move by move — it wins on the spot.",
    "fred_build": "Even inside a king hunt there are quiet moves: keep the material, add attackers, take squares from the king — then resume the checks.",
    "alapin_c3": "The repertoire's anti-Sicilian: skip the theory war, restore the classical centre, and play chess. Black's most principled tries all concede the pawn duo.",
    "cxd4_rebuild": "Mission accomplished: e4 + d4 stand, the c-file is half-open for the rook, and White plays a pleasant IQP-free version of the open games.",
    "c3_tempo": "The repertoire's standard answer to bishop checks: block with a pawn move that also gains time and takes squares.",
    "c3_solid": "The consolidation move of the anti-gambit lines: it takes d4 and b4 from Black's pieces and prepares to push the attackers back — the beginning of the end for the compensation.",
    "h3_question": "Part of the consolidation recipe: question the aggressive piece at the exact moment the tactics don't work for Black.",
    "Ng3_kick": "Typical of the two-knights treatment: gain time on the bishop and point the knight at the kingside.",
    "N_reroute": "System thinking: pieces go where the structure wants them, even if it takes an extra move — the closed centre gives White the time.",
    "OO_safe": "When the position is quiet the repertoire castles short and plays chess — not every line needs the O-O-O sword-fight.",
    "blk_e5_open": "Exactly what the repertoire hopes for — the Scotch strike is prepared and White knows these structures better.",
    "blk_scandi": "The repertoire's treatment: accept, develop with tempo on the queen, and meet the awkward checks with the cold-blooded Be2.",
    "blk_french": "The exchange keeps it simple: symmetric pawns, freer pieces, and none of the closed-centre theory Black hoped to show off.",
    "blk_caro": "Two good options are covered: the space-gaining Advance with the Short setup, and the trick-laden two-knights line with Qe2 ideas.",
    "blk_sicilian": "2.c3 turns the Sicilian into a fight about the centre instead of a memory contest — home ground for this repertoire.",
    "blk_pirc": "White takes the full centre and develops classically — the repertoire refuses to chase shadows on the wings.",
    "blk_alekhine": "Take the space, decline the provocation: e5 and d4, sensible development, no overextension.",
    "blk_petroff": "The repertoire's answer is the most testing one — and its traps (Qe2!) score heavily at club level.",
    "blk_stafford": "The repertoire includes the exact antidote setup — d3, Be2, c3 (and h3 at the right moment) — that neutralises every trap while keeping the pawn.",
    "blk_philidor": "The queen-trade endgame is the repertoire's answer: a small, permanent, risk-free edge — precisely the kind of position gambit-and-trap players hate defending.",
    "blk_damiano": "The repertoire punishes with development and the f7-diagonal rather than memorised sacrifices — winning without risk.",
    "blk_fred": "The repertoire carries the full refutation: accept, develop, then the forced king hunt. Worth one focused study session.",
    "blk_h5_lash": "This is the move the antidote setup exists for — with d3/Be2/c3 played, ...h5 is a bluff White has already called.",
}

SCOTCH_PREVENTS: Dict[str, str] = {
    "e4": "Takes d5 and f5 under control before Black can claim them, and denies Black any easy symmetric equaliser except entering White's preparation.",
    "Nf3_dev": "Stops liberating ideas based on ...d5 or holding e5 for free — Black must commit a defender first.",
    "d4_strike": "Denies Black the comfortable, slow development of the Italian/Ruy structures — the centre opens before Black's pieces are ready for it.",
    "Qxd4_central": "The immediate harassers are gone — that's the point. (...c5 or ...Bc5 can still nudge the queen later, which is exactly why the calm Qd1 retreat is part of the repertoire.)",
    "Bg5_pin": "Freezes the f6-knight: no ...Nxe4 tricks, and any ...h6/...g5 attempt to break the pin weakens Black's own king.",
    "OOO_pressure": "Removes the king from the centre before the d-file pressure begins — the same file the rook now owns.",
    "queen_trade_edge": "Denies Black a middlegame: without queens, the counter-attacking chances Black's setup relied on evaporate, while the structural problems remain.",
    "Qe2_pin": "Prevents Black from keeping the extra pawn comfortably: the e-file pin means the e4-knight can't just retreat (…Nf6?? loses the queen to the discovered Nc6+).",
    "alapin_c3": "Prevents the Sicilian's main achievement — a White knight on d4 as a target and an open c-file for Black. Here the c-pawn recaptures instead.",
    "c3_tempo": "Ends the pin/check nuisance on its first move — the bishop achieves nothing and must already explain itself.",
    "c3_solid": "Takes the d4- and b4-squares away from Black's minor pieces — the squares every Stafford/Scandinavian trick depends on.",
    "d3_solid": "Takes e4 away from Black's knight and bishop, cutting off the tactical melodies (…Ng4, …Nxe4-tricks) at the source.",
    "e5_space": "Denies Black's king knight its best square (f6) — a lasting concession you can play against for the whole game.",
    "Be2_calm": "Blocks the checks and pins without offering the queen trade or loosening a single pawn — Black's 'activity' bought nothing.",
    "h3_question": "Prevents the bishop from camping on g4 pinning the knight — and does it before Black can support the pin.",
    "d5_clamp": "Stops Black from maintaining the e5-strongpoint and denies the c6-knight its best square in one push.",
    "king_hunt": "Each check denies the king a breathing move — one slow move and Black would consolidate, so the sequence prevents exactly that.",
    "exd5_simplify": "Prevents Black from getting the open, active game the gambit promised — the extra tempo goes to White instead.",
}

SCOTCH_PLANS: Dict[str, List[str]] = {
    "e4": [
        "Develop knights before bishops — Nf3 hitting e5 comes first.",
        "Strike in the centre with d4 as early as possible (the Scotch idea).",
        "Castle quickly: short in the quiet lines, long in the Qxd4 lines.",
    ],
    "Nf3_dev": [
        "Play d4 next and open the centre while Black is committed to e5.",
        "Meet 2...Nf6 (Petroff) with the critical 3.Nxe5.",
        "Against 2...d6, exchange in the centre and aim for the queen-trade endgame.",
    ],
    "d4_strike": [
        "Recapture on d4 with the knight — it belongs in the centre.",
        "If knights come off on d4, retake with the queen and build Nc3/Bg5/O-O-O.",
        "Develop fast and keep Black reacting; the initiative is the point.",
    ],
    "d4_center": [
        "Keep the duo intact: c3, Nf3/Nc3 and natural piece play behind the pawns.",
        "Meet the pawn breaks (...c5, ...d5, ...f6) calmly — support, don't panic-trade.",
        "Castle short and expand on the side where you have more space.",
    ],
    "Qxd4_central": [
        "Develop Nc3 and Bg5 around the queen — she's a strongpoint, not a target.",
        "Castle long: the rook lands on d1 with tempo against d6/d7.",
        "If ...c5 arrives, retreat Qd1 calmly — the lead in development remains.",
    ],
    "Bg5_pin": [
        "Castle long next — O-O-O puts the rook on the d-file in one move.",
        "Keep the pin while it annoys; only take on f6 for a concrete reason.",
        "Centralise the rooks (Rd1/Rhe1) and probe the d6-pawn.",
    ],
    "OOO_pressure": [
        "Pile on the d-file: Rd1, sometimes doubling, against d6/d7.",
        "Your kingside pawns are free to advance — h3/g4 gains space without risking your king.",
        "Trade into favourable endgames; these positions convert themselves slowly.",
    ],
    "queen_trade_edge": [
        "Exploit the stuck king: Bc4 eyes f7, rooks come to d1 fast.",
        "Win or fix the e5-pawn (Nxe5 tricks when the defenders are gone).",
        "Trade pieces, not pawns — every exchange magnifies the endgame edge.",
    ],
    "petroff_Nxe5": [
        "Against ...Nxe4, play Qe2 and win the e-file battle (mind the Nc6+ trick).",
        "Against ...d6 first, retreat Nf3, develop and keep the freer game.",
        "Against ...Nc6 (Stafford), take, then consolidate: d3, Be2, c3, h3 at the right time.",
    ],
    "alapin_c3": [
        "Play d4 next and recapture with the c-pawn to keep the duo.",
        "Develop Bd3/Bc4 and Nf3, castle short, and use the space edge.",
        "Meet ...d5 with exd5 and play against the isolated or symmetric structure.",
    ],
    "e5_space": [
        "Support the spearhead: Nf3, Be2, c3 — the Short System shape.",
        "Answer ...c5 with Be3/c3, keeping the d4-point rock solid.",
        "Manoeuvre behind the chain: knights dream of d4, f4 and g5.",
    ],
    "accept_gambit": [
        "Keep the pawn and develop toward the kingside (d4, Bd3, Nf3).",
        "When the checks start, follow the forced sequence exactly.",
        "If Black deviates, just finish development — you are simply winning.",
    ],
    "Nc3_tempo": [
        "Collect the tempi: every queen move is answered by a developing move.",
        "Play d4 and natural piece moves; castle short.",
        "Punish early aggression with calm blocks (Be2!) rather than pawn weaknesses.",
    ],
    "queen_hunt_generic": [],
}

SCOTCH_HOOK: Dict[str, str] = {
    "e4": "Open games on purpose: centre, development, initiative.",
    "Nf3_dev": "Develop with a threat — make them defend, not develop.",
    "d4_strike": "Scotch = strike with d4 before they're ready.",
    "d4_center": "Two pawns abreast on e4+d4 — the dream centre, take it when offered.",
    "Nxd4_recapture": "Recapture toward the centre.",
    "Qxd4_central": "Qxd4 is safe here — the harasser (that c6-knight) is gone.",
    "Nc3_dev": "Knight out, e4 covered, long castle coming.",
    "Nc3_tempo": "Develop by hitting the queen — free moves add up.",
    "Bg5_pin": "Bg5 then O-O-O: pin first, castle into the d-file second.",
    "OOO_pressure": "Castling long = king safe + rook on d1, same tempo.",
    "OO_safe": "Quiet position? Castle short and just play chess.",
    "queen_trade_edge": "Qxd8+ ruins their day: their king gets stuck, yours doesn't.",
    "Nxe5_tactic": "When the defenders are traded, e5 just falls — count and take.",
    "petroff_Nxe5": "Petroff? Take e5 — and remember the Qe2 trick.",
    "Qe2_pin": "Qe2: the quiet move that wins queens (Nc6+!) — or mates on d6.",
    "N_retreat": "Grab, give back nothing, retreat with profit.",
    "Nxc6_keep_pawn": "Stafford antidote step one: trade pieces, keep the pawn.",
    "Nxc6_structure": "Double their c-pawns, then hit the knight with e5.",
    "Q_retreat": "Centralise while it's free, retreat (Qd1) the moment it's taxed.",
    "Bc4_f7": "Bishop c4 stares at f7 — the king's sore spot.",
    "Be3_challenge": "Meet ...Bc5 with Be3: every trade helps White.",
    "Be2_calm": "Early checks? Be2 — block, develop, shrug.",
    "d3_solid": "d3 first — no e4-square, no tricks, no Stafford magic.",
    "d5_clamp": "They delay ...exd4? Push d5 and take the space.",
    "e5_space": "Push e5: cramp them now, manoeuvre later.",
    "exd5_simplify": "Take on d5 and develop while they recapture.",
    "dxe5_clear": "Trade d for e — then Qxd8+ and grind.",
    "accept_gambit": "1...f5? Take it. Then follow the checklist to the king.",
    "king_hunt": "In the Fred: if it's not a check or a threat, it's not the move.",
    "fred_build": "Between the checks: add a piece, tighten the net.",
    "alapin_c3": "vs the Sicilian: c3 then d4 — rebuild the centre they tried to ban.",
    "cxd4_rebuild": "Recapture with the c-pawn — that was the whole point of 2.c3.",
    "c3_tempo": "Bishop checks? c3 — block with tempo.",
    "c3_solid": "c3: take away b4 and d4, then push them back.",
    "h3_question": "Ask the bishop the question while the answer hurts.",
    "blk_recapture_d8": "Kxd8 loses castling, Nxd8 buries the horse — smile either way.",
    "blk_stafford": "Stafford = traps; d3–Be2–c3 defuses them all.",
    "blk_h5_lash": "...h5 looks scary; with the setup played, it's a bluff.",
    "blk_exd4": "They released the tension — the centre is yours now.",
}

SCOTCH_DIFFICULTY: Dict[str, int] = {
    "e4": 1, "Nf3_dev": 1, "d4_strike": 2, "d4_center": 1,
    "Nxd4_recapture": 1, "Nxe4_recapture": 1, "Qxd4_central": 3,
    "Nc3_dev": 1, "Nc3_tempo": 2, "Bg5_pin": 2, "OOO_pressure": 3,
    "OO_safe": 1, "queen_trade_edge": 3, "Nxe5_tactic": 3,
    "petroff_Nxe5": 3, "Qe2_pin": 4, "N_retreat": 2,
    "Nxc6_keep_pawn": 2, "Nxc6_structure": 3, "Q_retreat": 2,
    "Qxf3_pair": 2, "Bc4_f7": 2, "Be3_challenge": 2, "Be2_calm": 2,
    "Bb5_check": 3, "d3_solid": 2, "d5_clamp": 3, "e5_space": 2,
    "exd5_simplify": 1, "dxe5_clear": 3, "accept_gambit": 3,
    "king_hunt": 5, "fred_build": 4, "alapin_c3": 2, "cxd4_rebuild": 1,
    "c3_tempo": 1, "c3_solid": 2, "h3_question": 2, "Ng3_kick": 2,
    "N_reroute": 2,
}

SCOTCH_MISTAKE: Dict[str, str] = {
    "e4": "Don't drift into 'hope chess' after 1.e4 — the repertoire's edge comes from knowing the plan against each defence, not from the first move itself.",
    "d4_strike": "Don't prepare d4 slowly with c3 here (that's a different opening) — the strike works *because* it comes before Black is developed.",
    "Qxd4_central": "Two opposite errors: avoiding 5.Qxd4 because 'the queen shouldn't come out early' (the rule's reason — tempo-gaining attacks — is absent here), and leaving her centralised once ...c5/...Bc5 arrive with real tempo. Centralise free, retreat taxed.",
    "Bg5_pin": "Don't cash the pin with Bxf6 without a concrete reason — the pin is worth more than the bishop pair here.",
    "OOO_pressure": "Castling long means pawn storms can matter — glance at Black's ...b5-b4 potential before committing. In these lines White's centre and activity outrun it, but always look once.",
    "queen_trade_edge": "Don't assume queen trades are drawish — this one wins time and castling rights. But count the captures on e5 carefully; the pawn-win tactic only works in the right order.",
    "Nxe5_tactic": "Grabbing e5 when the tactics *don't* work — always count defenders first. The pattern only works after the right pieces have been traded.",
    "petroff_Nxe5": "Don't 'win' material with the tricks in the wrong order — and remember it's *Black* who blunders with 4...Nf6?? 5.Nc6+. White's moves are natural if you know the one idea.",
    "Qe2_pin": "Knowing Qe2 without knowing why: if you can't name the threat (discovered check on the e-file / Nd6 ideas), review the line before playing it — the move only bites with follow-up.",
    "accept_gambit": "Half-remembering the Fred refutation and improvising in the middle — if unsure, keep the pawn, develop simply and castle; you don't need the most brutal line to be winning.",
    "king_hunt": "Inserting a 'safe' developing move in the middle of the forced sequence — that's the one way to let the king escape. Learn it as one chunk.",
    "alapin_c3": "Recapturing on d4 with a piece — the c-pawn recapture *is* the point of 2.c3. Also don't skip d4: without it, c3 was just a slow move.",
    "e5_space": "Overextending: pushing pawns beyond what your pieces support hands Black exactly the targets the defence was fishing for.",
    "blk_stafford": "Against the Stafford, greed and 'natural' moves lose games: don't grab the ...Ng4 bait, don't play h3 at the wrong moment. The setup d3, Be2, c3 (then h3) defuses everything.",
    "blk_h5_lash": "Panicking against ...h5 — or ignoring it entirely. Check the ...Ng4 tricks once, play the prophylactic move the position asks for, and keep the extra pawn.",
    "blk_recapture_d8": "Relaxing because 'it's just an endgame' — the edge is concrete (stuck king, weak e5) and evaporates if you trade lazily or let the king reconnect the rooks for free.",
}

SCOTCH_BOOK = OpeningBook(
    key="scotch",
    name="Scotch Game",
    logic_label="Scotch repertoire logic",
    classify=_classify_scotch,
    why=SCOTCH_WHY,
    logic=SCOTCH_LOGIC,
    prevents=SCOTCH_PREVENTS,
    plans=SCOTCH_PLANS,
    hook=SCOTCH_HOOK,
    difficulty=SCOTCH_DIFFICULTY,
    mistake=SCOTCH_MISTAKE,
    default_logic=(
        "This move follows the repertoire's guiding ideas: open the centre early "
        "(d4!), develop with threats, centralise — even the queen, when she cannot "
        "be harassed — and steer for structures Black must defend precisely: "
        "queen-trade endgames, doubled pawns, kings stuck in the centre."
    ),
    default_plans=[
        "Finish development quickly — every piece toward the centre.",
        "Keep or increase the central superiority; meet wing play with central play.",
        "Choose the right castling: short in quiet lines, long in the Qxd4 setup.",
    ],
    root_summary=(
        "The initial position. White opens 1.e4, heading for the Scotch after "
        "1...e5 2.Nf3 Nc6 3.d4 — and has a prepared answer to every other defence."
    ),
    root_logic=(
        "The plan: 1.e4, Nf3, d4 — open the centre fast, develop with threats, "
        "and centralise. Sidelines are met with simple, structurally sound "
        "solutions (2.c3 vs the Sicilian, the Advance vs the Caro-Kann, the "
        "exchange vs the French)."
    ),
    root_plans=[
        "Play 1.e4 and meet 1...e5 with Nf3 and the d4 strike (the Scotch).",
        "Against the Sicilian: 2.c3 and rebuild the full pawn centre.",
        "Against everything else: take the centre with d4 and develop naturally.",
    ],
    root_hook="e4 → Nf3 → d4: open it up.",
    llm_persona=(
        "You are a friendly, practical chess coach who specialises in a complete "
        "1.e4 repertoire for White built around the Scotch Game (1.e4 e5 2.Nf3 "
        "Nc6 3.d4), including: the Qxd4 centralisation lines with Nc3, Bg5 and "
        "O-O-O; the queen-trade endgames against ...d6 setups; the critical "
        "3.Nxe5 Petroff with the Qe2 e-file tricks and the Stafford Gambit "
        "antidote (d3, Be2, c3); the 2.c3 Alapin against the Sicilian; the "
        "Advance Caro-Kann with the Short System; the Exchange French; and "
        "sound refutations of early gambits."
    ),
)


# --------------------------------------------------------------------------- #
# Jobava London book (dictionaries defined above) + generic fallback
# --------------------------------------------------------------------------- #
JOBAVA_BOOK = OpeningBook(
    key="jobava",
    name="Jobava London",
    logic_label="Jobava London logic",
    classify=_classify_jobava,
    why=THEME_WHY,
    logic=THEME_JOBAVA,
    prevents=THEME_PREVENTS,
    plans=THEME_PLANS,
    hook=THEME_HOOK,
    difficulty=THEME_DIFFICULTY,
    mistake=THEME_MISTAKE,
    default_logic=(
        "This move supports the core Jobava London plan: rapid development "
        "with Nc3 and Bf4, pressure on c7/d5, and the option to break with "
        "e4 or launch the h-pawn depending on where Black puts the king."
    ),
    default_plans=[
        "Finish developing: e3, a knight to f3 or e2, the light-squared bishop, then castle.",
        "Look for the e4 break to open the centre once your pieces are ready.",
        "Watch for Nb5 ideas hitting c7 and d6 if Black neglects that square.",
    ],
    root_summary="The initial position. White is about to begin the Jobava London.",
    root_logic="The plan: 1.d4, then Nc3 and Bf4 — active, attacking London chess.",
    root_plans=["Play 1.d4 and aim for the Nc3 + Bf4 setup."],
    root_hook="d4, Nc3, Bf4 — the Jobava skeleton.",
    llm_persona=(
        "You are a friendly, practical chess coach who specialises in the "
        "Jobava London System (1.d4 followed by Nc3 and Bf4)."
    ),
)

GENERIC_BOOK = OpeningBook(
    key="generic",
    name="repertoire",
    logic_label="Opening logic",
    classify=lambda node, board: "generic",
    default_logic=(
        "This move follows sound opening principles: develop toward the centre, "
        "fight for the central squares, keep the king safe, and start concrete "
        "action only once development is complete."
    ),
    default_plans=[
        "Finish development and castle.",
        "Prepare the right pawn break to challenge the centre.",
        "Improve your worst-placed piece before starting an attack.",
    ],
    root_summary="The initial position of the loaded repertoire.",
    root_logic="Follow the repertoire's moves and study the ideas behind each branch.",
    root_plans=["Follow the mainline first, then explore the sidelines."],
    root_hook="Plans over moves: understand each branch before memorising it.",
    llm_persona=(
        "You are a friendly, practical chess coach explaining an opening "
        "repertoire from a PGN file."
    ),
)


def detect_book(tree: RepertoireTree) -> OpeningBook:
    """Pick the knowledge base that matches the loaded repertoire.

    Detection is deliberately simple: the mainline first move decides
    (1.d4 → Jobava London flavour, 1.e4 → Scotch flavour). Anything else gets
    the neutral principle-based book. The family texts are written generally
    enough that related repertoires still get sensible guidance.
    """
    try:
        kids = tree.children(tree.root_id)
    except Exception:
        return GENERIC_BOOK
    first = kids[0].move_san if kids else None
    if first == "d4":
        return JOBAVA_BOOK
    if first == "e4":
        return SCOTCH_BOOK
    return GENERIC_BOOK


__all__ = [
    "OpeningBook",
    "JOBAVA_BOOK",
    "SCOTCH_BOOK",
    "GENERIC_BOOK",
    "detect_book",
]
