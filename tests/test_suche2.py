from pathlib import Path
import time

import chess
import pytest

from Code.rl.suche2 import schnellsuche
from Code.rl.search import search, SearchOptions, CLASSIC
from Code.rl.valuenet import ValueNet


OPTS = [SearchOptions(cap=0.9, repetition_draw=True), CLASSIC]

STELLUNGEN = [
    (
        "Startstellung",
        chess.STARTING_FEN,
    ),
    (
        "Spanische Partie",
        "r1bqk1nr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ),
    (
        "Italienisch",
        "r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    ),
    (
        "Damengambit",
        "rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2",
    ),
    (
        "Mittelspiel mit Schlagzug eins",
        "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7",
    ),
    (
        "Mittelspiel mit Schlagzug zwei",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10",
    ),
    (
        "Mittelspiel mit Schlagzug drei",
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
    ),
    (
        "Endspiel mit Bauern",
        "8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 1",
    ),
    (
        "Endspiel mit Freibauer",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    ),
    (
        "Matt in eins",
        "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1",
    ),
    (
        "Matt in zwei",
        "7k/8/5K2/8/8/8/8/6R1 w - - 0 1",
    ),
    (
        "Patt-Falle",
        "7k/5Q2/8/8/8/8/8/K7 w - - 0 1",
    ),
]

MITTELSPIELE = [
    "r1bq1rk1/ppp2ppp/2n2n2/3pp3/1bPP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 7",
    "r2q1rk1/pp2bppp/2n1pn2/3p4/3P1B2/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10",
    "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQK2R b KQkq - 0 5",
]

TIEFE3_NAMEN = {
    "Endspiel mit Bauern",
    "Endspiel mit Freibauer",
    "Matt in eins",
    "Matt in zwei",
}

VERGLEICHSFAELLE = [
    (opts, fen, depth)
    for opts in OPTS
    for name, fen in STELLUNGEN
    for depth in (1, 2, 3)
    if depth < 3 or name in TIEFE3_NAMEN
]


@pytest.fixture(scope="module")
def netz() -> ValueNet:
    return ValueNet.load(Path(__file__).parent.parent / "models/full_v9.npz")


def _vergleichs_id(fall) -> str:
    opts, fen, depth = fall
    options_name = "cap-und-wiederholung" if opts is OPTS[0] else "classic"
    name = next(name for name, stellungs_fen in STELLUNGEN if stellungs_fen == fen)
    return f"{options_name}-{name}-tiefe-{depth}"


@pytest.mark.parametrize(
    "opts, fen, depth",
    VERGLEICHSFAELLE,
    ids=[_vergleichs_id(fall) for fall in VERGLEICHSFAELLE],
)
def test_gleichwertig_ohne_abkuerzungen(netz, opts, fen, depth):
    board = chess.Board(fen)
    assert board.is_valid(), fen

    schnell = schnellsuche(
        board,
        netz,
        depth,
        options=opts,
        tt=False,
        reduktionen=False,
    )
    referenz = search(board.copy(), netz, depth, options=opts)

    assert abs(schnell.value - referenz.value) < 1e-5
    assert schnell.move is not None
    assert board.is_legal(schnell.move)


@pytest.mark.parametrize(
    "opts, fen, depth",
    VERGLEICHSFAELLE,
    ids=[_vergleichs_id(fall) for fall in VERGLEICHSFAELLE],
)
def test_gleichwertig_mit_tt(netz, opts, fen, depth):
    board = chess.Board(fen)
    assert board.is_valid(), fen

    schnell = schnellsuche(
        board,
        netz,
        depth,
        options=opts,
        tt=True,
        reduktionen=False,
    )
    referenz = search(board.copy(), netz, depth, options=opts)

    assert abs(schnell.value - referenz.value) < 1e-5
    assert schnell.move is not None
    assert board.is_legal(schnell.move)


@pytest.mark.parametrize(
    "opts",
    OPTS,
    ids=["cap-und-wiederholung", "classic"],
)
@pytest.mark.parametrize("depth", [2, 3], ids=["tiefe-2", "tiefe-3"])
def test_matt_in_1(netz, opts, depth):
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1")
    ergebnis = schnellsuche(
        board,
        netz,
        depth,
        options=opts,
        tt=True,
        reduktionen=True,
    )

    assert ergebnis.move is not None
    assert board.is_legal(ergebnis.move)
    nach_dem_zug = board.copy()
    nach_dem_zug.push(ergebnis.move)
    assert nach_dem_zug.is_checkmate()
    assert abs(ergebnis.value - 1.0) < 1e-6


def test_keine_zuege(netz):
    board = chess.Board("R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1")
    assert board.is_valid()
    assert not any(board.legal_moves)

    ergebnis = schnellsuche(board, netz, 3)

    assert ergebnis.move is None
    assert ergebnis.value == 0.0


def test_brett_unveraendert(netz):
    board = chess.Board()
    for san in ("e4", "e5", "Nf3", "Nc6"):
        board.push_san(san)

    vorher_fen = board.fen()
    vorher_zuege = list(board.move_stack)

    schnellsuche(board, netz, 3)

    assert board.fen() == vorher_fen
    assert board.move_stack == vorher_zuege


def test_zeitgrenze(netz):
    board = chess.Board(MITTELSPIELE[0])
    assert board.is_valid()

    start = time.perf_counter()
    ergebnis = schnellsuche(board, netz, 6, zeit=0.5)
    dauer = time.perf_counter() - start

    assert dauer < 2.0
    assert ergebnis.move is not None
    assert board.is_legal(ergebnis.move)


def test_aufwand(netz):
    leaves_schnell = 0
    leaves_referenz = 0

    for fen in MITTELSPIELE:
        board = chess.Board(fen)
        assert board.is_valid()

        schnell = schnellsuche(
            board,
            netz,
            3,
            tt=True,
            reduktionen=True,
        )
        referenz = search(board.copy(), netz, 3)

        leaves_schnell += schnell.leaves
        leaves_referenz += referenz.leaves

    assert leaves_schnell < 0.6 * leaves_referenz
