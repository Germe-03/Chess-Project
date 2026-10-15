"""
Tests für das Vorausrechnen und die Beschleunigung der Bewertung.

Zwei Arten von Prüfungen:

1. Nichts Bestehendes hat sich verändert. Die Referenzdateien in fixtures/
   wurden VOR dem Einbau der Suche aufgenommen (Commit ec4d4c8): die Antworten
   von v1 bis v3 auf 40 Stellungen und Φ auf 611 Stellungen. Verglichen wird
   bitgenau, nicht ungefähr — wer v3 auswählt, soll exakt dieselbe Engine
   bekommen wie vorher.

2. Die Suche rechnet richtig. Alpha-Beta, Killer-Heuristik und das Abbrechen
   über die Ein-Zug-Bewertung sind reine Abkürzungen; sie dürfen das Ergebnis
   nicht verändern. Deshalb wird gegen eine Suche ohne jede Abkürzung geprüft.
"""
import json
import math
from pathlib import Path

import chess
import numpy as np
import pytest

from Code.main import choose_move
from Code.rl import shaping
from Code.rl.engine import score_moves, terminal_value, value_move
from Code.rl.features import FEATURE_SIZE, board_to_features
from Code.rl.search import _Searcher, leaf_of, search
from Code.rl.tdleaf import play_episode_tdleaf
from Code.rl.valuenet import ValueNet

FIXTURES = Path(__file__).parent / "fixtures"
MODELS = Path(__file__).resolve().parents[1] / "models"
GAMMA = 0.99


def _referenz(name: str) -> list[dict]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))["fälle"]


@pytest.fixture(scope="module")
def v3() -> ValueNet:
    return ValueNet.load(MODELS / "full_v3.npz")


def _gesamtwert(board: chess.Board, net: ValueNet) -> float:
    """V + Φ einer laufenden Stellung, so wie die Suche sie an den Blättern nutzt."""
    value = float(net.value(board_to_features(board)[None, :])[0])
    return value + shaping.potential(board) if net.shaped else value


# --------------------------------------------------------------------------- #
# 1. Bestehendes Verhalten bleibt bitgenau erhalten
# --------------------------------------------------------------------------- #
def test_v1_bis_v3_antworten_wie_vorher():
    """Gleicher Zug und gleiche Bewertung für alle 120 aufgenommenen Fälle."""
    for fall in _referenz("referenz_v1_v3.json"):
        auswahl = choose_move(chess.Board(fall["fen"]), fall["version"])
        assert auswahl.depth == 1
        assert auswahl.move.uci() == fall["best_move"], (fall["version"], fall["fen"])
        assert auswahl.evaluation == fall["evaluation"], (fall["version"], fall["fen"])


def test_potential_wie_vorher():
    """Die beschleunigte Fassung von Φ liefert dieselben Werte, Bit für Bit."""
    for fall in _referenz("referenz_potential.json"):
        board = chess.Board(fall["fen"])
        ich = board.turn
        assert shaping.potential(board) == fall["potential"], fall["fen"]
        assert shaping.material(board, ich) == fall["material"]
        assert shaping.center(board, ich) == fall["center"]
        assert shaping.exchange(board, ich) == fall["exchange"]
        assert shaping.best_exchange(board, ich) == fall["best_exchange_ich"]
        assert shaping.best_exchange(board, not ich) == fall["best_exchange_gegner"]
        if not board.is_game_over():
            assert shaping.potential_open(board) == fall["potential"]


def _merkmale_alt(board: chess.Board) -> np.ndarray:
    """Die frühere Kodierung über board.mirror(), als Massstab für die neue."""
    view = board if board.turn == chess.WHITE else board.mirror()
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)
    for square, piece in view.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        x[plane * 64 + square] = 1.0
    x[768] = float(view.has_kingside_castling_rights(chess.WHITE))
    x[769] = float(view.has_queenside_castling_rights(chess.WHITE))
    x[770] = float(view.has_kingside_castling_rights(chess.BLACK))
    x[771] = float(view.has_queenside_castling_rights(chess.BLACK))
    x[772] = float(view.ep_square is not None)
    x[773] = min(view.halfmove_clock / 100.0, 1.0)
    return x


def test_merkmale_wie_vorher():
    rng = np.random.default_rng(3)
    stellungen = [chess.Board(fall["fen"]) for fall in _referenz("referenz_potential.json")]
    stellungen += [
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1"),
        chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 7 1"),
        chess.Board("4k3/8/8/8/3pP3/8/8/4K3 b - e3 0 1"),
    ]
    for _ in range(200):
        board = chess.Board()
        for _ in range(int(rng.integers(0, 60))):
            zuege = list(board.legal_moves)
            if not zuege:
                break
            board.push(zuege[int(rng.integers(len(zuege)))])
        stellungen.append(board)
    for board in stellungen:
        assert np.array_equal(board_to_features(board), _merkmale_alt(board)), board.fen()


# --------------------------------------------------------------------------- #
# 2. Die Suche rechnet richtig
# --------------------------------------------------------------------------- #
def test_tiefe_1_ohne_ruhesuche_entspricht_der_bisherigen_zugwahl(v3):
    """Die Suche ist eine Verallgemeinerung von score_moves, keine zweite Logik."""
    for fall in _referenz("referenz_v1_v3.json")[80:]:          # die v3-Fälle
        board = chess.Board(fall["fen"])
        ergebnis = search(board, v3, 1, GAMMA, quiescence_plies=0)
        _, punkte = score_moves(board, v3, GAMMA)
        assert ergebnis.move == value_move(board, v3, GAMMA)
        assert ergebnis.value == float(punkte.max())


def _minimax(board, net, depth, quiescence_plies):
    """Dieselbe Definition wie die Suche, aber ohne jede Abkürzung."""
    best = -math.inf
    for move in board.legal_moves:
        board.push(move)
        outcome = terminal_value(board)
        if outcome is not None:
            value = outcome
        elif depth == 1:
            value = -GAMMA * _ruhe(board, net, quiescence_plies)
        else:
            value = -GAMMA * _minimax(board, net, depth - 1, quiescence_plies)
        board.pop()
        best = max(best, value)
    return best


def _ruhe(board, net, plies):
    best = _gesamtwert(board, net)
    if plies == 0:
        return best
    for move in list(_Searcher._tactical_moves(board)):
        if shaping.loses_material(board, move):
            continue
        board.push(move)
        outcome = terminal_value(board)
        value = outcome if outcome is not None else -GAMMA * _ruhe(board, net, plies - 1)
        board.pop()
        best = max(best, value)
    return best


@pytest.mark.parametrize(
    "fen, tiefe, ruhe",
    [
        ("4k3/8/3p4/2r1n3/3P4/2N2B2/8/4K3 w - - 0 1", 2, 4),
        ("r3k2r/8/8/3q4/8/2N5/8/R3K2R w KQkq - 0 1", 2, 4),
        ("4k3/8/3p4/4n3/3P4/2N5/8/4K3 b - - 0 1", 3, 2),
        ("6k1/5ppp/8/8/1n6/2Q5/5PPP/6K1 w - - 0 1", 3, 2),
    ],
)
def test_alpha_beta_rechnet_wie_minimax(fen, tiefe, ruhe):
    netz = ValueNet(hidden_size=16, seed=7, shaped=True)
    board = chess.Board(fen)
    ergebnis = search(board, netz, tiefe, GAMMA, quiescence_plies=ruhe)
    assert ergebnis.value == pytest.approx(_minimax(board, netz, tiefe, ruhe), abs=1e-5)


def test_hauptvariante_endet_im_bewerteten_blatt(v3):
    """Suchwert = (-gamma)^k * Gesamtwert des Blatts — die Grundlage für TD-Leaf."""
    for fen in (
        chess.STARTING_FEN,
        "r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8",
        "r3k3/pp3ppp/8/1N6/8/8/PP3PPP/4K3 w - - 0 1",
    ):
        board = chess.Board(fen)
        ergebnis = search(board, v3, 2, GAMMA)
        blatt = leaf_of(board, ergebnis.pv)
        assert terminal_value(blatt) is None
        k = len(ergebnis.pv)
        assert ergebnis.value == pytest.approx((-GAMMA) ** k * _gesamtwert(blatt, v3), abs=1e-5)


def test_brett_bleibt_unveraendert(v3):
    board = chess.Board("r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R2QK2R w KQ - 0 8")
    board.push_san("O-O")
    vorher = (board.fen(), list(board.move_stack))
    search(board, v3, 2, GAMMA)
    assert (board.fen(), list(board.move_stack)) == vorher


def test_matt_in_zwei_braucht_drei_halbzuege():
    """
    Mit einem Netz, das überall 0 liefert, zählen nur echte Partieausgänge.
    Ta7 sperrt die siebte Reihe, nach Kg8 setzt Tb8 matt.
    """
    netz = ValueNet(hidden_size=4, seed=0)
    netz.W2[:] = 0.0
    board = chess.Board("7k/8/8/8/8/8/R7/1R4K1 w - - 0 1")
    assert search(board, netz, 2, GAMMA).value < 0.5
    ergebnis = search(board, netz, 3, GAMMA)
    assert ergebnis.value == pytest.approx(GAMMA ** 2)
    assert leaf_of(board, ergebnis.pv).is_checkmate()


def test_springergabel_erst_mit_vorausschau(v3):
    """
    Sc7+ greift König und Turm an. Mit einem Halbzug sieht v3 das nicht, weil
    der Turm erst nach dem Ausweichen des Königs fällt; mit zwei schon.
    """
    board = chess.Board("r3k3/pp3ppp/8/1N6/8/8/PP3PPP/4K3 w - - 0 1")
    assert board.san(value_move(board, v3, GAMMA)) != "Nc7+"
    assert board.san(search(board, v3, 2, GAMMA).move) == "Nc7+"


# --------------------------------------------------------------------------- #
# 3. TD-Leaf: Lernziele für die Blätter der Suche
# --------------------------------------------------------------------------- #
def test_tdleaf_ohne_lernziel_wenn_die_suche_exakt_ist():
    """Endet die Hauptvariante im Matt, ist der Suchwert exakt — nichts zu lernen."""
    netz = ValueNet(hidden_size=4, seed=0)
    netz.W2[:] = 0.0
    X, ziele, ergebnis, halbzuege = play_episode_tdleaf(
        netz, np.random.default_rng(0), GAMMA, 0.0, 10, 0.9, 2,
        start=chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"),
    )
    assert (ergebnis, halbzuege, len(X), len(ziele)) == ("mate", 1, 0, 0)


def test_tdleaf_ziele_passen_zu_den_blaettern():
    netz = ValueNet(hidden_size=16, seed=3, shaped=True)
    X, ziele, ergebnis, halbzuege = play_episode_tdleaf(
        netz, np.random.default_rng(0), GAMMA, 0.0, 6, 0.9, 2, start=chess.Board()
    )
    assert (ergebnis, halbzuege) == ("truncated", 6)
    assert X.shape == (len(ziele), FEATURE_SIZE)
    assert 0 < len(ziele) <= 6
    assert np.all(np.abs(ziele) <= 1.0)


def test_version_bestimmt_die_suchtiefe():
    board = chess.Board()
    assert choose_move(board, "v3").depth == 1
    auswahl = choose_move(board, "v4")
    assert auswahl.depth == 2
    assert auswahl.pv[0] == auswahl.move
    assert len(auswahl.pv) >= 2
