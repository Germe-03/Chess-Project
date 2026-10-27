"""
Korrekturen der Suche ab v6: Deckel und Wiederholung = Remis.

Anlass war eine Diagnose von v5 gegen sehr schwache Gegner: In den nicht
gewonnenen Partien stand 871-mal ein Matt in 1 auf dem Brett, gespielt wurde es
7-mal, und mehrere Partien mit einer Dame mehr endeten durch dreifache
Wiederholung remis (siehe search.py). Geprüft wird beides an echten Stellungen
und an kleinen, durchschaubaren Netzen — und dass v1 bis v5 davon unberührt
bleiben.
"""
from pathlib import Path

import chess
import numpy as np
import pytest

from Code.main import choose_move
from Code.rl import testbench
from Code.rl.registry import find, search_options
from Code.rl.search import CLASSIC, SearchOptions, _Searcher, search
from Code.rl.tdleaf import play_episode_tdleaf
from Code.rl.valuenet import ValueNet

MODELS = Path(__file__).resolve().parents[1] / "models"
GAMMA = 0.99
V6 = SearchOptions(cap=0.9, repetition_draw=True)

# Aus einer Diagnosepartie von v5 gegen einen Zufallsspieler: zwei Damen mehr,
# Matt in 1 mit De8-b8 oder De8-c6. v5 spielte Sa3-b5 und später remis.
DIAGNOSE = "1q2QQ2/2k4p/4p2r/p1p1N1p1/2P2p2/NP1P3P/PB3PP1/1R3RK1 w - - 1 30"
DIAGNOSE_MATT = {"e8b8", "e8c6"}


def _konstant(wert: float, shaped: bool) -> ValueNet:
    """Netz, das jede Stellung mit tanh(wert) bewertet — für durchschaubare Fälle."""
    netz = ValueNet(hidden_size=4, seed=0, shaped=shaped)
    netz.W2[:] = 0.0
    netz.b2[:] = wert
    return netz


def _springertanz() -> chess.Board:
    """Nach Sf3 Sf6 Sg1 Sg8 Sf3 Sf6 führt Sg1 in eine schon dagewesene Stellung."""
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
        board.push_uci(uci)
    return board


@pytest.fixture(scope="module")
def v5() -> ValueNet:
    return ValueNet.load(MODELS / "full_v5.npz")


# --------------------------------------------------------------------------- #
# Deckel
# --------------------------------------------------------------------------- #
def test_deckel_ist_streng_monoton_und_umkehrbar():
    werte = np.array([-1.5, -0.6, -0.01, 0.0, 0.3, 0.95, 1.4], dtype=np.float32)
    gestaucht = V6.squash(werte)
    assert np.all(np.diff(gestaucht) > 0)
    assert np.all(np.abs(gestaucht) < 0.9)
    for wert, s in zip(werte, gestaucht):
        assert V6.unsquash(float(s)) == pytest.approx(float(wert), abs=1e-4)


def test_ohne_deckel_ist_haushoch_vorne_mehr_wert_als_matt():
    """
    Weiss hat Dame und zwei Türme mehr, Tg8 und Th8 sind matt. Das Netz hält
    jede Stellung für verloren für die Zugpartei (V = -1), Φ sagt dasselbe: Für
    Schwarz ergibt das V + Φ um -1.5, für Weiss also einen Zugwert um +1.47 —
    mehr als das Matt (1.0). Erst mit Deckel gewinnt das Matt.
    """
    board = chess.Board("k7/8/1K6/8/8/8/7R/6RQ w - - 0 1")
    matt = {"g1g8", "h2h8"}
    netz = _konstant(-20.0, shaped=True)
    ohne = search(board, netz, 1, GAMMA)
    assert ohne.move.uci() not in matt and ohne.value > 1.0
    mit = search(board, netz, 1, GAMMA, options=SearchOptions(cap=0.9))
    assert mit.move.uci() in matt and mit.value == 1.0


def test_v5_uebersieht_das_matt_v6_spielt_es(v5):
    board = chess.Board(DIAGNOSE)
    assert search(board, v5, 2, GAMMA).move.uci() not in DIAGNOSE_MATT
    assert search(board, v5, 2, GAMMA, options=V6).move.uci() in DIAGNOSE_MATT


# --------------------------------------------------------------------------- #
# Wiederholung = Remis
# --------------------------------------------------------------------------- #
def test_wiederholender_zug_ist_ein_exaktes_remis(v5):
    board = _springertanz()
    moves = list(board.legal_moves)
    zurueck = moves.index(chess.Move.from_uci("f3g1"))
    scores, totals, _ = _Searcher(v5, GAMMA, 0, V6)._children(board, moves)
    assert totals[zurueck] is None and scores[zurueck] == 0.0
    # Ohne die Regel ist es ein gewöhnlicher Zug, und das Brett bleibt unverändert.
    _, totals, _ = _Searcher(v5, GAMMA, 0)._children(board, moves)
    assert totals[zurueck] is not None
    assert board.fen() == _springertanz().fen()


def test_wer_hinten_steht_sucht_die_wiederholung():
    # Das Netz hält jede Stellung für gewonnen für die Zugpartei: Jeder Zug von
    # Weiss gibt Schwarz eine Gewinnstellung, nur die Wiederholung rettet.
    netz = _konstant(20.0, shaped=False)
    ergebnis = search(_springertanz(), netz, 1, GAMMA, options=V6)
    assert ergebnis.move.uci() == "f3g1" and ergebnis.value == 0.0


def test_wer_vorne_steht_meidet_die_wiederholung():
    netz = _konstant(-20.0, shaped=False)
    ergebnis = search(_springertanz(), netz, 1, GAMMA, options=V6)
    assert ergebnis.move.uci() != "f3g1" and ergebnis.value > 0.0


# --------------------------------------------------------------------------- #
# Versionen
# --------------------------------------------------------------------------- #
def test_v1_bis_v5_rechnen_ohne_korrekturen():
    for name in ("v1", "v2", "v3", "v4", "v5"):
        assert search_options(name) == CLASSIC
    werte = np.array([0.4, 1.3], dtype=np.float32)
    assert CLASSIC.squash(werte) is werte


def test_v6_ist_v5_mit_korrigierter_suche():
    assert search_options("v6") == V6
    assert find("v6")["file"] == find("v5")["file"]
    assert choose_move(chess.Board(DIAGNOSE), "v6").move.uci() in DIAGNOSE_MATT
    assert choose_move(chess.Board(DIAGNOSE), "v5").move.uci() not in DIAGNOSE_MATT


def test_pruefstand_unterscheidet_v5_und_v6():
    v5_key, v6_key = testbench.fingerprint(find("v5")), testbench.fingerprint(find("v6"))
    assert v6_key == v5_key + ":cap0.9:wdh"
    assert v5_key.count(":") == 1      # Kennungen von v1 bis v5 bleiben wie im Prüfbericht


# --------------------------------------------------------------------------- #
# Training (v7)
# --------------------------------------------------------------------------- #
def test_tdleaf_mit_korrekturen_liefert_erreichbare_ziele():
    netz = ValueNet(hidden_size=16, seed=3, shaped=True)
    X, ziele, ergebnis, halbzuege = play_episode_tdleaf(
        netz, np.random.default_rng(0), GAMMA, 0.0, 6, 0.9, 2, start=chess.Board(), options=V6
    )
    assert (ergebnis, halbzuege) == ("truncated", 6)
    assert len(X) == len(ziele) > 0
    assert np.all(np.abs(ziele) <= 1.0)


def test_tdleaf_lernt_nicht_an_einer_wiederholung():
    """Endet die Hauptvariante in einer Wiederholung, ist ihr Wert exakt — kein Lernziel."""
    # Tiefe 1, weil ein Netz mit konstanter Bewertung nur dort in sich stimmig ist:
    # Wie oben rettet nur die Wiederholung, und die Variante endet genau dort.
    netz = _konstant(20.0, shaped=False)
    X, ziele, _, halbzuege = play_episode_tdleaf(
        netz, np.random.default_rng(0), GAMMA, 0.0, 1, 0.9, 1, start=_springertanz(), options=V6
    )
    assert halbzuege == 1 and len(X) == len(ziele) == 0
