"""
Tests für den RL-Teil.

Geprüft wird bewusst nicht die Spielstärke — die schwankt und ist in einem Test
nicht sinnvoll festzunageln. Geprüft werden die Rechenregeln, bei denen ein
Fehler **still** passiert: falsche Vorzeichen, falsche Perspektive, falsch
herum gespiegelte Bretter. So etwas führt nicht zu einem Absturz, sondern zu
einem Agenten, der monatelang das Falsche lernt.
"""
import chess
import numpy as np

from Code.rl.engine import material_move, score_moves, terminal_value
from Code.rl.features import FEATURE_SIZE, board_to_features
from Code.rl.shaping import potential, static_exchange
from Code.rl.valuenet import ValueNet


# --------------------------------------------------------------------------- #
# Merkmale
# --------------------------------------------------------------------------- #
def test_merkmale_sind_perspektivisch():
    """Dieselbe Stellung aus beiden Sichten muss identisch kodiert werden."""
    weiss = chess.Board("8/8/4k3/8/8/2K5/4Q3/8 w - - 0 1")
    schwarz = weiss.mirror()
    assert np.array_equal(board_to_features(weiss), board_to_features(schwarz))


def test_merkmale_zaehlen_figuren():
    x = board_to_features(chess.Board("8/8/4k3/8/8/2K5/4Q3/8 w - - 0 1"))
    assert len(x) == FEATURE_SIZE
    assert x[:768].sum() == 3        # zwei Könige und eine Dame


def test_zusatzmerkmale_erfassen_rochade_und_zugzaehler():
    ohne = board_to_features(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"))
    mit = board_to_features(chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"))
    assert ohne[768:772].sum() == 0
    assert mit[768:772].sum() == 4


# --------------------------------------------------------------------------- #
# Partieende
# --------------------------------------------------------------------------- #
def test_matt_zaehlt_fuer_den_der_zog():
    """+1 gilt aus Sicht der Partei, die den Mattzug gemacht hat."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    board.push(board.parse_san("Ra8#"))
    assert terminal_value(board) == 1.0


def test_patt_und_laufende_partie():
    assert terminal_value(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")) == 0.0
    assert terminal_value(chess.Board()) is None


# --------------------------------------------------------------------------- #
# Zugbewertung
# --------------------------------------------------------------------------- #
def test_mattzug_bekommt_hoechste_bewertung():
    """Ein Matt in einem Zug muss die Bewertung 1.0 erhalten, nicht die Netzschätzung."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    netz = ValueNet(hidden_size=16, seed=0)
    zuege, punkte = score_moves(board, netz, 0.99)
    matt = [i for i, z in enumerate(zuege) if board.san(z) == "Ra8#"]
    assert matt, "Ra8# sollte unter den legalen Zügen sein"
    assert punkte[matt[0]] == 1.0
    assert punkte.max() == 1.0


def test_heuristik_nimmt_matt_in_eins():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
    assert board.san(material_move(board)) == "Ra8#"


# --------------------------------------------------------------------------- #
# Abtauschbewertung
# --------------------------------------------------------------------------- #
def test_see_guter_tausch():
    """Läufer schlägt gedeckten Turm: Gewinn 5 minus Läufer 3."""
    board = chess.Board("4k3/1p6/2p5/3r4/8/1B6/8/4K3 w - - 0 1")
    assert static_exchange(board, chess.D5, chess.WHITE) == 2


def test_see_schlechter_tausch_lohnt_nicht():
    """Turm gegen gedeckten Bauern: Der Schlagende würde verlieren, also 0."""
    board = chess.Board("4k3/2p5/3p4/3R4/8/8/8/4K3 w - - 0 1")
    assert static_exchange(board, chess.D6, chess.WHITE) == 0


def test_see_freier_bauer():
    board = chess.Board("4k3/8/3p4/3R4/8/8/8/4K3 w - - 0 1")
    assert static_exchange(board, chess.D6, chess.WHITE) == 1


# --------------------------------------------------------------------------- #
# Shaping
# --------------------------------------------------------------------------- #
def test_potential_ist_an_endstellungen_null():
    """Ohne das bricht die Policy-Invarianz des Shapings."""
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    board.push(board.parse_san("Ra8#"))
    assert potential(board) == 0.0


def test_potential_bevorzugt_materialvorsprung():
    vorsprung = potential(chess.Board("4k3/8/8/8/8/8/8/3QK3 w - - 0 1"))
    rueckstand = potential(chess.Board("3qk3/8/8/8/8/8/8/4K3 w - - 0 1"))
    assert vorsprung > 0 > rueckstand


def test_potential_bleibt_beschraenkt():
    """Ein Sieg zählt 1.0 und muss jeden Stellungsvorteil überwiegen."""
    extrem = chess.Board("4k3/8/8/8/8/8/QQQQQQQQ/QQQQKQQQ w - - 0 1")
    assert abs(potential(extrem)) <= 0.5


# --------------------------------------------------------------------------- #
# Value-Netz
# --------------------------------------------------------------------------- #
def test_netz_liegt_im_wertebereich():
    netz = ValueNet(hidden_size=16, seed=0)
    rng = np.random.default_rng(0)
    X = (rng.random((20, FEATURE_SIZE)) < 0.05).astype(np.float32)
    werte = netz.value(X)
    assert werte.shape == (20,)
    assert np.all(np.abs(werte) <= 1.0)


def test_netz_lernt():
    netz = ValueNet(hidden_size=32, seed=1)
    rng = np.random.default_rng(1)
    X = (rng.random((64, FEATURE_SIZE)) < 0.03).astype(np.float32)
    ziele = rng.uniform(-0.9, 0.9, size=64).astype(np.float32)
    zuerst = netz.update(X, ziele, 1e-3)
    for _ in range(300):
        zuletzt = netz.update(X, ziele, 1e-3)
    assert zuletzt < zuerst / 4


def test_verbreitern_erhaelt_die_funktion():
    """Net2Net: Ein breiteres Netz muss zunächst exakt dasselbe berechnen."""
    netz = ValueNet(hidden_size=32, seed=2)
    rng = np.random.default_rng(2)
    X = (rng.random((40, FEATURE_SIZE)) < 0.04).astype(np.float32)
    breiter = netz.widened(128, seed=3, noise=0.0)
    assert breiter.hidden_size == 128
    assert np.allclose(netz.value(X), breiter.value(X), atol=1e-5)


def test_modell_ueberlebt_speichern(tmp_path):
    netz = ValueNet(hidden_size=24, seed=4, shaped=True)
    pfad = tmp_path / "netz.npz"
    netz.save(pfad)
    geladen = ValueNet.load(pfad)
    assert geladen.shaped is True
    assert geladen.hidden_size == 24
    rng = np.random.default_rng(5)
    X = (rng.random((10, FEATURE_SIZE)) < 0.04).astype(np.float32)
    assert np.allclose(netz.value(X), geladen.value(X))
