from __future__ import annotations

import math

import chess
import pytest

from Code.rl.messlatte import elo_schaetzen, leistung
from Code.rl.stockfish import PartieErkennung, find_stockfish, stockfish_policy


def _zuege(*san_zuege: str) -> list[chess.Move]:
    """Erzeugt eine legale Zugfolge aus Schachnotation."""
    brett = chess.Board()
    zuege = []
    for san in san_zuege:
        zug = brett.parse_san(san)
        zuege.append(zug)
        brett.push(zug)
    return zuege


def test_elo_schaetzen_bei_ausgeglichenen_ergebnissen() -> None:
    # Bei 50 Prozent liegt die geschätzte Spielstärke auf derselben Stufe.
    elo, _ = elo_schaetzen([(1600, 100.0, 200)])

    assert elo == pytest.approx(1600, abs=1)


def test_elo_schaetzen_bei_76_prozent() -> None:
    elo, _ = elo_schaetzen([(1600, 152.0, 200)])

    assert elo == pytest.approx(1800, abs=5)


def test_elo_schaetzen_kombiniert_mehrere_stufen() -> None:
    elo, _ = elo_schaetzen([(1400, 152.0, 200), (1800, 48.0, 200)])

    assert elo == pytest.approx(1600, abs=5)


def test_elo_schaetzen_begrenzt_perfekte_ergebnisse() -> None:
    elo, _ = elo_schaetzen([(1600, 200.0, 200)])

    assert elo > 1600


def test_elo_schaetzen_begrenzt_ergebnisse_ohne_punkte() -> None:
    elo, _ = elo_schaetzen([(1600, 0.0, 200)])

    assert elo < 1600


def test_elo_schaetzen_standardfehler_wird_mit_mehr_partien_kleiner() -> None:
    _, fehler_wenige_partien = elo_schaetzen([(1600, 100.0, 200)])
    _, fehler_viele_partien = elo_schaetzen([(1600, 400.0, 800)])

    assert fehler_viele_partien < fehler_wenige_partien


def test_leistung_bei_50_prozent_entspricht_der_stufe() -> None:
    assert leistung(1600, 0.5) == pytest.approx(1600)


def test_leistung_begrenzt_100_prozent() -> None:
    assert leistung(1600, 1.0) == pytest.approx(1600 + 400 * math.log10(99))


def test_leistung_begrenzt_0_prozent_symmetrisch() -> None:
    assert leistung(1600, 0.0) == pytest.approx(1600 - 400 * math.log10(99))


def test_partieerkennung_erkennt_neue_partien() -> None:
    erkennung = PartieErkennung()
    e4 = _zuege("e4")[0]
    e4_e5 = _zuege("e4", "e5")

    # Ein leerer Anfang ist ebenfalls eine neue Partie.
    assert erkennung.ist_neu([]) is True
    assert erkennung.ist_neu([e4]) is False
    assert erkennung.ist_neu(e4_e5) is False

    # Gleiche oder kürzere Zugfolgen zeigen einen neuen Stack an.
    assert erkennung.ist_neu(e4_e5) is True
    assert erkennung.ist_neu([e4]) is True


def test_partieerkennung_erkennt_einen_anderen_anlaut() -> None:
    erkennung = PartieErkennung()

    assert erkennung.ist_neu(_zuege("e4", "e5", "Nf3")) is True
    assert erkennung.ist_neu(_zuege("d4", "d5", "c4", "c5")) is True


@pytest.mark.skipif(
    find_stockfish(None) is None,
    reason="Stockfish ist nicht installiert",
)
def test_stockfish_policy_gibt_einen_legalen_zug_zurueck() -> None:
    with stockfish_policy(zeit=0.05, elo=1500, neues_spiel=True) as policy:
        brett = chess.Board()
        zug = policy(brett)

    assert zug in brett.legal_moves
