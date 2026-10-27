"""
Engine-Test (/cheat): Zugfolge statt FEN, Partieende, Stockfish-Leiste.

Geprüft wird vor allem, was die Seite still verfälschen würde: eine Remis-
meldung einen Zug zu früh, ein Zug in der Folge, der nicht hätte gehen dürfen,
und dass ein fehlendes Stockfish den Dienst nicht mitreisst.
"""
from __future__ import annotations

import chess
import pytest
from fastapi import HTTPException

from Code import stockfish_analyse
from Code.main import (
    PartieEngineRequest, PartieZugRequest, StockfishRequest, partie_engine, partie_zug, stockfish,
)

NARRENMATT = ["f2f3", "e7e5", "g2g4"]
# Beide Springer hin und zurück: Nach acht Halbzügen steht die Grundstellung
# zum dritten Mal auf dem Brett.
SPRINGERTANZ = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]


def test_zug_wird_auf_die_zugfolge_gesetzt():
    antwort = partie_zug(PartieZugRequest(moves=["e2e4"], move="e7e5"))
    assert antwort.san == "e5"
    assert antwort.fen.startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w")
    assert not antwort.finished and antwort.result == "*"


def test_ungueltiger_zug_in_der_folge_wird_abgelehnt():
    with pytest.raises(HTTPException) as fehler:
        partie_zug(PartieZugRequest(moves=["e2e4", "e7e5", "e4e5"], move="a7a6"))
    assert fehler.value.status_code == 400


def test_umwandlungsfigur_wird_uebernommen():
    folge = ["e2e4", "d7d5", "e4d5", "c7c6", "d5c6", "g8f6", "c6b7", "b8d7"]
    antwort = partie_zug(PartieZugRequest(moves=folge, move="b7b8n"))
    assert antwort.uci == "b7b8n" and antwort.san == "b8=N"


def test_wiederholung_beendet_erst_beim_dritten_mal():
    # Vor dem letzten Halbzug könnte Schwarz das Remis zwar beanspruchen, die
    # Stellung ist aber erst zweimal dagewesen — die Partie läuft weiter.
    davor = partie_zug(PartieZugRequest(moves=SPRINGERTANZ[:6], move=SPRINGERTANZ[6]))
    assert not davor.finished
    danach = partie_zug(PartieZugRequest(moves=SPRINGERTANZ[:7], move=SPRINGERTANZ[7]))
    assert danach.finished and danach.result == "1/2-1/2"


def test_nach_matt_zieht_niemand_mehr():
    antwort = partie_zug(PartieZugRequest(moves=NARRENMATT, move="d8h4"))
    assert antwort.finished and antwort.result == "0-1"
    with pytest.raises(HTTPException) as fehler:
        partie_engine(PartieEngineRequest(moves=NARRENMATT + ["d8h4"]))
    assert fehler.value.status_code == 422


def test_engine_spielt_einen_legalen_zug_fuer_die_zugpartei():
    antwort = partie_engine(PartieEngineRequest(moves=["e2e4"]))
    board = chess.Board()
    board.push_uci("e2e4")
    assert chess.Move.from_uci(antwort.uci) in board.legal_moves
    assert antwort.engine in ("rl", "material") and antwort.version


def test_ohne_stockfish_meldet_die_leiste_503(monkeypatch):
    monkeypatch.setattr(stockfish_analyse, "_engine", None)
    monkeypatch.setattr(stockfish_analyse, "find_stockfish", lambda: None)
    with pytest.raises(HTTPException) as fehler:
        stockfish(StockfishRequest(fen=chess.STARTING_FEN))
    assert fehler.value.status_code == 503
