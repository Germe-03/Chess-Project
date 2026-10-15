"""
Tests für den Prüfstand (/test).

Zwei Arten:

1. Die Rechenregeln der Zugbewertung — Gewinnerwartung, Schwellen der Klassen,
   Genauigkeit. Ein Fehler hier färbt jede Partie im Bericht falsch ein, ohne
   dass es auffällt.

2. Die Release-Sperre. test_pruefbericht_passt_zur_registry schlägt fehl,
   sobald eine Version in der Registry steht, für die der Bericht nicht gerechnet
   wurde — oder deren Netz bzw. Suchtiefe sich seit der Prüfung geändert hat.
   So lässt sich keine neue Version ausliefern, ohne dass sie den Prüfstand
   durchlaufen hat. Abhilfe: python -m Code.rl.testbench
"""
import json

import chess.engine

from Code.rl import testbench
from Code.rl.registry import versions
from Code.rl.testbench import classify, expected_points, game_plan, move_accuracy


def _klasse(loss, **sonst):
    werte = dict(loss=loss, best=False, forced=False, only_move=False, sacrifice=False,
                 before=0.5, after=0.5 - loss, chance_missed=False)
    werte.update(sonst)
    return classify(**werte)


def test_gewinnerwartung():
    assert expected_points(chess.engine.Cp(0)) == 0.5
    assert expected_points(chess.engine.Mate(3)) == 1.0
    assert expected_points(chess.engine.Mate(-1)) == 0.0
    assert 0.5 < expected_points(chess.engine.Cp(100)) < expected_points(chess.engine.Cp(300)) < 1.0
    assert abs(expected_points(chess.engine.Cp(-200)) + expected_points(chess.engine.Cp(200)) - 1.0) < 1e-12


def test_schwellen_wie_bei_chess_com():
    assert _klasse(0.01) == "ausgezeichnet"
    assert _klasse(0.03) == "gut"
    assert _klasse(0.07) == "ungenau"
    assert _klasse(0.15) == "fehler"
    assert _klasse(0.35) == "patzer"


def test_sonderfaelle():
    assert _klasse(0.0, best=True) == "bester"
    assert _klasse(0.0, best=True, only_move=True) == "grossartig"
    assert _klasse(0.0, best=True, sacrifice=True, before=0.6, after=0.7) == "brillant"
    # Ein Opfer in ohnehin gewonnener Stellung ist nicht brillant.
    assert _klasse(0.0, best=True, sacrifice=True, before=0.99, after=0.99) == "bester"
    assert _klasse(0.3, forced=True) == "erzwungen"
    # Verpasst ersetzt Fehler und Patzer, nicht aber eine kleine Ungenauigkeit.
    assert _klasse(0.3, chance_missed=True) == "verpasst"
    assert _klasse(0.07, chance_missed=True) == "ungenau"


def test_genauigkeit():
    assert abs(move_accuracy(0.6, 0.6) - 100.0) < 0.01
    assert move_accuracy(0.6, 0.5) > move_accuracy(0.6, 0.3) > move_accuracy(0.9, 0.1) >= 0.0
    assert move_accuracy(0.4, 0.7) == move_accuracy(0.4, 0.4)    # Gewinn zählt nicht als Bonus


def test_jede_version_spielt_zehn_partien_gegen_jede():
    plan = game_plan(["a", "b", "c"])
    assert len(plan) == 3 * 10
    for paar in (("a", "b"), ("a", "c"), ("b", "c")):
        partien = [p for p in plan if {p[0], p[1]} == set(paar)]
        assert len(partien) == 10
        # jede Eröffnung einmal mit jeder Farbe
        assert sorted((w, o) for w, _, o in partien) == sorted(
            (spieler, o) for o in range(5) for spieler in paar
        )


def test_pruefbericht_passt_zur_registry():
    """Release-Sperre: Der mitgelieferte Bericht deckt jede Version in ihrem aktuellen Stand ab."""
    assert testbench.REPORT_PATH.exists(), "Kein Prüfbericht. Vor dem Release: python -m Code.rl.testbench"
    bericht = json.loads(testbench.REPORT_PATH.read_text(encoding="utf-8"))
    stellungen = json.loads(testbench.POSITIONS_PATH.read_text(encoding="utf-8"))["stellungen"]
    assert len(stellungen) == testbench.POSITION_COUNT
    assert [s["fen"] for s in bericht["stellungen"]] == [s["fen"] for s in stellungen]

    namen = [v["name"] for v in versions()]
    schluessel = {v["name"]: testbench.fingerprint(v) for v in versions()}
    for name in namen:
        loesung = bericht["loesungen"].get(name)
        assert loesung is not None, f"{name} fehlt im Prüfbericht. Vor dem Release: python -m Code.rl.testbench"
        assert loesung["schluessel"] == schluessel[name], f"{name} wurde seit der Prüfung verändert."
        assert len(loesung["antworten"]) == len(stellungen)

    erwartet = {testbench._game_key(w, b, o, schluessel) for w, b, o in game_plan(namen)}
    vorhanden = {p["schluessel"] for p in bericht["partien"]}
    assert erwartet <= vorhanden, "Turnierpartien fehlen. Vor dem Release: python -m Code.rl.testbench"
    assert len(bericht["partien"]) == len(erwartet)
