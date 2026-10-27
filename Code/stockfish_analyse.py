"""
Stockfish-Bewertung für die Analyseleiste auf /cheat.

Anders als in Code/rl/stockfish.py spielt Stockfish hier nicht mit, sondern
bewertet nur die Stellung — als unabhängiger Massstab neben der eigenen Engine
und dem Azure-Modell.

Der Engine-Prozess wird beim ersten Aufruf gestartet und dann weiterverwendet.
Ein Neustart pro Anfrage würde jedes Mal rund eine halbe Sekunde kosten, und
die Leiste fragt nach jedem Halbzug. Die Anfragen laufen in FastAPI auf
mehreren Threads, eine UCI-Engine kann aber nur eine Stellung zugleich rechnen;
deshalb das Schloss.

Stockfish liegt nicht im Repository (siehe Code/rl/stockfish.py). Auf
render.com fehlt es, dann meldet der Dienst 503 und die Leiste zeigt
«nicht verfügbar».

Konfiguration:
    STOCKFISH_ZEIT  Rechenzeit pro Stellung in Sekunden (Standard 0.4)
"""
from __future__ import annotations

import os
import threading
from typing import NamedTuple

import chess
import chess.engine

from .rl.stockfish import find_stockfish

ZEIT_SEKUNDEN = float(os.environ.get("STOCKFISH_ZEIT", "0.4"))

_schloss = threading.Lock()
_engine: chess.engine.SimpleEngine | None = None


class NichtVerfuegbar(RuntimeError):
    """Keine Stockfish-Programmdatei gefunden."""


class Bewertung(NamedTuple):
    cp: int | None       # Centipawns aus Sicht von Weiss, None bei Matt
    matt: int | None     # Matt in n Zügen, + für Weiss, - für Schwarz
    tiefe: int
    pv: list[chess.Move]


def _engine_holen() -> chess.engine.SimpleEngine:
    global _engine
    if _engine is None:
        pfad = find_stockfish()
        if pfad is None:
            raise NichtVerfuegbar("Stockfish nicht gefunden.")
        _engine = chess.engine.SimpleEngine.popen_uci(pfad)
    return _engine


def bewerte(board: chess.Board) -> Bewertung:
    """Stellung mit fester Rechenzeit bewerten."""
    global _engine
    with _schloss:
        engine = _engine_holen()
        try:
            info = engine.analyse(board, chess.engine.Limit(time=ZEIT_SEKUNDEN))
        except chess.engine.EngineTerminatedError:
            # Abgestürzt oder von aussen beendet: Beim nächsten Aufruf neu starten.
            _engine = None
            raise
    score = info["score"].white()
    return Bewertung(score.score(), score.mate(), int(info.get("depth", 0)), list(info.get("pv", [])))


def beenden() -> None:
    """Engine-Prozess beim Herunterfahren des Dienstes schliessen."""
    global _engine
    with _schloss:
        if _engine is not None:
            _engine.quit()
            _engine = None
