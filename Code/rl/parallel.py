"""
Self-Play und Auswertung auf mehreren Prozessorkernen.

Warum überhaupt? Mit Vorausrechnen kostet eine Trainingspartie nicht mehr
Sekundenbruchteile, sondern rund eine Minute. 3000 Partien auf einem Kern wären
zwei Tage. Der Entwicklungsrechner hat aber 14 Kerne, von denen das Training
bisher genau einen nutzte.

Warum Prozesse und nicht Threads? Python führt wegen des Global Interpreter
Lock immer nur einen Thread gleichzeitig aus. Die Arbeit hier ist reiner
Python-Code (Zugerzeugung, Φ), also bringen nur getrennte Prozesse echte
Parallelität.

Ablauf beim Training (asynchron)
--------------------------------
* Bis zu 'workers' Partien laufen gleichzeitig, jede mit einer Kopie der
  Gewichte zum Zeitpunkt ihres Starts.
* Ist eine fertig, macht der Hauptprozess den Lernschritt und startet sofort
  die nächste Partie, schon mit den neuen Gewichten.

Eine Partie wird dadurch mit Gewichten gespielt, die einige Lernschritte alt
sind — bei 12 Prozessen höchstens 11. Bei Tausenden Partien und kleiner
Lernrate ist das unerheblich; dasselbe Prinzip nutzen verteilte Verfahren wie
A3C und IMPALA. Der Preis: Die Reihenfolge der Lernschritte hängt von der
Rechenzeit ab, ein Lauf ist über den Seed nicht mehr exakt wiederholbar. Mit
workers = 1 läuft alles im Hauptprozess und bleibt reproduzierbar.

Vor jeder Zwischenauswertung wartet das Training, bis alle laufenden Partien
fertig sind. So misst die Auswertung einen klar definierten Stand.

BLAS-Threads: Jeder Prozess bekommt OPENBLAS_NUM_THREADS=1. Sonst startet jede
NumPy-Instanz eigene Rechenthreads, und zwölf Prozesse mit je zwanzig Threads
kämpfen um zwanzig Kerne.
"""
from __future__ import annotations

import multiprocessing
import multiprocessing.util
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack

import chess
import numpy as np

from .search import CLASSIC
from .valuenet import ValueNet

_THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

# Zustand je Prozess: das zuletzt ausgepackte Netz und offene Gegner-Engines.
_state: dict = {"token": None, "net": None, "opponents": {}, "stack": None}


class Runner:
    """Führt Aufgaben aus — im Hauptprozess (workers = 1) oder in einem Prozesspool."""

    def __init__(self, workers: int = 1):
        self.workers = max(1, workers)
        self._pool = None
        if self.workers > 1:
            # Die Kindprozesse erben die Umgebung des Hauptprozesses beim Start.
            for variable in _THREAD_VARIABLES:
                os.environ[variable] = "1"
            self._pool = ProcessPoolExecutor(
                self.workers, mp_context=multiprocessing.get_context("spawn")
            )

    def share(self, net: ValueNet, token: int):
        """
        Gewichte für eine Aufgabe verpacken. Im Hauptprozess genügt das Netz
        selbst; an andere Prozesse gehen die Arrays samt einer Kennung, damit
        ein Prozess dasselbe Netz nicht mehrfach auspacken muss.
        """
        if self._pool is None:
            return net
        return {
            "token": token, "W1": net.W1, "b1": net.b1, "W2": net.W2, "b2": net.b2, "shaped": net.shaped,
        }

    def map(self, function, jobs: list[tuple]) -> list:
        """Alle Aufgaben ausführen, Ergebnisse in der Reihenfolge der Aufgaben."""
        if self._pool is None:
            return [function(*job) for job in jobs]
        futures = [self._pool.submit(function, *job) for job in jobs]
        return [future.result() for future in futures]

    def stream(self, function, make_job, count: int, should_stop=None):
        """
        'count' Aufgaben ausführen, höchstens 'workers' gleichzeitig.

        make_job(i) wird erst beim Start der i-ten Aufgabe aufgerufen. Dadurch
        bekommt jede Partie die Gewichte, die in diesem Moment aktuell sind.
        Liefert (i, Ergebnis) in der Reihenfolge, in der die Aufgaben fertig
        werden.

        should_stop() wird alle paar Sekunden gefragt. Liefert es True, endet
        der Strom sofort; noch laufende Aufgaben beendet danach terminate().
        """
        if self._pool is None:
            for index in range(count):
                if should_stop is not None and should_stop():
                    return
                yield index, function(*make_job(index))
            return
        pending: dict = {}
        started = 0
        while pending or started < count:
            while started < count and len(pending) < self.workers:
                pending[self._pool.submit(function, *make_job(started))] = started
                started += 1
            done, _ = wait(pending, timeout=2.0, return_when=FIRST_COMPLETED)
            for future in done:
                yield pending.pop(future), future.result()
            if should_stop is not None and should_stop():
                return

    def terminate(self) -> None:
        """
        Laufende Aufgaben sofort abbrechen. shutdown() allein würde warten,
        bis jede angefangene Partie fertig ist — bei Tiefe 4 bis zu einer
        halben Stunde. Die Prozessliste ist ein internes Attribut von
        ProcessPoolExecutor; einen öffentlichen Weg gibt es dafür nicht.
        """
        if self._pool is not None:
            for process in list(getattr(self._pool, "_processes", {}).values()):
                process.terminate()
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
        close_opponents()

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(cancel_futures=True)
            self._pool = None
        close_opponents()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _as_net(shared) -> ValueNet:
    if isinstance(shared, ValueNet):
        return shared
    if _state["token"] != shared["token"]:
        net = ValueNet(
            input_size=shared["W1"].shape[0], hidden_size=shared["W1"].shape[1], shaped=shared["shaped"]
        )
        net.W1, net.b1, net.W2, net.b2 = shared["W1"], shared["b1"], shared["W2"], shared["b2"]
        _state.update(token=shared["token"], net=net)
    return _state["net"]


def episode_task(shared, params: dict):
    """
    Eine Trainingspartie. Rückgabe: Merkmale, Lernziele, Ausgang, Halbzüge,
    Rechenzeit in Sekunden. Gelernt wird nicht hier, sondern im Hauptprozess.
    """
    net = _as_net(shared)
    rng = np.random.default_rng(params["seed"])
    started = time.perf_counter()
    if params["depth"] > 1:
        from .tdleaf import play_episode_tdleaf

        X, targets, result, plies = play_episode_tdleaf(
            net, rng, params["gamma"], params["epsilon"], params["max_plies"],
            params["td_lambda"], params["depth"], options=params.get("options", CLASSIC),
        )
    else:
        from .train import play_episode

        X, targets, result, plies = play_episode(
            net, rng, params["task"], params["gamma"], params["epsilon"], params["max_plies"],
            params["difficulty"], params["td_lambda"],
        )
    return X, targets, result, plies, time.perf_counter() - started


def eval_game_task(shared, params: dict) -> str:
    """Eine Auswertungspartie aus der Grundstellung. Ergebnis aus Sicht des Agenten."""
    from .evaluate import make_policy, play_game

    net = _as_net(shared)
    rng = np.random.default_rng(params["seed"])
    agent = make_policy("rl", rng, net, params["gamma"], params["depth"], params.get("options", CLASSIC))
    color = chess.WHITE if params["game"] % 2 == 0 else chess.BLACK
    start = None
    if params.get("opening") is not None:
        from .positions import sample_start

        start = sample_start("full", np.random.default_rng(params["opening"]))
    opponent = _opponent(params["opponent"], params["gamma"])
    return play_game(agent, opponent, color, params["max_plies"], start)


def _opponent(spec: tuple, gamma: float):
    """
    Gegner aus einer Beschreibung, damit er sich an andere Prozesse schicken
    lässt: ("material",), ("stockfish", Pfad, Tiefe, Elo) oder
    ("modell", Datei, Tiefe), optional mit SearchOptions als viertem Eintrag.
    Engines bleiben je Prozess offen und werden erst beim Prozessende
    geschlossen.
    """
    if spec in _state["opponents"]:
        return _state["opponents"][spec]
    kind = spec[0]
    if kind == "material":
        from .engine import material_move

        policy = material_move
    elif kind == "stockfish":
        from .stockfish import stockfish_policy

        if _state["stack"] is None:
            _state["stack"] = ExitStack()
            # python-chess betreibt jede Engine in einem eigenen Thread. Beim
            # Beenden wartet Python auf alle Threads, BEVOR atexit-Funktionen
            # laufen — eine noch offene Engine liesse den Prozess also ewig
            # hängen. Finalize läuft in einem Kindprozess vor diesem Warten;
            # im Hauptprozess schliesst Runner.close() die Engines.
            multiprocessing.util.Finalize(None, close_opponents, exitpriority=10)
        _, path, depth, elo = spec
        policy = _state["stack"].enter_context(stockfish_policy(path, depth=depth, elo=elo))
    elif kind == "modell":
        from .evaluate import make_policy

        path, depth = spec[1], spec[2]
        options = spec[3] if len(spec) > 3 else CLASSIC
        policy = make_policy("rl", None, ValueNet.load(path), gamma, depth, options)
    else:
        raise ValueError(f"Unbekannter Gegner: {spec}")
    _state["opponents"][spec] = policy
    return policy


def close_opponents() -> None:
    """Offene Gegner-Engines dieses Prozesses beenden."""
    if _state["stack"] is not None:
        _state["stack"].close()
        _state["stack"] = None
    _state["opponents"].clear()
