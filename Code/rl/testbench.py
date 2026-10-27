"""
Prüfstand: jede Engine-Version unter immer denselben Bedingungen.

Läuft lokal vor jedem Release, denn er braucht Stockfish — und Stockfish gehört
nicht auf render.com (über 100 MB, viel zu rechenintensiv für 0.1 CPU). Das
Ergebnis ist ein Bericht, testbench/bericht.json, der mit dem Release gepusht und
unter /test angezeigt wird. Ein Test (tests/test_testbench.py) schlägt fehl,
solange der Bericht nicht zu allen Versionen der Registry passt; eine neue
Version lässt sich also nicht unbemerkt ohne Prüfung ausliefern.

Drei Teile
----------
1. Teststellungen. 50 Stellungen mit genau einem klar besten Zug, einmal
   erzeugt (testbench/stellungen.json) und danach nie mehr verändert. Jede
   Version bekommt dieselben. Erzeugt aus Partien von Stockfish gegen sich selbst
   auf Elo 1320, damit die Auswahl keine der eigenen Versionen bevorzugt, und von
   Stockfish ohne Begrenzung geprüft.

2. Turnier. Jede Version spielt gegen jede 10 Partien: 5 feste Eröffnungen, je
   einmal mit jeder Farbe. Gespielt wird über main.choose_move, also genau so,
   wie die Version auch auf der Webseite zieht.

3. Zugbewertung wie bei chess.com. Stockfish analysiert jede Stellung der
   Partien. Aus der Bewertung wird die Gewinnerwartung der Partei am Zug (0 bis 1,
   Formel von Lichess), und der Verlust an Gewinnerwartung durch den gespielten
   Zug bestimmt die Klasse:

       Verlust < 0.02  Ausgezeichnet      < 0.10  Ungenauigkeit
       Verlust < 0.05  Gut                < 0.20  Fehler      sonst  Patzer

   Dazu die Sonderfälle: Bester Zug (Stockfishs erste Wahl), Grossartig (der
   einzige gute Zug), Brillant (bester Zug mit Figurenopfer), Verpasst (eine
   Gelegenheit — Matt oder ein Fehler des Gegners — wird nicht genutzt),
   Erzwungen (nur ein legaler Zug) und Eröffnung (vorgegeben). Die Schwellen
   entsprechen denen, die chess.com für sein Expected-Points-Modell nennt.

Inkrementell: Jedes Ergebnis trägt einen Schlüssel aus dem Fingerabdruck der
Modelldatei und der Suchtiefe. Kommt eine Version dazu, werden nur ihre
Teststellungen und ihre Partien gegen die anderen gerechnet.

Aufruf:
    python -m Code.rl.testbench --generate-positions   einmalig: die 50 Stellungen
    python -m Code.rl.testbench                        vor jedem Release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing.util
import os
import time
from datetime import datetime
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .parallel import Runner
from .positions import sample_start
from .registry import MODELS_DIR, search_depth, search_options
from .registry import versions as registry_versions
from .shaping import loses_material
from .stockfish import find_stockfish

ROOT = Path(__file__).resolve().parents[2]
TESTBENCH_DIR = ROOT / "testbench"
POSITIONS_PATH = TESTBENCH_DIR / "stellungen.json"
REPORT_PATH = TESTBENCH_DIR / "bericht.json"

ANALYSIS_DEPTH = 14          # Stockfish beim Bewerten der Partien
POSITION_DEPTH = 18          # Stockfish beim Bestätigen der Teststellungen
MAX_PLIES = 300              # Abbruch einer Turnierpartie, als Remis gewertet
POSITION_COUNT = 50
UNIQUE_GAP = 0.25            # so viel Gewinnerwartung liegt der beste Zug mindestens vorn

OPENINGS = [
    ("Italienisch", "e4 e5 Nf3 Nc6 Bc4 Bc5"),
    ("Sizilianisch", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6"),
    ("Französisch", "e4 e6 d4 d5 Nc3 Nf6"),
    ("Damengambit", "d4 d5 c4 e6 Nc3 Nf6"),
    ("Englisch", "c4 e5 Nc3 Nf6 g3 d5"),
]

# Klasse -> (Bezeichnung, Symbol). Reihenfolge wie in der Übersicht der Seite.
CLASSES = {
    "brillant": ("Brillant", "!!"),
    "grossartig": ("Grossartig", "!"),
    "bester": ("Bester Zug", "★"),
    "ausgezeichnet": ("Ausgezeichnet", "👍"),
    "gut": ("Gut", "✓"),
    "eroeffnung": ("Eröffnung", "📖"),
    "erzwungen": ("Erzwungen", "→"),
    "ungenau": ("Ungenauigkeit", "?!"),
    "verpasst": ("Verpasst", "✗"),
    "fehler": ("Fehler", "?"),
    "patzer": ("Patzer", "??"),
}


# --------------------------------------------------------------------------- #
# Bewertung von Zügen
# --------------------------------------------------------------------------- #
def expected_points(score: chess.engine.Score) -> float:
    """
    Gewinnerwartung (0 bis 1) für die Partei, aus deren Sicht 'score' gilt.

    Centipawns werden mit der Formel von Lichess umgerechnet; ein Matt zählt als
    sicherer Sieg bzw. sichere Niederlage. Remis entspricht 0.5.
    """
    mate = score.mate()
    if mate is not None:
        return 1.0 if mate > 0 else 0.0
    return 1.0 / (1.0 + math.exp(-0.00368208 * score.score()))


def classify(
    *, loss: float, best: bool, forced: bool, only_move: bool, sacrifice: bool,
    before: float, after: float, chance_missed: bool,
) -> str:
    """Klasse eines Zuges aus dem Verlust an Gewinnerwartung und den Sonderfällen."""
    if forced:
        return "erzwungen"
    if best:
        # Ein Opfer ist nur brillant, wenn danach nicht verloren ist und vorher
        # nicht ohnehin alles gewonnen war.
        if sacrifice and after >= 0.5 and before < 0.97:
            return "brillant"
        return "grossartig" if only_move else "bester"
    if loss < 0.02:
        return "ausgezeichnet"
    if loss < 0.05:
        return "gut"
    if loss < 0.10:
        return "ungenau"
    if chance_missed:
        return "verpasst"
    return "fehler" if loss < 0.20 else "patzer"


def move_accuracy(before: float, after: float) -> float:
    """Genauigkeit eines Zuges in Prozent, Formel von Lichess."""
    drop = max(0.0, (before - after) * 100.0)
    return float(min(100.0, max(0.0, 103.1668 * math.exp(-0.04354 * drop) - 3.1669)))


def _white_score(score: chess.engine.PovScore) -> dict:
    white = score.white()
    return {"matt": white.mate()} if white.is_mate() else {"cp": white.score()}


# --------------------------------------------------------------------------- #
# Stockfish je Prozess
# --------------------------------------------------------------------------- #
_engines: dict = {}


def _stockfish(kind: str = "analyse"):
    """
    Eine Stockfish-Instanz je Prozess und Art: 'analyse' spielt ohne Begrenzung
    mit einem Thread, 'spieler' auf Elo 1320 für die Teststellungs-Partien.
    """
    if kind not in _engines:
        path = find_stockfish(None)
        if path is None:
            raise FileNotFoundError("Stockfish nicht gefunden (tools/stockfish/ oder STOCKFISH_PATH).")
        engine = chess.engine.SimpleEngine.popen_uci(path)
        engine.configure({"Threads": 1, "Hash": 64})
        if kind == "spieler":
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": 1320})
        if not _engines:
            # Wie in parallel.py: vor dem Warten auf Threads schliessen.
            multiprocessing.util.Finalize(None, _close_engines, exitpriority=10)
        _engines[kind] = engine
    return _engines[kind]


def _close_engines() -> None:
    for engine in _engines.values():
        try:
            engine.quit()
        except Exception:
            pass
    _engines.clear()


# --------------------------------------------------------------------------- #
# Teil 1: die 50 Teststellungen
# --------------------------------------------------------------------------- #
def _phase(board: chess.Board) -> str:
    pieces = len(board.piece_map())
    if board.fullmove_number <= 12 and pieces >= 28:
        return "Eröffnung"
    return "Endspiel" if pieces <= 12 else "Mittelspiel"


def _analysis(board: chess.Board, depth: int, multipv: int = 2) -> list[dict]:
    infos = _stockfish().analyse(board, chess.engine.Limit(depth=depth), multipv=multipv)
    return [info for info in infos if info.get("pv")]


def _candidates_from_game(seed: int) -> list[dict]:
    """Eine Partie Stockfish (Elo 1320) gegen sich selbst; Stellungen mit einem klar besten Zug."""
    board = sample_start("full", np.random.default_rng(seed))
    player = _stockfish("spieler")
    found = []
    for _ in range(160):
        if board.is_game_over() or board.is_repetition(3):
            break
        if board.ply() >= 8 and board.legal_moves.count() >= 2:
            infos = _analysis(board, 12)
            if len(infos) == 2:
                best = expected_points(infos[0]["score"].relative)
                second = expected_points(infos[1]["score"].relative)
                if best - second >= UNIQUE_GAP and best >= 0.15:
                    found.append({"fen": board.fen(), "quelle": seed})
        board.push(player.play(board, chess.engine.Limit(depth=1)).move)
    return found


def _confirm(candidate: dict) -> dict | None:
    """Kandidat mit grösserer Tiefe bestätigen und einordnen."""
    board = chess.Board(candidate["fen"])
    infos = _analysis(board, POSITION_DEPTH)
    if len(infos) < 2:
        return None
    best_score, second_score = infos[0]["score"].relative, infos[1]["score"].relative
    best, second = expected_points(best_score), expected_points(second_score)
    if best - second < UNIQUE_GAP or best < 0.15:
        return None
    move = infos[0]["pv"][0]
    if best_score.is_mate() and best_score.mate() > 0:
        category = "Matt"
    elif best <= 0.6:
        category = "Verteidigung"
    else:
        category = "Gewinn"
    return {
        "fen": candidate["fen"],
        "quelle": candidate["quelle"],
        "loesung_uci": move.uci(),
        "loesung_san": board.san(move),
        "kategorie": category,
        "phase": _phase(board),
        "bewertung": _white_score(infos[0]["score"]),
        "erwartung_beste": round(best, 4),
        "erwartung_zweitbeste": round(second, 4),
    }


def generate_positions(workers: int, games: int = 480, force: bool = False) -> list[dict]:
    """
    Die 50 Teststellungen erzeugen. Bewusst nur einmal: Bestehende Stellungen
    werden ohne --force nicht überschrieben, sonst wären die Ergebnisse der
    Versionen nicht mehr vergleichbar.
    """
    if POSITIONS_PATH.exists() and not force:
        raise SystemExit(f"{POSITIONS_PATH} gibt es schon. Ersetzen nur mit --force — "
                         "danach sind alte Ergebnisse nicht mehr vergleichbar.")
    with Runner(workers) as runner:
        print(f"Spiele {games} Partien Stockfish (Elo 1320) gegen sich selbst ...", flush=True)
        candidates = [c for batch in runner.map(_candidates_from_game, [(7000 + i,) for i in range(games)])
                      for c in batch]
        print(f"{len(candidates)} Kandidaten, bestätige mit Tiefe {POSITION_DEPTH} ...", flush=True)
        confirmed = [c for c in runner.map(_confirm, [(c,) for c in candidates]) if c]

    # Ausgewogen auswählen: je Kategorie eine Quote, höchstens zwei Stellungen
    # je Quellpartie, Reihenfolge über einen Hash statt Zufall — reproduzierbar.
    quotas = {"Matt": 15, "Gewinn": 20, "Verteidigung": 15}
    confirmed.sort(key=lambda c: hashlib.sha256(c["fen"].encode()).hexdigest())
    chosen: list[dict] = []
    per_source: dict[int, int] = {}
    seen: set[str] = set()
    for category, quota in quotas.items():
        pool = [c for c in confirmed if c["kategorie"] == category]
        # Phasen mischen: abwechselnd aus Eröffnung, Mittelspiel und Endspiel.
        by_phase = {phase: [c for c in pool if c["phase"] == phase]
                    for phase in ("Eröffnung", "Mittelspiel", "Endspiel")}
        taken = 0
        while taken < quota and any(by_phase.values()):
            for phase in list(by_phase):
                if taken >= quota or not by_phase[phase]:
                    continue
                candidate = by_phase[phase].pop(0)
                key = candidate["fen"].split(" ")[0]
                if key in seen or per_source.get(candidate["quelle"], 0) >= 2:
                    continue
                seen.add(key)
                per_source[candidate["quelle"]] = per_source.get(candidate["quelle"], 0) + 1
                chosen.append(candidate)
                taken += 1
    # Fehlt einer Kategorie etwas, wird mit den übrigen aufgefüllt.
    for candidate in confirmed:
        if len(chosen) >= POSITION_COUNT:
            break
        key = candidate["fen"].split(" ")[0]
        if key not in seen:
            seen.add(key)
            chosen.append(candidate)

    for number, position in enumerate(chosen, start=1):
        position["id"] = number
    data = {
        "beschreibung": (
            f"{len(chosen)} Teststellungen mit genau einem klar besten Zug (mindestens "
            f"{UNIQUE_GAP} Gewinnerwartung vor dem zweitbesten). Erzeugt am "
            f"{datetime.now():%Y-%m-%d} aus {games} Partien Stockfish (Elo 1320) gegen sich "
            f"selbst, bestätigt von Stockfish ohne Begrenzung mit Tiefe {POSITION_DEPTH}. "
            "Wird nicht mehr verändert, damit alle Versionen dieselbe Prüfung bekommen."
        ),
        "stellungen": chosen,
    }
    _write_json(POSITIONS_PATH, data, pretty=True)
    counts = {c: sum(p["kategorie"] == c for p in chosen) for c in quotas}
    print(f"{len(chosen)} Stellungen gespeichert: {counts}")
    return chosen


def _solve_positions(version: str, positions: list[dict]) -> list[dict]:
    """Eine Version löst alle Teststellungen; Stockfish bewertet ihre Antworten."""
    from ..main import choose_move

    answers = []
    for position in positions:
        board = chess.Board(position["fen"])
        move = choose_move(board, version).move
        solved = move.uci() == position["loesung_uci"]
        before = position["erwartung_beste"]
        if solved:
            after, loss = before, 0.0
        else:
            info = _stockfish().analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH), root_moves=[move])
            after = expected_points(info["score"].relative)
            loss = max(0.0, before - after)
        answers.append({
            "id": position["id"],
            "uci": move.uci(),
            "san": board.san(move),
            "geloest": solved,
            "verlust": round(loss, 4),
            "klasse": classify(
                loss=loss, best=solved, forced=False, only_move=True,
                sacrifice=solved and loses_material(board, move), before=before, after=after,
                chance_missed=position["kategorie"] == "Matt",
            ),
        })
    return answers


# --------------------------------------------------------------------------- #
# Teil 2 und 3: Turnier und Zugbewertung
# --------------------------------------------------------------------------- #
def _end_reason(board: chess.Board, plies: int) -> tuple[str, str] | None:
    """Ergebnis und Grund, wenn die Partie zu Ende ist, sonst None."""
    if board.is_checkmate():
        return ("0-1" if board.turn == chess.WHITE else "1-0"), "Matt"
    if board.is_stalemate():
        return "1/2-1/2", "Patt"
    if board.is_insufficient_material():
        return "1/2-1/2", "zu wenig Material"
    if board.halfmove_clock >= 100:
        return "1/2-1/2", "50-Züge-Regel"
    if board.is_repetition(3):
        return "1/2-1/2", "dreifache Stellungswiederholung"
    if plies >= MAX_PLIES:
        return "1/2-1/2", f"Zuglimit ({MAX_PLIES} Halbzüge)"
    return None


def _play_and_analyse(white: str, black: str, opening: int) -> dict:
    """Eine Turnierpartie spielen und jeden Zug von Stockfish bewerten lassen."""
    from ..main import choose_move

    name, line = OPENINGS[opening]
    board = chess.Board()
    played: list[tuple[chess.Move, bool]] = []          # (Zug, vorgegeben)
    for san in line.split():
        played.append((board.parse_san(san), True))
        board.push(played[-1][0])
    engine_plies = 0
    while (end := _end_reason(board, engine_plies)) is None:
        version = white if board.turn == chess.WHITE else black
        move = choose_move(board, version).move
        played.append((move, False))
        board.push(move)
        engine_plies += 1
    result, reason = end

    # Jede Stellung vor einem Zug analysieren; neues Spiel für Stockfish, damit
    # nichts aus der vorigen Partie im Speicher die Bewertung färbt.
    analysis_key = object()
    replay = chess.Board()
    analyses = []
    for move, _ in played:
        infos = _stockfish().analyse(
            replay, chess.engine.Limit(depth=ANALYSIS_DEPTH), multipv=2, game=analysis_key
        )
        analyses.append([info for info in infos if info.get("pv")])
        replay.push(move)

    # Bewertung nach dem letzten Zug: Matt und Remis nach den Regeln sind exakt;
    # wurde am Zuglimit abgebrochen, bewertet Stockfish die Schlussstellung.
    if board.is_checkmate():
        final = None
    elif reason.startswith("Zuglimit"):
        final = _stockfish().analyse(board, chess.engine.Limit(depth=ANALYSIS_DEPTH), game=analysis_key)["score"]
    else:
        final = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)

    replay = chess.Board()
    moves, previous_class = [], None
    for index, (move, given) in enumerate(played):
        infos = analyses[index]
        mover = replay.turn
        before = expected_points(infos[0]["score"].pov(mover))
        second = expected_points(infos[1]["score"].pov(mover)) if len(infos) > 1 else None
        best_move = infos[0]["pv"][0]
        san = replay.san(move)
        forced = replay.legal_moves.count() == 1
        sacrifice = loses_material(replay, move)
        mate_available = infos[0]["score"].pov(mover).is_mate() and infos[0]["score"].pov(mover).mate() > 0
        best_san = replay.san(best_move)
        replay.push(move)

        if index + 1 < len(played):
            after_score = analyses[index + 1][0]["score"]
        else:
            after_score = final
        if after_score is None:                 # Matt mit dem letzten Zug
            after, evaluation = 1.0, {"matt": 0, "sieger": "w" if mover == chess.WHITE else "b"}
        else:
            after, evaluation = expected_points(after_score.pov(mover)), _white_score(after_score)

        is_best = move == best_move
        loss = 0.0 if is_best else max(0.0, before - after)
        if given:
            klasse = "eroeffnung"
        else:
            klasse = classify(
                loss=loss, best=is_best, forced=forced,
                only_move=second is not None and before - second >= 0.20,
                sacrifice=sacrifice, before=before, after=after,
                chance_missed=mate_available or previous_class in ("fehler", "patzer"),
            )
        moves.append({
            "san": san, "uci": move.uci(), "fen": replay.fen(),
            "farbe": "w" if mover == chess.WHITE else "b",
            "klasse": klasse, "verlust": round(loss, 4),
            "vorher": round(before, 4), "nachher": round(after, 4),
            "bewertung": evaluation,
            "beste_san": None if is_best else best_san,
            "beste_uci": None if is_best else best_move.uci(),
        })
        previous_class = klasse

    accuracy = {}
    for color in ("w", "b"):
        own = [m for m in moves if m["farbe"] == color and m["klasse"] not in ("eroeffnung", "erzwungen")]
        accuracy[color] = round(float(np.mean([move_accuracy(m["vorher"], m["nachher"]) for m in own])), 1) \
            if own else None
    return {
        "weiss": white, "schwarz": black, "eroeffnung": name,
        "start_fen": chess.STARTING_FEN, "ergebnis": result, "grund": reason,
        "halbzuege": len(moves), "genauigkeit": accuracy,
        "start_bewertung": _white_score(analyses[0][0]["score"]) if analyses else {"cp": 0},
        "zuege": moves,
    }


# --------------------------------------------------------------------------- #
# Bericht
# --------------------------------------------------------------------------- #
def fingerprint(version: dict) -> str:
    """
    Kennung einer Version: Netzdatei, Suchtiefe und Korrekturen der Suche.
    Ändert sich eines, wird neu geprüft. Die Korrekturen hängen nur an, wenn
    sie eingeschaltet sind — so bleiben die Kennungen von v1 bis v5 gleich.
    """
    digest = hashlib.sha256((MODELS_DIR / version["file"]).read_bytes()).hexdigest()[:16]
    key = f"{digest}:{search_depth(version['name'])}"
    options = search_options(version["name"])
    if options.cap is not None:
        key += f":cap{options.cap}"
    if options.repetition_draw:
        key += ":wdh"
    return key


def game_plan(names: list[str]) -> list[tuple[str, str, int]]:
    """Alle Turnierpartien: jedes Paar, jede Eröffnung, beide Farben."""
    plan = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for opening in range(len(OPENINGS)):
                plan += [(a, b, opening), (b, a, opening)]
    return plan


def _game_key(white: str, black: str, opening: int, prints: dict) -> str:
    return f"{prints[white]}|{prints[black]}|{OPENINGS[opening][0]}|sf{ANALYSIS_DEPTH}|max{MAX_PLIES}"


def _summary(names: list[str], solved: dict, games: list[dict]) -> dict:
    table = {a: {b: 0.0 for b in names if b != a} for a in names}
    stats = {name: {"partien": 0, "punkte": 0.0, "genauigkeit": [], "klassen": {k: 0 for k in CLASSES}}
             for name in names}
    for game in games:
        white, black = game["weiss"], game["schwarz"]
        points = {"1-0": (1.0, 0.0), "0-1": (0.0, 1.0)}.get(game["ergebnis"], (0.5, 0.5))
        table[white][black] += points[0]
        table[black][white] += points[1]
        for name, color, score in ((white, "w", points[0]), (black, "b", points[1])):
            stats[name]["partien"] += 1
            stats[name]["punkte"] += score
            if game["genauigkeit"][color] is not None:
                stats[name]["genauigkeit"].append(game["genauigkeit"][color])
            for move in game["zuege"]:
                if move["farbe"] == color:
                    stats[name]["klassen"][move["klasse"]] += 1
    ranking = []
    for name in names:
        s = stats[name]
        ranking.append({
            "version": name,
            "geloest": sum(answer["geloest"] for answer in solved[name]),
            "partien": s["partien"],
            "punkte": s["punkte"],
            "genauigkeit": round(float(np.mean(s["genauigkeit"])), 1) if s["genauigkeit"] else None,
            "klassen": s["klassen"],
        })
    return {"rangliste": ranking, "kreuztabelle": table}


def _write_json(path: Path, data, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=1 if pretty else None,
                      separators=None if pretty else (",", ":"))
    temporary.write_text(text + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run(workers: int) -> dict:
    """Bericht auf den Stand der Registry bringen; nur Fehlendes wird gerechnet."""
    if not POSITIONS_PATH.exists():
        raise SystemExit("Es gibt noch keine Teststellungen. Zuerst einmalig: "
                         "python -m Code.rl.testbench --generate-positions")
    positions = json.loads(POSITIONS_PATH.read_text(encoding="utf-8"))["stellungen"]
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8")) if REPORT_PATH.exists() else {}
    registry = registry_versions()
    names = [v["name"] for v in registry]
    prints = {v["name"]: fingerprint(v) for v in registry}

    old_solutions = report.get("loesungen", {})
    solve = [name for name in names
             if old_solutions.get(name, {}).get("schluessel") != prints[name]]
    old_games = {g["schluessel"]: g for g in report.get("partien", [])}
    plan = game_plan(names)
    missing = [(w, b, o) for w, b, o in plan if _game_key(w, b, o, prints) not in old_games]
    print(f"{len(names)} Versionen | Teststellungen zu rechnen: {solve or 'keine'} | "
          f"Partien zu rechnen: {len(missing)} von {len(plan)}", flush=True)

    started = time.time()
    with Runner(workers) as runner:
        new_answers = runner.map(_solve_positions, [(name, positions) for name in solve])
        new_games = runner.map(_play_and_analyse, missing)
    solutions = {name: old_solutions[name] for name in names if name not in solve}
    for name, answers in zip(solve, new_answers):
        solutions[name] = {"schluessel": prints[name], "antworten": answers}
    for (white, black, opening), game in zip(missing, new_games):
        game["schluessel"] = _game_key(white, black, opening, prints)
        old_games[game["schluessel"]] = game

    games = []
    for white, black, opening in plan:
        game = old_games[_game_key(white, black, opening, prints)]
        game["id"] = f"{white}-{black}-{opening + 1}"
        games.append(game)

    engine_name = _engine_name()
    report = {
        "erstellt": datetime.now().isoformat(timespec="minutes"),
        "stockfish": {"name": engine_name, "tiefe": ANALYSIS_DEPTH},
        "klassen": {key: {"name": label, "symbol": symbol} for key, (label, symbol) in CLASSES.items()},
        "versionen": [
            {"name": v["name"], "tiefe": search_depth(v["name"]), "schluessel": prints[v["name"]],
             "episoden": v.get("episodes"), "notizen": v.get("notes", "")}
            for v in registry
        ],
        "stellungen": positions,
        "loesungen": solutions,
        "partien": games,
        "zusammenfassung": _summary(names, {n: solutions[n]["antworten"] for n in names}, games),
    }
    _write_json(REPORT_PATH, report)
    print(f"Bericht geschrieben: {REPORT_PATH} ({(time.time() - started) / 60:.1f} min)")
    return report


def _engine_name() -> str:
    """Name und Version der Stockfish-Programmdatei, für den Bericht."""
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish(None))
    try:
        return engine.id.get("name", "Stockfish")
    finally:
        engine.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prüfstand: alle Versionen unter gleichen Bedingungen")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2),
                        help="Partien und Analysen, die gleichzeitig laufen")
    parser.add_argument("--generate-positions", action="store_true",
                        help="Einmalig die 50 Teststellungen erzeugen")
    parser.add_argument("--force", action="store_true",
                        help="Mit --generate-positions: vorhandene Stellungen ersetzen")
    parser.add_argument("--games", type=int, default=480,
                        help="Mit --generate-positions: Quellpartien, aus denen ausgewählt wird")
    args = parser.parse_args()
    if args.generate_positions:
        generate_positions(args.workers, games=args.games, force=args.force)
        return
    report = run(args.workers)
    print("\nRangliste")
    for row in sorted(report["zusammenfassung"]["rangliste"], key=lambda r: -r["punkte"]):
        print(f"  {row['version']:<4} Teststellungen {row['geloest']:>2}/{len(report['stellungen'])} | "
              f"Turnier {row['punkte']:>4.1f}/{row['partien']} | Genauigkeit {row['genauigkeit']}")


if __name__ == "__main__":
    main()
