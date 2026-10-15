"""
Self-Play-Training der Bewertungsfunktion V(s) mit TD(0).

MDP-Formulierung
----------------
Zustand s   Stellung inklusive Zugpartei, kodiert als 768 Merkmale (features.py)
Aktion  a   ein legaler Zug
Reward  r   nur am Partieende, aus Sicht der Partei, die gezogen hat: +1 für
            Matt gesetzt, 0 für Remis. Während der Partie gibt es nichts.
Ende        Matt, Patt, ungenügendes Material, 50-Züge-Regel. Zusätzlich
            Abbruch nach max_plies - dort wird weiter gebootstrappt, weil die
            Partie nicht wirklich zu Ende ist (terminated vs. truncated).
gamma       0.99

Bewusst kein Schrittabzug: In einem Nullsummenspiel würde er beide Seiten gleich
bestrafen und dem Verteidiger fälschlich das schnelle Verlieren belohnen. Die
Motivation, schnell mattzusetzen, kommt stattdessen aus dem Discount gamma < 1 -
also aus der zweiten der beiden Varianten des Grid-World-Beispiels.

Mit --shaping kommen Zwischenbelohnungen dazu (siehe shaping.py). In ganzen
Partien ist das nötig, weil der Agent sonst hunderte Halbzüge lang gar kein
Signal bekommt. Die Belohnung wird dann um gamma·(-Φ(s')) - Φ(s) ergänzt, und
das Netz lernt nur noch die Korrektur zu Φ. Für die Endspiele bleibt es
ausgeschaltet: Dort ist Material konstant und der Weg zum Matt kurz genug.

Lernschritt
-----------
TD mit Funktionsapproximation, in Negamax-Form. Das Ein-Schritt-Ziel lautet

    Ziel  = r + gamma * ( -V(s') )
    V(s) <- V(s) + alpha * ( Ziel - V(s) )

Das Minuszeichen ist nötig, weil V(s') aus Sicht des Gegners gilt.

Reines TD(0) trägt den Mattwert aber nur einen Halbzug pro Update zurück, und
ein Matt liegt aus einer zufälligen Endspielstellung 20 bis 30 Halbzüge entfernt
- die Zuordnung von Belohnung zu Ursache dauert dann ewig. Deshalb TD(lambda) in
der Vorwärtssicht: Am Episodenende werden die Ziele rückwärts zu lambda-Returns
verrechnet,

    G_t = (1 - lambda) * Ziel_t  -  gamma * lambda * G_(t+1)

wobei der letzte Schritt sein Ein-Schritt-Ziel behält. lambda = 0 ergibt wieder
TD(0), lambda = 1 den reinen Monte-Carlo-Return. Das Minuszeichen vor G_(t+1)
ist erneut die Negamax-Regel.

Die Updates einer Episode werden gesammelt und am Episodenende als ein Batch
angewandt: deutlich schneller und ruhiger als ein Update pro Halbzug.

Self-Play: Ein einziges Netz spielt beide Seiten. Weil die Merkmale immer aus
Sicht der Zugpartei kodiert sind, ist das konsistent.

Exploration: epsilon-greedy, epsilon fällt linear von eps_start auf eps_end.
Gelernt wird auch aus den zufälligen Zügen (on-policy TD).

Training mit Vorausrechnen (ab v5)
----------------------------------
Mit --search-depth 2 oder mehr spielt die Engine im Training mit derselben
Alpha-Beta-Suche wie später im Einsatz, und gelernt wird mit TD-Leaf(λ)
(tdleaf.py). Ohne die Option bleibt alles beim Verfahren oben — damit lassen
sich v1 bis v3 jederzeit nachvollziehen.

Mit --workers N laufen N Partien gleichzeitig auf N Prozessorkernen
(parallel.py). Das ist für das Training mit Suche praktisch Pflicht: Eine
Partie dauert dann etwa eine Minute statt Sekundenbruchteile.

Aufruf:
    python -m Code.rl.train --task kqk --episodes 20000
    python -m Code.rl.train --task full --resume models/full_v3.npz --search-depth 2 --workers 12 ...
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np

from .engine import score_moves, terminal_value
from .evaluate import endgame_trial, make_policy, summarize
from .features import FEATURE_SIZE, board_to_features
from .parallel import Runner, episode_task, eval_game_task
from .positions import OPENING_PLIES, sample_start
from .shaping import potential
from .stockfish import find_stockfish
from .valuenet import ValueNet


def play_episode(
    net: ValueNet,
    rng,
    task: str,
    gamma: float,
    epsilon: float,
    max_plies: int,
    difficulty: float,
    td_lambda: float,
):
    """Eine Self-Play-Partie. Rückgabe: Merkmale, lambda-Returns, Ausgang, Halbzüge."""
    board = sample_start(task, rng, difficulty)
    states: list[np.ndarray] = []
    targets: list[float] = []
    phis: list[float] = []
    result, plies = "truncated", 0

    while plies < max_plies:
        moves, scores = score_moves(board, net, gamma)
        if not moves:
            break
        if rng.random() < epsilon:
            index = int(rng.integers(len(moves)))
        else:
            best = np.flatnonzero(scores == scores.max())
            index = int(rng.choice(best))

        # scores[index] ist bereits r + gamma * (-V(s')) - genau das TD-Ziel.
        states.append(board_to_features(board))
        targets.append(float(scores[index]))
        if net.shaped:
            phis.append(potential(board))

        board.push(moves[index])
        plies += 1
        outcome = terminal_value(board)
        if outcome is not None:
            result = "mate" if outcome == 1.0 else "draw"
            break

    X = np.stack(states) if states else np.zeros((0, FEATURE_SIZE), dtype=np.float32)
    returns = np.asarray(targets, dtype=np.float32)
    for t in range(len(returns) - 2, -1, -1):
        returns[t] = (1.0 - td_lambda) * returns[t] - gamma * td_lambda * returns[t + 1]

    # Mit Shaping ist der eben berechnete lambda-Return die Schätzung für die
    # Gesamtbewertung V + Φ. Das Netz lernt davon nur den Rest, also abzüglich Φ(s).
    if net.shaped:
        returns -= np.asarray(phis, dtype=np.float32)
    return X, returns, result, plies


def run_eval(
    net: ValueNet,
    task: str,
    gamma: float,
    games: int,
    max_plies: int,
    opponent: tuple = ("material",),
    runner: Runner | None = None,
    depth: int = 1,
    token: int = 0,
) -> tuple[float, dict]:
    """
    Zwischenstand für die Lernkurve und die Auswahl des besten Modells.

    Endspiele werden an der Mattquote gegen einen Zufallsgegner gemessen, ganze
    Partien an Siegen minus Niederlagen gegen den Gegner 'opponent' (eine
    Beschreibung wie ("stockfish", Pfad, Tiefe, Elo), siehe parallel.py). Die
    Partien laufen über den Runner, also auf Wunsch parallel. Gespielt wird mit
    derselben Suchtiefe wie im Training.

    Gewertet wird die Differenz statt nur der Siege, weil früh im Training fast
    alles remis endet oder ins Zuglimit läuft - eine reine Siegquote bliebe lange
    bei null.

    Zur Wahl des Gegners: Die Material-Heuristik ist als Massstab ab etwa 1000
    Episoden erschöpft, der Agent schlägt sie dann in jeder Partie. Eine Messlatte
    bei +100 Prozent misst nichts mehr, und die Auswahl des besten Modells wird
    zum Zufall. Für längere Läufe gehört deshalb --eval-opponent stockfish dazu.
    """
    if task == "full":
        runner = runner or Runner(1)
        shared = runner.share(net, token)
        jobs = [
            (shared, {"game": game, "seed": 99 + game, "gamma": gamma, "depth": depth,
                      "opponent": opponent, "max_plies": max_plies, "opening": 5000 + game // 2})
            for game in range(games)
        ]
        stats = summarize(runner.map(eval_game_task, jobs))
        return stats["win_rate"] - stats["loss_rate"], stats
    rng = np.random.default_rng(99)
    agent = make_policy("rl", rng, net, gamma)
    stats = endgame_trial(agent, make_policy("random", rng), task, rng, games, max_plies)
    return stats["mate_rate"], stats


def _linear(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


class RunDir:
    """
    Lauf-Mappe für ein Training, das unterbrochen und später fortgesetzt wird.

    Inhalt:
        state.json      Einstellungen des Laufs und Fortschritt
        checkpoint.npz  Gewichte samt Optimierer nach der letzten fertigen Partie
        train.csv       Protokoll je Partie, beim Fortsetzen ergänzt
        snapshots/      Zwischenstände für die Lernkurve, z. B. ep0025.npz
        STOP            Liegt diese Datei da, hält das Training sauber an.

    Gesichert wird nach jeder fertigen Partie. Ein Absturz, ein Neustart oder
    ein leerer Akku kosten deshalb höchstens die Partien, die gerade liefen.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.state_path = self.path / "state.json"
        self.checkpoint = self.path / "checkpoint.npz"
        self.stop_file = self.path / "STOP"

    def exists(self) -> bool:
        return self.state_path.exists() and self.checkpoint.exists()

    def load(self) -> dict:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def save(self, net: ValueNet, state: dict) -> None:
        """
        Erst in eine Hilfsdatei schreiben, dann umbenennen. Das Umbenennen ist
        ein einziger Schritt des Dateisystems; bricht der Rechner mitten im
        Schreiben ab, bleibt der vorige Stand unversehrt.
        """
        self.path.mkdir(parents=True, exist_ok=True)
        temporary = self.path / "checkpoint.tmp.npz"
        net.save(temporary, optimizer=True)
        os.replace(temporary, self.checkpoint)
        temporary = self.path / "state.tmp.json"
        temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def stop_requested(self) -> bool:
        # Im Windows-Explorer angelegt, heisst die Datei oft STOP.txt.
        return self.stop_file.exists() or self.stop_file.with_suffix(".txt").exists()

    def clear_stop(self) -> None:
        self.stop_file.unlink(missing_ok=True)
        self.stop_file.with_suffix(".txt").unlink(missing_ok=True)


# Einstellungen, die zur Maschine gehören und nicht zum Lauf. Sie dürfen sich
# beim Fortsetzen ändern, alles andere kommt aus state.json.
_SESSION_ARGS = ("workers", "run_dir", "stop")


def _progress(episode, total, difficulty, epsilon, mates, plies, errors, done, elapsed) -> str:
    """Fortschrittszeile mit geschätzter Restzeit aus dem Tempo dieser Sitzung."""
    rate = done / elapsed if elapsed > 0 else 0.0
    rest = (total - episode) / rate / 3600 if rate > 0 else float("nan")
    return (
        f"Episode {episode:>6}/{total} | difficulty {difficulty:4.2f} | epsilon {epsilon:4.2f} "
        f"| Matt {np.mean(mates):5.1%} | {np.mean(plies):5.1f} Halbzüge "
        f"| TD-Fehler {np.mean(errors):.4f} | {rate * 60:6.2f} Episoden/min | Rest ca. {rest:4.1f} h"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-Play-Training der Bewertungsfunktion")
    parser.add_argument("--task", default="kqk", choices=("kqk", "krk", "full"))
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--td-lambda", type=float, default=0.9, help="0 = TD(0), 1 = Monte Carlo")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument(
        "--shaping",
        action="store_true",
        help="Zwischenbelohnungen aus Material, Zentrum, Abtausch und Schach - für ganze Partien",
    )
    parser.add_argument("--eps-start", type=float, default=0.9)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-episodes", type=int, default=None, help="Standard: 70 Prozent der Episoden")
    parser.add_argument("--curriculum-episodes", type=int, default=None, help="Standard: 50 Prozent der Episoden")
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--eval-every", type=int, default=1000,
                        help="Zwischenprüfung alle N Episoden; 0 heisst keine")
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--eval-opponent", default="material", choices=("material", "stockfish"),
                        help="Gegner der Zwischenprüfung bei ganzen Partien. Die Heuristik ist ab "
                             "etwa 1000 Episoden erschöpft und misst dann nichts mehr")
    parser.add_argument("--eval-depth", type=int, default=None,
                        help="Vorausschau in der Zwischenprüfung; Standard ist die Trainingstiefe")
    parser.add_argument("--stockfish", nargs="?", const="", default=None,
                        help="Pfad zur Stockfish-Programmdatei; leer heisst automatisch suchen")
    parser.add_argument("--stockfish-elo", type=int, default=1320,
                        help="Spielstärke des Prüfgegners (Minimum 1320)")
    parser.add_argument("--stockfish-depth", type=int, default=1, help="Suchtiefe des Prüfgegners")
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="Vorhandenes Modell laden und weitertrainieren statt bei null zu beginnen")
    parser.add_argument("--widen", type=int, default=None,
                        help="Beim Fortsetzen die versteckte Schicht auf diese Breite bringen, "
                             "ohne die gelernte Funktion zu verändern (Net2Net)")
    parser.add_argument("--search-depth", type=int, default=1,
                        help="Vorausschau in Halbzügen beim Spielen. Ab 2 wird mit TD-Leaf gelernt "
                             "(nur --task full)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Partien, die gleichzeitig auf eigenen Prozessorkernen laufen")
    parser.add_argument("--run-dir", default=None,
                        help="Lauf-Mappe: Stand nach jeder Partie sichern; derselbe Befehl setzt "
                             "einen unterbrochenen Lauf fort")
    parser.add_argument("--snapshot-every", type=int, default=0,
                        help="Alle N Episoden einen Zwischenstand in <run-dir>/snapshots ablegen")
    parser.add_argument("--stop", action="store_true",
                        help="Laufendes Training in --run-dir sauber anhalten und sofort beenden")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    run = RunDir(args.run_dir) if args.run_dir else None
    if args.stop:
        if run is None:
            parser.error("--stop braucht --run-dir")
        run.path.mkdir(parents=True, exist_ok=True)
        run.stop_file.write_text("Training anhalten\n", encoding="utf-8")
        print(f"Stopp angefordert ({run.stop_file}). Das Training sichert seinen Stand und "
              "hält innerhalb weniger Sekunden an.")
        return

    resumed = run is not None and run.exists()
    if resumed:
        # Fortsetzen: Alle Einstellungen kommen aus dem Lauf selbst, damit sich
        # nicht versehentlich mitten im Training etwas ändert. Nur die Zahl der
        # Prozesse darf neu gewählt werden.
        state = run.load()
        if state.get("finished"):
            print(f"Der Lauf in {run.path} ist bereits abgeschlossen ({state['episodes_done']} Episoden).")
            return
        session_args = {name: getattr(args, name) for name in _SESSION_ARGS}
        args = argparse.Namespace(**state["config"], **session_args)
        net = ValueNet.load(run.checkpoint)
    else:
        if args.search_depth > 1 and args.task != "full":
            parser.error("--search-depth ab 2 ist nur für --task full vorgesehen")
        # Das Netz zuerst bauen: Beim Fortsetzen kommt die Shaping-Einstellung
        # aus der Modelldatei, nicht vom Aufruf - und das Protokoll soll den
        # echten Wert festhalten.
        if args.resume:
            net = ValueNet.load(args.resume)
            if args.widen and args.widen > net.hidden_size:
                net = net.widened(args.widen, seed=args.seed)
        else:
            net = ValueNet(hidden_size=args.hidden, seed=args.seed, shaped=args.shaping)
        state = {
            "config": {key: value for key, value in vars(args).items() if key not in _SESSION_ARGS},
            "episodes_done": 0,
            "next_game": 1,
            "best_score": None,
            "seconds": 0.0,
            "sessions": 0,
            "finished": False,
        }
    if run is not None:
        run.clear_stop()
    state["sessions"] += 1

    eps_episodes = args.eps_episodes or max(1, int(0.7 * args.episodes))
    curriculum_episodes = args.curriculum_episodes if args.curriculum_episodes is not None else int(0.5 * args.episodes)
    eval_depth = args.eval_depth or args.search_depth
    out_path = Path(args.out or f"models/value_{args.task}.npz")
    best_path = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    log_path = run.path / "train.csv" if run is not None else Path(args.log or f"logs/train_{args.task}.csv")
    eval_path = log_path.with_name(f"{log_path.stem}_eval.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    print("Hyperparameter")
    for key, value in (
        ("Aufgabe", args.task),
        ("Episoden", args.episodes),
        ("gamma", args.gamma),
        ("lambda", args.td_lambda),
        ("Lernrate", args.lr),
        ("Hidden-Units", net.hidden_size),
        ("Shaping", "ja" if net.shaped else "nein (nur Partieausgang)"),
        (
            "Start",
            (f"Fortsetzung von {args.resume}" + (f", verbreitert auf {args.widen}" if args.widen else ""))
            if args.resume else "neues Netz",
        ),
        ("epsilon", f"{args.eps_start} -> {args.eps_end} über {eps_episodes} Episoden"),
        (
            "Curriculum",
            f"Zufallseröffnung, 0 bis {OPENING_PLIES} Halbzüge"
            if args.task == "full"
            else f"difficulty 0 -> 1 über {curriculum_episodes} Episoden",
        ),
        ("max. Halbzüge", args.max_plies),
        (
            "Suchtiefe",
            f"{args.search_depth} Halbzüge + Ruhesuche, Lernverfahren TD-Leaf(lambda)"
            if args.search_depth > 1 else "1 Halbzug, Lernverfahren TD(lambda)",
        ),
        ("Prozesse", args.workers),
        (
            "Prüfgegner",
            "keine Zwischenprüfung" if args.eval_every <= 0
            else f"Stockfish Elo {args.stockfish_elo}, Tiefe {args.stockfish_depth}, wir mit Tiefe {eval_depth}"
            if args.task == "full" and args.eval_opponent == "stockfish"
            else ("Material-Heuristik" if args.task == "full" else "Zufall"),
        ),
        ("Seed", args.seed),
    ):
        print(f"  {key:<16} {value}")
    print(f"  {'Modell':<16} {out_path}")
    print(f"  {'Logs':<16} {log_path} / {eval_path}")
    if run is not None:
        stand = (f"Fortsetzung, {state['episodes_done']} von {args.episodes} Episoden fertig, "
                 f"Sitzung {state['sessions']}" if resumed else "neuer Lauf")
        print(f"  {'Lauf-Mappe':<16} {run.path} ({stand})")
        print(f"  {'Anhalten':<16} python -m Code.rl.train --run-dir {run.path} --stop")
    print()

    # Der Prüfgegner als Beschreibung statt als offene Engine: So lässt er sich
    # an andere Prozesse schicken, und jeder Prozess öffnet seine eigene.
    eval_opponent: tuple = ("material",)
    if args.task == "full" and args.eval_opponent == "stockfish" and args.eval_every > 0:
        path = find_stockfish(args.stockfish or None)
        # Bewusst vor der ersten Episode: Fehlt die Programmdatei, soll das
        # sofort auffallen und nicht erst nach einer Stunde Rechnen.
        if path is None:
            raise SystemExit("Stockfish nicht gefunden. Pfad über --stockfish angeben oder STOCKFISH_PATH setzen.")
        eval_opponent = ("stockfish", path, args.stockfish_depth, args.stockfish_elo)

    recent_mates = deque(maxlen=args.log_every)
    recent_plies = deque(maxlen=args.log_every)
    recent_errors = deque(maxlen=args.log_every)
    # -inf statt 0, weil der Wert bei ganzen Partien negativ sein kann
    # (mehr Niederlagen als Siege gegen die Heuristik).
    best_score = state["best_score"] if state["best_score"] is not None else float("-inf")
    if args.task != "full":
        eval_label = "gegen Zufall (Mattquote)"
    elif args.eval_opponent == "stockfish":
        eval_label = f"gegen Stockfish Elo {args.stockfish_elo} (Siege - Niederlagen)"
    else:
        eval_label = "gegen Heuristik (Siege - Niederlagen)"
    started = time.time()
    seconds_before = state["seconds"]

    def schedule(game: int) -> tuple[float, float]:
        """difficulty und epsilon für die Partie mit dieser Nummer."""
        difficulty = 1.0 if curriculum_episodes <= 0 else min(1.0, game / curriculum_episodes)
        return difficulty, _linear(args.eps_start, args.eps_end, game / eps_episodes)

    updates = state["episodes_done"]   # Kennung des aktuellen Gewichtsstands für die Prozesse

    def make_job(submitted: list[int]):
        # Partien werden fortlaufend nummeriert, auch über Unterbrechungen
        # hinweg. Jede Nummer hat ihren eigenen Zufallsstrom aus Seed und
        # Nummer; so hängt eine Partie nicht davon ab, welcher Prozess sie
        # spielt, und nach einer Pause wiederholt sich keine Partie.
        game = state["next_game"]
        state["next_game"] += 1
        submitted.append(game)
        difficulty, epsilon = schedule(game)
        params = {
            "seed": [args.seed, game], "task": args.task, "gamma": args.gamma, "epsilon": epsilon,
            "max_plies": args.max_plies, "difficulty": difficulty, "td_lambda": args.td_lambda,
            "depth": args.search_depth,
        }
        return runner.share(net, updates), params

    should_stop = run.stop_requested if run is not None else None
    log_mode = "a" if resumed and log_path.exists() else "w"
    episode = start_episode = state["episodes_done"]
    stopped = False
    runner = Runner(args.workers)
    try:
        with log_path.open(log_mode, newline="", encoding="utf-8") as log_file, \
                eval_path.open(log_mode, newline="", encoding="utf-8") as eval_file:
            log_writer = csv.writer(log_file)
            eval_writer = csv.writer(eval_file)
            if log_mode == "w":
                log_writer.writerow(
                    ["episode", "difficulty", "epsilon", "plies", "result", "td_error", "seconds", "session"]
                )
                eval_writer.writerow(
                    ["episode", "score", "mate_rate", "win_rate", "draw_rate", "loss_rate", "avg_mate_plies"]
                )

            while episode < args.episodes and not stopped:
                # Bis zur nächsten Zwischenauswertung spielen; dort warten, bis
                # alle laufenden Partien fertig sind, damit ein klarer Stand
                # gemessen wird.
                segment = args.episodes - episode
                if args.eval_every > 0:
                    segment = min(segment, args.eval_every - episode % args.eval_every)
                submitted: list[int] = []
                jobs = runner.stream(episode_task, lambda _: make_job(submitted), segment, should_stop)
                for index, (X, targets, result, plies, seconds) in jobs:
                    episode += 1
                    difficulty, epsilon = schedule(submitted[index])
                    td_error = net.update(X, targets, args.lr) if len(X) else 0.0
                    updates += 1

                    recent_mates.append(result == "mate")
                    recent_plies.append(plies)
                    recent_errors.append(td_error)
                    log_writer.writerow([
                        episode, f"{difficulty:.3f}", f"{epsilon:.3f}", plies, result,
                        f"{td_error:.5f}", f"{seconds:.1f}", state["sessions"],
                    ])
                    log_file.flush()

                    state["episodes_done"] = episode
                    state["seconds"] = seconds_before + time.time() - started
                    if run is not None:
                        run.save(net, state)
                        if args.snapshot_every > 0 and episode % args.snapshot_every == 0:
                            net.save(run.path / "snapshots" / f"ep{episode:04d}.npz")

                    if episode % args.log_every == 0:
                        print(_progress(episode, args.episodes, difficulty, epsilon, recent_mates,
                                        recent_plies, recent_errors, episode - start_episode,
                                        time.time() - started),
                              flush=True)

                if should_stop is not None and should_stop():
                    stopped = True
                    break

                if args.eval_every > 0 and episode % args.eval_every == 0:
                    score, stats = run_eval(
                        net, args.task, args.gamma, args.eval_games, args.max_plies, eval_opponent,
                        runner, eval_depth, updates,
                    )
                    eval_writer.writerow([
                        episode,
                        f"{score:.4f}",
                        f"{stats.get('mate_rate', float('nan')):.4f}",
                        f"{stats.get('win_rate', float('nan')):.4f}",
                        f"{stats.get('draw_rate', float('nan')):.4f}",
                        f"{stats.get('loss_rate', float('nan')):.4f}",
                        f"{stats.get('avg_mate_plies', float('nan')):.2f}",
                    ])
                    eval_file.flush()
                    # Immer den aktuellen Stand sichern. Vorher wurde nur bei
                    # einer Verbesserung gespeichert - sobald die Wertung ihr
                    # Maximum erreicht, stand danach kein Fortschritt mehr auf
                    # der Platte und das fertige Netz war nach Prozessende verloren.
                    net.save(out_path)
                    marker = ""
                    if score > best_score:
                        best_score = score
                        state["best_score"] = score
                        net.save(best_path)
                        marker = "  <- bester Stand"
                    if run is not None:
                        run.save(net, state)
                    print(f"   Evaluation {eval_label}: {score:+6.1%}{marker}", flush=True)

            if not stopped:
                net.save(out_path)
                # Es lief gar keine Zwischenevaluation, obwohl eine verlangt war.
                if args.eval_every > 0 and best_score == float("-inf"):
                    net.save(best_path)
                    best_score, _ = run_eval(
                        net, args.task, args.gamma, args.eval_games, args.max_plies, eval_opponent,
                        runner, eval_depth, updates,
                    )
    except KeyboardInterrupt:
        stopped = True
    finally:
        # Beim Anhalten nicht auf angefangene Partien warten, sondern die
        # Prozesse beenden; ihr Stand ginge ohnehin verloren.
        if stopped:
            runner.terminate()
        else:
            runner.close()

    hours = (time.time() - started) / 3600
    if stopped:
        if run is not None:
            state["seconds"] = seconds_before + time.time() - started
            run.save(net, state)
            run.clear_stop()
            print(f"\nAngehalten nach {episode} von {args.episodes} Episoden ({hours:.1f} h in dieser Sitzung). "
                  f"Fortsetzen mit demselben Befehl.")
        else:
            net.save(out_path)
            print(f"\nAbgebrochen nach {episode} Episoden. Aktueller Stand gesichert: {out_path}")
        return

    if run is not None:
        state["finished"] = True
        state["seconds"] = seconds_before + time.time() - started
        run.save(net, state)
    total_hours = state["seconds"] / 3600
    if args.eval_every > 0:
        print(f"\nFertig nach {total_hours:.1f} Stunden Rechenzeit. Bestes Modell {eval_label}: "
              f"{best_score:+.1%} -> {out_path}")
    else:
        print(f"\nFertig nach {total_hours:.1f} Stunden Rechenzeit -> {out_path}")


if __name__ == "__main__":
    main()
