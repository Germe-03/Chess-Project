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

Aufruf:
    python -m Code.rl.train --task kqk --episodes 20000
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np

from .engine import score_moves, terminal_value
from .evaluate import endgame_trial, make_policy, match
from .features import FEATURE_SIZE, board_to_features
from .positions import OPENING_PLIES, sample_start
from .shaping import potential
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


def run_eval(net: ValueNet, task: str, gamma: float, games: int, max_plies: int) -> tuple[float, dict]:
    """
    Zwischenstand für die Lernkurve und die Auswahl des besten Modells.

    Endspiele werden an der Mattquote gegen einen Zufallsgegner gemessen, ganze
    Partien an Siegen minus Niederlagen gegen die Material-Heuristik.

    Zwei bewusste Entscheidungen für ganze Partien. Gegner ist die Heuristik und
    nicht der Zufall, weil genau das die Frage ist: Schlägt der gelernte Agent
    die bisherige Engine? Und gewertet wird die Differenz statt nur der Siege,
    weil früh im Training fast alles remis endet oder ins Zuglimit läuft - eine
    reine Siegquote bliebe lange bei null, und als bestes Modell würde der erste
    Zwischenstand gespeichert. Nach Stunden Training wäre das Ergebnis weg.
    """
    rng = np.random.default_rng(99)
    agent = make_policy("rl", rng, net, gamma)
    if task == "full":
        stats = match(agent, make_policy("material", rng), rng, games, max_plies)
        return stats["win_rate"] - stats["loss_rate"], stats
    stats = endgame_trial(agent, make_policy("random", rng), task, rng, games, max_plies)
    return stats["mate_rate"], stats


def _linear(start: float, end: float, progress: float) -> float:
    return start + (end - start) * min(max(progress, 0.0), 1.0)


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
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", default=None,
                        help="Vorhandenes Modell laden und weitertrainieren statt bei null zu beginnen")
    parser.add_argument("--out", default=None)
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    eps_episodes = args.eps_episodes or max(1, int(0.7 * args.episodes))
    curriculum_episodes = args.curriculum_episodes if args.curriculum_episodes is not None else int(0.5 * args.episodes)
    out_path = Path(args.out or f"models/value_{args.task}.npz")
    best_path = out_path.with_name(f"{out_path.stem}_best{out_path.suffix}")
    log_path = Path(args.log or f"logs/train_{args.task}.csv")
    eval_path = log_path.with_name(f"{log_path.stem}_eval.csv")
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Das Netz zuerst bauen: Beim Fortsetzen kommt die Shaping-Einstellung aus
    # der Modelldatei, nicht vom Aufruf - und das Protokoll soll den echten
    # Wert festhalten.
    if args.resume:
        net = ValueNet.load(args.resume)
    else:
        net = ValueNet(hidden_size=args.hidden, seed=args.seed, shaped=args.shaping)

    print("Hyperparameter")
    for key, value in (
        ("Aufgabe", args.task),
        ("Episoden", args.episodes),
        ("gamma", args.gamma),
        ("lambda", args.td_lambda),
        ("Lernrate", args.lr),
        ("Hidden-Units", net.hidden_size),
        ("Shaping", "ja" if net.shaped else "nein (nur Partieausgang)"),
        ("Start", f"Fortsetzung von {args.resume}" if args.resume else "neues Netz"),
        ("epsilon", f"{args.eps_start} -> {args.eps_end} über {eps_episodes} Episoden"),
        (
            "Curriculum",
            f"Zufallseröffnung, 0 bis {OPENING_PLIES} Halbzüge"
            if args.task == "full"
            else f"difficulty 0 -> 1 über {curriculum_episodes} Episoden",
        ),
        ("max. Halbzüge", args.max_plies),
        ("Seed", args.seed),
    ):
        print(f"  {key:<16} {value}")
    print(f"  {'Modell':<16} {out_path}")
    print(f"  {'Logs':<16} {log_path} / {eval_path}\n")

    rng = np.random.default_rng(args.seed)

    recent_mates = deque(maxlen=args.log_every)
    recent_plies = deque(maxlen=args.log_every)
    recent_errors = deque(maxlen=args.log_every)
    # -inf statt 0, weil der Wert bei ganzen Partien negativ sein kann
    # (mehr Niederlagen als Siege gegen die Heuristik).
    best_score = float("-inf")
    eval_label = "gegen Heuristik (Siege - Niederlagen)" if args.task == "full" else "gegen Zufall (Mattquote)"
    started = time.time()

    with log_path.open("w", newline="", encoding="utf-8") as log_file, \
            eval_path.open("w", newline="", encoding="utf-8") as eval_file:
        log_writer = csv.writer(log_file)
        log_writer.writerow(["episode", "difficulty", "epsilon", "plies", "result", "td_error"])
        eval_writer = csv.writer(eval_file)
        eval_writer.writerow(
            ["episode", "score", "mate_rate", "win_rate", "draw_rate", "loss_rate", "avg_mate_plies"]
        )

        for episode in range(1, args.episodes + 1):
            difficulty = 1.0 if curriculum_episodes <= 0 else min(1.0, episode / curriculum_episodes)
            epsilon = _linear(args.eps_start, args.eps_end, episode / eps_episodes)

            X, targets, result, plies = play_episode(
                net, rng, args.task, args.gamma, epsilon, args.max_plies, difficulty, args.td_lambda
            )
            td_error = net.update(X, targets, args.lr) if len(X) else 0.0

            recent_mates.append(result == "mate")
            recent_plies.append(plies)
            recent_errors.append(td_error)
            log_writer.writerow([episode, f"{difficulty:.3f}", f"{epsilon:.3f}", plies, result, f"{td_error:.5f}"])

            if episode % args.log_every == 0:
                rate = time.time() - started
                print(
                    f"Episode {episode:>6} | difficulty {difficulty:4.2f} | epsilon {epsilon:4.2f} "
                    f"| Matt {np.mean(recent_mates):5.1%} | {np.mean(recent_plies):5.1f} Halbzüge "
                    f"| TD-Fehler {np.mean(recent_errors):.4f} | {episode / rate:5.1f} Episoden/s",
                    flush=True,
                )
                log_file.flush()

            if args.eval_every > 0 and episode % args.eval_every == 0:
                score, stats = run_eval(net, args.task, args.gamma, args.eval_games, args.max_plies)
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
                # Immer den aktuellen Stand sichern. Vorher wurde nur bei einer
                # Verbesserung gespeichert - sobald die Wertung ihr Maximum
                # erreicht, stand danach kein Fortschritt mehr auf der Platte
                # und das fertige Netz war nach Prozessende verloren.
                net.save(out_path)
                marker = ""
                if score > best_score:
                    best_score = score
                    net.save(best_path)
                    marker = "  <- bester Stand"
                print(f"   Evaluation {eval_label}: {score:+6.1%}{marker}", flush=True)

    net.save(out_path)
    if best_score == float("-inf"):   # es lief gar keine Zwischenevaluation
        net.save(best_path)
        best_score, _ = run_eval(net, args.task, args.gamma, args.eval_games, args.max_plies)

    minutes = (time.time() - started) / 60
    print(f"\nFertig in {minutes:.1f} Minuten. Bestes Modell {eval_label}: {best_score:+.1%} -> {out_path}")


if __name__ == "__main__":
    main()
