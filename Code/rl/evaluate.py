"""
Evaluation des RL-Agenten gegen Vergleichslinien.

Ohne Baseline ist eine Lernkurve wertlos, deshalb wird immer gegen zwei
Vergleichslinien gemessen:

    random    zufällige legale Züge
    material  die bisherige Platzhalter-Engine (Materialbilanz, ein Zug tief)

Endspiele (kqk, krk): Die getestete Strategie führt die starke Seite, ein
Zufallsgegner verteidigt. Gemessen wird, wie oft innerhalb des Limits mattgesetzt
wird und wie lange das dauert. Diese Aufgabe trennt sauber: Materialzählen hilft
hier nicht, weil sich das Material gar nicht ändert.

Volle Partien (full): direkter Vergleich mit abwechselnden Farben, gezählt
werden Siege, Remis und Niederlagen.

Ausgewertet wird über mehrere Seeds, angegeben als Mittelwert und
Standardabweichung - RL-Resultate streuen stark.

Aufruf:
    python -m Code.rl.evaluate --task kqk --model models/value_kqk.npz
"""
from __future__ import annotations

import argparse

import chess
import numpy as np

from .engine import material_evaluation, material_move, random_move, terminal_value, value_move
from .positions import sample_start
from .valuenet import ValueNet


def make_policy(name: str, rng, net: ValueNet | None = None, gamma: float = 0.99):
    """Name -> Funktion, die zu einer Stellung einen Zug liefert."""
    if name == "rl":
        if net is None:
            raise ValueError("Für die Strategie 'rl' wird ein Modell benötigt.")
        return lambda board: value_move(board, net, gamma)
    if name == "material":
        return material_move
    if name == "random":
        return lambda board: random_move(board, rng)
    raise ValueError(f"Unbekannte Strategie: {name}")


def endgame_trial(strong_policy, weak_policy, task: str, rng, games: int, max_plies: int) -> dict:
    """Starke Seite = getestete Strategie. Rückgabe: Mattquote und Zuglänge."""
    mates, draws, truncated, mate_plies = 0, 0, 0, []
    for _ in range(games):
        board = sample_start(task, rng, difficulty=1.0)
        strong = chess.WHITE if material_evaluation(board) > 0 else chess.BLACK
        plies, finished = 0, False
        while plies < max_plies:
            policy = strong_policy if board.turn == strong else weak_policy
            move = policy(board)
            if move is None:
                break
            board.push(move)
            plies += 1
            outcome = terminal_value(board)
            if outcome is not None:
                finished = True
                if board.is_checkmate() and board.turn != strong:
                    mates += 1
                    mate_plies.append(plies)
                else:
                    draws += 1
                break
        if not finished:
            truncated += 1
    return {
        "mate_rate": mates / games,
        "draw_rate": draws / games,
        "truncated_rate": truncated / games,
        "avg_mate_plies": float(np.mean(mate_plies)) if mate_plies else float("nan"),
    }


def match(policy_a, policy_b, rng, games: int, max_plies: int) -> dict:
    """Volle Partien mit abwechselnden Farben, Ergebnis aus Sicht von A."""
    wins, draws, losses = 0, 0, 0
    for game in range(games):
        board = chess.Board()
        color_a = chess.WHITE if game % 2 == 0 else chess.BLACK
        plies, result = 0, "draw"
        while plies < max_plies:
            policy = policy_a if board.turn == color_a else policy_b
            move = policy(board)
            if move is None:
                break
            board.push(move)
            plies += 1
            outcome = terminal_value(board)
            if outcome is not None:
                if board.is_checkmate():
                    result = "win" if board.turn != color_a else "loss"
                break
        wins += result == "win"
        losses += result == "loss"
        draws += result == "draw"
    return {"win_rate": wins / games, "draw_rate": draws / games, "loss_rate": losses / games}


def _aggregate(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: (float(np.nanmean([r[k] for r in runs])), float(np.nanstd([r[k] for r in runs]))) for k in keys}


def evaluate_task(task: str, net: ValueNet | None, games: int, max_plies: int, seeds: int, gamma: float) -> dict:
    """Jede Strategie tritt gegen denselben Zufallsgegner an, damit die Zeilen vergleichbar sind."""
    results: dict[str, dict] = {}
    names = (["rl"] if net is not None else []) + ["material", "random"]
    for name in names:
        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            policy = make_policy(name, rng, net, gamma)
            opponent = make_policy("random", rng)
            if task == "full":
                runs.append(match(policy, opponent, rng, games, max_plies))
            else:
                runs.append(endgame_trial(policy, opponent, task, rng, games, max_plies))
        results[f"{name} vs random"] = _aggregate(runs)

    if task == "full" and net is not None:
        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            runs.append(
                match(make_policy("rl", rng, net, gamma), make_policy("material", rng), rng, games, max_plies)
            )
        results["rl vs material"] = _aggregate(runs)
    return results


def print_results(task: str, results: dict, games: int, seeds: int) -> None:
    print(f"\nAufgabe {task} | {games} Partien x {seeds} Seeds (Mittelwert +/- Standardabweichung)")
    if task == "full":
        print(f"{'Paarung':<20} {'Sieg':>16} {'Remis':>16} {'Niederlage':>16}")
        for name, res in results.items():
            print(
                f"{name:<20} "
                f"{res['win_rate'][0]:>9.1%} +/-{res['win_rate'][1]:>5.1%} "
                f"{res['draw_rate'][0]:>9.1%} +/-{res['draw_rate'][1]:>5.1%} "
                f"{res['loss_rate'][0]:>9.1%} +/-{res['loss_rate'][1]:>5.1%}"
            )
        return
    print(f"{'Paarung':<20} {'Matt gesetzt':>16} {'Remis':>16} {'Halbzüge bis Matt':>20}")
    for name, res in results.items():
        plies = res["avg_mate_plies"][0]
        plies_text = f"{plies:>13.1f}" if not np.isnan(plies) else f"{'-':>13}"
        print(
            f"{name:<20} "
            f"{res['mate_rate'][0]:>9.1%} +/-{res['mate_rate'][1]:>5.1%} "
            f"{res['draw_rate'][0]:>9.1%} +/-{res['draw_rate'][1]:>5.1%} "
            f"{plies_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-Agent gegen Baselines messen")
    parser.add_argument("--task", default="kqk", choices=("kqk", "krk", "full"))
    parser.add_argument("--model", default=None, help="Pfad zum trainierten Modell (.npz)")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()

    net = ValueNet.load(args.model) if args.model else None
    results = evaluate_task(args.task, net, args.games, args.max_plies, args.seeds, args.gamma)
    print_results(args.task, results, args.games, args.seeds)


if __name__ == "__main__":
    main()
