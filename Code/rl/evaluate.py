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

Duell-Modus (--opponent)
------------------------
Ein Modell mit gewählter Suchtiefe gegen genau einen Gegner, verteilt auf
mehrere Prozesse. Jede Partie beginnt mit einer Zufallseröffnung (0 bis 8
Halbzüge, ohne Materialverlust), und jede Eröffnung wird zweimal gespielt, je
einmal mit jeder Farbe. Alle Kandidaten bekommen dieselben Eröffnungen.

Warum Eröffnungen? Zwei deterministische Engines spielen aus der Grundstellung
immer dieselbe Partie. Ein Vergleich über "50 Partien" bestünde dann in
Wahrheit aus zwei Partien, je 25-mal wiederholt — so geschehen bei den ersten
Messungen v3 gegen v2. Gegen Stockfish fällt das weniger auf, weil dessen
Elo-Begrenzung zufällige Fehler einbaut.

Aufruf:
    python -m Code.rl.evaluate --task kqk --model models/value_kqk.npz
    python -m Code.rl.evaluate --task full --model models/full_v3.npz --depth 2 \\
        --opponent stockfish --games 200 --workers 12
    python -m Code.rl.evaluate --task full --model models/full_v3.npz --depth 2 \\
        --opponent models/full_v3.npz --opponent-depth 1 --games 200 --workers 12
"""
from __future__ import annotations

import argparse

import chess
import numpy as np

from .engine import material_evaluation, material_move, random_move, terminal_value, value_move
from .positions import sample_start
from .search import search
from .stockfish import find_stockfish, stockfish_policy
from .valuenet import ValueNet


def make_policy(name: str, rng, net: ValueNet | None = None, gamma: float = 0.99, depth: int = 1):
    """
    Name -> Funktion, die zu einer Stellung einen Zug liefert.

    'depth' gilt für die Strategie 'rl': Tiefe 1 ist die bisherige Zugwahl
    (value_move), ab Tiefe 2 rechnet die Alpha-Beta-Suche.
    """
    if name == "rl":
        if net is None:
            raise ValueError("Für die Strategie 'rl' wird ein Modell benötigt.")
        if depth > 1:
            return lambda board: search(board, net, depth, gamma).move
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


def play_game(
    policy_a, policy_b, color_a: chess.Color, max_plies: int, start: chess.Board | None = None
) -> str:
    """
    Eine Partie, ohne 'start' aus der Grundstellung. Ergebnis aus Sicht von A:
    'win', 'draw' oder 'loss'.
    """
    board = start.copy() if start is not None else chess.Board()
    plies = 0
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
                return "win" if board.turn != color_a else "loss"
            break
    return "draw"


def summarize(results: list[str]) -> dict:
    """Ergebnisliste -> Anteile Sieg, Remis, Niederlage."""
    games = len(results)
    return {
        "win_rate": results.count("win") / games,
        "draw_rate": results.count("draw") / games,
        "loss_rate": results.count("loss") / games,
    }


def match(policy_a, policy_b, rng, games: int, max_plies: int) -> dict:
    """Volle Partien mit abwechselnden Farben, Ergebnis aus Sicht von A."""
    return summarize([
        play_game(policy_a, policy_b, chess.WHITE if game % 2 == 0 else chess.BLACK, max_plies)
        for game in range(games)
    ])


def _aggregate(runs: list[dict]) -> dict:
    keys = runs[0].keys()
    return {k: (float(np.nanmean([r[k] for r in runs])), float(np.nanstd([r[k] for r in runs]))) for k in keys}


def _against_stockfish(
    task: str, net: ValueNet | None, games: int, max_plies: int, seeds: int, gamma: float,
    path: str | None, depth: int | None, elo: int | None,
    nodes: int | None = None, skill: int | None = None, label_suffix: str = "stockfish",
) -> dict:
    """Dieselben Strategien gegen Stockfish, als härteren Massstab."""
    results: dict[str, dict] = {}
    with stockfish_policy(path, depth=depth, nodes=nodes, elo=elo, skill=skill) as engine_policy:
        contenders: list[tuple[str, object]] = []
        if net is not None:
            contenders.append(("rl", net))
            if net.shaped:
                contenders.append(("nur Shaping", ValueNet(hidden_size=net.hidden_size, seed=0, shaped=True)))
        for label, model in contenders:
            runs = []
            for seed in range(seeds):
                rng = np.random.default_rng(1000 + seed)
                runs.append(
                    match(make_policy("rl", rng, model, gamma), engine_policy, rng, games, max_plies)
                )
            results[f"{label} vs {label_suffix}"] = _aggregate(runs)

        runs = []
        for seed in range(seeds):
            rng = np.random.default_rng(1000 + seed)
            runs.append(match(make_policy("material", rng), engine_policy, rng, games, max_plies))
        results[f"material vs {label_suffix}"] = _aggregate(runs)
    return results


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

        # Kontrollversuch: dasselbe Verfahren mit einem untrainierten Netz. Weil
        # V(s) dann nichts beiträgt, misst diese Zeile allein die Stärke des
        # handgeschriebenen Φ. Erst der Abstand zur Zeile darüber ist das, was
        # das Lernen tatsächlich gebracht hat.
        if net.shaped:
            runs = []
            for seed in range(seeds):
                rng = np.random.default_rng(1000 + seed)
                blank = ValueNet(hidden_size=net.hidden_size, seed=0, shaped=True)
                runs.append(
                    match(make_policy("rl", rng, blank, gamma), make_policy("material", rng), rng, games, max_plies)
                )
            results["nur Shaping vs material"] = _aggregate(runs)
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


def duel(
    model: str, depth: int, opponent: tuple, games: int, max_plies: int, workers: int,
    gamma: float = 0.99, openings: bool = True,
) -> dict:
    """
    Modell gegen einen Gegner, siehe Duell-Modus oben. Punkte zählen ein
    Remis als halben Sieg, wie in den bisherigen Tabellen.
    """
    from .parallel import Runner, eval_game_task

    net = ValueNet.load(model)
    with Runner(workers) as runner:
        shared = runner.share(net, 0)
        jobs = [
            (shared, {"game": game, "seed": 1000 + game, "gamma": gamma, "depth": depth,
                      "opponent": opponent, "max_plies": max_plies,
                      "opening": 5000 + game // 2 if openings else None})
            for game in range(games)
        ]
        stats = summarize(runner.map(eval_game_task, jobs))
    stats["points"] = stats["win_rate"] + 0.5 * stats["draw_rate"]
    return stats


def tactics(model: str, depth: int, path: str, gamma: float = 0.99) -> tuple[int, int]:
    """
    Taktik-Test: In jeder Stellung der Datei gibt es laut Stockfish genau einen
    klar besten Zug (mindestens drei Bauern besser als der zweitbeste). Gezählt
    wird, wie oft das Modell ihn findet. Anders als Partien misst das gezielt die
    kurzen Taktiken, bei denen Vorausrechnen helfen sollte.
    """
    import json
    from pathlib import Path

    net = ValueNet.load(model)
    policy = make_policy("rl", None, net, gamma, depth)
    cases = json.loads(Path(path).read_text(encoding="utf-8"))["fälle"]
    solved = sum(policy(chess.Board(case["fen"])).uci() == case["zug"] for case in cases)
    return solved, len(cases)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL-Agent gegen Baselines messen")
    parser.add_argument("--task", default="kqk", choices=("kqk", "krk", "full"))
    parser.add_argument("--model", default=None, help="Pfad zum trainierten Modell (.npz)")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-plies", type=int, default=120)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--stockfish", nargs="?", const="", default=None,
                        help="Zusätzlich gegen Stockfish messen; optional der Pfad zur Programmdatei")
    parser.add_argument("--stockfish-depth", type=int, default=1, help="Suchtiefe von Stockfish")
    parser.add_argument("--stockfish-elo", type=int, default=1320,
                        help="Spielstärke von Stockfish (Minimum 1320). Gemessen die einzige "
                             "Einstellung, bei der unser Agent überhaupt punktet - ohne sie "
                             "verliert auch die Material-Heuristik 100 Prozent der Partien")
    parser.add_argument("--stockfish-skill", type=int, default=None,
                        help="Alternativer Schwächungsschalter, 0 bis 20 (wirkt stärker als Elo)")
    parser.add_argument("--depth", type=int, default=1, help="Vorausschau des Modells in Halbzügen")
    parser.add_argument("--opponent", default=None,
                        help="Duell-Modus: 'stockfish', 'material' oder Pfad zu einem Modell (.npz)")
    parser.add_argument("--opponent-depth", type=int, default=1,
                        help="Vorausschau des Gegners, wenn er ein Modell ist")
    parser.add_argument("--from-start", action="store_true",
                        help="Duell ohne Zufallseröffnungen, alle Partien aus der Grundstellung")
    parser.add_argument("--workers", type=int, default=1, help="Partien, die gleichzeitig laufen")
    parser.add_argument("--tactics", default=None,
                        help="Taktik-Test: Anteil der Stellungen aus dieser Datei, in denen das Modell "
                             "den klar besten Zug findet (z. B. tests/fixtures/taktikstellungen.json)")
    args = parser.parse_args()

    if args.tactics is not None:
        if not args.model:
            parser.error("Der Taktik-Test braucht --model")
        solved, total = tactics(args.model, args.depth, args.tactics, args.gamma)
        print(f"\n{args.model}, Tiefe {args.depth}: {solved} von {total} Taktikstellungen gelöst "
              f"({solved / total:.1%})")
        return

    if args.opponent is not None:
        if args.task != "full" or not args.model:
            parser.error("Der Duell-Modus braucht --task full und --model")
        if args.opponent == "stockfish":
            path = find_stockfish(args.stockfish or None)
            if path is None:
                parser.error("Stockfish nicht gefunden. Pfad über --stockfish angeben.")
            spec = ("stockfish", path, args.stockfish_depth, args.stockfish_elo)
            label = f"Stockfish (Elo {args.stockfish_elo})"
        elif args.opponent == "material":
            spec, label = ("material",), "Material-Heuristik"
        else:
            spec = ("modell", args.opponent, args.opponent_depth)
            label = f"{args.opponent}, Tiefe {args.opponent_depth}"
        stats = duel(args.model, args.depth, spec, args.games, args.max_plies, args.workers,
                     args.gamma, openings=not args.from_start)
        print(f"\n{args.model}, Tiefe {args.depth}  gegen  {label}  ({args.games} Partien)")
        print(f"  Sieg {stats['win_rate']:6.1%}   Remis {stats['draw_rate']:6.1%}   "
              f"Niederlage {stats['loss_rate']:6.1%}   Punkte {stats['points']:6.1%}")
        return

    net = ValueNet.load(args.model) if args.model else None
    results = evaluate_task(args.task, net, args.games, args.max_plies, args.seeds, args.gamma)

    if args.stockfish is not None and args.task != "full":
        print("Hinweis: Der Stockfish-Vergleich spielt ganze Partien und passt nicht zu den Endspielaufgaben.")
    elif args.stockfish is not None:
        results.update(
            _against_stockfish(
                args.task, net, args.games, args.max_plies, args.seeds, args.gamma,
                args.stockfish or None, args.stockfish_depth, args.stockfish_elo,
                skill=args.stockfish_skill,
                label_suffix=f"stockfish (Elo {args.stockfish_elo})",
            )
        )

    print_results(args.task, results, args.games, args.seeds)


if __name__ == "__main__":
    main()
