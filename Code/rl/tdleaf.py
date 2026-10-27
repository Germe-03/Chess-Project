"""
TD-Leaf(λ): Lernen mit Vorausrechnen.

Bis v3 lernt das Netz mit TD(λ) aus Ein-Zug-Werten: Für jede Stellung s_t der
Partie wird V(s_t) an den λ-Return herangezogen (train.play_episode). Spielt die
Engine aber mit Suche, ist das nicht mehr ganz passend — sie bewertet dann gar
nicht s_t, sondern die Stellungen am Ende ihrer Varianten, meist ruhige
Stellungen nach einem Schlagabtausch. Genau dort sollte das Netz gut sein.

TD-Leaf (Baxter, Tridgell & Weaver 1998, Schachprogramm KnightCap) zieht deshalb
nicht V(s_t) nach, sondern V(Blatt_t): die Stellung am Ende der Hauptvariante,
deren Bewertung den Suchwert W_t bestimmt hat (siehe search.py):

    W_t = (-gamma)^k * Gesamtbewertung(Blatt_t),   k = Länge der Variante

Ablauf einer Episode
--------------------
1. Zufallseröffnung wie bisher (positions.py).
2. In jeder Stellung s_t rechnet die Suche mit Tiefe d: Suchwert W_t,
   Hauptvariante, Blatt_t.
3. Gezogen wird der beste Zug, mit Wahrscheinlichkeit epsilon ein zufälliger.
4. Ein-Schritt-Ziel wie bisher, nur mit Suchwerten statt Ein-Zug-Werten:

       Ziel_t = r                    wenn der Zug die Partie beendet
       Ziel_t = -gamma * W_(t+1)     sonst

   Wird die Partie am Zuglimit abgebrochen, liefert eine letzte Suche W_T —
   abgebrochen ist nicht beendet (truncated vs. terminated).
5. λ-Return mit derselben Rekursion wie in train.py:

       G_t = (1 - λ) * Ziel_t  -  gamma * λ * G_(t+1)

6. Das Ziel für das Blatt ergibt sich durch Zurückrechnen über den Faktor:

       Gesamtbewertung(Blatt_t) soll G_t / (-gamma)^k  werden
       V(Blatt_t)               soll G_t / (-gamma)^k - Φ(Blatt_t)  werden

   Endet die Hauptvariante in einem Matt oder Remis, ist der Suchwert exakt;
   dann gibt es nichts zu lernen, und das Blatt fällt aus dem Batch. Sein
   Wert fliesst trotzdem über die λ-Returns in die anderen Ziele ein.

Mit Tiefe 1 und ohne Ruhesuche wäre Blatt_t die Stellung nach dem gewählten
Zug. Für Tiefe 1 bleibt das Training aber bewusst beim bisherigen Verfahren in
train.py — so lassen sich v1 bis v3 jederzeit genau nachvollziehen.

Mit den Korrekturen ab v6 (search.SearchOptions)
------------------------------------------------
Mit Deckel bewertet die Suche ein Blatt mit cap · tanh(V + Φ). Das Lernziel
für V wird deshalb zuerst zurückgestaucht:

    V(Blatt_t) soll  unsquash(G_t / (-gamma)^k) - Φ(Blatt_t)  werden

Mit der Wiederholungsregel ist ein Blatt, das eine schon dagewesene Stellung
wiederholt, ein exaktes Remis — es fällt wie Matt und Patt aus dem Batch.
"""
from __future__ import annotations

import chess
import numpy as np

from .engine import terminal_value
from .features import FEATURE_SIZE, board_to_features
from .positions import sample_start
from .search import CLASSIC, QUIESCENCE_PLIES, SearchOptions, leaf_of, search
from .shaping import potential_open
from .valuenet import ValueNet


def play_episode_tdleaf(
    net: ValueNet,
    rng,
    gamma: float,
    epsilon: float,
    max_plies: int,
    td_lambda: float,
    depth: int,
    quiescence_plies: int = QUIESCENCE_PLIES,
    start: chess.Board | None = None,
    options: SearchOptions = CLASSIC,
):
    """
    Eine Self-Play-Partie mit Suche. Rückgabe wie train.play_episode:
    Merkmale der Blätter, Lernziele für V an diesen Blättern, Ausgang, Halbzüge.

    'start' ersetzt die Zufallseröffnung, etwa für Tests mit fester Stellung.
    """
    board = start.copy() if start is not None else sample_start("full", rng)
    search_values: list[float] = []          # W_t
    outcomes: list[float | None] = []        # Reward, falls der Zug die Partie beendet
    leaves: list[tuple[int, np.ndarray, float, float]] = []   # (t, Merkmale, Φ, Faktor)
    result, plies = "truncated", 0

    while plies < max_plies:
        found = search(board, net, depth, gamma, quiescence_plies, options)
        if found.move is None:
            break
        leaf = leaf_of(board, found.pv)
        exact = terminal_value(leaf) is not None or (options.repetition_draw and leaf.is_repetition(2))
        if not exact:
            phi = potential_open(leaf) if net.shaped else 0.0
            leaves.append((plies, board_to_features(leaf), phi, (-gamma) ** len(found.pv)))
        search_values.append(found.value)

        if rng.random() < epsilon:
            moves = list(board.legal_moves)
            move = moves[int(rng.integers(len(moves)))]
        else:
            move = found.move
        board.push(move)
        plies += 1

        outcome = terminal_value(board)
        outcomes.append(outcome)
        if outcome is not None:
            result = "mate" if outcome == 1.0 else "draw"
            break

    if not search_values:
        return np.zeros((0, FEATURE_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32), result, plies

    # W_(t+1) für den letzten Schritt: Bei einem Abbruch am Zuglimit geht die
    # Partie eigentlich weiter, also wird mit einer letzten Suche gebootstrappt.
    following = search_values[1:]
    if outcomes[-1] is None:
        following.append(search(board, net, depth, gamma, quiescence_plies, options).value)

    targets = np.array(
        [
            outcome if outcome is not None else -gamma * following[t]
            for t, outcome in enumerate(outcomes)
        ],
        dtype=np.float64,
    )
    returns = targets.copy()
    for t in range(len(returns) - 2, -1, -1):
        returns[t] = (1.0 - td_lambda) * targets[t] - gamma * td_lambda * returns[t + 1]

    if not leaves:
        return np.zeros((0, FEATURE_SIZE), dtype=np.float32), np.zeros(0, dtype=np.float32), result, plies

    X = np.stack([features for _, features, _, _ in leaves])
    leaf_targets = np.array(
        [options.unsquash(returns[t] / factor) - phi for t, _, phi, factor in leaves], dtype=np.float32
    )
    # V liegt wegen tanh in [-1, 1]. Durch das Zurückrechnen über (-gamma)^k
    # kann ein Ziel knapp darüber liegen; das Netz könnte es nie erreichen.
    np.clip(leaf_targets, -1.0, 1.0, out=leaf_targets)
    return X, leaf_targets, result, plies
