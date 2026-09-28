"""
Zugwahl: aus einer Stellung den besten Zug bestimmen.

Drei Varianten mit absichtlich gleicher Struktur - einen Zug vorausschauen, die
Folgestellung bewerten, das Maximum nehmen:

    random_move    Zufall, unterste Vergleichslinie
    material_move  Baseline: handgeschriebene Materialbilanz (die bisherige
                   Platzhalter-Engine des Projekts)
    value_move     RL-Agent: gelernte Bewertungsfunktion V(s)

Der einzige Unterschied liegt in der Bewertung der Folgestellung. Ehrliche
Einordnung für die Dokumentation: Gelernt - und damit Reinforcement Learning -
ist ausschliesslich V(s). Die Vorausschau um einen Zug ist klassische Suche, denn
das Modell der Umwelt (die Schachregeln) ist bekannt.
"""
from __future__ import annotations

import chess
import numpy as np

from .features import board_to_features
from .shaping import PIECE_VALUES as PAWN_VALUES
from .shaping import potential
from .valuenet import ValueNet

# Centipawns für die API-Ausgabe. Die Figurenwerte selbst stehen nur einmal im
# Projekt, nämlich in shaping.py.
PIECE_VALUES = {piece_type: value * 100 for piece_type, value in PAWN_VALUES.items()}


def material_evaluation(board: chess.Board) -> float:
    """Materialbilanz in Centipawns, positiv = Vorteil Weiss."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))
    return float(score)


def terminal_value(board: chess.Board) -> float | None:
    """
    Ausgang der Partie aus Sicht der Partei, die soeben gezogen hat:
    +1 Matt gesetzt, 0 Remis, None wenn die Partie weitergeht.

    Remis umfasst Patt, ungenügendes Material und die 50-Züge-Regel. Die
    Wiederholungsregel wird bewusst weggelassen, weil ihre Prüfung im Trainings-
    loop zu teuer ist; stattdessen begrenzt das Training die Episodenlänge.
    """
    if board.is_checkmate():
        return 1.0
    if board.is_stalemate() or board.is_insufficient_material() or board.halfmove_clock >= 100:
        return 0.0
    return None


def score_moves(board: chess.Board, net: ValueNet, gamma: float):
    """
    Bewertet alle legalen Züge aus Sicht der Zugpartei mit r + gamma * (-V(s')).

    Das Minuszeichen ist die Negamax-Regel: V(s') gilt aus Sicht des Gegners, der
    nach dem Zug am Zug ist. Endet der Zug die Partie, zählt direkt der
    terminale Reward und es wird nicht mehr gebootstrappt.

    Wurde das Modell mit Reward Shaping trainiert, tritt an die Stelle von V(s')
    die Gesamtbewertung V(s') + Φ(s'). Das muss so sein: Φ war beim Lernen Teil
    der Belohnung, also gehört es beim Ausführen auch in die Zugwahl.
    """
    moves = list(board.legal_moves)
    scores = np.zeros(len(moves), dtype=np.float32)
    open_indices: list[int] = []
    open_features: list[np.ndarray] = []
    open_potentials: list[float] = []

    for index, move in enumerate(moves):
        board.push(move)
        outcome = terminal_value(board)
        if outcome is None:
            open_indices.append(index)
            open_features.append(board_to_features(board))
            if net.shaped:
                open_potentials.append(potential(board))
        else:
            scores[index] = outcome
        board.pop()

    if open_indices:
        values = net.value(np.stack(open_features))
        if net.shaped:
            values = values + np.asarray(open_potentials, dtype=np.float32)
        scores[open_indices] = -gamma * values
    return moves, scores


def value_move(board: chess.Board, net: ValueNet, gamma: float = 0.99, rng=None) -> chess.Move | None:
    """Zug mit der höchsten gelernten Bewertung."""
    moves, scores = score_moves(board, net, gamma)
    if not moves:
        return None
    best = np.flatnonzero(scores == scores.max())
    index = int(rng.choice(best)) if rng is not None else int(best[0])
    return moves[index]


def material_move(board: chess.Board) -> chess.Move | None:
    """
    Baseline: material-gieriger Blick um einen Zug.

    Gegenüber der ursprünglichen Platzhalter-Engine mit einer Ergänzung: Ein
    Matt in einem Zug wird genommen. Ohne das wäre der Vergleich im Endspiel
    unfair, weil eine reine Materialzählung ein Matt gar nicht als etwas
    Besonderes erkennt.
    """
    best_move, best_score = None, float("-inf")
    white_to_move = board.turn == chess.WHITE
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            board.pop()
            return move
        score = material_evaluation(board)
        board.pop()
        if not white_to_move:
            score = -score
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def random_move(board: chess.Board, rng) -> chess.Move | None:
    moves = list(board.legal_moves)
    if not moves:
        return None
    return moves[int(rng.integers(len(moves)))]
