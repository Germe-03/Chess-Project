"""
Startstellungen für das Self-Play.

Aufgaben:
    kqk   König + Dame gegen König
    krk   König + Turm gegen König
    full  volle Partie aus der Grundstellung

Warum ein Curriculum? Gemessen an zufälligem Spiel endet ein KQK-Endspiel nur in
rund 1 Prozent der Fälle mit Matt, ein KRK-Endspiel praktisch nie - fast alle
Partien enden mit Patt oder geschlagener Dame. Der Agent sähe also kaum je eine
positive Belohnung und hätte nichts zu lernen (klassisches Sparse-Reward-
Problem).

Deshalb beginnt das Training mit Stellungen, in denen der schwache König schon
in einer Ecke steht und der starke König daneben; das Matt ist dort wenige Züge
entfernt. Über das Training wird der Anteil dieser einfachen Stellungen auf null
gefahren (difficulty 0 -> 1).

Entscheidend: Die Belohnungsfunktion bleibt unverändert. Verändert wird nur die
Verteilung der Startzustände. Damit entfällt das Risiko von Reward Hacking, und
die Evaluation läuft immer auf gleichverteilten Stellungen (difficulty = 1).
"""
from __future__ import annotations

import chess

TASKS = ("kqk", "krk", "full")

_STRONG_PIECE = {"kqk": chess.QUEEN, "krk": chess.ROOK}
_CORNERS = ((0, 0), (0, 7), (7, 0), (7, 7))
_RING2 = [
    (dr, df)
    for dr in (-2, -1, 0, 1, 2)
    for df in (-2, -1, 0, 1, 2)
    if max(abs(dr), abs(df)) == 2
]


def sample_start(task: str, rng, difficulty: float = 1.0) -> chess.Board:
    """
    Zufällige Startstellung. difficulty = 0 heisst nur einfache Ecken-
    stellungen, difficulty = 1 heisst gleichverteilt über das ganze Brett.
    """
    if task == "full":
        return chess.Board()
    piece = _STRONG_PIECE[task]
    easy = rng.random() >= difficulty
    while True:
        board = _build_easy(rng, piece) if easy else _build_uniform(rng, piece)
        if board is not None:
            return board


def _build_uniform(rng, piece: int) -> chess.Board | None:
    squares = rng.choice(64, size=3, replace=False)
    return _assemble(rng, piece, int(squares[0]), int(squares[1]), int(squares[2]))


def _build_easy(rng, piece: int) -> chess.Board | None:
    """Schwacher König am Rand einer Ecke, starker König zwei Felder entfernt."""
    corner_rank, corner_file = _CORNERS[int(rng.integers(4))]
    weak_rank = corner_rank + (1 if corner_rank == 0 else -1) * int(rng.integers(2))
    weak_file = corner_file + (1 if corner_file == 0 else -1) * int(rng.integers(2))

    candidates = [
        (weak_rank + dr, weak_file + df)
        for dr, df in _RING2
        if 0 <= weak_rank + dr <= 7 and 0 <= weak_file + df <= 7
    ]
    strong_rank, strong_file = candidates[int(rng.integers(len(candidates)))]

    weak_square = chess.square(weak_file, weak_rank)
    strong_square = chess.square(strong_file, strong_rank)
    free = [sq for sq in range(64) if sq not in (weak_square, strong_square)]
    piece_square = free[int(rng.integers(len(free)))]
    return _assemble(rng, piece, weak_square, strong_square, piece_square)


def _assemble(rng, piece: int, weak_square: int, strong_square: int, piece_square: int) -> chess.Board | None:
    if len({weak_square, strong_square, piece_square}) != 3:
        return None
    strong = chess.WHITE if rng.random() < 0.5 else chess.BLACK
    board = chess.Board(None)
    board.set_piece_at(weak_square, chess.Piece(chess.KING, not strong))
    board.set_piece_at(strong_square, chess.Piece(chess.KING, strong))
    board.set_piece_at(piece_square, chess.Piece(piece, strong))
    board.turn = chess.WHITE if rng.random() < 0.5 else chess.BLACK
    if not board.is_valid() or board.is_game_over():
        return None
    return board
