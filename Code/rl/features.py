"""
Kodierung einer Stellung als Merkmalsvektor für das Value-Netz.

Konvention: Der Vektor beschreibt die Stellung IMMER aus Sicht der Partei, die am
Zug ist. Die Ebenen 0-5 enthalten die eigenen Figuren (Bauer, Springer, Läufer,
Turm, Dame, König), die Ebenen 6-11 die des Gegners. Ist Schwarz am Zug, wird
das Brett gespiegelt: Farben tauschen und Reihen kippen.

Dadurch muss das Netz die Bewertung nur einmal lernen statt getrennt für Weiss
und Schwarz, und die Negamax-Regel V_ich(s) = -V_gegner(s) gilt exakt.

Nach den 768 Figurenfeldern folgen sechs Zusatzmerkmale (Rochaderechte, En
passant, Zugzähler). Für Endspiele sind sie bedeutungslos, für ganze Partien
nötig — ohne den Zugzähler sieht der Agent das 50-Züge-Remis nicht kommen.

Die Zusatzmerkmale stehen bewusst am Ende: Ältere Modelle, die nur mit 768
Merkmalen trainiert wurden, bleiben dadurch gültig und lesen einfach den vorderen
Teil des Vektors (siehe ValueNet.value).
"""
from __future__ import annotations

import chess
import numpy as np

NUM_PLANES = 12
PIECE_FEATURES = NUM_PLANES * 64
EXTRA_FEATURES = 6
FEATURE_SIZE = PIECE_FEATURES + EXTRA_FEATURES


def board_to_features(board: chess.Board) -> np.ndarray:
    """Stellung -> Merkmalsvektor aus Sicht der Zugpartei."""
    view = board if board.turn == chess.WHITE else board.mirror()
    x = np.zeros(FEATURE_SIZE, dtype=np.float32)

    for square, piece in view.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        x[plane * 64 + square] = 1.0

    # Nach dem Spiegeln ist Weiss immer "ich".
    x[PIECE_FEATURES + 0] = float(view.has_kingside_castling_rights(chess.WHITE))
    x[PIECE_FEATURES + 1] = float(view.has_queenside_castling_rights(chess.WHITE))
    x[PIECE_FEATURES + 2] = float(view.has_kingside_castling_rights(chess.BLACK))
    x[PIECE_FEATURES + 3] = float(view.has_queenside_castling_rights(chess.BLACK))
    x[PIECE_FEATURES + 4] = float(view.ep_square is not None)
    x[PIECE_FEATURES + 5] = min(view.halfmove_clock / 100.0, 1.0)
    return x
