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
    """
    Stellung -> Merkmalsvektor aus Sicht der Zugpartei.

    Gerechnet wird direkt auf den Bitmasken von python-chess: Jede der zwölf
    Ebenen ist eine 64-Bit-Zahl, Bit i steht für Feld i (a1 = 0, h8 = 63). Das
    Spiegeln für Schwarz ist dann ein Byte-Tausch (flip_vertical), und das
    Auspacken in 768 Nullen und Einsen erledigt NumPy in einem Schritt. Das
    Ergebnis ist identisch mit dem früheren Weg über board.mirror() und
    piece_map(), aber schneller — wichtig, weil beim Vorausrechnen jede
    bewertete Stellung hier durch muss.
    """
    me = board.turn
    own, opp = board.occupied_co[me], board.occupied_co[not me]
    pieces = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings)
    planes = [mask & own for mask in pieces] + [mask & opp for mask in pieces]
    if me == chess.BLACK:
        planes = [chess.flip_vertical(plane) for plane in planes]

    x = np.empty(FEATURE_SIZE, dtype=np.float32)
    bits = np.array(planes, dtype="<u8").view(np.uint8)
    x[:PIECE_FEATURES] = np.unpackbits(bits, bitorder="little")

    # "ich" ist die Zugpartei, genau wie nach dem Spiegeln Weiss.
    x[PIECE_FEATURES + 0] = float(board.has_kingside_castling_rights(me))
    x[PIECE_FEATURES + 1] = float(board.has_queenside_castling_rights(me))
    x[PIECE_FEATURES + 2] = float(board.has_kingside_castling_rights(not me))
    x[PIECE_FEATURES + 3] = float(board.has_queenside_castling_rights(not me))
    x[PIECE_FEATURES + 4] = float(board.ep_square is not None)
    x[PIECE_FEATURES + 5] = min(board.halfmove_clock / 100.0, 1.0)
    return x
