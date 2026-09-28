"""
Φ(s): die handgeschriebene Grundbewertung für das Reward Shaping.

Warum überhaupt? In ganzen Partien gibt es nur am Schluss eine Belohnung. Bis
dahin bekommt der Agent tausende Halbzüge lang kein Signal. Φ liefert eine
Zwischenrückmeldung — aber nicht als direkte Belohnung, sondern als **Differenz**
zwischen zwei Stellungen:

    Belohnung = r_Ende  −  gamma · Φ(s')  −  Φ(s)

Das ist Potential-Based Reward Shaping (Ng, Harada & Russell 1999). Über eine
ganze Partie heben sich die Φ-Terme paarweise weg; übrig bleibt der echte Ausgang
minus einer Konstanten, die vom Verhalten des Agenten unabhängig ist. Deshalb
kann der Agent das System nicht ausnutzen: Eine Figur ins Zentrum und wieder
zurück ergibt exakt null, und wer mit Material-Vorsprung verliert, zahlt den
Vorschuss am Ende vollständig zurück.

Φ wird immer aus Sicht der Partei berechnet, die am Zug ist, und ist an
Endstellungen zwingend 0 — ohne das bricht die Garantie.

Vier Bausteine, alle in Bauerneinheiten:

    M  Material        Bauer 1, Springer 3, Läufer 3, Turm 5, Dame 9
    Z  Zentrum         sättigend, damit das Aufgeben der Oberhand teuer ist,
                       das Verkleinern eines schon bestehenden Vorsprungs aber
                       fast gratis
    T  Abtausch        statische Abtauschbewertung (SEE) beider Seiten, damit
                       "Läufer schlägt Turm" positiv und "Turm schlägt gedeckten
                       Bauern" negativ ist
    S  Schach          kleiner Abzug im Schach, grosser bei Doppelschach
"""
from __future__ import annotations

import math

import chess

# Figurenwerte in Bauerneinheiten. Einzige Quelle im Projekt; engine.py rechnet
# sie für die API in Centipawns um.
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}

INNER_4 = [chess.D4, chess.E4, chess.D5, chess.E5]
INNER_16 = [chess.square(file, rank) for rank in range(2, 6) for file in range(2, 6)]

CENTER_4_WEIGHT = 0.20
CENTER_16_WEIGHT = 0.05
CENTER_SATURATION = 0.7

OWN_THREAT_WEIGHT = 0.7
OPPONENT_THREAT_WEIGHT = 0.3

CHECK_PENALTY = 0.3
DOUBLE_CHECK_PENALTY = 1.5

SCALE = 8.0
AMPLITUDE = 0.5


def potential(board: chess.Board) -> float:
    """Φ(s) im Bereich [-0.5, 0.5], aus Sicht der Zugpartei."""
    if board.is_game_over():
        return 0.0
    me = board.turn
    total = material(board, me) + center(board, me) + exchange(board, me) + check(board)
    return AMPLITUDE * math.tanh(total / SCALE)


def material(board: chess.Board, me: chess.Color) -> float:
    """Materialbilanz in Bauerneinheiten, eigene minus gegnerische."""
    return float(
        sum(
            value * (len(board.pieces(piece_type, me)) - len(board.pieces(piece_type, not me)))
            for piece_type, value in PIECE_VALUES.items()
        )
    )


def center(board: chess.Board, me: chess.Color) -> float:
    """
    Zentrumskontrolle, pro Feld sättigend bewertet.

    tanh sorgt dafür, dass der Sprung von "Gleichstand" auf "Oberhand" viel mehr
    zählt als das Ausbauen eines Vorsprungs, den man ohnehin schon hat. Eine
    Figur aus dem Zentrum zu nehmen kostet daher fast nichts, solange die
    Oberhand bleibt — und viel, sobald sie kippt.
    """
    score = 0.0
    for squares, weight in ((INNER_4, CENTER_4_WEIGHT), (INNER_16, CENTER_16_WEIGHT)):
        for square in squares:
            balance = len(board.attackers(me, square)) - len(board.attackers(not me, square))
            score += weight * math.tanh(CENTER_SATURATION * balance)
    return score


def exchange(board: chess.Board, me: chess.Color) -> float:
    """
    Abtauschbilanz: eigene Drohung minus gegnerische.

    Die Zugpartei ist höher gewichtet, weil sie ihre Drohung sofort ausführen
    kann, während die Gegenseite noch einen Zug zum Parieren hat. Genau daraus
    entsteht die Strafe fürs Hängenlassen: Wer eine ungedeckte Figur stehen
    lässt, ist danach nicht mehr am Zug.
    """
    return (
        OWN_THREAT_WEIGHT * best_exchange(board, me)
        - OPPONENT_THREAT_WEIGHT * best_exchange(board, not me)
    )


def check(board: chess.Board) -> float:
    """
    Schachabzug für die Zugpartei.

    Immer null oder negativ, denn im legalen Schach kann nur die Partei am Zug
    im Schach stehen. Belohnt wird dadurch der Übergang ins Schach; pariert der
    Gegner es, fliesst der Betrag wieder ab. Sinnlose Dauerschachs bringen also
    unter dem Strich nichts.
    """
    checkers = len(board.checkers())
    if checkers >= 2:
        return -DOUBLE_CHECK_PENALTY
    if checkers == 1:
        return -CHECK_PENALTY
    return 0.0


def best_exchange(board: chess.Board, side: chess.Color) -> int:
    """Grösster Materialgewinn, den 'side' mit einem Schlagzug statisch erzwingt."""
    best = 0
    targets = board.occupied_co[not side] & ~board.kings
    for square in chess.scan_forward(targets):
        if board.attackers_mask(side, square):
            gain = static_exchange(board, square, side)
            if gain > best:
                best = gain
    return best


def static_exchange(board: chess.Board, square: int, side: chess.Color) -> int:
    """
    Static Exchange Evaluation: Netto-Gewinn für 'side', wenn auf 'square'
    abgetauscht wird.

    Beide Seiten schlagen immer mit der billigsten verfügbaren Figur, und jede
    Seite darf abbrechen, wenn der Tausch sich nicht mehr lohnt. Dadurch ist
    "Läufer schlägt gedeckten Turm" +2 wert, "Turm schlägt gedeckten Bauern"
    nichts, und "Turm schlägt freien Bauern" +1.

    Gerechnet wird über Belegungsmasken statt über Brettkopien — sonst wäre es
    für den Trainingsloop zu langsam. Weil attackers_mask die Belegung als
    Parameter nimmt, rücken Röntgenangriffe dabei automatisch korrekt nach.
    """
    target = board.piece_at(square)
    if target is None or target.color == side:
        return 0

    occupied = board.occupied
    attackers = board.attackers_mask(chess.WHITE, square, occupied) | board.attackers_mask(
        chess.BLACK, square, occupied
    )
    gains: list[int] = []
    captured_value = PIECE_VALUES[target.piece_type]
    turn = side

    while True:
        available = attackers & board.occupied_co[turn] & occupied
        if not available:
            break
        from_square = min(
            chess.scan_forward(available), key=lambda sq: PIECE_VALUES[board.piece_type_at(sq)]
        )
        gains.append(captured_value)
        captured_value = PIECE_VALUES[board.piece_type_at(from_square)]
        occupied &= ~chess.BB_SQUARES[from_square]
        attackers |= board.attackers_mask(chess.WHITE, square, occupied) | board.attackers_mask(
            chess.BLACK, square, occupied
        )
        attackers &= occupied
        turn = not turn

    value = 0
    for gain in reversed(gains):
        value = max(0, gain - value)
    return value
