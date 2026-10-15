"""
Φ(s): die handgeschriebene Grundbewertung für das Reward Shaping.

Warum überhaupt? In ganzen Partien gibt es nur am Schluss eine Belohnung. Bis
dahin bekommt der Agent tausende Halbzüge lang kein Signal. Φ liefert eine
Zwischenrückmeldung — aber nicht als direkte Belohnung, sondern als **Differenz**
zwischen zwei Stellungen:

    Belohnung = r_Ende  −  gamma · Φ(s')  −  Φ(s)

Das ist Potential-Based Reward Shaping (Ng, Harada & Russell 1999). Die
Eigenschaft, auf die es ankommt, ist

    V_geshaped(s) = V*(s) - Phi(s)

Weil Phi(s) für alle Züge aus derselben Stellung derselbe Wert ist, verschiebt es
die Reihenfolge der Züge nicht: Die beste Wahl bleibt die beste Wahl.

Achtung, hier lauert ein Irrtum: Das übliche Teleskop-Argument - die Phi-Terme
höben sich über eine Episode paarweise weg - gilt in der Negamax-Form NICHT.
Dort steht -gamma*Phi(s') und im nächsten Halbzug -Phi(s'); beide haben dasselbe
Vorzeichen und addieren sich. An einer echten Partie nachgerechnet summierten
sich die Belohnungen auf +17.35 statt auf null. Die Policy-Invarianz gilt
trotzdem, aber über die Gleichung oben.

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

# Zentrumsfelder in genau der Reihenfolge, in der sie aufsummiert werden. Die
# Reihenfolge ist nicht beliebig: Fliesskomma-Additionen sind nicht assoziativ,
# und Φ muss für die bereits trainierten Modelle bitgenau gleich bleiben.
_CENTER_SQUARES = [
    (square, weight)
    for squares, weight in ((INNER_4, CENTER_4_WEIGHT), (INNER_16, CENTER_16_WEIGHT))
    for square in squares
]


def potential(board: chess.Board) -> float:
    """Φ(s) im Bereich [-0.5, 0.5], aus Sicht der Zugpartei."""
    if board.is_game_over():
        return 0.0
    return _potential(board)


def potential_open(board: chess.Board) -> float:
    """
    Φ(s) für eine Stellung, von der der Aufrufer schon weiss, dass sie nicht
    durch Matt, Patt, Materialmangel oder die 50-Züge-Regel beendet ist.

    Liefert exakt denselben Wert wie potential(), spart aber die doppelte
    Prüfung auf Partieende: Die Zugbewertung hat sie mit terminal_value() eben
    erst gemacht, und is_game_over() erzeugt dafür noch einmal legale Züge.
    Übrig bleibt die fünffache Wiederholung, die terminal_value() nicht kennt —
    sie wird hier weiterhin geprüft, damit das Ergebnis gleich bleibt.
    """
    if board.is_fivefold_repetition():
        return 0.0
    return _potential(board)


def _potential(board: chess.Board) -> float:
    me = board.turn
    total = material(board, me) + center(board, me) + exchange(board, me) + check(board)
    return AMPLITUDE * math.tanh(total / SCALE)


def material(board: chess.Board, me: chess.Color) -> float:
    """Materialbilanz in Bauerneinheiten, eigene minus gegnerische."""
    return float(
        sum(
            value * (chess.popcount(board.pieces_mask(piece_type, me))
                     - chess.popcount(board.pieces_mask(piece_type, not me)))
            for piece_type, value in PIECE_VALUES.items()
        )
    )


def attackers_both(board: chess.Board, square: chess.Square, occupied: int) -> int:
    """
    Alle Angreifer beider Farben auf ein Feld, bei frei wählbarer Belegung.

    Dasselbe wie attackers_mask(WHITE) | attackers_mask(BLACK) aus python-chess,
    aber in einem Durchgang statt in zweien. Diese Abfrage ist der innerste
    Baustein von Zentrum und Abtauschbewertung und läuft beim Vorausrechnen
    hunderttausendfach; die Hälfte der Arbeit zu sparen lohnt sich hier.
    """
    rank_pieces = chess.BB_RANK_MASKS[square] & occupied
    file_pieces = chess.BB_FILE_MASKS[square] & occupied
    diag_pieces = chess.BB_DIAG_MASKS[square] & occupied
    queens_and_rooks = board.queens | board.rooks
    queens_and_bishops = board.queens | board.bishops
    return (
        (chess.BB_KING_ATTACKS[square] & board.kings)
        | (chess.BB_KNIGHT_ATTACKS[square] & board.knights)
        | (chess.BB_RANK_ATTACKS[square][rank_pieces] & queens_and_rooks)
        | (chess.BB_FILE_ATTACKS[square][file_pieces] & queens_and_rooks)
        | (chess.BB_DIAG_ATTACKS[square][diag_pieces] & queens_and_bishops)
        | (chess.BB_PAWN_ATTACKS[chess.BLACK][square] & board.pawns & board.occupied_co[chess.WHITE])
        | (chess.BB_PAWN_ATTACKS[chess.WHITE][square] & board.pawns & board.occupied_co[chess.BLACK])
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
    mine, theirs, occupied = board.occupied_co[me], board.occupied_co[not me], board.occupied
    for square, weight in _CENTER_SQUARES:
        attackers = attackers_both(board, square, occupied)
        balance = chess.popcount(attackers & mine) - chess.popcount(attackers & theirs)
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
    """
    Grösster Materialgewinn, den 'side' mit einem Schlagzug statisch erzwingt.

    Die Ziele werden von der wertvollsten Figur abwärts geprüft. Ein Abtausch
    kann nie mehr einbringen als die zuerst geschlagene Figur; sobald der beste
    Gewinn diesen Wert erreicht, kann kein weiteres Ziel ihn mehr übertreffen,
    und die Prüfung bricht ab. Am Ergebnis ändert das nichts, denn das Maximum
    hängt nicht von der Reihenfolge ab.
    """
    best = 0
    enemy, mine, occupied = board.occupied_co[not side], board.occupied_co[side], board.occupied
    for pieces, value in (
        (board.queens, PIECE_VALUES[chess.QUEEN]),
        (board.rooks, PIECE_VALUES[chess.ROOK]),
        (board.bishops | board.knights, PIECE_VALUES[chess.BISHOP]),
        (board.pawns, PIECE_VALUES[chess.PAWN]),
    ):
        if best >= value:
            break
        for square in chess.scan_forward(pieces & enemy):
            if attackers_both(board, square, occupied) & mine:
                gain = static_exchange(board, square, side)
                if gain > best:
                    best = gain
                    if best >= value:
                        break
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
    für den Trainingsloop zu langsam. Weil die Angreiferabfrage die Belegung als
    Parameter nimmt, rücken Röntgenangriffe dabei automatisch korrekt nach.

    Geschlagen wird immer mit der billigsten Figur. Der König zählt mit Wert 0
    als billigste, Springer und Läufer teilen sich den Wert 3; bei gleichem Wert
    entscheidet das niedrigste Feld. Diese Reihenfolge ist festgeschrieben, weil
    die trainierten Modelle mit genau ihr gelernt haben: Wer zuerst den Läufer
    statt des Springers wegnimmt, öffnet unter Umständen eine andere
    Röntgenlinie und bekommt einen anderen Wert.
    """
    if not board.occupied_co[not side] & chess.BB_SQUARES[square]:
        return 0       # leeres Feld oder eigene Figur

    occupied = board.occupied
    attackers = attackers_both(board, square, occupied)
    gains: list[int] = []
    captured_value = PIECE_VALUES[board.piece_type_at(square)]
    turn = side
    by_value = (
        (board.kings, PIECE_VALUES[chess.KING]),
        (board.pawns, PIECE_VALUES[chess.PAWN]),
        (board.knights | board.bishops, PIECE_VALUES[chess.KNIGHT]),
        (board.rooks, PIECE_VALUES[chess.ROOK]),
        (board.queens, PIECE_VALUES[chess.QUEEN]),
    )

    while True:
        available = attackers & board.occupied_co[turn] & occupied
        if not available:
            break
        for pieces, value in by_value:
            candidates = available & pieces
            if candidates:
                lowest = candidates & -candidates
                break
        gains.append(captured_value)
        captured_value = value
        occupied &= ~lowest
        attackers |= attackers_both(board, square, occupied)
        attackers &= occupied
        turn = not turn

    value = 0
    for gain in reversed(gains):
        value = max(0, gain - value)
    return value


def loses_material(board: chess.Board, move: chess.Move) -> bool:
    """
    Verschenkt dieser Zug Material? Geschlagenes gegengerechnet.

    Gebraucht an zwei Stellen: bei der Zufallseröffnung (positions.py), damit
    keine Partie mit einer verschenkten Figur beginnt, und in der Ruhesuche
    (search.py), die nur Schlagzüge weiterverfolgt, die sich lohnen.
    """
    captured = board.piece_at(move.to_square)
    if captured is not None:
        gain = PIECE_VALUES[captured.piece_type]
    else:
        gain = PIECE_VALUES[chess.PAWN] if board.is_en_passant(move) else 0
    board.push(move)
    risk = static_exchange(board, move.to_square, board.turn)
    board.pop()
    return risk - gain > 0
