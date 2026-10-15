"""
Vorausrechnen: Alpha-Beta-Suche über die gelernte Bewertung.

Bis v3 schaut die Engine genau einen Halbzug voraus: eigenen Zug ausführen,
Folgestellung bewerten, den besten nehmen (engine.value_move). Was der Gegner
darauf antwortet, sieht sie nicht. Eine Springergabel, die erst nach dem
Ausweichen des Königs eine Figur gewinnt, bleibt so unsichtbar.

Diese Suche rechnet mehrere Halbzüge tief. Die Bewertung an den Blättern ist
dieselbe wie bisher, nämlich V(s) bzw. V(s) + Φ(s) bei Modellen mit Shaping
(engine.evaluate_children). Neu ist nur, wie weit vorausgeschaut wird.

Negamax mit Diskont
-------------------
Wie im Training gilt jeder Wert aus Sicht der Partei, die am Zug ist. Der Wert
eines Zuges ist

    Ende der Partie:   der terminale Reward (+1 Matt, 0 Remis)
    sonst:             -gamma * W(s')

und W(s) ist das Maximum über alle Züge. Mit Tiefe 1 ist das exakt die Formel
von score_moves(); die Suche ist also deren Verallgemeinerung, keine zweite
Bewertungslogik. gamma < 1 sorgt dafür, dass ein schnelles Matt mehr zählt als
ein langsames.

Alpha-Beta
----------
Ein Zug muss nicht fertig gerechnet werden, sobald feststeht, dass er schlechter
ist als ein bereits gefundener. Das spart je nach Zugsortierung den grössten Teil
der Arbeit, ohne das Ergebnis zu verändern. Damit es greift, werden die Züge in
jedem Knoten nach ihrer Ein-Zug-Bewertung sortiert — die starken zuerst.

Ruhesuche (Quiescence)
----------------------
Eine Suche mit fester Tiefe hat einen blinden Fleck, den Horizonteffekt: Endet
die Rechnung mitten in einem Schlagabtausch, wird eine Stellung bewertet, in der
gerade eine Figur hängt. Deshalb rechnet die Suche an den Blättern weiter, aber
nur noch Schlagzüge (und Umwandlungen zur Dame), die nach der statischen
Abtauschbewertung kein Material verschenken. Die Partei am Zug darf jederzeit
aufhören ("stand pat"), wenn kein Schlagzug ihre Lage verbessert.

Beispiel Springergabel Sc7+: Mit Tiefe 2 sieht die Suche Sc7+ und das Ausweichen
des Königs. In der Stellung danach hängt der Turm; das erfasst schon die
Abtauschbewertung in Φ, und die Ruhesuche prüft Sxa8 nach. Mit einem Halbzug
Vorausschau bleibt die Gabel dagegen unsichtbar, weil der König noch im Weg
steht.

Bekannte Vereinfachung: Auch im Schach darf die Ruhesuche stehen bleiben, statt
alle Ausweichzüge zu prüfen. Das hält sie klein; ein Matt am Horizont kann ihr
dadurch entgehen. Φ enthält ohnehin einen Abzug für das Stehen im Schach.

Hauptvariante
-------------
Die Suche liefert neben dem Zug die erwartete Zugfolge (principal variation).
Sie endet in der Stellung, deren Bewertung den Suchwert bestimmt hat:

    Suchwert = (-gamma)^k * Gesamtbewertung(Blatt),   k = Länge der Variante

Genau diese Blattstellung trainiert TD-Leaf (tdleaf.py), und die Oberfläche
zeigt die Variante als "die Engine rechnet mit ..." an.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from .engine import evaluate_children
from .shaping import loses_material
from .valuenet import ValueNet

# Wie viele Halbzüge die Ruhesuche höchstens an ein Blatt anhängt. Längere
# Schlagfolgen sind selten; die Grenze verhindert, dass eine wilde Stellung die
# Rechenzeit sprengt.
QUIESCENCE_PLIES = 6

INF = float("inf")

_PROMOTION_RANKS = {chess.WHITE: chess.BB_RANK_7, chess.BLACK: chess.BB_RANK_2}


@dataclass
class SearchResult:
    move: chess.Move | None                     # bester Zug, None ohne legale Züge
    value: float                                # Suchwert aus Sicht der Zugpartei
    pv: list[chess.Move] = field(default_factory=list)   # Hauptvariante, beginnt mit move
    leaves: int = 0                             # bewertete Stellungen, Mass für den Aufwand


def search(
    board: chess.Board,
    net: ValueNet,
    depth: int,
    gamma: float = 0.99,
    quiescence_plies: int = QUIESCENCE_PLIES,
) -> SearchResult:
    """
    Besten Zug mit 'depth' Halbzügen Vorausschau bestimmen.

    Das Brett wird während der Suche verändert, ist danach aber wieder im
    Ausgangszustand. Bei gleich guten Zügen gewinnt derjenige, der in der
    Sortierung zuerst kommt — die Suche ist damit deterministisch.
    """
    if depth < 1:
        raise ValueError("Die Suchtiefe muss mindestens 1 sein.")
    if not any(board.legal_moves):
        return SearchResult(None, 0.0)
    searcher = _Searcher(net, gamma, quiescence_plies)
    value, line = searcher.negamax(board, depth, -INF, INF)
    return SearchResult(line[0], value, line, searcher.leaves)


class _Searcher:
    def __init__(self, net: ValueNet, gamma: float, quiescence_plies: int):
        self.net = net
        self.gamma = gamma
        self.quiescence_plies = quiescence_plies
        self.leaves = 0
        self.killers: dict[int, chess.Move] = {}   # Halbzug ab Wurzel -> letzter Widerlegungszug

    def _children(self, board: chess.Board, moves: list[chess.Move]):
        """
        Ein-Zug-Bewertung aller Züge (wie score_moves) und die Gesamtbewertung
        der Folgestellung, die die Ruhesuche als Ausgangswert braucht. None
        heisst: Der Zug beendet die Partie, sein Wert ist exakt.
        """
        outcomes, open_indices, totals = evaluate_children(board, self.net, moves)
        self.leaves += len(moves)
        scores = np.zeros(len(moves), dtype=np.float32)
        child_totals: list[float | None] = [None] * len(moves)
        for index, outcome in enumerate(outcomes):
            if outcome is not None:
                scores[index] = outcome
        if open_indices:
            scores[open_indices] = -self.gamma * totals
            for position, index in enumerate(open_indices):
                child_totals[index] = float(totals[position])
        # Stabil sortieren: Bei Gleichstand bleibt die Reihenfolge von
        # python-chess erhalten, wie bei value_move().
        order = np.argsort(-scores, kind="stable")
        return scores, child_totals, order

    def negamax(self, board: chess.Board, depth: int, alpha: float, beta: float, ply: int = 0):
        """
        Wert und Hauptvariante einer laufenden Stellung, aus Sicht der Zugpartei.

        Liegt der Wert ausserhalb des Fensters (alpha, beta), ist das Ergebnis
        nur eine Schranke — für die Entscheidung im Elternknoten reicht das.
        """
        best_value, best_line = -INF, []
        tried = None

        # Killer-Heuristik, nur im letzten Halbzug vor den Blättern: Gemessen
        # schneidet dort in über 90 Prozent der Knoten schon der erste
        # Antwortzug ab, und oft ist es derselbe, der eben einen Nachbarzug
        # widerlegt hat. Dann genügt es, diesen einen Zug zu bewerten, statt
        # alle rund 30. Am Ergebnis ändert das nichts, nur an der Arbeit.
        killer = self.killers.get(ply) if depth == 1 else None
        if killer is not None and board.is_legal(killer):
            tried = killer
            scores, totals, _ = self._children(board, [killer])
            best_value, best_line = self._frontier_value(
                board, killer, float(scores[0]), totals[0], alpha, beta
            )
            if best_value >= beta:
                return best_value, best_line
            alpha = max(alpha, best_value)

        moves = [move for move in board.legal_moves if move != tried]
        scores, totals, order = self._children(board, moves)

        for index in order:
            move, static = moves[index], float(scores[index])
            if totals[index] is None:
                value, line = static, [move]
            elif depth == 1:
                # Mehr als die Ein-Zug-Bewertung kann dieser Zug nicht wert sein:
                # Die Ruhesuche kann die Lage des Gegners nur verbessern. Ist
                # also schon die Ein-Zug-Bewertung nicht besser als alpha, gilt
                # das für alle folgenden Züge auch, denn sie sind absteigend
                # sortiert.
                if static <= alpha:
                    if static > best_value:
                        best_value, best_line = static, [move]
                    break
                value, line = self._frontier_value(board, move, static, totals[index], alpha, beta)
            else:
                board.push(move)
                reply, reply_line = self.negamax(
                    board, depth - 1, -beta / self.gamma, -alpha / self.gamma, ply + 1
                )
                board.pop()
                value, line = -self.gamma * reply, [move] + reply_line

            if value > best_value:
                best_value, best_line = value, line
            if best_value >= beta:
                self.killers[ply] = move
                break
            alpha = max(alpha, best_value)
        return best_value, best_line

    def _frontier_value(self, board, move, static: float, total: float | None, alpha: float, beta: float):
        """Wert eines Zuges im letzten Halbzug: Ein-Zug-Bewertung plus Ruhesuche."""
        if total is None or static <= alpha:
            return static, [move]
        board.push(move)
        reply, reply_line = self.quiesce(
            board, total, -beta / self.gamma, -alpha / self.gamma, self.quiescence_plies
        )
        board.pop()
        # Ohne Schlagfolge bleibt es exakt beim Wert von score_moves().
        value = static if not reply_line else -self.gamma * reply
        return value, [move] + reply_line

    def quiesce(self, board: chess.Board, stand_pat: float, alpha: float, beta: float, plies: int):
        """
        Ruhesuche: nur noch lohnende Schlagzüge, bis die Stellung ruhig ist.

        'stand_pat' ist die Gesamtbewertung der Stellung selbst, also der Wert,
        wenn die Zugpartei auf weitere Schlagzüge verzichtet.
        """
        best_value, best_line = stand_pat, []
        if plies == 0 or best_value >= beta:
            return best_value, best_line
        alpha = max(alpha, best_value)

        moves = [move for move in self._tactical_moves(board) if not loses_material(board, move)]
        if not moves:
            return best_value, best_line
        scores, totals, order = self._children(board, moves)

        for index in order:
            move, static = moves[index], float(scores[index])
            if static <= alpha:
                break
            if totals[index] is None:
                value, line = static, [move]
            else:
                board.push(move)
                reply, reply_line = self.quiesce(
                    board, totals[index], -beta / self.gamma, -alpha / self.gamma, plies - 1
                )
                board.pop()
                value = static if not reply_line else -self.gamma * reply
                line = [move] + reply_line
            if value > best_value:
                best_value, best_line = value, line
            if best_value >= beta:
                break
            alpha = max(alpha, best_value)
        return best_value, best_line

    @staticmethod
    def _tactical_moves(board: chess.Board):
        """Schlagzüge und Umwandlungen; Unterverwandlungen bleiben draussen."""
        for move in board.generate_legal_captures():
            if move.promotion in (None, chess.QUEEN):
                yield move
        pawns = board.pawns & board.occupied_co[board.turn] & _PROMOTION_RANKS[board.turn]
        if pawns:
            for move in board.generate_legal_moves(pawns, ~board.occupied):
                if move.promotion == chess.QUEEN:
                    yield move


def leaf_of(board: chess.Board, line: list[chess.Move]) -> chess.Board:
    """Stellung am Ende einer Variante, als Kopie."""
    leaf = board.copy(stack=True)
    for move in line:
        leaf.push(move)
    return leaf
