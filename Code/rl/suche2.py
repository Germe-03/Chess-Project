"""
Schnelle Suche ab v11: dieselbe Bewertung wie search.py, aber weniger Arbeit.

Die Stellungsbewertung (Merkmale, Netz, Φ) ist teuer; die alte Suche schafft
rund 600 Stellungen pro Sekunde und bewertet in inneren Knoten alle Kinder vorab,
nur um sie zu sortieren. Diese Suche spart an vier Stellen:

1. Bewertungs-Cache: Jede Stellung wird höchstens einmal bewertet, auch über
   mehrere Iterationen und Zugumstellungen hinweg.
2. Iterative Vertiefung mit Transpositionstabelle: Tiefe 1, 2, ... nacheinander;
   der beste Zug der vorigen Runde kommt zuerst, gespeicherte Schranken schneiden
   ab. Mit 'zeit' bricht die Suche ab und liefert die letzte fertige Tiefe.
3. Zugsortierung in inneren Knoten ohne Vorab-Bewertung: Tabellenzug,
   Schlagzüge nach Opferwert, Killerzüge, dann ruhige Züge nach Erfolgsgeschichte.
   An der Front bleibt es wie in search.py (alle Kinder in einem Netzaufruf).
4. Reduktionen (abschaltbar): Nullzug und späte Züge mit weniger Tiefe.

Ohne Tabelle und Reduktionen liefert sie an der Wurzel exakt den Wert von
search.py (tests/test_suche2.py). Grenzen: Die Tabelle kennt die Vorgeschichte
nicht, mit der Wiederholungsregel kann ein Treffer daher eine Wiederholung
übersehen; Tabellentreffer und Nullzug kürzen die Hauptvariante. Für TD-Leaf
(braucht die volle Variante) ist diese Suche darum nicht gedacht.
"""
from __future__ import annotations

import time
from typing import TypeAlias

import chess
import numpy as np

from . import features, shaping
from .engine import terminal_value
from .search import (
    CLASSIC,
    INF,
    QUIESCENCE_PLIES,
    SearchOptions,
    SearchResult,
    _Searcher,
)
from .valuenet import ValueNet

_TranspositionKey: TypeAlias = tuple[object, int]
_TTEntry: TypeAlias = tuple[int, float, str, chess.Move | None]

_EXAKT = "EXAKT"
_UNTERE = "UNTERE"
_OBERE = "OBERE"


class _Zeitende(Exception):
    """Interne Ausnahme zum Abbrechen einer unvollständigen Iteration."""


class _Schnellsucher(_Searcher):
    def __init__(
        self,
        net: ValueNet,
        gamma: float,
        quiescence_plies: int,
        options: SearchOptions,
        tt: bool,
        reduktionen: bool,
    ):
        super().__init__(net, gamma, quiescence_plies, options)
        self.terms: dict[_TranspositionKey, float | None] = {}
        self.totals: dict[_TranspositionKey, float] = {}
        self.use_tt = tt
        self.tt: dict[_TranspositionKey, _TTEntry] = {}
        self.reduktionen = reduktionen
        self.killers2: dict[int, list[chess.Move]] = {}
        self.history: dict[tuple[chess.Square, chess.Square], int] = {}
        self.root_best: chess.Move | None = None
        self.root_order: list[chess.Move] = []
        self.deadline: float | None = None
        self.zeit_aktiv = False

    @staticmethod
    def _key(board: chess.Board) -> _TranspositionKey:
        """Schlüssel der Stellung samt Zugzähler für die 50-Züge-Regel."""
        return board._transposition_key(), board.halfmove_clock

    def _terminal(self, board: chess.Board) -> float | None:
        """Terminalen Reward der aktuellen Stellung aus dem Cache lesen."""
        key = self._key(board)
        if key not in self.terms:
            self.terms[key] = terminal_value(board)
        return self.terms[key]

    def _total(self, board: chess.Board) -> float:
        """Rohe Gesamtbewertung der aktuellen Stellung ohne Squash."""
        key = self._key(board)
        if key not in self.totals:
            x = features.board_to_features(board)
            value = self.net.value(np.stack([x]))[0]
            if self.net.shaped:
                value = value + np.asarray([shaping.potential_open(board)], dtype=np.float32)[0]
            self.totals[key] = float(value)
            self.leaves += 1
        return self.totals[key]

    def _children(self, board: chess.Board, moves: list[chess.Move]):
        """
        Ein-Zug-Bewertung mit gemeinsamem Netzaufruf und Cache für Folgestellungen.
        """
        keys: list[_TranspositionKey] = []
        outcomes: list[float | None] = []
        irreversible: list[bool] = []

        for move in moves:
            irreversible.append(board.is_irreversible(move))
            board.push(move)
            try:
                key = self._key(board)
                keys.append(key)
                if key not in self.terms:
                    self.terms[key] = terminal_value(board)
                outcomes.append(self.terms[key])
            finally:
                board.pop()

        fehlend: dict[_TranspositionKey, tuple[np.ndarray, float | None]] = {}
        for index, key in enumerate(keys):
            if outcomes[index] is None and key not in self.totals and key not in fehlend:
                board.push(moves[index])
                try:
                    x = features.board_to_features(board)
                    potential = shaping.potential_open(board) if self.net.shaped else None
                    fehlend[key] = (x, potential)
                finally:
                    board.pop()

        if fehlend:
            cache_keys = list(fehlend)
            values = self.net.value(np.stack([fehlend[key][0] for key in cache_keys]))
            if self.net.shaped:
                potentials = np.asarray(
                    [fehlend[key][1] for key in cache_keys], dtype=np.float32
                )
                values = values + potentials
            for key, value in zip(cache_keys, values):
                self.totals[key] = float(value)
            self.leaves += len(cache_keys)

        repetitions = [False] * len(moves)
        if self.options.repetition_draw:
            for index, move in enumerate(moves):
                if outcomes[index] is not None or irreversible[index]:
                    continue
                board.push(move)
                try:
                    repetitions[index] = board.is_repetition(2)
                finally:
                    board.pop()

        totals_array = np.asarray(
            [
                self.totals[key]
                for index, key in enumerate(keys)
                if outcomes[index] is None and not repetitions[index]
            ],
            dtype=np.float32,
        )
        squashed = self.options.squash(totals_array)

        scores = np.zeros(len(moves), dtype=np.float32)
        child_totals: list[float | None] = [None] * len(moves)
        total_position = 0
        for index, outcome in enumerate(outcomes):
            if outcome is not None:
                scores[index] = outcome
            elif repetitions[index]:
                scores[index] = 0.0
            else:
                total = float(squashed[total_position])
                total_position += 1
                scores[index] = -self.gamma * total
                child_totals[index] = total

        order = np.argsort(-scores, kind="stable")
        return scores, child_totals, order

    def _time_check(self) -> None:
        """Bricht eine aktive Iteration nach Ablauf ihres Zeitbudgets ab."""
        if self.zeit_aktiv and self.deadline is not None and time.perf_counter() > self.deadline:
            raise _Zeitende

    def _remember_killer(self, ply: int, move: chess.Move) -> None:
        """Speichert bis zu zwei unterschiedliche Widerlegungszüge."""
        moves = self.killers2.setdefault(ply, [])
        if move in moves:
            moves.remove(move)
        moves.insert(0, move)
        del moves[2:]
        self.killers[ply] = move

    def _ordered_inner_moves(
        self, board: chess.Board, moves: list[chess.Move], tt_move: chess.Move | None, ply: int
    ) -> list[chess.Move]:
        """Ordnet Züge ohne Vorabbewertung der Folgestellungen."""
        ordered: list[chess.Move] = []
        enthalten: set[chess.Move] = set()

        def add(move: chess.Move) -> None:
            if move not in enthalten:
                ordered.append(move)
                enthalten.add(move)

        if tt_move is not None and tt_move in moves:
            add(tt_move)

        tactical: list[tuple[int, int, int, chess.Move]] = []
        quiet: list[tuple[int, chess.Move]] = []
        for index, move in enumerate(moves):
            if move in enthalten:
                continue
            capture = board.is_capture(move)
            queen_promotion = move.promotion == chess.QUEEN
            if capture or queen_promotion:
                if board.is_en_passant(move):
                    victim_value = shaping.PIECE_VALUES[chess.PAWN]
                else:
                    victim = board.piece_at(move.to_square)
                    victim_value = (
                        shaping.PIECE_VALUES[victim.piece_type] if victim is not None else 0
                    )
                if queen_promotion and not capture:
                    victim_value = shaping.PIECE_VALUES[chess.QUEEN] - shaping.PIECE_VALUES[chess.PAWN]
                attacker = board.piece_at(move.from_square)
                attacker_value = (
                    shaping.PIECE_VALUES[attacker.piece_type]
                    if attacker is not None
                    else shaping.PIECE_VALUES[chess.PAWN]
                )
                tactical.append((-victim_value, attacker_value, index, move))
            else:
                quiet.append(
                    (
                        -self.history.get((move.from_square, move.to_square), 0),
                        move,
                    )
                )

        for _, _, _, move in sorted(tactical):
            add(move)

        for killer in self.killers2.get(ply, []):
            if killer in moves:
                add(killer)

        for _, move in sorted(quiet, key=lambda item: item[0]):
            add(move)

        return ordered

    def _store_tt(
        self,
        key: _TranspositionKey,
        depth: int,
        value: float,
        alpha0: float,
        beta0: float,
        move: chess.Move | None,
    ) -> None:
        """Speichert einen fail-soft-Wert, wenn er die vorhandene Tiefe erreicht."""
        if not self.use_tt:
            return
        if value <= alpha0:
            kind = _OBERE
        elif value >= beta0:
            kind = _UNTERE
        else:
            kind = _EXAKT
        old = self.tt.get(key)
        if old is None or depth >= old[0]:
            self.tt[key] = (depth, value, kind, move)

    def negamax(
        self,
        board: chess.Board,
        depth: int,
        alpha: float,
        beta: float,
        ply: int = 0,
        root: bool = False,
        null_ok: bool = True,
    ) -> tuple[float, list[chess.Move]]:
        """Berechnet den Wert und die Hauptvariante aus Sicht der Zugpartei."""
        self._time_check()

        if depth <= 0:
            if not any(board.legal_moves):
                return 0.0, []
            stand = float(
                self.options.squash(
                    np.array([self._total(board)], dtype=np.float32)
                )[0]
            )
            value, _ = self.quiesce(
                board, stand, alpha, beta, self.quiescence_plies
            )
            return value, []

        alpha0, beta0 = alpha, beta
        key = self._key(board)
        tt_entry = self.tt.get(key) if self.use_tt else None
        tt_move: chess.Move | None = None

        if tt_entry is not None:
            tt_depth, tt_value, tt_kind, tt_move = tt_entry
            if ply > 0 and not root and tt_depth >= depth:
                if tt_kind == _EXAKT:
                    return tt_value, [tt_move] if tt_move is not None else []
                if tt_kind == _UNTERE and tt_value >= beta:
                    return tt_value, [tt_move] if tt_move is not None else []
                if tt_kind == _OBERE and tt_value <= alpha:
                    return tt_value, [tt_move] if tt_move is not None else []

        moves = list(board.legal_moves)
        if not moves:
            return 0.0, []

        if root:
            if depth == 1:
                scores, totals, order = self._children(board, moves)
                ordered_moves = [moves[int(index)] for index in order]
                self.root_order = ordered_moves.copy()
                best_value, best_line = -INF, []
                for index in order:
                    move = moves[int(index)]
                    static = float(scores[int(index)])
                    total = totals[int(index)]
                    if total is None:
                        value, line = static, [move]
                    elif static <= alpha:
                        if static > best_value:
                            best_value, best_line = static, [move]
                        break
                    else:
                        value, line = self._frontier_value(
                            board, move, static, total, alpha, beta
                        )
                    if value > best_value:
                        best_value, best_line = value, line
                    if best_value >= beta:
                        break
                    alpha = max(alpha, best_value)
                self._store_tt(
                    key, depth, best_value, alpha0, beta0,
                    best_line[0] if best_line else None,
                )
                return best_value, best_line

            root_moves: list[chess.Move] = []
            if self.root_best is not None and self.root_best in moves:
                root_moves.append(self.root_best)
            for move in self.root_order:
                if move in moves and move not in root_moves:
                    root_moves.append(move)
            for move in moves:
                if move not in root_moves:
                    root_moves.append(move)
        else:
            root_moves = []

        if not root:
            if self.reduktionen and depth >= 3 and ply > 0 and null_ok:
                has_non_pawn = bool(
                    board.occupied_co[board.turn] & ~(board.pawns | board.kings)
                )
                if (
                    not board.is_check()
                    and has_non_pawn
                    and beta < 0.85
                    and beta > -INF
                ):
                    board.push(chess.Move.null())
                    try:
                        reply, _ = self.negamax(
                            board,
                            depth - 3,
                            -beta / self.gamma,
                            -(beta - 1e-4) / self.gamma,
                            ply + 1,
                            False,
                            null_ok=False,
                        )
                    finally:
                        board.pop()
                    value = -self.gamma * reply
                    if value >= beta:
                        return value, []

            root_moves = self._ordered_inner_moves(board, moves, tt_move, ply)

        best_value, best_line = -INF, []
        tried: chess.Move | None = None

        if depth == 1 and not root:
            candidate = self.killers.get(ply)
            if candidate is None or not board.is_legal(candidate):
                candidate = tt_move if tt_move is not None and board.is_legal(tt_move) else None
            if candidate is not None and candidate in moves:
                tried = candidate
                scores, totals, _ = self._children(board, [candidate])
                best_value, best_line = self._frontier_value(
                    board, candidate, float(scores[0]), totals[0], alpha, beta
                )
                if best_value >= beta:
                    self._remember_killer(ply, candidate)
                    self._store_tt(key, depth, best_value, alpha0, beta0, candidate)
                    return best_value, best_line
                alpha = max(alpha, best_value)

            remaining = [move for move in moves if move != tried]
            scores, totals, order = self._children(board, remaining)
            for raw_index in order:
                index = int(raw_index)
                move, static = remaining[index], float(scores[index])
                if totals[index] is None:
                    value, line = static, [move]
                elif static <= alpha:
                    if static > best_value:
                        best_value, best_line = static, [move]
                    break
                else:
                    value, line = self._frontier_value(
                        board, move, static, totals[index], alpha, beta
                    )
                if value > best_value:
                    best_value, best_line = value, line
                if best_value >= beta:
                    self._remember_killer(ply, move)
                    break
                alpha = max(alpha, best_value)

            self._store_tt(
                key, depth, best_value, alpha0, beta0,
                best_line[0] if best_line else None,
            )
            return best_value, best_line

        in_check = board.is_check()
        for index, move in enumerate(root_moves):
            gives_check = board.gives_check(move)
            is_capture = board.is_capture(move)
            is_irreversible = board.is_irreversible(move)
            quiet = not is_capture and move.promotion is None

            board.push(move)
            try:
                term = self._terminal(board)
                if term is not None:
                    value, line = term, [move]
                elif self.options.repetition_draw and not is_irreversible and board.is_repetition(2):
                    value, line = 0.0, [move]
                else:
                    child_depth = depth - 1
                    if (
                        self.reduktionen
                        and depth >= 3
                        and index >= 3
                        and quiet
                        and not in_check
                        and not gives_check
                    ):
                        reply, _ = self.negamax(
                            board,
                            depth - 2,
                            -beta / self.gamma,
                            -alpha / self.gamma,
                            ply + 1,
                            False,
                        )
                        reduced_value = -self.gamma * reply
                        if reduced_value > alpha:
                            reply, reply_line = self.negamax(
                                board,
                                child_depth,
                                -beta / self.gamma,
                                -alpha / self.gamma,
                                ply + 1,
                                False,
                            )
                            value, line = -self.gamma * reply, [move] + reply_line
                        else:
                            value, line = reduced_value, [move]
                    else:
                        reply, reply_line = self.negamax(
                            board,
                            child_depth,
                            -beta / self.gamma,
                            -alpha / self.gamma,
                            ply + 1,
                            False,
                        )
                        value, line = -self.gamma * reply, [move] + reply_line
            finally:
                board.pop()

            if value > best_value:
                best_value, best_line = value, line
            if best_value >= beta:
                if quiet:
                    self._remember_killer(ply, move)
                    history_key = (move.from_square, move.to_square)
                    self.history[history_key] = (
                        self.history.get(history_key, 0) + depth * depth
                    )
                break
            alpha = max(alpha, best_value)

        self._store_tt(
            key, depth, best_value, alpha0, beta0,
            best_line[0] if best_line else None,
        )
        return best_value, best_line


def schnellsuche(
    board: chess.Board,
    net: ValueNet,
    depth: int,
    gamma: float = 0.99,
    quiescence_plies: int = QUIESCENCE_PLIES,
    options: SearchOptions = CLASSIC,
    zeit: float | None = None,
    tt: bool = True,
    reduktionen: bool = True,
) -> SearchResult:
    """Sucht iterativ vertieft und gibt das Ergebnis der letzten fertigen Runde zurück."""
    if depth < 1:
        raise ValueError("Die Suchtiefe muss mindestens 1 sein.")

    moves = list(board.legal_moves)
    if not moves:
        return SearchResult(None, 0.0)

    searcher = _Schnellsucher(
        net, gamma, quiescence_plies, options, tt, reduktionen
    )
    stack_length = len(board.move_stack)
    best_value, best_line = 0.0, []

    started = time.perf_counter()
    searcher.deadline = started + zeit if zeit is not None else None
    try:
        for current_depth in range(1, depth + 1):
            searcher.zeit_aktiv = zeit is not None and current_depth >= 2
            try:
                value, line = searcher.negamax(
                    board, current_depth, -INF, INF, ply=0, root=True
                )
            except _Zeitende:
                break

            best_value, best_line = value, line
            searcher.root_best = line[0] if line else None

            if zeit is not None and time.perf_counter() - started > 0.4 * zeit:
                break
    finally:
        while len(board.move_stack) > stack_length:
            board.pop()

    return SearchResult(
        best_line[0] if best_line else None,
        best_value,
        best_line,
        searcher.leaves,
    )
