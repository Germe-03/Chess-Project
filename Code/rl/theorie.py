"""
Schachtheorie für die Engine (ab v10): Eröffnungsbuch und Endspieltabellen.

Die Theorie wird nicht gelernt, sondern nachgeschlagen. Steht eine Stellung im
Eröffnungsbuch (theorie/buch.bin, gebaut von buch.py aus Partien starker
Lichess-Spieler) oder hat sie höchstens fünf Steine (Syzygy-Tabellen in
theorie/syzygy), spielt die Version den Theoriezug; sonst rechnet sie wie ihr
Netz. Welche Version welche Theorie bekommt, steht in der Registry (Feld
"theorie"). So lässt sich getrennt messen, was Buch und Tabellen bringen.

Die Dateien liegen nur lokal (rund 1 GB, nicht im Repo). Fehlen sie, liefert
zug() einfach None, und die Engine spielt wie ohne Theorie.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import NamedTuple

import chess
import chess.polyglot
import chess.syzygy


THEORIE_DIR = Path(__file__).resolve().parents[2] / "theorie"
BUCH_PATH = THEORIE_DIR / "buch.bin"
SYZYGY_DIR = THEORIE_DIR / "syzygy"


@dataclass(frozen=True)
class TheorieOptionen:
    """Schaltet Eröffnungsbuch und Endspieltabellen ein."""

    eroeffnung: bool = False
    endspiel: bool = False

    def __bool__(self) -> bool:
        return self.eroeffnung or self.endspiel

    def kennung(self) -> str:
        """Kurze Kennung der eingeschalteten Theorieteile."""
        if self.eroeffnung and self.endspiel:
            return "buch+tb"
        if self.eroeffnung:
            return "buch"
        if self.endspiel:
            return "tb"
        return ""


KEINE = TheorieOptionen()


class TheorieZug(NamedTuple):
    """Ein Theoriezug und seine Quelle."""

    move: chess.Move
    quelle: str
    wert: float | None


def buch_verfuegbar(pfad: Path = BUCH_PATH) -> bool:
    """Gibt an, ob ein nicht leeres Eröffnungsbuch vorhanden ist."""
    try:
        return pfad.is_file() and pfad.stat().st_size > 0
    except OSError:
        return False


def endspiel_verfuegbar(pfad: Path = SYZYGY_DIR) -> bool:
    """Gibt an, ob der Ordner mindestens eine WDL-Tabelle enthält."""
    try:
        return pfad.is_dir() and next(pfad.glob("*.rtbw"), None) is not None
    except OSError:
        return False


def verfuegbar(optionen: TheorieOptionen) -> bool:
    """Gibt an, ob alle verlangten Theorieteile vorhanden sind."""
    return (
        (not optionen.eroeffnung or buch_verfuegbar())
        and (not optionen.endspiel or endspiel_verfuegbar())
    )


class Theorie:
    """Lädt Eröffnungsbuch und Endspieltabellen bei Bedarf."""

    def __init__(
        self,
        optionen: TheorieOptionen,
        buch_pfad: Path = BUCH_PATH,
        syzygy_pfad: Path = SYZYGY_DIR,
    ) -> None:
        self.optionen = optionen
        self.buch_pfad = buch_pfad
        self.syzygy_pfad = syzygy_pfad
        self._reader: chess.polyglot.MemoryMappedReader | None = None
        self._tablebase: chess.syzygy.Tablebase | None = None

    def zug(self, board: chess.Board) -> TheorieZug | None:
        """Sucht zuerst in den Endspieltabellen und danach im Eröffnungsbuch."""
        if self.optionen.endspiel:
            ergebnis = self._endspiel_zug(board)
            if ergebnis is not None:
                return ergebnis

        if self.optionen.eroeffnung:
            return self._buch_zug(board)

        return None

    def _endspiel_zug(self, board: chess.Board) -> TheorieZug | None:
        if chess.popcount(board.occupied) > 5 or board.castling_rights != 0:
            return None

        legale_zuege = list(board.legal_moves)
        if not legale_zuege or self._tabellen() is None:
            return None

        kandidaten: list[tuple[int, int, chess.Move]] = []
        try:
            for move in legale_zuege:
                board.push(move)
                try:
                    if board.is_checkmate():
                        return TheorieZug(move, "endspiel", 1.0)

                    wdl = self._tablebase.probe_wdl(board)
                    dtz = self._tablebase.probe_dtz(board)
                    eigenes_ergebnis = -wdl
                    kandidaten.append((eigenes_ergebnis, dtz, move))
                finally:
                    board.pop()
        except KeyError:
            return None

        if not kandidaten:
            return None

        def schluessel(kandidat: tuple[int, int, chess.Move]) -> tuple[int, int, str]:
            eigenes_ergebnis, dtz, move = kandidat
            if eigenes_ergebnis > 0:
                dtz_rang = abs(dtz)
            elif eigenes_ergebnis < 0:
                dtz_rang = -abs(dtz)
            else:
                dtz_rang = 0
            return -eigenes_ergebnis, dtz_rang, move.uci()

        eigenes_ergebnis, _, bester_zug = min(kandidaten, key=schluessel)
        if eigenes_ergebnis == 2:
            wert = 1.0
        elif eigenes_ergebnis == -2:
            wert = -1.0
        else:
            wert = 0.0
        return TheorieZug(bester_zug, "endspiel", wert)

    def _buch_zug(self, board: chess.Board) -> TheorieZug | None:
        kandidaten = self.buch_zuege(board)
        if not kandidaten:
            return None
        move, _ = min(kandidaten, key=lambda kandidat: (-kandidat[1], kandidat[0].uci()))
        return TheorieZug(move, "buch", None)

    def buch_zuege(self, board: chess.Board) -> list[tuple[chess.Move, int]]:
        """Alle legalen Buchzüge der Stellung mit Gewicht (Anzahl Partien); leer ausserhalb des Buchs.

        Unabhängig von optionen.eroeffnung, damit das Training Startstellungen aus dem Buch ziehen kann.
        """
        reader = self._buch()
        if reader is None:
            return []
        legale_zuege = set(board.legal_moves)
        return [(eintrag.move, eintrag.weight) for eintrag in reader.find_all(board) if eintrag.move in legale_zuege]

    def endspiel_wert(self, board: chess.Board) -> float | None:
        """Ausgang laut Endspieltabelle aus Sicht der Seite am Zug: +1.0, 0.0 oder -1.0.

        None, wenn die Stellung nicht abgedeckt ist (mehr als 5 Steine, Rochaderecht, Tabelle fehlt).
        Verfluchte Gewinne und gerettete Verluste zählen wegen der 50-Züge-Regel als remis.
        Unabhängig von optionen.endspiel, damit das Training Partien danach werten kann.
        """
        if chess.popcount(board.occupied) > 5 or board.castling_rights != 0:
            return None
        tabellen = self._tabellen()
        if tabellen is None:
            return None
        try:
            wdl = tabellen.probe_wdl(board)
        except KeyError:
            return None
        return 1.0 if wdl == 2 else -1.0 if wdl == -2 else 0.0

    def _buch(self) -> chess.polyglot.MemoryMappedReader | None:
        if self._reader is None:
            if not buch_verfuegbar(self.buch_pfad):
                return None
            try:
                self._reader = chess.polyglot.open_reader(str(self.buch_pfad))
            except OSError:
                return None
        return self._reader

    def _tabellen(self) -> chess.syzygy.Tablebase | None:
        if self._tablebase is None:
            if not self.syzygy_pfad.is_dir():
                return None
            try:
                self._tablebase = chess.syzygy.open_tablebase(str(self.syzygy_pfad))
            except OSError:
                return None
        return self._tablebase

    def close(self) -> None:
        """Schliesst geöffnete Dateien und setzt die Handles zurück."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._tablebase is not None:
            self._tablebase.close()
            self._tablebase = None

    def __enter__(self) -> Theorie:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


_theorie_cache: dict[TheorieOptionen, Theorie] = {}


def theorie_fuer(optionen: TheorieOptionen) -> Theorie | None:
    """Gibt die zwischengespeicherte Theorie zu den Optionen zurück."""
    if not optionen:
        return None
    if optionen not in _theorie_cache:
        _theorie_cache[optionen] = Theorie(optionen)
    return _theorie_cache[optionen]
