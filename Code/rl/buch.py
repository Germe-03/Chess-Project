"""
Eröffnungsbuch aus PGN-Partien erstellen (Schachtheorie ab v10, siehe theorie.py).

Gezählt wird, welche Züge starke Spieler in den ersten Halbzügen jeder Stellung
gespielt haben und wie viele Punkte die ziehende Seite damit holte. Ins Buch
kommen nur Züge, die oft genug gespielt wurden und nicht schlecht abschnitten;
das Gewicht ist die Anzahl Partien. Die Engine spielt in einer Buchstellung den
Zug mit dem höchsten Gewicht. Geschrieben wird im Polyglot-Format, das
python-chess direkt lesen kann; daneben buch.json mit Quelle und Parametern.

Aufruf:
    python -m Code.rl.buch --pgn theorie/quellen/lichess_elite_2025-11.zip --out theorie/buch.bin
"""
from __future__ import annotations

import argparse
import io
import json
import struct
import time
import zipfile
from datetime import date
from functools import partial
from pathlib import Path
from typing import TextIO

import chess
import chess.pgn
import chess.polyglot


def _uci_polyglot(board: chess.Board, move: chess.Move) -> str:
    """UCI-Zug im Polyglot-Format, bei Rochaden mit dem Turm als Zielfeld."""
    if board.is_kingside_castling(move):
        zielfeld = chess.H1 if board.turn == chess.WHITE else chess.H8
        return chess.square_name(move.from_square) + chess.square_name(zielfeld)
    if board.is_queenside_castling(move):
        zielfeld = chess.A1 if board.turn == chess.WHITE else chess.A8
        return chess.square_name(move.from_square) + chess.square_name(zielfeld)
    return board.uci(move)


def _raw_aus_uci(uci: str) -> int:
    """Erzeugt den Polyglot-Rohzug aus einem UCI-Zug."""
    if len(uci) not in {4, 5}:
        raise ValueError(f"UCI-Zug muss vier oder fünf Zeichen haben: {uci!r}")

    von = chess.parse_square(uci[:2])
    nach = chess.parse_square(uci[2:4])
    promo = 0
    if len(uci) == 5:
        promo = {"n": 1, "b": 2, "r": 3, "q": 4}.get(uci[4].lower(), -1)
        if promo < 0:
            raise ValueError(f"Ungültige Umwandlungsfigur im UCI-Zug: {uci!r}")

    return nach | (von << 6) | (promo << 12)


def polyglot_zug(board: chess.Board, move: chess.Move) -> int:
    """Erzeugt den Polyglot-Rohzug, einschliesslich Polyglot-Rochaden."""
    return _raw_aus_uci(_uci_polyglot(board, move))


class _PartieVisitor(chess.pgn.BaseVisitor):
    """Sammelt Header und die ersten Hauptzüge einer PGN-Partie."""

    def __init__(self, max_halbzuege: int) -> None:
        super().__init__()
        self.max_halbzuege = max_halbzuege
        self.headers: dict[str, str] = {}
        self.zuege: list[tuple[int, str, bool]] = []

    def begin_headers(self) -> None:
        self.headers = {}

    def visit_header(self, tagname: str, tagvalue: str) -> None:
        self.headers[tagname] = tagvalue

    def begin_variation(self) -> chess.pgn.SkipType:
        return chess.pgn.SKIP

    def visit_move(self, board: chess.Board, move: chess.Move) -> None:
        if len(self.zuege) >= self.max_halbzuege:
            return
        self.zuege.append(
            (
                chess.polyglot.zobrist_hash(board),
                _uci_polyglot(board, move),
                board.turn,
            )
        )

    def handle_error(self, error: Exception) -> None:
        pass

    def result(self) -> tuple[dict[str, str], list[tuple[int, str, bool]]]:
        return self.headers, self.zuege


def zaehle_partien(
    pgn_pfad: Path,
    max_halbzuege: int = 20,
    max_partien: int | None = None,
) -> tuple[dict[tuple[int, str], list[float]], int]:
    """Liest PGN-Partien und zählt Züge sowie Punkte für jede Stellung."""
    zaehler: dict[tuple[int, str], list[float]] = {}
    verwertete_partien = 0
    gelesene_partien = 0

    def verarbeite(handle: TextIO) -> None:
        nonlocal gelesene_partien, verwertete_partien

        visitor = partial(_PartieVisitor, max_halbzuege=max_halbzuege)
        while max_partien is None or gelesene_partien < max_partien:
            ergebnis = chess.pgn.read_game(handle, Visitor=visitor)
            if ergebnis is None:
                break

            headers, zuege = ergebnis
            gelesene_partien += 1
            if gelesene_partien % 20_000 == 0:
                print(f"{gelesene_partien} Partien gelesen...", flush=True)

            partie_resultat = headers.get("Result")
            if partie_resultat not in {"1-0", "0-1", "1/2-1/2"}:
                continue
            if "Variant" in headers and headers["Variant"] != "Standard":
                continue
            if "SetUp" in headers or "FEN" in headers:
                continue

            verwertete_partien += 1
            for stellung_hash, uci, ziehende_seite in zuege:
                if partie_resultat == "1/2-1/2":
                    punkte = 0.5
                else:
                    weisser_sieg = partie_resultat == "1-0"
                    punkte = float(ziehende_seite == weisser_sieg)

                eintrag = zaehler.setdefault((stellung_hash, uci), [0, 0.0])
                eintrag[0] += 1
                eintrag[1] += punkte

    if pgn_pfad.suffix.lower() == ".zip":
        with zipfile.ZipFile(pgn_pfad) as archiv:
            dateien = sorted(
                (info for info in archiv.infolist() if info.filename.lower().endswith(".pgn")),
                key=lambda info: info.filename,
            )
            for info in dateien:
                if max_partien is not None and gelesene_partien >= max_partien:
                    break
                with archiv.open(info) as datei:
                    with io.TextIOWrapper(datei, encoding="utf-8", errors="replace") as handle:
                        verarbeite(handle)
    else:
        with pgn_pfad.open("r", encoding="utf-8", errors="replace") as handle:
            verarbeite(handle)

    return zaehler, verwertete_partien


def _schreibe_buch_details(
    zaehler: dict[tuple[int, str], list[float]],
    ziel: Path,
    min_partien: int,
    min_punkte: float,
) -> tuple[int, int]:
    """Schreibt das Buch und liefert Anzahl Einträge sowie verschiedene Stellungen."""
    eintraege: list[tuple[int, int, int]] = []
    for (stellung_hash, uci), werte in zaehler.items():
        anzahl = int(werte[0])
        punkte = werte[1]
        if anzahl < min_partien or punkte / anzahl < min_punkte:
            continue
        eintraege.append((stellung_hash, _raw_aus_uci(uci), min(anzahl, 65535)))

    eintraege.sort(key=lambda eintrag: (eintrag[0], -eintrag[2]))
    ziel.parent.mkdir(parents=True, exist_ok=True)
    with ziel.open("wb") as datei:
        for stellung_hash, raw_zug, gewicht in eintraege:
            datei.write(struct.pack(">QHHI", stellung_hash, raw_zug, gewicht, 0))

    return len(eintraege), len({eintrag[0] for eintrag in eintraege})


def schreibe_buch(
    zaehler: dict[tuple[int, str], list[float]],
    ziel: Path,
    min_partien: int = 20,
    min_punkte: float = 0.45,
) -> int:
    """Schreibt gefilterte Züge im Polyglot-Format und liefert die Eintragszahl."""
    anzahl, _ = _schreibe_buch_details(zaehler, ziel, min_partien, min_punkte)
    return anzahl


def main() -> None:
    parser = argparse.ArgumentParser(description="Eröffnungsbuch aus PGN-Partien erstellen")
    parser.add_argument(
        "--pgn",
        type=Path,
        default=Path("theorie/quellen/lichess_elite_2025-11.zip"),
        help="PGN-Datei oder ZIP-Archiv mit PGN-Dateien",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("theorie/buch.bin"),
        help="Zieldatei im Polyglot-Format",
    )
    parser.add_argument(
        "--max-halbzuege",
        type=int,
        default=20,
        help="Höchstens so viele Hauptzüge pro Partie auswerten",
    )
    parser.add_argument(
        "--min-partien",
        type=int,
        default=20,
        help="Mindestzahl an Partien pro Bucheintrag",
    )
    parser.add_argument(
        "--min-punkte",
        type=float,
        default=0.45,
        help="Mindestpunktzahl pro Partie für einen Bucheintrag",
    )
    parser.add_argument(
        "--max-partien",
        type=int,
        default=None,
        help="Höchstens so viele Partien insgesamt lesen",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    zaehler, partien = zaehle_partien(
        args.pgn,
        max_halbzuege=args.max_halbzuege,
        max_partien=args.max_partien,
    )
    eintraege, stellungen = _schreibe_buch_details(
        zaehler,
        args.out,
        min_partien=args.min_partien,
        min_punkte=args.min_punkte,
    )
    dauer = time.perf_counter() - start

    metadaten = {
        "Quelle": args.pgn.name,
        "Datum": date.today().isoformat(),
        "Parameter": {
            "max_halbzuege": args.max_halbzuege,
            "min_partien": args.min_partien,
            "min_punkte": args.min_punkte,
            "max_partien": args.max_partien,
        },
        "Partien": partien,
        "Einträge": eintraege,
        "Stellungen": stellungen,
        "Dauer Sekunden": dauer,
    }
    metadaten_pfad = args.out.with_name("buch.json")
    metadaten_pfad.parent.mkdir(parents=True, exist_ok=True)
    metadaten_pfad.write_text(
        json.dumps(metadaten, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        f"Buch geschrieben: {args.out} — {partien} Partien, "
        f"{eintraege} Einträge in {stellungen} Stellungen "
        f"({dauer:.1f} Sekunden)."
    )


if __name__ == "__main__":
    main()
