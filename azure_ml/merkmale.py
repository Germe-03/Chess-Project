"""
Merkmale einer Schachstellung für das AutoML-Modell in Azure ML (Projektaufgabe 2).

Das Modell sagt aus einer Stellung voraus, wer besser steht: "weiss_besser",
"ausgeglichen" oder "schwarz_besser". Es bekommt keine FEN, sondern eine Zeile
mit Zahlen, wie AutoML sie für tabellarische Daten erwartet.

Zwei Runden, damit sich "trainieren, deployen, verbessern, deployen" messen lässt:

  Runde 1  nur das Material: wie viele Figuren jeder Art jede Seite hat, und wer
           am Zug ist. Das ist die naheliegendste Grundlage, sieht aber nichts von
           dem, was im nächsten Zug passiert.
  Runde 2  zusätzlich Merkmale, die die Engine des Projekts schon kennt: die
           Materialbilanz nach dem besten Schlagzug (statische Abtauschbewertung
           aus Code/rl/shaping.py), Mobilität, Schach, Druck auf den König und
           Freibauern.

Dieselbe Funktion erzeugt die Trainingsdaten (azure_ml/datensatz.py) und die
Anfragen an den Endpoint (azure_ml/endpoint_aufruf.ipynb). So bekommt das Modell
im Betrieb genau die Merkmale, mit denen es trainiert wurde.
"""
from __future__ import annotations

import chess

from Code.rl.shaping import PIECE_VALUES, best_exchange

FIGUREN = [
    ("bauern", chess.PAWN),
    ("springer", chess.KNIGHT),
    ("laeufer", chess.BISHOP),
    ("tuerme", chess.ROOK),
    ("damen", chess.QUEEN),
]

SPALTEN_RUNDE_1 = ["weiss_am_zug"] + [
    f"{prefix}_{name}" for prefix in ("w", "s") for name, _ in FIGUREN
]

SPALTEN_RUNDE_2 = SPALTEN_RUNDE_1 + [
    "materialbilanz",
    "bilanz_nach_bestem_schlag",
    "schlaggewinn_weiss",
    "schlaggewinn_schwarz",
    "legale_zuege",
    "mobilitaet_weiss",
    "mobilitaet_schwarz",
    "im_schach",
    "koenigsdruck_auf_weiss",
    "koenigsdruck_auf_schwarz",
    "freibauern_weiss",
    "freibauern_schwarz",
    "figurenmaterial",
]

ZIELSPALTE = "lage"

# Ab diesem Vorteil laut Stockfish (in Centipawns, aus Sicht von Weiss) gilt eine
# Seite als besser. 150 entspricht anderthalb Bauern, laut der Lichess-Formel
# rund 64 Prozent Gewinnerwartung.
SCHWELLE_CP = 150


def lage(bewertung: dict) -> str:
    """Zielklasse aus der Stockfish-Bewertung des Prüfberichts ({"cp": ...} oder {"matt": ...})."""
    if "matt" in bewertung:
        return "weiss_besser" if bewertung["matt"] > 0 else "schwarz_besser"
    cp = bewertung["cp"]
    if cp > SCHWELLE_CP:
        return "weiss_besser"
    if cp < -SCHWELLE_CP:
        return "schwarz_besser"
    return "ausgeglichen"


def merkmale(board: chess.Board, runde: int = 2) -> dict:
    """Merkmale der Stellung als Spaltenname → Zahl, in der Reihenfolge der Runde."""
    zeile = {"weiss_am_zug": int(board.turn == chess.WHITE)}
    for prefix, farbe in (("w", chess.WHITE), ("s", chess.BLACK)):
        for name, typ in FIGUREN:
            zeile[f"{prefix}_{name}"] = len(board.pieces(typ, farbe))
    if runde == 1:
        return zeile

    bilanz = _material(board, chess.WHITE) - _material(board, chess.BLACK)
    gewinn_weiss = best_exchange(board, chess.WHITE)
    gewinn_schwarz = best_exchange(board, chess.BLACK)
    # Wer am Zug ist, kann seinen Schlaggewinn sofort einstreichen.
    sofort = gewinn_weiss if board.turn == chess.WHITE else -gewinn_schwarz

    zeile.update(
        materialbilanz=bilanz,
        bilanz_nach_bestem_schlag=bilanz + sofort,
        schlaggewinn_weiss=gewinn_weiss,
        schlaggewinn_schwarz=gewinn_schwarz,
        legale_zuege=board.legal_moves.count(),
        mobilitaet_weiss=_mobilitaet(board, chess.WHITE),
        mobilitaet_schwarz=_mobilitaet(board, chess.BLACK),
        im_schach=int(board.is_check()),
        koenigsdruck_auf_weiss=_koenigsdruck(board, chess.WHITE),
        koenigsdruck_auf_schwarz=_koenigsdruck(board, chess.BLACK),
        freibauern_weiss=_freibauern(board, chess.WHITE),
        freibauern_schwarz=_freibauern(board, chess.BLACK),
        figurenmaterial=sum(
            PIECE_VALUES[typ] * len(board.pieces(typ, farbe))
            for farbe in (chess.WHITE, chess.BLACK)
            for typ in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN)
        ),
    )
    return {spalte: zeile[spalte] for spalte in SPALTEN_RUNDE_2}


def _material(board: chess.Board, farbe: chess.Color) -> int:
    return sum(PIECE_VALUES[typ] * len(board.pieces(typ, farbe)) for _, typ in FIGUREN)


def _mobilitaet(board: chess.Board, farbe: chess.Color) -> int:
    """Pseudolegale Züge von 'farbe', auch wenn die andere Seite am Zug ist."""
    kopie = board.copy(stack=False)
    kopie.turn = farbe
    kopie.ep_square = None
    return kopie.pseudo_legal_moves.count()


def _koenigsdruck(board: chess.Board, farbe: chess.Color) -> int:
    """Angriffe des Gegners auf den König von 'farbe' und die Felder um ihn herum."""
    koenig = board.king(farbe)
    if koenig is None:
        return 0
    zone = chess.SquareSet(chess.BB_KING_ATTACKS[koenig]) | chess.SquareSet.from_square(koenig)
    return sum(len(board.attackers(not farbe, feld)) for feld in zone)


def _freibauern(board: chess.Board, farbe: chess.Color) -> int:
    """Bauern von 'farbe', die auf ihrer und den Nachbarlinien keinen gegnerischen Bauern mehr vor sich haben."""
    gegner = board.pieces(chess.PAWN, not farbe)
    anzahl = 0
    for feld in board.pieces(chess.PAWN, farbe):
        linie, reihe = chess.square_file(feld), chess.square_rank(feld)
        vor_ihm = (
            (lambda r: r > reihe) if farbe == chess.WHITE else (lambda r: r < reihe)
        )
        blockiert = any(
            abs(chess.square_file(g) - linie) <= 1 and vor_ihm(chess.square_rank(g))
            for g in gegner
        )
        anzahl += not blockiert
    return anzahl
