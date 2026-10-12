"""
Tests für die Stellungserkennung.

Die beiden Testbilder decken die Stile ab, die das Modell können muss: ein
chess.com-Brett und ein Lichess-Brett. Der zweite Fall ist der wichtigere —
er schlägt an, wenn ein Nachtraining die ursprüngliche Fähigkeit zerstört hat
(katastrophales Vergessen). Das wäre sonst erst aufgefallen, wenn jemand einen
Lichess-Screenshot einfügt.

Alle Tests kommen ohne Netzzugriff aus.
"""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from Code.vision.finder import find_board
from Code.vision.recognizer import FEN_CHARS, TileClassifier, tiles_to_fen
from Code.vision.tiles import TILE_SIZE, board_to_tiles

FIXTURES = Path(__file__).parent / "fixtures"
BRETTER = [
    ("brett_chesscom.png", "r3kb1r/pppqpppp/2n2n2/3p1b2/3P1B2/3BPN2/PPP2PPP/RN1QK2R"),
    ("brett_lichess.png", "8/5pk1/6p1/7p/8/4R1PK/5P1P/3r4"),
]

hat_modell = pytest.mark.skipif(
    not TileClassifier.available(), reason="models/tile_classifier.npz fehlt"
)


# --------------------------------------------------------------------------- #
# Kacheln
# --------------------------------------------------------------------------- #
def test_kacheln_form_und_wertebereich():
    bild = Image.open(FIXTURES / BRETTER[0][0])
    kacheln = board_to_tiles(bild)
    assert kacheln.shape == (64, TILE_SIZE, TILE_SIZE, 1)
    assert kacheln.min() >= 0.0 and kacheln.max() <= 1.0


def test_kachelreihenfolge_beginnt_oben_links():
    """Index 0 muss a8 sein, Index 63 h1 — sonst steht die FEN auf dem Kopf."""
    bild = Image.new("RGB", (256, 256), (255, 255, 255))
    for x in range(32):
        for y in range(32):
            bild.putpixel((x, y), (0, 0, 0))     # nur a8 schwarz einfärben
    kacheln = board_to_tiles(bild)
    assert kacheln[0].mean() < 0.1
    assert kacheln[63].mean() > 0.9


# --------------------------------------------------------------------------- #
# FEN-Zusammenbau
# --------------------------------------------------------------------------- #
def test_leeres_brett():
    assert tiles_to_fen(np.zeros(64, dtype=int)) == "8/8/8/8/8/8/8/8"


def test_fen_fasst_leere_felder_zusammen():
    klassen = np.zeros(64, dtype=int)
    klassen[0] = FEN_CHARS.index("r")        # a8
    klassen[7] = FEN_CHARS.index("r")        # h8
    klassen[60] = FEN_CHARS.index("K")       # e1
    assert tiles_to_fen(klassen).split("/")[0] == "r6r"
    assert tiles_to_fen(klassen).split("/")[7] == "4K3"


# --------------------------------------------------------------------------- #
# Bretterkennung im Bild
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("datei,_fen", BRETTER)
def test_findet_brett_im_zugeschnittenen_bild(datei, _fen):
    bild = Image.open(FIXTURES / datei)
    kasten = find_board(bild)
    assert kasten is not None
    breite, hoehe = kasten[2] - kasten[0], kasten[3] - kasten[1]
    assert abs(breite - bild.width) <= 4 and abs(hoehe - bild.height) <= 4


@pytest.mark.parametrize("datei,_fen", BRETTER)
def test_findet_brett_trotz_rand(datei, _fen):
    """Der eigentliche Zweck: ein Screenshot hat Rand, Koordinaten, Umgebung."""
    brett = Image.open(FIXTURES / datei).convert("RGB")
    gross = Image.new("RGB", (brett.width + 160, brett.height + 120), (30, 34, 42))
    gross.paste(brett, (90, 70))
    kasten = find_board(gross)
    assert kasten is not None
    assert max(abs(a - b) for a, b in zip(kasten, (90, 70, 90 + brett.width, 70 + brett.height))) <= 4


def test_kein_brett_im_leeren_bild():
    """Ohne Brett lieber nichts melden als etwas erfinden."""
    assert find_board(Image.new("RGB", (400, 300), (200, 200, 200))) is None


# --------------------------------------------------------------------------- #
# Erkennung von Ende zu Ende
# --------------------------------------------------------------------------- #
@hat_modell
@pytest.mark.parametrize("datei,fen", BRETTER)
def test_erkennt_stellung(datei, fen):
    netz = TileClassifier.load()
    klassen, sicher = netz.predict(board_to_tiles(Image.open(FIXTURES / datei)))
    assert tiles_to_fen(klassen) == fen
    assert sicher.min() > 0.8


@hat_modell
@pytest.mark.parametrize("datei,fen", BRETTER)
def test_erkennt_stellung_aus_screenshot_mit_rand(datei, fen):
    brett = Image.open(FIXTURES / datei).convert("RGB")
    gross = Image.new("RGB", (brett.width + 160, brett.height + 120), (30, 34, 42))
    gross.paste(brett, (90, 70))
    netz = TileClassifier.load()
    klassen, _ = netz.predict(board_to_tiles(gross.crop(find_board(gross))))
    assert tiles_to_fen(klassen) == fen
