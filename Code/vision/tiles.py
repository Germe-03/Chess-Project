"""
Screenshot -> 64 Felder.

Die Vorverarbeitung folgt exakt dem Modell von linrock/chessboard-recognizer,
sonst passen die Eingaben nicht zu den gelernten Gewichten:

  1. Das Brettbild auf 256x256 skalieren (bilinear)
  2. In Graustufen wandeln, mit den Gewichten 0.2989 / 0.5870 / 0.1140
  3. In 64 Kacheln à 32x32 schneiden, Reihenfolge a8, b8, ... g1, h1
  4. Werte auf [0, 1] bringen

Jede Abweichung davon — andere Skalierung, andere Graustufenformel, andere
Reihenfolge — liefert stillschweigend falsche Erkennungen statt eines Fehlers.
Deshalb steht es hier so ausdrücklich.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

BOARD_SIZE = 256
TILE_SIZE = 32
GRAYSCALE_WEIGHTS = (0.2989, 0.5870, 0.1140, 0.0)


def board_to_tiles(image: Image.Image) -> np.ndarray:
    """Brettbild -> (64, 32, 32, 1) in der Reihenfolge a8 bis h1."""
    board = image.convert("RGB").resize([BOARD_SIZE, BOARD_SIZE], Image.BILINEAR)
    gray = np.asarray(board.convert("L", GRAYSCALE_WEIGHTS), dtype=np.float32) / 255.0

    tiles = np.empty((64, TILE_SIZE, TILE_SIZE, 1), dtype=np.float32)
    for rank in range(8):
        for file in range(8):
            patch = gray[rank * TILE_SIZE:(rank + 1) * TILE_SIZE,
                         file * TILE_SIZE:(file + 1) * TILE_SIZE]
            tiles[rank * 8 + file, :, :, 0] = patch
    return tiles
