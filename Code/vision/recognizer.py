"""
Figurenerkennung: 64 Kacheln -> FEN.

Das Netz stammt aus linrock/chessboard-recognizer (MIT-Lizenz) und wurde dort
mit TensorFlow trainiert. Hier läuft nur der Vorwärtsdurchlauf, und zwar in
reinem NumPy — aus demselben Grund wie beim Value-Netz: Der Web-Dienst soll
ohne schweres ML-Rahmenwerk auskommen. TensorFlow installiert wäre rund 600 MB
und würde die 512-MB-Instanz auf render.com sprengen; die Gewichte allein sind
0.5 MB.

Architektur (unverändert übernommen):

    Eingabe 32x32x1, Graustufen in [0, 1]
    Conv2D(32, 3x3) + ReLU   ->  MaxPool 2x2
    Conv2D(64, 3x3) + ReLU   ->  MaxPool 2x2
    Conv2D(64, 3x3) + ReLU
    Flatten -> Dense(64) + ReLU -> Dense(13) + Softmax

Ehrliche Einordnung für die Dokumentation: Dieses Modell ist **übernommen**,
nicht selbst trainiert. Es ist damit eine dritte Kategorie neben dem selbst
gelernten V(s) und dem handgeschriebenen Φ.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Index 0 steht für ein leeres Feld, danach weisse und schwarze Figuren.
FEN_CHARS = "1RNBQKPrnbqkp"
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "tile_classifier.npz"


def _conv2d(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Gültige Faltung über einen Stapel Bilder: (n, h, w, c_in) -> (n, h-2, w-2, c_out).

    Umgesetzt als eine einzige Matrixmultiplikation über alle Fensterpositionen
    (im Deep Learning als im2col bekannt). Eine Schleife über die Pixel wäre in
    NumPy um Grössenordnungen langsamer.
    """
    n, h, w, c_in = x.shape
    kh, kw, _, c_out = kernel.shape
    oh, ow = h - kh + 1, w - kw + 1

    # Fenster ohne Kopie über die Schrittweiten des Speichers auslegen
    s = x.strides
    windows = np.lib.stride_tricks.as_strided(
        x, shape=(n, oh, ow, kh, kw, c_in), strides=(s[0], s[1], s[2], s[1], s[2], s[3])
    )
    flat = windows.reshape(n * oh * ow, kh * kw * c_in)
    out = flat @ kernel.reshape(kh * kw * c_in, c_out) + bias
    return out.reshape(n, oh, ow, c_out)


def _maxpool2d(x: np.ndarray) -> np.ndarray:
    """2x2-Maximum ohne Überlappung; ungerade Ränder fallen weg, wie bei Keras."""
    n, h, w, c = x.shape
    h, w = h - h % 2, w - w % 2
    return x[:, :h, :w, :].reshape(n, h // 2, 2, w // 2, 2, c).max(axis=(2, 4))


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class TileClassifier:
    """Klassifiziert 32x32-Kacheln in eine der 13 FEN-Klassen."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights

    @classmethod
    def load(cls, path: str | Path = MODEL_PATH) -> "TileClassifier":
        data = np.load(Path(path))
        return cls({k: data[k].astype(np.float32) for k in data.files})

    @classmethod
    def available(cls, path: str | Path = MODEL_PATH) -> bool:
        return Path(path).exists()

    def predict(self, tiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(64, 32, 32, 1) -> Klassenindizes und zugehörige Wahrscheinlichkeiten."""
        w = self.w
        x = _maxpool2d(np.maximum(_conv2d(tiles, w["conv1_w"], w["conv1_b"]), 0.0))
        x = _maxpool2d(np.maximum(_conv2d(x, w["conv2_w"], w["conv2_b"]), 0.0))
        x = np.maximum(_conv2d(x, w["conv3_w"], w["conv3_b"]), 0.0)
        x = x.reshape(x.shape[0], -1)
        x = np.maximum(x @ w["dense1_w"] + w["dense1_b"], 0.0)
        probs = _softmax(x @ w["dense2_w"] + w["dense2_b"])
        return probs.argmax(axis=1), probs.max(axis=1)


def tiles_to_fen(classes: np.ndarray) -> str:
    """
    64 Klassenindizes (a8 bis h1) -> Figurenteil einer FEN.

    Leere Felder werden wie üblich zu Zahlen zusammengefasst.
    """
    rows = []
    for rank in range(8):
        row, leer = "", 0
        for file in range(8):
            zeichen = FEN_CHARS[int(classes[rank * 8 + file])]
            if zeichen == "1":
                leer += 1
                continue
            if leer:
                row += str(leer)
                leer = 0
            row += zeichen
        if leer:
            row += str(leer)
        rows.append(row)
    return "/".join(rows)
