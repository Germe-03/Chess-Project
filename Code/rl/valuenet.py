"""
V(s): die gelernte Bewertungsfunktion als kleines neuronales Netz.

    768 Merkmale -> ReLU-Schicht -> tanh -> V(s) im Bereich [-1, 1]

V(s) schätzt den diskontierten Ausgang der Partie aus Sicht der Zugpartei:
+1 = gewonnen, 0 = remis, -1 = verloren. Die tanh-Ausgabe passt genau zu diesem
Wertebereich.

Bewusst nur NumPy statt PyTorch: Das Training läuft in Minuten auf der CPU, und
der Web-Service braucht dadurch kein schweres ML-Framework - relevant für die
512-MB-Instanz auf render.com. Der Gradient ist von Hand implementiert; es ist
der Semi-Gradient von TD, das heisst das TD-Ziel wird als Konstante behandelt
und es wird nicht durch V(s') zurückpropagiert.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import FEATURE_SIZE


class ValueNet:
    def __init__(
        self,
        input_size: int = FEATURE_SIZE,
        hidden_size: int = 128,
        seed: int = 0,
        shaped: bool = False,
    ):
        # shaped merkt sich, ob dieses Modell mit Reward Shaping trainiert wurde.
        # Es muss beim Ausführen genauso gerechnet werden wie beim Training,
        # sonst wählt der Agent systematisch die falschen Züge.
        self.shaped = shaped
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((input_size, hidden_size)) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden_size, 1)) * np.sqrt(1.0 / hidden_size)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self._reset_adam()

    def _reset_adam(self) -> None:
        self._moments = [(np.zeros_like(p), np.zeros_like(p)) for p in self._params()]
        self._step_count = 0

    def _params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]

    @property
    def hidden_size(self) -> int:
        return int(self.W1.shape[1])

    @property
    def input_size(self) -> int:
        return int(self.W1.shape[0])

    def _fit_width(self, X: np.ndarray) -> np.ndarray:
        """
        Schneidet den Merkmalsvektor auf die Breite, für die dieses Modell
        gebaut wurde. Die Zusatzmerkmale stehen hinten, daher lesen ältere
        Modelle ohne sie einfach den vorderen Teil weiter korrekt.
        """
        return X if X.shape[1] == self.input_size else X[:, : self.input_size]

    def value(self, X: np.ndarray) -> np.ndarray:
        """V(s) für eine Matrix von Stellungen -> (n,)."""
        X = self._fit_width(X)
        hidden = np.maximum(X @ self.W1 + self.b1, 0.0)
        return np.tanh(hidden @ self.W2 + self.b2)[:, 0]

    def update(self, X: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """
        Ein TD-Lernschritt auf einem Batch: Zieht V(s) an das TD-Ziel heran
        (quadratischer Fehler, Adam als Optimierer). Rückgabe: mittlerer
        absoluter TD-Fehler - die Kennzahl für die Lernkurve.
        """
        X = self._fit_width(X)
        n = X.shape[0]
        hidden_pre = X @ self.W1 + self.b1
        hidden = np.maximum(hidden_pre, 0.0)
        v = np.tanh(hidden @ self.W2 + self.b2)

        error = v - targets.reshape(-1, 1)
        delta_out = error * (1.0 - v * v) / n
        grad_W2 = hidden.T @ delta_out
        grad_b2 = delta_out.sum(axis=0)
        delta_hidden = (delta_out @ self.W2.T) * (hidden_pre > 0)
        grad_W1 = X.T @ delta_hidden
        grad_b1 = delta_hidden.sum(axis=0)

        self._adam_step([grad_W1, grad_b1, grad_W2, grad_b2], lr)
        return float(np.mean(np.abs(error)))

    def _adam_step(self, grads, lr: float, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self._step_count += 1
        bias1 = 1.0 - beta1 ** self._step_count
        bias2 = 1.0 - beta2 ** self._step_count
        for param, grad, (m, v) in zip(self._params(), grads, self._moments):
            m *= beta1
            m += (1.0 - beta1) * grad
            v *= beta2
            v += (1.0 - beta2) * (grad * grad)
            param -= lr * (m / bias1) / (np.sqrt(v / bias2) + eps)

    def widened(self, hidden_size: int, seed: int = 0, noise: float = 1e-3) -> "ValueNet":
        """
        Kopie mit breiterer versteckter Schicht, die zunächst exakt dieselbe
        Funktion berechnet (Net2Net, Chen et al. 2016).

        Jede zusätzliche Einheit kopiert eine vorhandene. Damit die Ausgabe
        gleich bleibt, werden die Ausgangsgewichte durch die Anzahl der Kopien
        geteilt — die Summe über alle Kopien ergibt dann wieder das alte
        Gewicht. Gemessen an echten Stellungen beträgt die Abweichung 6e-08,
        also reines Fliesskomma-Rauschen.

        Das kleine Rauschen auf W1 ist nicht Kosmetik, sondern nötig: Identische
        Einheiten bekommen identische Gradienten und blieben für immer gleich.
        Die gewonnene Kapazität läge ohne diesen Anstoss brach.
        """
        if hidden_size < self.hidden_size:
            raise ValueError("Ein Netz lässt sich nur verbreitern, nicht verschmälern.")
        rng = np.random.default_rng(seed)
        quelle = np.concatenate(
            [np.arange(self.hidden_size), rng.integers(0, self.hidden_size, size=hidden_size - self.hidden_size)]
        )
        kopien = np.bincount(quelle, minlength=self.hidden_size)

        breiter = ValueNet(self.input_size, hidden_size, seed=seed, shaped=self.shaped)
        breiter.W1 = self.W1[:, quelle].copy()
        breiter.b1 = self.b1[quelle].copy()
        breiter.W2 = (self.W2[quelle] / kopien[quelle][:, None]).astype(np.float32)
        breiter.b2 = self.b2.copy()
        if noise:
            breiter.W1 += rng.normal(0, noise, breiter.W1.shape).astype(np.float32)
        breiter._reset_adam()
        return breiter

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2, shaped=np.array(self.shaped)
        )

    @classmethod
    def load(cls, path: str | Path) -> "ValueNet":
        data = np.load(Path(path))
        # Die zuerst trainierten Endspielmodelle kennen das Feld noch nicht.
        shaped = bool(data["shaped"]) if "shaped" in data.files else False
        net = cls(
            input_size=int(data["W1"].shape[0]),
            hidden_size=int(data["W1"].shape[1]),
            shaped=shaped,
        )
        net.W1 = data["W1"].astype(np.float32)
        net.b1 = data["b1"].astype(np.float32)
        net.W2 = data["W2"].astype(np.float32)
        net.b2 = data["b2"].astype(np.float32)
        net._reset_adam()
        return net
