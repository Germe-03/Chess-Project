"""
Tests für unterbrechbares Training.

Ein Lauf mit Vorausrechnen dauert Stunden und wird absichtlich in mehreren
Sitzungen gerechnet. Ein Fehler beim Fortsetzen fiele erst nach Stunden auf —
etwa ein Optimierer, der still neu beginnt, oder ein Protokoll, das beim
zweiten Start überschrieben wird. Deshalb wird der ganze Ablauf hier im Kleinen
durchgespielt, mit Endspielpartien, die Millisekunden dauern.
"""
import json
import sys

import numpy as np

from Code.rl import train
from Code.rl.features import FEATURE_SIZE
from Code.rl.train import RunDir
from Code.rl.valuenet import ValueNet


def _daten(seed: int):
    rng = np.random.default_rng(seed)
    X = (rng.random((32, FEATURE_SIZE)) < 0.04).astype(np.float32)
    return X, rng.uniform(-0.8, 0.8, size=32).astype(np.float32)


def test_optimierer_ueberlebt_speichern(tmp_path):
    """Nach dem Laden macht das Netz exakt denselben nächsten Lernschritt."""
    X, ziele = _daten(1)
    netz = ValueNet(hidden_size=16, seed=1)
    for _ in range(3):
        netz.update(X, ziele, 1e-3)
    netz.save(tmp_path / "stand.npz", optimizer=True)
    kopie = ValueNet.load(tmp_path / "stand.npz")

    netz.update(X, ziele, 1e-3)
    kopie.update(X, ziele, 1e-3)
    assert np.array_equal(netz.value(X), kopie.value(X))


def test_modelldatei_bleibt_ohne_optimierer():
    """Die Modelle in models/ sollen klein bleiben; nur Checkpoints tragen Adam mit."""
    assert "adam_steps" not in np.load("models/full_v3.npz").files


def test_lauf_mappe_sichert_ohne_reste(tmp_path):
    lauf = RunDir(tmp_path / "lauf")
    lauf.save(ValueNet(hidden_size=8), {"episodes_done": 3})
    assert lauf.exists()
    assert lauf.load()["episodes_done"] == 3
    assert not list(lauf.path.glob("*.tmp*"))

    assert not lauf.stop_requested()
    (lauf.path / "STOP.txt").write_text("anhalten", encoding="utf-8")   # so legt der Explorer sie an
    assert lauf.stop_requested()
    lauf.clear_stop()
    assert not lauf.stop_requested()


def test_training_laesst_sich_anhalten_und_fortsetzen(tmp_path, monkeypatch):
    """Zwei Sitzungen ergeben zusammen einen vollständigen, lückenlosen Lauf."""
    aufrufe = {"n": 0}

    def stopp_nach_zwei_partien(self):
        aufrufe["n"] += 1
        return aufrufe["n"] > 2

    befehl = [
        "train", "--task", "kqk", "--episodes", "5", "--eval-every", "0", "--log-every", "1",
        "--max-plies", "20", "--run-dir", str(tmp_path / "lauf"), "--out", str(tmp_path / "modell.npz"),
    ]
    monkeypatch.setattr(sys, "argv", befehl)
    monkeypatch.setattr(RunDir, "stop_requested", stopp_nach_zwei_partien)
    train.main()

    zustand = json.loads((tmp_path / "lauf" / "state.json").read_text(encoding="utf-8"))
    assert zustand["episodes_done"] == 2
    assert not zustand["finished"]
    assert not (tmp_path / "modell.npz").exists()

    # Beim Fortsetzen gelten die gespeicherten Einstellungen, nicht der Aufruf.
    monkeypatch.setattr(sys, "argv", befehl[:4] + ["99"] + befehl[5:])
    monkeypatch.setattr(RunDir, "stop_requested", lambda self: False)
    train.main()

    zustand = json.loads((tmp_path / "lauf" / "state.json").read_text(encoding="utf-8"))
    assert zustand["episodes_done"] == 5
    assert zustand["finished"]
    assert zustand["sessions"] == 2
    assert ValueNet.load(tmp_path / "lauf" / "checkpoint.npz")._step_count == 5

    zeilen = (tmp_path / "lauf" / "train.csv").read_text(encoding="utf-8").splitlines()
    assert len(zeilen) == 1 + 5                                  # Kopfzeile plus fünf Partien
    assert [zeile.split(",")[0] for zeile in zeilen[1:]] == ["1", "2", "3", "4", "5"]
    assert (tmp_path / "modell.npz").exists()
