"""
Tests für die Anbindung der Schachtheorie (ab v10) an Registry, Zugwahl,
Messung und Prüfstand.

Die echten Theorie-Dateien liegen nur lokal. Darum bauen die Tests ein
winziges Eröffnungsbuch selbst und hängen es über theorie_fuer ein.
"""
import json
import struct

import chess
import chess.polyglot
import pytest

from Code import main
from Code.rl import evaluate, registry, testbench
from Code.rl.search import SearchOptions
from Code.rl.theorie import KEINE, Theorie, TheorieOptionen
from Code.rl.valuenet import ValueNet


def _buch(pfad):
    """Grundstellung: d2d4 (Gewicht 30) vor e2e4 (Gewicht 10)."""
    key = chess.polyglot.zobrist_hash(chess.Board())
    eintraege = [(key, chess.D4 | (chess.D2 << 6), 30), (key, chess.E4 | (chess.E2 << 6), 10)]
    pfad.write_bytes(b"".join(struct.pack(">QHHI", k, m, w, 0) for k, m, w in eintraege))
    return pfad


@pytest.fixture
def test_registry(tmp_path, monkeypatch):
    """Eigene Registry mit v9 und einer Theorie-Version, beide mit dem Netz von v9."""
    daten = {"active": "v9", "versions": [
        {"name": "v9", "file": "full_v9.npz", "trained": "2026-11-13", "episodes": 0, "parent": None,
         "search_depth": 2, "cap": 0.9, "repetition_draw": True, "notes": "", "results": {}},
    ]}
    pfad = tmp_path / "registry.json"
    pfad.write_text(json.dumps(daten), encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", pfad)
    registry.register("vt", "full_v9.npz", 0, "v9", depth=2, activate=False,
                      options=SearchOptions(cap=0.9, repetition_draw=True),
                      theorie=TheorieOptionen(eroeffnung=True))
    buch = _buch(tmp_path / "buch.bin")
    monkeypatch.setattr(main, "theorie_fuer", lambda o: Theorie(o, buch_pfad=buch) if o else None)
    monkeypatch.setattr(evaluate, "theorie_fuer", lambda o: Theorie(o, buch_pfad=buch) if o else None)
    return pfad


def test_registry_traegt_theorie_nur_ein_wenn_eingeschaltet(test_registry):
    eintraege = {v["name"]: v for v in registry.versions()}
    assert "theorie" not in eintraege["v9"]
    assert eintraege["vt"]["theorie"] == {"eroeffnung": True, "endspiel": False}
    assert registry.theorie_optionen("vt") == TheorieOptionen(eroeffnung=True)
    assert registry.theorie_optionen("v9") == KEINE


def test_versionen_ohne_theorie_dateien_sind_ausgeblendet(test_registry, monkeypatch):
    monkeypatch.setattr(registry, "verfuegbar", lambda optionen: not optionen)
    assert [v["name"] for v in registry.sichtbare_versionen()] == ["v9"]
    monkeypatch.setattr(registry, "verfuegbar", lambda optionen: True)
    assert [v["name"] for v in registry.sichtbare_versionen()] == ["v9", "vt"]


def test_zugwahl_nimmt_buchzug_und_sonst_das_netz(test_registry):
    buchzug = main.choose_move(chess.Board(), "vt")
    assert buchzug.move == chess.Move.from_uci("d2d4")
    assert buchzug.engine == "theorie" and buchzug.label == "vt-buch"
    ohne = main.choose_move(chess.Board(), "v9")
    assert ohne.engine == "rl" and ohne.depth == 2
    # Ausserhalb des Buchs rechnet die Theorie-Version genau wie v9.
    board = chess.Board()
    board.push_uci("g1f3")
    assert main.choose_move(board, "vt").move == main.choose_move(board, "v9").move


def test_messung_nimmt_buchzug_nur_mit_theorie(test_registry):
    netz = ValueNet.load(registry.MODELS_DIR / "full_v9.npz")
    optionen = SearchOptions(cap=0.9, repetition_draw=True)
    mit = evaluate.make_policy("rl", None, netz, 0.99, 2, optionen, TheorieOptionen(eroeffnung=True))
    ohne = evaluate.make_policy("rl", None, netz, 0.99, 2, optionen)
    assert mit(chess.Board()) == chess.Move.from_uci("d2d4")
    board = chess.Board()
    board.push_uci("g1f3")
    assert mit(board) == ohne(board)


def test_pruefstand_unterscheidet_theorie_versionen(test_registry):
    eintraege = {v["name"]: v for v in registry.versions()}
    ohne, mit = testbench.fingerprint(eintraege["v9"]), testbench.fingerprint(eintraege["vt"])
    assert mit == ohne + ":buch"
