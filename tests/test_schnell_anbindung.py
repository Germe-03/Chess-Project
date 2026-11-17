"""
Tests für die Anbindung der schnellen Suche (ab v11) an Registry, Zugwahl und
Prüfstand: Schalter im Manifest, Kennung im Prüfbericht und Ausblenden von
Versionen, die tiefer rechnen, als der Server erlaubt.
"""
import json

import chess
import pytest

from Code import main
from Code.rl import registry, testbench
from Code.rl.search import SearchOptions, search
from Code.rl.valuenet import ValueNet


@pytest.fixture
def test_registry(tmp_path, monkeypatch):
    """Eigene Registry: v9 und eine Version mit schneller Suche und Tiefe 3."""
    daten = {"active": "v9", "versions": [
        {"name": "v9", "file": "full_v9.npz", "trained": "2026-11-13", "episodes": 0, "parent": None,
         "search_depth": 2, "cap": 0.9, "repetition_draw": True, "notes": "", "results": {}},
    ]}
    pfad = tmp_path / "registry.json"
    pfad.write_text(json.dumps(daten), encoding="utf-8")
    monkeypatch.setattr(registry, "REGISTRY_PATH", pfad)
    registry.register("vs", "full_v9.npz", 0, "v9", depth=3, activate=False,
                      options=SearchOptions(cap=0.9, repetition_draw=True, schnell=True))
    return pfad


def test_schalter_steht_im_manifest_und_kommt_zurueck(test_registry):
    eintraege = {v["name"]: v for v in registry.versions()}
    assert "schnelle_suche" not in eintraege["v9"]
    assert eintraege["vs"]["schnelle_suche"] is True
    assert registry.search_options("vs") == SearchOptions(cap=0.9, repetition_draw=True, schnell=True)
    assert registry.search_options("v9").schnell is False


def test_pruefstand_unterscheidet_schnelle_suche(test_registry):
    eintraege = {v["name"]: v for v in registry.versions()}
    assert testbench.fingerprint(eintraege["vs"]).endswith(":wdh:schnell")
    assert ":schnell" not in testbench.fingerprint(eintraege["v9"])


def test_tiefere_versionen_sind_auf_dem_server_verborgen(test_registry, monkeypatch):
    monkeypatch.setattr(main, "MAX_SEARCH_DEPTH", 2)
    assert [v["name"] for v in main.list_versions()["versions"]] == ["v9"]
    monkeypatch.setattr(main, "MAX_SEARCH_DEPTH", 99)
    assert [v["name"] for v in main.list_versions()["versions"]] == ["v9", "vs"]


def test_search_leitet_mit_schalter_auf_die_schnelle_suche_um(test_registry):
    netz = ValueNet.load(registry.MODELS_DIR / "full_v9.npz")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    schnell = search(board, netz, 2, options=SearchOptions(cap=0.9, repetition_draw=True, schnell=True))
    klassisch = search(board, netz, 2, options=SearchOptions(cap=0.9, repetition_draw=True))
    assert schnell.move == chess.Move.from_uci("h5f7")       # Schäfermatt
    assert abs(schnell.value - klassisch.value) < 1e-5
