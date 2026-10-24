"""
Lagebewertung über Azure ML (POST /lage).

Kein Netzzugriff: Der Aufruf an Azure wird durch eine nachgebaute Antwort
ersetzt. Geprüft wird, was still schiefgehen kann — falsche Spaltenreihenfolge
(das Modell würde dann unbemerkt Unsinn rechnen), ein Schlüssel, der ungewollt
fehlt, und dass ein abgeschalteter Endpoint den Dienst nicht mitreisst.
"""
from __future__ import annotations

import io
import json
import urllib.error

import chess
import pytest
from fastapi import HTTPException

from Code import azure_lage
from Code.main import LageRequest, lage
from azure_ml.merkmale import SPALTEN_RUNDE_2

DAME_HAENGT = "rnb1kbnr/pppp1ppp/8/4p3/4P2q/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"


class Antwort(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def ohne_schluessel(monkeypatch, tmp_path):
    monkeypatch.delenv("AZURE_ML_KEY", raising=False)
    monkeypatch.setattr(azure_lage, "SCHLUESSEL_DATEI", tmp_path / "fehlt.txt")


@pytest.fixture
def mit_schluessel(monkeypatch):
    monkeypatch.setenv("AZURE_ML_KEY", "testschluessel")
    monkeypatch.delenv("AZURE_ML_DEPLOYMENT", raising=False)


def test_ohne_schluessel_ist_die_funktion_aus(ohne_schluessel):
    assert not azure_lage.konfiguriert()
    with pytest.raises(HTTPException) as fehler:
        lage(LageRequest(fen=chess.STARTING_FEN))
    assert fehler.value.status_code == 503


def test_anfrage_hat_die_spalten_des_trainings(mit_schluessel, monkeypatch):
    gesendet = {}

    def urlopen(anfrage, timeout):
        gesendet["kopf"] = dict(anfrage.header_items())
        gesendet["daten"] = json.loads(anfrage.data)
        return Antwort(json.dumps(["weiss_besser"]).encode())

    monkeypatch.setattr(azure_lage.urllib.request, "urlopen", urlopen)
    antwort = lage(LageRequest(fen=DAME_HAENGT))

    assert antwort.lage == "weiss_besser"
    eingabe = gesendet["daten"]["input_data"]
    assert eingabe["columns"] == SPALTEN_RUNDE_2
    assert len(eingabe["data"][0]) == 24
    zeile = dict(zip(eingabe["columns"], eingabe["data"][0]))
    assert zeile["bilanz_nach_bestem_schlag"] == 9      # Weiss schlägt die Dame
    assert gesendet["kopf"]["Authorization"] == "Bearer testschluessel"


def test_abgeschalteter_endpoint_gibt_502(mit_schluessel, monkeypatch):
    def urlopen(anfrage, timeout):
        raise urllib.error.URLError("Name or service not known")

    monkeypatch.setattr(azure_lage.urllib.request, "urlopen", urlopen)
    with pytest.raises(HTTPException) as fehler:
        lage(LageRequest(fen=chess.STARTING_FEN))
    assert fehler.value.status_code == 502


def test_unerwartete_antwort_gibt_502(mit_schluessel, monkeypatch):
    monkeypatch.setattr(azure_lage.urllib.request, "urlopen",
                        lambda anfrage, timeout: Antwort(b'{"error": "boom"}'))
    with pytest.raises(HTTPException) as fehler:
        lage(LageRequest(fen=chess.STARTING_FEN))
    assert fehler.value.status_code == 502


def test_ungueltige_fen_gibt_400(mit_schluessel):
    with pytest.raises(HTTPException) as fehler:
        lage(LageRequest(fen="kein schach"))
    assert fehler.value.status_code == 400
