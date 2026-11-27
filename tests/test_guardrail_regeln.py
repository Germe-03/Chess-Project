"""
Guardrail-Regeln (Projektaufgabe 3.1) — security/guardrail/regeln.json.

Die Regeln laufen produktiv im Anonymizer des Dozenten. Hier wird geprüft,
dass der Dienst sie beim Speichern annimmt, dass sie die erfundenen Schlüssel
treffen und harmlose Schachfragen in Ruhe lassen, und dass keine Regel in den
Timeout von 0.2 s läuft (der Dienst würde dann jede Anfrage blockieren).
"""
from __future__ import annotations

import pytest

from ops import guardrail as g

KONFIG = g.laden()


def test_regeln_wuerden_im_dienst_gespeichert():
    assert g.pruefen(KONFIG) == []


@pytest.mark.parametrize("art,text,erwartet", g.BEISPIELE,
                         ids=[f"{i:02d}-{b[0]}" for i, b in enumerate(g.BEISPIELE)])
def test_beispiel(art, text, erwartet):
    assert set(g.treffer(KONFIG, text)) == erwartet


@pytest.mark.parametrize("geheim", [g.FAKE_AZURE_KEY, g.FAKE_KLEIN, "0f3a" * 10])
def test_maskierte_schluessel_erreichen_das_llm_nicht(geheim):
    for art, text, erwartet in g.BEISPIELE:
        if art == "treffer" and geheim in text:
            status, ausgabe = g.anwenden(KONFIG, text)
            assert status == "OK"
            assert geheim not in ausgabe


@pytest.mark.parametrize("text", [g.FAKE_GITHUB, g.FAKE_RENDER, g.FAKE_PRIVAT, "Passwort: x"])
def test_block_regeln_halten_die_anfrage_an(text):
    assert g.anwenden(KONFIG, f"Hilf mir bitte: {text}")[0] == "BLOCK"


@pytest.mark.parametrize("boese", [
    "a" * 20_000,
    "Aa1" * 7_000,
    "api_key=" * 2_000,
    "token=" + "0" * 20_000,
    "-----BEGIN " * 2_000,
    "passwort ist " * 1_500,
])
def test_keine_regel_laeuft_in_den_timeout(boese):
    for regel in KONFIG["rules"]:
        list(g.kompilieren(regel).finditer(boese, timeout=g.TIMEOUT))
