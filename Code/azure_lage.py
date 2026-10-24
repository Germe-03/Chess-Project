"""
Lagebewertung über das Azure-ML-Modell aus Projektaufgabe 2.

Das AutoML-Modell (Deployment "runde2" im Endpoint schach-lage-dgerm) sagt aus
24 Merkmalen einer Stellung voraus, wer besser steht: "weiss_besser",
"ausgeglichen" oder "schwarz_besser". Es ergänzt die eigene Engine um eine
zweite, unabhängig trainierte Einschätzung.

Der Dienst ruft den Endpoint serverseitig auf. Der Schlüssel bleibt damit auf
dem Server und gelangt nie in den Browser.

Konfiguration:
    AZURE_ML_URL         REST-Endpunkt (Standard: schach-lage-dgerm)
    AZURE_ML_KEY         Primärschlüssel des Endpoints. Lokal genügt stattdessen
                         die Datei azure_ml/endpoint_key.txt (steht in .gitignore).
    AZURE_ML_DEPLOYMENT  optional: eine bestimmte Bereitstellung ansprechen,
                         sonst entscheidet die Verkehrsverteilung im Endpoint

Ohne Schlüssel ist die Funktion abgeschaltet. Läuft der Endpoint nicht (etwa
weil er nach der Abgabe gelöscht wurde), meldet der Aufruf das als Fehler und
die Oberfläche zeigt «nicht verfügbar» — der Rest des Dienstes ist davon nicht
betroffen.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

import chess

from azure_ml.merkmale import SPALTEN_RUNDE_2, merkmale

STANDARD_URL = "https://schach-lage-dgerm.switzerlandnorth.inference.ml.azure.com/score"
SCHLUESSEL_DATEI = Path(__file__).resolve().parent.parent / "azure_ml" / "endpoint_key.txt"
TIMEOUT_SEKUNDEN = 15
KLASSEN = ("weiss_besser", "ausgeglichen", "schwarz_besser")


class NichtKonfiguriert(RuntimeError):
    """Kein Schlüssel vorhanden — die Azure-Bewertung ist abgeschaltet."""


class AzureFehler(RuntimeError):
    """Der Endpoint war nicht erreichbar oder hat unerwartet geantwortet."""


def url() -> str:
    return os.environ.get("AZURE_ML_URL") or STANDARD_URL


def schluessel() -> str | None:
    wert = os.environ.get("AZURE_ML_KEY", "").strip()
    if not wert and SCHLUESSEL_DATEI.exists():
        wert = SCHLUESSEL_DATEI.read_text(encoding="utf-8").strip()
    return wert or None


def konfiguriert() -> bool:
    return schluessel() is not None


def nutzlast(board: chess.Board) -> dict:
    """Anfrage im MLflow-Format, mit denselben Merkmalen wie beim Training."""
    zeile = merkmale(board, runde=2)
    return {"input_data": {"columns": SPALTEN_RUNDE_2, "index": [0],
                           "data": [[zeile[s] for s in SPALTEN_RUNDE_2]]}}


def bewerte(board: chess.Board) -> str:
    """Fragt den Endpoint und gibt eine der drei Klassen zurück."""
    key = schluessel()
    if key is None:
        raise NichtKonfiguriert("Kein Schlüssel für den Azure-ML-Endpoint hinterlegt.")

    kopf = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    deployment = os.environ.get("AZURE_ML_DEPLOYMENT")
    if deployment:
        kopf["azureml-model-deployment"] = deployment
    anfrage = urllib.request.Request(url(), json.dumps(nutzlast(board)).encode(), kopf)
    try:
        with urllib.request.urlopen(anfrage, timeout=TIMEOUT_SEKUNDEN) as antwort:
            daten = json.loads(antwort.read())
    except (urllib.error.URLError, TimeoutError, ValueError) as fehler:
        raise AzureFehler(f"Azure-Endpoint nicht erreichbar: {fehler}") from fehler

    if isinstance(daten, str):
        daten = json.loads(daten)
    if isinstance(daten, dict):
        daten = daten.get("Results") or daten.get("predictions") or daten.get("result")
    if not isinstance(daten, list) or not daten or daten[0] not in KLASSEN:
        raise AzureFehler(f"Unerwartete Antwort vom Azure-Endpoint: {str(daten)[:200]}")
    return daten[0]
