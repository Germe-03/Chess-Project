"""
Trainings- und Testdaten für das AutoML-Modell aus dem Prüfbericht.

Quelle ist testbench/bericht.json: 100 Turnierpartien zwischen v1 und v5, in
denen Stockfish 19 jede Stellung mit Tiefe 14 bewertet hat. Das ergibt rund
7 500 verschiedene Stellungen mit einer verlässlichen Bewertung, ohne dass
Stockfish dafür noch einmal rechnen muss.

Aufgeteilt wird nach Partien, nicht nach Stellungen: Stellungen aus derselben
Partie ähneln sich stark. Lägen sie zufällig verteilt in Training und Test,
würde der Test zu gut aussehen. Jede fünfte Partie kommt deshalb in den Test,
und eine Stellung, die auch im Training vorkommt (etwa aus einer gemeinsamen
Eröffnung), wird aus dem Test entfernt.

Aufruf (aus dem Projektordner):
    python -m azure_ml.datensatz

Ergebnis in azure_ml/daten/:
    runde1/               für den ersten AutoML-Auftrag (nur Material)
    runde2/               für den zweiten (mit den zusätzlichen Merkmalen)
    test_stellungen.csv   FEN und richtige Klasse, nur für das Notebook

Die Ordner runde1/ und runde2/ werden in Azure ML als Ganzes hochgeladen. Eine
Datenressource vom Typ "Tabelle" verlangt neben der CSV-Datei eine MLTable-Datei,
die beschreibt, wie die CSV zu lesen ist.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import chess

from .merkmale import SPALTEN_RUNDE_1, SPALTEN_RUNDE_2, ZIELSPALTE, lage, merkmale

ROOT = Path(__file__).resolve().parent.parent
BERICHT = ROOT / "testbench" / "bericht.json"
ZIEL = Path(__file__).resolve().parent / "daten"
TEST_JEDE_N_TE_PARTIE = 5

# Die Spaltentypen stehen fest in der MLTable-Datei. Azure ML errät sie sonst aus
# den ersten Zeilen und hielt w_damen und s_damen für Wahrheitswerte, weil zwei
# Damen selten sind (44 von 6 200 Stellungen); eine spätere Umwandlung in
# Ganzzahlen ergab dann nur noch Fehlerwerte. Deshalb wird gar nicht geraten
# (infer_column_types: false), alles als Text gelesen und jedes Merkmal
# ausdrücklich in eine Ganzzahl umgewandelt.
MLTABLE = """\
$schema: https://azuremlschemas.azureedge.net/latest/MLTable.schema.json
type: mltable
paths:
  - file: ./{datei}
transformations:
  - read_delimited:
      delimiter: ','
      encoding: utf8
      header: all_files_same_headers
      infer_column_types: false
  - convert_column_types:
      - columns: [{spalten}]
        column_type: int
"""


def stellungen() -> tuple[list[dict], list[dict]]:
    """Stellungen aus dem Bericht, getrennt nach Trainings- und Testpartien, ohne Doppelte."""
    bericht = json.loads(BERICHT.read_text(encoding="utf-8"))
    training, test, gesehen = [], [], set()
    for nummer, partie in enumerate(bericht["partien"]):
        ist_test = nummer % TEST_JEDE_N_TE_PARTIE == TEST_JEDE_N_TE_PARTIE - 1
        for zug in partie["zuege"]:
            eintrag = {"fen": zug["fen"], "bewertung": zug["bewertung"], "partie": nummer}
            if ist_test:
                test.append(eintrag)
            elif zug["fen"] not in gesehen:
                gesehen.add(zug["fen"])
                training.append(eintrag)
    test = list({e["fen"]: e for e in test if e["fen"] not in gesehen}.values())
    return training, test


def schreiben(pfad: Path, spalten: list[str], zeilen: list[dict]) -> None:
    with pfad.open("w", newline="", encoding="utf-8") as datei:
        writer = csv.DictWriter(datei, fieldnames=spalten)
        writer.writeheader()
        writer.writerows(zeilen)


def main() -> None:
    training, test = stellungen()
    ZIEL.mkdir(exist_ok=True)

    for runde, spalten in ((1, SPALTEN_RUNDE_1), (2, SPALTEN_RUNDE_2)):
        zeilen = [
            {**merkmale(chess.Board(e["fen"]), runde), ZIELSPALTE: lage(e["bewertung"])}
            for e in training
        ]
        ordner = ZIEL / f"runde{runde}"
        ordner.mkdir(exist_ok=True)
        datei = f"runde{runde}_training.csv"
        schreiben(ordner / datei, spalten + [ZIELSPALTE], zeilen)
        (ordner / "MLTable").write_text(
            MLTABLE.format(datei=datei, spalten=", ".join(spalten)), encoding="utf-8"
        )

    schreiben(
        ZIEL / "test_stellungen.csv",
        ["fen", ZIELSPALTE, "stockfish"],
        [
            {
                "fen": e["fen"],
                ZIELSPALTE: lage(e["bewertung"]),
                "stockfish": f"#{e['bewertung']['matt']}" if "matt" in e["bewertung"] else e["bewertung"]["cp"],
            }
            for e in test
        ],
    )

    for name, daten in (("Training", training), ("Test", test)):
        verteilung = Counter(lage(e["bewertung"]) for e in daten)
        partien = len({e["partie"] for e in daten})
        print(f"{name}: {len(daten)} Stellungen aus {partien} Partien, {dict(verteilung)}")
    print(f"Geschrieben nach {ZIEL}")


if __name__ == "__main__":
    main()
