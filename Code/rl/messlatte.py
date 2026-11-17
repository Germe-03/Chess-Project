"""Messlatte für eine nachvollziehbare Elo-Schätzung des RL-Agenten.

Warum: Die alte Messung gegen Stockfish mit UCI_Elo 1320 und Tiefe 1 ist
keine echte Elo-Messung. Bei 400 Partien streute v9 zwischen 61.6 und 70.8 %.

Wie: Es werden mehrere Elo-Stufen mit fester Zeit pro Zug und ohne
Tiefenbegrenzung gemessen. Stockfish erhält mit ucinewgame einen frischen Hash
je Partie. Alle Stufen spielen dieselben festen Zufallseröffnungen.

Grenzen: UCI_Elo von Stockfish ist für längere Bedenkzeit geeicht. Bei 0.2 s
pro Zug ist Stockfish eher schwächer, die Schätzung also eher zu hoch. Das
Ergebnis ist keine FIDE-Elo.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable


def leistung(stufe: float, anteil: float) -> float:
    """Berechnet die Leistung aus einer Elo-Stufe und dem erzielten Punkteanteil."""
    p = min(0.99, max(0.01, anteil))
    return stufe + 400 * math.log10(p / (1 - p))


def elo_schaetzen(ergebnisse: list[tuple[int, float, int]]) -> tuple[float, float]:
    """
    Schätzt die eigene Elo per Maximum-Likelihood und berechnet den Standardfehler.

    Jedes Ergebnis enthält Elo-Stufe, erzielte Punkte und Partien.
    """
    if not ergebnisse or sum(partien for _, _, partien in ergebnisse) <= 0:
        raise ValueError("Für die Elo-Schätzung sind Partien erforderlich.")

    ln10_durch_400 = math.log(10) / 400

    def erwartung(stufe: int, elo: float) -> float:
        return 1 / (1 + 10 ** ((stufe - elo) / 400))

    def score(elo: float) -> float:
        return sum(
            punkte - partien * erwartung(stufe, elo)
            for stufe, punkte, partien in ergebnisse
        )

    unten, oben = 0.0, 4000.0
    if score(unten) <= 0:
        schaetzung = unten
    elif score(oben) >= 0:
        schaetzung = oben
    else:
        for _ in range(100):
            mitte = (unten + oben) / 2
            if score(mitte) > 0:
                unten = mitte
            else:
                oben = mitte
        schaetzung = (unten + oben) / 2

    information = sum(
        partien
        * max(erwartung(stufe, schaetzung) * (1 - erwartung(stufe, schaetzung)), 1e-6)
        * ln10_durch_400**2
        for stufe, _, partien in ergebnisse
    )
    return schaetzung, 1 / math.sqrt(information)


def messe(
    version: str,
    stufen: list[int],
    partien: int,
    sf_zeit: float,
    workers: int,
    tiefe: int | None = None,
    bei_stufe: Callable[[dict], None] | None = None,
) -> dict:
    """Misst eine Modellversion gegen Stockfish auf mehreren Elo-Stufen."""
    from . import evaluate, registry
    from .stockfish import find_stockfish

    eintrag = registry.find(version)
    if eintrag is None:
        raise ValueError(f"Unbekannte Version: {version}")
    if partien <= 0:
        raise ValueError("Die Anzahl der Partien muss grösser als null sein.")

    modellpfad = str(registry.MODELS_DIR / eintrag["file"])
    tiefe = tiefe if tiefe is not None else registry.search_depth(version)
    options = registry.search_options(version)
    theorie = registry.theorie_optionen(version)

    sf = find_stockfish(None)
    if sf is None:
        raise FileNotFoundError("Stockfish nicht gefunden.")

    import chess.engine

    engine = chess.engine.SimpleEngine.popen_uci(sf)
    try:
        stockfish_name = engine.id.get("name", "Stockfish")
    finally:
        engine.quit()

    start = perf_counter()
    stufen_ergebnisse = []
    schaetzungsdaten = []

    for stufe in stufen:
        res = evaluate.duel(
            modellpfad,
            tiefe,
            ("stockfish", sf, None, stufe, sf_zeit, True),
            partien,
            400,
            workers,
            options=options,
            theorie=theorie,
        )
        sieg = int(round(res["win_rate"] * partien))
        remis = int(round(res["draw_rate"] * partien))
        niederlage = int(round(res["loss_rate"] * partien))
        punkte = sieg + 0.5 * remis
        stufen_dict = {
            "stufe": stufe,
            "sieg": sieg,
            "remis": remis,
            "niederlage": niederlage,
            "punkte": punkte,
            "leistung": round(leistung(stufe, punkte / partien), 0),
        }
        stufen_ergebnisse.append(stufen_dict)
        schaetzungsdaten.append((stufe, punkte, partien))
        if bei_stufe is not None:
            bei_stufe(stufen_dict)

    elo, fehler = elo_schaetzen(schaetzungsdaten)
    return {
        "version": version,
        "tiefe": tiefe,
        "datum": datetime.now().isoformat(timespec="minutes"),
        "stockfish": stockfish_name,
        "sf_zeit": sf_zeit,
        "partien": partien,
        "stufen": stufen_ergebnisse,
        "elo": round(elo, 0),
        "fehler": round(fehler, 1),
        "dauer_minuten": round((perf_counter() - start) / 60, 1),
    }


def main() -> None:
    """Führt die Messung aus und hängt das Ergebnis an die Ergebnisdatei an."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="RL-Agent auf einer Elo-Messlatte messen")
    parser.add_argument("--version", required=True, help="Zu messende Modellversion")
    parser.add_argument("--tiefe", type=int, default=None, help="Suchtiefe des Modells")
    parser.add_argument("--stufen", type=int, nargs="+", default=[1400, 1600, 1800, 2000])
    parser.add_argument("--partien", type=int, default=200)
    parser.add_argument("--sf-zeit", type=float, default=0.2)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) - 2),
    )
    args = parser.parse_args()

    ordner = Path(__file__).resolve().parents[2] / "messlatte"

    def zeige_stufe(ergebnis: dict) -> None:
        print(
            f"Stufe {ergebnis['stufe']}: {ergebnis['sieg']} S / "
            f"{ergebnis['remis']} R / {ergebnis['niederlage']} N -> "
            f"{ergebnis['punkte'] / args.partien:.1%} "
            f"(Leistung {ergebnis['leistung']:.0f})",
            flush=True,
        )
        # Jede fertige Stufe sofort sichern: Bricht eine lange Messung ab, muss
        # nur die fehlende Stufe nachgeholt werden (--stufen).
        ordner.mkdir(parents=True, exist_ok=True)
        zeile = {"version": args.version, "tiefe": args.tiefe, "sf_zeit": args.sf_zeit,
                 "partien": args.partien, "zeit": datetime.now().isoformat(timespec="minutes"), **ergebnis}
        with (ordner / "stufen.jsonl").open("a", encoding="utf-8") as datei:
            datei.write(json.dumps(zeile, ensure_ascii=False) + "\n")

    ergebnis = messe(
        args.version,
        args.stufen,
        args.partien,
        args.sf_zeit,
        args.workers,
        tiefe=args.tiefe,
        bei_stufe=zeige_stufe,
    )
    print(f"Elo ≈ {ergebnis['elo']:.0f} ± {ergebnis['fehler']:.1f}")

    pfad = Path(__file__).resolve().parents[2] / "messlatte" / "ergebnisse.json"
    pfad.parent.mkdir(parents=True, exist_ok=True)
    ergebnisse = json.loads(pfad.read_text(encoding="utf-8")) if pfad.exists() else []
    ergebnisse.append(ergebnis)
    pfad.write_text(
        json.dumps(ergebnisse, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
