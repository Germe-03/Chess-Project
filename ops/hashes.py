"""
Lieferkette absichern: Hashes der Modelldateien (OWASP LLM04).

Modellgewichte sind Teil der Supply Chain. Die gleiche Datei unter dem
gleichen Namen, aber mit anderem Inhalt, fällt nicht auf -- sie fällt
nur über den Hash auf. Deshalb schreibt und prüft dieses Modul
models/SHA256SUMS im Format von sha256sum.

Aufruf:
    python -m ops.hashes --write     Manifest neu schreiben
    python -m ops.hashes --verify    Arbeitsbaum gegen das Manifest prüfen
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess  # nosec B404 # nur für den festen git-Aufruf unten
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "models/SHA256SUMS"

BLOCK = 1024 * 1024  # 1 MiB


def sha256_datei(pfad: Path) -> str:
    """SHA-256 einer Datei als kleiner Hex-String; gelesen wird in 1-MiB-Blöcken."""
    digest = hashlib.sha256()
    with pfad.open("rb") as datei:
        while block := datei.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def manifest_lesen(pfad: Path) -> dict[str, str]:
    """Manifest lesen: Zeilen '<64hex>  <pfad>' werden zu {pfad: hash}; leere Zeilen werden ignoriert."""
    eintraege: dict[str, str] = {}
    for zeile in pfad.read_text(encoding="utf-8").splitlines():
        if not zeile.strip():
            continue
        soll, _, name = zeile.partition("  ")
        eintraege[name] = soll
    return eintraege


def pruefen(root: Path) -> list[str]:
    """
    Arbeitsbaum gegen das Manifest prüfen -- ganz ohne git.

    Die Pfade im Manifest stehen mit '/' relativ zu root. Zurück kommt
    eine Liste von Fehlerzeilen; ist sie leer, ist alles in Ordnung.
    """
    fehler: list[str] = []
    manifest_pfad = root / MANIFEST
    if not manifest_pfad.is_file():
        return [f"MANIFEST FEHLT {MANIFEST}"]

    eintraege = manifest_lesen(manifest_pfad)
    for pfad, soll in eintraege.items():
        datei = root / pfad
        if not datei.exists():
            fehler.append(f"FEHLT {pfad}")
        elif sha256_datei(datei) != soll:
            fehler.append(f"ABWEICHUNG {pfad}")

    # Jede in registry.json eingetragene Version muss auch im Manifest stehen.
    registry_pfad = root / "models" / "registry.json"
    if registry_pfad.is_file():
        daten = json.loads(registry_pfad.read_text(encoding="utf-8"))
        for version in daten.get("versions", []):
            pfad = "models/" + version["file"]
            if pfad not in eintraege:
                fehler.append(f"NICHT IM MANIFEST {pfad}")
    return fehler


def schreiben(root: Path) -> int:
    """
    Manifest neu schreiben: alle von git verfolgten Modelldateien hashen.

    Das Manifest selbst und Dateien, die es im Arbeitsbaum nicht gibt,
    bleiben draussen. Zurück kommt die Zahl der eingetragenen Dateien.
    """
    ergebnis = subprocess.run(  # nosec B603 B607 # fester Befehl ohne Shell und ohne Benutzereingabe
        ["git", "ls-files", "models", "testbench/stellungen.json"],
        cwd=root, capture_output=True, text=True, check=True,
    )
    zeilen: list[str] = []
    for pfad in sorted(zeile for zeile in ergebnis.stdout.splitlines() if zeile):
        if pfad == MANIFEST:
            continue  # das Manifest selbst wird nicht mit sich selbst gehasht
        datei = root / pfad
        if not datei.is_file():
            continue  # nur Dateien hashen, die es im Arbeitsbaum gibt
        zeilen.append(f"{sha256_datei(datei)}  {pfad}\n")
    manifest_pfad = root / MANIFEST
    manifest_pfad.parent.mkdir(parents=True, exist_ok=True)
    with manifest_pfad.open("w", encoding="utf-8", newline="\n") as manifest:
        manifest.writelines(zeilen)
    return len(zeilen)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Modellgewichte hashen: {MANIFEST} schreiben oder damit prüfen"
    )
    modus = parser.add_mutually_exclusive_group(required=True)
    modus.add_argument("--write", action="store_true",
                       help="Manifest neu schreiben (alle von git verfolgten Modelldateien)")
    modus.add_argument("--verify", action="store_true",
                       help="Dateien im Arbeitsbaum gegen das Manifest prüfen")
    args = parser.parse_args()

    if args.write:
        print(f"{schreiben(ROOT)} Dateien nach {MANIFEST} geschrieben")
        return

    fehler = pruefen(ROOT)
    if fehler:
        for zeile in fehler:
            print(zeile)
        print(f"FEHLER: {len(fehler)} Abweichungen")
        sys.exit(1)
    print(f"OK: {len(manifest_lesen(ROOT / MANIFEST))} Dateien geprüft")


if __name__ == "__main__":
    main()
