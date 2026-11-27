"""
Schlüssel-Suche im ganzen Git-Verlauf (Projektaufgabe 3.2, Supply Chain).

detect-secrets prüft nur den heutigen Stand der Dateien. Ein Schlüssel, der
einmal committet und danach wieder gelöscht wurde, steckt aber weiter im
Verlauf und ist bei einem öffentlichen Repo für alle lesbar. Darum geht dieses
Skript jede Änderung jedes Commits durch.

    python -m ops.schluessel_suche

Treffer werden nur maskiert ausgegeben (erste vier Zeichen und Länge).
Liegt azure_ml/endpoint_key.txt lokal vor, wird zusätzlich der echte
Endpoint-Schlüssel exakt gesucht, ohne ihn anzuzeigen.
Exit-Code 1, sobald etwas gefunden wird.
"""
from __future__ import annotations

import re
import subprocess  # nosec B404 # nur für den festen git-Aufruf unten
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCHLUESSEL_DATEI = ROOT / "azure_ml" / "endpoint_key.txt"

MUSTER = {
    "GitHub-Token": r"gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,}",
    "Render-Deploy-Hook": r"api\.render\.com/deploy/srv-[a-z0-9]+\?key=[A-Za-z0-9_-]+",
    "Jupyter-Token": r"[?&]token=[0-9a-f]{32,}",
    "Privater Schlüssel": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "Azure-Storage-Key": r"AccountKey=[A-Za-z0-9+/=]{40,}",
    "OpenAI/Anthropic-Key": r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}",
    "Guardrail-Key": r"\bano_[A-Za-z0-9_-]{16,}",
    "Schlüssel mit Kontext": (
        r"(?i)(?:azure_ml_key|api[_-]?key|secret|passwor[dt]|bearer)"
        r"(?:[\"']?\s*[:=]\s*[\"']?|\s+)([A-Za-z0-9+/_-]{24,}={0,2})"
    ),
}


def verlauf(root: Path = ROOT) -> str:
    """Alle Änderungen aller Commits als Text (ohne Modelle und Bilder)."""
    return subprocess.run(  # nosec B603 B607 # fester Befehl ohne Shell und ohne Benutzereingabe
        ["git", "log", "--all", "-p", "--no-color", "--text",
         "--format=@@COMMIT %h", "--", ".",
         ":(exclude)*.npz", ":(exclude)*.png", ":(exclude)*.ico"],
        cwd=root, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=True,
    ).stdout


def maskiert(wert: str) -> str:
    return f"{wert[:4]}… ({len(wert)} Zeichen)"


def suchen(text: str) -> list[tuple[str, str, str, str]]:
    """Treffer als (Commit, Datei, Art, maskierter Wert)."""
    treffer = []
    commit = datei = "?"
    for zeile in text.splitlines():
        if zeile.startswith("@@COMMIT "):
            commit = zeile.split()[1]
        elif zeile.startswith("+++ b/"):
            datei = zeile[6:]
        elif zeile.startswith(("+", "-")) and not zeile.startswith(("+++", "---")):
            for art, muster in MUSTER.items():
                for fund in re.finditer(muster, zeile):
                    wert = fund.group(1) if fund.groups() else fund.group(0)
                    treffer.append((commit, datei, art, maskiert(wert)))
    return treffer


def main() -> int:
    text = verlauf()
    treffer = suchen(text)
    print(f"Commits durchsucht: {text.count('@@COMMIT ')}")
    print(f"Treffer nach Mustern: {len(treffer)}")
    for commit, datei, art, wert in treffer:
        print(f"  {commit}  {datei}  {art}: {wert}")

    echter_key = False
    if SCHLUESSEL_DATEI.exists():
        key = SCHLUESSEL_DATEI.read_text(encoding="utf-8").strip()
        echter_key = bool(key) and key in text
        print("Echter Azure-Endpoint-Schlüssel im Verlauf:", "JA" if echter_key else "nein")
    else:
        print("Echter Azure-Endpoint-Schlüssel: nicht geprüft (keine lokale Schlüsseldatei)")
    return 1 if treffer or echter_key else 0


if __name__ == "__main__":
    sys.exit(main())
