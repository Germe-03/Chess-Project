"""
Guardrail-Regeln lokal prüfen (Projektaufgabe 3.1, OWASP LLM02).

Die Regeln in security/guardrail/regeln.json laufen produktiv im Anonymizer
des Dozenten (guardrail.romanbrun.ch) vor dem LLM. Dieses Modul bildet nach,
was der Dienst damit macht, damit die Regeln versioniert und getestet sind,
bevor sie in die Weboberfläche kommen:

- gleiche Prüfung beim Speichern (Name, Länge, Regex gültig, kein Leer-Treffer),
- gleiche Bibliothek (regex) und gleicher Timeout von 0.2 s,
- block: die Anfrage geht gar nicht ans LLM; mask: Platzhalter [NAME_1].

    python -m ops.guardrail      Beispiele durchspielen und Ergebnis zeigen

Alle Schlüssel in den Beispielen sind erfunden und werden zur Laufzeit
zusammengesetzt, damit die Schlüssel-Suche (detect-secrets) nicht anschlägt.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import regex

ROOT = Path(__file__).resolve().parent.parent
REGELN = ROOT / "security" / "guardrail" / "regeln.json"

# Grenzen wie im Dienst (app/guardrail.py im Repo rbocre/anonymizer)
AKTIONEN = ("mask", "redact", "block")
MAX_REGELN = 20
MAX_MUSTER_LAENGE = 300
TIMEOUT = 0.2
NAME_RE = re.compile(r"^[A-Z][A-Z_]{1,29}$")
EINGEBAUT = {"PERSON", "LOCATION", "ADDRESS", "DATE_OF_BIRTH", "CREDENTIAL", "EMAIL_ADDRESS",
             "PHONE_NUMBER", "IBAN_CODE", "CREDIT_CARD", "IP_ADDRESS", "AHV_NUMBER"}


def laden(pfad: Path = REGELN) -> dict:
    return json.loads(pfad.read_text(encoding="utf-8"))


def pruefen(konfig: dict) -> list[str]:
    """Dieselben Prüfungen wie beim Speichern im Dienst. Leere Liste = in Ordnung."""
    fehler = []
    regeln = konfig.get("rules", [])
    if len(regeln) > MAX_REGELN:
        fehler.append(f"Mehr als {MAX_REGELN} Regeln")
    namen = set()
    for r in regeln:
        name = r.get("name", "")
        if not NAME_RE.match(name):
            fehler.append(f"{name}: Name ungültig (nur Grossbuchstaben und _, 2–30 Zeichen)")
        if name in EINGEBAUT or name in namen:
            fehler.append(f"{name}: Name eingebaut oder doppelt")
        namen.add(name)
        muster = r.get("pattern", "")
        if not muster or len(muster) > MAX_MUSTER_LAENGE:
            fehler.append(f"{name}: Muster fehlt oder länger als {MAX_MUSTER_LAENGE} Zeichen")
            continue
        try:
            kompiliert = kompilieren(r)
        except regex.error as e:
            fehler.append(f"{name}: ungültiger Regex ({e})")
            continue
        if kompiliert.search("") is not None:
            fehler.append(f"{name}: Muster passt auf leeren Text")
        if r.get("action") not in AKTIONEN:
            fehler.append(f"{name}: Aktion muss mask, redact oder block sein")
    return fehler


def kompilieren(regel: dict) -> regex.Pattern:
    return regex.compile(regel["pattern"], regex.IGNORECASE if regel.get("ignore_case") else 0)


def treffer(konfig: dict, text: str) -> dict[str, list[str]]:
    """Welche Regel trifft was? {Regelname: [gefundene Stellen]}."""
    ergebnis: dict[str, list[str]] = {}
    for r in konfig["rules"]:
        if not r.get("enabled", True):
            continue
        funde = [m.group(0) for m in kompilieren(r).finditer(text, timeout=TIMEOUT)]
        if funde:
            ergebnis[r["name"]] = funde
    return ergebnis


def anwenden(konfig: dict, text: str) -> tuple[str, str]:
    """
    Was beim LLM ankommt: ("BLOCK", Regelname) oder ("OK", Text mit Platzhaltern).
    Bei Überlappung gewinnt wie im Dienst der längere Treffer.
    """
    stellen = []
    for r in konfig["rules"]:
        if not r.get("enabled", True):
            continue
        for m in kompilieren(r).finditer(text, timeout=TIMEOUT):
            if r["action"] == "block":
                return "BLOCK", r["name"]
            stellen.append((m.start(), m.end(), r))
    stellen.sort(key=lambda s: (-(s[1] - s[0]), s[0]))
    gewaehlt: list[tuple[int, int, dict]] = []
    for s in stellen:
        if all(s[1] <= g[0] or s[0] >= g[1] for g in gewaehlt):
            gewaehlt.append(s)
    nummern: dict[tuple[str, str], int] = {}
    ausgabe = text
    for start, ende, r in sorted(gewaehlt, key=lambda s: s[0], reverse=True):
        if r["action"] == "redact":
            ersatz = r.get("replacement") or "[ENTFERNT]"
        else:
            schluessel = (r["name"], text[start:ende])
            if schluessel not in nummern:
                nummern[schluessel] = sum(1 for n, _ in nummern if n == r["name"]) + 1
            ersatz = f"[{r['name']}_{nummern[schluessel]}]"
        ausgabe = ausgabe[:start] + ersatz + ausgabe[ende:]
    return "OK", ausgabe


# --------------------------------------------------------------------------- #
# Beispiele: erfundene Schlüssel, zur Laufzeit zusammengesetzt
# --------------------------------------------------------------------------- #
FAKE_AZURE_KEY = ("Q7wErT9yUi" * 9)[:84]          # wie der echte: 84 Zeichen, gemischt
FAKE_GITHUB = "ghp" + "_" + "Ab3De5Gh7Jk9" * 3
FAKE_GITHUB_PAT = "github" + "_pat_" + "11ABCDEFG0_" + "x9Y8z7W6v5" * 6
FAKE_RENDER = "https://api.render.com/deploy/srv-" + "d4abc123xyz" + "?key=" + "Zx9_" * 3
FAKE_JUPYTER = "http://localhost:8888/tree?token=" + "0f3a" * 10
FAKE_ENDPOINT = "https://beispiel-lage.switzerlandnorth.inference.ml.azure.com/score"
FAKE_PRIVAT = "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIEow...\n-----END RSA PRIVATE KEY-----"
FAKE_KLEIN = "k3y" + "abcdef1234" * 3                 # nur Kleinbuchstaben und Ziffern  # pragma: allowlist secret

# (Art, Text, Regeln, die treffen müssen; leer = keine)
BEISPIELE: list[tuple[str, str, set[str]]] = [
    ("treffer", f'$env:AZURE_ML_KEY = "{FAKE_AZURE_KEY}"', {"AZURE_ML_KEY", "ZUFALLSSCHLUESSEL"}),
    ("treffer", f"curl -H 'Authorization: Bearer {FAKE_AZURE_KEY}' ...", {"AZURE_ML_KEY", "ZUFALLSSCHLUESSEL"}),
    ("treffer", f"Mein Primärschlüssel ist {FAKE_AZURE_KEY}, warum kommt 401?", {"ZUFALLSSCHLUESSEL"}),
    ("treffer", f"api_key: {FAKE_KLEIN}", {"AZURE_ML_KEY"}),
    ("treffer", f"git push https://{FAKE_GITHUB}@github.com/x/chess.git", {"GITHUB_TOKEN", "ZUFALLSSCHLUESSEL"}),
    ("treffer", f"GH_TOKEN={FAKE_GITHUB_PAT}", {"GITHUB_TOKEN", "ZUFALLSSCHLUESSEL"}),
    ("treffer", f"Deploy-Hook: {FAKE_RENDER}", {"RENDER_DEPLOY_HOOK"}),
    ("treffer", f"{FAKE_PRIVAT}", {"PRIVATER_SCHLUESSEL"}),
    ("treffer", "Mein Passwort: 1234", {"PASSWORT"}),
    ("treffer", "mein passwort=1234", {"PASSWORT"}),
    ("treffer", "pw: Sommer2026", {"PASSWORT"}),
    ("treffer", "Mein Passwort ist 1234", {"PASSWORT"}),
    ("treffer", f"Jupyter läuft unter {FAKE_JUPYTER}", {"JUPYTER_TOKEN"}),
    ("treffer", f"POST an {FAKE_ENDPOINT} gibt 424", {"AZURE_ENDPOINT"}),
    ("kein Treffer", "Warum ist 1.e4 e5 2.Sf3 besser als 2.Lc4?", set()),
    ("kein Treffer", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", set()),
    ("kein Treffer", "SHA-256: " + "5152784207c8a1f0" * 4, set()),  # pragma: allowlist secret
    ("kein Treffer", "Commit " + "387d344a" * 5, set()),
    ("kein Treffer", 'key = os.environ.get("AZURE_ML_KEY", "")', set()),
    ("kein Treffer", "api_key_environment_variable_name", set()),
    ("kein Treffer", "Bearer-Token werden im Header mitgeschickt.", set()),
    ("kein Treffer", "Das Passwort-Feld bleibt leer.", set()),
    ("kein Treffer", "Das Passwort ist zu kurz, wie prüfe ich die Länge?", set()),
    ("kein Treffer", "hash = bcrypt.hashpw(pw_bytes, salt)", set()),
    ("kein Treffer", "Modell schachlagerunde19:2, Deployment runde2", set()),
    ("Umgehung", "P a s s w o r t: 1234", set()),
    ("Umgehung", "Mein Passwort ist geheim", set()),
    ("Umgehung", " ".join(FAKE_AZURE_KEY[i:i + 28] for i in range(0, 84, 28)), set()),
]


def main() -> int:
    konfig = laden()
    fehler = pruefen(konfig)
    print(f"Regeln: {len(konfig['rules'])}, Prüfung wie im Dienst: {'OK' if not fehler else fehler}")
    falsch = 0
    for art, text, erwartet in BEISPIELE:
        getroffen = set(treffer(konfig, text))
        status, ausgabe = anwenden(konfig, text)
        ok = getroffen == erwartet
        falsch += not ok
        kurz = text.replace("\n", " ")
        kurz = kurz if len(kurz) <= 70 else kurz[:67] + "..."
        ans_llm = f"BLOCK ({ausgabe})" if status == "BLOCK" else ausgabe.replace("\n", " ")[:70]
        print(f"{'ok ' if ok else 'XX '} {art:<12} {kurz}\n      -> LLM sieht: {ans_llm}")
    print(f"\n{len(BEISPIELE) - falsch} von {len(BEISPIELE)} Beispielen wie erwartet")
    return 1 if fehler or falsch else 0


if __name__ == "__main__":
    sys.exit(main())
