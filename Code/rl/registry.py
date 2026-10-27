"""
Model Registry für die Vollpartien-Modelle.

Jeder Trainingslauf erzeugt eine neue Version, und keine wird je überschrieben.
Das Manifest models/registry.json hält fest, was eine Version eigentlich ist —
Episodenzahl, Elternversion, Datum und gemessene Resultate. Ohne diese Angaben
sind "v1" und "v2" nach ein paar Wochen zwei bedeutungslose Dateien.

Umgeschaltet wird auf zwei Ebenen:
  * global über den Eintrag "active" im Manifest — das ist das Rollback
  * pro Anfrage über das Feld "version" in der API — damit lassen sich zwei
    Versionen zur selben Stellung direkt vergleichen, ohne Neustart

Nur die Vollpartien-Modelle sind versioniert. Die Endspielmodelle (kqk, krk)
sind Spezialisten für eine feste Aufgabe und ändern sich nicht mehr.

Eine Version ist Netz UND Suchtiefe
-----------------------------------
Zu jeder Version gehört das Feld "search_depth": wie viele Halbzüge die Engine
mit diesem Netz vorausrechnet. Erst beides zusammen legt fest, wie eine Version
spielt. v1 bis v3 rechnen fest einen Halbzug voraus; wer sie auswählt, bekommt
exakt dieselben Züge wie vor dem Einbau der Suche (abgesichert durch
tests/test_search.py). Einträge ohne das Feld gelten als Tiefe 1.

Deshalb gibt es auch keinen freien Tiefe-Regler in der Oberfläche: Eine
Version soll einen festen, nachvollziehbaren Stand des Projekts zeigen.

Aus demselben Grund gehören die Korrekturen der Suche ab v6 zur Version
(Felder "cap" und "repetition_draw", siehe search.SearchOptions). Fehlen sie,
rechnet die Suche wie bei v1 bis v5.

Aufruf:
    python -m Code.rl.registry                  Versionen anzeigen
    python -m Code.rl.registry --activate v1    aktive Version umschalten
    python -m Code.rl.registry --register v5 --file full_v5.npz --depth 2 ...
    python -m Code.rl.registry --register v6 --file full_v5.npz --depth 2 --cap 0.9 --repetition-draw ...
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from .search import SearchOptions
from .valuenet import ValueNet

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

_cache: dict[str, ValueNet] = {}


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"active": None, "versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def save_registry(data: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def versions() -> list[dict]:
    return load_registry().get("versions", [])


def find(name: str) -> dict | None:
    return next((v for v in versions() if v["name"] == name), None)


def active_name() -> str | None:
    data = load_registry()
    active = data.get("active")
    if active and find(active):
        return active
    available = data.get("versions", [])
    return available[-1]["name"] if available else None


def set_active(name: str) -> None:
    if find(name) is None:
        raise ValueError(f"Unbekannte Version: {name}")
    data = load_registry()
    data["active"] = name
    save_registry(data)


def search_depth(name: str | None = None) -> int:
    """Vorausschau einer Version in Halbzügen; ohne Namen die der aktiven."""
    entry = find(name or active_name() or "")
    return int(entry.get("search_depth", 1)) if entry else 1


def search_options(name: str | None = None) -> SearchOptions:
    """Korrekturen der Suche, die zu einer Version gehören; ohne Namen die der aktiven."""
    entry = find(name or active_name() or "") or {}
    return SearchOptions(cap=entry.get("cap"), repetition_draw=bool(entry.get("repetition_draw", False)))


def load_model(name: str | None = None) -> ValueNet | None:
    """
    Modell einer Version laden. Ohne Namen die aktive Version.

    Geladene Netze werden behalten, damit die API nicht bei jeder Anfrage eine
    Datei von der Platte liest.
    """
    name = name or active_name()
    if name is None:
        return None
    entry = find(name)
    if entry is None:
        return None
    # Zwischengespeichert wird pro Datei, nicht pro Version: Zwei Versionen
    # können dasselbe Netz mit unterschiedlicher Suchtiefe verwenden.
    if entry["file"] in _cache:
        return _cache[entry["file"]]
    path = MODELS_DIR / entry["file"]
    if not path.exists():
        return None
    _cache[entry["file"]] = ValueNet.load(path)
    return _cache[entry["file"]]


def register(
    name: str,
    file: str,
    episodes: int,
    parent: str | None = None,
    notes: str = "",
    depth: int = 1,
    activate: bool = True,
    options: SearchOptions = SearchOptions(),
) -> dict:
    """Neue Version eintragen. Bestehende Einträge werden nicht überschrieben."""
    if find(name) is not None:
        raise ValueError(f"Version {name} gibt es bereits.")
    data = load_registry()
    entry = {
        "name": name,
        "file": file,
        "trained": date.today().isoformat(),
        "episodes": episodes,
        "parent": parent,
        "search_depth": depth,
    }
    # Nur eintragen, was eingeschaltet ist: Einträge ohne die Felder rechnen wie v1 bis v5.
    if options.cap is not None:
        entry["cap"] = options.cap
    if options.repetition_draw:
        entry["repetition_draw"] = True
    entry.update({"notes": notes, "results": {}})
    data.setdefault("versions", []).append(entry)
    if activate:
        data["active"] = name
    save_registry(data)
    return entry


def main() -> None:
    parser = argparse.ArgumentParser(description="Modellversionen anzeigen und umschalten")
    parser.add_argument("--activate", help="Diese Version als aktive setzen")
    parser.add_argument("--register", help="Name einer neuen Version, z. B. v3")
    parser.add_argument("--file", help="Dateiname in models/ für --register")
    parser.add_argument("--episodes", type=int, default=0, help="Trainierte Episoden für --register")
    parser.add_argument("--parent", default=None, help="Version, auf der aufgesetzt wurde")
    parser.add_argument("--notes", default="", help="Kurzbeschreibung des Laufs")
    parser.add_argument("--depth", type=int, default=1, help="Vorausschau in Halbzügen für --register")
    parser.add_argument("--keep-active", action="store_true",
                        help="Neue Version nur eintragen, die aktive bleibt unverändert")
    parser.add_argument("--cap", type=float, default=None,
                        help="Gesamtbewertung unter diesen Wert stauchen, damit Matt immer gewinnt (ab v6: 0.9)")
    parser.add_argument("--repetition-draw", action="store_true",
                        help="Stellungswiederholung in der Suche als Remis werten (ab v6)")
    args = parser.parse_args()

    if args.register:
        if not args.file:
            parser.error("--register braucht --file")
        register(args.register, args.file, args.episodes, args.parent, args.notes,
                 depth=args.depth, activate=not args.keep_active,
                 options=SearchOptions(cap=args.cap, repetition_draw=args.repetition_draw))
        print(f"Version {args.register} eingetragen"
              + (".\n" if args.keep_active else " und aktiviert.\n"))
    if args.activate:
        set_active(args.activate)
        print(f"Aktive Version ist jetzt {args.activate}.\n")

    active = active_name()
    print(f"{'':2} {'Version':<8} {'Datum':<12} {'Episoden':>9} {'Tiefe':>6} {'Eltern':<8} Beschreibung")
    for entry in versions():
        marker = "->" if entry["name"] == active else "  "
        print(
            f"{marker:2} {entry['name']:<8} {entry['trained']:<12} {entry['episodes']:>9} "
            f"{entry.get('search_depth', 1):>6} {str(entry['parent'] or '-'):<8} {entry['notes']}"
        )
    print("\n-> = aktive Version, Tiefe = vorausgerechnete Halbzüge")
    korrigiert = [v["name"] for v in versions() if "cap" in v or v.get("repetition_draw")]
    if korrigiert:
        print(f"Mit korrigierter Suche (Deckel, Wiederholung = Remis): {', '.join(korrigiert)}")


if __name__ == "__main__":
    main()
