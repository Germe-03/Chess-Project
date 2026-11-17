"""Training des Schach-Wertnetzes als Azure-ML-Auftrag.

Das Skript startet auf Azure ML dasselbe Training wie der lokale Lauf v7b.
Details und Hintergründe stehen in azure_ml/TRAINING_AZURE.md.
Die vier Befehle sind:
    python azure_ml/training/starte_training.py starten --lauf probe
    python azure_ml/training/starte_training.py starten --lauf haupt --vm Standard_F16s_v2
    python azure_ml/training/starte_training.py starten --lauf r2a --vm Standard_F16s_v2  (Runde 2: r2a bis r2f)
    python azure_ml/training/starte_training.py status <auftragsname>
    python azure_ml/training/starte_training.py holen <auftragsname>
    python azure_ml/training/starte_training.py umgebung
Der Probelauf prüft Anmeldung, Upload, Umgebung und Download.
Mit --trockenlauf lässt sich der Start ohne Azure testen.
Status zeigt den Zustand eines Auftrags; holen lädt seine Ausgaben herunter.
umgebung zeigt die verwendete Microsoft-Umgebung.
Beim ersten Aufruf öffnet sich für die Anmeldung ein Browserfenster.
Die Anmeldung wird lokal gespeichert, damit spätere Aufrufe einfacher sind.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

# Abo, Ressourcengruppe, Workspace und Tenant stehen in einer lokalen Datei,
# nicht im öffentlichen Repo (gitignored). Vorlage: ziel.beispiel.json.
ZIEL = Path(__file__).with_name("ziel.json")
# Keine eigene Image-Erstellung: Azure legt so keine täglich kostenpflichtige Container-Registry an.
ENVIRONMENT = "azureml://registries/azureml/environments/sklearn-1.5/labels/latest"
EXPERIMENT = "schach-training"
PROJEKT = Path(__file__).resolve().parents[2]
# Enthält nur den AuthenticationRecord, also Kontoname und Tenant, keinen Token; ist gitignored.
ANMELDUNG = Path(__file__).with_name(".anmeldung.json")
PAKETE = "numpy==2.1.3 chess==1.11.1"
KERNE = {
    "Standard_DS3_v2": 4,
    "Standard_D4as_v4": 4,
    "Standard_F8s_v2": 8,
    "Standard_D16as_v4": 16,
    "Standard_F16s_v2": 16,
}
LAEUFE: dict[str, dict[str, int | str | float | None]] = {
    "probe": {
        "episodes": 2,
        "search_depth": 2,
        "max_plies": 120,
        "snapshot_every": 0,
        "out": "outputs/full_probe.npz",
        "timeout_hours": 0.5,
    },
    "haupt": {
        "episodes": 50,
        "search_depth": 4,
        "max_plies": 400,
        "snapshot_every": 25,
        "out": "outputs/full_v7b_azure.npz",
        "timeout_hours": None,
    },
    # Runde 1 des grossen Trainings (TRAINING_AZURE.md): vier Varianten
    # gleichzeitig, jede misst ihre Zwischenstände danach selbst gegen v6.
    "r1a": {  # Lernrate kleiner
        "episodes": 100, "search_depth": 4, "max_plies": 400, "snapshot_every": 25,
        "lr": "1e-4", "out": "outputs/netz.npz", "timeout_hours": 3, "messen": True,
    },
    "r1b": {  # Training so tief wie das Spiel
        "episodes": 1000, "search_depth": 2, "max_plies": 400, "snapshot_every": 250,
        "lr": "5e-4", "out": "outputs/netz.npz", "timeout_hours": 1, "messen": True,
    },
    "r1c": {  # beides
        "episodes": 1000, "search_depth": 2, "max_plies": 400, "snapshot_every": 250,
        "lr": "1e-4", "out": "outputs/netz.npz", "timeout_hours": 1, "messen": True,
    },
    "r1d": {  # altes Rezept, doppelt so lang, zum Vergleich
        "episodes": 100, "search_depth": 4, "max_plies": 400, "snapshot_every": 25,
        "lr": "5e-4", "seed": 3, "out": "outputs/netz.npz", "timeout_hours": 3, "messen": True,
    },
}
# Runde 2 (TRAINING_AZURE.md): sechs Varianten gleichzeitig, Start beim Netz von
# v8, Grundrezept wie v8 (2 Halbzüge, Lernrate 5e-4), je 3000 Partien. Jede misst
# danach ihre sechs Zwischenstände selbst gegen v8. Notbremse 2.25 h je Auftrag,
# damit sechs Aufträge zusammen sicher unter 10 CHF bleiben.
RUNDE2: dict[str, int | str | float | None] = {
    "episodes": 3000, "search_depth": 2, "max_plies": 400, "snapshot_every": 500,
    "lr": "5e-4", "start": "full_v8.npz", "gegner": "full_v8.npz",
    "out": "outputs/netz.npz", "timeout_hours": 2.25, "messen": True,
}
LAEUFE.update({
    "r2a": {**RUNDE2, "seed": 1},  # Rezept von v8, drei Seeds zeigen die Streuung
    "r2b": {**RUNDE2, "seed": 2},
    "r2c": {**RUNDE2, "seed": 3},
    "r2d": {**RUNDE2, "seed": 1, "lr": "2.5e-4"},  # halbe Lernrate für die Feinarbeit
    "r2e": {**RUNDE2, "seed": 1, "cap": 0.9},  # Deckel 0.9 auch im Training, ganz wie im Spiel
    "r2f": {**RUNDE2, "seed": 1, "eps_start": 0.2},  # mehr Erkundung am Anfang
})
# Runde 3: Training mit Schachtheorie. Start beim Netz von v9, Rezept von v9.
# Die Hälfte der Partien beginnt in einer Stellung aus dem Eröffnungsbuch, und
# Stellungen mit höchstens 5 Steinen werden nach den Endspieltabellen gewertet.
# Gemessen wird mit Theorie auf beiden Seiten gegen v10.2 (Netz von v9 mit Buch
# und Tabellen); so zeigt sich, was das neue Netz unter gleicher Theorie bringt.
RUNDE3: dict[str, int | str | float | None] = {
    **RUNDE2, "start": "full_v9.npz", "gegner": "full_v9.npz",
    "buch_anteil": 0.5, "endspiel": True, "theorie_messen": True,
}
LAEUFE.update({
    "r3a": {**RUNDE3, "seed": 1},  # Buchstarts und Tabellen, drei Seeds
    "r3b": {**RUNDE3, "seed": 2},
    "r3c": {**RUNDE3, "seed": 3},
    "r3d": {**RUNDE3, "seed": 1, "endspiel": False},  # nur Buchstarts
    "r3e": {**RUNDE3, "seed": 1, "buch_anteil": 0.0},  # nur Tabellen
    "r3f": {**RUNDE3, "seed": 1, "lr": "2.5e-4"},  # beides, halbe Lernrate
})
# Endspieltabellen (Syzygy, bis 5 Steine, 984 MB) als Datenbestand im Workspace;
# einmal hochladen mit: starte_training.py tabellen-hochladen
TABELLEN_ASSET = "syzygy-5"
TABELLEN_VERSION = "1"
# Messung wie bei v6: 2 Halbzüge, Deckel 0.9, Wiederholung = Remis, auf beiden Seiten.
MESSPARTIEN = 400


def trainingsbefehl(lauf: str, workers: int, seed: int = 0) -> str:
    profil = LAEUFE[lauf]
    teile = [
        "pip install --quiet",
        PAKETE,
        "&&",
        "python -m Code.rl.train",
        "--task full",
        f"--resume models/{profil.get('start', 'full_v5.npz')}",
        f"--search-depth {profil['search_depth']}",
        "--repetition-draw",
        f"--workers {workers}",
        f"--episodes {profil['episodes']}",
        f"--max-plies {profil['max_plies']}",
        f"--eps-start {profil.get('eps_start', 0.1)}",
        "--eps-end 0.05",
        f"--lr {profil.get('lr', '5e-4')}",
        "--eval-every 0",
        f"--snapshot-every {profil['snapshot_every']}",
        "--log-every 1",
        "--run-dir outputs/run",
        f"--out {profil['out']}",
    ]
    # Seed 0 ist der Standard von train.py; so bleibt der Befehl gleich wie bei v7b.
    seed = seed or int(profil.get("seed", 0))
    if seed:
        teile.append(f"--seed {seed}")
    if profil.get("cap") is not None:
        teile.append(f"--cap {profil['cap']}")
    if profil.get("buch_anteil"):
        teile.append(f"--buch-anteil {profil['buch_anteil']}")
    if profil.get("endspiel"):
        teile.append("--endspieltabellen")
    befehl = " ".join(teile)
    if braucht_tabellen(lauf):
        # Die Tabellen kommen als Datenbestand; train.py und evaluate.py suchen sie unter theorie/syzygy.
        befehl = "mkdir -p theorie && ln -sfn ${{inputs.syzygy}} theorie/syzygy && " + befehl
    if profil.get("messen"):
        befehl += " && " + messbefehl(workers, str(profil.get("gegner", "full_v5.npz")),
                                       bool(profil.get("theorie_messen")))
    return befehl


def braucht_tabellen(lauf: str) -> bool:
    profil = LAEUFE[lauf]
    return bool(profil.get("endspiel") or profil.get("theorie_messen"))


def braucht_buch(lauf: str) -> bool:
    profil = LAEUFE[lauf]
    return bool(profil.get("buch_anteil") or profil.get("theorie_messen"))


def messbefehl(workers: int, gegner: str = "full_v5.npz", theorie: bool = False) -> str:
    """Jeden Zwischenstand gegen das Gegnernetz messen (Runde 1: v6, Runde 2: v8, Runde 3: v10.2).

    Mit theorie spielen beide Seiten mit Eröffnungsbuch und Endspieltabellen.
    Die Ergebnisse landen in outputs/messung.txt.
    """
    schalter = "--eroeffnung --endspiel --opponent-eroeffnung --opponent-endspiel " if theorie else ""
    duell = (
        "python -m Code.rl.evaluate --task full --model $f --depth 2 --cap 0.9 --repetition-draw "
        f"--opponent models/{gegner} --opponent-depth 2 --opponent-cap 0.9 "
        f"--opponent-repetition-draw {schalter}--games {MESSPARTIEN} --max-plies 400 --workers {workers}"
    )
    return f"(for f in outputs/run/snapshots/*.npz; do {duell}; done) 2>&1 | tee outputs/messung.txt"


def hoechstlaufzeit(lauf: str, kerne: int) -> int:
    stunden = LAEUFE[lauf]["timeout_hours"]
    if stunden is not None:
        return int(stunden * 3600)
    # Reserve für das Laden der Umgebung und die langen Partien am Schluss.
    return 3 * 3600 if kerne >= 16 else 6 * 3600


def paket_vorbereiten(lauf: str) -> Path:
    profil = LAEUFE[lauf]
    namen = sorted({str(profil.get("start", "full_v5.npz")), str(profil.get("gegner", "full_v5.npz"))})
    for name in namen:
        if not (PROJEKT / "models" / name).is_file():
            raise SystemExit(f"Modell nicht gefunden: {PROJEKT / 'models' / name}")

    ziel = Path(tempfile.mkdtemp(prefix="schach-azure-"))
    shutil.copytree(
        PROJEKT / "Code",
        ziel / "Code",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    modelle = ziel / "models"
    modelle.mkdir()
    for name in namen:
        shutil.copy2(PROJEKT / "models" / name, modelle / name)
    if braucht_buch(lauf):
        # Das Buch ist klein (rund 250 KB) und reist im Code-Paket mit.
        buch = PROJEKT / "theorie" / "buch.bin"
        if not buch.is_file():
            raise SystemExit(f"Eröffnungsbuch fehlt: {buch} (python -m Code.rl.buch ...)")
        (ziel / "theorie").mkdir()
        shutil.copy2(buch, ziel / "theorie" / "buch.bin")

    groesse = sum(datei.stat().st_size for datei in ziel.rglob("*") if datei.is_file())
    print(f"Paketgrösse: {groesse / (1024 * 1024):.1f} MB")
    return ziel


def ziel() -> dict[str, str]:
    if not ZIEL.is_file():
        raise SystemExit(
            f"{ZIEL.name} fehlt. Vorlage ziel.beispiel.json kopieren und "
            "subscription_id, resource_group, workspace und tenant_id eintragen."
        )
    return json.loads(ZIEL.read_text(encoding="utf-8"))


def anmelden():
    from azure.identity import (
        AuthenticationRecord,
        InteractiveBrowserCredential,
        TokenCachePersistenceOptions,
    )

    cache = TokenCachePersistenceOptions(name="schach-azure-ml")
    if ANMELDUNG.exists():
        record = AuthenticationRecord.deserialize(ANMELDUNG.read_text(encoding="utf-8"))
        return InteractiveBrowserCredential(
            tenant_id=ziel()["tenant_id"],
            cache_persistence_options=cache,
            authentication_record=record,
        )

    credential = InteractiveBrowserCredential(
        tenant_id=ziel()["tenant_id"],
        cache_persistence_options=cache,
    )
    record = credential.authenticate(scopes=["https://management.azure.com/.default"])
    ANMELDUNG.write_text(record.serialize(), encoding="utf-8")
    print("Anmeldung gespeichert.")
    return credential


def ml_client(credential):
    from azure.ai.ml import MLClient

    z = ziel()
    return MLClient(credential, z["subscription_id"], z["resource_group"], z["workspace"])


def cmd_starten(args: argparse.Namespace) -> None:
    kerne = args.kerne or KERNE.get(args.vm)
    if kerne is None:
        raise SystemExit("Unbekannte VM-Grösse, bitte --kerne angeben.")
    workers = args.workers or kerne
    befehl = trainingsbefehl(args.lauf, workers, args.seed)
    timeout = hoechstlaufzeit(args.lauf, kerne)
    # Wirksamer Seed: aus --seed oder aus dem Profil, wie in trainingsbefehl().
    args.seed = args.seed or int(LAEUFE[args.lauf].get("seed", 0))

    print(
        f"Lauf: {args.lauf}\n"
        f"VM: {args.vm}\n"
        f"Kerne: {kerne}\n"
        f"Prozesse: {workers}\n"
        f"Seed: {args.seed}\n"
        f"Höchstlaufzeit: {timeout / 3600:.1f} Stunden\n"
        f"Umgebung: {ENVIRONMENT}\n"
        f"Befehl: {befehl}"
    )
    if args.trockenlauf:
        print("Trockenlauf: nichts an Azure geschickt.")
        return

    code_dir = paket_vorbereiten(args.lauf)
    try:
        from azure.ai.ml import Input, command
        from azure.ai.ml.constants import AssetTypes, InputOutputModes

        eingaben = {}
        if braucht_tabellen(args.lauf):
            eingaben["syzygy"] = Input(
                type=AssetTypes.URI_FOLDER,
                path=f"azureml:{TABELLEN_ASSET}:{TABELLEN_VERSION}",
                mode=InputOutputModes.RO_MOUNT,
            )
        job = command(
            code=str(code_dir),
            command=befehl,
            inputs=eingaben or None,
            environment=ENVIRONMENT,
            environment_variables={
                "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "PYTHONUNBUFFERED": "1",
            },
            instance_type=args.vm,
            instance_count=1,
            timeout=timeout,
            job_tier="Standard",
            experiment_name=EXPERIMENT,
            display_name=f"{args.lauf}-seed{args.seed}-{args.vm}-{time.strftime('%Y%m%d-%H%M')}",
            tags={"lauf": args.lauf, "vm": args.vm, "prozesse": str(workers), "seed": str(args.seed)},
        )
        erstellt = ml_client(anmelden()).jobs.create_or_update(job)
        print(f"Auftrag: {erstellt.name}")
        print(f"Status: {erstellt.status}")
        print(f"Studio: {erstellt.studio_url}")
    finally:
        shutil.rmtree(code_dir, ignore_errors=True)


def cmd_status(args: argparse.Namespace) -> None:
    job = ml_client(anmelden()).jobs.get(args.name)
    print(f"Auftrag: {job.name}")
    print(f"Anzeigename: {job.display_name}")
    print(f"Status: {job.status}")
    print(f"Studio: {job.studio_url}")
    erstellt = getattr(getattr(job, "creation_context", None), "created_at", None)
    if erstellt is not None:
        print(f"Erstellt am: {erstellt}")


def cmd_holen(args: argparse.Namespace) -> None:
    ziel = PROJEKT / "runs" / "azure" / args.name
    ml_client(anmelden()).jobs.download(args.name, download_path=str(ziel), all=True)
    print("Heruntergeladene Dateien:")
    for datei in sorted(ziel.rglob("*")):
        if datei.is_file() and datei.suffix in {".npz", ".csv", ".json", ".txt"}:
            print(datei.relative_to(ziel))


def cmd_tabellen_hochladen(args: argparse.Namespace) -> None:
    """Endspieltabellen einmal als Datenbestand in den Workspace laden (für Runde 3)."""
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data

    ordner = PROJEKT / "theorie" / "syzygy"
    dateien = sorted(ordner.glob("*.rtb[wz]"))
    if not dateien:
        raise SystemExit(f"Keine Tabellen in {ordner}")
    groesse = sum(d.stat().st_size for d in dateien) / 1e6
    print(f"{len(dateien)} Dateien, {groesse:.0f} MB")
    if args.trockenlauf:
        print("Trockenlauf: nichts hochgeladen.")
        return
    daten = Data(
        name=TABELLEN_ASSET,
        version=TABELLEN_VERSION,
        type=AssetTypes.URI_FOLDER,
        path=str(ordner),
        description="Syzygy-Endspieltabellen bis 5 Steine (WDL und DTZ) von tablebase.lichess.ovh",
    )
    erstellt = ml_client(anmelden()).data.create_or_update(daten)
    print(f"Datenbestand: {erstellt.name}:{erstellt.version}")


def cmd_umgebung(args: argparse.Namespace) -> None:
    from azure.ai.ml import MLClient

    credential = anmelden()
    reg = MLClient(credential, registry_name="azureml")
    name = ENVIRONMENT.split("/environments/", 1)[1].split("/", 1)[0]
    env = reg.environments.get(name=name, label="latest")
    print(f"Name: {env.name}")
    print(f"Version: {env.version}")
    print(f"Image: {env.image}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Schach-Wertnetz auf Azure ML trainieren.")
    unter = parser.add_subparsers(dest="befehl", required=True)

    starten = unter.add_parser("starten", help="Trainingsauftrag starten")
    starten.add_argument("--lauf", choices=list(LAEUFE), required=True)
    starten.add_argument("--vm", default="Standard_DS3_v2")
    starten.add_argument("--kerne", type=int, default=None)
    starten.add_argument("--workers", type=int, default=None)
    starten.add_argument("--seed", type=int, default=0, help="Zufalls-Startwert für train.py")
    starten.add_argument("--trockenlauf", action="store_true")
    starten.set_defaults(handler=cmd_starten)

    status = unter.add_parser("status", help="Auftragsstatus anzeigen")
    status.add_argument("name")
    status.set_defaults(handler=cmd_status)

    holen = unter.add_parser("holen", help="Auftragsausgaben herunterladen")
    holen.add_argument("name")
    holen.set_defaults(handler=cmd_holen)

    umgebung = unter.add_parser("umgebung", help="Azure-ML-Umgebung anzeigen")
    umgebung.set_defaults(handler=cmd_umgebung)

    tabellen = unter.add_parser("tabellen-hochladen", help="Endspieltabellen als Datenbestand hochladen")
    tabellen.add_argument("--trockenlauf", action="store_true")
    tabellen.set_defaults(handler=cmd_tabellen_hochladen)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
