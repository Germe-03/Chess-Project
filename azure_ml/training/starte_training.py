"""Training des Schach-Wertnetzes als Azure-ML-Auftrag.

Das Skript startet auf Azure ML dasselbe Training wie der lokale Lauf v7b.
Details und Hintergründe stehen in azure_ml/TRAINING_AZURE.md.
Die vier Befehle sind:
    python azure_ml/training/starte_training.py starten --lauf probe
    python azure_ml/training/starte_training.py starten --lauf haupt --vm Standard_F16s_v2
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
}


def trainingsbefehl(lauf: str, workers: int, seed: int = 0) -> str:
    profil = LAEUFE[lauf]
    teile = [
        "pip install --quiet",
        PAKETE,
        "&&",
        "python -m Code.rl.train",
        "--task full",
        "--resume models/full_v5.npz",
        f"--search-depth {profil['search_depth']}",
        "--repetition-draw",
        f"--workers {workers}",
        f"--episodes {profil['episodes']}",
        f"--max-plies {profil['max_plies']}",
        "--eps-start 0.1",
        "--eps-end 0.05",
        "--lr 5e-4",
        "--eval-every 0",
        f"--snapshot-every {profil['snapshot_every']}",
        "--log-every 1",
        "--run-dir outputs/run",
        f"--out {profil['out']}",
    ]
    # Seed 0 ist der Standard von train.py; so bleibt der Befehl gleich wie bei v7b.
    if seed:
        teile.append(f"--seed {seed}")
    return " ".join(teile)


def hoechstlaufzeit(lauf: str, kerne: int) -> int:
    stunden = LAEUFE[lauf]["timeout_hours"]
    if stunden is not None:
        return int(stunden * 3600)
    # Reserve für das Laden der Umgebung und die langen Partien am Schluss.
    return 3 * 3600 if kerne >= 16 else 6 * 3600


def paket_vorbereiten() -> Path:
    modell = PROJEKT / "models" / "full_v5.npz"
    if not modell.is_file():
        raise SystemExit(f"Startmodell nicht gefunden: {modell}")

    ziel = Path(tempfile.mkdtemp(prefix="schach-azure-"))
    shutil.copytree(
        PROJEKT / "Code",
        ziel / "Code",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    modelle = ziel / "models"
    modelle.mkdir()
    shutil.copy2(modell, modelle / "full_v5.npz")

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

    code_dir = paket_vorbereiten()
    try:
        from azure.ai.ml import command

        job = command(
            code=str(code_dir),
            command=befehl,
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
        if datei.is_file() and datei.suffix in {".npz", ".csv", ".json"}:
            print(datei.relative_to(ziel))


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
    starten.add_argument("--lauf", choices=["probe", "haupt"], required=True)
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

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
