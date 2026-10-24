# Projektaufgabe 2 — Azure ML: Schritt für Schritt

**Auftrag** (Einführungsfolien, Semesterprogramm, Vorlage Projektarbeit): Mit
**Azure ML Studio (AutoML)** ein eigenes ML-Modell trainieren, deployen,
**verbessern und erneut deployen**.

**Abgabe:**
- **Kapitel 3** der Arbeit, max. 1 A4-Seite.
- **Screenvideo**: der deployte Endpoint und ein Beispielaufruf mit Ergebnis in einem Notebook.
- **Kapitel 3.1 Kosten**: verursachte Kosten und Kostentreiber (¼ A4), Sparpotenzial (¼ A4), erwartete Kosten on-prem (½ A4).

**Unser Modell:** Aus einer Schachstellung vorhersagen, wer besser steht:
`weiss_besser`, `ausgeglichen` oder `schwarz_besser`. Die Daten stammen aus
dem eigenen Prüfstand: 100 Partien, jede Stellung von Stockfish bewertet.

```
FEN ──► Merkmale (merkmale.py) ──► Azure-ML-Endpoint ──► "weiss_besser"
```

| Runde | Merkmale | lokal gemessen (Vorabtest) |
|---|---|---|
| 1 | nur Material (11 Spalten) | 78–85 % richtig |
| 2 | + Bilanz nach bestem Schlag, Mobilität, Schach, Königsdruck, Freibauern (24 Spalten) | 88–89 % richtig |

Die Zahlen stammen aus einem lokalen Vorabtest mit scikit-learn: trainiert auf
80 Partien, getestet auf 20 anderen. Sie zeigen nur, dass die Verbesserung echt
ist. In der Arbeit zählen die Werte aus Azure.

**Dateien in diesem Ordner**

| Datei | Zweck |
|---|---|
| `daten/runde1/` | Ordner-Upload für AutoML-Auftrag 1: CSV (6 200 Stellungen) + `MLTable` |
| `daten/runde2/` | Ordner-Upload für AutoML-Auftrag 2: dieselben Stellungen, mehr Merkmale + `MLTable` |
| `daten/test_stellungen.csv` | 1 373 Stellungen aus 20 anderen Partien, nur für das Notebook |
| `merkmale.py` | FEN → Merkmale, gleich für Training und Aufruf |
| `datensatz.py` | erzeugt die CSV-Dateien neu: `python -m azure_ml.datensatz` |
| `endpoint_aufruf.ipynb` | Notebook für das Screenvideo |

**Kosten:** Laut Folie unter **CHF 40 für zwei Tage**. Die Kosten überwachst
du selbst. Der grösste Kostentreiber ist der Endpoint, weil er **rund um die Uhr**
läuft, auch ohne Anfragen. → Am Ende unbedingt [aufräumen](#10-aufräumen-wichtig).

---

## 1. Zugang einrichten (einmalig)

1. **Cisco VPN Client** installieren (Download und Anleitung im FHNW-Intranet,
   Link auf Folie 11 der Einführung) und mit dem FHNW-Konto verbinden.
2. **MS Authenticator** auf dem Handy einrichten, falls noch nicht geschehen.
3. Prüfen, ob die E-Mail angekommen ist, dass du einer Ressourcengruppe
   hinzugefügt wurdest (Folie «Prepare (I)»). Ohne diese E-Mail fehlt die
   Berechtigung → beim Dozenten melden.

## 2. Vor jeder Sitzung: «pimmen»

Die Rechte sind nur zeitlich begrenzt aktiv und müssen über PIM eingeschaltet werden.

1. VPN verbinden.
2. https://aka.ms/PIM öffnen → **Meine Rollen** → **Azure-Ressourcen**.
3. Unter *Berechtigte Zuweisungen* **beide Rollen aktivieren**:
   - **Contributor** auf deiner persönlichen Ressourcengruppe
   - **Reader** auf `DefaultResourceGroup-switzerlandnorth`
4. Auf die E-Mail warten, dass die Rollen aktiv sind.
5. **Abmelden und neu anmelden**, erst dann gelten die Rechte.
6. https://ml.azure.com öffnen.

Eine Fehlermeldung über fehlende Berechtigung heisst fast immer: neu pimmen,
abmelden, anmelden.

## 3. Arbeitsbereich (Workspace) erstellen

ml.azure.com → **Arbeitsbereich erstellen** (*Create workspace*)

| Feld | Wert |
|---|---|
| Name | z. B. `ws-schach-dgerm` |
| Abonnement | das FHNW-Abonnement |
| Ressourcengruppe | **deine persönliche** |
| Region | **Switzerland North** |

Das dauert einige Minuten. Azure legt dabei auch Speicherkonto, Key Vault und
Application Insights an. Diese kosten fast nichts, zählen aber zu den Kosten in
Kapitel 3.1.

## 4. Daten hochladen (Runde 1) ✅ erledigt am 23.10.2026

Im Workspace links **Daten** (*Data*) → **Erstellen**:

| Schritt | Wert |
|---|---|
| Name | `schach-stellungen` |
| Typ | **Tabelle (mltable)** |
| Quelle | **Aus lokalen Dateien** → Datenspeicher `workspaceblobstore` |
| MLTable-Auswahl | **Ordner hochladen** → `azure_ml/daten/runde1` (enthält die CSV und die Datei `MLTable`) |

Eine Datenressource vom Typ «Tabelle» lässt sich nur als **Ordner mit einer
`MLTable`-Datei** hochladen. Die MLTable beschreibt, wie die CSV zu lesen ist.

**Was dabei schiefging und warum es lehrreich ist (für die Arbeit):**
- **Version 1** zeigte in der Vorschau in `weiss_am_zug`, `w_damen` und
  `s_damen` nur **«Fehler»**. Azure hatte die Typen aus den ersten Zeilen
  erraten und die Spalten für Wahrheitswerte gehalten, weil dort meist nur 0 und
  1 stehen. Zwei Damen kommen nur 44-mal vor.
- **Version 2** liest ohne Typerkennung (`infer_column_types: false`) und
  wandelt jedes Merkmal ausdrücklich in eine Ganzzahl um. Die Vorschau ist korrekt.
- Das ist Datenversionierung in der Praxis: Der Fehler bleibt als Version 1
  nachvollziehbar, das Training verwendet Version 2.
- Die zuerst angelegte Ressource `schach-lage` hatte dasselbe Problem und wird
  nicht verwendet (archivieren).

Ergebnis: Datenressource `schach-stellungen`, **Version 2** = Runde 1.

## 5. AutoML-Auftrag Runde 1 ✅ gestartet am 23.10.2026, 11:33

Links **Automatisiertes ML** → **Neuer Auftrag für automatisiertes ML**

| Seite | Einstellung |
|---|---|
| Grundeinstellungen | Auftragsname `schach-lage-runde1`, Experiment **neu**: `schach-lage` |
| Aufgabentyp & Daten | **Klassifizierung**, Daten `schach-stellungen` (Version 2, über «Daten anzeigen» prüfen: keine «Fehler») |
| Aufgabeneinstellungen | Zielspalte **`lage`**, primäre Metrik **AUC gewichtet** (Standard, robust bei ungleich grossen Klassen) |
| Zusätzliche Konfiguration | **«Bestes Modell erklären»** aktivieren (liefert die Merkmalswichtigkeit für die Arbeit) |
| Grenzwerte | max. Testversionen **20**, gleichzeitig **2**, max. Knoten **2**, Timeout Experiment **60 min**, Timeout Iteration **15 min**, vorzeitige Beendigung **an** |
| Validierung | **Automatisch** |
| Compute | **Serverlos**, CPU, dediziert, **Standard_DS3_v2** (4 Kerne, 14 GB, 0.30 $/h), **2 Instanzen** |

Serverlos heisst: Azure stellt die VMs nur für die Dauer des Auftrags bereit,
es bleibt nichts laufen. Kosten höchstens 2 × 0.30 $ × 1 h.
Absenden. Der Auftrag dauert etwa 30–60 Minuten.

**Wichtig:** Der Assistent merkt sich die Datenversion. Wurde während des
Ausfüllens eine neue Version hochgeladen, den Assistenten abbrechen und neu
starten, sonst trainiert er auf der alten Version.

**Danach ansehen und festhalten** (Screenshots für die Arbeit):
- Registerkarte **Modelle**: bestes Modell (Algorithmus, z. B. VotingEnsemble oder LightGBM)
- **Metriken**: Genauigkeit (*accuracy*), ausgewogene Genauigkeit, AUC gewichtet, Konfusionsmatrix
- **Erklärungen**: wichtigste Merkmale. In Runde 1 vermutlich Damen und Türme.

### Ergebnis Runde 1 (Auftrag `schach-lage-runde1`, 18 min, 20 Modelle)

Bestes Modell: **VotingEnsemble** aus LightGBM, 4 × XGBoost, RandomForest und
LogisticRegression. Die Werte stammen aus der Kreuzvalidierung in Azure:

| AUC gewichtet | Genauigkeit | ausgewogene Genauigkeit | F1 gewichtet |
|---|---|---|---|
| 0.959 | 86.9 % | 81.4 % | 0.868 |

Konfusionsmatrix (Zeilen = richtig, Spalten = vorhergesagt; die Zeilensummen
ergeben die Klassengrössen 927 / 2 412 / 2 861):

| | ausgeglichen | schwarz_besser | weiss_besser | Trefferquote |
|---|---|---|---|---|
| **ausgeglichen** | 573 | 159 | 195 | 61.8 % |
| **schwarz_besser** | 148 | 2187 | 77 | 90.7 % |
| **weiss_besser** | 167 | 67 | 2627 | 91.8 % |

Die Kreuzvalidierung mischt Stellungen derselben Partie in Training und
Validierung, deshalb liegt die Genauigkeit über dem lokalen Vorabtest (78–85 %).
Ehrlicher ist die Auswertung im Notebook: Teststellungen aus 20 Partien, die
nie im Training waren.

## 6. Runde 1 deployen ✅ erledigt am 23.10.2026, 11:55–12:06

Bestes Modell (VotingEnsemble) → **Bereitstellen** → **Echtzeitendpunkt**

| Feld | Wert |
|---|---|
| Endpunkt | **neu**: `schach-lage-dgerm` |
| Bereitstellungsname | **`runde1`** |
| VM | **Standard_D2a_v4** (2 Kerne, 8 GB RAM, 50 GB, 0.14 $/h) |
| Instanzanzahl | **1** (vorgeschlagen waren 3 × DS3_v2, also rund 0.90 $/h) |
| Authentifizierung | schlüsselbasiert |

**Warum nicht die kleinste VM:** Standard_DS1_v2 (0.07 $/h) und F2s_v2
(0.11 $/h) wären günstiger. Azure warnt aber selbst, dass sie für grössere
Modelle zu wenig Arbeitsspeicher oder Speicherplatz haben und der Container
abbricht. Das AutoML-Ensemble braucht ein mehrere GB grosses Software-Paket.
D2a_v4 ist die günstigste Grösse ohne diese Warnung.

Achtung: Beim Absenden zeigte das gesperrte Feld kurz wieder DS3_v2 an. Die
Details des Endpoints bestätigen aber **SKU Standard_D2a_v4**. Im Zweifel dort
nachsehen.

- REST-Endpunkt: `https://schach-lage-dgerm.switzerlandnorth.inference.ml.azure.com/score`
- Format: **`input_data`** (MLflow) → im Notebook `FORMAT = "mlflow"`
- Registriertes Modell: `schachlagerunde19:1`
- Test im Studio mit den drei Beispielstellungen: `["ausgeglichen", "ausgeglichen", "schwarz_besser"]`.
  Grundstellung und «Weiss kann die Dame schlagen» haben in Runde 1
  **identische Merkmale** (gleiches Material), deshalb kann Runde 1 die hängende
  Dame nicht erkennen. Das ist das Argument für Runde 2.
- Den **Primärschlüssel** findest du unter *Nutzen*. Er gehört nicht ins Repo
  und nicht ins Video.

## 7. Notebook ausführen

In VS Code den Ordner `Chess-Project` öffnen, im Terminal:

```powershell
.\.venv\Scripts\activate
pip install -r azure_ml\requirements.txt
```

`azure_ml/endpoint_aufruf.ipynb` öffnen, Kernel **.venv** wählen.

- URL und `FORMAT = "mlflow"` sind schon eingetragen, `DEPLOYMENTS` enthält vorerst nur `runde1`.
- Alles ausführen und beim Aufruf den Primärschlüssel eingeben (aus *Nutzen*).
  Er wird nicht gespeichert. **Im Video nicht zeigen.**

Ergebnis: drei Beispielstellungen mit Brett und Vorhersage, dann die Auswertung
aller 1 373 Teststellungen mit Konfusionsmatrix.

## 8. Verbessern: Runde 2 ✅ Daten v3 hochgeladen, Auftrag gestartet am 23.10.2026, 13:40

1. **Daten** → `schach-stellungen` → **Neue Version** → Ordner hochladen
   `azure_ml/daten/runde2` → ergibt **Version 3**. Das ist die versionierte
   Datenressource, ein sichtbarer MLOps-Baustein. In der Vorschau prüfen, dass
   keine «Fehler» stehen.
2. **Danach** einen neuen AutoML-Auftrag `schach-lage-runde2` im **selben
   Experiment** `schach-lage` starten (Vorhandenes auswählen), Daten
   `schach-stellungen` Version 3, sonst gleiche Einstellungen wie in Runde 1.
3. Im Experiment beide Aufträge vergleichen und Metriken nebeneinander
   festhalten. Bei den Erklärungen sollte jetzt `bilanz_nach_bestem_schlag`
   weit oben stehen.

### Ergebnis Runde 2 (Auftrag `schach-lage-runde2`, 17 min, 20 Modelle)

Bestes Modell wieder ein **VotingEnsemble**, registriert als
`schachlagerunde19:2`, also als Version 2 desselben Modellnamens.

| Kennzahl (Kreuzvalidierung) | Runde 1 | Runde 2 |
|---|---|---|
| AUC gewichtet | 0.959 | **0.975** |
| Genauigkeit | 86.9 % | **90.2 %** |
| ausgewogene Genauigkeit | 81.4 % | **84.7 %** |
| F1 gewichtet | 0.868 | **0.899** |

| Runde 2: richtig ↓ / vorhergesagt → | ausgeglichen | schwarz_besser | weiss_besser | Trefferquote |
|---|---|---|---|---|
| **ausgeglichen** | 601 | 146 | 180 | 64.8 % |
| **schwarz_besser** | 93 | 2268 | 51 | 94.0 % |
| **weiss_besser** | 109 | 30 | 2722 | 95.1 % |

Alle drei Klassen werden besser erkannt. «Ausgeglichen» bleibt die schwierigste
Klasse: Sie ist die kleinste (15 %), und die Grenze bei ±150 Centipawns ist
fliessend.

## 9. Runde 2 deployen, ohne Runde 1 sofort abzulösen ✅ ausgelöst am 23.10.2026, 14:02

1. Bestes Modell aus Runde 2 → **Bereitstellen** → Echtzeitendpunkt →
   **vorhandener Endpunkt** `schach-lage-dgerm`, Bereitstellungsname **`runde2`**,
   1 × Standard_D2a_v4. Auch hier waren 3 × DS3_v2 vorgeschlagen.
2. Datenverkehr zuerst bei **runde1 100 %, runde2 0 %** lassen. Der
   Schnell-Dialog vergibt einer neuen Bereitstellung automatisch 0 %.
   - **Prüfen, welches Modell die Bereitstellung verwendet.** Nach dem
     Absenden stand bei `runde2` die Modell-ID `schachlagerunde19:1`, also das
     Modell von Runde 1.
   - **Befund vom 23.10.2026 (Fehler im Schnell-Dialog):** Der Dialog
     «Bereitstellen» auf der Seite des Auftrags hat das Modell von Runde 2 als
     neue Version `schachlagerunde19:2` registriert, die Bereitstellung aber
     mit `:1` verknüpft. Nachweis im Test-Tab: `runde2` liefert auf die drei
     Beispielstellungen dasselbe wie `runde1` und nimmt auch eine Anfrage mit
     nur den 11 Spalten von Runde 1 an. Ein Modell von Runde 2 würde die
     fehlenden 13 Spalten verlangen. MLflow ignoriert überzählige Spalten, der
     Fehler fällt deshalb nicht von selbst auf.
   - **Lehre für die Arbeit (Governance):** Nach jedem Deployment prüfen, welche
     Modellversion wirklich läuft, nicht nur, ob der Status «Erfolgreich» ist.
   - **Korrektur ✅ (14:33–14:44):** Bereitstellung `runde2` gelöscht und über
     *Modelle → schachlagerunde19:2 → Dieses Modell verwenden →
     Echtzeitendpunkt* neu angelegt (vorhandener Endpunkt, 1 × D2a_v4).
     Diesmal steht im Titel des Dialogs «schachlagerunde19:2 bereitstellen»,
     und die Details zeigen Modell-ID `schachlagerunde19:2`, 0 % Verkehr.
   - **Nachweis im Test-Tab:** Eine Anfrage mit nur 11 Spalten an `runde2`
     schlägt jetzt fehl, wie es sein muss. Mit 24 Spalten antwortet `runde2`
     auf die drei Beispielstellungen `["ausgeglichen", "weiss_besser",
     "schwarz_besser"]`. Die hängende Dame erkennt Runde 2 also, Runde 1
     sagte dort «ausgeglichen».
3. Notebook: `DEPLOYMENTS = {"runde1": 1, "runde2": 2}`. Das Notebook spricht
   jede Bereitstellung gezielt an (Header `azureml-model-deployment`) und
   vergleicht beide auf denselben Teststellungen.
4. Ist Runde 2 besser: Datenverkehr auf **runde2 100 %** umstellen. Danach
   `runde1` löschen, das halbiert die laufenden Kosten.
   ✅ Am 23.10.2026 um 14:52 umgestellt (*Datenverkehr aktualisieren*: runde1 0 %,
   runde2 100 %), danach `runde1` gelöscht. Im Endpoint läuft nur noch
   `runde2` mit 1 × D2a_v4 (0.14 $/h). Das Notebook spricht nur noch `runde2` an.
   Die Werte von Runde 1 stehen oben in den Tabellen.

Das entspricht **Blue-Green-Deployment** aus der DevOps-Lektion: Die neue
Version läuft schon, bekommt aber erst nach der Prüfung den Verkehr
(Release ≠ Deployment).

Reicht das Kontingent nicht für zwei Bereitstellungen gleichzeitig, kannst du
eine von zwei Varianten wählen:
- `runde1` vor dem Deploy von Runde 2 löschen, nachdem du ihre Notebook-Ergebnisse festgehalten hast.
- Runde 2 als eigenen Endpunkt anlegen.

Die gewählte Variante in der Arbeit kurz begründen.

### Ergebnis im Notebook (runde2, Endpoint live, 23.10.2026)

Drei Beispielstellungen: Grundstellung → ausgeglichen, Weiss kann die Dame
schlagen → weiss_besser, Schwarz hat zwei Türme mehr → schwarz_besser.

| richtig ↓ / vorhergesagt → | weiss_besser | ausgeglichen | schwarz_besser | Trefferquote |
|---|---|---|---|---|
| **weiss_besser** | 615 | 20 | 18 | 94.2 % |
| **ausgeglichen** | 30 | 99 | 59 | 52.7 % |
| **schwarz_besser** | 17 | 17 | 498 | 93.6 % |

Gesamt **1 212 von 1 373 richtig (88.3 %)**. Das liegt nur wenig unter der
Kreuzvalidierung in Azure (90.2 %), das Modell verallgemeinert also auf neue
Partien. Schwach bleibt «ausgeglichen».

**Notebook starten:** Jupyter läuft über `.claude/launch.json` im Ordner
AI Operations (Name `jupyter-azure-ml`), Adresse
`http://localhost:8888/notebooks/azure_ml/endpoint_aufruf.ipynb?token=…`. Den
Schlüssel liest das Notebook aus `azure_ml/endpoint_key.txt` (in `.gitignore`).

## Screenvideo (ca. 2 min, Ton optional)

1. Studio → **Endpunkte** → `schach-lage-dgerm`: Status, Bereitstellung `runde2` mit 100 % Verkehr und Modell-ID `schachlagerunde19:2`. Den Wechsel von runde1 zu runde2 im Text erklären (Blue-Green), er ist nicht mehr live zu sehen.
2. Kurz die beiden AutoML-Aufträge im Experiment zeigen: Metriken Runde 1 gegen Runde 2.
3. Notebook: Zelle mit den Beispielstellungen ausführen, Brett und Vorhersage mit dem Mauszeiger zeigen.
4. Auswertung: Genauigkeit und Konfusionsmatrix von `runde1` und `runde2`.
5. **Den Schlüssel nie einblenden.**

## Kosten festhalten (Kapitel 3.1)

Portal https://portal.azure.com → **deine Ressourcengruppe** →
**Kostenverwaltung** → **Kostenanalyse**:
- Ansicht nach **Ressource** und nach **Dienst** gruppieren, Screenshot.
- Kosten erscheinen mit **bis zu 24–48 h Verzögerung**. Den Screenshot also ein bis zwei Tage nach dem Aufräumen machen.

Erwartete Kostentreiber für den Text:
- **Managed Online Endpoint**: VM-Stunden rund um die Uhr, pro Bereitstellung.
- **Compute für das Training**: nur während der AutoML-Aufträge.
- Speicher, Key Vault, Application Insights: wenige Rappen.

## 10. Aufräumen (wichtig)

Sobald Video, Screenshots und Notebook-Ergebnisse gesichert sind:
1. **Endpunkte** → `schach-lage-dgerm` → **Löschen**. Das ist der grösste Kostentreiber.
2. **Compute** → Cluster löschen, oder sicherstellen, dass min. Knoten = 0.
3. Workspace und Daten dürfen bleiben (fast kostenlos), falls du später nochmals deployen willst.

## Für die Arbeit festhalten

| | Runde 1 | Runde 2 |
|---|---|---|
| Datenversion | `schach-stellungen` v2 | `schach-stellungen` v3 |
| bestes Modell (Algorithmus) | VotingEnsemble (`schachlagerunde19:1`) | VotingEnsemble (`schachlagerunde19:2`) |
| AUC gewichtet (Azure) | 0.959 | 0.975 |
| Genauigkeit (Azure) | 86.9 % | 90.2 % |
| ausgewogene Genauigkeit (Azure) | 81.4 % | 84.7 % |
| Genauigkeit im Notebook (1 373 Teststellungen aus 20 ungesehenen Partien) | nicht mehr gemessen (runde1 vorher gelöscht; lokaler Vorabtest 78–85 %) | **88.3 %** (1 212 richtig, gemessen am 23.10.2026) |
| wichtigste 3 Merkmale | | |
| Trainingsdauer | 18 min (serverlos, 2 × DS3_v2) | 17 min (serverlos, 2 × DS3_v2) |
| VM Endpoint | 1 × Standard_D2a_v4 | 1 × Standard_D2a_v4 |

Für das **Hilfsmittelverzeichnis**: «Claude Code | Merkmalsextraktion,
Datensatz-Skript, Notebook für den Endpoint-Aufruf, Anleitung | Kap. 3,
`azure_ml/`».

## Fehlerbehebung

| Meldung | Ursache und Lösung |
|---|---|
| Fehlende Berechtigung | neu pimmen (beide Rollen), abmelden, anmelden |
| «Fehler» in der Datenvorschau | MLTable aus `python -m azure_ml.datensatz` verwenden (feste Ganzzahl-Typen) und als neue Version hochladen |
| Kontingent überschritten (*quota*) | andere kleine VM-Grösse oder weniger Instanzen |
| Notebook: `HTTP Error 401` | falscher Schlüssel oder Schlüssel des falschen Endpoints |
| Notebook: `HTTP Error 400` / `424` | `FORMAT` falsch gewählt, oder die Runde passt nicht zur Bereitstellung (Runde 1 erwartet 11 Spalten, Runde 2 erwartet 24) |
| Notebook: `HTTP Error 404` | Bereitstellungsname in `DEPLOYMENTS` stimmt nicht mit Azure überein |
| Endpoint antwortet lange nicht | Bereitstellung noch nicht «Erfolgreich», oder Datenverkehr steht bei 0 % und der Header fehlt |
