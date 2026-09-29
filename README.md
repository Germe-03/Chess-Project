# Chess-Project — Schach-Assistent

FHNW **AI Operations**. Web-Service, der zu einer Schachstellung den besten Zug
liefert — und gegen den man auch selbst spielen kann. Die Züge kommen von
Reinforcement-Learning-Agenten, die per Self-Play trainiert wurden: zwei
Endspiel-Spezialisten und ein versioniertes Modell für ganze Partien.
Oberfläche nach den Prinzipien der Mensch-Maschine-Interaktion.

## Lokal starten (Aufgabe 1.1 — DevOps lokal)

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows
pip install -r requirements.txt
uvicorn Code.main:app --reload
```

- UI:      http://127.0.0.1:8000/
- API-Doku: http://127.0.0.1:8000/docs

## Deployment (Aufgabe 1.2 — DevOps Cloud, render.com)

Neuer **Web Service** auf render.com, mit diesem GitHub-Repo verbunden:

| Einstellung   | Wert                                                  |
|---------------|-------------------------------------------------------|
| Branch        | `main`                                                |
| Build Command | `pip install -r requirements.txt`                     |
| Start Command | `uvicorn Code.main:app --host 0.0.0.0 --port $PORT`   |
| Instance Type | Free                                                  |

Alternativ die mitgelieferte `render.yaml` als **Blueprint** verwenden.

Das Training braucht kein PyTorch und der Service auch nicht: Das Value-Netz ist
in reinem NumPy implementiert, die Modelldateien sind wenige hundert Kilobyte
gross. Damit bleibt die 512-MB-Instanz auf render.com ausreichend.

## Projektstruktur

```
Code/main.py        FastAPI-App (API + liefert beide Oberflächen aus)
Code/rl/            Reinforcement Learning: Umgebung, Training, Agent, Registry
web/index.html      Analyse: Stellung stellen, besten Zug berechnen
web/spielen.html    gegen die gewählte Modellversion spielen
models/             trainierte Modelle (.npz) und registry.json
logs/               Trainingsprotokolle (nicht im Repo)
tools/              Stockfish für die Auswertung (nicht im Repo, 103 MB)
requirements.txt    Abhängigkeiten
render.yaml         render.com Blueprint
```

Stockfish wird **nur zum Messen** gebraucht, nie zum Ausliefern. Die Datei lädt
beim Start ein eigenes neuronales Netz nach und würde die 512-MB-Instanz
sprengen; sie ist deshalb per `.gitignore` ausgeschlossen.

## API

| Endpoint                     | Eingabe                        | Ausgabe                                     |
|------------------------------|--------------------------------|---------------------------------------------|
| `GET /health`                | –                              | `{"status":"ok"}`                           |
| `GET /versions`              | –                              | verfügbare Modellversionen, aktive markiert |
| `POST /best-move`            | `{fen, version?}`              | `{best_move, san, evaluation, engine, version}` |
| `POST /legal-moves`          | `{fen, square}`                | erreichbare Felder dieser Figur             |
| `POST /spielen/zug`          | `{fen, move, version?}`        | Stellung nach beiden Zügen, Status          |
| `POST /best-move-from-image` | Bild (multipart `image`)       | dito, Stellungserkennung noch Platzhalter   |

Das Feld `engine` sagt, wer den Zug gewählt hat: `rl` oder `material`. Die Skala
von `evaluation` hängt davon ab — Centipawns bei `material`, erwarteter
Partieausgang in `[-1, 1]` bei `rl`. `version` nennt das antwortende Modell,
etwa `v2` oder `endspiel-queen`.

## Oberflächen

| Adresse | Zweck |
|---|---|
| `/` | Stellung von Hand setzen und den besten Zug berechnen lassen. Eine Figur antippen zeigt ihre regelkonformen Züge — inklusive Fesselung und der Regel, dass der König nicht ins Schach ziehen darf. |
| `/spielen` | Eine ganze Partie gegen die gewählte Modellversion spielen, mit Farbwahl, Zugliste und Zurücknehmen. |
| `/docs` | Automatisch erzeugte API-Dokumentation |

Der Server hält dabei **keine Partie im Speicher**. Der Browser schickt die
Stellung als FEN mit und bekommt die neue zurück. Dadurch braucht es weder
Sitzungen noch eine Datenbank, und der Dienst verträgt Neustarts und mehrere
Instanzen — Voraussetzung für render.com und später Azure.

## Modellversionen

Die Vollpartien-Modelle sind versioniert, jede Version ist eine eigene Datei und
wird nie überschrieben. Das Manifest `models/registry.json` hält fest, was eine
Version ausmacht: Episodenzahl, Elternversion, Datum und gemessene Resultate.

```bash
python -m Code.rl.registry                  # Versionen anzeigen
python -m Code.rl.registry --activate v1    # Rollback auf eine ältere Version
```

Umgeschaltet wird auf zwei Ebenen. Global über den Eintrag `active` im Manifest,
und **pro Anfrage** über das Feld `version`. Das zweite erlaubt den direkten
Vergleich: dieselbe Stellung von v1 und v2 rechnen lassen, ohne Neustart. In der
Grundstellung empfiehlt v1 `b3`, v2 dagegen `e4` — der Lernfortschritt ist so
unmittelbar sichtbar.

Die Endspielmodelle sind bewusst **nicht** versioniert. Sie lösen eine feste
Aufgabe und ändern sich nicht mehr; bei genau drei Figuren antworten immer sie.

---

# Reinforcement Learning

## Was gelernt wird

Gelernt wird eine **Bewertungsfunktion** `V(s)`: Wie gut steht die Partei, die am
Zug ist? Der Zug selbst entsteht dann durch einen Blick einen Halbzug voraus —
für jeden legalen Zug wird die Folgestellung bewertet und die beste genommen.

Ehrliche Einordnung: Reinforcement Learning ist hier **ausschliesslich** das
Lernen von `V(s)` aus Self-Play. Die Vorausschau um einen Zug ist klassische
Suche, denn das Modell der Umwelt (die Schachregeln) ist bekannt. Genau diese
Struktur hatte die bisherige Platzhalter-Engine auch — ersetzt wurde nur die
handgeschriebene Materialzählung durch eine gelernte Funktion.

## Aufgabenstellung: Endspiele statt ganzer Partien

Trainiert wird auf **König + Dame gegen König** (`kqk`) und **König + Turm gegen
König** (`krk`). Das ist bewusst gewählt:

- Ein AlphaZero-artiges Training auf dem vollen Brett braucht Rechenleistung, die
  hier nicht zur Verfügung steht.
- Die Endspiele trennen sauber zwischen gelernt und heuristisch: Das Material
  ändert sich nicht, also ist die Materialbilanz als Bewertung **wertlos**. Wer
  hier mattsetzt, hat wirklich etwas gelernt.
- Das Ergebnis ist messbar: Wie oft gelingt das Matt, und wie schnell?

`--task full` für ganze Partien ist implementiert, aber nicht das Ziel dieses
Laufs.

## MDP-Formulierung

| Element | Definition |
|---|---|
| Zustand `s` | Stellung inklusive Zugpartei, kodiert als 768 Merkmale (12 Figurenebenen × 64 Felder) |
| Aktion `a` | ein legaler Zug |
| Reward `r` | nur am Partieende, aus Sicht der Partei, die gezogen hat: **+1** Matt gesetzt, **0** Remis |
| Episodenende | Matt, Patt, ungenügendes Material, 50-Züge-Regel; zusätzlich Abbruch nach `max_plies` |
| `gamma` | 0.99 |

Zwei Entscheidungen, die zum Reward-Engineering aus dem Modul gehören:

**Kein Schrittabzug.** Im Grid-World-Beispiel gibt es zwei Wege, kurze Lösungen zu
belohnen: −1 pro Schritt oder `gamma < 1`. In einem Nullsummenspiel ist der
Schrittabzug falsch — er bestraft beide Seiten gleich und würde dem Verteidiger
das schnelle Verlieren belohnen. Mit `gamma < 1` stimmen die Anreize für beide:
Der Angreifer will das Matt früh (weniger Diskontierung auf +1), der Verteidiger
will es spät (die −1 wird stärker diskontiert).

**Perspektive statt Farbe.** Die Merkmale sind immer aus Sicht der Zugpartei
kodiert; bei Schwarz am Zug wird das Brett gespiegelt. Dadurch gilt exakt
`V_ich(s) = −V_gegner(s)` (Negamax), und ein einziges Netz spielt im Self-Play
beide Seiten.

## Algorithmus

TD(λ) mit Funktionsapproximation. Das Ein-Schritt-Ziel in Negamax-Form:

```
Ziel_t = r + gamma * ( -V(s') )
```

Reines TD(0) trägt den Mattwert aber nur einen Halbzug pro Update zurück,
während ein Matt aus einer zufälligen Endspielstellung 20 bis 30 Halbzüge
entfernt liegt. Deshalb werden die Ziele am Episodenende rückwärts zu
λ-Returns verrechnet:

```
G_t = (1 - lambda) * Ziel_t  -  gamma * lambda * G_(t+1)
```

Der letzte Schritt behält sein Ein-Schritt-Ziel. `lambda = 0` ergibt wieder
TD(0), `lambda = 1` den reinen Monte-Carlo-Return.

- **Netz:** 768 → 128 (ReLU) → 1 (tanh), rund 98 000 Parameter, reines NumPy mit
  handgeschriebenem Gradienten und Adam.
- **Exploration:** ε-greedy, ε linear von 0.9 auf 0.05. Gelernt wird auch aus den
  zufälligen Zügen (on-policy).
- **Updates:** pro Episode ein Batch statt eines Updates pro Halbzug — schneller
  und ruhiger, bei identischem Ziel.

## Curriculum gegen das Sparse-Reward-Problem

Gemessen (siehe unten): Unter zufälligem Spiel endet ein KQK-Endspiel nur in
**1 %** der Fälle mit Matt, ein KRK-Endspiel praktisch **nie** — fast alle Partien
enden mit Patt oder geschlagener Dame. Ein Agent, der nie eine positive Belohnung
sieht, lernt nichts.

Deshalb startet das Training mit Stellungen, in denen der schwache König bereits
in einer Ecke steht und der starke König daneben; das Matt ist dann wenige Züge
entfernt. Das hebt die Mattquote bei Zufallsspiel auf **9.3 %** (KQK) bzw.
**3.7 %** (KRK). Über das Training wird der Anteil dieser einfachen Stellungen
auf null gefahren (`difficulty` 0 → 1).

Wichtig für die Bewertung: Die **Belohnungsfunktion bleibt unangetastet** —
verändert wird nur die Verteilung der Startzustände. Damit entsteht kein Risiko
von Reward Hacking, und die Evaluation läuft immer auf gleichverteilten
Stellungen (`difficulty = 1`).

## Training und Evaluation starten

```bash
python -m Code.rl.train --task kqk --episodes 30000
python -m Code.rl.train --task krk --episodes 30000
python -m Code.rl.train --task full --shaping --episodes 5000 --max-plies 160
python -m Code.rl.train --task full --resume models/full_v2.npz --episodes 5000 --eps-start 0.3
python -m Code.rl.evaluate --task kqk --model models/value_kqk.npz --games 200
python -m Code.rl.evaluate --task full --model models/full_v2.npz --stockfish
```

Mit `--resume` setzt ein Lauf auf einem vorhandenen Modell auf, statt bei null zu
beginnen. Das Ergebnis wird danach als neue Version eingetragen:

```bash
python -m Code.rl.registry --register v3 --file full_v3.npz --episodes 5000 --parent v2 --notes "..."
```

Für den Stockfish-Vergleich muss die Programmdatei unter `tools/stockfish/`
liegen oder über `STOCKFISH_PATH` erreichbar sein.

Wichtigste Optionen: `--td-lambda`, `--gamma`, `--lr`, `--hidden`, `--eps-start` /
`--eps-end`, `--curriculum-episodes`, `--resume`, `--seed`. Das Training schreibt
eine Lernkurve nach `logs/train_<task>.csv` (pro Episode) und die
Zwischenevaluationen nach `logs/train_<task>_eval.csv`.

Gespeichert wird bei **jeder** Zwischenevaluation der aktuelle Stand, zusätzlich
der beste unter `<name>_best.npz`. Das ist wörtlich zu nehmen: Eine frühere
Fassung speicherte nur bei einer Verbesserung — als die Wertung gegen die
Heuristik ihr Maximum von +100 Prozent erreichte, stand danach kein Fortschritt
mehr auf der Platte, und ein fünfstündiger Lauf war am Ende nicht zu gebrauchen.
Wer als Messlatte etwas wählt, das sättigen kann, darf die Speicherung nicht
davon abhängig machen.

## Zwischenbelohnungen für ganze Partien

Für Endspiele reicht der Partieausgang als einziges Signal. In ganzen Partien
nicht — dort vergehen hunderte Halbzüge ohne jede Rückmeldung. Mit `--shaping`
kommt deshalb eine Grundbewertung Φ dazu (`Code/rl/shaping.py`), die als
**Differenz** zwischen zwei Stellungen ausgezahlt wird:

```
Belohnung = r_Ende  −  gamma · Φ(s')  −  Φ(s)
```

Das ist Potential-Based Reward Shaping (Ng, Harada & Russell 1999). Über eine
ganze Partie heben sich die Φ-Terme paarweise weg; übrig bleibt der echte Ausgang
minus einer Konstanten, die vom Verhalten des Agenten unabhängig ist. Genau
deshalb ist das System nicht ausnutzbar: Eine Figur ins Zentrum und wieder zurück
ergibt exakt null, und wer mit Materialvorsprung verliert, zahlt den Vorschuss
am Ende vollständig zurück.

Φ besteht aus vier Teilen, alle in Bauerneinheiten, zusammen gestaucht auf ±0.5:

| Teil | Inhalt |
|---|---|
| Material | Bauer 1, Springer 3, Läufer 3, Turm 5, Dame 9 |
| Zentrum | je Feld `0.20 · tanh(0.7·d)` auf d4/e4/d5/e5 und `0.05 · tanh(0.7·d)` auf c3–f6, mit `d` = eigene minus gegnerische Angreifer |
| Abtausch | `0.7 ·` eigenes bestes SEE `−` `0.3 ·` gegnerisches bestes SEE |
| Schach | `−0.3` im Schach, `−1.5` bei Doppelschach |

Zwei Details, die den Unterschied machen. Die **Sättigung** im Zentrum macht das
Aufgeben der Oberhand teuer, das Verkleinern eines ohnehin bestehenden
Vorsprungs aber fast gratis. Und die **statische Abtauschbewertung** (SEE)
rechnet den Schlagabtausch auf einem Feld zu Ende — dadurch ist ein guter Tausch
schon positiv, bevor der Gegner zurückschlägt, und ein schlechter negativ.

Der Schachabzug ist nie positiv, weil im legalen Schach nur die Partei am Zug im
Schach stehen kann. Belohnt wird dadurch der Übergang ins Schach; pariert der
Gegner, fliesst der Betrag wieder ab. Dauerschachs bringen also nichts — genau
die Eigenschaft, die eine direkte Schachbelohnung nicht hätte.

Gemessene Belohnungen einzelner Ereignisse:

| Ereignis | Belohnung |
|---|---|
| Matt setzen | bis **+1.000** |
| Läufer schlägt gedeckten Turm (SEE +2) | +0.118 |
| Doppelschach geben | +0.042 |
| Turm schlägt freien Bauern (SEE +1) | +0.017 |
| Schach geben | +0.010 |
| gedeckte Figur stehen lassen | +0.001 |
| ungedeckte Figur stehen lassen | −0.058 |
| Turm schlägt gedeckten Bauern (SEE −4) | −0.136 |

Der Preis: Φ muss auch beim Ausführen mitgerechnet werden, sonst wählt der Agent
systematisch falsch. Jede Modelldatei merkt sich deshalb im `.npz`, ob sie mit
Shaping trainiert wurde, und die Zugwahl richtet sich danach. Gemessene
Mehrkosten: **Faktor 2.4** pro Halbzug (2.7 ms auf 6.5 ms), also rund 1.5 bis 3
Stunden für 10 000 ganze Partien.

## Resultate

Je 200 Partien × 3 Seeds. Die getestete Strategie führt die starke Seite gegen
einen Zufallsverteidiger, die Startstellungen sind gleichverteilt über das Brett.

**König + Dame gegen König**

| Strategie | Matt gesetzt | Halbzüge bis Matt |
|---|---|---|
| **RL-Agent** | **97.8 %** ± 0.2 % | 18.1 |
| Material-Heuristik | 6.5 % ± 2.5 % | 16.9 |
| Zufall | 1.5 % ± 0.0 % | 45.7 |

**König + Turm gegen König**

| Strategie | Matt gesetzt | Halbzüge bis Matt |
|---|---|---|
| **RL-Agent** | **94.8 %** ± 0.6 % | 25.9 |
| Material-Heuristik | 5.3 % ± 1.3 % | 24.9 |
| Zufall | 0.5 % ± 0.0 % | 43.7 |

Die 5 bis 7 Prozent der Material-Heuristik stammen ausschliesslich aus dem
Matt-in-eins-Zusatz. Reines Materialzählen kann in diesen Endspielen
definitionsgemäss nichts ausrichten, weil sich das Material gar nicht ändert.

**Lernverlauf KQK:** 23 % nach 1 000 Episoden, 59 % nach 9 000, ab 13 000 stabil
über 93 %. Die Mattlänge fiel dabei von 40 auf 17 Halbzüge. 30 000 Episoden in
44 Minuten.

**TD(λ) gegen TD(0)**, vorab per A/B-Lauf geprüft (KQK, identische Einstellungen,
nur λ verschieden):

| λ | Mattquote nach 1 000 Episoden |
|---|---|
| 0.0 (reines TD(0)) | 24.2 % |
| 0.9 | **44.2 %** |

### Ganze Partien

Trainiert mit Reward Shaping und Zufallseröffnung. Gemessen über 25 Partien × 2
Seeds je Paarung, Farben abwechselnd, Abbruch nach 160 Halbzügen. Die
Punktewertung zählt ein Remis als halben Sieg.

| Paarung | Sieg | Remis | Niederlage | Punkte |
|---|---|---|---|---|
| **v2 gegen v1** | **52 %** | 48 % | **0 %** | 76 % |
| v2 gegen Material-Heuristik | 100 % | 0 % | 0 % | 100 % |
| v2 gegen Stockfish (Elo 1320) | 10 % | **36 %** | 54 % | **28 %** |
| v1 gegen Stockfish (Elo 1320) | 12 % | 20 % | 68 % | 22 % |
| Material-Heuristik gegen Stockfish | 0 % | 6 % | 94 % | 3 % |
| nur Φ, untrainiert, gegen Material | 50 % | 50 % | 0 % | 75 % |

Drei Dinge lassen sich daraus ablesen.

**Das Training wirkt.** v2 verliert keine einzige Partie gegen v1. Gegen
Stockfish steigt die Punktequote von 22 auf 28 Prozent.

**Gelernt wurde Verteidigung, nicht Angriff.** Die Siege gegen Stockfish blieben
gleich (12 → 10 Prozent), aber die Remis haben sich fast verdoppelt (20 → 36
Prozent) und die Niederlagen fielen von 68 auf 54 Prozent. Für eine Belohnung,
die Sieg und Remis klar trennt, ist das die naheliegende erste Stufe: Niederlagen
zu vermeiden ist leichter, als Gewinne zu erzwingen.

**Etwa die Hälfte der Spielstärke ist handgeschrieben.** Ein völlig untrainiertes
Netz schlägt mit Φ allein die Material-Heuristik bereits mit 75 Punkteprozent.
Der Abstand von dort zu v2 ist das, was das Lernen beigetragen hat — und der
Abstand der Heuristik zu Stockfish (3 gegen 28 Prozent) zeigt, dass der gelernte
Agent in einer anderen Liga spielt als die Engine, die er ersetzt hat.

**Warum Stockfish auf Elo 1320:** Ohne Begrenzung gewinnt Stockfish 20 von 20
Partien gegen alles hier — auch gegen die Heuristik. Selbst `depth 1` reicht
nicht, weil die Engine eine Quiescence-Suche anhängt und die Blätter mit einem
neuronalen Netz bewertet. Erst die Elo-Begrenzung macht den Vergleich
aussagekräftig; Knotenlimits allein wirken nicht.

**Lernverlauf v1 → v2:** Der TD-Fehler fiel über 5 000 Episoden von 0.124 auf
0.099 und lief danach flach. Trainingsdauer 104 Minuten.

## Grenzen und nächste Schritte

- Gegen Stockfish holt der Vollpartien-Agent 28 Punkteprozent, verliert also die
  Mehrheit der Partien. Für ein Verfahren mit einem Halbzug Vorausschau und
  98 000 Parametern ist das zu erwarten und wird hier bewusst so ausgewiesen.
- Die Material-Heuristik kommt nur noch zum Zug, wenn gar kein Modell zuständig
  ist. Sie bleibt trotzdem im Code, weil erst sie die Resultate einordnet.
- Die Bilderkennung (`/best-move-from-image`) ist weiterhin ein **Platzhalter**
  und liefert unabhängig vom hochgeladenen Bild die Grundstellung.
- Beim Shaping ist ein Teil der Spielstärke handgeschrieben (Material, Zentrum,
  Abtausch, Schach); `V` lernt nur noch die Korrektur darüber hinaus. Für die
  Endspielmodelle gilt das nicht — die sind ohne Shaping trainiert.
- Eine tiefere Suche (Alpha-Beta über `V`) würde die Spielstärke sofort erhöhen,
  ist aber Suche und nicht Lernen — deshalb hier bewusst bei einem Halbzug
  belassen.
- Naheliegende Ausbauschritte: gemeinsames Modell für mehrere Endspiele,
  Gymnasium-Wrapper für Vergleiche mit DQN/PPO aus stable-baselines3, und
  Lernkurven-Plots aus den CSV-Dateien für die Dokumentation.
