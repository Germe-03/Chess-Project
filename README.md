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
Code/main.py         FastAPI-App (API + liefert beide Oberflächen aus)
Code/rl/             Reinforcement Learning: Umgebung, Training, Agent, Registry
Code/vision/         Stellungserkennung aus Screenshots
web/index.html       Analyse: Screenshot einfügen oder Stellung stellen
web/spielen.html     gegen die gewählte Modellversion spielen
models/              trainierte Modelle (.npz) und registry.json
tests/               Testsuite (pytest), kommt ohne Netzzugriff aus
logs/                Trainingsprotokolle (nicht im Repo)
tools/               Stockfish für die Auswertung (nicht im Repo, 103 MB)
requirements.txt     Abhängigkeiten des Dienstes
requirements-dev.txt zusätzlich für die Tests
render.yaml          render.com Blueprint
```

Stockfish wird **nur zum Messen** gebraucht, nie zum Ausliefern. Die Datei lädt
beim Start ein eigenes neuronales Netz nach und würde die 512-MB-Instanz
sprengen; sie ist deshalb per `.gitignore` ausgeschlossen.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -q
```

30 Tests, unter einer Sekunde, ohne Netzzugriff. Geprüft wird bewusst **nicht**
die Spielstärke — die schwankt und lässt sich nicht sinnvoll festnageln.
Geprüft werden die Rechenregeln, bei denen ein Fehler **still** passiert:

- Vorzeichen und Perspektive (Negamax, gespiegelte Brettkodierung)
- Φ ist an Endstellungen null und bleibt betragsmässig unter dem Siegwert
- die statische Abtauschbewertung an bekannten Fällen
- Net2Net verbreitert funktionserhaltend
- die Kachelreihenfolge beginnt oben links — sonst steht jede FEN auf dem Kopf
- die Erkennung auf **beiden** Brettstilen, mit und ohne Rand

Der letzte Punkt ist der wichtigste: Er schlägt an, wenn ein Nachtraining die
ursprüngliche Fähigkeit zerstört hat. Das wäre sonst erst aufgefallen, wenn
jemand einen Lichess-Screenshot einfügt.

## API

| Endpoint                     | Eingabe                        | Ausgabe                                     |
|------------------------------|--------------------------------|---------------------------------------------|
| `GET /health`                | –                              | `{"status":"ok"}`                           |
| `GET /versions`              | –                              | verfügbare Modellversionen, aktive markiert |
| `POST /best-move`            | `{fen, version?}`              | `{best_move, san, evaluation, engine, version}` |
| `POST /legal-moves`          | `{fen, square}`                | erreichbare Felder dieser Figur             |
| `POST /spielen/zug`          | `{fen, move, version?}`        | Stellung nach beiden Zügen, Status          |
| `POST /recognize`            | Bild, `side`, `flipped`        | erkannte Stellung, Sicherheit, Hinweis      |
| `POST /best-move-from-image` | Bild, `side`, `flipped`        | erkannte Stellung **und** bester Zug        |

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

# Stellungserkennung

Auf der Analyseseite genügt **Strg+V**: Screenshot einfügen, die Stellung landet
im Brett, der Zug wird vorgeschlagen. Ziehen und Dateiauswahl gehen auch.

## Das Modell

Der Figurenerkenner stammt aus
[linrock/chessboard-recognizer](https://github.com/linrock/chessboard-recognizer)
(MIT) und klassifiziert jedes der 64 Felder einzeln:

```
32×32×1 Graustufen
Conv2D(32, 3×3) + ReLU  →  MaxPool 2×2
Conv2D(64, 3×3) + ReLU  →  MaxPool 2×2
Conv2D(64, 3×3) + ReLU
Flatten → Dense(64) + ReLU → Dense(13) + Softmax
```

122 189 Parameter, 13 Klassen (`1RNBQKPrnbqkp`, Index 0 = leer).

**Ausgeliefert wird es in reinem NumPy**, aus demselben Grund wie beim
Value-Netz: TensorFlow installiert wäre rund 600 MB und würde die 512-MB-Instanz
sprengen, die Gewichte allein sind 0.46 MB. TensorFlow kam genau einmal zum
Einsatz, um die Gewichte aus dem Checkpoint zu lesen, und ist weder im
Repository noch im Deployment.

Die Faltung läuft über `as_strided` als eine einzige Matrixmultiplikation; eine
Schleife über Pixel wäre in NumPy um Grössenordnungen langsamer.

Ehrliche Einordnung: Das ist eine **dritte Kategorie** neben dem selbst
gelernten `V(s)` und dem handgeschriebenen Φ — ein übernommenes Modell, das wir
anschliessend auf die eigene Domäne nachgezogen haben.

## Das Brett im Bild finden

Ohne diesen Schritt ist die Erkennung unbrauchbar. Die Kacheln entstehen durch
stures Achteln des Bildes; sobald Rand, Koordinaten oder eine Seitenleiste mit
drauf sind, liegen die Kachelgrenzen mitten in den Feldern. Gemessen fiel die
Sicherheit dann von 1.000 auf 0.65, und die Stellung war Unsinn.

Ein erster Versuch über **gleichmässig verteilte Kanten** — ein Brett hat neun
Gitterlinien je Richtung — scheiterte: Figuren erzeugen zu viele konkurrierende
Kanten, nur einer von sechs Testfällen stimmte.

Was funktioniert, ist die **Farbe**: Ein digitales Brett besteht aus zwei
abwechselnden Flächenfarben und ist meist das grösste solche Farbpaar im Bild.
Das allein genügt aber nicht, weil ein dunkler Rahmen in ähnlicher Farbe den
Ausschnitt mitwachsen lässt. Deshalb wird jeder Kandidat zusätzlich geprüft, ob
seine 8×8-Unterteilung wirklich abwechselnd hell und dunkel ist — abgetastet
wird die **Ecke** jeder Zelle, nicht die Mitte, dort stehen die Figuren.

Damit: 8 von 8 Testfällen, Abweichung 0 bis 1 Pixel, auch bei einem echten
Screenshot mit 1460×1434 Pixeln.

## Nachtrainiert auf die eigene Domäne

Das übernommene Modell erkannte Lichess-Bretter fehlerfrei, scheiterte aber an
einem chess.com-Screenshot: 13 von 64 Feldern falsch. Die Fehler waren
systematisch — 10 davon weisse Figuren, 9 davon auf hellen Feldern. Eine weisse
Figur auf einem cremefarbenen Feld verschwindet in Graustufen fast.

Besonders unangenehm: Das Modell war **selbstsicher falsch**, Sicherheit 1.00
für ein leeres Feld, auf dem ein Turm stand. Die Warnschwelle greift dort nicht.

Kontrastnormierung half nicht, andere Farbkanäle auch nicht. Die Information
fehlte dem Modell, nicht dem Bild.

**Die Trainingsdaten entstanden aus einem einzigen Screenshot.** Die
Feldhintergründe sind einfarbig, also lassen sich die Figuren als Maske
freistellen — alle zwölf Typen waren in der Stellung vorhanden. Daraus 500
Bretter mit zufälligen Belegungen gerendert, Labels per Konstruktion richtig.
Dazu 220 gerenderte Lichess-Bretter mit bekannten Stellungen. Macht 48 896
beschriftete Kacheln, ohne eine einzige von Hand zu beschriften.

Drei Entscheidungen gaben den Ausschlag:

**Zufällige statt realistischer Stellungen.** Der Klassifikator sieht jede
Kachel einzeln, ohne Zusammenhang — Gleichverteilung der zwölf Figurentypen
hilft ihm mehr als Realismus. Der erste Versuch mit realistischen Stellungen
lieferte 5 weisse Damen gegen 2467 leere Felder.

**Beide Stile gemischt** gegen katastrophales Vergessen.

**Störungen**: wechselnde Bildgrössen, versetzte Zuschnitte, zufällig markierte
Felder wie nach einem Zug.

Ergebnis nach 12 Epochen und 90 Sekunden Training:

| | übernommenes Modell | nachtrainiert |
|---|---|---|
| chess.com-Screenshot | 51/64 Felder, falsch | **64/64, richtig**, Sicherheit 1.00 |
| Lichess, 12 Bretter | 12/12, 768/768 Felder | **12/12, 768/768 Felder** |
| Prüfdatensatz | 76.5 % | **99.74 %** |

Die alte Fassung liegt als `models/tile_classifier_v1.npz` daneben.

## Grenzen

Zwei Dinge kann ein Bild nicht hergeben, sie werden deshalb abgefragt: **wer am
Zug ist** und ob das Brett **aus Sicht von Schwarz** dargestellt ist. Letzteres
wird durch Umkehren der 64 Felder gelöst, nicht durch Drehen der
FEN-Zeichenkette — das wäre wegen der Zifferncodierung subtil fehleranfällig.

Erkennung und Zugberechnung sind **getrennte Endpunkte**. Liefert die Erkennung
eine unspielbare Stellung, gibt `/recognize` sie trotzdem zurück, mit einem
Hinweis statt eines Fehlers. Ein Fehler würde die erkannte Stellung verwerfen
und den Nutzer bei null anfangen lassen; so landet sie im Brett und ist
nachbesserbar.

Andere Brettdesigns als die beiden trainierten sind ungetestet. Das Verfahren
lässt sich aber wiederholen: ein Screenshot mit bekannter Stellung genügt, um
einen weiteren Stil nachzuziehen.

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
python -m Code.rl.train --task full --shaping --episodes 5000 --max-plies 400
python -m Code.rl.train --task full --resume models/full_v2.npz --widen 512 \
    --episodes 3000 --max-plies 400 --eval-opponent stockfish
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

### Die Wahl des Prüfgegners

`--eval-opponent` bestimmt, woran der Fortschritt während des Trainings gemessen
wird. Die Material-Heuristik ist ab rund 1 000 Episoden erschöpft — der Agent
schlägt sie dann in jeder Partie, und eine Messlatte bei +100 Prozent misst
nichts mehr. Für längere Läufe gehört deshalb

```bash
--eval-opponent stockfish --stockfish-elo 1320
```

dazu. Der Unterschied ist sofort sichtbar: Wo die Heuristik konstant +100 Prozent
zurückmeldet, liefert Stockfish Werte wie −16.7 Prozent, die sich mit dem
Training bewegen können.

Die Engine wird **einmal für den ganzen Lauf** geöffnet, und zwar vor der ersten
Episode. Fehlt die Programmdatei, bricht das Training sofort mit einer klaren
Meldung ab, statt nach einer Stunde Rechnen bei der ersten Prüfung. Ein
ausdrücklich angegebener Pfad wird dabei nie stillschweigend durch eine andere
gefundene Engine ersetzt.

## Zwischenbelohnungen für ganze Partien

Für Endspiele reicht der Partieausgang als einziges Signal. In ganzen Partien
nicht — dort vergehen hunderte Halbzüge ohne jede Rückmeldung. Mit `--shaping`
kommt deshalb eine Grundbewertung Φ dazu (`Code/rl/shaping.py`), die als
**Differenz** zwischen zwei Stellungen ausgezahlt wird:

```
Belohnung = r_Ende  −  gamma · Φ(s')  −  Φ(s)
```

Das ist Potential-Based Reward Shaping (Ng, Harada & Russell 1999). Entscheidend
ist, dass für die geshapte Bewertung

```
V_geshaped(s) = V*(s) − Φ(s)
```

gilt. Weil `Φ(s)` für alle Züge aus derselben Stellung derselbe Wert ist,
verschiebt es die Reihenfolge der Züge nicht — die beste Wahl bleibt die beste
Wahl. Das Shaping ändert also nur, **wann** die Rückmeldung kommt, nicht **was**
sich lohnt.

**Eine Falle, in die wir zuerst getappt sind:** Das übliche Teleskop-Argument —
die Φ-Terme höben sich über eine Episode paarweise weg, übrig bleibe nur der
Ausgang — gilt hier **nicht**. In der Negamax-Form steht `−γ·Φ(s')` und im
nächsten Halbzug `−Φ(s')`; beide haben dasselbe Vorzeichen und addieren sich,
statt sich aufzuheben. An einer echten Partie nachgerechnet summierten sich die
Belohnungen auf **+17.35** statt auf null. Das Teleskop gilt für einen
Einzelagenten, nicht für zwei abwechselnde Parteien. Die Policy-Invarianz bleibt
trotzdem bestehen, aber über die Gleichung oben.

Dass in der Praxis nichts ausufert, lässt sich an den Lernzielen einer Episode
ablesen: Sie liegen zwischen −0.34 und +0.40, wechseln bei 304 von 333 Halbzügen
das Vorzeichen und haben einen Mittelwert von +0.02 — genau das Bild, das man bei
abwechselnder Bewertung zweier Seiten erwartet.

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
| **v3 gegen v2** | **100 %** | 0 % | **0 %** | **100 %** |
| v2 gegen v1 | 52 % | 48 % | 0 % | 76 % |
| v3 gegen Material-Heuristik | 100 % | 0 % | 0 % | 100 % |
| **v3 gegen Stockfish (Elo 1320)** | **24 %** | 10 % | 66 % | **29 %** |
| v2 gegen Stockfish (Elo 1320) | 16 % | 22 % | 62 % | 27 % |
| v1 gegen Stockfish (Elo 1320) | 12 % | 20 % | 68 % | 22 % |
| Material-Heuristik gegen Stockfish | 0 % | 6 % | 94 % | 3 % |
| nur Φ, untrainiert, gegen Material | 50 % | 50 % | 0 % | 75 % |

Vier Dinge lassen sich daraus ablesen.

**Jede Version schlägt ihren Vorgänger.** v2 verliert keine Partie gegen v1, v3
gewinnt alle 50 gegen v2.

**Der Spielstil kippt von Sicherheit auf Gewinn.** v2 hatte vor allem gelernt,
nicht zu verlieren: gegen Stockfish 16 Prozent Siege bei 22 Prozent Remis. v3
dreht das um — 24 Prozent Siege bei nur noch 10 Prozent Remis. Unter dem Strich
ändert das gegen den stärkeren Gegner wenig (27 auf 29 Punkteprozent), gegen den
gleich starken Vorgänger ist es vernichtend.

**Der TD-Fehler taugt nicht als Fortschrittsmass.** Bei v3 blieb er über alle
3 000 Episoden flach bei 0.11 und lag damit sogar über dem Endwert von v2
(0.099) — während v3 gleichzeitig 100 Prozent gegen v2 gewinnt. Er misst, wie
gut der Agent seine **eigenen** Partien vorhersagt; werden die Partien länger und
schwerer, kann er steigen, obwohl die Spielstärke wächst. Nur der direkte
Vergleich zählt.

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

**Was jede Version ausmacht:**

| | Episoden | Netz | Abbruchlimit | Dauer |
|---|---|---|---|---|
| v1 | 1 000 | 128 Neuronen, 99 k Parameter | 160 Halbzüge | – |
| v2 | 5 000 | 128 Neuronen, 99 k Parameter | 160 Halbzüge | 104 min |
| **v3** | 3 000 | **512 Neuronen, 397 k Parameter** | **400 Halbzüge** | 134 min |

Zwei Änderungen machten v3 aus. Das Netz wurde per **Net2Net** verbreitert, also
funktionserhaltend: v3 startete exakt auf v2-Niveau statt bei null, die Abweichung
lag bei 6·10⁻⁸. Gemessen kostet die vierfache Kapazität nur 3 Prozent Rechenzeit,
weil nicht das Netz der Flaschenhals ist, sondern Φ mit seiner SEE-Rechnung
(3.7 von 6.7 ms pro Halbzug).

Und das **Abbruchlimit** stieg von 160 auf 400 Halbzüge. Das war der grössere
Hebel: Bei v2 endeten 55 Prozent der Episoden am Limit, der Agent bekam dort also
nie ein echtes Ergebnis zu sehen, sondern bootstrappte aus der eigenen Schätzung.
Bei v3 sind es noch 1.9 Prozent. Gemessen an 30 Partien mit Limit 600 endet
**jede** Partie von selbst, im Median bei Halbzug 238.

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
