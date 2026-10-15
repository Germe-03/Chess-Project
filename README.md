# Chess-Project — Schach-Assistent

FHNW **AI Operations**. Web-Service, der zu einer Schachstellung den besten Zug
liefert — und gegen den man auch selbst spielen kann. Die Züge kommen von
Reinforcement-Learning-Agenten, die per Self-Play trainiert wurden: zwei
Endspiel-Spezialisten und ein versioniertes Modell für ganze Partien. Ab v4
rechnet die Engine mehrere Halbzüge voraus (Alpha-Beta-Suche über die gelernte
Bewertung). Stellungen lassen sich per Screenshot einfügen.
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

**Konfiguration über Umgebungsvariablen:**

| Variable | Standard | Bedeutung |
|---|---|---|
| `MAX_SEARCH_DEPTH` | `2` | Obergrenze der Vorausschau in Halbzügen. Auf der Free-Instanz (0.1 CPU) dauert ein Zug mit Tiefe 2 etwa 1 bis 3 Sekunden, mit Tiefe 3 eher 15 bis 30 — zu lang für eine Web-Anfrage. Lokal lässt sich die Grenze anheben. |
| `OPENBLAS_NUM_THREADS` (und `OMP_`/`MKL_`) | `1` | Rechenthreads von NumPy. Wird in `Code/main.py` gesetzt, bevor NumPy geladen wird. |

**Warum ein einziger Rechenthread:** Nach der Aktivierung von v5 dauerte ein Zug
im Mittelspiel auf render.com rund 15 Sekunden — lokal sind es 0.09. Auch ein
einfacher Zug von v3 brauchte 1.2 statt 0.005 Sekunden, während Anfragen ohne
Rechnung (`/health`) schnell blieben. Alles Rechnende war rund 170-mal langsamer,
nicht nur 10-mal, wie die 0.1 CPU erwarten liessen. Die Ursache ist ein
bekanntes Containerproblem: NumPys Matrixbibliothek startet so viele Threads,
wie der Server Kerne hat, nicht so viele, wie dem Container zugeteilt sind. Die
Threads warten aktiv, verbrauchen dabei das Kontingent, und der Container
drosselt daraufhin den ganzen Prozess. Für die kleinen Matrizen hier bringt
Mehrfädigkeit ohnehin nichts.

Die Grenze steht als Umgebungsvariable und nicht im Code, weil sie zur
Umgebung gehört und nicht zum Modell: Derselbe Stand läuft lokal tiefer als auf
der kleinen Cloud-Instanz. Greift sie, meldet die API im Feld `depth`, mit
welcher Tiefe tatsächlich gerechnet wurde. In `render.yaml` ist sie gesetzt;
wurde der Dienst von Hand angelegt, gilt ohne Eintrag der Standardwert.

## Projektstruktur

```
Code/main.py         FastAPI-App (API + liefert beide Oberflächen aus)
Code/rl/             Reinforcement Learning: Umgebung, Training, Agent, Registry
Code/vision/         Stellungserkennung aus Screenshots
web/index.html       Analyse: Screenshot einfügen oder Stellung stellen
web/spielen.html     gegen die gewählte Modellversion spielen
web/test.html        Prüfstand: Versionen im Vergleich, Partien mit Zugbewertung
models/              trainierte Modelle (.npz) und registry.json
tests/               Testsuite (pytest), kommt ohne Netzzugriff aus
testbench/           Prüfstand: die 50 Teststellungen und der Bericht für /test
logs/                Trainingsprotokolle (nicht im Repo)
tools/               Stockfish für die Auswertung (nicht im Repo, 103 MB)
runs/                Lauf-Mappen unterbrechbarer Trainings (nicht im Repo)
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

49 Tests, rund 10 Sekunden, ohne Netzzugriff. Geprüft wird bewusst **nicht**
die Spielstärke — die schwankt und lässt sich nicht sinnvoll festnageln.
Geprüft werden die Rechenregeln, bei denen ein Fehler **still** passiert:

- Vorzeichen und Perspektive (Negamax, gespiegelte Brettkodierung)
- Φ ist an Endstellungen null und bleibt betragsmässig unter dem Siegwert
- die statische Abtauschbewertung an bekannten Fällen
- Net2Net verbreitert funktionserhaltend
- die Kachelreihenfolge beginnt oben links — sonst steht jede FEN auf dem Kopf
- die Erkennung auf **beiden** Brettstilen, mit und ohne Rand
- **v1 bis v3 antworten bitgenau wie vor dem Einbau der Suche** (siehe unten)
- die Suche rechnet wie eine Minimax-Suche ohne jede Abkürzung
- der Suchwert passt zur Stellung am Ende der Hauptvariante — darauf baut
  TD-Leaf auf
- Matt in zwei wird mit drei Halbzügen gefunden, die Springergabel mit zwei
- ein Training lässt sich anhalten und fortsetzen, ohne dass Optimierer oder
  Protokoll etwas verlieren

Der Test zu v1 bis v3 ist der wichtigste der neuen. Vor jeder Änderung an der
Zugwahl wurden zwei Referenzen aufgenommen (`tests/fixtures/`): die Antworten
von v1, v2 und v3 auf 40 Stellungen — Zug **und** Bewertung — sowie Φ auf 611
Stellungen. Verglichen wird mit `==`, nicht ungefähr. Wer in der Oberfläche v3
wählt, bekommt also nachweislich dieselbe Engine wie vorher, obwohl Φ und die
Brettkodierung darunter neu geschrieben wurden.

Bei der Erkennung schlägt der Test auf beiden Brettstilen an, wenn ein
Nachtraining die ursprüngliche Fähigkeit zerstört hat. Das wäre sonst erst
aufgefallen, wenn jemand einen Lichess-Screenshot einfügt.

## API

| Endpoint                     | Eingabe                        | Ausgabe                                     |
|------------------------------|--------------------------------|---------------------------------------------|
| `GET /health`                | –                              | `{"status":"ok"}`                           |
| `GET /versions`              | –                              | verfügbare Modellversionen, aktive markiert |
| `POST /best-move`            | `{fen, version?}`              | `{best_move, san, evaluation, engine, version, depth, pv}` |
| `POST /legal-moves`          | `{fen, square}`                | erreichbare Felder dieser Figur             |
| `POST /spielen/zug`          | `{fen, move, version?}`        | Stellung nach beiden Zügen, Status          |
| `POST /recognize`            | Bild, `side`, `flipped`        | erkannte Stellung, Sicherheit, Hinweis      |
| `POST /best-move-from-image` | Bild, `side`, `flipped`        | erkannte Stellung **und** bester Zug        |
| `GET /test/bericht.json`     | –                              | Prüfbericht des letzten Releases            |

Das Feld `engine` sagt, wer den Zug gewählt hat: `rl` oder `material`. Die Skala
von `evaluation` hängt davon ab — Centipawns bei `material`, erwarteter
Partieausgang in `[-1, 1]` bei `rl`. `version` nennt das antwortende Modell,
etwa `v2` oder `endspiel-queen`.

`depth` ist die Vorausschau in Halbzügen, mit der gerechnet wurde, und `pv` die
erwartete Zugfolge (Hauptvariante) in lesbarer Notation, beginnend mit dem
empfohlenen Zug — etwa `["Nc7+", "Kd7"]`. Ab Tiefe 2 ist `evaluation`
der Suchwert, also die Bewertung **nach** dem Vorausrechnen; bei Tiefe 1 bleibt
es wie bisher die Netzbewertung der Stellung selbst. `/spielen/zug` liefert
`depth` und `pv` ebenfalls.

## Oberflächen

| Adresse | Zweck |
|---|---|
| `/` | Stellung von Hand setzen und den besten Zug berechnen lassen. Eine Figur antippen zeigt ihre regelkonformen Züge — inklusive Fesselung und der Regel, dass der König nicht ins Schach ziehen darf. |
| `/spielen` | Eine ganze Partie gegen die gewählte Modellversion spielen, mit Farbwahl, Zugliste und Zurücknehmen. |
| `/test` | Prüfstand: Teststellungen, Turnier aller Versionen und jede Partie mit Stockfish-Bewertung jedes Zuges (siehe [Prüfstand](#prüfstand-test)). |

Der Versionswähler zeigt zu jeder Version, wie viele Halbzüge sie vorausrechnet.
Ab Tiefe 2 erscheint im Ergebnis zusätzlich die **erwartete Folge**, also womit
die Engine nach ihrem Zug rechnet. Das macht nachvollziehbar, warum sie einen
Zug spielt — etwa ein Schach, das erst zwei Halbzüge später einen Turm gewinnt.
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

**Eine Version ist Netz und Suchtiefe.** Jeder Eintrag hat das Feld
`search_depth`: wie viele Halbzüge die Engine mit diesem Netz vorausrechnet.
Erst beides zusammen legt fest, wie eine Version spielt.

| Version | Netz | Vorausschau | Was sie zeigt |
|---|---|---|---|
| v1 | `full_v1.npz` | 1 Halbzug | erster Lauf mit Reward Shaping |
| v2 | `full_v2.npz` | 1 Halbzug | längeres Training, Zufallseröffnungen |
| v3 | `full_v3.npz` | 1 Halbzug | breiteres Netz, längere Partien |
| v4 | `full_v3.npz` | **2 Halbzüge** + Ruhesuche | was die Suche **allein** bringt, ohne neues Training |
| v5 | `full_v5.npz` | **2 Halbzüge** + Ruhesuche | was **Lernen mit Suche** (TD-Leaf) bringt: Netz von v3, 200 Partien mit 4 Halbzügen weitertrainiert |

v1 bis v3 rechnen fest einen Halbzug voraus und spielen damit **genau wie vor
dem Einbau der Suche** — abgesichert durch einen Test gegen eine vorher
aufgenommene Referenz (siehe [Tests](#tests)). Einen freien Tiefe-Regler gibt es
deshalb bewusst nicht: Eine Version soll einen festen, nachvollziehbaren Stand
des Projekts zeigen. v4 verwendet absichtlich dieselbe Netzdatei wie v3; so
trennt der Vergleich v3 gegen v4 sauber, was die Suche beiträgt und was das
Lernen.

Eine neue Version wird eingetragen mit

```bash
python -m Code.rl.registry --register v5 --file full_v5.npz --episodes 3000 \
    --parent v3 --depth 2 --keep-active --notes "..."
```

`--keep-active` trägt nur ein, ohne umzuschalten — aktiviert wird erst, wenn
die Messung zeigt, dass die neue Version besser spielt.

**Aktiv ist seit dem 15. Oktober 2026 v5.** Es schlägt v4 direkt und holt gegen
Stockfish fast doppelt so viele Punkte (siehe
[Lernen mit Vorausrechnen](#lernen-mit-vorausrechnen-v5)). Auf render.com spielt
damit standardmässig v5 mit zwei Halbzügen Vorausschau; v1 bis v4 bleiben über
den Versionswähler erreichbar.

Die Endspielmodelle sind bewusst **nicht** versioniert. Sie lösen eine feste
Aufgabe und ändern sich nicht mehr; bei genau drei Figuren antworten immer sie,
mit einem Halbzug Vorausschau wie im Training.

---

# Prüfstand (`/test`)

Jede Version durchläuft vor dem Release dieselbe Prüfung, und das Ergebnis ist
öffentlich unter `/test` einsehbar. Die Seite beantwortet drei Fragen: Wie gut
löst eine Version taktische Aufgaben? Wie schneidet sie gegen die anderen
Versionen ab? Und wo genau macht sie ihre Fehler?

**Teststellungen.** 50 Stellungen mit genau einem klar besten Zug
(`testbench/stellungen.json`): 15 Matt, 20 Gewinn, 15 Verteidigung, gemischt
aus Eröffnung, Mittelspiel und Endspiel. Sie stammen aus Partien von Stockfish
gegen sich selbst auf Elo 1320 — bewusst nicht aus eigenen Partien, damit keine
Version im Vorteil ist — und wurden von Stockfish ohne Begrenzung mit Tiefe 18
bestätigt: Der beste Zug liegt mindestens 25 Prozentpunkte Gewinnerwartung vor
dem zweitbesten. Die Stellungen wurden einmal erzeugt und werden nie mehr
verändert; jede Version bekommt dieselben.

**Turnier.** Jede Version spielt gegen jede 10 Partien: 5 feste Eröffnungen
(Italienisch, Sizilianisch, Französisch, Damengambit, Englisch), je einmal mit
jeder Farbe. Gezogen wird über dieselbe Funktion wie auf der Webseite
(`main.choose_move`), also mit der Suchtiefe der Version und den
Endspiel-Spezialisten. Abbruch nach 300 Halbzügen, dreifacher Wiederholung oder
der 50-Züge-Regel.

**Zugbewertung wie bei chess.com.** Stockfish 19 analysiert jede Stellung jeder
Partie mit Tiefe 14. Die Bewertung wird in die Gewinnerwartung der Partei am Zug
umgerechnet (0 bis 1, Formel von Lichess), und was ein Zug davon verliert,
bestimmt seine Klasse — mit den Schwellen, die chess.com für sein
Expected-Points-Modell nennt:

| Klasse | Bedingung |
|---|---|
| Brillant `!!` | bester Zug, der eine Figur opfert; danach nicht verloren, vorher nicht ohnehin gewonnen |
| Grossartig `!` | bester Zug, und jeder andere wäre mindestens 20 Punkte schlechter |
| Bester Zug `★` | Stockfishs erste Wahl |
| Ausgezeichnet | Verlust unter 2 Prozentpunkten |
| Gut | unter 5 |
| Ungenauigkeit `?!` | 5 bis 10 |
| Verpasst `✗` | ab 10, obwohl ein Matt möglich war oder der Gegner eben einen Fehler gemacht hatte |
| Fehler `?` | 10 bis 20 |
| Patzer `??` | ab 20 |
| Erzwungen, Eröffnung | nur ein legaler Zug bzw. vorgegebener Eröffnungszug |

Pro Partie und Seite kommt eine Genauigkeit dazu (Formel von Lichess je Zug,
gemittelt). Die exakten Formeln von chess.com sind nicht öffentlich; die
Schwellen sind übernommen, die Umrechnung in Gewinnerwartung und die Genauigkeit
folgen den veröffentlichten Formeln von Lichess.

**Die Seite.** Übersicht mit Rangliste, Kreuztabelle und Legende; eine Tabelle
der Teststellungen (Klick zeigt Stellung, Lösung und die Antwort jeder Version);
die Partienliste mit Filter nach Paarung. Ein Klick auf eine Partie öffnet die
Analyse: Brett mit Bewertungsbalken, jeder Zug mit seiner Klasse, der bessere
Zug als Pfeil, Gewinnerwartung vorher und nachher. Durch die Partie geht es mit
den Pfeiltasten, `Pos1`/`Ende` oder per Klick in die Zugliste. Jede Ansicht hat
eine eigene Adresse (`/test#partie=v5-v4-1&zug=23`), lässt sich also verlinken,
und die Zurück-Taste des Browsers funktioniert.

### Ablauf vor jedem Release

Stockfish läuft nicht auf render.com — er ist über 100 MB gross und viel zu
rechenintensiv für 0.1 CPU. Der Prüfstand rechnet deshalb lokal, und der Dienst
liefert nur den fertigen Bericht aus (`testbench/bericht.json`, gzip-komprimiert).

```bash
python -m Code.rl.testbench                        # vor jedem Release
python -m Code.rl.testbench --generate-positions   # nur einmal, hat schon stattgefunden
```

Gerechnet wird nur, was fehlt. Jedes Ergebnis trägt einen Schlüssel aus dem
Fingerabdruck der Netzdatei und der Suchtiefe; kommt eine Version dazu, laufen
nur ihre 50 Teststellungen und ihre Partien gegen die anderen.

Damit das nicht vergessen geht, gibt es eine **Release-Sperre**: Der Test
`test_pruefbericht_passt_zur_registry` schlägt fehl, sobald eine Version in der
Registry steht, die der Bericht nicht abdeckt — oder deren Netz oder Suchtiefe
sich seit der Prüfung geändert hat. Eine neue Version lässt sich so nicht
unbemerkt ohne Prüfung ausliefern. In der Sprache des DevOps-Zyklus ist der
Prüfstand die Stufe **Test** zwischen Build und Release.

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
Zug ist? Der Zug selbst entsteht dann durch Vorausschau. Bei v1 bis v3 ist das
ein Halbzug — für jeden legalen Zug wird die Folgestellung bewertet und die
beste genommen. Ab v4 rechnet eine Alpha-Beta-Suche mehrere Halbzüge tief
(siehe [Vorausrechnen](#vorausrechnen-alpha-beta-suche)).

Ehrliche Einordnung: Reinforcement Learning ist hier **ausschliesslich** das
Lernen von `V(s)` aus Self-Play. Die Vorausschau ist klassische Suche, denn das
Modell der Umwelt (die Schachregeln) ist bekannt. Mit einem Halbzug hatte die
bisherige Platzhalter-Engine genau diese Struktur auch — ersetzt wurde nur die
handgeschriebene Materialzählung durch eine gelernte Funktion. Wo Suche und
Lernen ineinandergreifen, nämlich beim Training mit TD-Leaf, ist das unten
eigens beschrieben.

| Modul (`Code/rl/`) | Aufgabe |
|---|---|
| `features.py` | Stellung → 774 Merkmale aus Sicht der Zugpartei |
| `valuenet.py` | das Netz `V(s)`, NumPy, Gradient von Hand, Adam, Net2Net |
| `shaping.py` | Φ(s) für das Reward Shaping, statische Abtauschbewertung (SEE) |
| `engine.py` | Zugwahl mit einem Halbzug, Bewertung aller Folgestellungen in einem Netzaufruf |
| `search.py` | Alpha-Beta-Suche mit Ruhesuche und Hauptvariante |
| `positions.py` | Startstellungen: Endspiel-Curriculum, Zufallseröffnungen |
| `train.py` | Self-Play-Training mit TD(λ), ab Tiefe 2 mit TD-Leaf(λ) |
| `tdleaf.py` | eine Trainingspartie mit Suche und die Lernziele für TD-Leaf |
| `parallel.py` | Partien auf mehreren Prozessorkernen |
| `evaluate.py` | Messungen gegen Zufall, Heuristik, Stockfish und andere Versionen |
| `stockfish.py` | Stockfish als Messlatte (nur lokal, nie im Dienst) |
| `registry.py` | Modellversionen: Netz, Suchtiefe, Herkunft, Resultate |
| `testbench.py` | Prüfstand vor jedem Release: Teststellungen, Turnier, Zugbewertung mit Stockfish |

## Aufgabenstellung: Endspiele statt ganzer Partien

Trainiert wird auf **König + Dame gegen König** (`kqk`) und **König + Turm gegen
König** (`krk`). Das ist bewusst gewählt:

- Ein AlphaZero-artiges Training auf dem vollen Brett braucht Rechenleistung, die
  hier nicht zur Verfügung steht.
- Die Endspiele trennen sauber zwischen gelernt und heuristisch: Das Material
  ändert sich nicht, also ist die Materialbilanz als Bewertung **wertlos**. Wer
  hier mattsetzt, hat wirklich etwas gelernt.
- Das Ergebnis ist messbar: Wie oft gelingt das Matt, und wie schnell?

Auf ganzen Partien (`--task full`) wurde danach mit Reward Shaping
weitertrainiert; daraus sind die Versionen v1 bis v3 entstanden (siehe
[Zwischenbelohnungen](#zwischenbelohnungen-für-ganze-partien) und
[Resultate](#resultate)).

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

- **Netz:** 774 → 128 (ReLU) → 1 (tanh), rund 99 000 Parameter, ab v3 mit 512
  Neuronen und rund 397 000 Parametern; reines NumPy mit handgeschriebenem
  Gradienten und Adam. Die 774 Merkmale sind 768 Figurenfelder plus Rochaderechte,
  En passant und Zugzähler; die Endspielmodelle lesen nur die ersten 768.
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

# Duell-Modus: ein Modell mit gewählter Tiefe gegen genau einen Gegner, parallel
python -m Code.rl.evaluate --task full --model models/full_v3.npz --depth 2 \
    --opponent stockfish --games 200 --max-plies 400 --workers 12
python -m Code.rl.evaluate --task full --model models/full_v3.npz --depth 2 \
    --opponent models/full_v3.npz --opponent-depth 1 --games 200 --max-plies 400 --workers 12
```

Im Duell-Modus beginnt jede Partie mit einer Zufallseröffnung (0 bis 8
Halbzüge, ohne Materialverlust), und jede Eröffnung wird zweimal gespielt, je
einmal mit jeder Farbe. Alle Kandidaten bekommen dieselben Eröffnungen. Ohne
das spielen zwei deterministische Engines aus der Grundstellung immer dieselbe
Partie (siehe [Resultate](#ganze-partien)).

Mit `--resume` setzt ein Lauf auf einem vorhandenen Modell auf, statt bei null zu
beginnen. Das Ergebnis wird danach als neue Version eingetragen:

```bash
python -m Code.rl.registry --register v3 --file full_v3.npz --episodes 5000 --parent v2 --notes "..."
```

Für den Stockfish-Vergleich muss die Programmdatei unter `tools/stockfish/`
liegen oder über `STOCKFISH_PATH` erreichbar sein.

Wichtigste Optionen: `--td-lambda`, `--gamma`, `--lr`, `--hidden`, `--eps-start` /
`--eps-end`, `--curriculum-episodes`, `--resume`, `--seed`, `--search-depth`
(Vorausschau beim Spielen, ab 2 mit TD-Leaf) und `--workers` (Partien
gleichzeitig auf mehreren Kernen). Das Training schreibt eine Lernkurve nach
`logs/train_<task>.csv` (pro Episode, mit Rechenzeit) und die
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

Ob Stockfish vorhanden ist, wird **vor der ersten Episode** geprüft. Fehlt die
Programmdatei, bricht das Training sofort mit einer klaren Meldung ab, statt
nach einer Stunde Rechnen bei der ersten Prüfung. Ein ausdrücklich angegebener
Pfad wird dabei nie stillschweigend durch eine andere gefundene Engine ersetzt.
Jeder Prozess öffnet seine Engine einmal und behält sie für den ganzen Lauf.

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

## Vorausrechnen: Alpha-Beta-Suche

Mit einem Halbzug Vorausschau sieht die Engine nur ihren eigenen Zug, nicht die
Antwort darauf. Eine Springergabel, die erst nach dem Ausweichen des Königs eine
Figur gewinnt, bleibt so unsichtbar: In der Stellung
`r3k3/pp3ppp/8/1N6/8/8/PP3PPP/4K3 w` spielt v3 `b4`, v4 dagegen `Sc7+` mit der
erwarteten Folge `Sc7+ Kd7`. Dass danach der Turm auf a8 hängt, erfasst schon
die Abtauschbewertung in Φ an diesem Blatt; mit Tiefe 3 steht `Sxa8` dann auch
ausgeschrieben in der Folge. Die Suche liegt in `Code/rl/search.py`.

**Negamax mit Diskont.** Wie im Training gilt jeder Wert aus Sicht der Partei am
Zug. Ein Zug ist `r` wert, wenn er die Partie beendet, sonst `−gamma · W(s')`, und
`W(s)` ist das Maximum über alle Züge. Mit Tiefe 1 ist das exakt die bisherige
Zugbewertung — die Suche ist deren Verallgemeinerung, keine zweite Logik. Bewertet
werden die Blätter wie bisher mit `V + Φ`.

**Alpha-Beta.** Ein Zug muss nicht fertig gerechnet werden, sobald feststeht, dass
er schlechter ist als ein bereits gefundener. Am Ergebnis ändert das nichts, ein
Test vergleicht dazu mit einer Minimax-Suche ohne jede Abkürzung. Damit früh
abgeschnitten wird, werden die Züge in jedem Knoten nach ihrer Ein-Zug-Bewertung
sortiert.

**Killer-Heuristik.** Gemessen schnitt im letzten Halbzug vor den Blättern in 98
von 105 Knoten schon der **erste** Antwortzug ab — die übrigen rund 30
Folgestellungen wurden umsonst bewertet. Oft ist es derselbe Antwortzug, der eben
einen Nachbarzug widerlegt hat. Er wird deshalb zuerst allein geprüft; erst wenn
er nicht reicht, kommen alle anderen dran. Im Mittelspiel sank die Zahl der
bewerteten Stellungen bei Tiefe 2 dadurch von 1 484 auf 463.

**Ruhesuche.** Eine Suche mit fester Tiefe hat einen blinden Fleck, den
Horizonteffekt: Endet die Rechnung mitten in einem Schlagabtausch, wird eine
Stellung bewertet, in der gerade eine Figur hängt. An den Blättern rechnet die
Suche deshalb weiter, aber nur noch Schlagzüge und Umwandlungen zur Dame, die nach
der Abtauschbewertung kein Material verschenken — höchstens sechs Halbzüge. Die
Partei am Zug darf jederzeit aufhören, wenn kein Schlagzug ihre Lage verbessert.
Bewusst vereinfacht: Auch im Schach darf sie aufhören, statt alle Ausweichzüge zu
prüfen. Das hält die Ruhesuche klein; ein Matt genau am Horizont kann ihr dadurch
entgehen.

**Hauptvariante.** Die Suche liefert die erwartete Zugfolge mit. Sie endet in
der Stellung, deren Bewertung den Suchwert bestimmt hat:

```
Suchwert = (−gamma)^k · Gesamtbewertung(Blatt),   k = Länge der Variante
```

Die Oberfläche zeigt diese Folge an, und das Training mit TD-Leaf lernt genau an
diesem Blatt.

### Schneller rechnen, ohne etwas zu verändern

Beim Vorausrechnen wird jede Folgestellung bewertet, im Mittelspiel mit Tiefe 2
einige hundert pro Zug. Der grösste Posten war Φ, vor allem die Abtauschbewertung.
Φ und die Brettkodierung wurden deshalb neu geschrieben — mit einer harten
Bedingung: **dieselben Werte, Bit für Bit**. Sonst würden die bereits trainierten
Modelle in einzelnen Stellungen anders ziehen.

| Baustein | vorher | nachher | Faktor | Werte |
|---|---|---|---|---|
| Φ an einem Blatt | 0.266 ms | 0.101 ms | 2.6 | identisch auf 611 Stellungen |
| Abtauschbewertung | 0.136 ms | 0.044 ms | 3.1 | identisch |
| Brettkodierung | 0.142 ms | 0.023 ms | 6.3 | identisch auf 3 615 Stellungen |

Woher die Faktoren kommen: Angreifer beider Farben werden in einem Durchgang über
Bitmasken bestimmt statt in zweien, die Abtauschbewertung prüft die Ziele von der
wertvollsten Figur abwärts und hört auf, sobald kein Ziel mehr mehr einbringen
kann, und die Brettkodierung packt die zwölf Figurenebenen mit NumPy in einem
Schritt aus, statt das Brett zu kopieren und zu spiegeln. Eine Falle dabei: Die
Abtauschbewertung schlägt immer mit der billigsten Figur, und Springer und Läufer
sind beide 3 wert. Bei Gleichstand entschied bisher das niedrigste Feld — genau
diese Regel musste erhalten bleiben, denn wer zuerst den Läufer statt des
Springers wegnimmt, öffnet unter Umständen eine andere Röntgenlinie.

Gemessene Rechenzeit pro Zug, lokal (Intel, 14 Kerne, ein Kern genutzt):

| Stellung | Tiefe 1 | Tiefe 2 | Tiefe 3 |
|---|---|---|---|
| Grundstellung | 0.01 s | 0.05 s | 0.52 s |
| Mittelspiel | 0.02 s | 0.13 s | 1.54 s |

Auf render.com (0.1 CPU) ist mit etwa dem Zehnfachen zu rechnen. Tiefe 2 passt
damit in eine Web-Anfrage, Tiefe 3 nicht — daher `MAX_SEARCH_DEPTH=2`.

## Training mit Vorausrechnen: TD-Leaf(λ)

Spielt die Engine mit Suche, bewertet sie gar nicht mehr die Stellung, in der sie
am Zug ist, sondern die Stellungen am Ende ihrer Varianten — meist ruhige
Stellungen nach einem Schlagabtausch. Genau dort sollte das Netz gut sein.
**TD-Leaf(λ)** (Baxter, Tridgell & Weaver 1998, Schachprogramm KnightCap) lernt
deshalb nicht an `s_t`, sondern am Blatt der Hauptvariante (`Code/rl/tdleaf.py`):

1. In jeder Stellung `s_t` rechnet die Suche: Suchwert `W_t`, Hauptvariante,
   Blatt.
2. Gezogen wird der beste Zug, mit Wahrscheinlichkeit ε ein zufälliger.
3. Ein-Schritt-Ziel wie bisher, nur mit Suchwerten: `Ziel_t = r` am Partieende,
   sonst `−gamma · W_(t+1)`. Bei Abbruch am Zuglimit liefert eine letzte Suche
   den Wert zum Bootstrappen.
4. λ-Return mit derselben Rekursion wie bisher.
5. Zurückgerechnet aufs Blatt: `V(Blatt_t)` soll `G_t / (−gamma)^k − Φ(Blatt_t)`
   werden. Endet die Variante im Matt oder Remis, ist der Suchwert exakt, und
   das Blatt fällt aus dem Lernbatch.

Für Tiefe 1 bleibt das Training beim bisherigen TD(λ). So bleibt das Verfahren
hinter v1 bis v3 jederzeit nachvollziehbar.

### Partien parallel auf allen Kernen

Mit Vorausrechnen dauert eine Trainingspartie nicht mehr Sekundenbruchteile,
sondern im Mittel 37 Sekunden. 3 000 Partien wären auf einem Kern über 30
Stunden. Der Rechner hat aber 14 Kerne, von denen das Training bisher genau einen
nutzte. `--workers N` verteilt die Partien auf N Prozesse (`Code/rl/parallel.py`):

- Bis zu N Partien laufen gleichzeitig, jede mit den Gewichten zum Zeitpunkt
  ihres Starts. Ist eine fertig, lernt der Hauptprozess daraus und startet sofort
  die nächste, schon mit den neuen Gewichten. Eine Partie spielt dadurch mit
  Gewichten, die höchstens N−1 Lernschritte alt sind — bei Tausenden Partien und
  kleiner Lernrate unerheblich; dasselbe Prinzip nutzen A3C und IMPALA.
- Prozesse statt Threads, weil Python wegen des Global Interpreter Lock nur einen
  Thread gleichzeitig rechnen lässt.
- Jeder Prozess rechnet NumPy mit **einem** Thread. Sonst starten zwölf Prozesse
  je zwanzig Rechenthreads und behindern sich gegenseitig.
- Vor jeder Zwischenauswertung wartet das Training, bis alle laufenden Partien
  fertig sind. Die Auswertungspartien laufen ebenfalls parallel, jeder Prozess
  mit seiner eigenen Stockfish-Instanz.
- Der Preis: Die Reihenfolge der Lernschritte hängt von der Rechenzeit ab, ein
  Lauf ist nicht mehr exakt wiederholbar. Mit `--workers 1` läuft alles im
  Hauptprozess und bleibt es.

Ein Fehler, der dabei auffiel und für jeden Einsatz von python-chess mit Prozessen
lehrreich ist: Der erste Probelauf blieb am Ende stehen. python-chess betreibt
jede Engine in einem eigenen Thread, und Python wartet beim Beenden auf alle
Threads, **bevor** die `atexit`-Aufräumfunktionen laufen. Die Stockfish-Instanzen
wurden also nie geschlossen, und die Prozesse warteten ewig auf sich selbst.
Gelöst mit `multiprocessing.util.Finalize`, das in einem Kindprozess vor diesem
Warten läuft.

**Probelauf** mit 24 Partien, Tiefe 2, 12 Prozessen, ausgehend von v3: **0.47
Partien pro Sekunde**, eine Partie zwischen 18 und 62 Sekunden, Länge 48 bis 400
Halbzüge, zwei Drittel enden mit Matt. Hochgerechnet dauern 3 000 Partien damit
knapp **zwei Stunden** — eine Nacht reicht auch für deutlich mehr.

### v5: wenige Partien, tiefe Suche

Zur Wahl standen zwei Wege mit ähnlichem Rechenaufwand:

| Variante | Episoden | Vorausschau im Training | Dauer auf 12 Kernen |
|---|---|---|---|
| A | 3 000 | 2 Halbzüge | ca. 2 h |
| **C** | **200** | **4 Halbzüge** | 5.1 h, in drei Sitzungen gerechnet |

Gewählt wurde C, nach einer Diskussion mit dem Dozenten: lieber wenige Partien,
in denen jedes Lernziel aus einer tiefen Suche stammt, als viele mit flacher
Suche. Genau so hat KnightCap mit TD-Leaf in wenigen hundert Partien gelernt.
Noch tiefer geht nicht: Gemessen dauert ein Zug im Mittelspiel mit Tiefe 4 rund
16 Sekunden, mit Tiefe 5 schon 93 Sekunden; jeder weitere Doppelschritt kostet
etwa das Hundert- bis Zweihundertfache. Acht Halbzüge, wie zuerst überlegt,
wären mehrere Tage pro Zug.

```bash
python -m Code.rl.train --task full --resume models/full_v3.npz \
    --search-depth 4 --workers 12 --episodes 200 --max-plies 400 \
    --eps-start 0.1 --eps-end 0.05 --lr 5e-4 \
    --eval-every 0 --snapshot-every 25 --log-every 1 \
    --run-dir runs/v5 --out models/full_v5.npz
```

Die Wahl der Einstellungen:

- **Lernrate 5·10⁻⁴**, halb so gross wie bei v1 bis v3. Kleiner, weil es eine
  Feinabstimmung eines brauchbaren Netzes ist; aber nicht zu klein, weil 200
  Partien nur 200 Lernschritte sind.
- **Wenig Zufall** (ε von 0.1 auf 0.05): Das Netz spielt schon brauchbar, die
  Vielfalt kommt aus den Zufallseröffnungen.
- **Keine Zwischenprüfung während des Trainings.** Mit Tiefe 4 dauerte eine
  einzige Prüfung über 100 Partien gegen Stockfish über eine Stunde. Stattdessen
  legt der Lauf alle 25 Partien einen Zwischenstand ab, und die werden danach im
  Duell-Modus mit je 200 Partien gemessen. Das gibt eine genauere Lernkurve und
  bremst das Training nicht.
- Gespielt wird v5 später mit **Tiefe 2** — auf Render geht nicht mehr. Das
  passt: Das Netz lernt, Stellungen zu bewerten, unabhängig davon, mit welcher
  Tiefe es danach eingesetzt wird.

### Unterbrechen und fortsetzen

Ein Lauf über mehrere Stunden soll sich auf mehrere Sitzungen verteilen lassen —
etwa zwei Stunden heute und vier morgen. Dafür gibt es die **Lauf-Mappe**
(`--run-dir`):

| Datei | Inhalt |
|---|---|
| `state.json` | Einstellungen des Laufs, Fortschritt, Zahl der Sitzungen |
| `checkpoint.npz` | Gewichte **samt Zustand des Optimierers** nach der letzten fertigen Partie |
| `train.csv` | Protokoll je Partie mit Sitzungsnummer, wird beim Fortsetzen ergänzt |
| `snapshots/ep0025.npz` … | Zwischenstände für die Lernkurve |

- **Fortsetzen:** derselbe Befehl noch einmal. Liegt in der Mappe ein Stand,
  geht es dort weiter, mit den gespeicherten Einstellungen. Nur `--workers` darf
  sich ändern, etwa wenn der Rechner nebenher gebraucht wird.
- **Anhalten:** `python -m Code.rl.train --run-dir runs/v5 --stop`, eine Datei
  `STOP` (oder `STOP.txt`) in der Mappe oder Ctrl+C. Das Training sichert und
  beendet sich innerhalb weniger Sekunden.
- **Absturz, Neustart, leerer Akku:** Gesichert wird nach jeder fertigen Partie,
  verloren gehen also nur die Partien, die gerade liefen.

Zwei Details, die das zuverlässig machen. Der Zustand von Adam wird
mitgesichert; ohne ihn liefe der Optimierer nach jeder Pause neu an und das
Training anders als ohne Unterbrechung. Und jede Datei wird zuerst unter einem
Hilfsnamen geschrieben und dann in einem Schritt umbenannt — bricht der Rechner
mitten im Schreiben ab, bleibt der vorige Stand unversehrt. Ein Test spielt
Anhalten und Fortsetzen im Kleinen durch (`tests/test_training.py`).

Nach Abschluss liegt das Netz unter `models/full_v5.npz` und wird als v5 mit
`--depth 2 --keep-active` eingetragen. Aktiviert wird es erst, wenn die Messung
gegen v4 zeigt, dass es besser spielt.

**Wie es tatsächlich lief:** drei Sitzungen über zwei Tage. Die erste wurde nach
79 Partien wie geplant mit `--stop` angehalten. Die zweite endete ungeplant nach
Partie 140, weil der Terminal-Tab neu gestartet wurde und den Prozess mitnahm —
ohne Stopp-Anforderung, also der Ernstfall. Der Stand nach Partie 140 war
trotzdem vollständig und lesbar; verloren gingen nur die zwölf Partien, die
gerade liefen. Die dritte Sitzung lief bis Partie 200 durch. Von den 200
Partien endeten 176 mit Matt und 24 remis, im Mittel nach 116 Halbzügen. Mit
sinkendem ε wurden die Partien länger (anfangs rund 65, gegen Ende rund 130
Halbzüge), weil weniger zufällige Patzer sofort bestraft werden.

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

Alle Paarungen nach demselben Protokoll (Duell-Modus von `Code.rl.evaluate`):
200 Partien, 100 Zufallseröffnungen mit 0 bis 8 Halbzügen ohne Materialverlust,
jede Eröffnung mit beiden Farben gespielt, Abbruch nach 400 Halbzügen. Die
Punktewertung zählt ein Remis als halben Sieg. Bei 200 Partien liegt die
statistische Unsicherheit der Punkte bei etwa ±3 Prozentpunkten (eine
Standardabweichung). Gemessen am 14. Oktober 2026.

| Paarung | Sieg | Remis | Niederlage | Punkte |
|---|---|---|---|---|
| v2 gegen v1 | 50.5 % | 36.0 % | 13.5 % | 68.5 % |
| v3 gegen v2 | 36.0 % | 35.5 % | 28.5 % | 53.8 % |
| **v4 gegen v3** | **97.5 %** | 2.5 % | **0 %** | **98.8 %** |
| v3 gegen Material-Heuristik | 96.0 % | 4.0 % | 0 % | 98.0 % |
| nur Φ, untrainiert, gegen Material | 85.5 % | 14.5 % | 0 % | 92.8 % |
| v1 gegen Stockfish (Elo 1320) | 8.5 % | 8.0 % | 83.5 % | 12.5 % |
| v2 gegen Stockfish (Elo 1320) | 14.5 % | 13.0 % | 72.5 % | 21.0 % |
| v3 gegen Stockfish (Elo 1320) | 13.5 % | 11.0 % | 75.5 % | 19.0 % |
| **v4 gegen Stockfish (Elo 1320)** | 15.5 % | 11.5 % | 73.0 % | **21.2 %** |

Was sich daraus ablesen lässt:

**v2 ist klar besser als v1, v3 kaum besser als v2.** 68.5 Punkteprozent für v2
gegen v1 sind ein deutlicher Fortschritt. 53.8 Prozent für v3 gegen v2 liegen nur
knapp über dem Gleichstand, und gegen Stockfish sind v2 und v3 im Rahmen der
Messunsicherheit gleich stark (21.0 gegen 19.0). Das breitere Netz und die
längeren Partien haben v3 also nicht messbar stärker gemacht.

**Vorausrechnen schlägt dasselbe Netz ohne Vorausrechnen fast immer.** v4 gewinnt
97.5 Prozent der Partien gegen v3 — mehr dazu unten in
[Suche allein](#suche-allein-v3-gegen-v4).

**Der TD-Fehler taugt nicht als Fortschrittsmass.** Bei v3 blieb er über alle
3 000 Episoden flach bei 0.11 und lag damit über dem Endwert von v2 (0.099). Er
misst, wie gut der Agent seine **eigenen** Partien vorhersagt; werden die
Partien länger und schwerer, kann er steigen, ohne dass die Spielstärke sinkt.
Nur der direkte Vergleich zählt.

**Ein grosser Teil der Spielstärke ist handgeschrieben.** Ein völlig
untrainiertes Netz schlägt mit Φ allein die Material-Heuristik bereits mit 92.8
Punkteprozent — fast so deutlich wie v3 mit 98.0. Gegen die Heuristik misst man
damit vor allem Φ. Was das Lernen beiträgt, zeigt erst ein stärkerer Gegner:
Gegen Stockfish liegt v2 mit 21 Punkteprozent deutlich vor v1 mit 12.5.

#### Korrektur der ersten Messungen

Die erste Fassung dieser Tabelle meldete «v3 gegen v2: 100 Prozent» und schloss
daraus, jede Version schlage ihren Vorgänger klar. Gemessen wurde damals aus der
Grundstellung, 25 Partien × 2 Seeds. Unsere Engines sind aber deterministisch:
Aus derselben Stellung spielen sie immer denselben Zug. Zwischen zwei eigenen
Versionen gab es deshalb nur **zwei verschiedene Partien**, eine mit jeder
Farbe, und die wurden je 25-mal wiederholt. Gewinnt v3 diese beiden, stehen 100
Prozent da. Mit Zufallseröffnungen sind es 53.8 Prozent.

Gegen Stockfish war der Fehler kleiner, weil dessen Elo-Begrenzung zufällige
Fehler einbaut und die Partien dadurch auseinanderlaufen. Direkt vergleichbar
sind die alten Werte trotzdem nicht: andere Startstellungen, weniger Partien und
ein Abbruch schon nach 160 Halbzügen, der als Remis zählt. Zur
Nachvollziehbarkeit die erste Messung im Original (in `models/registry.json` mit
dem Zusatz `_grundstellung`):

| Paarung (erste Messung, aus der Grundstellung) | Sieg | Remis | Niederlage | Punkte |
|---|---|---|---|---|
| v3 gegen v2 | 100 % | 0 % | 0 % | 100 % |
| v2 gegen v1 | 52 % | 48 % | 0 % | 76 % |
| v3 gegen Material-Heuristik | 100 % | 0 % | 0 % | 100 % |
| v3 gegen Stockfish (Elo 1320) | 24 % | 10 % | 66 % | 29 % |
| v2 gegen Stockfish (Elo 1320) | 16 % | 22 % | 62 % | 27 % |
| v1 gegen Stockfish (Elo 1320) | 12 % | 20 % | 68 % | 22 % |
| Material-Heuristik gegen Stockfish | 0 % | 6 % | 94 % | 3 % |
| nur Φ, untrainiert, gegen Material | 50 % | 50 % | 0 % | 75 % |

Die Lehre daraus: Wer deterministische Engines vergleicht, braucht verschiedene
Startstellungen, sonst zählt er Wiederholungen statt Partien.

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
| v4 | – | Netz von v3, rechnet 2 Halbzüge voraus | – | – |
| **v5** | 200 | 512 Neuronen, TD-Leaf mit 4 Halbzügen | 400 Halbzüge | 5.1 h auf 12 Kernen |

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

### Suche allein: v3 gegen v4

v4 ist Bit für Bit das Netz von v3, rechnet aber zwei Halbzüge voraus statt
einem, plus Ruhesuche. Jeder Unterschied zwischen den beiden geht also allein
auf die Suche zurück — kein neues Training, keine andere Bewertung.

| Messung | v3 (1 Halbzug) | v4 (2 Halbzüge) |
|---|---|---|
| direkt gegeneinander, 200 Partien | 1.2 Punkteprozent | **98.8 Punkteprozent**, keine Niederlage |
| gegen Stockfish (Elo 1320), 200 Partien | 19.0 Punkteprozent | 21.2 Punkteprozent |
| Taktik-Test, 150 Stellungen mit genau einem klar besten Zug | 64.0 % gelöst | **71.3 % gelöst** |
| Rechenzeit pro Zug im Mittelspiel, lokal | 0.02 s | 0.13 s |

Zwei Ergebnisse, die auf den ersten Blick nicht zusammenpassen. Gegen v3 ist v4
erdrückend überlegen: v3 verliert fast jede Partie an einer kurzen Taktik — einer
Gabel, einer hängenden Figur, einem Schlag, dessen Antwort es nicht sieht. Genau
das deckt der zweite Halbzug auf. Gegen Stockfish dagegen bringt die Suche nur
gut zwei Punkte, und das liegt innerhalb der Messunsicherheit. Stockfish macht
diese kurzen Fehler auch bei Elo 1320 selten; dort entscheiden Pläne und
Taktiken, die tiefer liegen als zwei Halbzüge, und die Qualität der Bewertung
selbst.

**Der Taktik-Test** (`tests/fixtures/taktikstellungen.json`) prüft das gezielt.
Die 150 Stellungen stammen aus Partien von v3 und wurden von Stockfish 19 ohne
Begrenzung bewertet (Tiefe 12, die zwei besten Züge). Aufgenommen wurde eine
Stellung nur, wenn der beste Zug mindestens drei Bauern besser ist als der
zweitbeste. Gezählt wird, wie oft die Engine genau diesen Zug spielt: v1 62.0,
v2 63.3, v3 64.0, v4 **71.3** Prozent. Die drei trainierten Versionen liegen
fast gleichauf, die zweite Ebene der Suche bringt sieben Punkte. Dass es nicht
mehr sind, liegt an der Zusammensetzung: Viele dieser Stellungen löst schon die
Abtauschbewertung in Φ mit einem Halbzug, etwa das Schlagen einer hängenden
Figur. Einschränkung: Weil die Stellungen aus Partien von v3 stammen, sind sie
auf dessen typische Fehler zugeschnitten.

```bash
python -m Code.rl.evaluate --task full --model models/full_v3.npz --depth 2 \
    --tactics tests/fixtures/taktikstellungen.json
```

Daraus folgte der nächste Schritt. Das Netz von v3 wurde nur dafür trainiert,
Stellungen mit einem Halbzug Vorausschau zu beurteilen. Ob es mit TD-Leaf lernt,
die Blätter der Suche besser zu bewerten, und ob das gegen Stockfish etwas
bringt, zeigt v5.

### Lernen mit Vorausrechnen: v5

v5 ist das Netz von v3, weitertrainiert mit TD-Leaf: 200 Partien mit 4
Halbzügen Vorausschau, 5.1 Stunden auf 12 Kernen, verteilt auf drei Sitzungen
(siehe [v5: wenige Partien, tiefe Suche](#v5-wenige-partien-tiefe-suche)).
Gespielt und gemessen wird es mit 2 Halbzügen, also mit derselben Suche wie v4.
Jeder Unterschied zwischen v4 und v5 geht damit allein auf das Training zurück.

| Messung (200 Partien, Protokoll wie oben) | v4 | **v5** |
|---|---|---|
| gegen Stockfish (Elo 1320) | 21.2 Punkteprozent | **38.8 Punkteprozent** |
| direkt gegeneinander | 44.8 Punkteprozent | **55.2 Punkteprozent** |
| Taktik-Test | 71.3 % gelöst | 73.3 % gelöst |

Gegen Stockfish fast eine Verdoppelung: 27.5 Prozent Siege, 22.5 Prozent Remis,
50 Prozent Niederlagen — v5 verliert damit erstmals nur noch die Hälfte der
Partien. Das ist mit Abstand der grösste Sprung im ganzen Projekt, grösser als
alle Trainingsläufe von v1 bis v3 zusammen, und das mit nur 200 Partien.

**Die Lernkurve** — jeder Zwischenstand mit 2 Halbzügen gegen Stockfish, je 200
Partien:

| nach Partie | 0 (= v4) | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 (= v5) |
|---|---|---|---|---|---|---|---|---|---|
| Punkteprozent | 21.2 | 28.2 | 34.8 | 35.8 | 37.2 | 32.5 | 39.0 | 41.5 | 38.8 |

Der grösste Teil des Gewinns kommt in den ersten 50 Partien. Danach steigt die
Kurve nur noch langsam, die Ausschläge (32.5 bei 125, 41.5 bei 175) liegen im
Rahmen der Messunsicherheit von etwa ±3.4 Punkten. Als v5 gilt bewusst der
Endstand nach 200 Partien und nicht der beste Zwischenstand: Wer den besten aus
neun Messungen auswählt, wählt zum Teil den, bei dem der Zufall am günstigsten
war.

**Netz und Suche gehören zusammen.** Die aufschlussreichste Messung ist die
Gegenprobe mit nur einem Halbzug Vorausschau:

| gegen Stockfish | 1 Halbzug | 2 Halbzüge |
|---|---|---|
| Netz von v3 | 19.0 % | 21.2 % (v4) |
| Netz von v5 | 18.8 % | **38.8 %** (v5) |

Mit einem Halbzug ist v5 kein bisschen stärker als v3 — auch im Taktik-Test
nicht (64.0 Prozent, genau wie v3). Das Training hat das Netz also nicht
allgemein verbessert. Es hat gelernt, genau die Stellungen zu bewerten, bei
denen die Suche ankommt: ruhige Stellungen nach einem Schlagabtausch, einige
Halbzüge in der Zukunft. Erst zusammen mit der Suche wirkt das. Das ist der Kern
von TD-Leaf, und hier ist er direkt messbar.

Die Taktik bleibt dagegen Sache der Suche: Die Stellungen mit einem kurzen,
klaren Gewinnzug löst v5 kaum öfter als v4 (73.3 gegen 71.3 Prozent).

**Zur Frage aus der Diskussion mit dem Dozenten** — viele Partien mit flacher
Suche oder wenige mit tiefer? Gerechnet wurde nur Variante C. Dass 200 Partien
einen so grossen Sprung bringen, während 3 000 Partien ohne Suche von v2 zu v3
gegen Stockfish nichts Messbares brachten, spricht deutlich für die tiefe Suche
im Training. Ein sauberer Vergleich bräuchte zusätzlich Variante A (3 000
Partien mit 2 Halbzügen, rund 2 Stunden).

## Grenzen und nächste Schritte

- Gegen Stockfish (Elo 1320) holt auch v5 erst 39 Punkteprozent und verliert
  die Hälfte der Partien. Für ein Netz mit 397 000 Parametern und zwei Halbzügen
  Vorausschau ist das zu erwarten und wird hier bewusst so ausgewiesen.
- Die Stärke von v5 hängt an der Suche: Mit nur einem Halbzug spielt das Netz
  nicht besser als v3 (siehe [v5](#lernen-mit-vorausrechnen-v5)). Auf render.com
  ist die Tiefe auf 2 begrenzt; trainiert wurde mit 4.
- Der Vergleich «wenige Partien, tiefe Suche» gegen «viele Partien, flache
  Suche» ist nur zur Hälfte gerechnet (Variante C, nicht A).
- Die Ruhesuche darf auch im Schach stehen bleiben. Ein Matt genau am Horizont
  kann ihr dadurch entgehen.
- Die Material-Heuristik kommt nur noch zum Zug, wenn gar kein Modell zuständig
  ist. Sie bleibt trotzdem im Code, weil erst sie die Resultate einordnet.
- Beim Shaping ist ein Teil der Spielstärke handgeschrieben (Material, Zentrum,
  Abtausch, Schach); `V` lernt nur noch die Korrektur darüber hinaus. Für die
  Endspielmodelle gilt das nicht — die sind ohne Shaping trainiert.
- Training mit mehreren Prozessen ist nicht exakt wiederholbar, weil die
  Reihenfolge der Lernschritte von der Rechenzeit abhängt (`--workers 1` ist es).
- Naheliegende Ausbauschritte: gemeinsames Modell für mehrere Endspiele,
  Gymnasium-Wrapper für Vergleiche mit DQN/PPO aus stable-baselines3, und
  Lernkurven-Plots aus den CSV-Dateien für die Dokumentation.
