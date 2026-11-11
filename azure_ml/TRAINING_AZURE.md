# Plan: Die Schach-Engine auf Azure ML trainieren

Stand 11.11.2026. Probelauf und drei Hauptläufe auf Azure gelaufen und
gemessen, Ergebnisse unten.

**Ziel:** Genau das Training, das schon lokal lief, einmal auf Azure ML rechnen
lassen. Danach vergleichen: Wie viel Aufwand ist es? Wie lange dauert es? Was
kostet es? Kommt dasselbe heraus?

## Was trainiert wird

- **Startnetz:** `models/full_v5.npz`. Das ist das Netz der neuesten Engine v6
  (v6 = Netz von v5 + korrigierte Suche).
- **Rezept:** dasselbe wie beim lokalen Testlauf v7b, Zeichen für Zeichen:

  ```bash
  python -m Code.rl.train --task full --resume models/full_v5.npz \
      --search-depth 4 --repetition-draw --workers <Kerne> --episodes 50 \
      --max-plies 400 --eps-start 0.1 --eps-end 0.05 --lr 5e-4 \
      --eval-every 0 --snapshot-every 25 --log-every 1 \
      --run-dir outputs/run --out outputs/full_v7b_azure.npz
  ```

  Einziger Unterschied: Alles, was entsteht, landet unter `outputs/`. Diesen
  Ordner lädt Azure nach dem Lauf automatisch hoch.
- **Warum genau v7b:** Es ist der kürzeste Lauf, zu dem es lokale Messwerte
  gibt. Mit einem neuen Rezept wüsste man am Ende nicht, ob ein Unterschied
  von Azure kommt oder vom Rezept.

Lokale Werte von v7b zum Vergleich: 50 Partien in 67 Minuten mit 12
Prozessen. Das sind rund 12.75 Kernstunden, im Mittel 15 Minuten pro Partie
auf einem Kern. Ergebnis: 34.8 % der Punkte gegen v6, 44.2 % gegen Stockfish
1320.

## So läuft es auf Azure

| Frage | Entscheidung | Warum |
|---|---|---|
| Art des Auftrags | Befehlsauftrag (*command job*), kein AutoML | Unser eigenes Training, Azure stellt nur den Rechner |
| Was hochgeladen wird | nur `Code/` und `models/full_v5.npz`, rund 2 MB | `.venv`, `runs/` und `tools/` braucht Azure nicht |
| Umgebung | fertige Python-Umgebung von Microsoft, beim Start `pip install numpy==2.1.3 chess==1.11.1` | Azure baut so kein eigenes Image und legt keine Container-Registry an, die jeden Tag etwas kostet |
| Rechner | serverlos, 1 Instanz, nur CPU | Nichts bleibt stehen: Ist der Auftrag fertig, ist der Rechner weg. Das Netz ist klein, eine Grafikkarte bringt nichts |
| Grösse | 16 Kerne (z.B. F16s_v2), falls das Kontingent reicht, sonst DS3_v2 mit 4 Kernen | DS3_v2 lief schon bei AutoML. `--workers` = Anzahl Kerne |
| Notbremse | feste Höchstlaufzeit im Auftrag: 3 h bei 16 Kernen, 6 h bei 4 Kernen | Hängt etwas, bricht Azure von selbst ab |
| Start | kleines Skript `azure_ml/training/starte_training.py` mit dem Azure-ML-SDK (`azure-ai-ml`), Anmeldung im Browser | Wiederholbar und im Repo festgehalten. Den Weg per Klick im Studio notiere ich zum Vergleich |

Was am Code nicht geändert werden muss: `keep_awake` tut unter Linux nichts,
und die Prozesse starten mit `spawn`, das geht auch dort. Die BLAS-Threads sind
in `parallel.py` schon auf 1 gesetzt.

## Schritte

1. **Vorbereiten, lokal (rund 1 h):** Das Startskript schreiben und einen
   lokalen Kurzlauf mit 1 Partie machen. Er prüft, dass alles unter
   `outputs/` landet.
2. **Kontingent und Preis prüfen:** Im Studio unter *Kontingent* nachsehen,
   welche VM-Familien wie viele Kerne haben. Den Preis pro Stunde im Dialog
   ablesen. Davor die PIM-Rolle «Mitwirkender» aktivieren.
3. **Probelauf auf Azure:** 2 Partien mit Tiefe 2 auf kleiner VM,
   Höchstlaufzeit 30 Minuten. Er prüft Anmeldung, Upload, Umgebung und den
   Download der Ausgaben. Kosten: ein paar Rappen.
4. **Hauptlauf:** 50 Partien wie oben. Erwartet werden rund 1 bis 1.5 h mit
   16 Kernen oder 3.5 bis 4.5 h mit 4 Kernen. Mitlesen geht im Studio unter
   *Ausgaben + Protokolle*.
5. **Herunterladen und messen, lokal:** Gemessen wird genau wie bei v7b, je
   200 Partien gegen v6 und gegen Stockfish 1320.
6. **Vergleich aufschreiben**, Tabelle unten. Das ist gleich Stoff für den
   Bericht in AI Operations.
7. **Aufräumen:** Prüfen, dass kein Rechner und kein Endpoint läuft. Nach 24
   bis 48 Stunden einen Screenshot der Kostenanalyse machen.

## Geprüft am 10.11.2026

- **Kontingent** im Abo `d-sub-hsw`, Switzerland North: 998 Kerne frei,
  davon 300 für die F-Serie (FSv2). 16 Kerne gehen also problemlos. Die 2
  belegten Kerne der DAv4-Familie stammen nicht von uns, unser Endpoint ist
  leer. Das Kontingent gilt fürs ganze Abo, also auch für andere
  Ressourcengruppen.
- **Preise** laut öffentlicher Azure-Preisliste in CHF pro Stunde (Linux,
  normale Priorität): DS3_v2 (4 Kerne) 0.25, F8s_v2 (8) 0.35, **F16s_v2
  (16) 0.71**, D16as_v4 (16) 0.91. F16s_v2 ist pro Kern am günstigsten und
  am schnellsten. Damit wird es die VM für Probe- und Hauptlauf.
- **Lokaler Kurzlauf** mit genau dem Paket, das hochgeladen wird (1.6 MB):
  2 Partien in 25 Sekunden, Netz und Protokolle landen unter `outputs/`.
- **Probelauf auf Azure** (`brave_beet_pbj16rhr6y`, F16s_v2, 2 Partien,
  Tiefe 2): abgeschickt um 14:27, nach 1.4 Minuten lief er, nach 2.8
  Minuten war er fertig. Für das Anwerfen von Rechner und Umgebung
  brauchte Azure also nur gut eine Minute. Kosten: rund 2 Rappen.
  Heruntergeladen: das Netz `full_probe.npz`, die Lauf-Mappe mit
  `train.csv` und das Protokoll `std_log.txt`. Die 2 Partien dauerten auf
  Azure 5 und 7 Sekunden. Das Netz hat sich wie erwartet nur leicht
  verändert (grösste Änderung eines Gewichts 0.001). `pip` meldet einen
  Konflikt mit dem vorinstallierten `scipy`, das stört nicht, weil der
  Code `scipy` nicht braucht.
- **Hauptlauf auf Azure** (`careful_plate_10bk0hr20y`, F16s_v2, 16
  Prozesse): abgeschickt um 19:36, nach 3.1 Minuten lief er, nach 32.3
  Minuten war er fertig und heruntergeladen. Reine Trainingszeit 27.9
  Minuten statt 67 lokal. Zusammengezählt rechneten die Partien 6.7
  Kernstunden statt 12.75 lokal: Ein Azure-Kern ist für dieses Training
  also fast doppelt so schnell wie ein Prozess auf dem Laptop. Kosten: rund
  0.5 h × 0.71 CHF, also etwa 35 Rappen.
- **Gleiche Partien:** Die ersten 2 Partien verliefen auf Azure genau wie
  lokal (gleiche Länge, gleicher Ausgang, gleicher Lernfehler). Danach
  gehen die Läufe auseinander, weil die Partien in anderer Reihenfolge
  fertig werden und das Netz in anderer Reihenfolge lernt. Ergebnis: 45
  Matts und 5 Remis (lokal 47 und 3), im Mittel 124 Halbzüge (lokal 104).
- **Stolperstein:** Die PIM-Rolle lief ab, bevor die Ausgaben
  heruntergeladen waren. Ohne Rolle lässt Azure nicht einmal den Status
  lesen. Vor dem Herunterladen also immer prüfen, ob die Rolle noch gilt.

## Was verglichen wird

| | lokal (v7b, 28.10.) | Azure ML |
|---|---|---|
| Aufwand bis zum ersten Lauf | keiner, lief schon | ein halber Tag: Startskript, Anmeldung, PIM-Rolle, Kontingent und Preise prüfen, Probelauf |
| Laufzeit 50 Partien | 67 min mit 12 Prozessen | 28 min mit 16 Prozessen, dazu 3 min Warten auf den Rechner |
| Kosten | Strom, Laptop über eine Stunde blockiert | rund 35 Rappen, Probelauf 2 Rappen, Laptop frei |
| Punkte gegen v6 | 34.8 % (nach 25 Partien 38.5 %) | 53.0 % (nach 25 Partien 52.2 %) |
| Punkte gegen Stockfish 1320 | 44.2 % | 45.0 % |
| Stolpersteine | Standby über Nacht (seither `keep_awake`) | PIM-Rolle läuft nach 4 h ab, dann gehen weder Status noch Download. `pip` warnt wegen `scipy` (harmlos) |
| Mitlesen, Abbrechen | Terminal, STOP-Datei | Protokolle im Studio, Knopf «Abbrechen» |

Gleich wird das Ergebnis nicht sein. Mit mehreren Prozessen ist das Training
nicht exakt wiederholbar, und eine Messung mit 200 Partien schwankt um rund
3.5 Punkte. Liegt Azure innerhalb von etwa 5 Punkten der lokalen Werte,
gilt das als «dasselbe Ergebnis».

## Ergebnis vom 10.11.2026

- **Gegen Stockfish dasselbe Ergebnis:** 45.0 % auf Azure, 44.2 % lokal.
  Das liegt klar innerhalb der Schwankung.
- **Gegen v6 nicht:** 53.0 % auf Azure, 34.8 % lokal. 18 Punkte sind weit
  mehr als die Schwankung von rund 3.5 Punkten. Beide Netze haben sich
  gleich weit vom Startnetz entfernt (Abstand je 1.32), aber in
  verschiedene Richtungen (Abstand zueinander 1.58). Das Rezept streut
  also stark von Lauf zu Lauf. Das Urteil «v7b ist schwächer als v6» hängt
  damit zum Teil am Zufall dieses einen Laufs.
- **Besser als v6 ist auch das Azure-Netz nicht:** Gegen v6 ist es gleich
  stark (53 % liegt in der Schwankung um 50 %), gegen Stockfish etwas
  schwächer (45.0 % statt 48.0 %). v6 bleibt aktiv.
- **Fazit zu Azure:** Das bestehende Training liess sich ohne Änderung am
  Code auf Azure rechnen. Es lief mehr als doppelt so schnell und kostete
  unter 50 Rappen, und der Laptop blieb frei. Der Aufwand lag fast nur im
  Drumherum: PIM-Rolle, Kontingent, Anmeldung und Startskript. Ist das
  einmal eingerichtet, ist ein weiterer Lauf ein einziger Befehl.

## Streuung: drei Läufe auf Azure (11.11.2026)

Weil der erste Azure-Lauf gegen v6 so anders abschnitt als der lokale,
liefen zwei weitere Läufe mit anderem Zufall (`--seed 1` und `--seed 2`)
gleichzeitig auf je einer F16s_v2. Sonst war alles gleich.

| Lauf | Trainingszeit | gegen v6 nach 25 | gegen v6 nach 50 | gegen Stockfish 1320 |
|---|---|---|---|---|
| lokal, Seed 0 (v7b) | 67 min | 38.5 % | 34.8 % | 44.2 % |
| Azure, Seed 0 | 28 min | 52.2 % | 53.0 % | 45.0 % |
| Azure, Seed 1 | 39 min | 45.2 % | 50.2 % | 44.0 % |
| Azure, Seed 2 | 35 min | 61.0 % | 48.8 % | 48.0 % |
| v6 zum Vergleich | | 50 % | 50 % | 48.0 % |

- **Alle drei Azure-Läufe** landen nach 50 Partien gegen v6 bei rund 50 %
  (48.8 bis 53.0) und gegen Stockfish bei 44 bis 48 %. Das Rezept macht
  v6 also weder besser noch schlechter.
- **Der lokale Lauf fällt aus der Reihe.** Ob das Pech war oder an einem
  Unterschied liegt (12 statt 16 Prozesse, Windows statt Linux), lässt
  sich mit diesen Läufen nicht sagen. «v7b ist schwächer als v6» gilt
  also nur für diesen einen Lauf. Genauer ist: Weitertrainieren mit
  diesem Rezept bringt nichts.
- **Seed 2 nach 25 Partien** fällt mit 61.0 % auf, rund 3
  Schwankungsbreiten über 50 %. Bei acht Messungen kann so ein Wert aber
  auch Zufall sein. Vor einer Aktivierung müsste man diesen Zwischenstand
  mit mehr Partien nachmessen.
- **Zwei Läufe gleichzeitig** brauchten 35 und 39 Minuten statt 28,
  obwohl die Partien eher kürzer waren. Die Azure-Rechner sind also nicht
  immer gleich schnell. Lokal hätten die beiden Läufe nacheinander über
  2 Stunden gedauert.
- **Kosten:** zusammen gut 90 Rappen. Für das ganze Azure-Training
  (Probelauf und drei Hauptläufe) sind es damit rund 1.30 CHF.
- **Stolperstein:** Über Nacht verlangte die FHNW das Kennwort neu, für
  das Portal und das Startskript. Danach musste die PIM-Rolle wieder
  aktiviert werden.

## Plan: Grosses Training mit Ausscheidung (ab 11.11.2026)

Ziel: ein Netz, das v6 schlägt. Ein einzelner langer Lauf mit dem alten
Rezept bringt laut den drei Läufen oben nichts. Deshalb laufen mehrere
Varianten gleichzeitig, und jede misst ihre Zwischenstände selbst gegen v6.
Weiter geht es nur mit einem Stand, der v6 klar schlägt.

**Runde 1:** vier Aufträge gleichzeitig, je eine F16s_v2, Start immer
`models/full_v5.npz`:

- **A:** Training mit 4 Halbzügen wie bisher, aber Lernrate `1e-4` statt
  `5e-4`, 100 Partien. Prüft: Ist die Lernrate für Feinarbeit zu gross?
- **B:** Training mit 2 Halbzügen, so tief wie die Engine spielt, Lernrate
  `5e-4`, 1000 Partien. Prüft: Passt das Training mit 4 nicht zum Spiel mit 2?
- **C:** wie B, aber Lernrate `1e-4`.
- **D:** altes Rezept mit doppelt so vielen Partien (100, Seed 3), zum
  Vergleich.

Jeder Auftrag misst nach dem Training seine vier Zwischenstände mit je 400
Partien gegen v6, noch auf Azure. Ein Stand gilt als besser, wenn er
mindestens 55 % holt (bei 400 Partien schwankt der Wert um rund 2.5
Punkte). Diesen Stand prüfe ich danach lokal gegen Stockfish, im
Taktik-Test und im Prüfstand. Erst dann wird er v8. Aktiviert wird nur mit
deinem OK.

**Runde 2:** vom Sieger aus mit der besten Variante weiter, wieder vier
Läufe gleichzeitig. Schlägt in einer Runde kein Stand v6, ist Schluss.

Erwartete Kosten Runde 1: rund 3 CHF, höchstens etwa 6 CHF (Notbremse 3 h
bei A und D, 1 h bei B und C).

### Runde 1: Ergebnis (11.11.2026)

Gestartet um 16:50, je 400 Partien gegen v6 pro Zwischenstand, Punkte in %:

| Lauf | Auftrag | Stand 1 | Stand 2 | Stand 3 | Stand 4 | fertig |
|---|---|---|---|---|---|---|
| A: 4 Halbzüge, `1e-4`, 100 Partien | `frank_book_x4ym9n9gqq` | 50.2 | 49.0 | 43.6 | 45.1 | 18:08 |
| B: 2 Halbzüge, `5e-4`, 1000 Partien | `crimson_mango_721vp6c5rc` | 63.9 | 60.4 | 51.6 | **76.0** | 17:48 |
| C: 2 Halbzüge, `1e-4`, 1000 Partien | `red_pencil_nzlknf7mr1` | 54.1 | 53.9 | 64.0 | 56.4 | 17:48 |
| D: altes Rezept, 100 Partien | `nice_quince_0t974g4y7d` | 53.2 | 56.2 | 51.6 | 53.5 | 18:29 |

Stände bei A und D nach 25, 50, 75, 100 Partien, bei B und C nach 250,
500, 750, 1000.

- **Training mit 2 Halbzügen wirkt.** B und C schlagen v6 in fast allen
  Ständen, B nach 1000 Partien mit 76.0 %. Damit ist die Vermutung aus dem
  README bestätigt: Das Training mit 4 Halbzügen passte nicht zum Spiel mit 2.
- **Kleinere Lernrate allein hilft nicht:** A bleibt bei oder unter v6.
- **Das alte Rezept** (D) liegt mit 100 Partien knapp über v6 (51.6 bis
  56.2), wie die drei Läufe mit 50 Partien.
- **Dauer und Kosten:** Training mit 2 Halbzügen ist schnell: 1000 Partien
  in 15 bis 17 Minuten, dazu rund 35 Minuten Messung. A und D brauchten 54
  und 68 Minuten Training. Zusammen liefen die vier Rechner rund 4.6
  Stunden, also etwa 3.30 CHF.
- **Die Stände schwanken stark** (B: 63.9, 60.4, 51.6, 76.0). Darum zählt
  nur ein Stand, der auch lokal gegen Stockfish und im Taktik-Test besteht.
- **Nachmessung** des Zwischenstands von Seed 2 nach 25 Partien (61.0 %
  bei 200 Partien): 60.8 % bei 400 Partien, 60.5 % nur auf den 200 neuen
  Eröffnungen, gegen Stockfish 49.1 % (400 Partien). Er hält also.
- **Stolperstein:** Die PIM-Rolle lief um 18:17 ab, bevor D fertig war.
  D rechnete trotzdem zu Ende und wurde nach neuer Aktivierung um 22:09
  heruntergeladen.

**Lokale Prüfung des Siegers** (B nach 1000 Partien,
`runs/azure/crimson_mango_721vp6c5rc/artifacts/outputs/run/snapshots/ep1000.npz`),
gespielt wie v6 mit 2 Halbzügen, Deckel und Wiederholungsregel:

| Messung | v6 | B nach 1000 Partien |
|---|---|---|
| direkt gegen v6, 400 Partien | 50 % | **76.0 %** |
| gegen Stockfish 1320, dieselben 400 Partien | 42.1 % | **57.5 %** |
| Taktik-Test (150 Stellungen) | 72.7 % | **77.3 %** |

v6 holte auf den ersten 200 Eröffnungen 48.0 %, auf allen 400 nur 42.1 %:
Die zweiten 200 sind schwerer. Verglichen wird darum immer auf denselben
Eröffnungen. B ist in allen drei Messungen klar besser und damit der
Kandidat für v8. Eingetragen als v8 (`models/full_v8.npz`). Im Prüfstand
erster Platz: Teststellungen 45/50 (v6 41), Turnier 54.5/60 (v6 40.0),
Genauigkeit 91.7 (v6 88.2).

## Kosten

Der Preis hängt kaum von der VM-Grösse ab. Bezahlt werden Kernstunden, und
davon braucht der Lauf rund 13. Bei 0.06 bis 0.08 $ pro Kernstunde (DS3_v2:
0.30 $/h für 4 Kerne) kommt das auf **etwa 1 bis 2 $** für den Hauptlauf. Der
Probelauf kostet unter 0.10 $. Insgesamt bleibt es unter 5 $, auch mit einem
zweiten Lauf.

Der Unterschied zum Endpoint, der im Oktober gut 22 $ gekostet hat: Ein
Trainingsauftrag endet von selbst. Dann läuft nichts mehr weiter.

## Danach, nur wenn der Hauptlauf glatt lief

Teilweise erledigt am 11.11.2026: zwei Läufe gleichzeitig, aber mit
anderem Zufall statt anderem Rezept (siehe «Streuung»). Offen sind die
beiden Rezepte unten.

Ein oder zwei weitere Läufe **gleichzeitig**, je mit einer der offenen
Ursachen aus dem README: Lernrate `1e-4` statt `5e-4`, oder Training mit 2
statt 4 Halbzügen. Daran sieht man den eigentlichen Vorteil der Cloud: Lokal
läuft immer nur ein Training, auf Azure mehrere nebeneinander.

## Risiken

- **Kontingent zu klein:** Dann DS3_v2 mit 4 Kernen nehmen. Der Lauf dauert
  länger, kostet aber etwa gleich viel.
- **PIM-Rolle läuft ab:** Die Rolle braucht es nur zum Starten und
  Herunterladen. Der Auftrag selbst läuft weiter.
- **Andere Paketversionen in der Microsoft-Umgebung:** `pip install` mit
  festen Versionen wie in `requirements.txt` legt sie fest.

## Wer macht was

- **Du:** dich anmelden und vor jedem Start auf Azure dein OK geben.
- **Ich:** Startskript, auf Wunsch die PIM-Rolle aktivieren, Aufträge starten
  und überwachen, messen, Vergleich schreiben.
