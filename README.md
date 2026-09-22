# Chess-Project — Schach-Assistent

FHNW **AI Operations**. Web-Service, der zu einer Schachstellung den (aktuell per
Platzhalter-Engine bestimmten) besten Zug liefert. Oberfläche nach den Prinzipien
der Mensch-Maschine-Interaktion; später wird die Zug-Auswahl durch den
Reinforcement-Learning-Agenten ersetzt.

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

## Projektstruktur

```
Code/main.py     FastAPI-App (API + liefert die UI aus)
web/index.html   Oberfläche (Brett-Editor, Zugpartei, Ergebnis)
requirements.txt Abhängigkeiten
render.yaml      render.com Blueprint
```

## API

| Endpoint                 | Eingabe                    | Ausgabe                                      |
|--------------------------|----------------------------|----------------------------------------------|
| `GET /health`            | –                          | `{"status":"ok"}`                            |
| `POST /best-move`        | `{"fen": "..."}`           | `{best_move, san, evaluation, side_to_move}` |
| `POST /best-move-from-image` | Bild (multipart `image`) | dito, Stellungserkennung noch Platzhalter    |
