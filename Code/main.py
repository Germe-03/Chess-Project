"""
Schach-Assistent — AI Operations (FHNW)
========================================
FastAPI-Backend für die Web-Oberfläche in ../web/index.html.

Kern-Use-Case (CUJ): Screenshot einer Stellung  ->  optimaler Zug.

Pipeline (zweistufig):
    1) Bild -> Stellung erkennen: Brett im Bild finden (Code/vision/finder.py),
       dann jedes der 64 Felder mit einem CNN klassifizieren
    2) Stellung -> bester Zug: das Vollpartien-Modell der gewählten Version mit
       ihrer Suchtiefe (Code/rl/), bei drei Figuren ein Endspiel-Spezialist

Start (lokal):
    .venv aktivieren, dann aus dem Projekt-Wurzelverzeichnis:
        uvicorn Code.main:app --reload
    UI:      http://127.0.0.1:8000/
    API-Doku: http://127.0.0.1:8000/docs

Deploy (render.com):
    Build Command: pip install -r requirements.txt
    Start Command: uvicorn Code.main:app --host 0.0.0.0 --port $PORT

Konfiguration über Umgebungsvariablen:
    MAX_SEARCH_DEPTH  Obergrenze der Vorausschau in Halbzügen (Standard 2).
                      Schützt die kleine Render-Instanz vor Versionen, deren
                      Suche dort zu lange dauern würde.
"""
import os
from io import BytesIO
from pathlib import Path
from typing import NamedTuple

import chess
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .rl.engine import material_evaluation, material_move, value_move
from .rl.features import board_to_features
from .rl.registry import active_name, load_model, search_depth
from .rl.registry import versions as registry_versions
from .rl.search import search
from .rl.valuenet import ValueNet
from .vision.finder import find_board
from .vision.recognizer import TileClassifier, tiles_to_fen
from .vision.tiles import board_to_tiles

# --------------------------------------------------------------------------- #
# App-Grundgerüst
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="Schach-Assistent API",
    version="0.1.0",
    description="Erkennt eine Schachstellung (aus Bild oder FEN) und liefert den optimalen Zug.",
)

# CORS offen halten, damit die UI auch von einem anderen Origin (z. B. lokalem
# Dev-Server / geöffnete Datei) auf die API zugreifen kann. Für ein Studien-
# projekt vertretbar; produktiv würde man die erlaubten Origins einschränken.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
START_FEN = chess.STARTING_FEN
GAMMA = 0.99

# Auf render.com (0.1 CPU) dauert ein Zug etwa zehnmal so lange wie lokal. Mit
# Tiefe 2 sind das rund 1 bis 3 Sekunden, mit Tiefe 3 eher 15 bis 30 — zu lang
# für eine Web-Anfrage. Wird die Grenze gezogen, meldet die Antwort im Feld
# "depth", mit welcher Tiefe tatsächlich gerechnet wurde.
MAX_SEARCH_DEPTH = int(os.environ.get("MAX_SEARCH_DEPTH", "2"))


# --------------------------------------------------------------------------- #
# Datenschemas (Ein-/Ausgabe der API)
# --------------------------------------------------------------------------- #
class FenRequest(BaseModel):
    fen: str = Field(..., description="Stellung in FEN-Notation", examples=[START_FEN])
    version: str | None = Field(None, description="Modellversion; leer heisst die aktive")


class MoveResponse(BaseModel):
    fen: str = Field(..., description="Zugrunde gelegte Stellung (FEN)")
    best_move: str = Field(..., description="Empfohlener Zug in UCI-Notation, z. B. e2e4")
    san: str = Field("", description="Empfohlener Zug in lesbarer SAN-Notation, z. B. e4")
    evaluation: float = Field(
        0.0,
        description=(
            "Bewertung aus Sicht von Weiss (+ = Vorteil Weiss). Skala je nach Engine: "
            "Centipawns bei 'material', erwarteter Partieausgang in [-1, 1] bei 'rl'."
        ),
    )
    side_to_move: str = Field(..., description="'white' oder 'black'")
    recognition: str = Field(
        "n/a", description="Wie die Stellung ermittelt wurde: 'fen', 'placeholder', ..."
    )
    engine: str = Field(
        "material", description="Welche Engine den Zug gewählt hat: 'rl' oder 'material'"
    )
    version: str = Field(
        "", description="Welches Modell geantwortet hat, z. B. 'v2' oder 'endspiel-queen'"
    )
    confidence: float = Field(
        0.0,
        description=(
            "Nur bei Bilderkennung: niedrigste Sicherheit über alle 64 Felder. "
            "Werte unter 0.9 heissen, dass die Stellung geprüft werden sollte."
        ),
    )
    depth: int = Field(1, description="Mit wie vielen Halbzügen Vorausschau gerechnet wurde")
    pv: list[str] = Field(
        default_factory=list,
        description="Erwartete Zugfolge (Hauptvariante) in SAN, beginnend mit dem empfohlenen Zug",
    )


class PlayRequest(BaseModel):
    fen: str = Field(..., description="Aktuelle Stellung", examples=[START_FEN])
    move: str = Field(
        "", description="Zug des Menschen in UCI-Notation. Leer lassen, damit die Engine eröffnet."
    )
    version: str | None = Field(None, description="Modellversion; leer heisst die aktive")


class PlayResponse(BaseModel):
    fen: str = Field(..., description="Stellung nach beiden Zügen")
    player_san: str = Field(..., description="Zug des Menschen in lesbarer Notation")
    engine_uci: str = Field("", description="Antwortzug der Engine, UCI")
    engine_san: str = Field("", description="Antwortzug der Engine, lesbar")
    status: str = Field(..., description="Klartext für die Anzeige")
    finished: bool = Field(False, description="Ist die Partie zu Ende?")
    evaluation: float = Field(0.0, description="Bewertung aus Sicht von Weiss")
    engine: str = Field("", description="'rl' oder 'material'")
    version: str = Field("", description="Welches Modell geantwortet hat")
    depth: int = Field(1, description="Mit wie vielen Halbzügen Vorausschau gerechnet wurde")
    pv: list[str] = Field(
        default_factory=list, description="Erwartete Zugfolge der Engine in SAN, beginnend mit ihrem Zug"
    )


class RecognitionResponse(BaseModel):
    fen: str = Field(..., description="Erkannte Stellung in FEN-Notation")
    confidence: float = Field(..., description="Niedrigste Sicherheit über alle 64 Felder")
    playable: bool = Field(..., description="Lässt sich aus dieser Stellung überhaupt ein Zug berechnen?")
    note: str = Field("", description="Hinweis, falls die Stellung nicht spielbar ist")


class LegalMovesRequest(BaseModel):
    fen: str = Field(..., description="Stellung in FEN-Notation", examples=[START_FEN])
    square: str = Field(..., description="Ausgangsfeld der Figur, z. B. 'e1'", examples=["e2"])


class LegalMovesResponse(BaseModel):
    square: str = Field(..., description="Abgefragtes Ausgangsfeld")
    targets: list[str] = Field(
        default_factory=list, description="Felder, die diese Figur regelkonform erreichen kann"
    )
    in_check: bool = Field(False, description="Steht diese Partei im Schach?")


# --------------------------------------------------------------------------- #
# Zug-Auswahl: RL-Agent wo zuständig, sonst die Material-Heuristik
# --------------------------------------------------------------------------- #
# Die trainierten Modelle sind Endspiel-Spezialisten (König + Dame bzw. König +
# Turm gegen König). Ausserhalb dieser Stellungen bleibt es bei der Heuristik —
# eine ehrliche Zuständigkeitsgrenze ist besser als ein Modell, das weit
# ausserhalb seiner Trainingsverteilung rät.
RL_MODEL_FILES = {
    chess.QUEEN: MODEL_DIR / "value_kqk.npz",
    chess.ROOK: MODEL_DIR / "value_krk.npz",
}
RL_AGENTS = {
    piece_type: ValueNet.load(path)
    for piece_type, path in RL_MODEL_FILES.items()
    if path.exists()
}

# Erkennungsmodell einmal beim Start laden; 0.5 MB, kostet nichts.
RECOGNIZER = TileClassifier.load() if TileClassifier.available() else None


def select_agent(board: chess.Board, version: str | None = None) -> tuple[ValueNet | None, str, int]:
    """
    Zuständiges Modell für die Stellung, ein Name für die Antwort und die
    Suchtiefe in Halbzügen.

    Bei genau drei Figuren übernimmt der passende Endspiel-Spezialist. Die sind
    bewusst nicht versioniert — sie lösen eine feste Aufgabe, ändern sich nicht
    mehr und rechnen wie beim Training einen Halbzug voraus. Für alles andere
    kommt das Vollpartien-Modell aus der Registry, entweder die angefragte
    Version oder die aktive, mit der Tiefe, die zu dieser Version gehört.
    """
    pieces = board.piece_map()
    if len(pieces) == 3:
        for piece in pieces.values():
            agent = RL_AGENTS.get(piece.piece_type)
            if agent is not None:
                return agent, f"endspiel-{chess.piece_name(piece.piece_type)}", 1

    model = load_model(version)
    if model is not None:
        name = version or active_name() or "?"
        return model, name, max(1, min(search_depth(name), MAX_SEARCH_DEPTH))
    return None, "heuristik", 1


class Choice(NamedTuple):
    move: chess.Move | None
    evaluation: float          # aus Sicht von Weiss
    engine: str                # 'rl' oder 'material'
    label: str                 # Modellname
    depth: int                 # tatsächlich gerechnete Halbzüge
    pv: list[chess.Move]       # erwartete Zugfolge


def choose_move(board: chess.Board, version: str | None = None) -> Choice:
    """
    Zug der gewählten Version samt Bewertung aus Sicht von Weiss.

    Tiefe 1 läuft absichtlich über denselben Code wie vor dem Einbau der Suche
    (value_move und die Netzbewertung der aktuellen Stellung). Damit spielen v1
    bis v3 Zug für Zug wie bisher; tests/test_search.py prüft das gegen eine
    Referenz, die vor der Änderung aufgenommen wurde.

    Ab Tiefe 2 rechnet die Alpha-Beta-Suche, und als Bewertung wird ihr
    Suchwert gemeldet — also das, was die Engine nach dem Vorausrechnen
    erwartet, nicht der Blick auf die Stellung allein.
    """
    agent, label, depth = select_agent(board, version)
    if agent is None:
        move = material_move(board)
        return Choice(move, material_evaluation(board), "material", "heuristik", 1, [move] if move else [])

    if depth == 1:
        move = value_move(board, agent, GAMMA)
        value = float(agent.value(board_to_features(board)[None, :])[0])
        pv = [move] if move else []
    else:
        result = search(board, agent, depth, GAMMA)
        move, value, pv = result.move, result.value, result.pv
    evaluation = value if board.turn == chess.WHITE else -value
    return Choice(move, evaluation, "rl", label, depth, pv)


def line_san(board: chess.Board, line: list[chess.Move]) -> list[str]:
    """Zugfolge in lesbarer Notation, ausgehend von 'board'."""
    replay = board.copy(stack=False)
    sans = []
    for move in line:
        sans.append(replay.san(move))
        replay.push(move)
    return sans


def analyse(board: chess.Board, recognition: str, version: str | None = None) -> MoveResponse:
    """Gemeinsame Auswertung einer Stellung -> Antwortobjekt."""
    if board.is_game_over():
        raise HTTPException(status_code=422, detail=f"Partie beendet: {board.result()}")

    choice = choose_move(board, version)
    if choice.move is None:
        raise HTTPException(status_code=422, detail="Keine legalen Züge in dieser Stellung.")

    return MoveResponse(
        fen=board.fen(),
        best_move=choice.move.uci(),
        san=board.san(choice.move),
        evaluation=choice.evaluation,
        side_to_move="white" if board.turn == chess.WHITE else "black",
        recognition=recognition,
        engine=choice.engine,
        version=choice.label,
        depth=choice.depth,
        pv=line_san(board, choice.pv),
    )


def board_from_fen(fen: str) -> chess.Board:
    try:
        return chess.Board(fen)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültige FEN-Notation.")


# --------------------------------------------------------------------------- #
# API-Endpunkte
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["system"])
def health() -> dict:
    """Einfacher Health-Check für die Statusanzeige der UI."""
    return {"status": "ok"}


@app.post("/best-move", response_model=MoveResponse, tags=["schach"])
def best_move(req: FenRequest) -> MoveResponse:
    """Optimalen Zug zu einer als FEN gegebenen Stellung berechnen."""
    board = board_from_fen(req.fen)
    return analyse(board, recognition="fen", version=req.version)


async def recognize_board(image: UploadFile, side: str, flipped: bool) -> tuple[chess.Board, float]:
    """Screenshot -> Stellung und Sicherheit. Gemeinsame Grundlage beider Endpunkte."""
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Leere Datei erhalten.")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Bitte eine Bilddatei (PNG/JPG) hochladen.")
    if RECOGNIZER is None:
        raise HTTPException(
            status_code=503, detail="Das Erkennungsmodell fehlt (models/tile_classifier.npz)."
        )
    try:
        picture = Image.open(BytesIO(data))
        # Zuerst das Brett im Bild suchen. Ohne diesen Schritt liegen die
        # Kachelgrenzen mitten in den Feldern, sobald ein Rand mit auf dem
        # Screenshot ist, und die Erkennung liefert stillschweigend Unsinn.
        ausschnitt = find_board(picture)
        if ausschnitt is not None:
            picture = picture.crop(ausschnitt)
        classes, probabilities = RECOGNIZER.predict(board_to_tiles(picture))
    except Exception:
        raise HTTPException(status_code=400, detail="Das Bild konnte nicht gelesen werden.")

    if flipped:
        # Aus Schwarzsicht aufgenommen: Das Brett ist um 180 Grad gedreht, also
        # kehrt sich die Reihenfolge der 64 Felder einfach um.
        classes = classes[::-1]

    seite = "b" if side == "b" else "w"
    return board_from_fen(f"{tiles_to_fen(classes)} {seite} - - 0 1"), float(probabilities.min())


def describe_position(board: chess.Board) -> tuple[str, bool]:
    """Klartext-Status einer Stellung und ob die Partie zu Ende ist."""
    if board.is_checkmate():
        gewinner = "Schwarz" if board.turn == chess.WHITE else "Weiss"
        return f"Schachmatt — {gewinner} gewinnt", True
    if board.is_stalemate():
        return "Patt — remis", True
    if board.is_insufficient_material():
        return "Remis — zu wenig Material zum Mattsetzen", True
    if board.can_claim_fifty_moves():
        return "Remis nach der 50-Züge-Regel", True
    if board.can_claim_threefold_repetition():
        return "Remis durch dreifache Stellungswiederholung", True
    am_zug = "Weiss" if board.turn == chess.WHITE else "Schwarz"
    if board.is_check():
        return f"Schach — {am_zug} muss reagieren", False
    return f"{am_zug} am Zug", False


def parse_move(board: chess.Board, uci: str) -> chess.Move:
    """
    UCI-Zug einlesen. Fehlt bei einem Bauernzug auf die letzte Reihe die
    Umwandlungsfigur, wird zur Dame umgewandelt — das erspart der Oberfläche
    einen Auswahldialog und trifft in aller Regel das Gewollte.
    """
    try:
        move = chess.Move.from_uci(uci)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültige Zugnotation.")
    if move in board.legal_moves:
        return move
    als_dame = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
    if als_dame in board.legal_moves:
        return als_dame
    raise HTTPException(status_code=400, detail="Dieser Zug ist nicht erlaubt.")


@app.get("/versions", tags=["modelle"])
def list_versions() -> dict:
    """Verfügbare Modellversionen für die Auswahl in der Oberfläche."""
    return {"active": active_name(), "versions": registry_versions()}


@app.post("/spielen/zug", response_model=PlayResponse, tags=["schach"])
def play_move(req: PlayRequest) -> PlayResponse:
    """
    Einen Zug des Menschen ausführen und die Engine antworten lassen.

    Der Server merkt sich bewusst keine Partie: Die Stellung kommt als FEN mit
    der Anfrage und geht als FEN zurück. Dadurch braucht es weder Sitzungen noch
    eine Datenbank, und der Dienst verträgt Neustarts und mehrere Instanzen —
    wichtig für render.com und später Azure.
    """
    board = board_from_fen(req.fen)
    if board.is_game_over():
        raise HTTPException(status_code=422, detail="Die Partie ist bereits beendet.")

    player_san = ""
    if req.move:
        move = parse_move(board, req.move)
        player_san = board.san(move)
        board.push(move)

    status, finished = describe_position(board)
    if finished:
        return PlayResponse(
            fen=board.fen(), player_san=player_san, status=status, finished=True,
            evaluation=material_evaluation(board),
        )

    choice = choose_move(board, req.version)
    if choice.move is None:
        raise HTTPException(status_code=422, detail="Die Engine findet keinen Zug.")
    engine_san = board.san(choice.move)
    pv = line_san(board, choice.pv)
    board.push(choice.move)
    status, finished = describe_position(board)

    return PlayResponse(
        fen=board.fen(),
        player_san=player_san,
        engine_uci=choice.move.uci(),
        engine_san=engine_san,
        status=status,
        finished=finished,
        evaluation=choice.evaluation,
        engine=choice.engine,
        version=choice.label,
        depth=choice.depth,
        pv=pv,
    )


@app.post("/legal-moves", response_model=LegalMovesResponse, tags=["schach"])
def legal_moves(req: LegalMovesRequest) -> LegalMovesResponse:
    """
    Erreichbare Felder einer einzelnen Figur — für die Zugvorschau im Brett-Editor.

    Die Regelprüfung übernimmt python-chess. Dadurch sind Fesselungen, Rochade und
    En passant automatisch korrekt, und für den König gilt insbesondere: Felder,
    auf denen er im Schach stünde, sind nicht dabei.

    Anders als beim Spielen wird hier die Zugpartei auf die Farbe der angeklickten
    Figur gesetzt. Im Editor soll man ja auch die Seite anschauen dürfen, die
    gerade nicht am Zug ist.
    """
    board = board_from_fen(req.fen)
    try:
        square = chess.parse_square(req.square)
    except ValueError:
        raise HTTPException(status_code=400, detail="Ungültiges Feld.")

    piece = board.piece_at(square)
    if piece is None:
        return LegalMovesResponse(square=req.square, targets=[])

    board.turn = piece.color
    targets = sorted(
        {chess.square_name(move.to_square) for move in board.legal_moves if move.from_square == square}
    )
    return LegalMovesResponse(square=req.square, targets=targets, in_check=board.is_check())


@app.post("/best-move-from-image", response_model=MoveResponse, tags=["schach"])
async def best_move_from_image(
    image: UploadFile = File(...),
    side: str = Form("w"),
    flipped: bool = Form(False),
    version: str | None = Form(None),
) -> MoveResponse:
    """
    Stellung aus einem Screenshot erkennen und den besten Zug berechnen.

    Das Erkennungsmodell stammt aus linrock/chessboard-recognizer (MIT) und
    klassifiziert jedes der 64 Felder einzeln. Es läuft hier in reinem NumPy,
    damit der Dienst ohne TensorFlow auskommt.

    Zwei Dinge kann das Bild nicht hergeben und müssen mitgeliefert werden: wer
    am Zug ist, und ob das Brett aus Sicht von Schwarz dargestellt ist. Die
    Oberfläche fragt beides ab und lässt die erkannte Stellung nachbessern —
    bei einem Erkennungsfehler soll man nicht von vorn anfangen müssen.
    """
    board, confidence = await recognize_board(image, side, flipped)
    antwort = analyse(board, recognition="cnn", version=version)
    antwort.confidence = confidence
    return antwort


@app.post("/recognize", response_model=RecognitionResponse, tags=["schach"])
async def recognize(
    image: UploadFile = File(...),
    side: str = Form("w"),
    flipped: bool = Form(False),
) -> RecognitionResponse:
    """
    Nur erkennen, nicht rechnen.

    Bewusst getrennt von der Zugberechnung: Erkennt das Modell eine Stellung, aus
    der sich kein Zug ableiten lässt — ein leeres Brett, ein fehlender König, eine
    bereits beendete Partie —, soll der Nutzer trotzdem sehen, was erkannt wurde,
    und es nachbessern können. Ein Fehler statt eines Ergebnisses würde die
    erkannte Stellung verwerfen und ihn bei null anfangen lassen.
    """
    board, confidence = await recognize_board(image, side, flipped)
    note = ""
    if not board.is_valid():
        note = "Die Stellung ist so nicht regelkonform — fehlt ein König?"
    elif board.is_game_over():
        note = f"Aus dieser Stellung gibt es keinen Zug mehr: {board.result()}"
    return RecognitionResponse(
        fen=board.fen(), confidence=confidence, playable=not note, note=note
    )


# --------------------------------------------------------------------------- #
# Statische UI ausliefern (derselbe Server liefert Frontend + API)
# --------------------------------------------------------------------------- #
def _symbol(datei: str, typ: str):
    """
    Symbol für den Browsertab.

    Feste Pfade statt eines Platzhalters: Ein Pfadparameter würde hier auch
    /spielen abfangen, weil die Routen der Reihe nach geprüft werden. Und
    feste Pfade legen genau fest, welche Dateien aus web/ nach aussen gelangen.
    """
    pfad = WEB_DIR / datei
    if not pfad.exists():
        raise HTTPException(status_code=404, detail="Nicht gefunden.")
    return FileResponse(pfad, media_type=typ,
                        headers={"Cache-Control": "public, max-age=86400"})


# Die .ico-Fassung ist kein Zierrat: Manche Browser fragen /favicon.ico an,
# ohne vorher ins HTML zu schauen.
@app.get("/favicon.ico", include_in_schema=False)
def favicon_ico():
    return _symbol("favicon.ico", "image/x-icon")


@app.get("/favicon.svg", include_in_schema=False)
def favicon_svg():
    return _symbol("favicon.svg", "image/svg+xml")


@app.get("/favicon.png", include_in_schema=False)
def favicon_png():
    return _symbol("favicon.png", "image/png")


@app.get("/spielen", include_in_schema=False)
def spielen():
    page = WEB_DIR / "spielen.html"
    if page.exists():
        return FileResponse(page, headers={"Cache-Control": "no-cache"})
    return {"message": "Spielseite nicht gefunden."}


@app.get("/", include_in_schema=False)
def index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        # Ohne no-cache liefert der Browser nach einer UI-Änderung weiter die
        # alte Seite aus — beim Entwickeln verwirrend, in Produktion unkritisch,
        # weil die Datei ohnehin klein ist.
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {"message": "UI nicht gefunden. API-Doku unter /docs."}
