"""
Schach-Assistent — AI Operations (FHNW)
========================================
FastAPI-Backend für die Web-Oberfläche in ../web/index.html.

Kern-Use-Case (CUJ): Screenshot einer Stellung  ->  optimaler Zug.

Pipeline (zweistufig):
    1) Bild -> Stellung erkennen (Computer Vision)   [hier noch PLATZHALTER]
    2) Stellung -> bester Zug: RL-Agent, wo er zuständig ist (Endspiele mit drei
       Figuren, trainiert in Code/rl/), sonst die Material-Heuristik

Start (lokal):
    .venv aktivieren, dann aus dem Projekt-Wurzelverzeichnis:
        uvicorn Code.main:app --reload
    UI:      http://127.0.0.1:8000/
    API-Doku: http://127.0.0.1:8000/docs

Deploy (render.com):
    Build Command: pip install -r requirements.txt
    Start Command: uvicorn Code.main:app --host 0.0.0.0 --port $PORT
"""
from pathlib import Path

import chess
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .rl.engine import material_evaluation, material_move, value_move
from .rl.features import board_to_features
from .rl.valuenet import ValueNet

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


# --------------------------------------------------------------------------- #
# Datenschemas (Ein-/Ausgabe der API)
# --------------------------------------------------------------------------- #
class FenRequest(BaseModel):
    fen: str = Field(..., description="Stellung in FEN-Notation", examples=[START_FEN])


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


def select_agent(board: chess.Board) -> ValueNet | None:
    """Passendes Endspiel-Modell für die Stellung, sonst None."""
    pieces = board.piece_map()
    if len(pieces) != 3:
        return None
    for piece in pieces.values():
        agent = RL_AGENTS.get(piece.piece_type)
        if agent is not None:
            return agent
    return None


def analyse(board: chess.Board, recognition: str) -> MoveResponse:
    """Gemeinsame Auswertung einer Stellung -> Antwortobjekt."""
    if board.is_game_over():
        raise HTTPException(status_code=422, detail=f"Partie beendet: {board.result()}")

    agent = select_agent(board)
    if agent is not None:
        move = value_move(board, agent, GAMMA)
        value = float(agent.value(board_to_features(board)[None, :])[0])
        evaluation = value if board.turn == chess.WHITE else -value
        engine = "rl"
    else:
        move = material_move(board)
        evaluation = material_evaluation(board)
        engine = "material"

    if move is None:
        raise HTTPException(status_code=422, detail="Keine legalen Züge in dieser Stellung.")

    return MoveResponse(
        fen=board.fen(),
        best_move=move.uci(),
        san=board.san(move),
        evaluation=evaluation,
        side_to_move="white" if board.turn == chess.WHITE else "black",
        recognition=recognition,
        engine=engine,
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
    return analyse(board, recognition="fen")


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
async def best_move_from_image(image: UploadFile = File(...)) -> MoveResponse:
    """
    Optimalen Zug aus einem hochgeladenen Screenshot berechnen.

    Schritt 1 (Bilderkennung -> FEN) ist noch ein PLATZHALTER: aktuell wird die
    Grundstellung angenommen. Die UI zeigt die 'erkannte' FEN an und lässt sie
    korrigieren (Fehlertoleranz), bis die echte Erkennung implementiert ist.
    """
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Leere Datei erhalten.")
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=415, detail="Bitte eine Bilddatei (PNG/JPG) hochladen.")

    # TODO: echte Erkennung — Brett zuschneiden, 64 Felder klassifizieren -> FEN.
    recognized_fen = START_FEN
    board = board_from_fen(recognized_fen)
    return analyse(board, recognition="placeholder")


# --------------------------------------------------------------------------- #
# Statische UI ausliefern (derselbe Server liefert Frontend + API)
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def index():
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        # Ohne no-cache liefert der Browser nach einer UI-Änderung weiter die
        # alte Seite aus — beim Entwickeln verwirrend, in Produktion unkritisch,
        # weil die Datei ohnehin klein ist.
        return FileResponse(index_file, headers={"Cache-Control": "no-cache"})
    return {"message": "UI nicht gefunden. API-Doku unter /docs."}
