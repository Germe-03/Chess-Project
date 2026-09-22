"""
Schach-Assistent — AI Operations (FHNW)
========================================
FastAPI-Backend für die Web-Oberfläche in ../web/index.html.

Kern-Use-Case (CUJ): Screenshot einer Stellung  ->  optimaler Zug.

Pipeline (zweistufig):
    1) Bild -> Stellung erkennen (Computer Vision)   [hier noch PLATZHALTER]
    2) Stellung -> bester Zug (Engine / RL-Agent)    [hier einfache Heuristik]

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
START_FEN = chess.STARTING_FEN


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
        0.0, description="Bewertung in Centipawns aus Sicht von Weiss (+ = Vorteil Weiss)"
    )
    side_to_move: str = Field(..., description="'white' oder 'black'")
    recognition: str = Field(
        "n/a", description="Wie die Stellung ermittelt wurde: 'fen', 'placeholder', ..."
    )


# --------------------------------------------------------------------------- #
# Business-Logik (Platzhalter — später durch RL-Agent / stärkere Engine ersetzen)
# --------------------------------------------------------------------------- #
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def material_evaluation(board: chess.Board) -> float:
    """Einfache Materialbilanz in Centipawns, positiv = Vorteil Weiss."""
    score = 0
    for piece_type, value in PIECE_VALUES.items():
        score += value * len(board.pieces(piece_type, chess.WHITE))
        score -= value * len(board.pieces(piece_type, chess.BLACK))
    return float(score)


def choose_best_move(board: chess.Board) -> chess.Move | None:
    """
    PLATZHALTER-'Engine': material-gieriger Ein-Zug-Blick.

    Bewertet jeden legalen Zug nach der Materialbilanz danach (aus Sicht der
    Zugpartei) und wählt den besten. Liefert immer einen LEGALEN Zug — genug,
    damit die UI end-to-end funktioniert. Wird später durch den RL-Agenten
    (bzw. eine echte Suche) ersetzt.
    """
    best_move, best_score = None, float("-inf")
    white_to_move = board.turn == chess.WHITE
    for move in board.legal_moves:
        board.push(move)
        score = material_evaluation(board)
        if not white_to_move:
            score = -score  # aus Sicht der ziehenden Partei
        board.pop()
        if score > best_score:
            best_score, best_move = score, move
    return best_move


def analyse(board: chess.Board, recognition: str) -> MoveResponse:
    """Gemeinsame Auswertung einer Stellung -> Antwortobjekt."""
    if board.is_game_over():
        raise HTTPException(status_code=422, detail=f"Partie beendet: {board.result()}")

    move = choose_best_move(board)
    if move is None:
        raise HTTPException(status_code=422, detail="Keine legalen Züge in dieser Stellung.")

    san = board.san(move)
    return MoveResponse(
        fen=board.fen(),
        best_move=move.uci(),
        san=san,
        evaluation=material_evaluation(board),
        side_to_move="white" if board.turn == chess.WHITE else "black",
        recognition=recognition,
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
        return FileResponse(index_file)
    return {"message": "UI nicht gefunden. API-Doku unter /docs."}
