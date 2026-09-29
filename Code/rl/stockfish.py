"""
Stockfish als Massstab für die Auswertung.

Warum überhaupt? Die Material-Heuristik ist als Gegner schnell ausgereizt — wenn
der Agent 100 Prozent der Partien gewinnt, misst sie keinen Fortschritt mehr.
Ein stärkerer Gegner hält die Messung aussagekräftig.

Wichtig ist die eingestellte Stärke. Gemessen: Selbst mit `depth=1` gewinnt
Stockfish 20 von 20 Partien gegen unseren Agenten — und ebenso gegen die
Material-Heuristik. Als Massstab ist das genauso nutzlos wie ein Gegner, gegen
den immer gewonnen wird.

Der Grund: Stockfishs Suchtiefe 1 ist kein reiner Ein-Halbzug-Blick. Die Engine
hängt eine Quiescence-Suche an, löst also alle Schlagabtausche darüber hinaus
auf, und bewertet die Blätter mit einem neuronalen Netz.

Brauchbar wird der Vergleich erst über eine **Knotenbegrenzung** (`nodes`): Sie
deckelt, wie viele Stellungen Stockfish überhaupt anschauen darf, und lässt sich
fein genug einstellen, um die Stärke unseres Agenten einzuordnen. Ergänzend gibt
es `skill` (0 bis 20, baut absichtliche Fehler ein) und `elo` (ab 1320).

Stockfish gehört nicht ins Repository und wird für den Web-Service nicht
gebraucht; es ist reines Auswertungswerkzeug. Der Pfad kommt entweder als
Argument, aus der Umgebungsvariable STOCKFISH_PATH oder aus dem PATH.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path

import chess
import chess.engine


TOOLS_DIR = Path(__file__).resolve().parents[2] / "tools" / "stockfish"


def find_stockfish(path: str | None = None) -> str | None:
    """
    Pfad zur Stockfish-Programmdatei, oder None wenn keine zu finden ist.

    Gesucht wird der Reihe nach: übergebener Pfad, STOCKFISH_PATH, PATH und
    zuletzt tools/stockfish/ im Projekt. Letzteres ist per .gitignore
    ausgeschlossen — die Datei ist über 100 MB gross und gehört nicht ins Repo.
    """
    # Eine ausdrücklich angegebene Datei wird nicht stillschweigend übergangen:
    # Wer einen Pfad nennt, will genau diese Engine und keine andere.
    if path:
        return str(path) if Path(path).exists() else None

    candidates = [os.environ.get("STOCKFISH_PATH"), shutil.which("stockfish")]
    if TOOLS_DIR.is_dir():
        candidates += [str(found) for found in sorted(TOOLS_DIR.glob("stockfish*")) if found.is_file()]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


@contextmanager
def stockfish_policy(
    path: str | None = None,
    depth: int | None = None,
    nodes: int | None = None,
    elo: int | None = None,
    skill: int | None = None,
):
    """
    Liefert eine Zugfunktion wie die anderen Strategien auch.

    Als Kontextmanager, damit der Engine-Prozess am Ende zuverlässig beendet
    wird — sonst bleiben bei einer Auswertung über mehrere Seeds mehrere
    Stockfish-Prozesse liegen.
    """
    executable = find_stockfish(path)
    if executable is None:
        raise FileNotFoundError(
            "Stockfish nicht gefunden. Pfad über --stockfish angeben oder "
            "STOCKFISH_PATH setzen."
        )

    engine = chess.engine.SimpleEngine.popen_uci(executable)
    try:
        if elo is not None:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": elo})
        if skill is not None:
            engine.configure({"Skill Level": skill})
        limit = chess.engine.Limit(depth=depth, nodes=nodes)

        def policy(board: chess.Board) -> chess.Move | None:
            if board.is_game_over():
                return None
            return engine.play(board, limit).move

        yield policy
    finally:
        engine.quit()
