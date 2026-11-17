import zipfile

import chess
import chess.polyglot
import pytest

from Code.rl.buch import polyglot_zug, zaehle_partien, schreibe_buch


@pytest.fixture
def pgn_pfad(tmp_path):
    """Erstellt eine kleine PGN-Datei mit drei gewerteten Partien."""
    pgn = """[Event "Sieg mit Rochade"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. O-O Nf6 1-0

[Event "Sieg fuer Schwarz"]
[Result "0-1"]

1. e4 c5 0-1

[Event "Remis mit Variante"]
[Result "1/2-1/2"]

1. d4 d5 (1... Nf6 2. c4) 2. c4 1/2-1/2

[Event "Ungewertete Partie"]
[Result "*"]

1. c4 e5 *
"""
    pfad = tmp_path / "partien.pgn"
    pfad.write_text(pgn, encoding="utf-8")
    return pfad


def _buch_eintraege(reader, board):
    """Liefert die Zuege und Gewichte fuer eine Stellung."""
    return {eintrag.move.uci(): eintrag.weight for eintrag in reader.find_all(board)}


def test_polyglot_zug_kodiert_normale_zuege_umwandlungen_und_rochaden():
    board = chess.Board()
    assert polyglot_zug(board, chess.Move.from_uci("e2e4")) == (
        chess.E4 | chess.E2 << 6
    )

    board = chess.Board("8/P6k/8/8/8/8/8/K7 w - - 0 1")
    assert polyglot_zug(board, chess.Move.from_uci("a7a8q")) == (
        chess.A8 | chess.A7 << 6 | 4 << 12
    )

    rochade_fen = "r3k2r/8/8/8/8/8/8/R3K2R"
    rochaden = [
        ("w KQkq - 0 1", "e1g1", "e1h1"),
        ("w KQkq - 0 1", "e1c1", "e1a1"),
        ("b KQkq - 0 1", "e8g8", "e8h8"),
        ("b KQkq - 0 1", "e8c8", "e8a8"),
    ]
    for zusatz, zug_uci, turm_uci in rochaden:
        board = chess.Board(f"{rochade_fen} {zusatz}")
        zug = chess.Move.from_uci(zug_uci)
        assert zug in board.legal_moves
        turm = chess.Move.from_uci(turm_uci)
        assert polyglot_zug(board, zug) == (
            turm.to_square | zug.from_square << 6
        )


def test_zaehle_partien_ignoriert_ungewertete_partie_und_varianten(pgn_pfad):
    zaehler, anzahl = zaehle_partien(pgn_pfad)

    assert anzahl == 3

    board = chess.Board()
    board.push_uci("d2d4")
    board.push_uci("d7d5")
    hash_stellung = chess.polyglot.zobrist_hash(board)
    assert (hash_stellung, "g8f6") not in zaehler


def test_schreibe_buch_grundstellung_rochade_und_variantenzug(pgn_pfad, tmp_path):
    zaehler, anzahl = zaehle_partien(pgn_pfad)
    assert anzahl == 3

    buch_pfad = tmp_path / "buch.bin"
    schreibe_buch(zaehler, buch_pfad, min_partien=1, min_punkte=0.0)

    with chess.polyglot.open_reader(buch_pfad) as reader:
        grundstellung = _buch_eintraege(reader, chess.Board())
        assert grundstellung["e2e4"] == 2
        assert grundstellung["d2d4"] == 1
        assert "c2c4" not in grundstellung

        board = chess.Board()
        for zug in ("e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "f8c5"):
            board.push_uci(zug)

        eintraege = _buch_eintraege(reader, board)
        assert "e1g1" in eintraege
        assert chess.Move.from_uci("e1g1") in board.legal_moves

        board = chess.Board()
        board.push_uci("d2d4")
        board.push_uci("d7d5")
        assert "g8f6" not in _buch_eintraege(reader, board)


def test_schreibe_buch_min_partien(pgn_pfad, tmp_path):
    zaehler, _ = zaehle_partien(pgn_pfad)
    buch_pfad = tmp_path / "buch_min_partien.bin"
    schreibe_buch(zaehler, buch_pfad, min_partien=2, min_punkte=0.0)

    with chess.polyglot.open_reader(buch_pfad) as reader:
        grundstellung = _buch_eintraege(reader, chess.Board())
        assert grundstellung == {"e2e4": 2}


def test_schreibe_buch_min_punkte(pgn_pfad, tmp_path):
    zaehler, _ = zaehle_partien(pgn_pfad)

    board = chess.Board()
    board.push_uci("e2e4")

    buch_ohne_punktefilter = tmp_path / "buch_ohne_punktefilter.bin"
    schreibe_buch(
        zaehler,
        buch_ohne_punktefilter,
        min_partien=1,
        min_punkte=0.0,
    )
    with chess.polyglot.open_reader(buch_ohne_punktefilter) as reader:
        assert "e7e5" in _buch_eintraege(reader, board)

    buch_mit_punktefilter = tmp_path / "buch_mit_punktefilter.bin"
    schreibe_buch(
        zaehler,
        buch_mit_punktefilter,
        min_partien=1,
        min_punkte=0.45,
    )
    with chess.polyglot.open_reader(buch_mit_punktefilter) as reader:
        assert "e7e5" not in _buch_eintraege(reader, board)


def test_zaehle_partien_liest_zipdatei(pgn_pfad, tmp_path):
    zip_pfad = tmp_path / "partien.zip"
    with zipfile.ZipFile(zip_pfad, "w", compression=zipfile.ZIP_DEFLATED) as archiv:
        archiv.write(pgn_pfad, arcname="partien.pgn")

    zaehler_pgn, anzahl_pgn = zaehle_partien(pgn_pfad)
    zaehler_zip, anzahl_zip = zaehle_partien(zip_pfad)

    assert zaehler_zip == zaehler_pgn
    assert anzahl_zip == anzahl_pgn == 3


def test_zaehle_partien_begrenzt_partienzahl(pgn_pfad):
    _, anzahl = zaehle_partien(pgn_pfad, max_partien=1)

    assert anzahl == 1
