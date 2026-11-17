import pickle
import struct
from pathlib import Path

import chess
import chess.polyglot
import pytest

from Code.rl.theorie import KEINE, Theorie, TheorieOptionen, buch_verfuegbar, endspiel_verfuegbar, theorie_fuer


def test_theorie_optionen_bool_kennung_hashbarkeit_und_pickle():
    faelle = [
        (TheorieOptionen(), False, ""),
        (TheorieOptionen(eroeffnung=True), True, "buch"),
        (TheorieOptionen(endspiel=True), True, "tb"),
        (TheorieOptionen(eroeffnung=True, endspiel=True), True, "buch+tb"),
    ]

    for optionen, erwartet_bool, erwartet_kennung in faelle:
        assert bool(optionen) is erwartet_bool
        assert optionen.kennung() == erwartet_kennung
        assert {optionen: erwartet_kennung}[optionen] == erwartet_kennung
        assert pickle.loads(pickle.dumps(optionen)) == optionen

    assert not KEINE
    assert theorie_fuer(KEINE) is None


def _schreibe_polyglot_buch(pfad, board, zuege):
    key = chess.polyglot.zobrist_hash(board)
    eintraege = []

    for uci, weight in zuege:
        zug = chess.Move.from_uci(uci)
        raw_move = zug.to_square | (zug.from_square << 6)
        if zug.promotion:
            promo = {
                chess.KNIGHT: 1,
                chess.BISHOP: 2,
                chess.ROOK: 3,
                chess.QUEEN: 4,
            }[zug.promotion]
            raw_move |= promo << 12
        eintraege.append((key, raw_move, weight, 0))

    with pfad.open("wb") as datei:
        for eintrag in sorted(eintraege):
            datei.write(struct.pack(">QHHI", *eintrag))


def test_buch_liefert_zug_mit_hoechstem_gewicht_und_ignoriert_andere_stellungen(tmp_path):
    buch_pfad = tmp_path / "eroeffnung.bin"
    start = chess.Board()
    _schreibe_polyglot_buch(
        buch_pfad,
        start,
        [("e2e4", 10), ("d2d4", 30)],
    )

    assert buch_verfuegbar(buch_pfad)

    with Theorie(TheorieOptionen(eroeffnung=True), buch_pfad=buch_pfad) as theorie:
        ergebnis = theorie.zug(chess.Board())
        assert ergebnis is not None
        assert ergebnis.move == chess.Move.from_uci("d2d4")
        assert ergebnis.quelle == "buch"
        assert ergebnis.wert is None

        andere_stellung = chess.Board()
        andere_stellung.push_uci("a2a3")
        assert theorie.zug(andere_stellung) is None

    with Theorie(TheorieOptionen(), buch_pfad=buch_pfad) as theorie:
        assert theorie.zug(chess.Board()) is None


def test_verfuegbarkeit_von_buch_und_endspieltabellen(tmp_path):
    fehlendes_buch = tmp_path / "fehlt.bin"
    leeres_buch = tmp_path / "leer.bin"
    leeres_buch.touch()

    assert not buch_verfuegbar(fehlendes_buch)
    assert not buch_verfuegbar(leeres_buch)

    assert not endspiel_verfuegbar(tmp_path)

    tabellen_ordner = tmp_path / "tabellen"
    tabellen_ordner.mkdir()
    (tabellen_ordner / "KQvK.rtbw").touch()
    assert endspiel_verfuegbar(tabellen_ordner)

    assert not endspiel_verfuegbar(tmp_path / "nicht-vorhanden")


def test_fehlende_pfade_liefern_keinen_zug(tmp_path):
    with Theorie(
        TheorieOptionen(eroeffnung=True, endspiel=True),
        buch_pfad=tmp_path / "fehlendes-buch.bin",
        syzygy_pfad=tmp_path / "fehlende-tabellen",
    ) as theorie:
        assert theorie.zug(chess.Board()) is None


@pytest.mark.skipif(not endspiel_verfuegbar(), reason="Keine Syzygy-Tabellen verfügbar")


def test_endspiel_kqvk_ist_gewonnen():
    board = chess.Board("8/8/8/8/8/2k5/8/K6Q w - - 0 1")
    assert board.is_valid()

    with Theorie(TheorieOptionen(endspiel=True)) as theorie:
        ergebnis = theorie.zug(board)

    assert ergebnis is not None
    assert ergebnis.wert == 1.0
    assert ergebnis.quelle == "endspiel"


@pytest.mark.skipif(not endspiel_verfuegbar(), reason="Keine Syzygy-Tabellen verfügbar")


def test_endspiel_knvk_ist_remis():
    board = chess.Board("8/8/8/4k3/8/8/8/KN6 w - - 0 1")
    assert board.is_valid()

    with Theorie(TheorieOptionen(endspiel=True)) as theorie:
        ergebnis = theorie.zug(board)

    assert ergebnis is not None
    assert ergebnis.wert == 0.0
    assert ergebnis.quelle == "endspiel"


def test_endspiel_probe_ohne_passende_stellung_wird_uebersprungen(tmp_path):
    with Theorie(
        TheorieOptionen(endspiel=True),
        syzygy_pfad=tmp_path,
    ) as theorie:
        assert theorie.zug(chess.Board()) is None

        board_mit_rochaderecht = chess.Board("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1")
        assert theorie.zug(board_mit_rochaderecht) is None
