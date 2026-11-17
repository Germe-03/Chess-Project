import chess
import numpy as np

from Code.rl.positions import buch_eroeffnung, sample_start


_E4_MOVE = chess.Move.from_uci("e2e4")
_D4_MOVE = chess.Move.from_uci("d2d4")
_C5_MOVE = chess.Move.from_uci("c7c5")

_E4_BOARD = chess.Board()
_E4_BOARD.push(_E4_MOVE)
_E4_FEN = _E4_BOARD.fen()


class FakeTheorie:
    def buch_zuege(self, board: chess.Board) -> list[tuple[chess.Move, int]]:
        if board.fen() == chess.STARTING_FEN:
            return [(_E4_MOVE, 30), (_D4_MOVE, 10)]
        if board.fen() == _E4_FEN:
            return [(_C5_MOVE, 5)]
        return []


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_buch_eroeffnung_mit_ohne_zusatzhalbzuege_nur_buchstellungen() -> None:
    theorie = FakeTheorie()

    e4_board = chess.Board()
    e4_board.push(_E4_MOVE)

    d4_board = chess.Board()
    d4_board.push(_D4_MOVE)

    e4_c5_board = e4_board.copy()
    e4_c5_board.push(_C5_MOVE)

    erwartete_fens = {
        chess.Board().fen(),
        e4_board.fen(),
        d4_board.fen(),
        e4_c5_board.fen(),
    }

    for seed in range(2000):
        board = buch_eroeffnung(
            _rng(seed),
            theorie,
            extra_halbzuege=0,
        )
        assert board.fen() in erwartete_fens


def test_erster_buchzug_hat_erwartete_gewichtung() -> None:
    theorie = FakeTheorie()
    erste_zuege: list[chess.Move] = []

    for seed in range(2000):
        board = buch_eroeffnung(
            _rng(seed),
            theorie,
            extra_halbzuege=0,
        )
        if board.move_stack:
            erste_zuege.append(board.move_stack[0])

    anteil_e4 = sum(move == _E4_MOVE for move in erste_zuege) / len(erste_zuege)
    assert 0.65 <= anteil_e4 <= 0.85


def test_gleiche_seeds_ergeben_gleiche_fen() -> None:
    theorie = FakeTheorie()

    erste_runde = [
        buch_eroeffnung(_rng(seed), theorie, extra_halbzuege=4).fen()
        for seed in range(50)
    ]
    zweite_runde = [
        buch_eroeffnung(_rng(seed), theorie, extra_halbzuege=4).fen()
        for seed in range(50)
    ]

    assert erste_runde == zweite_runde


def test_zusatzhalbzuege_ergeben_gueltige_stellungen() -> None:
    theorie = FakeTheorie()

    for seed in range(200):
        board = buch_eroeffnung(
            _rng(seed),
            theorie,
            extra_halbzuege=4,
        )
        assert board.is_valid()


def test_sample_start_full_bleibt_rueckwaertskompatibel() -> None:
    theorie = FakeTheorie()

    for seed in range(100):
        ohne_buch = sample_start("full", _rng(seed))
        buch_ausgeschaltet = sample_start(
            "full",
            _rng(seed),
            buch_anteil=0.0,
            theorie=theorie,
        )
        ohne_theorie = sample_start(
            "full",
            _rng(seed),
            buch_anteil=1.0,
            theorie=None,
        )

        assert ohne_buch.fen() == buch_ausgeschaltet.fen()
        assert ohne_buch.fen() == ohne_theorie.fen()


def test_sample_start_mit_buch_beginnt_meistens_mit_buchzug() -> None:
    theorie = FakeTheorie()
    passende_erste_zuege = 0

    for seed in range(300):
        board = sample_start(
            "full",
            _rng(seed),
            buch_anteil=1.0,
            theorie=theorie,
        )
        if board.move_stack and board.move_stack[0] in (_E4_MOVE, _D4_MOVE):
            passende_erste_zuege += 1

    assert passende_erste_zuege >= 150
