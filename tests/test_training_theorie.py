from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

import chess
import numpy as np
import pytest

from Code.rl import tdleaf, train
from Code.rl.features import FEATURE_SIZE
from Code.rl.search import CLASSIC
from Code.rl.theorie import TheorieOptionen, endspiel_verfuegbar, theorie_fuer
from Code.rl.valuenet import ValueNet


def neues_netz() -> ValueNet:
    return ValueNet(input_size=FEATURE_SIZE, hidden_size=8, seed=0)


def test_ohne_theorie_unveraendert() -> None:
    net = neues_netz()
    argumente = dict(
        gamma=0.99,
        epsilon=0.0,
        max_plies=6,
        td_lambda=0.5,
        depth=2,
        options=CLASSIC,
    )

    ohne_theorie = tdleaf.play_episode_tdleaf(
        net,
        np.random.default_rng(23),
        **argumente,
    )
    mit_defaults = tdleaf.play_episode_tdleaf(
        net,
        np.random.default_rng(23),
        theorie=None,
        buch_anteil=0.0,
        endspiel=False,
        **argumente,
    )

    assert np.array_equal(ohne_theorie[0], mit_defaults[0])
    assert np.array_equal(ohne_theorie[1], mit_defaults[1])
    assert ohne_theorie[2:] == mit_defaults[2:]


class FakeTheorie:
    def __init__(self, wert: float) -> None:
        self.wert = wert

    def endspiel_wert(self, board: chess.Board) -> float | None:
        if len(board.piece_map()) <= 5:
            return self.wert
        return None


def fake_suche(
    board: chess.Board,
    net: ValueNet,
    depth: int,
    gamma: float,
    quiescence_plies: int,
    options,
    *,
    mit_pv: bool = True,
    zug_waehlen: Callable[[chess.Board], chess.Move] | None = None,
) -> SimpleNamespace:
    zug = zug_waehlen(board) if zug_waehlen is not None else next(iter(board.legal_moves))
    return SimpleNamespace(move=zug, pv=[zug] if mit_pv else [], value=0.0)


def test_endspiel_fake_theorie(monkeypatch: pytest.MonkeyPatch) -> None:
    start = chess.Board("8/8/8/4k3/8/8/3P4/4K2Q w - - 0 1")

    def suche_mit_pv(
        board: chess.Board,
        net: ValueNet,
        depth: int,
        gamma: float,
        quiescence_plies: int,
        options,
    ) -> SimpleNamespace:
        return fake_suche(board, net, depth, gamma, quiescence_plies, options)

    monkeypatch.setattr(tdleaf, "search", suche_mit_pv)
    _, _, result, plies = tdleaf.play_episode_tdleaf(
        neues_netz(),
        np.random.default_rng(0),
        gamma=0.99,
        epsilon=0.0,
        max_plies=6,
        td_lambda=0.0,
        depth=2,
        start=start,
        theorie=FakeTheorie(1.0),
        endspiel=True,
    )

    assert plies == 1
    assert result == "tabelle"


@pytest.mark.parametrize(
    ("wert", "erwartetes_vorzeichen"),
    [(1.0, -1.0), (-1.0, 1.0)],
)
def test_endspiel_fake_theorie_zielvorzeichen(
    monkeypatch: pytest.MonkeyPatch,
    wert: float,
    erwartetes_vorzeichen: float,
) -> None:
    start = chess.Board("8/8/8/4k3/8/8/3P4/4K2Q w - - 0 1")

    def suche_ohne_pv(
        board: chess.Board,
        net: ValueNet,
        depth: int,
        gamma: float,
        quiescence_plies: int,
        options,
    ) -> SimpleNamespace:
        return fake_suche(
            board,
            net,
            depth,
            gamma,
            quiescence_plies,
            options,
            mit_pv=False,
        )

    monkeypatch.setattr(tdleaf, "search", suche_ohne_pv)
    _, targets, result, plies = tdleaf.play_episode_tdleaf(
        neues_netz(),
        np.random.default_rng(0),
        gamma=0.99,
        epsilon=0.0,
        max_plies=6,
        td_lambda=0.0,
        depth=2,
        start=start,
        theorie=FakeTheorie(wert),
        endspiel=True,
    )

    assert plies == 1
    assert result == "tabelle"
    assert targets.shape == (1,)
    assert targets[0] * erwartetes_vorzeichen > 0


@pytest.mark.skipif(
    not endspiel_verfuegbar(),
    reason="Keine Syzygy-Endspieltabellen verfügbar",
)
def test_echte_tabellen(monkeypatch: pytest.MonkeyPatch) -> None:
    theorie = theorie_fuer(TheorieOptionen(eroeffnung=False, endspiel=True))
    assert theorie is not None
    start = chess.Board("8/8/8/4k3/8/8/8/4KQ2 w - - 0 1")
    zug = chess.Move.from_uci("e1d1")
    assert start.is_legal(zug)

    def suche_mit_koenigszug(
        board: chess.Board,
        net: ValueNet,
        depth: int,
        gamma: float,
        quiescence_plies: int,
        options,
    ) -> SimpleNamespace:
        return fake_suche(
            board,
            net,
            depth,
            gamma,
            quiescence_plies,
            options,
            mit_pv=False,
            zug_waehlen=lambda _: zug,
        )

    monkeypatch.setattr(tdleaf, "search", suche_mit_koenigszug)
    if True:
        _, targets, result, plies = tdleaf.play_episode_tdleaf(
            neues_netz(),
            np.random.default_rng(0),
            gamma=0.99,
            epsilon=0.0,
            max_plies=6,
            td_lambda=0.0,
            depth=2,
            start=start,
            theorie=theorie,
            endspiel=True,
        )

    assert result == "tabelle"
    assert plies >= 1
    assert targets.shape[0] >= 1
    assert targets[0] > 0


@pytest.mark.parametrize(
    "argumente",
    [
        ["--endspieltabellen", "--search-depth", "1"],
        ["--buch-anteil", "0.5", "--search-depth", "1"],
        ["--endspieltabellen", "--task", "kqk", "--search-depth", "2"],
    ],
)
def test_train_argumente(
    monkeypatch: pytest.MonkeyPatch,
    argumente: list[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["train", *argumente])

    with pytest.raises(SystemExit):
        train.main()
