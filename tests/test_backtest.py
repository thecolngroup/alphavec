import sys
import os
from pathlib import PurePath
from functools import partial
import logging

import numpy as np
import pandas as pd

from alphavec.backtest import backtest

workspace_root = str(PurePath(os.getcwd()))
sys.path.append(workspace_root)


def ohlcv_from_csv(filename):
    return pd.read_csv(
        filename,
        index_col=["symbol", "dt"],
        parse_dates=["dt"],
        dtype={
            "o": np.float64,
            "h": np.float64,
            "l": np.float64,
            "c": np.float64,
            "v": np.float64,
        },
        dayfirst=True,
    )


def load_close_prices(symbols: list):
    prices_filename = f"{workspace_root}/tests/testdata/binance-margin-1d.csv"
    market = ohlcv_from_csv(prices_filename)
    market = market[~market.index.duplicated()]
    market = market.unstack(level=0).sort_index(axis=1).stack()
    prices = pd.DataFrame(
        market.loc[:, ["c"]].unstack(level=1).droplevel(level=0, axis=1)
    )[symbols]
    return prices


def test_backtest_fixed_weights():
    """Assert that asset performance is equal to strategy performance when using fixed weights."""

    prices = load_close_prices(["BTCUSDT"])
    weights = prices.copy()
    weights[:] = 1

    perf, _, _, _, _, _ = backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        shift_periods=0,
    )
    assert (
        perf.loc["BTCUSDT", ("asset", "annual_sharpe")]
        == perf.loc["BTCUSDT", ("strategy", "annual_sharpe")]
    )


def test_backtest_external_validation():
    """Assert that portfolio performance is equal to a known external source (portfolioslab.com)."""
    prices = load_close_prices(["ETHUSDT", "BTCUSDT"])
    weights = prices.copy()
    weights[:] = 0.5

    _, _, perf_sr, _, _, _ = backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        shift_periods=1,
    )
    assert perf_sr.loc["2022-10-01T00:00:00.000", ("portfolio", 0)].round(2) == -0.74
