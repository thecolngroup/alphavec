import sys
import os
from pathlib import PurePath
from functools import partial
import logging

import numpy as np
import pandas as pd

import alphavec.backtest as bt

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

    perf, _, _, _, _, _ = bt.backtest(
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

    _, _, perf_sr, _, _, _ = bt.backtest(
        weights,
        prices,
        freq_day=1,
        trading_days_year=252,
        shift_periods=1,
    )
    assert perf_sr.loc["2022-10-01T00:00:00.000", ("portfolio", 0)].round(2) == -0.74


def test_pct_commission():
    weights = pd.Series([0, np.nan, 0, 1, -2.5])
    prices = pd.Series([10, 10, 10, 10, 10])
    act = bt.pct_commission(weights, prices, 0.1)

    assert np.isnan(act.iloc[0])  # Case: no fee, NaN introduced due to diff()
    assert act.iloc[1] == 0  # Case: no fee, zero to NaN
    assert act.iloc[2] == 0  # Case: no fee, NaN to zero
    assert act.iloc[3] == 1.0  # Case: fee for zero to 1
    assert act.iloc[4] == 3.5  # Case: fee for 1 to -2.5


def test_turnover():
    weights = pd.Series([0, np.nan, 0, 1, -2.5])
    prices = pd.Series([10, 20, 40, 80, 40])
    returns = weights * prices.pct_change()

    act = bt._ann_turnover(returns, weights)
    logging.info(act)


def test_spread():
    weights = pd.Series([np.nan, 0.5, -2.5])
    prices = pd.Series([10, 10, 10])

    act = bt._spread(weights, prices, 0.02)
    logging.info(act)
    assert act.iloc[2] == 0.3  # Case: spread cost


def test_borrow():
    weights = pd.Series([0.5, -2.5])
    prices = pd.Series([10, 10])
    rate = 0.1
    periods = 10

    act = bt._borrow(weights, prices, rate, periods)
    assert act.iloc[0] == 0  # Case: zero leverage
    assert act.iloc[1].round(2) == 0.36  # Case: weight with leverage
