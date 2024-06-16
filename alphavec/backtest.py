"""Backtest module for evaluating trading strategies."""

from audioop import mul
import logging
from typing import Callable, Tuple, Union

import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length


logger = logging.getLogger(__name__)

DEFAULT_TRADING_DAYS_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02


def zero_commission(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Zero trading commission.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.

    Returns:
        Always returns 0.
    """
    return pd.DataFrame(0, index=weights.index, columns=weights.columns)


def flat_commission(
    weights: pd.DataFrame, prices: pd.DataFrame, fee: float
) -> pd.DataFrame:
    """Flat commission applies a fixed fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee: Fixed fee per trade.

    Returns:
        Fixed fee per trade.
    """
    diff = weights.fillna(0).diff().abs() != 0
    tx = diff.astype(int)
    commissions = tx * fee
    return commissions


def pct_commission(
    weights: pd.DataFrame, prices: pd.DataFrame, fee: float
) -> pd.DataFrame:
    """Percentage commission applies a percentage fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee: Percentage fee per trade.

    Returns:
        Returns a percentage of the total value of the trade.
    """
    size = weights.fillna(0).diff().abs()
    value = size * prices
    commissions = value * fee
    return commissions


def nav(log_rets: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the cumulative net asset value (NAV) from log returns.

    Use this function in conjunction with log returns from the backtest.
    E.G. to calcuate portfolio value based on an initial investment of 1000:
    equity_in_currency_units = 1000 * nav(port_rets)

    Args:
        log_rets: Log returns of the assets in the portfolio.

    Returns:
        Cumulative net asset value (NAV) of the portfolio.
    """
    return np.exp(log_rets).cumprod()  # type: ignore


def backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    freq_day: int = 1,
    trading_days_year: int = DEFAULT_TRADING_DAYS_YEAR,
    shift_periods: int = 1,
    commission_func: Callable[
        [pd.DataFrame, pd.DataFrame], pd.DataFrame
    ] = zero_commission,
    ann_borrow_rate: float = 0,
    spread_pct: float = 0,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Backtest a trading strategy.

    Strategy is simulated using the given weights, prices, and cost parameters.
    Zero costs are calculated by default: no commission, no borrowing, no spread.

    To prevent look-ahead bias by default the returns are shifted 1 period relative to the weights during backtest.
    The default shift assumes close prices and an ability to trade at the close,
    this is reasonable for 24 hour markets such as crypto, but not for traditional markets with fixed trading hours.
    For traditional markets, you should set shift periods to at least 2.

    Daily periods are default.
    If your prices and weights have a different periodocity pass in the appropriate freq_day value.
    E.G. for 8 hour periods in a 24-hour market such as crypto, you should pass in 3.

    Performance is reported both asset-wise and as a portfolio.
    Annualized metrics use the default trading days per year of 252.

    Args:
        weights:
            Weights (e.g. -1 to +1) of the assets at each period.
            Each column should be the weights for a specific asset, with column name = asset name.
            Column names should match prices.
            Index should be a DatetimeIndex.
            Shape must match prices.
        prices:
            Prices of the assets at each period used to calculate returns and costs.
            Each column should be the mark prices for a specific asset, with column name = asset name.
            Column names should match weights.
            Index should be a DatetimeIndex.
            Shape must match weights.
        freq_day: Number of periods in a trading day. Defaults to 1 for daily data.
        trading_days_year: Number of trading days in a year. Defaults to 252.
        shift_periods: Positive integer for n periods to shift returns relative to weights. Defaults to 1.
        commission_func: Function to calculate commission cost. Defaults to zero_commission.
        ann_borrow_rate: Annualized borrowing rate applied when asset weight > 1. Defaults to 0.
        spread_pct: Spread cost percentage. Defaults to 0.
        ann_risk_free_rate: Annualized risk-free rate used to calculate Sharpe ratio. Defaults to 0.02.

    Returns:
        A tuple containing five data sets:
            1. Asset-wise performance table
            2. Asset-wise equity curves
            3. Asset-wise rolling annualized Sharpes
            4. Portfolio performance table
            5. Portfolio (log) returns
    """

    assert weights.shape == prices.shape, "Weights and prices must have the same shape"
    assert (
        weights.columns.tolist() == prices.columns.tolist()
    ), "Weights and prices must have the same column (asset) names"

    # Calc the number of data intervals in a trading year for annualized metrics
    freq_year = freq_day * trading_days_year

    # Backtest each asset so that we can assess the relative performance of the strategy
    # Asset returns approximate a baseline buy and hold scenario
    # Truncate the asset returns to account for shifting to ensure the asset and strategy performance is comparable.
    asset_rets = _log_rets(prices)
    asset_rets = asset_rets.iloc[:-shift_periods] if shift_periods > 0 else asset_rets
    asset_nav = nav(asset_rets)

    asset_perf = pd.concat(
        [
            _ann_sharpe(
                asset_rets, freq_year=freq_year, ann_risk_free_rate=ann_risk_free_rate
            ),
            _ann_vol(asset_rets, freq_year=freq_year),
            _cagr(asset_rets, freq_year=freq_year),
            _max_drawdown(asset_rets),
        ],  # type: ignore
        keys=["annual_sharpe", "annual_volatility", "cagr", "max_drawdown"],
        axis=1,
    )

    # Backtest a cost-aware strategy as defined by the given weights:
    # 1. Calc costs
    # 2. Evaluate asset-wise performance
    # 3. Evaluate portfolio performance

    # Calc each cost component in percentage terms so we can
    # deduct them from the strategy returns
    cmn_costs = commission_func(weights, prices) / prices
    borrow_costs = _borrow(weights, prices, ann_borrow_rate, freq_day) / prices
    spread_costs = _spread(weights, prices, spread_pct) / prices
    costs = cmn_costs + borrow_costs + spread_costs

    # Evaluate the cost-aware strategy returns and key performance metrics
    # Use the shift arg to prevent look-ahead bias
    # Truncate the returns to remove the empty intervals resulting from the shift
    strat_rets = _log_rets(prices) - costs
    strat_rets = weights * strat_rets.shift(-shift_periods)
    strat_rets = strat_rets.iloc[:-shift_periods] if shift_periods > 0 else strat_rets
    strat_nav = nav(strat_rets)

    # Calc the number of valid trading periods for each asset
    strat_valid_periods = weights.apply(
        lambda col: col.loc[col.first_valid_index() :].count()
    )
    strat_total_days = strat_valid_periods / freq_day

    # Calc the annual turnover for each asset
    strat_ann_turnover = _turnover(weights, strat_rets) * (
        trading_days_year / strat_total_days
    )

    # Evaluate the strategy asset-wise performance
    strat_perf = pd.concat(
        [
            _ann_sharpe(
                strat_rets, freq_year=freq_year, ann_risk_free_rate=ann_risk_free_rate
            ),
            _ann_vol(strat_rets, freq_year=freq_year),
            _cagr(strat_rets, freq_year=freq_year),
            _max_drawdown(strat_rets),
            strat_ann_turnover,
        ],  # type: ignore
        keys=[
            "annual_sharpe",
            "annual_volatility",
            "cagr",
            "max_drawdown",
            "annual_turnover",
        ],
        axis=1,
    )

    # Evaluate the strategy portfolio performance
    port_rets = strat_rets.sum(axis=1)
    port_nav = nav(port_rets)

    # Approximate the portfolio turnover as the weighted average sum of the asset-wise turnover
    port_ann_turnover = (strat_ann_turnover * weights.mean().abs()).sum()

    port_perf = pd.DataFrame(
        {
            "annual_sharpe": _ann_sharpe(
                port_rets, freq_year=freq_year, ann_risk_free_rate=ann_risk_free_rate
            ),
            "annual_volatility": _ann_vol(port_rets, freq_year=freq_year),
            "cagr": _cagr(port_rets, freq_year=freq_year),
            "max_drawdown": _max_drawdown(port_rets),
            "annual_turnover": port_ann_turnover,
        },
        index=["portfolio"],
    )

    # Combine the asset and strategy performance metrics into a single dataframe for comparison
    perf = pd.concat(
        [asset_perf, strat_perf],
        keys=["asset", "strategy"],
        axis=1,
    )

    perf_nav = pd.concat(
        [port_nav, asset_nav, strat_nav],
        keys=["portfolio", "asset", "strategy"],
        axis=1,
    ).rename(columns={0: "NAV"})

    perf_roll_sr = pd.concat(
        [
            _ann_roll_sharpe(
                port_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                asset_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                strat_rets,
                window=freq_year,
                freq_year=freq_year,
                ann_risk_free_rate=ann_risk_free_rate,
            ),
        ],
        keys=["portfolio", "asset", "strategy"],
        axis=1,
    ).rename(columns={0: "NAV"})

    return (
        perf,
        perf_nav,
        perf_roll_sr,
        port_perf,
        port_rets,
    )


def random_window_test(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    backtest_func: Callable[
        [pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series],
    ],
    test_n: int = 1000,
    window_size: int = DEFAULT_TRADING_DAYS_YEAR,
    allow_nan: bool = False,
    seed: int = 1,
) -> pd.DataFrame:
    """Random window test samples random contiguous periods (windows) to evaluate the robustness of a strategy.

    Goal is to test the strategy across different market regimes.
    See the other test functions for bootstrap and monte carlo tests.

    Args:
        weights: Weights of the assets in the portfolio (see backtest).
        prices: Prices of the assets in the portfolio (see backtest).
        backtest_func: Function to backtest the strategy that accepts weights and prices.
        test_n: Number of random contiguous samples to test. Defaults to 1000.
        window_size: Size in periods of each window. Defaults to DEFAULT_TRADING_DAYS_YEAR.
        allow_nan: Rejects a sample if NaN is found in weights or prices. Defaults to False.
        seed: Seed to reproduce results. Defaults to 1.

    Returns:
        Dataframe of portfolio performance for each window.
    """

    assert weights.shape == prices.shape, "Weights and prices must have the same shape"

    assert (
        len(prices) > window_size
    ), "Weights and prices must have more than sample_length periods"

    results = {}
    rs = RandomState(MT19937(SeedSequence(seed)))

    for i in range(test_n):
        start = rs.randint(0, len(prices) - window_size)

        sample_prices = prices.iloc[start : start + window_size].copy()
        sample_weights = weights.loc[sample_prices.index].copy()

        if not allow_nan:
            if sample_prices.isna().any().any() or sample_weights.isna().any().any():
                logging.debug(f"Skipping sample {i} due to NaN values")
                continue

        _, _, _, port_perf, _ = backtest_func(sample_weights, sample_prices)
        results[i] = port_perf

    return pd.concat(results).droplevel(1)


def monte_carlo_test(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    backtest_func: Callable[
        [pd.DataFrame, pd.DataFrame],
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series],
    ],
    test_n: int = 1000,
    seed: int = 1,
) -> pd.DataFrame:
    """Monte carlo test simulates synthetic price series to evaluate the robustness of a strategy.

    Args:
        weights: Weights of the assets in the portfolio (see backtest).
        prices: Prices of the assets in the portfolio (see backtest).
        backtest_func: Function to backtest the strategy that accepts weights and prices.
        test_n: Number of sythetic price series to test. Defaults to 1000.
        seed: Seed to reproduce results. Defaults to 1.
    Returns:
        Dataframe of portfolio performance for each sample.
    """

    assert weights.shape == prices.shape, "Weights and prices must have the same shape"

    results = {}

    rets = _log_rets(prices)
    rs = RandomState(MT19937(SeedSequence(seed)))

    for i in range(test_n):
        sim_rets = rets.apply(lambda x: rs.choice(x, x.shape))  # type: ignore
        sim_prices = 1 * nav(sim_rets)
        _, _, _, port_perf, _ = backtest_func(weights, sim_prices)
        results[i] = port_perf

    return pd.concat(results).droplevel(1)


def _log_rets(
    data: Union[pd.DataFrame, pd.Series],
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate log returns from data."""
    return np.log(data / data.shift(1))  # type: ignore


def _ann_to_period_rate(ann_rate: float, periods_year: int) -> float:
    """Calculate the annualized rate given the return periodocity."""
    return (1 + ann_rate) ** (1 / periods_year) - 1


def _ann_sharpe(
    rets: Union[pd.DataFrame, pd.Series],
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.Series:
    """Calculate annualized Sharpe ratio."""
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.mean()
    sigma = rets.std()
    sr = (mu - rfr) / sigma
    return sr * np.sqrt(freq_year)


def _ann_roll_sharpe(
    rets: Union[pd.DataFrame, pd.Series],
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    window: int = DEFAULT_TRADING_DAYS_YEAR,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate rolling annualized Sharpe ratio."""
    rfr = _ann_to_period_rate(ann_risk_free_rate, freq_year)
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()
    sr = (mu - rfr) / sigma
    return sr * np.sqrt(freq_year)


def _ann_vol(
    rets: Union[pd.DataFrame, pd.Series], freq_year: int = DEFAULT_TRADING_DAYS_YEAR
) -> pd.Series:
    """Calculate annualized volatility."""
    return rets.std() * np.sqrt(freq_year)


def _cagr(
    log_rets: Union[pd.DataFrame, pd.Series],
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> Union[pd.Series, float]:
    """Calculate CAGR."""
    n_years = len(log_rets) / freq_year
    final = np.exp(log_rets.sum()) - 1
    cagr = (1 + final) ** (1 / n_years) - 1
    return cagr  # type: ignore


def _max_drawdown(log_rets: Union[pd.DataFrame, pd.Series]) -> Union[pd.Series, float]:
    """Calculate the max drawdown in pct."""
    curve = nav(log_rets)
    hwm = curve.cummax()
    dd = (curve - hwm) / hwm
    return dd.min()  # type: ignore


def _turnover(
    weights: Union[pd.DataFrame, pd.Series],
    log_rets: Union[pd.DataFrame, pd.Series],
) -> pd.Series | float:
    """Calculate the turnover for each position in the strategy."""
    # Assume capital of 1000
    capital = 1000
    # Calculate the delta of the weight between each interval
    # Buy will be +ve, sell will be -ve
    diff = weights.fillna(0).diff()
    # Capital is fixed (uncompounded) for each interval so we can calculate the trade volume
    # Sum the volume of the buy and sell trades
    buy_volume = (diff.where(lambda x: x.gt(0), 0).abs() * capital).sum()
    sell_volume = (diff.where(lambda x: x.lt(0), 0).abs() * capital).sum()
    # Trade volume is the minimum of the buy and sell volumes
    # Wrap in Series in case of scalar volume sum (when weights is a Series)
    trade_volume = pd.concat(
        [pd.Series(buy_volume), pd.Series(sell_volume)], axis=1
    ).min(axis=1)
    # Calculate the average portfolio value
    # Finally take the ratio of trading volume to mean portfolio value
    nav_mu = (capital * nav(log_rets)).mean()
    turnover = trade_volume / nav_mu
    return turnover


def _spread(
    weights: Union[pd.DataFrame, pd.Series],
    prices: Union[pd.DataFrame, pd.Series],
    spread_pct: float = 0,
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the spread costs for each position in the strategy."""
    size = weights.fillna(0).diff().abs()
    value = size * prices
    costs = value * (spread_pct * 0.5)
    return costs


def _borrow(
    weights: Union[pd.DataFrame, pd.Series],
    prices: Union[pd.DataFrame, pd.Series],
    ann_borrow_rate: float = 0,
    freq_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the borrowing costs for each position in the strategy."""
    rate = _ann_to_period_rate(ann_borrow_rate, freq_year)
    # Position value from absolute weights and prices
    size = weights.abs().fillna(0)
    value = size * prices
    # Leverage is defined as an absolute weight > 1
    # Zero for all other positions
    lev = (size - 1).clip(lower=0)
    # Costs are the product of the position value, rate and leverage
    costs = value * rate * lev
    return costs
