from typing import Callable
import numpy as np
import pandas as pd


TRADING_DAYS_YEAR = 252


def zero_commission(weights: pd.DataFrame, prices: pd.DataFrame) -> float:
    """
    Zero commission will always return 0.

    Parameters:
    weights (pd.DataFrame): The weights of the assets in the portfolio.
    prices (pd.DataFrame): The prices of the assets in the portfolio.

    Returns:
    float: Commission cost is 0.
    """
    return 0


def backtest(
    strategy_weights: pd.DataFrame,
    mark_prices: pd.DataFrame,
    leverage: float = 1,
    freq_day: int = 1,
    commission_func: Callable[[pd.DataFrame, pd.DataFrame], float] = zero_commission,
    ann_borrow_pct: float = 0,
    spread_pct: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Backtests a trading strategy based on provided weights, prices, and cost parameters.
    Zero costs are calculated by default: no borrowing, and no spread.

    Daily interval data is assumed by default.
    If you want to use a different interval, you must pass in the appropriate freq_day value
    e.g. if you are using hourly data in a 24-hour market such as crypto, you should pass in 24.

    Performance is reported both asset-wise and as a portfolio.
    A baseline zero-cost buy-and-hold comparison is provided for each asset and the portfolio.
    Note: the comparative baseline portfolio is formed using equal-weights of the baseline asset returns
    i.e. 1 / number of assets.
    Annualised metrics always use a 252 day trading year.

    Parameters:
    strategy_weights (pd.DataFrame): Weights (-1 to 1) of the assets in the strategy at each interval.
    Each column should be the weights for a specific asset, with the column name being the asset name.
    Column names should match strategy_weights.
    Index should be a DatetimeIndex.
    Shape must match mark_prices.

    mark_prices (pd.DataFrame):
    Mark prices used to calculate returns of the assets at each interval.
    The mark price should be the realistic price at which the asset can be traded each interval.
    Each column should be the mark prices for a specific asset, with the column name being the asset name.
    Column names should match strategy_weights.
    Index should be a DatetimeIndex.
    Shape must match strategy_weights.

    leverage (float, optional): Leverage used in the strategy. Defaults to 1.

    freq_day (int, optional):
    Number of strategy intervals in a trading day. Defaults to 1.

    commission_func (Callable[[pd.DataFrame, pd.DataFrame], float], optional):
    Function to calculate commission cost. Defaults to zero_commission.

    ann_borrow_pct (float, optional):
    Annual borrowing cost percentage applied when leverage > 1. Defaults to 0.

    spread_pct (float, optional): Spread cost percentage. Defaults to 0.

    Returns:
    tuple: A tuple containing five DataFrames that report backtest performance:
        - perf: Asset-wise performance.
        - perf_cum: Asset-wise equity curve.
        - perf_roll_sr: Asset-wise rolling annual Sharpe ratio.
        - port_perf: Portfolio performance.
        - port_cum: Portoflio equity curve.
    """

    assert (
        strategy_weights.shape == mark_prices.shape
    ), "Weights and prices must have the same shape"
    assert (
        strategy_weights.columns == mark_prices.columns
    ), "Weights and prices must have the same column names"

    # Calc the number of data intervals in a trading year for annualised metrics
    freq_year = freq_day * TRADING_DAYS_YEAR

    # Backtest a baseline buy and hold scenario for each asset so that we can assess
    # the relative performance of the strategy
    # Use pct returns rather than log returns since all costs are in pct terms too
    asset_rets = mark_prices.pct_change()
    asset_cum = (1 + asset_rets).cumprod() - 1
    asset_perf = pd.concat(
        [
            asset_rets.apply(ann_sharpe, periods=freq_year),
            asset_rets.apply(ann_vol, periods=freq_year),
            asset_rets.apply(cagr, periods=freq_year),
        ],
        keys=["sharpe", "volatility", "cagr"],
        axis=1,
    )

    # Evaluate the baseline portfolio performance using a naive equal-weighted approach
    asset_port_weights = 1 / len(strategy_weights.columns)
    baseline_port_rets = asset_rets.mul(asset_port_weights).sum(axis=1)
    baseline_port_cum = asset_cum.mul(asset_port_weights).sum(axis=1)
    baseline_port_perf = pd.DataFrame(
        {
            "sharpe": ann_sharpe(baseline_port_rets, periods=freq_year),
            "volatility": ann_vol(baseline_port_rets, periods=freq_year),
            "cagr": cagr(baseline_port_rets, periods=freq_year),
            "profit_cost_ratio": np.NAN,
        },
        index=["baseline"],
    )

    # Backtest a cost-aware strategy as defined by the given weights.
    # 1. Calc costs
    # 2. Evaluate asset-wise performance
    # 3. Evalute portfolio performance

    # Adjust the weights for leverage
    strategy_weights *= leverage

    # Calc the number of valid trading periods for each asset
    # in order to support performance calcs over a ragged time series
    #  with older and newer assets
    strat_valid_periods = strategy_weights.apply(
        lambda col: col.loc[col.first_valid_index() :].count()
    )
    strat_days = strat_valid_periods / freq_day

    # Calc each cost component in percentage terms so we can
    # deduct them from the strategy returns
    cmn_costs = commission_func(strategy_weights, mark_prices) / mark_prices
    borrow_costs = (
        borrow(strategy_weights, mark_prices, (ann_borrow_pct / freq_year), leverage)
        / mark_prices
    )
    spread_costs = spread(strategy_weights, mark_prices, spread_pct) / mark_prices
    costs = cmn_costs + borrow_costs + spread_costs

    # Evaluate the cost-aware strategy returns and key performance metrics
    # Shift the strategy weights by 1 period to prevent look-ahead bias
    strat_rets = strategy_weights.shift(1) * (asset_rets - costs)
    strat_cum = (1 + strat_rets).cumprod() - 1
    profit_cost_ratio = strat_cum.iloc[-1] / costs.sum()
    strat_perf = pd.concat(
        [
            strat_rets.apply(ann_sharpe, periods=freq_year),
            strat_rets.apply(ann_vol, periods=freq_year),
            strat_rets.apply(cagr, periods=freq_year),
            trade_count(strategy_weights) / strat_days,
            profit_cost_ratio,
        ],
        keys=["sharpe", "volatility", "cagr", "trades_per_day", "profit_cost_ratio"],
        axis=1,
    )

    # Combine the baseline and strategy asset-wise performance metrics
    # into a single dataframe for comparison
    perf = pd.concat([asset_perf, strat_perf], keys=["baseline", "strat"], axis=1)
    perf_cum = pd.concat([asset_cum, strat_cum], keys=["baseline", "strat"], axis=1)
    perf_roll_sr = pd.concat(
        [
            roll_sharpe(asset_rets, window=freq_day, periods=freq_year),
            roll_sharpe(strat_rets, window=freq_day, periods=freq_year),
        ],
        keys=["baseline", "strat"],
        axis=1,
    )

    # Evaluate the strategy portfolio performance
    strat_port_rets = strat_rets.sum(axis=1)
    strat_port_cum = strat_cum.sum(axis=1)
    strat_port_perf = pd.DataFrame(
        {
            "sharpe": ann_sharpe(strat_port_rets, periods=freq_year),
            "volatility": ann_vol(strat_port_rets, periods=freq_year),
            "cagr": cagr(strat_port_rets, periods=freq_year),
            "profit_cost_ratio": profit_cost_ratio.sum().sum(),
        },
        index=["strat"],
    )

    # Combine the baseline and strategy portfolio performance metrics
    # into a single dataframe for comparison
    port_perf = pd.concat([baseline_port_perf, strat_port_perf], axis=0)
    port_cum = pd.concat(
        [baseline_port_cum, strat_port_cum], keys=["baseline", "strat"], axis=1
    )

    return (
        perf,
        perf_cum,
        perf_roll_sr,
        port_perf,
        port_cum,
    )


def ann_sharpe(
    returns: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0,
    periods: int = TRADING_DAYS_YEAR,
):
    mu = returns.mean()
    sigma = returns.std()
    sr = (mu - risk_free_rate) / sigma
    return sr * np.sqrt(periods)


def roll_sharpe(rets, rf=0, window=TRADING_DAYS_YEAR, periods=TRADING_DAYS_YEAR):
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()
    sr = (mu - rf) / sigma
    return sr * np.sqrt(periods)


def cagr(rets, periods=TRADING_DAYS_YEAR):
    cumprod = (1 + rets).cumprod().dropna()
    if len(cumprod) == 0:
        return 0

    final = cumprod.iloc[-1]
    if final <= 0:
        return 0

    n = len(cumprod) / periods
    cagr = final ** (1 / n) - 1

    return cagr


def ann_vol(rets, periods=TRADING_DAYS_YEAR):
    return rets.std() * np.sqrt(periods)


def trade_count(weights):
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    return tx.sum()


def spread(weights, prices, spread_pct):
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    costs = tx * (spread_pct * 0.5) * prices
    return costs.fillna(0)


def borrow(weights, prices, borrow_pct, lev: float = 1):
    size = weights.abs().fillna(0)
    value = size * prices
    costs = value * borrow_pct * (lev - 1)
    return costs.fillna(0)


def flat_commission(weights, prices, fee):
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    commissions = tx * fee
    return commissions.fillna(0)


def pct_commission(weights, prices, fee):
    size = weights.abs().diff().fillna(0)
    value = size * prices
    commissions = value * fee
    return commissions.fillna(0)
