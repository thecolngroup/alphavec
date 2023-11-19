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
    By default calculates no commission, no borrowing, and no spread.
    By default assumes weights and strategy prices are at day interval. If you want to use a different interval,
    you must pass in the appropriate freq_day value. 
    E.G. if you are using hourly data in a 24-hour market such as crypto, you should pass in 24.
    Performance is reported on a per-asset basis and also as a combined portfolio.
    Annualised metrics always assume a 252 day trading year.
    Note: the portfolio report assumes an equal-weighted approach.

    Parameters:
    strategy_weights (pd.DataFrame): The weights (-1 to 1) of the assets in the strategy at each interval.
    Shape must match that of mark_prices.
    mark_prices (pd.DataFrame): The mark prices of the assets at each interval.
    Shape must match that of strategy_weights.
    leverage (float, optional): The leverage used in the strategy. Defaults to 1.
    freq_day (int, optional): The number of strategy intervals in a trading day. Defaults to TRADING_DAYS_YEAR.
    commission_func (Callable[[pd.DataFrame, pd.DataFrame], float], optional): A function that calculates the commission. Defaults to zero_commission.
    ann_borrow_pct (float, optional): The annual borrowing percentage applied when leverage > 1. Defaults to 0.
    spread_pct (float, optional): The spread percentage. Defaults to 0.

    Returns:
    tuple: A tuple containing five DataFrames for :
        - strat_perf: Asset-wise performance of the strategy.
        - strat_perf_cum: Asset-wise equity curve of the strategy.
        - strat_perf_roll_sr: Asset-wise rolling annual Sharpe ratio of the strategy.
        - strat_port_perf: Performance of the portfolio.
        - strat_port_cum: Equity curve of the portfolio.
    """

    assert strategy_weights.shape == mark_prices.shape, "Weights and prices must have the same shape"

    # Calc the number of data intervals in a trading year for annualised metrics
    freq_year = freq_day * TRADING_DAYS_YEAR

    # Backtest a baseline buy and hold scenario for each asset so that we can assess
    # the relative performance of the strategy
    asset_rets = mark_prices.pct_change()
    asset_cum = (1 + asset_rets).cumprod() - 1
    asset_perf = pd.concat(
        [
            asset_rets.apply(ann_sharpe, periods=freq_year),
            asset_rets.apply(ann_vol, periods=freq_year),
            asset_rets.apply(cagr, periods=freq_year),
        ],
        keys=["sr", "vol", "cagr"],
        axis=1,
    )

    # Evaluate the baseline portfolio performance using a naive equal-weighted approach
    asset_port_weights = 1 / len(strategy_weights.columns)
    baseline_port_rets = asset_rets.mul(asset_port_weights).sum(axis=1)
    baseline_port_cum = asset_cum.mul(asset_port_weights).sum(axis=1)
    baseline_port_perf = pd.DataFrame(
        {
            "sr": ann_sharpe(baseline_port_rets, periods=freq_year),
            "vol": ann_vol(baseline_port_rets, periods=freq_year),
            "cagr": cagr(baseline_port_rets, periods=freq_year),
            "profit_cost_ratio": np.NAN,
        },
        index=["baseline"],
    )

    # Backtest a cost-aware strategy as defined by the given weights.
    # 1. Calc costs
    # 2. Evaluate strategy performance per-asset
    # 3. Evalute portfolio performance

    # Adjust the weights for leverage
    strategy_weights *= leverage

    # Calc the number of valid trading periods for each asset
    # this supports certain performance calcs over a ragged series of prices with older and newer assets
    strat_start_index = strategy_weights.apply(
        lambda col: col.loc[col.first_valid_index() :].count()
    )
    strat_years = strat_start_index / freq_year

    # Calc each cost component in percentage terms so we can deduct them from the strategy returns
    cmn_costs = commission_func(strategy_weights, mark_prices) / mark_prices
    borrow_costs = (
        borrow(strategy_weights, mark_prices, (ann_borrow_pct / freq_year), leverage)
        / mark_prices
    )
    spread_costs = spread(strategy_weights, spread_pct)
    costs = cmn_costs + borrow_costs + spread_costs

    # Evaluate the cost-aware strategy returns and key performance metrics
    strat_rets = strategy_weights.shift(1) * (asset_rets - costs)
    strat_cum = (1 + strat_rets).cumprod() - 1
    profit_cost_ratio = strat_cum.iloc[-1] / costs.sum()
    strat_perf = pd.concat(
        [
            strat_rets.apply(ann_sharpe, periods=freq_year),
            strat_rets.apply(ann_vol, periods=freq_year),
            strat_rets.apply(cagr, periods=freq_year),
            trade_count(strategy_weights) / strat_years,
            profit_cost_ratio,
        ],
        keys=["sr", "vol", "cagr", "trades_pa", "profit_cost_ratio"],
        axis=1,
    )

    # Combine the baseline and strategy per-asset performance metrics into a single dataframe for comparison
    strat_perf = pd.concat([asset_perf, strat_perf], keys=["baseline", "strat"], axis=1)
    strat_perf_cum = pd.concat(
        [asset_cum, strat_cum], keys=["baseline", "strat"], axis=1
    )
    strat_perf_roll_sr = pd.concat(
        [
            roll_sharpe(asset_rets, window=freq_day, periods=freq_year),
            roll_sharpe(strat_rets, window=freq_day, periods=freq_year),
        ],
        keys=["baseline", "strat"],
        axis=1,
    )

    # Evaluate the strategy portfolio performance
    strat_port_rets = strat_rets.mul(asset_port_weights).sum(axis=1)
    strat_port_cum = strat_cum.mul(asset_port_weights).sum(axis=1)
    strat_port_perf = pd.DataFrame(
        {
            "sr": ann_sharpe(strat_port_rets, periods=freq_year),
            "vol": ann_vol(strat_port_rets, periods=freq_year),
            "cagr": cagr(strat_port_rets, periods=freq_year),
            "profit_cost_ratio": profit_cost_ratio.mul(asset_port_weights).sum().sum(),
        },
        index=["strat"],
    )

    # Combine the baseline and strategy portfolio performance metrics into a single dataframe for comparison
    strat_port_perf = pd.concat([baseline_port_perf, strat_port_perf], axis=0)
    strat_port_cum = pd.concat(
        [baseline_port_cum, strat_port_cum], keys=["baseline", "strat"], axis=1
    )

    return (
        strat_perf,
        strat_perf_cum,
        strat_perf_roll_sr,
        strat_port_perf,
        strat_port_cum,
    )


def ann_sharpe(returns:pd.DataFrame | pd.Series, risk_free_rate:float=0, periods:int=TRADING_DAYS_YEAR):
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


def spread(weights, spread_pct):
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    costs = tx * (spread_pct * 0.5)
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
