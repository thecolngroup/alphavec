from typing import Callable
import numpy as np
import pandas as pd


TRADING_DAYS_YEAR = 252


def zero_commission(weights: pd.DataFrame, prices: pd.DataFrame) -> float:
    """Zero trading commission.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.

    Returns:
        Always returns 0.
    """
    return 0


def flat_commission(weights: pd.DataFrame, prices: pd.DataFrame, fee: float) -> float:
    """Flat commission applies a fixed fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee: Fixed fee per trade.

    Returns:
        Always returns fee.
    """
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    commissions = tx * fee
    return commissions.fillna(0)


def pct_commission(weights: pd.DataFrame, prices: pd.DataFrame, fee: float) -> float:
    """Percentage commission applies a percentage fee per trade.

    Args:
        weights: Weights of the assets in the portfolio.
        prices: Prices of the assets in the portfolio.
        fee: Percentage fee per trade.

    Returns:
        Returns a percentage of the total value of the trade.
    """
    size = weights.abs().diff().fillna(0)
    value = size * prices
    commissions = value * fee
    return commissions.fillna(0)


def backtest(
    strategy_weights: pd.DataFrame,
    mark_prices: pd.DataFrame,
    leverage: float = 1,
    freq_day: int = 1,
    commission_func: Callable[[pd.DataFrame, pd.DataFrame], float] = zero_commission,
    ann_borrow_pct: float = 0,
    spread_pct: float = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Backtests a trading strategy.

    Strategy is simulated using the give weights, prices, and cost parameters.
    Zero costs are calculated by default: no commission, borrowing, and no spread.

    Daily interval data is assumed by default.
    If you want to use a different interval, you must pass in the appropriate freq_day value
    e.g. if you are using hourly data in a 24-hour market such as crypto, you should pass in 24.

    Performance is reported both asset-wise and as a portfolio.
    Annualised metrics always use a 252 day trading year.

    Args:
        strategy_weights:
            Weights (-1 to 1) of the assets in the strategy at each interval.
            Each column should be the weights for a specific asset, with the column name being the asset name.
            Column names should match strategy_weights.
            Index should be a DatetimeIndex.
            Shape must match mark_prices.
        mark_prices:
            Mark prices used to calculate returns of the assets at each interval.
            The mark price should be the realistic price at which the asset can be traded each interval.
            Each column should be the mark prices for a specific asset, with the column name being the asset name.
            Column names should match strategy_weights.
            Index should be a DatetimeIndex.
            Shape must match strategy_weights.
        leverage: Leverage used in the strategy. Defaults to 1.
        freq_day: Number of strategy intervals in a trading day. Defaults to 1.
        commission_func: Function to calculate commission cost. Defaults to zero_commission.
        ann_borrow_pct: Annual borrowing cost percentage applied when leverage > 1. Defaults to 0.
        spread_pct: Spread cost percentage. Defaults to 0.

    Returns:
        A tuple containing five DataFrames that report backtest performance:
            1. Asset-wise performance.
            2. Asset-wise equity curve.
            3. Asset-wise rolling annual Sharpe ratio.
            4. Portfolio performance.
            5. Portoflio equity curve.
    """

    assert (
        strategy_weights.shape == mark_prices.shape
    ), "Weights and prices must have the same shape"
    assert (
        strategy_weights.columns.tolist() == mark_prices.columns.tolist()
    ), "Weights and prices must have the same column names"

    # Calc the number of data intervals in a trading year for annualised metrics
    freq_year = freq_day * TRADING_DAYS_YEAR

    # Backtest each asset so that we can assess the relative performance of the strategy
    # Asset returns approximate a baseline buy and hold scenario
    # Use pct returns rather than log returns since all costs are in pct terms too
    asset_rets = mark_prices.pct_change()
    asset_cum = (1 + asset_rets).cumprod() - 1
    asset_perf = pd.concat(
        [
            asset_rets.apply(_sharpe, periods=freq_year),
            asset_rets.apply(_vol, periods=freq_year),
            asset_rets.apply(_cagr, periods=freq_year),
            asset_rets.apply(_max_drawdown),
        ],
        keys=["sharpe", "volatility", "cagr", "max_drawdown"],
        axis=1,
    )

    # Backtest a cost-aware strategy as defined by the given weights:
    # 1. Calc costs
    # 2. Evaluate asset-wise performance
    # 3. Evalute portfolio performance

    # Adjust the weights for leverage
    strategy_weights *= leverage

    # Calc the number of valid trading periods for each asset
    # in order to support performance calcs over a ragged time series
    # with older and newer assets
    strat_valid_periods = strategy_weights.apply(
        lambda col: col.loc[col.first_valid_index() :].count()
    )
    strat_days = strat_valid_periods / freq_day

    # Calc each cost component in percentage terms so we can
    # deduct them from the strategy returns
    cmn_costs = commission_func(strategy_weights, mark_prices) / mark_prices
    borrow_costs = (
        _borrow(strategy_weights, mark_prices, (ann_borrow_pct / freq_year), leverage)
        / mark_prices
    )
    spread_costs = _spread(strategy_weights, mark_prices, spread_pct) / mark_prices
    costs = cmn_costs + borrow_costs + spread_costs

    # Evaluate the cost-aware strategy returns and key performance metrics
    # Shift the strategy weights by 1 period to prevent look-ahead bias
    strat_rets = strategy_weights.shift(1) * (asset_rets - costs)
    strat_cum = (1 + strat_rets).cumprod() - 1
    strat_profit_cost_ratio = strat_cum.iloc[-1] / costs.sum()
    strat_perf = pd.concat(
        [
            strat_rets.apply(_sharpe, periods=freq_year),
            strat_rets.apply(_vol, periods=freq_year),
            strat_rets.apply(_cagr, periods=freq_year),
            strat_rets.apply(_max_drawdown),
            _trade_count(strategy_weights) / strat_days,
            strat_profit_cost_ratio,
        ],
        keys=[
            "annual_sharpe",
            "annual_volatility",
            "cagr",
            "max_drawdown,",
            "trades_per_day",
            "profit_cost_ratio",
        ],
        axis=1,
    )

    # Evaluate the strategy portfolio performance
    port_rets = strat_rets.sum(axis=1)
    port_cum = strat_cum.sum(axis=1)
    port_profit_cost_ratio = port_cum.iloc[-1] / costs.sum().sum()
    port_perf = pd.DataFrame(
        {
            "annual_sharpe": _sharpe(port_rets, periods=freq_year),
            "annual_volatility": _vol(port_rets, periods=freq_year),
            "cagr": _cagr(port_rets, periods=freq_year),
            "max_drawdown": _max_drawdown(port_rets),
            "profit_cost_ratio": port_profit_cost_ratio,
        },
        index=["portfolio"],
    )

    # Combine the asset and strategy performance metrics
    # into a single dataframe for comparison
    perf = pd.concat(
        [asset_perf, strat_perf],
        keys=["asset", "strategy"],
        axis=1,
    )
    perf_cum = pd.concat(
        [asset_cum, strat_cum, port_cum],
        keys=["asset", "strategy", "portfolio"],
        axis=1,
    )
    perf_roll_sr = pd.concat(
        [
            _roll_sharpe(asset_rets, window=freq_year, periods=freq_year),
            _roll_sharpe(strat_rets, window=freq_year, periods=freq_year),
            _roll_sharpe(port_rets, window=freq_year, periods=freq_year),
        ],
        keys=["asset", "strategy", "portfolio"],
        axis=1,
    )

    return (
        perf,
        perf_cum,
        perf_roll_sr,
        port_perf,
        port_cum,
    )


def _sharpe(
    rets: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0,
    periods: int = TRADING_DAYS_YEAR,
) -> float:
    mu = rets.mean()
    sigma = rets.std()
    sr = (mu - risk_free_rate) / sigma
    return sr * np.sqrt(periods)


def _roll_sharpe(
    rets: pd.DataFrame | pd.Series,
    risk_free_rate: float = 0,
    window: int = TRADING_DAYS_YEAR,
    periods: int = TRADING_DAYS_YEAR,
) -> pd.DataFrame | pd.Series:
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()
    sr = (mu - risk_free_rate) / sigma
    return sr * np.sqrt(periods)


def _cagr(
    rets: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS_YEAR
) -> pd.DataFrame | pd.Series:
    cumprod = (1 + rets).cumprod().dropna()
    if len(cumprod) == 0:
        return 0

    final = cumprod.iloc[-1]
    if final <= 0:
        return 0

    n = len(cumprod) / periods
    cagr = final ** (1 / n) - 1

    return cagr


def _vol(
    rets: pd.DataFrame | pd.Series, periods: int = TRADING_DAYS_YEAR
) -> pd.DataFrame | pd.Series:
    return rets.std() * np.sqrt(periods)


def _max_drawdown(rets: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    cumprod = (1 + rets).cumprod()
    cummax = cumprod.cummax()
    max_drawdown = ((cummax - cumprod) / cummax).max()
    return max_drawdown


def _trade_count(weights: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    return tx.sum()


def _spread(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    spread_pct: float = 0,
) -> pd.DataFrame | pd.Series:
    diff = weights.abs().diff().fillna(0) != 0
    tx = diff.astype(int)
    costs = tx * (spread_pct * 0.5) * prices
    return costs.fillna(0)


def _borrow(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    borrow_pct: float = 0,
    lev: float = 1,
) -> pd.DataFrame | pd.Series:
    size = weights.abs().fillna(0)
    value = size * prices
    costs = value * borrow_pct * (lev - 1)
    return costs.fillna(0)
