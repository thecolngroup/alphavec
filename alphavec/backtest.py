from typing import Callable
import numpy as np
import pandas as pd


DEFAULT_TRADING_DAYS_YEAR = 252
DEFAULT_RISK_FREE_RATE = 0.02


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
        Fixed fee per trade.
    """
    diff = weights.fillna(0).diff().abs() != 0
    tx = diff.astype(int)
    commissions = tx * fee
    return commissions


def pct_commission(weights: pd.DataFrame, prices: pd.DataFrame, fee: float) -> float:
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


def backtest(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    freq_day: int = 1,
    trading_days_year: int = DEFAULT_TRADING_DAYS_YEAR,
    shift_periods: int = 1,
    commission_func: Callable[[pd.DataFrame, pd.DataFrame], float] = zero_commission,
    ann_borrow_rate: float = 0,
    spread_pct: float = 0,
    ann_risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    """Backtest a trading strategy.

    Strategy is simulated using the given weights, returns, and cost parameters.
    Zero costs are calculated by default: no commission, no borrowing, no spread.

    To prevent look-ahead bias by default the returns will be shifted 1 interval
    relative to the weights during backtest.

    Daily interval data is assumed by default.
    If you wish to use a different interval pass in the appropriate freq_day value
    e.g. if you are using hourly data in a 24-hour market such as crypto, you should pass in 24.

    Performance is reported both asset-wise and as a portfolio.
    Annualized metrics use the default trading days per year of 252.

    Args:
        weights:
            Weights (-1 to 1) of the assets in the strategy at each interval.
            Each column should be the weights for a specific asset, with the column name being the asset name.
            Column names should match returns.
            Index should be a DatetimeIndex.
            Shape must match returns.
        prices:
            Prices of the assets at each interval used to calculate returns ans costs.
            Each column should be the mark prices for a specific asset, with the column name being the asset name.
            Column names should match weights.
            Index should be a DatetimeIndex.
            Shape must match weights.
        freq_day: Number of strategy intervals in a trading day. Defaults to 1 for daily data.
        trading_days_year: Number of trading days in a year. Defaults to 252.
        shift_periods: Positive integer for number of intervals to shift returns relative to weights. Defaults to 1.
        commission_func: Function to calculate commission cost. Defaults to zero_commission.
        ann_borrow_rate: Annualized borrowing rate applied when asset weight > 1. Defaults to 0.
        spread_pct: Spread cost percentage. Defaults to 0.
        ann_risk_free_rate: Annualized risk-free rate used to calculate Sharpe ratio. Defaults to 0.02.

    Returns:
        A tuple containing five DataFrames that report backtest performance:
            1. Asset-wise performance.
            2. Asset-wise equity curve.
            3. Asset-wise rolling annual Sharpe ratio.
            4. Portfolio performance.
            5. Portfolio returns.
            5. Portoflio equity curve.
    """

    assert weights.shape == prices.shape, "Weights and prices must have the same shape"
    assert (
        weights.columns.tolist() == prices.columns.tolist()
    ), "Weights and prices must have the same column (asset) names"

    # Calc the number of data intervals in a trading year for annualised metrics
    freq_year = freq_day * trading_days_year

    # Backtest each asset so that we can assess the relative performance of the strategy
    # Asset returns approximate a baseline buy and hold scenario
    # Truncate the asset wise returns to account for shifting to ensure the
    # asset and strategy performance metrics are comparable.
    asset_rets = prices.pct_change()
    asset_rets = asset_rets.iloc[:-shift_periods] if shift_periods > 0 else asset_rets
    asset_cum = (1 + asset_rets).cumprod() - 1
    asset_perf = pd.concat(
        [
            asset_rets.apply(
                _ann_sharpe, periods=freq_year, risk_free_rate=ann_risk_free_rate
            ),
            asset_rets.apply(_ann_vol, periods=freq_year),
            asset_rets.apply(_cagr, periods=freq_year),
            asset_rets.apply(_max_drawdown),
        ],
        keys=["annual_sharpe", "annual_volatility", "cagr", "max_drawdown"],
        axis=1,
    )

    # Backtest a cost-aware strategy as defined by the given weights:
    # 1. Calc costs
    # 2. Evaluate asset-wise performance
    # 3. Evalute portfolio performance

    # Calc each cost component in percentage terms so we can
    # deduct them from the strategy returns
    cmn_costs = commission_func(weights, prices) / prices
    borrow_costs = _borrow(weights, prices, ann_borrow_rate, freq_day) / prices
    spread_costs = _spread(weights, prices, spread_pct) / prices
    costs = cmn_costs + borrow_costs + spread_costs

    # Evaluate the cost-aware strategy returns and key performance metrics
    # Use the shift arg to prevent look-ahead bias
    # Truncate the returns to remove the empty intervals resulting from the shift
    strat_rets = weights * (prices.pct_change() - costs).shift(-shift_periods)
    strat_rets = strat_rets.iloc[:-shift_periods] if shift_periods > 0 else strat_rets
    strat_cum = (1 + strat_rets).cumprod() - 1

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
            strat_rets.apply(
                _ann_sharpe, periods=freq_year, risk_free_rate=ann_risk_free_rate
            ),
            strat_rets.apply(_ann_vol, periods=freq_year),
            strat_rets.apply(_cagr, periods=freq_year),
            strat_rets.apply(_max_drawdown),
            strat_ann_turnover,
            _trade_count(weights) / strat_total_days,
        ],
        keys=[
            "annual_sharpe",
            "annual_volatility",
            "cagr",
            "max_drawdown,",
            "annual_turnover",
            "trades_per_day",
        ],
        axis=1,
    )

    # Evaluate the strategy portfolio performance
    port_rets = strat_rets.sum(axis=1)
    port_cum = strat_cum.sum(axis=1)

    # Aproximate the portfolio turnover as the weighted average sum of the asset-wise turnover
    port_ann_turnover = (strat_ann_turnover * weights.mean().abs()).sum()

    port_perf = pd.DataFrame(
        {
            "annual_sharpe": _ann_sharpe(
                port_rets, periods=freq_year, risk_free_rate=ann_risk_free_rate
            ),
            "annual_volatility": _ann_vol(port_rets, periods=freq_year),
            "cagr": _cagr(port_rets, periods=freq_year),
            "max_drawdown": _max_drawdown(port_rets),
            "annual_turnover": port_ann_turnover,
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
            _ann_roll_sharpe(
                asset_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                strat_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=ann_risk_free_rate,
            ),
            _ann_roll_sharpe(
                port_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=ann_risk_free_rate,
            ),
        ],
        keys=["asset", "strategy", "portfolio"],
        axis=1,
    )

    return (
        perf,
        perf_cum,
        perf_roll_sr,
        port_perf,
        port_rets,
        port_cum,
    )


def _ann_sharpe(
    rets: pd.DataFrame | pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods: int = DEFAULT_TRADING_DAYS_YEAR,
) -> float:
    ann_rfr = (1 + risk_free_rate) ** (1 / periods) - 1
    mu = rets.mean()
    sigma = rets.std()
    sr = (mu - ann_rfr) / sigma
    return sr * np.sqrt(periods)


def _ann_roll_sharpe(
    rets: pd.DataFrame | pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    window: int = DEFAULT_TRADING_DAYS_YEAR,
    periods: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.DataFrame | pd.Series:
    ann_rfr = (1 + risk_free_rate) ** (1 / periods) - 1
    mu = rets.rolling(window).mean()
    sigma = rets.rolling(window).std()
    sr = (mu - ann_rfr) / sigma
    return sr * np.sqrt(periods)


def _cagr(
    rets: pd.DataFrame | pd.Series, periods: int = DEFAULT_TRADING_DAYS_YEAR
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


def _ann_vol(
    rets: pd.DataFrame | pd.Series, periods: int = DEFAULT_TRADING_DAYS_YEAR
) -> pd.DataFrame | pd.Series:
    return rets.std() * np.sqrt(periods)


def _max_drawdown(rets: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    cumprod = (1 + rets).cumprod()
    cummax = cumprod.cummax()
    max_drawdown = ((cummax - cumprod) / cummax).max()
    return max_drawdown


def _turnover(
    weights: pd.DataFrame | pd.Series,
    rets: pd.DataFrame | pd.Series,
) -> pd.Series | float:
    """Calculate the turnover for each position in the strategy."""
    # Assume capital of 1000
    capital = 1000
    # Calculate the delta of the weight between each interval
    # Buy will be +ve, sell will be -ve
    diff = weights.fillna(0).diff()
    # Capital is fixed (uncompounded) for each interval so we can calculate the trade volume
    # Sum the volume of the buy and sell trades
    buy_volume = (diff.where(diff > 0, 0).abs() * capital).sum()
    sell_volume = (diff.where(diff < 0, 0).abs() * capital).sum()
    # Trade volume is the minimum of the buy and sell volumes
    # Wrap in Series in case of scalar volume sum (when weights is a Series)
    trade_volume = pd.concat(
        [pd.Series(buy_volume), pd.Series(sell_volume)], axis=1
    ).min(axis=1)
    # Calculate the uncompounded returns to get the average of the portfolio
    # Finally take the ratio of trading volume to mean portfolio value
    equity = capital + (capital * rets.cumsum())
    turnover = trade_volume / equity.mean()
    return turnover


def _trade_count(weights: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    diff = weights.fillna(0).diff().abs() != 0
    tx = diff.astype(int)
    return tx.sum()


def _spread(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    spread_pct: float = 0,
) -> pd.DataFrame | pd.Series:
    size = weights.fillna(0).diff().abs()
    value = size * prices
    costs = value * (spread_pct * 0.5)
    return costs


def _borrow(
    weights: pd.DataFrame | pd.Series,
    prices: pd.DataFrame | pd.Series,
    ann_borrow_rate: float = 0,
    periods_per_year: int = DEFAULT_TRADING_DAYS_YEAR,
) -> pd.DataFrame | pd.Series:
    """Calculate the borrowing costs for each position in the strategy."""
    # Annual rate to per period rate
    rate = (1 + ann_borrow_rate) ** (1 / periods_per_year) - 1
    # Position value from absolute weights and prices
    size = weights.abs().fillna(0)
    value = size * prices
    # Leverage is defined as an absolute weight > 1
    # Zero for all other positions
    lev = (size - 1).clip(lower=0)
    # Costs are the product of the position value, rate and leverage
    costs = value * rate * lev
    return costs
