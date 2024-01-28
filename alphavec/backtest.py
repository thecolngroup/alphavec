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
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    freq_day: int = 1,
    shift_periods: int = 1,
    commission_func: Callable[[pd.DataFrame, pd.DataFrame], float] = zero_commission,
    ann_borrow_pct: float = 0,
    spread_pct: float = 0,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
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
        freq_day: Number of strategy intervals in a trading day. Defaults to 1.
        shift_periods: Positive integer for number of intervals to shift returns relative to weights. Defaults to 1 which is suitable for close-to-close returns.
        commission_func: Function to calculate commission cost. Defaults to zero_commission.
        ann_borrow_pct: Annual borrowing cost percentage applied when asset weight > 1. Defaults to 0.
        spread_pct: Spread cost percentage. Defaults to 0.
        risk_free_rate: Risk-free rate used to calculate Sharpe ratio. Defaults to 0.02.

    Returns:
        A tuple containing five DataFrames that report backtest performance:
            1. Asset-wise performance.
            2. Asset-wise equity curve.
            3. Asset-wise rolling annual Sharpe ratio.
            4. Portfolio performance.
            5. Portoflio equity curve.
    """

    assert weights.shape == prices.shape, "Weights and prices must have the same shape"
    assert (
        weights.columns.tolist() == prices.columns.tolist()
    ), "Weights and prices must have the same column (asset) names"

    # Calc the number of data intervals in a trading year for annualised metrics
    freq_year = freq_day * DEFAULT_TRADING_DAYS_YEAR

    # Backtest each asset so that we can assess the relative performance of the strategy
    # Asset returns approximate a baseline buy and hold scenario
    # Truncate the asset wise returns to account for shifting to ensure the
    # asset and strategy performance metrics are comparable.
    asset_rets = prices.pct_change()[:-shift_periods]
    asset_cum = (1 + asset_rets).cumprod() - 1
    asset_perf = pd.concat(
        [
            asset_rets.apply(_sharpe, periods=freq_year, risk_free_rate=risk_free_rate),
            asset_rets.apply(_vol, periods=freq_year),
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
    borrow_costs = _borrow(weights, prices, (ann_borrow_pct / freq_year)) / prices
    spread_costs = _spread(weights, prices, spread_pct) / prices
    costs = cmn_costs + borrow_costs + spread_costs

    # Calc the number of valid trading periods for each asset
    # to calculate correct number of trades
    strat_valid_periods = weights.apply(
        lambda col: col.loc[col.first_valid_index() :].count()
    )
    strat_days = strat_valid_periods / freq_day

    # Evaluate the cost-aware strategy returns and key performance metrics
    # Use the shift arg to prevent look-ahead bias
    # Truncate the returns to remove the empty intervals resulting from the shift
    strat_rets = (
        weights * (prices.pct_change() - costs).shift(-shift_periods)[:-shift_periods]
    )
    strat_cum = (1 + strat_rets).cumprod() - 1
    strat_profit_cost_ratio = strat_cum.iloc[-1] / costs.sum()
    strat_perf = pd.concat(
        [
            strat_rets.apply(_sharpe, periods=freq_year, risk_free_rate=risk_free_rate),
            strat_rets.apply(_vol, periods=freq_year),
            strat_rets.apply(_cagr, periods=freq_year),
            strat_rets.apply(_max_drawdown),
            _trade_count(weights) / strat_days,
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
            "annual_sharpe": _sharpe(
                port_rets, periods=freq_year, risk_free_rate=risk_free_rate
            ),
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
            _roll_sharpe(
                asset_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=risk_free_rate,
            ),
            _roll_sharpe(
                strat_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=risk_free_rate,
            ),
            _roll_sharpe(
                port_rets,
                window=freq_year,
                periods=freq_year,
                risk_free_rate=risk_free_rate,
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
        port_cum,
    )


def _sharpe(
    rets: pd.DataFrame | pd.Series,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    periods: int = DEFAULT_TRADING_DAYS_YEAR,
) -> float:
    ann_rfr = (1 + risk_free_rate) ** (1 / periods) - 1
    mu = rets.mean()
    sigma = rets.std()
    sr = (mu - ann_rfr) / sigma
    return sr * np.sqrt(periods)


def _roll_sharpe(
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


def _vol(
    rets: pd.DataFrame | pd.Series, periods: int = DEFAULT_TRADING_DAYS_YEAR
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
) -> pd.DataFrame | pd.Series:
    size = weights.abs().fillna(0)
    value = size * prices
    lev = (size - 1).clip(lower=0)
    costs = value * borrow_pct * lev
    return costs.fillna(0)
