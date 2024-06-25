"""Backtest module for evaluating trading strategies."""

from calendar import c
from ensurepip import bootstrap
import logging
from typing import Callable, Tuple, Union, List

import numpy as np
from numpy.random import RandomState, SeedSequence, MT19937
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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


def pnl(log_rets: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """Calculate the cumulative profit and loss from log returns.

    Use this function in conjunction with log returns from the backtest.
    E.G. to calcuate portfolio value based on an initial investment of 1000:
    equity_in_currency_units = 1000 * pnl(port_rets)

    Args:
        log_rets: Log returns of the assets in the portfolio.

    Returns:
        Cumulative profit and loss of the portfolio.
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
    bootstrap_n: int = 1000,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
]:
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
        bootstrap_n: Number of bootstrap iterations to validate portfolio performance. Defaults to 1000.

    Returns:
        A tuple containing five data sets:
            1. Asset-wise performance table
            2. Asset-wise profit and loss curves
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
    asset_pnl = pnl(asset_rets)

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
    strat_pnl = pnl(strat_rets)

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
    port_pnl = pnl(port_rets)

    # Approximate the portfolio turnover as the weighted average sum of the asset-wise turnover
    # port_ann_turnover = _turnover(weights.abs().sum(axis=1), port_rets) * (
    #    trading_days_year / strat_total_days.max()
    # )

    port_ann_turnover = (strat_ann_turnover * weights.abs().mean()).sum()

    # Combine the asset and strategy performance metrics into a single dataframe for comparison
    perf = pd.concat(
        [asset_perf, strat_perf],
        keys=["asset", "strategy"],
        axis=1,
    )

    perf_pnl = pd.concat(
        [port_pnl, asset_pnl, strat_pnl],
        keys=["portfolio", "asset", "strategy"],
        axis=1,
    ).rename(columns={0: "PNL"})

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
    ).rename(columns={0: "SR"})

    def calc_port_metrics(port_rets: pd.Series):
        return pd.DataFrame(
            {
                "annual_sharpe": _ann_sharpe(
                    port_rets,
                    freq_year=freq_year,
                    ann_risk_free_rate=ann_risk_free_rate,
                ),
                "annual_volatility": _ann_vol(port_rets, freq_year=freq_year),
                "cagr": _cagr(port_rets, freq_year=freq_year),
                "max_drawdown": _max_drawdown(port_rets),
                "annual_turnover": port_ann_turnover,
            },
            index=["observed"],
        )

    port_perf = calc_port_metrics(port_rets)
    if bootstrap_n > 0:
        sampled_rets = _bootstrap_sampling(
            port_rets, n=bootstrap_n, stationary_method=True
        )
        sampled_perf = pd.concat([calc_port_metrics(rets) for rets in sampled_rets])

        def describe(x):
            return pd.Series(
                {
                    "mean": x.mean(),
                    "std": x.std(),
                    "median": x.median(),
                    "ucl.95": np.percentile(x, 97.5),
                    "lcl.95": np.percentile(x, 2.5),
                }
            )

        port_perf = pd.concat([port_perf, sampled_perf.apply(describe)]).round(4)

    return (perf, perf_pnl, perf_roll_sr, port_perf, port_rets)


def _bootstrap_sampling(
    x: pd.Series,
    n: int = 1000,
    seed: int = 1,
    stationary_method: bool = False,
) -> List[pd.Series]:

    samples = []

    rs = RandomState(MT19937(SeedSequence(seed)))

    if stationary_method:
        block_size = optimal_block_length(x.dropna())["stationary"].squeeze()
        bs = StationaryBootstrap(block_size, x.dropna().values, seed=rs)  # type: ignore
        for sample in bs.bootstrap(n):
            sample = pd.Series(sample[0][0], index=x.index)  # type: ignore
            sample[x.isna()] = np.nan
            samples.append(sample)
    else:
        for _ in range(n):
            sample = rs.choice(x.dropna(), size=x.shape, replace=True)  # type: ignore
            sample = pd.Series(sample, index=x.index)
            sample[x.isna()] = np.nan
            samples.append(sample)

    return samples


def plot_distribution(observed: float, sampled: pd.Series):
    """Plot the distribution of sampled values against the observed value.

    Args:
        observed: Observed value e.g. your original backtested annualized Sharpe.
        sampled: Sampled values e.g. bootstrapped Sharpes.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(sampled, bins=100, alpha=0.75, color="grey")

    # Plot standard deviation lines
    mu = float(np.mean(sampled))
    std = float(np.std(sampled))
    for i in range(1, 4):
        plt.axvline(
            mu + (i * std),
            color="grey",
            linestyle="dashed",
            linewidth=1,
        )
        plt.axvline(
            mu - (i * std),
            color="grey",
            linestyle="dashed",
            linewidth=1,
        )

    # Plot statistical significance percentile
    simulated_sorted = np.sort(sampled)
    upper_ci = np.percentile(simulated_sorted, 97.5)
    plt.axvline(
        upper_ci,
        color="green",
        linestyle="solid",
        linewidth=1,
        label=f"Upper 95% CI ({upper_ci:.2f})",
    )
    lower_ci = np.percentile(simulated_sorted, 2.5)
    plt.axvline(
        lower_ci,
        color="green",
        linestyle="solid",
        linewidth=1,
        label=f"Lower 95% CI ({lower_ci:.2f})",
    )

    # Plot baseline
    pctile = stats.percentileofscore(sampled, observed)
    plt.axvline(
        observed,
        color="red",
        linestyle="solid",
        linewidth=1,
        label=f"Baseline {observed:.2f} ({pctile:.2f}%)",
    )

    plt.title("Distribution of Sampled vs Observed")
    plt.legend()
    plt.show()


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
    curve = pnl(log_rets)
    hwm = curve.cummax()
    dd = (curve - hwm) / hwm
    return dd.min()  # type: ignore


def _turnover(
    weights: Union[pd.DataFrame, pd.Series],
    log_rets: Union[pd.DataFrame, pd.Series],
) -> float:
    """Calculate the non-annualized turnover for each position in the strategy using the post-cost returns."""

    if isinstance(weights, pd.Series):
        weights = weights.to_frame()
    traded = weights.abs().sum(axis=1).fillna(0).diff().abs().sum()

    mu = pnl(log_rets).mean()

    turnover = traded / mu
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
