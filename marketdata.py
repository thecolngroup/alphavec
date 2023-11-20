from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import zscore
from pandas_datareader import data as pdr
import yfinance as yf

yf.pdr_override()


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
    )


def ohlcv_from_yahoo(tickers: list, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Fetches OHLCV (Open, High, Low, Close, Volume) data for a list of tickers from Yahoo Finance.

    Parameters:
    tickers (list): A list of ticker symbols for which to fetch data.
    start (datetime): The start date for the data fetch.
    end (datetime): The end date for the data fetch.

    Returns:
    DataFrame: A pandas DataFrame containing the OHLCV data for the specified tickers and date range.

    The function performs the following steps:
    1. Fetches data from Yahoo Finance using the pandas_datareader library. The 'auto_adjust' parameter is set to True to adjust the OHLC data for dividends and splits.
    2. Stacks the DataFrame and swaps the levels to bring the ticker symbols to the index.
    3. Renames the columns to lowercase abbreviations of their original names.
    """
    data = (
        pdr.get_data_yahoo(tickers, start=start, end=end, auto_adjust=True)
        .stack(level=1)
        .swaplevel()
        .rename(
            columns={"Open": "o", "High": "h", "Low": "l", "Close": "c", "Volume": "v"}
        )
    )
    return data


def clean_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the given price data by removing NaNs and extreme outliers.

    Parameters:
    prices (DataFrame): A pandas DataFrame containing price data.

    Returns:
    DataFrame: A cleaned pandas DataFrame with the same structure as the input.

    The cleaning process involves the following steps:
    1. Calculate the percentage change in prices, which gives the returns.
    2. Compute the z-scores of the returns. This standardizes the returns data to have a mean of 0 and standard deviation of 1.
        The 'nan_policy' parameter set to 'omit' means that NaN values are automatically excluded from the calculation.
    3. Replace any price with a z-score greater than 4 or less than -4 with NaN. This step removes extreme outliers from the data.
    4. Forward fill NaN values. This means that any NaN value is replaced with the value from the previous row (or the next valid observation if the NaN is at the start of the series).
    """
    rets = prices.pct_change()
    z_scores = zscore(rets, nan_policy="omit")
    prices[np.abs(z_scores) > 4] = np.nan
    prices = prices.fillna(method="ffill")
    return prices
