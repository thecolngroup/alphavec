# alphavec

```
          $$\           $$\                                               
          $$ |          $$ |                                              
 $$$$$$\  $$ | $$$$$$\  $$$$$$$\   $$$$$$\ $$\    $$\  $$$$$$\   $$$$$$$\ 
 \____$$\ $$ |$$  __$$\ $$  __$$\  \____$$\\$$\  $$  |$$  __$$\ $$  _____|
 $$$$$$$ |$$ |$$ /  $$ |$$ |  $$ | $$$$$$$ |\$$\$$  / $$$$$$$$ |$$ /      
$$  __$$ |$$ |$$ |  $$ |$$ |  $$ |$$  __$$ | \$$$  /  $$   ____|$$ |      
\$$$$$$$ |$$ |$$$$$$$  |$$ |  $$ |\$$$$$$$ |  \$  /   \$$$$$$$\ \$$$$$$$\ 
 \_______|\__|$$  ____/ \__|  \__| \_______|   \_/     \_______| \_______|
              $$ |                                                        
              $$ |                                                        
              \__|                                                                                                         
```

Alphavec is a lightning fast, minimalist, cost-aware vectorized backtest engine inspired by https://github.com/Robot-Wealth/rsims.

The backtest input is the natural output of a typical quant research process - a time series of portfolio weights. You simply provide a dataframe of strategy weights and a dataframe of asset prices, along with some optional cost parameters and the backtest returns a streamlined performance report with insight into the key metrics of sharpe, volatility, CAGR, drawdown et al.

In addition, onvenience functions are included for robustness checking such as sampling of random contiguous periods and adding noise to a price series.

## Rationale

Alphavec is an antidote to the various bloated and complex backtest frameworks.

To validate ideas all you really need is...

``` weights * returns.shift(-1) ```

The goal was to add just enough extra complexity to this basic formula in order to support sound development of cost-aware systematic trading strategies.

## Install

``` pip install git+https://github.com/breaded-xyz/alphavec@main```

## Usage

See the notebook ```example.ipynb``` for a walkthrough of designing and testing a strategy using Alphavec.

```python

from functools import partial
import alphavec as av

prices = load_asset_prices()
weights = generate_weights()

result = av.backtest(
    weights,
    prices,
    freq_day=1,
    trading_days_year=365,
    shift_periods=1,
    commission_func=partial(av.pct_commission, fee=0.001),
    spread_pct=0.001,
    ann_borrow_rate=0.05,
    ann_risk_free_rate=0.002
)
```
