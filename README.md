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

Alphavec is a fast, minimalist, and cost-aware vectorized backtester inspired by https://github.com/Robot-Wealth/rsims.

The backtest input is the natural output of a typical quant research process: simply 2 dataframes, one of strategy weights, and one of asset prices.

A streamlined performance report provides insight into the key metrics of sharpe, CAGR, and volatility et al.

Utilities are also provided to assist with preparing weights such as applying a trade buffer heuristic.

## Rationale

Alphavec is an antidote to the various bloated and complex backtest frameworks.

For quick validation of ideas all you realy need is Pandas and...

``` weights.shift(1) * log_returns ```

The goal was to add just enough extra complexity to this basic building block of vectorized backtesting in order to support sound development of systematic trading strategies.

Alphavec adds support for various cost components, performance reporting and baseline comparisons.

## Install

``` pip install git+https://github.com/thecolngroup/alphavec ```


