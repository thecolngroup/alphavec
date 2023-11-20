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

Alphavec is a fast, minimalist, and cost-aware vectorized backtest engine inspired by https://github.com/Robot-Wealth/rsims.

The backtest input is the natural output of a typical quant research process. You simply provide a dataframe of strategy weights and a dataframe of asset prices, along with some cost parameters and it returns a streamlined performance report with insight into the key metrics of sharpe, CAGR, and volatility et al.

## Rationale

Alphavec is an antidote to the various bloated and complex backtest frameworks.

To validate ideas all you really need is...

``` weights.shift(1) * log_returns ```

The goal was to add just enough extra complexity to this basic building block in order to support sound development of cost-aware systematic trading strategies.

## Install

``` pip install git+https://github.com/thecolngroup/alphavec ```


