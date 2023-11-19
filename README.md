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

Alphavec is a fast, minimalist, and cost-aware backtester inspired by https://github.com/Robot-Wealth/rsims.

The backtest input is the natural output of a typical quant research process, simply 2 datframes, one of strategy weights, and one of asset prices.

Suport is also provided for optimising the weights using a trade buffer heuristic.

A streamlined performance report provides insight into the key metrics of sharpe et al.

The implementation is fully vectorized, so its fast... just don't use that as an excuse to overfit!