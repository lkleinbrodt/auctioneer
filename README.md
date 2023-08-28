# auctioneer

I used to have my own backtesting implementation, but now this works with backtrader.

Implement a strategy in `strategies.py`
    - Strategy class must have a method `act` which takes in a data_feed, which is a pandas Series like object, and outputs one of "buy", "sell" or None
    - sizing of orders is handled by a sizer object

Backtest the strategy using `backtest.py`
    - in the main function, input start date, end date, time interval, symbol to trade, your strategy, and your sizing strategy.