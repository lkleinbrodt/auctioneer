import backtrader as bt
import yfinance as yf
import pandas as pd
import strategies





class BackTest(bt.Strategy):
    
    def __init__(self, strategy):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.strategy = strategy
        self.order = None
        # self.sizer = bt.sizers.PercentSizer()

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None


    def next(self):
        if self.order:
            return

        close_series = pd.Series({
                self.datas[0].datetime.datetime(-i):
                self.dataclose[-i]
                for i in range(len(self.dataclose))
            },
            dtype = float
        )
        price = close_series.iloc[0]
        action = self.strategy.act(close_series)
        if action == 'buy':
            self.order = self.buy()
        elif action == 'sell':
            self.order = self.sell()
            
        # self.log('Close, %.2f' % self.dataclose[0])


if __name__ == '__main__':
    #TODO: move data loading into the class or maybe it's own class
    
    symbol = 'AAPL'
    start_date = '2023-02-05'
    end_date = '2023-08-11'
    interval = '1h'
    strategy = strategies.SlopeStrategy(window_size=40, threshold=.05)
    # strategy = strategies.CrossoverStrategy(5, 10)
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BackTest, (strategy))
    cerebro.addsizer(strategies.AllInSizer)
    cerebro.broker.setcommission(commission=0)
    data = bt.feeds.PandasData(dataname=yf.download(symbol,start_date ,end_date, auto_adjust=True, interval=interval))
    cerebro.adddata(data, name = symbol)
    # data = bt.feeds.PandasData(dataname=yf.download('TSLA',start_date ,end_date, auto_adjust=True, interval=interval))
    # cerebro.adddata(data, name = 'TSLA')
    cerebro.broker.setcash(10000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.plot()
    
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategies.LongStrategy)
    cerebro.addsizer(strategies.AllInSizer)
    cerebro.adddata(data)
    cerebro.broker.setcash(10000.0)
    cerebro.run()
    print('Long Strategy Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    