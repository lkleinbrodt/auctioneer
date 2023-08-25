import backtrader as bt
import yfinance as yf
from slope import SlopeStrategy
import pandas as pd
import math

class Sizer(bt.Sizer):

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return math.floor(cash / data[0])
        else:
            position = self.broker.getposition(data)
            return position.size


class LongStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.is_first = True
        self.sizer = Sizer()
    
    def next(self):
        if self.is_first:
            self.order = self.buy()
            self.is_first = False
            print(self.order.size)

class TestStrategy(bt.Strategy):
    
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        self.strategy = SlopeStrategy()
        self.order = None
        self.sizer = Sizer()

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
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
            i: self.dataclose[-i]
            for i in range(self.strategy.window_size)
        })
        close_series = close_series.sort_index(ascending = False)
        # print(self.dataclose)
        price = close_series.iloc[0]
        action = self.strategy.act(close_series)
        if action == 'buy':
            self.order = self.buy()
        elif action == 'sell':
            self.order = self.sell()
            
        # self.log('Close, %.2f' % self.dataclose[0])
    

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    cerebro.addstrategy(TestStrategy)
    cerebro.broker.setcommission(commission=.001)
    data = bt.feeds.PandasData(dataname=yf.download('SPY', '2023-07-06', '2023-08-05', auto_adjust=True, interval='5m'))
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LongStrategy)
    cerebro.adddata(data)
    cerebro.broker.setcash(1000.0)
    cerebro.run()
    print('Long Strategy Final Portfolio Value: %.2f' % cerebro.broker.getvalue())