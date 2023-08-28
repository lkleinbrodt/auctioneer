import pandas as pd
from scipy.stats import linregress
import backtrader as bt

class CrossoverStrategy:
    def __init__(self, short_window: int, long_window: int):
        self.short_window = short_window
        self.long_window = long_window
        self.status = 'below'
        
    def act(self, data_feed):
        data_feed = pd.Series(data_feed).sort_index()
        long_ma = data_feed.tail(self.long_window).mean()
        short_ma = data_feed.tail(self.short_window).mean()
        if (short_ma > long_ma) & (self.status == 'below'):
            self.status = 'above'
            return 'buy'
        elif (short_ma < long_ma) & (self.status == 'above'):
            self.status = 'below'
            return 'sell'
        

class SlopeStrategy:
    def __init__(self, window_size: int, threshold: float):
        self.window_size = window_size
        self.threshold = threshold
    
    #this one is only going to work on a single series
    def act(self, data_feed):
        data_feed = pd.Series(data_feed)
        data_feed = data_feed.sort_index().tail(self.window_size)
        # print(data_feed)
        if len(data_feed) < self.window_size:
            return None
        slope = linregress(data_feed.reset_index().index, data_feed.values)[0]
        
        if slope >= self.threshold:
            return 'buy'
        elif slope <= - self.threshold:
            return 'sell'
        else:
            return None
        
class LongStrategy(bt.Strategy):
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.is_first = True
    
    def next(self):
        if self.is_first:
            self.order = self.buy()
            self.is_first = False
            # print(self.order.size)
            print('Starting Price', self.dataclose[0])
        self.price = self.dataclose[0]
        
        
class AllInSizer(bt.Sizer):

    def _getsizing(self, comminfo, cash, data, isbuy):
        if isbuy:
            return math.floor(cash / data[0])
        else:
            position = self.broker.getposition(data)
            return position.size