import pandas as pd
from scipy.stats import linregress

class SlopeStrategy:
    def __init__(self, window_size: int = 10, threshold: float = .20):
        self.window_size = window_size
        self.threshold = threshold
    
    #this one is only going to work on a single series
    def act(self, data_feed):
        data_feed = pd.Series(data_feed).tail(self.window_size)
        slope = linregress(data_feed.reset_index().index, data_feed.values)[0]
        
        if slope >= self.threshold:
            return 'buy'
        elif slope <= - self.threshold:
            return 'sell'
        else:
            return None
        
        