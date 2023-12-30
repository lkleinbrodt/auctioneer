from config import *
from lstm import *

### 1) Load Data
    # price and returns data
    # identify the test period
    # load the models

# For each time step ( or multiple)
    # 1) predict expected return at t + n
    # complication: when treating this as ezxpected return, remember that it is cumulative return * 100
    # so it may need to be annualized? to be in prpoper format? or perhaps not, since it should only affect the numerator?
    # 2) Calculate historical volatility from 0:t
    # 3) optimize weights given 1+2
    # 4) balance portfolio accordingly, keeping track of value + holdings
    # store activity
    
