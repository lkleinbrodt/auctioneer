

SECURITY = 'ETH/USD' #BTC
START_DATE = "09-01-2010"
END_DATE = "11-01-2022"

STARTING_BALANCE = 100_000
N_DAYS = 10
THRESHOLD = 1



from auctioneer import *

import pandas as pd
from scipy.stats import linregress
import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)

# with open('data/paper_api_keys.txt') as api_file:
    # api_keys = api_file.read().replace('\n', '').split(',')
    # alpaca_api = {a.split('=')[0]: a.split('=')[1] for a in api_keys}

# No keys required for crypto data


START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)



from dateutil.relativedelta import relativedelta
def n_day_momentum(date, portfolio, data, n_days, threshold):
    STRATEGY = 'max'
    # n_days_ago = date + relativedelta(days = n_days)

    rel_data = data.loc[:date].tail(n_days)

    closes = pd.Series(rel_data['close'].values)
    slope = linregress(closes.index, closes.values)[0]

    if slope > threshold:
        if STRATEGY == 'max':
            amount = 'max'
        elif STRATEGY == 'singles':
            amount = 1
        elif STRATEGY == 'percentage':
            amount = (portfolio.balance * .1) // closes.iloc[-1]
        else:
            raise ValueError(f"Unrecognized strategy: {STRATEGY}")

        return {SECURITY: {'action': 'buy', 'amount': amount}}
    elif slope < -threshold:
        if STRATEGY == 'max':
            amount = 'max'
        elif STRATEGY == 'singles':
            amount = 1
        elif STRATEGY == 'percentage':
            amount = portfolio.holdings.get(SECURITY, 0) // 10
        else:
            raise ValueError(f"Unrecognized strategy: {STRATEGY}")

        return {SECURITY: {'action': 'sell', 'amount': amount}}
    else:
        return None


def long_return(data):
    long_portfolio = Portfolio(STARTING_BALANCE, data, {})
    long_portfolio.execute(data.index[0], {SECURITY: {'action': 'buy', 'amount': 'max'}})
    return long_portfolio.value()['Total']

def backtest_slope(data, n_days, threshold):

    strategy_portfolio = Portfolio(STARTING_BALANCE, data, {})
    
    for i, date in enumerate(data.index):
        if i < n_days:
            continue
        order = n_day_momentum(date, strategy_portfolio, data, n_days, threshold)
        if order is not None:
            strategy_portfolio.execute(date, order)

    return strategy_portfolio
        

def windowed_backtest(data, window_size, n_days, threshold, stride = 1):
    results = []
    for i in range(0, data.shape[0]-window_size, stride):
        tmp = data.iloc[i:i+window_size]
        result = backtest_slope(tmp, n_days, threshold)
        log = result.transaction_log_summary()

        long_result = long_return(tmp)
        results.append({
            'StartDate': data.index[i], 
            'EndValue': result.value()['Total'],
            'LongValue': long_result,
            'N_Buys': log['buys'][0],
            'N_Sells': log['sells'][0],
        })
    
    return pd.DataFrame(results)


def main():
    logger.info('---START---')
    data = pull_data(SECURITY, START_DATE, END_DATE)

    results_list = []
    i = 0
    for n in [5, 10, 20, 50]:
        for t in [1, 2, 5, 10]:
            logger.info(i)
            results = windowed_backtest(data, 365, n, t, 5)
            results = results.assign(N=n, Threshold=t)
            results_list.append(results)
            i+=1
    backtest_results = pd.concat(results_list)
    backtest_results.to_csv('backtested_results.csv', index = False)
    logger.info('---END---')

if __name__ == '__main__':
    main()