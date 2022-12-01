from auctioneer import *
import pandas as pd
import logging
logging.basicConfig(level = logging.DEBUG)
logger = logging.getLogger(__name__)


STARTING_BALANCE = 100_000
START_DATE = "11-01-2022"
END_DATE = "11-30-2022"
DATA_PATH = '../minute_crypto_data.csv' #None to pull cryto data

HISTORY_STEPS = 240#480
TARGET_STEPS = 30#120
MAX_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = .001
STARTING_BALANCE = 100_000
WINDOW_SIZE = 60*24*14
STRIDE = 60

REFRESH_RATE = 50

#begin

START_DATE = pd.to_datetime(START_DATE)
END_DATE = pd.to_datetime(END_DATE)

def main():
    logger.info('---START---')

    soi = get_crypto_symbols()
    soi = soi[:3]

    if DATA_PATH is not None:
        logger.info(f"Pulling data from: {DATA_PATH}")
        price_data = pd.read_csv(DATA_PATH)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data = price_data[price_data['symbol'].isin(soi)]
        price_data = price_data.loc[START_DATE:END_DATE]
    else:
        logger.info("Pulling data from Alpaca API")
        price_data = pull_crypto_prices(soi, START_DATE, END_DATE, timeframe='day')

    data = pivot_price_data(price_data)

    #TODO: this could be better
    global REFRESH_STEPS
    REFRESH_STEPS = [data.index[i] for i in range(0, data.shape[0], REFRESH_RATE)]

    results = windowed_backtest(data, window_size = WINDOW_SIZE, stride = STRIDE)

    results.to_csv('backtest_nn_results.csv', index = False)
    logger.info('---END---')

def one_day(date, portfolio, data, model, scalers):
    
    data = data.loc[:date]

    if date in REFRESH_STEPS:
        logger.info('Training fresh encoder model')
        model, scalers = train_encoder_model(data, HISTORY_STEPS, TARGET_STEPS, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE)
    
    predictions = predict_forward(data, model, HISTORY_STEPS)
    terminal_prices = {data.columns[i]: predictions[-1,i] for i in range(data.shape[1])}

    soi = get_crypto_symbols()
    orders = {}
    for symbol in soi:
        current_price = data[symbol].values[-1]
        predicted_price = terminal_prices[symbol]

        if predicted_price > current_price:
            amount = 1
            orders[symbol] = {'action': 'buy', 'amount': amount}
        elif predicted_price < current_price:
            amount = portfolio.holdings.get(symbol, 0)
            amount = max([1, amount // 10])
            orders[symbol] = {'action': 'sell', 'amount': amount}
    
    return orders

def backtest(data):
    portfolio = Portfolio(STARTING_BALANCE, data)

    start_offset = max([HISTORY_STEPS, REFRESH_RATE])

    model, scalers = train_encoder_model(
        data.iloc[:start_offset], 
        HISTORY_STEPS, 
        TARGET_STEPS, 
        MAX_EPOCHS, 
        BATCH_SIZE, 
        LEARNING_RATE
    )

    for i, date in enumerate(data.index):
        if i < start_offset:
            continue
        orders = one_day(date, portfolio, data, model, scalers)
        if orders is not None:
            portfolio.execute(date, orders)
    return portfolio

def windowed_backtest(data, window_size, stride = 1):
    results = []
    for i in range(0, data.shape[0]-window_size, stride):
        tmp = data.iloc[i:i+window_size]
        result = backtest(tmp)
        log = result.transaction_log_summary()

        long_result = long_return(tmp, STARTING_BALANCE)
        results.append({
            'StartDate': data.index[i], 
            'EndValue': result.value()['Total'],
            'LongValue': long_result,
            'N_Buys': log['buys'][0],
            'N_Sells': log['sells'][0],
        })
    
    return pd.DataFrame(results)





