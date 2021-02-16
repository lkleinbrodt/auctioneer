import os
import warnings
import logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tensorflow as tf
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as alp
import pandas_market_calendars as mcal
import logging
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.objective_functions import L2_reg
from pypfopt.discrete_allocation import DiscreteAllocation
logging.basicConfig(filename = 'Logs/backtest_log.log', level = 'INFO', filemode='w')

with open('Data/paper_api_keys.txt') as api_file:
    api_keys = api_file.read().replace('\n', '').split(',')
    alpaca_api = {a.split('=')[0]: a.split('=')[1] for a in api_keys}
    
api = alp.REST(key_id=alpaca_api['APCA_API_KEY_ID'], secret_key = alpaca_api['APCA_API_SECRET_KEY'], base_url=alpaca_api['APCA_API_BASE_URL'])

from Functions import *

########## PARAMETERS ##############

START_DATE = '2019-11-01'
END_DATE = '2020-12-01'

MOVEMENT_THRESHOLD = 0
MODEL_REFRESH_DAYS = 31
N_DATA_POINTS = 200_000

STARTING_CASH = 100_000
MAX_CONCURRENT_SECURITIES = 15

HISTORY_STEPS = 10_000#480
TARGET_STEPS = 1000#120
MAX_EPOCHS = 20
BATCH_SIZE = 32
LEARNING_RATE = .001

########## END PARAMETERS #############

logging.info('Starting Script: ')
logging.info(datetime.now())

#Get Alpaca APIs

### Get portfolio
###
#portfolio = pd.DataFrame([{'Symbol': p.symbol, 'Quantity': float(p.qty), 'Value': float(p.market_value)} for p in api.list_positions()])

portfolio = pd.DataFrame(columns = ['Symbol', 'Quantity', 'Value'])
portfolio['Symbol'] = portfolio['Symbol'].astype(str)
portfolio['Quantity'] = portfolio['Quantity'].astype(float)
portfolio['Value'] = portfolio['Value'].astype(float)

model_refresh_days = [CheckHoliday(x) for x in pd.date_range(START_DATE, END_DATE, freq='BMS')]

### Define One Day of trading behavior, should take in a date, and output the actions that will be taken

def TradingDay(current_day, portfolio, buying_power, api, scalers, model = None, cov_matrix = None):

    if (current_day in model_refresh_days) or (model is None): 
        portfolio_stocks = portfolio['Symbol'].tolist()
        if len(portfolio_stocks) > MAX_CONCURRENT_SECURITIES:
            stocks_to_predict = portfolio_stocks
        else:
            soi = IdentifyStocksOfInterest()
            stocks_to_predict = list(set(portfolio_stocks + soi))
        
        data = GetHistoricalData(stocks_to_predict, end_date = current_day, api = api, n_data_points = N_DATA_POINTS)
        logging.debug('Training Data')
        logging.debug(data.tail())
        model, scalers = TrainEncoderModel(data, HISTORY_STEPS, TARGET_STEPS, MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE)
        cov_matrix = CovarianceShrinkage(data).ledoit_wolf()
        
    else:
        soi = list(scalers.keys())
        stocks_to_predict = list(set(portfolio['Symbol'].tolist() + soi))
        data = GetHistoricalData(stocks_to_predict, api, day, n_data_points=HISTORY_STEPS)

    stocks_to_predict = [col for col in stocks_to_predict if col in data.columns]

    inference_data = data[-HISTORY_STEPS:].copy()
    logging.debug('Inference Data:')
    logging.debug(inference_data.tail())
    
    current_prices = {data.columns[i]: data.iloc[-1, i] for i in range(data.shape[1])}
    
    for col in inference_data.columns:
        scaler = scalers[col]
        norm = scaler.transform(inference_data[col].values.reshape(-1,1))
        norm = np.reshape(norm, len(norm))
        inference_data[col] = norm
    
    inference_data = np.array(inference_data).reshape((1, inference_data.shape[0], -1))

    predictions = model.predict(inference_data).squeeze() 

    for i, col in enumerate(data.columns):
        scaler = scalers[col]
        predictions[:,i] = scaler.inverse_transform(predictions[:,i].reshape(-1,1)).reshape(-1)

    terminal_prices = {data.columns[i]: predictions[-1,i] for i in range(data.shape[1])}

    ef = EfficientFrontier(pd.Series(terminal_prices), cov_matrix)
    ef.add_objective(L2_reg, gamma = .01)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()

    ### Reconcile Current Portfolio with Optimal Portfolio
    portfolio_value = buying_power + np.sum(portfolio['Value'])
    da = DiscreteAllocation(cleaned_weights, pd.Series(current_prices), portfolio_value)
    new_allocation, _ = da.lp_portfolio()
    logging.info('Target Allocation')
    logging.info(new_allocation)
    orders = dict([])
    for symbol in new_allocation.keys():

        target_quantity = new_allocation[symbol]

        if symbol in list(portfolio['Symbol']):
            current_quantity = portfolio[portfolio['Symbol'] == symbol]['Quantity'][0]
        else:
            current_quantity = 0

        if target_quantity > current_quantity:
            side = 'buy'
            quantity = (target_quantity - current_quantity)
        elif target_quantity < current_quantity:
            side = 'sell'
            quantity = (current_quantity - target_quantity)
        else:
            continue

        orders[symbol] = {'Side': side, 'Quantity': quantity}
    
    return predictions, orders, current_prices, model, scalers, cov_matrix


def Execution(day_of_order, orders, portfolio, buying_power):

    if not isinstance(day_of_order, pd.Timestamp):
        day_of_order = pd.Timestamp(day_of_order, tz = 'America/New_York')

    ###First, ensure your dates are valid
    calendar = api.get_calendar(start = day_of_order, end = day_of_order + relativedelta(days = 5))

    assert calendar[0].date == day_of_order

    ### Now determine which day these trades will be placed
    day_of_execution = calendar[1].date
    
    prices_to_pull = list(set(list(orders.keys()) + list(portfolio['Symbol'])))
    next_day_prices = dict(GetDayQuotes(prices_to_pull, api, day_of_execution, 'open'))
    orders = {symbol: orders[symbol] for symbol in orders.keys() if symbol in next_day_prices.keys()}

    sell_symbols = [symbol for symbol in orders.keys() if orders[symbol]['Side'] == 'sell']
    buy_symbols = [symbol for symbol in orders.keys() if orders[symbol]['Side'] == 'buy']

    for symbol in sell_symbols:
        quantity = orders[symbol]['Quantity']
        price = next_day_prices[symbol]
        cost = quantity * price

        logging.debug('Selling {} shares of {} at {} per share for a total sale of {}.'.format(quantity, symbol, np.round(price,5), np.round(cost,5)))
        buying_power += cost
        portfolio.loc[(portfolio['Symbol'] == symbol), 'Quantity'] -= quantity
        logging.debug('New buying power: ' + str(buying_power))
    
    for symbol in buy_symbols:
        quantity = orders[symbol]['Quantity']
        price = next_day_prices[symbol]
        cost = quantity * price
        logging.debug('Trying to buy {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
        
        while cost > buying_power:
            logging.debug('Cannot buy, total cost is {} and we only have {}'.format(cost, buying_power))
            quantity -= 1
            cost = quantity * price
        if quantity == 0:
            logging.debug('not enough cash to buy a single share')
            continue
        else:
            buying_power -= cost
            if symbol in list(portfolio['Symbol']):
                portfolio.loc[portfolio['Symbol']==symbol, 'Quantity'] += quantity
            else:
                portfolio = portfolio.append(pd.DataFrame([{'Symbol': symbol, 'Quantity': quantity, 'Value': quantity * price}]))
            logging.debug('Successfully bought {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
            logging.debug('New buying power: ' + str(buying_power))
        portfolio.loc[portfolio['Symbol']==symbol, 'Value'] = price * portfolio.loc[portfolio['Symbol']==symbol, 'Quantity']
        
    portfolio = portfolio[portfolio['Quantity'] != 0].copy()
    for symbol in next_day_prices.keys():
        portfolio.loc[portfolio['Symbol']==symbol, 'Value'] = next_day_prices[symbol] * portfolio.loc[portfolio['Symbol']==symbol, 'Quantity']

    return next_day_prices, portfolio, buying_power

day = pd.to_datetime(START_DATE)
end_day = pd.to_datetime(END_DATE)

order_history = []
price_history = []
portfolio_history = []
buying_power = STARTING_CASH
scalers = {}
model = None
cov_matrix = None

while day < end_day:
    logging.info('-----------------------------------------------------------')
    logging.info('-----------------------------------------------------------')
    
    day_ts = pd.Timestamp(day, tz = 'America/New_York')
    calendar = api.get_calendar(start = day_ts.isoformat(), end = (day_ts + relativedelta(days = 1)).isoformat())
    
    if calendar[0].date != day:
        logging.info('Skipping {}, the market is closed.'.format(day.strftime('%Y-%m-%d')))

        day = day + relativedelta(days = 1)
        continue

    logging.info('Working on {}'.format(day.strftime('%Y-%m-%d')))

    preds, orders, prices, model, scalers, cov_matrix = TradingDay(day, portfolio, buying_power, api, scalers, model, cov_matrix)
    logging.info('----Closing prices for the day:' + str(day))
    logging.info(prices)
    logging.info('----Predictions')
    logging.info(preds)
    logging.info('----Orders for next day open:')
    logging.info(orders)
    
    if len(orders)==0:
        logging.info('No orders for the day')
        day = day + relativedelta(days = 1)
        continue

    next_day_prices, portfolio, buying_power = Execution(day, orders, portfolio, buying_power)
    
    logging.info('----Prices at Open of Next Day:')
    logging.info(next_day_prices)
    logging.info('-----Portfolio:')
    logging.info(portfolio)
    logging.info('Buying Power {}'.format(buying_power))
    logging.info('Portfolio Value {}'.format(buying_power + np.sum(portfolio['Value'])))

    order_history.append(orders)
    price_history.append(prices)
    portfolio_history.append(portfolio)

    day = day + relativedelta(days = 1)

logging.info('Final buying power: {}'.format(buying_power))
logging.info('Final asset value: {}'.format(np.sum(portfolio['Value'])))
logging.info('Final portfolio value: {}'.format(buying_power + np.sum(portfolio['Value'])))
logging.info('Final rate of return: {}'.format((buying_power + np.sum(portfolio['Value'])) / STARTING_CASH))

logging.info(portfolio)

logging.info('-----------------------------------------------------------')
logging.info('----------Baseline Return')
logging.info('-----------------------------------------------------------')

baseline = GetLongReturns(IdentifyStocksOfInterest(), api, START_DATE, END_DATE)

logging.info(baseline)

logging.info(datetime.now())


import pickle as pk
with open('./data/logs/order_history', 'wb') as f:
    pk.dump(order_history, f)

with open('./data/logs/price_history', 'wb') as f:
    pk.dump(price_history, f)

with open('./data/logs/portfolio_history', 'wb') as f:
    pk.dump(portfolio_history, f)


# with open('./test_pickle', 'rb') as f:
#     ok = pk.load(f)