import os
import warnings
import logging
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from dateutil.relativedelta import relativedelta
import alpaca_trade_api as alp

with open('paper_api_keys.txt') as api_file:
    api_keys = api_file.read().replace('\n', '').split(',')
    alpaca_api = {a.split('=')[0]: a.split('=')[1] for a in api_keys}
    
api = alp.REST(key_id=alpaca_api['APCA_API_KEY_ID'], secret_key = alpaca_api['APCA_API_SECRET_KEY'], base_url=alpaca_api['APCA_API_BASE_URL'])

from Functions import *

########## PARAMETERS ##############

start_date = '2019-11-01'
end_date = '2020-12-01'

movement_threshold = 0

starting_cash = 100_000
max_per_buy = .01
max_per_sell = .25
max_epochs = 25

########## END PARAMETERS #############

print('Starting Script: ')
print(datetime.now())

#Get Alpaca APIs

### Get portfolio
###
#portfolio = pd.DataFrame([{'Symbol': p.symbol, 'Quantity': float(p.qty), 'Value': float(p.market_value)} for p in api.list_positions()])

portfolio = pd.DataFrame(columns = ['Symbol', 'Quantity', 'Value'])
portfolio['Symbol'] = portfolio['Symbol'].astype(str)
portfolio['Quantity'] = portfolio['Quantity'].astype(float)
portfolio['Value'] = portfolio['Value'].astype(float)

### Define One Day of trading behavior, should take in a date, and output the actions that will be taken

def TradingDay(current_day, portfolio, buying_power, model_dict, api, epochs):
    ###Get stocks of Interest
    #soi = IdentifyStocksOfInterest()
    soi = ['SPY']
    ###combine with symbols from portfolio
    stocks_to_predict = list(set(portfolio['Symbol'].tolist() + soi))

    ### Make predictions
    preds = []
    current_prices = dict([])
    
    ### For now we are using same data for all of them, but in future data grab
    ### should be inside for loop
    
    data = GetHistoricalData(stocks_to_predict, end = current_day, api = api)
    
    stocks_to_predict = [col for col in data if col in stocks_to_predict]
    
    for symbol in stocks_to_predict:
        if symbol not in model_dict.keys():
            print('Creating model for' + str(symbol) + str(current_day))
            model_dict[symbol] = CreateModel(data)
            model_dict[symbol] = TrainModel(model_dict[symbol], symbol, data, epochs) 
        elif current_day.day in [1, 15]:
            print('training' + str(symbol) + str(current_day))
            model_dict[symbol] = TrainModel(model_dict[symbol], symbol, data, epochs)    
        preds.append(Predict7DayHigh(model_dict[symbol], symbol, data))
        current_prices[symbol] = data.iloc[-1][symbol]
    preds = dict(preds)
    adjusted_movements = {sym: (preds[sym]/current_prices[sym]) for sym in preds.keys()}
    predicted_stocks = list(preds.keys())

    ### Determine which to buy and which to sell
    orders = dict([])
    for symbol in predicted_stocks:
        
        if adjusted_movements[symbol] > movement_threshold:
            side = 'buy'
            quantity = (buying_power * max_per_buy) // current_prices[symbol]

        elif (adjusted_movements[symbol] < -movement_threshold) and (symbol in list(portfolio['Symbol'])):
            side = 'sell'
            quantity = np.max([1, np.floor(portfolio[portfolio['Symbol']==symbol]['Value'][0] * max_per_sell)])
            #quantity = portfolio[portfolio['Symbol']==symbol]['Quantity'].astype(int)[0]
        
        else:
            continue

        orders[symbol] = {'Side': side, 'Quantity': quantity}
    
    return preds, orders, current_prices, model_dict


def Execution(day_of_order, orders, portfolio, buying_power):

    if not isinstance(day_of_order, pd.Timestamp):
        day_of_order = pd.Timestamp(day_of_order, tz = 'America/New_York')

    ###First, ensure your dates are valid
    calendar = api.get_calendar(start = day_of_order, end = day_of_order + relativedelta(days = 5))

    assert calendar[0].date == day_of_order

    ### Now determine which day these trades will be placed
    day_of_execution = calendar[1].date
    
    next_day_prices = dict(GetDayQuotes(orders.keys(), api, day_of_execution, 'open'))
    
    for symbol in orders.keys():

        quantity = orders[symbol]['Quantity']
        price = next_day_prices[symbol]
        cost = quantity * next_day_prices[symbol]

        if symbol in list(portfolio['Symbol']):
            #print('Found {} in portfolio'.format(symbol))
            if orders[symbol]['Side'] == 'buy':
                #print('Trying to buy {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
                if cost > buying_power:
                    #print('Did not buy, total cost is {} and we only have {}'.format(cost, buying_power))
                    continue
                buying_power -= cost
                portfolio.loc[portfolio['Symbol']==symbol, 'Quantity'] += quantity
                portfolio.loc[portfolio['Symbol']==symbol, 'Value'] = portfolio.loc[portfolio['Symbol']==symbol, 'Quantity'] * price
                #print('Successfully bought {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
                #print('New buying power: ' + str(buying_power))
            elif orders[symbol]['Side'] == 'sell':
                #print('Selling {} shares of {} at {} per share for a total sale of {}.'.format(quantity, symbol, np.round(price,5), np.round(cost,5)))
                
                buying_power += cost
                portfolio = portfolio[portfolio['Symbol'] != symbol]
                #print('New buying power: ' + str(buying_power))
        else:
            #print('{} not found in portfolio'.format(symbol))
            #print('Trying to buy {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
            if cost > buying_power:
                #print('Did not buy, total cost is {} and we only have {}'.format(cost, buying_power))
                continue
            buying_power -= cost
            portfolio = portfolio.append(pd.DataFrame([{'Symbol': symbol, 'Quantity': quantity, 'Value': quantity * price}]))
            #print('Successfully bought {} shares of {} at {} per share for a total cost of {}'.format(quantity, symbol, np.round(price, 5), np.round(cost,5)))
            #print('New buying power: ' + str(buying_power))

    return next_day_prices, portfolio, buying_power

day = pd.to_datetime(start_date)
end_day = pd.to_datetime(end_date)

order_history = []
price_history = []
portfolio_history = []
buying_power = starting_cash
model_dict = dict()
 
while day < end_day:
    print('-----------------------------------------------------------')
    print('-----------------------------------------------------------')
    
    day_ts = pd.Timestamp(day, tz = 'America/New_York')
    calendar = api.get_calendar(start = day_ts.isoformat(), end = (day_ts + relativedelta(days = 1)).isoformat())
    
    if calendar[0].date != day:
        print('Skipping {}, the market is closed.'.format(day.strftime('%Y-%m-%d')))

        day = day + relativedelta(days = 1)
        continue

    print('Working on {}'.format(day.strftime('%Y-%m-%d')))

    preds, orders, prices, model_dict = TradingDay(day, portfolio, buying_power, model_dict, api, epochs = max_epochs)
    print('----Closing prices for the day:')
    print(prices)
    print('----Predictions for next 7 days:')
    print(preds)
    print('----Orders for next day open:')
    print(orders)
    
    if len(orders)==0:
        print('No orders for the day')
        day = day + relativedelta(days = 1)
        continue

    next_day_prices, portfolio, buying_power = Execution(day, orders, portfolio, buying_power)
    
    print('----Prices at Open:')
    print(next_day_prices)
    print('-----Portfolio:')
    print(portfolio)
    print('Buying Power {}'.format(buying_power))
    print('Portfolio Value {}'.format(buying_power + np.sum(portfolio['Value'])))


    order_history.append(orders)
    price_history.append(prices)
    portfolio_history.append(portfolio)

    day = day + relativedelta(days = 1)


print('Final buying power: {}'.format(buying_power))
print('Final asset value: {}'.format(np.sum(portfolio['Value'])))
print('Final portfolio value: {}'.format(buying_power + np.sum(portfolio['Value'])))
print('Final rate of return: {}'.format((buying_power + np.sum(portfolio['Value'])) / starting_cash))

print(portfolio)

print('-----------------------------------------------------------')
print('----------Baseline Return')
print('-----------------------------------------------------------')

baseline = GetLongReturns(IdentifyStocksOfInterest(), api, start_date, end_date)

print(baseline)

print(datetime.now())


import pickle as pk
with open('./data/logs/order_history', 'wb') as f:
    pk.dump(order_history, f)

with open('./data/logs/price_history', 'wb') as f:
    pk.dump(price_history, f)

with open('./data/logs/portfolio_history', 'wb') as f:
    pk.dump(portfolio_history, f)


# with open('./test_pickle', 'rb') as f:
#     ok = pk.load(f)