#%%
from auctioneer import *
import pandas as pd
from alpaca.data import CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from scipy.stats import linregress
import numpy as np
import pandas as pd
import datetime as dt

import os
# APCA_API_KEY = os.getenv('APCA_API_KEY')
# APCA_SECRET_KEY = os.getenv('APCA_SECRET_KEY')
APCA_API_KEY='PKPSF4L3O68TT9B96VQ6'
APCA_SECRET_KEY='7CVn20cy5ZzLW7BWEVQYTFaDcjvtSYHQvoJ1BPSR'

trading_client = TradingClient(APCA_API_KEY, APCA_SECRET_KEY, paper=True)


# Date Variables
WINDOW_SIZE = 100
SLEEP_TIME = 60

#%%
def check_positions(symbol):
    positions = trading_client.get_all_positions()
    if symbol in str(positions):
        return 1
    return 0

def pull_closes(security, start_date, end_date):
    
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[security],
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date,
        limit = WINDOW_SIZE,
    )

    bars_df = client.get_crypto_bars(request_params).df
    data = bars_df.droplevel('symbol').copy()['close']

    return data
#%%

#%%
def slope_bot():
    # print('running')
    # try:

    symbols = ['BTC/USD','ETH/USD','DOGE/USD','SHIB/USD','MATIC/USD','ALGO/USD','AVAX/USD','LINK/USD','SOL/USD']

    dfl = []
    from datetime import datetime
    from dateutil.relativedelta import relativedelta
    
    start_date = pd.to_datetime(datetime.now()) - relativedelta(minutes = WINDOW_SIZE)
    end_date = pd.to_datetime(datetime.now())

    for symbol in symbols:
        try:
            data = pull_closes(symbol, start_date, end_date)
            data = pd.DataFrame(data).rename(columns={"close": str(symbol)})
            dfl.append(data)
        except:
            pass

    df = pd.concat(dfl, axis = 1)

    def slope(x):
        x = x[~x.isna()]
        if len(x) > 0:
            return linregress(x.reset_index().index, x.values)[0]
        else:
            return 0
    slopes = df.apply(slope)
    print(slopes)
    
    buy_symbols = slopes[slopes > 0.05].index
    sell_symbols = slopes[slopes < -0.05].index
    
    for symbol in buy_symbols:
        account_info = trading_client.get_account()
        balance = float(account_info.non_marginable_buying_power)
        price = df[symbol].dropna().iloc[-1]
        amount = (balance/50)# / price

        if amount > 1:
            market_order_data = MarketOrderRequest(
                symbol = symbol,
                notional=amount,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC
            )
            trading_client.submit_order(
                order_data=market_order_data)
            print(f"Bought {symbol} at approx. {price}")

    positions = trading_client.get_all_positions()
    positions = {x.symbol.replace('USD', '/USD'): x.qty for x in positions}
    for symbol in sell_symbols:
        
        qty = float(positions.get(symbol, 0))

        if qty > 0:
                price = df[symbol].dropna().iloc[-1]

                market_order_data = MarketOrderRequest(
                    symbol = symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC
                )
                trading_client.submit_order(
                    order_data=market_order_data)
                print(f"Sold {symbol} at approx. {price}")
        

    # except Exception as e:
    #     print (e)

def main():
    runs = 0
    from time import sleep
    while True:
        runs +=1
        slope_bot()
        print(runs)
        try:
            sleep(SLEEP_TIME)
        except KeyboardInterrupt:
            return True

if __name__ == '__main__':
    main()

# #%%
# async def quote_data_handler(data):
#     slope_bot()
# #%%
# crypto_stream = CryptoDataStream(APCA_API_KEY, APCA_SECRET_KEY, raw_data=True)
# crypto_stream.subscribe_bars(quote_data_handler, 'BTC//USD','ETH/USD','DOGE/USD','SHIB/USD','MATIC/USD','ALGO/USD','AVAX/USD','LINK/USD','SOL/USD')
# crypto_stream.run()
# # %%