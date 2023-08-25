#%%
from datetime import datetime
from dateutil.relativedelta import relativedelta
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
from logging import Logger
logger = Logger('main')
import os
import dotenv
dotenv.load_dotenv()

class SlopeBot:
    def __init__(self, window_size = 100, threshold: float = .05):
        self.window_size = window_size
        self.threshold = threshold
        
        self.symbols = ['BTC/USD','ETH/USD','DOGE/USD','SHIB/USD','MATIC/USD','ALGO/USD','AVAX/USD','LINK/USD','SOL/USD']
        
        try:
            APCA_API_KEY = os.environ['APCA_API_KEY']
            APCA_SECRET_KEY = os.environ['APCA_SECRET_KEY']
        except KeyError:
            raise KeyError("Must set APCA_API_KEY and APCA_SECRET_KEY in environments variable")
        self.trading_client =  TradingClient(APCA_API_KEY, APCA_SECRET_KEY, paper=True)

    def check_positions(self, symbol: str):
        positions = self.trading_client.get_all_positions()
        if symbol in str(positions):
            return 1
        return 0

    def pull_closes(self, security: str, start_date: datetime, end_date: datetime):
        
        client = CryptoHistoricalDataClient()
        request_params = CryptoBarsRequest(
            symbol_or_symbols=[security],
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date,
            limit = self.window_size,
        )

        bars_df = client.get_crypto_bars(request_params).df
        data = bars_df.droplevel('symbol').copy()['close']

        return data
    
    def get_recent_data(self):
        
        dfl = []
        
        now = datetime.now() - relativedelta(days = 365) #TODO: alpaca doesnt give recent data... alpaca sucks
        start_date = pd.to_datetime(now) - relativedelta(minutes = self.window_size)
        end_date = pd.to_datetime(now)
        
        dfl = []
        for symbol in self.symbols:
            try:
                data = self.pull_closes(symbol, start_date, end_date)
                data = pd.DataFrame(data).rename(columns={"close": str(symbol)})
                dfl.append(data)
            except Exception as e:
                logger.error(e)
                pass
        
        df = pd.concat(dfl, axis = 1)
        
        return df
    
    def slope(self, x):
        x = x[~x.isna()]
        if len(x) > 0:
            return linregress(x.reset_index().index, x.values)[0]
        else:
            return 0
        
    def sell(self, sell_symbols, data = None):
        # positions = self.trading_client.get_all_positions() it's broken. alpaca sucks
        # positions = {x.symbol.replace('USD', '/USD'): x.qty for x in positions}
        for symbol in sell_symbols:
            response = self.trading_client.get(f"/positions/{symbol}")
            
            # qty = round(float(positions.get(symbol, 0)), 2)
            qty = float(response['qty'])

            if qty > 0:
                    
                    market_order_data = MarketOrderRequest(
                        symbol = symbol,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.GTC
                    )
                    self.trading_client.submit_order(
                        order_data=market_order_data
                    )
                    if data is not None:
                        price = data[symbol].dropna().iloc[-1]
                    else:
                        price = '--'
                    print(f"Sold {symbol} at approx. {price}")

    def buy(self, buy_symbols, data = None):
        for symbol in buy_symbols:
            account_info = self.trading_client.get_account()
            balance = float(account_info.non_marginable_buying_power)
            
            amount = round(balance/50, 2)# / price

            if amount > 0:
                market_order_data = MarketOrderRequest(
                    symbol = symbol,
                    notional=amount,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC
                )
                self.trading_client.submit_order(
                    order_data=market_order_data
                )

                if data is not None:
                    price = data[symbol].dropna().iloc[-1]
                else:
                    price = '--'
                print(f"Bought {amount} {symbol} at approx. {price}")
        
    def act(self):
        
        
        data = self.get_recent_data()

        slopes = data.apply(self.slope)
        # print(slopes)
        
        buy_symbols = slopes[slopes > self.threshold].index
        sell_symbols = slopes[slopes < -self.threshold].index
        
        self.sell(sell_symbols, data)
        self.buy(buy_symbols, data)

def main():
    slope_bot = SlopeBot(
        window_size=100,
        threshold = .05
    )
    runs = 0
    from time import sleep
    while True:
        runs +=1
        slope_bot.act()
        print(runs)
        try:
            sleep(5)
        except KeyboardInterrupt:
            return True

if __name__ == '__main__':
    main()