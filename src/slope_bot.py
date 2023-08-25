# %%
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
import dataloader

logger = Logger("main")
import os
import dotenv

dotenv.load_dotenv()


class SlopeBot:
    def __init__(self, window_size=100, threshold: float = 0.05, symbol = None):
        raise NotImplemented('Use SlopeStrategy')
        self.window_size = window_size
        self.threshold = threshold
        self.dataloader = dataloader.AlpacaAPI()

        if symbol is None:
            self.symbols =['SPY']
            # [
            #     "SPY",
            #     "GLD",
            #     "TLT",
            #     "TSLA",
            # ]
        else:
            self.symbols = [symbol]
        
        self.quote_history = pd.DataFrame(dtype=float, columns=self.symbols)
        self.quote_history.index.name = 'Time'

        try:
            APCA_API_KEY = os.environ["APCA_API_KEY"]
            APCA_SECRET_KEY = os.environ["APCA_SECRET_KEY"]
        except KeyError:
            raise KeyError(
                "Must set APCA_API_KEY and APCA_SECRET_KEY in environments variable"
            )
        self.trading_client = TradingClient(APCA_API_KEY, APCA_SECRET_KEY, paper=True)

    def check_positions(self, symbol: str):
        positions = self.trading_client.get_all_positions()
        if symbol in str(positions):
            return 1
        return 0

    # def get_recent_data(self):
        
    #     symbols = self.symbols
    #     symbols = [self.dataloader.alpaca_to_coin[symbol] for symbol in symbols]
    #     now = datetime.utcnow().replace(second=0, microsecond=0)
    #     start_date = now - relativedelta(seconds = 100_000)
    #     df = self.dataloader.get_historical_prices(symbols, time_start = start_date)
    #     df.columns = df.columns.map(self.dataloader.coin_to_alpaca)
    #     return df
    
    def get_current_data(self, backtest_feed = None):
        
        #back from using COINAPI
        # symbols = self.symbols
        # symbols = [self.dataloader.alpaca_to_coin[symbol] for symbol in symbols]
        
        current_quotes = self.dataloader.get_current_quotes(self.symbols)
        series = pd.Series(current_quotes)
        self.quote_history.loc[datetime.now()] = series
        #TODO: drop after you get to window_size

    def slope(self, x):
        x = x[~x.isna()]
        if len(x) > 0:
            return linregress(x.reset_index().index, x.values)[0]
        else:
            return 0

    def sell(self, sell_symbols, data=None):
        # positions = self.trading_client.get_all_positions() it's broken. alpaca sucks
        # positions = {x.symbol.replace('USD', '/USD'): x.qty for x in positions}
        for symbol in sell_symbols:
            response = self.trading_client.get(f"/positions/{symbol}")

            # qty = round(float(positions.get(symbol, 0)), 2)
            qty = float(response["qty"])

            if qty > 0:
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC,
                )
                self.trading_client.submit_order(order_data=market_order_data)
                if data is not None:
                    price = data[symbol].dropna().iloc[-1]
                else:
                    price = "--"
                print(f"Sold {symbol} at approx. {price}")

    def buy(self, buy_symbols, data=None):
        for symbol in buy_symbols:
            account_info = self.trading_client.get_account()
            balance = float(account_info.non_marginable_buying_power)

            amount = round(balance / 50, 2)  # / price

            if amount > 0:
                market_order_data = MarketOrderRequest(
                    symbol=symbol,
                    notional=amount,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC, #TODO: not GTC
                )
                self.trading_client.submit_order(order_data=market_order_data)

                if data is not None:
                    price = data[symbol].dropna().iloc[-1]
                else:
                    price = "--"
                print(f"Bought {amount} {symbol} at approx. {price}")

    def act(self, backtest_feed = None):
        if backtest_feed is not None:
            self.quote_history = backtest_feed
        else:
            self.get_current_data()
        data = self.quote_history
        if data.shape[0] < self.window_size:
            print(f'Skipping, only have access to: {data.shape[0]} records')
            return

        slopes = data.apply(self.slope)
        # print(slopes)

        buy_symbols = slopes[slopes > self.threshold].index
        sell_symbols = slopes[slopes < -self.threshold].index

        self.sell(sell_symbols, data)
        self.buy(buy_symbols, data)


def main():
    slope_bot = SlopeBot(window_size=100, threshold=0.05)
    runs = 0
    from time import sleep

    while True:
        runs += 1
        slope_bot.act()
        print(runs)
        try:
            sleep(5)
        except KeyboardInterrupt:
            return True

#%%
bot = SlopeBot(5)
bot.act()
#%%
if __name__ == "__main__":
    main()

# %%
