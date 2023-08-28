import dataloader
import pandas as pd
import os
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from datetime import datetime
import strategies

class SlopeBot:
    def __init__(self, symbol, window_size=100, threshold: float = 0.05):
        self.window_size = window_size
        self.threshold = threshold
        self.dataloader = dataloader.AlpacaAPI()
        self.strategy = strategies.SlopeStrategy(window_size, threshold)
        
        self.symbol = symbol #TODO: support multiple symbols?
        
        self.quote_history = pd.Series(dtype = float)
        
        try:
            APCA_API_KEY = os.environ["APCA_API_KEY"]
            APCA_SECRET_KEY = os.environ["APCA_SECRET_KEY"]
        except KeyError:
            raise KeyError(
                "Must set APCA_API_KEY and APCA_SECRET_KEY in environments variable"
            )
        self.trading_client = TradingClient(APCA_API_KEY, APCA_SECRET_KEY, paper=True)
        
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
                    price = data.dropna().iloc[-1]
                    # price = data[symbol].dropna().iloc[-1], only using one symbol now
                else:
                    price = "--"
                print(f"Bought {amount} {symbol} at approx. {price}")
    
    def get_current_data(self):
        
        current_quote = self.dataloader.get_current_quotes(self.symbol)
        current_quote = current_quote[self.symbol]
        print("Current Price: ", current_quote)
        self.quote_history.loc[datetime.now()] = current_quote
           
    def act(self):
        self.get_current_data()
        action = self.strategy(self.quote_history) #TODO: need to sort?
        if action == 'buy':
            self.buy(buy_symbols=[self.symbol], data = self.quote_history)
        elif action == 'sell':
            self.sell(sell_symbols=[self.symbol], data = self.quote_history)