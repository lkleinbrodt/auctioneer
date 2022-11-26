import matplotlib.pyplot as plt
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.requests import CryptoBarsRequest

class Portfolio:

    def __init__(self, balance, data, holdings = {}):
        self.balance = balance
        self.data = data
        self.holdings = holdings
        self.transaction_log = []

    def value(self, date = None):
        #only supports one security for now
        if date is not None:
            price = self.data.loc[date]['close']
        else:
            price = self.data.iloc[-1]['close']
        
        aum = 0
        for security, amount in self.holdings.items():
            aum += price * amount
        
        out = {
            'Cash': self.balance,
            'AUM': aum,
            'Total': self.balance + aum
        }

        return out

    def execute(self, date, order_dict):
        if (order_dict is None) | (order_dict == {}):
            return 
        #just goes in order of order
        for security, order in order_dict.items():
            price = self.data.loc[date]['open']


            if order['action'] == 'buy':

                if order['amount'] == 'max':
                    amount = self.balance // price
                else:
                    amount = order['amount']# // price
                
                if amount > 0:
                    if amount * price > self.balance:
                        amount = self.balance // price
                    self.holdings[security] = self.holdings.get(security, 0) + amount
                    self.balance -= amount * price
                    self.transaction_log += [{'date': date, 'action': 'buy', 'amount': amount, 'value': amount*price}]

            elif order['action'] == 'sell':

                if order['amount'] == 'max':
                    amount = self.holdings.get(security, 0)
                else:
                    #TODO: not right
                    amount = order['amount']# // price

                if self.holdings.get(security, 0) == 0:
                    continue
                if amount > self.holdings[security]:
                    amount = self.holdings[security]
                
                self.holdings[security] -= amount
                self.balance += amount * price
                self.transaction_log += [{'date': date, 'action': 'sell', 'amount': amount, 'value': amount*price}]

    def transaction_log_summary(self):
        buys = []
        sells = []
        for order in self.transaction_log:
            if order['action'] == 'buy':
                buys += [order['value']]
            if order['action'] == 'sell':
                sells += [order['value']]
        
        # print(f"""
        # {len(buys)+len(sells)} total transactions.
        # {len(buys)} buy orders, totalling {sum(buys)}.
        # {len(sells)} sell orders, totalling {sum(sells)}.
        # """)
        return {'buys': (len(buys), sum(buys)), 'sells': (len(sells), sum(sells))}

    def plot_transactions(self):
        transaction_log = self.transaction_log

        for t in transaction_log:
            if t['action'] == 'buy':
                color = 'green'
            else:
                color='red'
            plt.axvline(t['date'], color=color, alpha = .05)
            
        plt.plot(self.data['close'])
        plt.show()

def pull_data(security, start_date, end_date):
    client = CryptoHistoricalDataClient()
    request_params = CryptoBarsRequest(
        symbol_or_symbols=[security],
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date,
        limit = 10_000
    )

    bars_df = client.get_crypto_bars(request_params).df
    data = bars_df.droplevel('symbol').copy()

    return data