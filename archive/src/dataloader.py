#%%
import pandas as pd
import json
import os
import dotenv
dotenv.load_dotenv()
import requests
from datetime import datetime
from alpaca.data import CryptoHistoricalDataClient, StockHistoricalDataClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
# %%

class AlpacaAPI:
    def __init__(self, type='stock'):
        self.type = type
        if type == 'stock':
            self.data_client = StockHistoricalDataClient(os.environ["APCA_API_KEY"],  os.environ["APCA_SECRET_KEY"])
        elif type == 'crypto':
            raise NotImplementedError('alpacas crypto doesnt have recent data... past 7/14')
            self.data_client = CryptoHistoricalDataClient()
        else:
            raise ValueError(type)
        
    def get_current_quotes(self, symbols):
        if self.type == 'stock':
            request_params = StockLatestQuoteRequest(symbol_or_symbols=symbols)
            response = self.data_client.get_stock_latest_quote(request_params)
        elif self.type == 'crypto':
            request_params = CryptoLatestQuoteRequest(symbol_or_symbols=symbols)
            response = self.data_client.get_crypto_latest_quote(request_params)

        quotes = {
            symbol: data.ask_price
            for symbol, data in response.items()
        }
        return quotes

class CoinAPI:
    def __init__(self):
        self.api_key = os.environ['COINAPI_API_KEY']
        self.headers = {'X-CoinAPI-Key' : self.api_key}
        self.base_url = 'https://rest.coinapi.io/v1/'
        
    def get(self, url, use_base = True):     
        if use_base:
            url = self.base_url + url
        print(url)
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response
        
    def get_all_symbols(self, filter = True):
        url = 'symbols'
        if filter:
            # url += f"?filter_symbol_id={filter_symbol_id}&filter_exchange_id={filter_exchange_id}&filter_asset_id={filter_asset_id}
            url += f"?filter_exchange_id=COINBASE"
            url += f"&filter_asset_id=USD"
        
        response = self.get(url)
        
        return response
    
    def get_historical_quotes(self, symbol_id, time_start: datetime, time_end: datetime=None, limit=None):

        url = f"quotes/{symbol_id}/history?"
        params = []
        if time_start:
            # assert isinstance(time_start, datetime)
            params.append(f"time_start={time_start.isoformat()}")
        if time_end:
            # assert isinstance(time_end, datetime)
            params.append(f"time_end={time_end.isoformat()}")
        if limit:
            params.append(f"limit={limit}")
        url += '&'.join(params)
        return self.get(url)
    
    def get_current_quote(self, symbol_id):

        url = f"quotes/{symbol_id}/current"
        response = self.get(url)
        response.raise_for_status()
        return response
    
    def get_historical_prices(self, symbol_id, time_start: datetime, time_end: datetime=None, limit=None):
        data = self.get_historical_quotes(symbol_id, time_start, time_end, limit)
        print(data.text)
        data = pd.DataFrame.from_records(json.loads(data.text))
        #TODO: ask price or bid price
        out = data.set_index('time_exchange')['ask_price']
        out.name = symbol_id
        out.index.name = 'Time'
        #Valdiate that the time_exchange and time_coinbase are close enough
        return out

    def get_symbol_df(self, filter = True):
        response = self.get_all_symbols(filter=filter)
        return pd.DataFrame.from_records(json.loads(response.text))

class DataLoader:
    def __init__(self, ):
        raise NotImplementedError('Now just using alpaca API')
        self.api = CoinAPI()
        # self.symbol_df = self.api.get_symbol_df(filter=True)
        self.symbol_map = {
            'BTC': {'coin_api': 'COINBASE_SPOT_BTC_USD', 'alpaca': 'BTC/USD'},
            'ETH': {'coin_api': 'COINBASE_SPOT_ETH_USD', 'alpaca': 'ETH/USD'},
            'LTC': {'coin_api': 'COINBASE_SPOT_LTC_USD', 'alpaca': 'LTC/USD'},
        }
        self.coin_to_alpaca = {
            val['coin_api']: val['alpaca']
            for val in self.symbol_map.values()
        }
        self.alpaca_to_coin = {
            val['alpaca']: val['coin_api']
            for val in self.symbol_map.values()
        }
        
    def get_historical_prices(self, symbol_id, time_start: datetime, time_end: datetime=None, limit=None):
        if isinstance(symbol_id, list):
            df = pd.concat([
                self.get_historical_prices(id, time_start, time_end, limit)
                for id in symbol_id
            ])
        elif isinstance(symbol_id, str):
            df = self.api.get_historical_prices(symbol_id, time_start, time_end, limit)
        else:
            raise TypeError(f"Symbol is of unsupported type: {symbol_id}")
        return df
    
    def get_current_quote(self, symbol_id):
        if isinstance(symbol_id, list):
            return {
                id: self.get_current_quote(id)
                for id in symbol_id
            }
        else:
            response = self.api.get_current_quote(symbol_id)
            data = json.loads(response.text)
            return data['ask_price']
        
