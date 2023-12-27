import sys
import os
import os
import glob
import pandas as pd

#adds auctioneer/src to path for allowing imports
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))

from config import *
from coinbase_client import Client
from coinbase_trader import Trader
import pandas as pd
import datetime

logger = create_logger(__name__, file = ROOT_DIR/'data'/'logs'/'pull_price_data.log')

OUTPUT_DIR = ROOT_DIR/ 'data/top_10'
os.mkdir(OUTPUT_DIR)

client = Client()
trader = Trader(client)

START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2023, 12, 24)
GRANULARITY = 'ONE_MINUTE'

def get_ids():
    products = client.listProducts(product_type = 'SPOT')
    products = products['products']
    usdc_products = [product for product in products if product['quote_currency_id'] == 'USD']

    volumes = pd.Series({x['product_id']: x['volume_24h'] for x in usdc_products})
    volumes[volumes == ''] = 0
    volumes = volumes.astype(float)

    prices = pd.Series({x['product_id']: x['price'] for x in usdc_products})
    prices[prices == ''] = 0
    prices = prices.astype(float)

    adj_volumes = volumes * prices

    rel_ids = adj_volumes.sort_values().tail(10).index.tolist()
    
    return rel_ids

def get_periods(start_date, end_date):
    # Splitting the period into 1-week length periods
    periods = []
    current_start = start_date
    current_end = start_date + datetime.timedelta(days=7)
    periods.append((current_start, current_end))
    
    while current_end < end_date:
        current_start = current_end
        current_end += datetime.timedelta(days=7)
        current_end = min([current_end, end_date])
        periods.append((current_start, current_end))

    return periods


def batch_job(rel_ids, start, end, granularity):
    try:
        price_data = trader.get_candles(
            rel_ids, 
            start,
            end, 
            granularity,
            progress=True
        )

        price_data.to_parquet(
            OUTPUT_DIR / f"{granularity}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet",
            index=True
        )
    except:
        logger.exception(f"Failed to pull data from {start} to {end}")
    return True


def concatenate_files():

    parquet_files = glob.glob(os.path.join(ROOT_DIR/'data/one_minute_2023/', '*.parquet'))

    df_list = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df_list.append(df)


    df = pd.concat(df_list)
    
    # df = df.reset_index()
    # df['start'] = df['start'].astype(int)
    # min_time = df.reset_index()['start'].min()
    # max_time = df.reset_index()['start'].max()
    # all_times = np.arange(min_time, max_time, 60)
    # all_times = set(all_times)
    # available_times = df['start'].unique()
    # available_times = set(available_times)
    # [c for c in all_times if c not in available_times]
    
    df = df.reset_index()
    for c in ['start']:
        df[c] = df[c].astype(int)
    for c in ['low', 'high', 'open', 'close', 'volume']:
        df[c] = df[c].astype(float)
    # df.to_parquet(ROOT_DIR / 'data' / 'top_100_one_minute_20230601_20231224.parquet', index=True)
    return df

def main(start_date, end_date, granularity):
    logger.info(f"Pulling product data from {start_date} to {end_date} with granularity {granularity}")
    ids = get_ids()
    periods = get_periods(start_date, end_date)
    
    #for each period in period, run batch_job(rel_ids, period[0], period[1])
    #run it in parallel processing using as many cores as possible
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=5) as executor:
        results = [
            executor.submit(batch_job, ids, period[0], period[1], granularity) 
            for period in periods
        ]

        # Wait for all tasks to complete
        for future in results:
            future.result()

    
if __name__ == '__main__':
    main(START_DATE, END_DATE, GRANULARITY)