#%%
import sys
import os
import os
import glob
import pandas as pd
import numpy as np

#adds auctioneer/src to path for allowing imports
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src')))

from config import *
from coinbase_client import Client
from coinbase_trader import Trader
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor


OUTPUT_DIR = ROOT_DIR/ 'data/top_5_15m'

if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR / 'batches/'):
    os.mkdir(OUTPUT_DIR / 'batches/')

client = Client()
trader = Trader(client)
logger = create_logger(__name__, file = OUTPUT_DIR/'log.log')

START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2023, 12, 24)
GRANULARITY = 'FIFTEEN_MINUTE'
K_CURRENCIES = 5

granularity_in_seconds = GRANULARITY_TO_SECONDS[GRANULARITY]

def get_top_ids(k):
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

    rel_ids = adj_volumes.sort_values().tail(k).index.tolist()
    
    return rel_ids

def get_periods(start_date, end_date, n_days = 7):
    periods = []
    current_start = start_date
    current_end = start_date + datetime.timedelta(days=n_days)
    current_end = min([current_end, end_date])
    periods.append((current_start, current_end))
    
    while current_end < end_date:
        current_start = current_end
        current_end += datetime.timedelta(days=n_days)
        current_end = min([current_end, end_date])
        periods.append((current_start, current_end))

    return periods


def pull_and_save(rel_ids, start, end, granularity):
    try:
        price_data = trader.get_candles(
            rel_ids, 
            start,
            end, 
            granularity,
            progress=True
        )

        filepath = OUTPUT_DIR / f"batches/{granularity}_{start.strftime('%Y%m%d_%H%M%S')}_{end.strftime('%Y%m%d_%H%M%S')}.parquet"
        price_data.to_parquet(filepath, index=True)
    except:
        logger.exception(f"Failed to pull data from {start} to {end}")
    return True

def async_pull_and_save(periods, ids, granularity):
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = [
            executor.submit(pull_and_save, ids, period[0], period[1], granularity) 
            for period in periods
        ]

    # Wait for all tasks to complete
    for future in results:
        future.result()
    
    return True

def find_missing_times(df):
    times = df.reset_index()['start'].astype(int)
    min_time = times.min()
    max_time = times.max()
    all_times = np.arange(min_time, max_time, granularity_in_seconds)
    all_times = set(all_times)
    available_times = times.unique()
    available_times = set(available_times)
    missing_times = [c for c in all_times if c not in available_times]
    missing_times = sorted(missing_times)
    
    return missing_times

def times_to_windows(times):
    
    windows = []
    start = times[0]
    prev_time = start
    for time in times[1:]:
        if (time - prev_time) > granularity_in_seconds:
            windows.append((start, prev_time))
            start = time
            prev_time = time
        else:
            prev_time = time
    windows.append((start, prev_time))
    return windows
    

def pull_large_missing_windows(df):
    """this likely wont pull all of the missing data, but it should pull a lot of it"""

    missing_times = find_missing_times(df)
    missing_windows = times_to_windows(missing_times)
    large_missing_windows = [window for window in missing_windows if window[1] - window[0] > (granularity_in_seconds * 2)]

    import itertools
    periods = [
        get_periods(
            datetime.datetime.fromtimestamp(window[0]),
            datetime.datetime.fromtimestamp(window[1]),
            n_days = 1
        ) for window in large_missing_windows
    ]
    periods = list(itertools.chain.from_iterable(periods))

    async_pull_and_save(periods, df['product_id'].unique().tolist(), 'ONE_MINUTE')
    
    return True


def concatenate_files(dir):
    logger.info(f"Concatenating files in {dir}")

    parquet_files = glob.glob(os.path.join(dir, '*.parquet'))

    df_list = []
    for file in parquet_files:
        df = pd.read_parquet(file)
        df_list.append(df)


    df = pd.concat(df_list)
    df = df[~df.index.duplicated(keep='first')]
    df = df.reset_index()
    
    for c in ['start']:
        df[c] = df[c].astype(int)
    for c in ['low', 'high', 'open', 'close', 'volume']:
        df[c] = df[c].astype(float)

    return df

def main(start_date, end_date, granularity):
    # logger.info(f"Pulling product data from {start_date} to {end_date} with granularity {granularity}")
    ids = get_top_ids(K_CURRENCIES)
    periods = get_periods(start_date, end_date)
    async_pull_and_save(periods, ids, granularity)
    
    df = concatenate_files(OUTPUT_DIR / 'batches')
    pull_large_missing_windows(df)
    df = concatenate_files(OUTPUT_DIR / 'batches')
    
    df.to_parquet(OUTPUT_DIR / 'data.parquet', index=True)

#%%
if __name__ == '__main__':
    main(START_DATE, END_DATE, GRANULARITY)