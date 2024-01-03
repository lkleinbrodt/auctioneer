#%%
import sys
import os
import os
import glob
import pandas as pd
import numpy as np

#adds auctioneer/src to path for allowing imports
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from config import *
from coinbase_client import Client
from coinbase_trader import Trader
import pandas as pd
import datetime
from concurrent.futures import ThreadPoolExecutor

GRANULARITY = 'FIFTEEN_MINUTE'

#either set currencies or set k to get teh top k most popular
CURRENCIES = [
]

K_CURRENCIES = 30
if CURRENCIES:
    K_CURRENCIES = None

START_DATE = datetime.datetime(2020, 1, 1)
END_DATE = datetime.datetime(2023, 12, 24)
granularity_in_seconds = GRANULARITY_TO_SECONDS[GRANULARITY]

client = Client()
trader = Trader(client)
logger = create_logger(__name__, file = ROOT_DIR/'data/pull_price_log.log')

OUTPUT_DIR = ROOT_DIR/f"data/prices/{GRANULARITY}/"
os.makedirs(OUTPUT_DIR, exist_ok = True)

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

def pull_and_save(id, start, end, granularity):
    assert isinstance(id, str)
    
    filepath = OUTPUT_DIR/f"{id}/raw/{start.strftime('%Y%m%d_%H%M%S')}_{end.strftime('%Y%m%d_%H%M%S')}.parquet"
    if filepath.exists():
        logger.info(f"Already pulled {filepath}")
        return True
    
    try:
        price_data = trader.get_candles(
            id, 
            start,
            end, 
            granularity,
            progress=True
        )
        
        price_data.to_parquet(filepath, index=True)
    except:
        logger.exception(f"Failed to pull data from {start} to {end}")
    return True

def async_pull_and_save(periods, id, granularity):
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = [
            executor.submit(pull_and_save, id, period[0], period[1], granularity) 
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
    
    id = df['product_id'].unique().tolist()
    assert len(id) == 1, 'current version is meant to do one id at a time'
    id = id[0]

    missing_times = find_missing_times(df)
    if len(missing_times) == 0:
        return True
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

    async_pull_and_save(periods, id, 'ONE_MINUTE')
    
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

def upload_to_s3(ids, granularity):
    from s3 import S3Client
    s3 = S3Client()
    for id in ids:
        s3.upload_file(ROOT_DIR/f'data/prices/{granularity}/{id}/data.parquet', f'data/prices/{granularity}/{id}/data.parquet')

def main(start_date, end_date, granularity):
    if CURRENCIES:
        ids = CURRENCIES
    else:
        ids = get_top_ids(K_CURRENCIES)
    
    periods = get_periods(start_date, end_date, n_days = 7)
    for id in ids:
        output_dir = OUTPUT_DIR/f"{id}/raw/"
        os.makedirs(output_dir, exist_ok = True)
        async_pull_and_save(periods, id, granularity)
        df = concatenate_files(output_dir)
        pull_large_missing_windows(df)
        df = concatenate_files(output_dir)
        
        df.to_parquet(OUTPUT_DIR/f'{id}/data.parquet', index=True)
    
    upload_to_s3(ids, granularity)

#%%
if __name__ == '__main__':
    main(START_DATE, END_DATE, GRANULARITY)