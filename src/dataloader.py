from config import *
import pandas as pd
from s3 import S3Client
from functools import lru_cache

@lru_cache()
def load_price_data(granularity, ids, s3 = False):
    #TODO: improve loader to be a dataloader which has local loader and s3 loader
    assert granularity in GRANULARITY_TO_SECONDS.keys(), f"granularity must be one of {list(GRANULARITY_TO_SECONDS.keys())}"
    
    if isinstance(ids, str):
        ids = [ids]
    
    l = []
    
    if s3:
        s3 = S3Client()
        for id in ids:
            l.append(s3.read_parquet(f'data/prices/{granularity}/{id}/data.parquet'))
    else:
        for id in ids:
            l.append(pd.read_parquet(ROOT_DIR/f'data/prices/{granularity}/{id}/data.parquet'))
    
    return pd.concat(l)

@lru_cache()
def load_returns(granularity, ids, s3 = False):
    prices = load_price_data(granularity, ids, s3)
    prices['start'] = pd.to_datetime(prices['start'], unit='s')
    prices = prices[['start', 'product_id', 'close']].pivot(index = 'start', columns = 'product_id', values = 'close')
    returns = prices.pct_change().iloc[1:]
    return returns