from config import *
import pandas as pd

def load_price_data(granularity, ids):
    assert granularity in GRANULARITY_TO_SECONDS.keys(), f"granularity must be one of {list(GRANULARITY_TO_SECONDS.keys())}"
    
    if isinstance(ids, str):
        ids = [ids]
    
    l = []
    for id in ids:
        l.append(pd.read_parquet(ROOT_DIR/f'data/prices/{granularity}/{id}/data.parquet'))
    
    return pd.concat(l)

def load_returns(granularity, ids):
    prices = load_price_data(granularity, ids)
    prices['start'] = pd.to_datetime(prices['start'], unit='s')
    prices = prices[['start', 'product_id', 'close']].pivot(index = 'start', columns = 'product_id', values = 'close')
    returns = prices.pct_change().iloc[1:]
    return returns