
# %load_ext autoreload
# %autoreload 2
# from lstm import *
# from s3 import S3Client

# s3 = S3Client()
# # s3.download_file('models/test/lstm_best.pt', 'lstm_best.pt')
# # s3.download_file('models/test/lstm_best_startup_params.json', 'lstm_best_startup_params.json')
# # s3.read_csv('models/test/lstm_history.csv').set_index('epoch').plot()
# model = load_model_from_params('lstm_best.pt')
# time_series = get_time_series() 
# train, val, test = split_time_series(time_series)
# pd.Series(time_series).plot()
# s3.read_csv('models/test/test_preds.csv')[['actual', 'preds']].plot()

#%%
from config import *
import glob
import os
import pandas as pd

parquet_files = glob.glob(os.path.join(ROOT_DIR/'data/top_10/', '*.parquet'))

df_list = []
for file in parquet_files:
    df = pd.read_parquet(file)
    df_list.append(df)

df = pd.concat(df_list)

#%%
import numpy as np
df = df.reset_index()
df['start'] = df['start'].astype(int)
min_time = df.reset_index()['start'].min()
max_time = df.reset_index()['start'].max()
all_times = np.arange(min_time, max_time, 60)
all_times = set(all_times)
available_times = df['start'].sort_values().unique()
available_times = set(available_times)

#%%
missing_times = [c for c in all_times if c not in available_times]
missing_times = sorted(missing_times)
missing_windows = []
start = missing_times[0]
prev_time = start
for time in missing_times[1:]:
    if (time - prev_time) > 60:
        missing_windows.append((start, prev_time))
        start = time
        prev_time = time
    else:
        prev_time = time
missing_windows.append((start, prev_time))


#%%
from config import *
from coinbase_client import Client
from coinbase_trader import Trader
import datetime
client = Client()
trader = Trader(client)

ids = df['product_id'].unique().tolist()

data_list = []
for start, end in missing_windows:
    if end == start:
        end += 60
    
    try:
        data = trader.get_candles(
            ids,
            datetime.datetime.fromtimestamp(start),
            datetime.datetime.fromtimestamp(end),
            'ONE_MINUTE',
            progress = True,
            candles_per_batch=150
        )
    except:
        print(f"Failed to pull data from {datetime.datetime.fromtimestamp(start)} to {datetime.datetime.fromtimestamp(end)}")
    
    data_list.append(data) 
    
#%%
missing_df = pd.concat(data_list)
missing_df = missing_df[~missing_df.index.duplicated(keep='first')]

#%%
final_df = pd.concat([missing_df.reset_index(), df])

df.to_parquet(ROOT_DIR / 'data' / 'top_10_one_minute.parquet', index=False)

# %%
def concatenate_files():

    


    
    
    
    
    df = df.reset_index()
    for c in ['start']:
        df[c] = df[c].astype(int)
    for c in ['low', 'high', 'open', 'close', 'volume']:
        df[c] = df[c].astype(float)
    # df.to_parquet(ROOT_DIR / 'data' / 'top_100_one_minute_20230601_20231224.parquet', index=True)
    return df