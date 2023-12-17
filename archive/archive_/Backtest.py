# %%
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from dateutil import parser
import alpaca_trade_api as alpaca
import logging
from Functions import IdentifyStocksOfInterest

logging.basicConfig(
    filename="Logs/backtest_log.log",
    level="DEBUG",
    filemode="w",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

START_DATE = "2017-01-01"
END_DATE = "2017-09-01"

TIME_STEP = "15Min"

### Process
start_date_ts = pd.Timestamp(START_DATE, tz="America/New_York")
end_date_ts = pd.Timestamp(END_DATE, tz="America/New_York")
start_date_str = start_date_ts.isoformat()
end_date_str = end_date_ts.isoformat()

soi = IdentifyStocksOfInterest("SP")
# %%
with open("Data/paper_api_keys.txt") as api_file:
    api_keys = api_file.read().replace("\n", "").split(",")
    alpaca_api = {a.split("=")[0]: a.split("=")[1] for a in api_keys}

api = alpaca.REST(
    key_id=alpaca_api["APCA_API_KEY_ID"],
    secret_key=alpaca_api["APCA_API_SECRET_KEY"],
    base_url=alpaca_api["APCA_API_BASE_URL"],
)

# %%
#### Get Historical Data
### 100 symbols per request
soi_chunks = [soi[i : i + 100] for i in range(0, len(soi), 100)]

historical_data = pd.concat(
    [
        api.get_barset(
            symbols=symbols[:5],
            timeframe=TIME_STEP,
            after=start_date_str,
            until=end_date_str,
        ).df
        for symbols in soi_chunks
    ],
    axis=1,
)

# %%
# for some stupid fucking reason, the api starts with most recent data and then goes back
###Want data up until the first trading day after/on the start date
first_trading_day = pd.Timestamp(
    api.get_calendar(start=start_date_str, end=start_date_str)[0].date,
    tz="America/New_York",
)

print("First Data Day:", historical_data.index.min())
print("Want data starting", first_trading_day)
# %%
min_date = historical_data.index.min()
while min_date > first_trading_day:
    print("min date:" + str(min_date))
    new_data = pd.concat(
        [
            api.get_barset(
                symbols=symbols[:5],
                timeframe=TIME_STEP,
                limit=1000,
                after=start_date_str,
                until=min_date.isoformat(),
            ).df
            for symbols in soi_chunks
        ]
    )
    if new_data.shape[0] == 0:
        min_date = first_trading_day
    else:
        historical_data = pd.concat([historical_data, new_data])
        min_date = historical_data.index.min()


# %%

# historical_data = pd.concat([api.get_barset(symbols = symbols[:5], timeframe = '15Min', limit = 100, start = START_DATE).df for symbols in soi_chunks])
historical_data = api.get_barset(
    symbols=soi[:5], timeframe="15Min", limit=50, after=START_DATE, until=END_DATE
).df

# %%
