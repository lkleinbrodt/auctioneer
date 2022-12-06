from auctioneer import *
logger = create_logger(__name__)

SYMBOLS_TO_PULL = None
START_DATE = '2021-01-01'

def pull_data():
    if SYMBOLS_TO_PULL is None:
        soi = get_crypto_symbols()
    else:
        soi = SYMBOLS_TO_PULL
    
    start_date = pd.to_datetime(START_DATE).tz_localize('US/Pacific')
    end_date = pd.to_datetime(datetime.now()).tz_localize('US/Pacific')

    dates = pd.date_range(start_date, end_date, freq = '7D')

    l = []

    for i in range(1, len(dates)):
        logger.info(f'Pulling from {dates[i-1]} to {dates[i]}')
        d = pull_crypto_prices(
            soi, 
            dates[i-1], 
            dates[i], 
            timeframe='minute'
        )
        if d.shape[0] == 0:
            logger.warning('WARNING: empty dataframe')
        logger.info(d.shape)
        l.append(d)

    df = pd.concat(l)

    if df.index[-1] < end_date:
        logger.info('Making additional pull to fill gap')

        final_pull = pull_crypto_prices(
            soi, 
            df.index[-1],
            end_date,
            timeframe = 'minute'
        )

        additional_times = final_pull.index.difference(df.index)
        if len(additional_times) > 0:
            df = pd.concat([df, final_pull.loc[additional_times]])

    return df

def main():
    data = pull_data()
    logger.info('Done pulling data')

    s3 = create_s3()
    try:
        data.to_csv('../data/minute_crypto_prices.csv')
    except Exception as e:
        logger.exception('unable to save to local')
    save_s3_csv(s3, data, 'minute_crypto_prices.csv')

if __name__ == '__main__':
    main()