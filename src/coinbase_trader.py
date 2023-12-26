from config import *

from coinbase_client import Client
import pandas as pd
import math
import time
import datetime

logger = create_logger(__name__)

class Trader:
    
    def __init__(self, client: Client = None):
        if client is None:
            client = Client()
        self.client = client
        
    def get_candles(self, products, start_datetime, end_datetime, granularity, progress = False):
    
        assert isinstance(products, (list, str)), "products must be a list or a string"
        
        if isinstance(products, str):
            products = [products]
        
        if not isinstance(start_datetime, datetime.datetime):
            start_datetime = datetime.datetime(start_datetime)
        if not isinstance(end_datetime, datetime.datetime):
            end_datetime = datetime.datetime(end_datetime)
            
        assert end_datetime > start_datetime, "end_datetime must be greater than start_datetime"
        
        granularity_to_seconds = {
            'ONE_MINUTE': 60,
            'FIVE_MINUTE': 60 * 5,
            'FIFTEEN_MINUTE': 60 * 15,
            'THIRTY_MINUTE': 60 * 30,
            'ONE_HOUR': 60 * 60,
            'TWO_HOUR': 60 * 60 * 2,
            'SIX_HOUR': 60 * 60 * 6,
            'ONE_DAY': 60 * 60 * 24
        }

        requested_seconds = (end_datetime - start_datetime).total_seconds()
        try:
            n_requested_candles = math.ceil(requested_seconds / granularity_to_seconds[granularity])
        except KeyError:
            raise ValueError(f"granularity must be one of {list(granularity_to_seconds.keys())}")

        MAX_POINTS = 300 #max # of candles allowed by coinbase per request
        
        #TODO: daylight savings breaks this
        #ex. trying to pull minute data for Nov 5 2023, which is when daylight savings hits
        #pulling "5 hours" will actually try to request more than 5 hours worht of data. 
        #pretty weird bug, dont feel like fixing it rn
        
        if n_requested_candles > MAX_POINTS:
            stime = start_datetime
            etime = stime
            results = []
            
            n_necessary_requests = math.ceil(n_requested_candles / MAX_POINTS)
            counter = 0
            timer = datetime.datetime.now()
            while etime < end_datetime:
                
                etime = stime + datetime.timedelta(seconds = MAX_POINTS * granularity_to_seconds[granularity])
                etime = min([etime, end_datetime])

                tmp_results = self.get_candles(products, stime, etime, granularity)
                results.append(tmp_results)
                stime = etime
                counter += 1
                if progress:
                    elapsed_seconds = (datetime.datetime.now() - timer).total_seconds()
                    eta = (n_necessary_requests - counter) * (elapsed_seconds / counter) / 60
                    logger.info(f"Request {counter}/{n_necessary_requests} complete. ETA: {eta:.2f} minutes")
                
            return pd.concat(results)
        
        output = {}
        for product in products:
            output[product] = self.client.getProductCandles(
                product_id = product,
                start = int(start_datetime.timestamp()),
                end = int(end_datetime.timestamp()),
                granularity=granularity
            )
            time.sleep(.025) # to slightly avoid rate limiting, coinbase throttles to 30 per second
        
        if len(output) == 1:
            return pd.DataFrame(output[products[0]]['candles']).set_index('start')
        else:
            l = []
            for product in products:
                try:
                    tmp = pd.DataFrame(output[product]['candles']).assign(product_id=product)
                    l.append(tmp)
                except KeyError:
                    logger.exception(f"Failed to pull data for {product} from {start_datetime} to {end_datetime}")
                    logger.error('Output:')
                    logger.exception(output[product])
            
            return pd.concat(l).set_index(['product_id', 'start'])
            
    