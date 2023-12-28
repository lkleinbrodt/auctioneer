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
        
    def get_candles(self, products, start_datetime, end_datetime, granularity, progress = False, candles_per_batch = 300):
        #TODO: improve logging and error handling, it's a mess right now
        assert isinstance(products, (list, str)), "products must be a list or a string"
        
        if isinstance(products, str):
            products = [products]
        
        if not isinstance(start_datetime, datetime.datetime):
            start_datetime = datetime.datetime(start_datetime)
        if not isinstance(end_datetime, datetime.datetime):
            end_datetime = datetime.datetime(end_datetime)
            
        assert end_datetime > start_datetime, "end_datetime must be greater than start_datetime"
        
        

        requested_seconds = (end_datetime - start_datetime).total_seconds()
        try:
            n_requested_candles = math.ceil(requested_seconds / GRANULARITY_TO_SECONDS[granularity])
        except KeyError:
            raise ValueError(f"granularity must be one of {list(GRANULARITY_TO_SECONDS.keys())}")

        
        #MAX_POINTS = 300 #max # of candles allowed by coinbase per request
        #TODO: daylight savings breaks this
        #ex. trying to pull minute data for Nov 5 2023, which is when daylight savings hits
        #pulling "5 hours" will actually try to request more than 5 hours worht of data. 
        #pretty weird bug, dont feel like fixing it rn
        
        if n_requested_candles > candles_per_batch:
            stime = start_datetime
            etime = stime
            results = []
            
            n_necessary_requests = math.ceil(n_requested_candles / candles_per_batch)
            counter = 0
            timer = datetime.datetime.now()
            while etime < end_datetime:
                
                etime = stime + datetime.timedelta(seconds = candles_per_batch * GRANULARITY_TO_SECONDS[granularity])
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
        
        l = []
        for product in products:
            response = self.client.getProductCandles(
                product_id = product,
                start = int(start_datetime.timestamp()),
                end = int(end_datetime.timestamp()),
                granularity=granularity
            )
            
            tmp = pd.DataFrame(response['candles']).assign(product_id=product)
            if tmp.empty:
                logger.warning(f"Empty dataframe returned for {product} from {start_datetime} to {end_datetime}")
                continue
            l.append(tmp)
        
        if len(l) == 0:
            raise ValueError(f"Unable to pull any data for {products} from {start_datetime} to {end_datetime}")
        out_df = pd.concat(l)
        return out_df.set_index(['product_id', 'start'])

            
    