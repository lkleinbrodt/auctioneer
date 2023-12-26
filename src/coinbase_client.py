

from config import *

import jwt
from cryptography.hazmat.primitives import serialization
import time
import os
import json
import http.client
import json
import time
from urllib.parse import urlencode
from typing import Union, Dict
from urllib.error import HTTPError


logger = create_logger(__name__)

class Client:
    """
    A client class for interacting with the Coinbase Advanced Trade API.

    This class provides methods for authenticating requests, sending requests to the API, and retrieving information about accounts, transactions, products, and orders.
    
    This class should implement all of the base API calls for coinbase, but not more than that. Helper functions should go elsewhere

    :param api_key: The API key for authentication. If not provided, it will be retrieved from the COINBASE_API_KEY environment variable.
    :param secret_key: The secret key for authentication. If not provided, it will be retrieved from the COINBASE_SECRET_KEY environment variable.
    """

    def __init__(self, api_key=None, secret_key=None):
        
        if api_key is None:
            try:
                api_key = os.environ['COINBASE_API_KEY']
            except KeyError:
                raise Exception("No API key provided and no COINBASE_API_KEY found in environment variables")
        if secret_key is None:
            try:
                secret_key = os.environ['COINBASE_SECRET_KEY']
            except KeyError:
                raise Exception("No secret key provided and no COINBASE_SECRET_KEY found in environment variables")
        self.api_key = api_key
        self.secret_key = secret_key
        
    def build_jwt(self, request_path):
        # copied from: https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-auth
            
        request_method = "GET"
        request_host   = "api.coinbase.com"
        
        service   = "retail_rest_api_proxy"
        uri = f"{request_method} {request_host}{request_path}"
        
        private_key_bytes = self.secret_key.encode('utf-8')
        private_key = serialization.load_pem_private_key(private_key_bytes, password=None)
        jwt_payload = {
            'sub': self.api_key,
            'iss': "coinbase-cloud",
            'nbf': int(time.time()),
            'exp': int(time.time()) + 60,
            'aud': [service],
            'uri': uri,
        }
        
        jwt_token = jwt.encode(
            jwt_payload,
            private_key,
            algorithm='ES256',
            headers={'kid': self.api_key, 'nonce': str(int(time.time()))},
        )
        
        return jwt_token
    
    def send_request(self, method, path, body_encoded, headers):
        conn = http.client.HTTPSConnection("api.coinbase.com")
        try:
            conn.request(method, path, body_encoded, headers = headers)
            res = conn.getresponse()
            data = res.read()
            
            # if res.status != 200:
            #     raise HTTPError(
            #         url=path,
            #         code=res.status,
            #         msg=res.reason,
            #         hdrs=res.headers,
            #         fp=res.fp,
            #     )
            response_data = json.loads(data.decode("utf-8"))
            if 'error_details' in response_data:
                logger.exception([method, path, body_encoded, headers])
                raise ValueError(response_data['error_details'])

            return response_data
        except json.JSONDecodeError:
            print("Error: Unable to decode JSON response. Raw response data:", data)
            return None
        finally:
            conn.close()
            
    def add_query_params(self, path, params):
        if params:
            query_params = urlencode(params)
            path = f'{path}?{query_params}'
        return path
    
    def prepare_body(self, body):
        return json.dumps(body).encode('utf-8') if body else None
    
    def create_headers(self, request_path):
        return {
            'Content-Type': 'application/json',
            "Authorization": "Bearer " + self.build_jwt(request_path)
        }
        
    def __call__(self, method: str, path: str, body: Union[Dict, str] = '', params: Dict[str, str] = None) -> Dict:
        """
        Prepare and send an authenticated request to the Coinbase API.

        :param method: HTTP method (e.g., 'GET', 'POST')
        :param path: API endpoint path
        :param body: Request payload
        :param params: URL parameters
        :return: Response from the Coinbase API as a dictionary
        """
        full_path = self.add_query_params(path, params)
        body_encoded = self.prepare_body(body)
        headers = self.create_headers(path)

        return self.send_request(method, full_path, body_encoded, headers)
    
    def listAccounts(self, limit=49, cursor=None):
        """
        Get a list of authenticated accounts for the current user.

        This function uses the GET method to retrieve a list of authenticated accounts from the Coinbase Advanced Trade API.

        :param limit: A pagination limit with default of 49 and maximum of 250. If has_next is true, additional orders are available to be fetched with pagination and the cursor value in the response can be passed as cursor parameter in the subsequent request.
        :param cursor: Cursor used for pagination. When provided, the response returns responses after this cursor.
        :return: A dictionary containing the response from the server. A successful response will return a 200 status code. An unexpected error will return a default error response.
        """
        return self('GET', '/api/v3/brokerage/accounts', {'limit': limit, 'cursor': cursor})


    def getAccount(self, account_uuid):
        """
        Get a list of information about an account, given an account UUID.

        This function uses the GET method to retrieve information about an account from the Coinbase Advanced Trade API.

        :param account_uuid: The account's UUID. Use listAccounts() to find account UUIDs.
        :return: A dictionary containing the response from the server. A successful response will return a 200 status code. An unexpected error will return a default error response.
        """
        return self('GET', f'/api/v3/brokerage/accounts/{account_uuid}')

    def getTransactionsSummary(self, start_date, end_date, user_native_currency='USD', product_type='SPOT', contract_expiry_type='UNKNOWN_CONTRACT_EXPIRY_TYPE'):
        """
        Get a summary of transactions with fee tiers, total volume, and fees.

        This function uses the GET method to retrieve a summary of transactions from the Coinbase Advanced Trade API.

        :param start_date: The start date of the transactions to retrieve, in datetime format.
        :param end_date: The end date of the transactions to retrieve, in datetime format.
        :param user_native_currency: The user's native currency. Default is 'USD'.
        :param product_type: The type of product. Default is 'SPOT'.
        :param contract_expiry_type: Only orders matching this contract expiry type are returned. Only filters response if ProductType is set to 'FUTURE'. Default is 'UNKNOWN_CONTRACT_EXPIRY_TYPE'.
        :return: A dictionary containing the response from the server. This will include details about each transaction, such as fee tiers, total volume, and fees.
        """
        params = {
            'start_date': start_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'end_date': end_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'user_native_currency': user_native_currency,
            'product_type': product_type,
            'contract_expiry_type': contract_expiry_type
        }
        return self('GET', '/api/v3/brokerage/transaction_summary', params = params)
    
    def listProducts(self, **kwargs):
        """
        Get a list of the available currency pairs for trading.

        This function uses the GET method to retrieve a list of products from the Coinbase Advanced Trade API.

        :param limit: An optional integer describing how many products to return. Default is None.
        :param offset: An optional integer describing the number of products to offset before returning. Default is None.
        :param product_type: An optional string describing the type of products to return. Default is None.
        :param product_ids: An optional list of strings describing the product IDs to return. Default is None.
        :param contract_expiry_type: An optional string describing the contract expiry type. Default is 'UNKNOWN_CONTRACT_EXPIRY_TYPE'.
        :return: A dictionary containing the response from the server. This will include details about each product, such as the product ID, product type, and contract expiry type.
        """
        return self('GET', '/api/v3/brokerage/products', params=kwargs)


    def getProduct(self, product_id):
        """
        Get information on a single product by product ID.

        This function uses the GET method to retrieve information about a single product from the Coinbase Advanced Trade API.

        :param product_id: The ID of the product to retrieve information for.
        :return: A dictionary containing the response from the server. This will include details about the product, such as the product ID, product type, and contract expiry type.
        """
        response = self(
            'GET', f'/api/v3/brokerage/products/{product_id}')

        # Check if there's an error in the response
        if 'error' in response and response['error'] == 'PERMISSION_DENIED':
            print(
                f"Error: {response['message']}. Details: {response['error_details']}")
            return None

        return response


    def getProductCandles(self, product_id, start, end, granularity):
        """
        Get rates for a single product by product ID, grouped in buckets.

        This function uses the GET method to retrieve rates for a single product from the Coinbase Advanced Trade API.

        :param product_id: The trading pair.
        :param start: Timestamp for starting range of aggregations, in UNIX time.
        :param end: Timestamp for ending range of aggregations, in UNIX time.
        :param granularity: The time slice value for each candle.
        :return: A dictionary containing the response from the server. This will include details about each candle, such as the open, high, low, close, and volume.
        """
        
        assert granularity in [
            'UNKNOWN_GRANULARITY', #not sure if this one works?
            'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR', 'SIX_HOUR', 'ONE_DAY'
        ]
        
        params = {
            'start': start,
            'end': end,
            'granularity': granularity
        }

        return self('GET', f'/api/v3/brokerage/products/{product_id}/candles', params=params)
    
    def createOrder(self, client_order_id, product_id, side, order_type, order_configuration):
        """
        Create an order with the given parameters.

        :param client_order_id: A unique ID generated by the client for this order.
        :param product_id: The ID of the product to order.
        :param side: The side of the order (e.g., 'buy' or 'sell').
        :param order_type: The type of order (e.g., 'limit_limit_gtc').
        :param order_configuration: A dictionary containing order details such as price, size, and post_only.
        :return: A dictionary containing the response from the server.
        """
        payload = {
            "client_order_id": client_order_id,
            "product_id": product_id,
            "side": side,
            "order_configuration": {
                order_type: order_configuration
            }
        }
        # print("Payload being sent to server:", payload)  # For debugging
        return self('POST', '/api/v3/brokerage/orders', payload)

    def cancelOrders(self, order_ids):
        """
        Initiate cancel requests for one or more orders.

        This function uses the POST method to initiate cancel requests for one or more orders on the Coinbase Advanced Trade API.

        :param order_ids: A list of order IDs for which cancel requests should be initiated.
        :return: A dictionary containing the response from the server. A successful response will return a 200 status code. An unexpected error will return a default error response.
        """
        body = json.dumps({"order_ids": order_ids})
        return self('POST', '/api/v3/brokerage/orders/batch_cancel', body)


    def listOrders(self, **kwargs):
        """
        Retrieve a list of historical orders.

        This function uses the GET method to retrieve a list of historical orders from the Coinbase Advanced Trade API.
        The orders are returned in a batch format.

        :param kwargs: Optional parameters that can be passed to the API. These can include:
            'product_id': Optional string of the product ID. Defaults to null, or fetch for all products.
            'order_status': A list of order statuses.
            'limit': A pagination limit with no default set.
            'start_date': Start date to fetch orders from, inclusive.
            'end_date': An optional end date for the query window, exclusive.
            'user_native_currency': (Deprecated) String of the users native currency. Default is `USD`.
            'order_type': Type of orders to return. Default is to return all order types.
            'order_side': Only orders matching this side are returned. Default is to return all sides.
            'cursor': Cursor used for pagination.
            'product_type': Only orders matching this product type are returned. Default is to return all product types.
            'order_placement_source': Only orders matching this placement source are returned. Default is to return RETAIL_ADVANCED placement source.
            'contract_expiry_type': Only orders matching this contract expiry type are returned. Filter is only applied if ProductType is set to FUTURE in the request.
        :return: A dictionary containing the response from the server. This will include details about each order, such as the order ID, product ID, side, type, and status.
        """
        return self('GET', '/api/v3/brokerage/orders/historical/batch', params=kwargs)


    def listFills(self, **kwargs):
        """
        Retrieve a list of fills filtered by optional query parameters.

        This function uses the GET method to retrieve a list of fills from the Coinbase Advanced Trade API.
        The fills are returned in a batch format.

        :param kwargs: Optional parameters that can be passed to the API. These can include:
            'order_id': Optional string of the order ID.
            'product_id': Optional string of the product ID.
            'start_sequence_timestamp': Start date. Only fills with a trade time at or after this start date are returned.
            'end_sequence_timestamp': End date. Only fills with a trade time before this start date are returned.
            'limit': Maximum number of fills to return in response. Defaults to 100.
            'cursor': Cursor used for pagination. When provided, the response returns responses after this cursor.
        :return: A dictionary containing the response from the server. This will include details about each fill, such as the fill ID, product ID, side, type, and status.
        """
        return self('GET', '/api/v3/brokerage/orders/historical/fills', params=kwargs)


    def getOrder(self, order_id):
        """
        Retrieve a single order by order ID.

        This function uses the GET method to retrieve a single order from the Coinbase Advanced Trade API.

        :param order_id: The ID of the order to retrieve.
        :return: A dictionary containing the response from the server. This will include details about the order, such as the order ID, product ID, side, type, and status.
        """
        return self('GET', f'/api/v3/brokerage/orders/historical/{order_id}')