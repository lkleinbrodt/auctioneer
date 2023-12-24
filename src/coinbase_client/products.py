from coinbase_client.client import Client

client = Client()

def listProducts(**kwargs):
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
    return client('GET', '/api/v3/brokerage/products', params=kwargs)


def getProduct(product_id):
    """
    Get information on a single product by product ID.

    This function uses the GET method to retrieve information about a single product from the Coinbase Advanced Trade API.

    :param product_id: The ID of the product to retrieve information for.
    :return: A dictionary containing the response from the server. This will include details about the product, such as the product ID, product type, and contract expiry type.
    """
    response = client(
        'GET', f'/api/v3/brokerage/products/{product_id}')

    # Check if there's an error in the response
    if 'error' in response and response['error'] == 'PERMISSION_DENIED':
        print(
            f"Error: {response['message']}. Details: {response['error_details']}")
        return None

    return response


def getProductCandles(product_id, start, end, granularity):
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
        'ONE_MINUTE', 'FIVE_MINUTE', 'FIFTEEN_MINUTE', 'THIRTY_MINUTE', 'ONE_HOUR', 'TWO_HOUR' 'SIX_HOUR', 'ONE_DAY'
    ]
    
    params = {
        'start': start,
        'end': end,
        'granularity': granularity
    }
    return client('GET', f'/api/v3/brokerage/products/{product_id}/candles', params=params)