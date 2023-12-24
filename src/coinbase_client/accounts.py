from coinbase_client.client import Client

client = Client()

def listAccounts(limit=49, cursor=None):
    """
    Get a list of authenticated accounts for the current user.

    This function uses the GET method to retrieve a list of authenticated accounts from the Coinbase Advanced Trade API.

    :param limit: A pagination limit with default of 49 and maximum of 250. If has_next is true, additional orders are available to be fetched with pagination and the cursor value in the response can be passed as cursor parameter in the subsequent request.
    :param cursor: Cursor used for pagination. When provided, the response returns responses after this cursor.
    :return: A dictionary containing the response from the server. A successful response will return a 200 status code. An unexpected error will return a default error response.
    """
    return client('GET', '/api/v3/brokerage/accounts', {'limit': limit, 'cursor': cursor})


def getAccount(account_uuid):
    """
    Get a list of information about an account, given an account UUID.

    This function uses the GET method to retrieve information about an account from the Coinbase Advanced Trade API.

    :param account_uuid: The account's UUID. Use listAccounts() to find account UUIDs.
    :return: A dictionary containing the response from the server. A successful response will return a 200 status code. An unexpected error will return a default error response.
    """
    return client('GET', f'/api/v3/brokerage/accounts/{account_uuid}')

def getTransactionsSummary(start_date, end_date, user_native_currency='USD', product_type='SPOT', contract_expiry_type='UNKNOWN_CONTRACT_EXPIRY_TYPE'):
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
    return client('GET', '/api/v3/brokerage/transaction_summary', params)