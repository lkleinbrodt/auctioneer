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

class Client:
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
        
    def build_jwt(self):
        # copied from: https://docs.cloud.coinbase.com/advanced-trade-api/docs/rest-api-auth
            
        request_method = "GET"
        request_host   = "api.coinbase.com"
        request_path   = "/api/v3/brokerage/accounts"
        
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
            conn.request(method, path, body_encoded, headers)
            res = conn.getresponse()
            data = res.read()

            if res.status == 401:
                print("Error: Unauthorized. Please check your API key and secret.")
                return None

            response_data = json.loads(data.decode("utf-8"))
            if 'error_details' in response_data and response_data['error_details'] == 'missing required scopes':
                print(
                    "Error: Missing Required Scopes. Please update your API Keys to include more permissions.")
                return None

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
        return json.dumps(body).encode('utf-8') if body else b''
    
    def create_headers(self):
        return {
            'Content-Type': 'application/json',
            "Authorization": "Bearer " + self.build_jwt()
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
        path = self.add_query_params(path, params)
        body_encoded = self.prepare_body(body)
        headers = self.create_headers()
        return self.send_request(method, path, body_encoded, headers)