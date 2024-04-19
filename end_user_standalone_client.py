import asyncio
import json
import httpx
import os
import logging
import shutil
import queue
import zstandard as zstd
import base64
import pytz
import hashlib
import urllib.parse as urlparse
import re
import random
import time
import traceback
import uuid
from decimal import Decimal
import decimal
import pandas as pd
from datetime import datetime, date
import datetime as dt
from typing import List, Dict, Union, Any, Optional, Tuple
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from httpx import AsyncClient, Limits, Timeout
from decouple import Config as DecoupleConfig, RepositoryEnv
from pydantic import field_validator
from sqlmodel import SQLModel, Field, Column, JSON, UUID

# Note: you must have `minrelaytxfee=0.00001` in your pastel.conf to allow "dust" transactions for the inference request confirmation transactions to work!

logger = logging.getLogger("pastel_supernode_messaging_client")

config = DecoupleConfig(RepositoryEnv('.env'))
MESSAGING_TIMEOUT_IN_SECONDS = config.get("MESSAGING_TIMEOUT_IN_SECONDS", default=60, cast=int)
MY_LOCAL_PASTELID = config.get("MY_LOCAL_PASTELID", cast=str)
# MY_PASTELID_PASSPHRASE = config.get("MY_PASTELID_PASSPHRASE", cast=str)
MY_PASTELID_PASSPHRASE = "5QcX9nX67buxyeC"
MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING = config.get("MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING", default=0.1, cast=float)
MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS = config.get("MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS", default=1, cast=int)
TARGET_VALUE_PER_CREDIT_IN_USD = config.get("TARGET_VALUE_PER_CREDIT_IN_USD", default=0.1, cast=float)
TARGET_PROFIT_MARGIN = config.get("TARGET_PROFIT_MARGIN", default=0.1, cast=float)
MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING = config.get("MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING", default=0.1, cast=float)
MAXIMUM_PER_CREDIT_PRICE_IN_PSL_FOR_CLIENT = config.get("MAXIMUM_PER_CREDIT_PRICE_IN_PSL_FOR_CLIENT", default=100.0, cast=float)

def setup_logger():
    if logger.handlers:
        return logger
    old_logs_dir = 'old_logs'
    if not os.path.exists(old_logs_dir):
        os.makedirs(old_logs_dir)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    log_file_path = 'pastel_supernode_messaging_client.log'
    log_queue = queue.Queue(-1)  # Create a queue for the handlers
    fh = RotatingFileHandler(log_file_path, maxBytes=10*1024*1024, backupCount=5)
    fh.setFormatter(formatter)
    def namer(default_log_name):  # Function to move rotated logs to the old_logs directory
        return os.path.join(old_logs_dir, os.path.basename(default_log_name))
    def rotator(source, dest):
        shutil.move(source, dest)
    fh.namer = namer
    fh.rotator = rotator
    sh = logging.StreamHandler()  # Stream handler
    sh.setFormatter(formatter)
    queue_handler = QueueHandler(log_queue)  # Create QueueHandler
    queue_handler.setFormatter(formatter)
    logger.addHandler(queue_handler)
    listener = QueueListener(log_queue, fh, sh)  # Create QueueListener with real handlers
    listener.start()
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)  # Configure SQLalchemy logging
    return logger

logger = setup_logger()

def get_local_rpc_settings_func(directory_with_pastel_conf=os.path.expanduser("~/.pastel/")):
    with open(os.path.join(directory_with_pastel_conf, "pastel.conf"), 'r') as f:
        lines = f.readlines()
    other_flags = {}
    rpchost = '127.0.0.1'
    rpcport = '19932'
    for line in lines:
        if line.startswith('rpcport'):
            value = line.split('=')[1]
            rpcport = value.strip()
        elif line.startswith('rpcuser'):
            value = line.split('=')[1]
            rpcuser = value.strip()
        elif line.startswith('rpcpassword'):
            value = line.split('=')[1]
            rpcpassword = value.strip()
        elif line.startswith('rpchost'):
            pass
        elif line == '\n':
            pass
        else:
            current_flag = line.strip().split('=')[0].strip()
            current_value = line.strip().split('=')[1].strip()
            other_flags[current_flag] = current_value
    return rpchost, rpcport, rpcuser, rpcpassword, other_flags

def write_rpc_settings_to_env_file_func(rpc_host, rpc_port, rpc_user, rpc_password, other_flags):
    with open('.env', 'w') as f:
        f.write(f"RPC_HOST={rpc_host}\n")
        f.write(f"RPC_PORT={rpc_port}\n")
        f.write(f"RPC_USER={rpc_user}\n")
        f.write(f"RPC_PASSWORD={rpc_password}\n")
        for current_flag in other_flags:
            current_value = other_flags[current_flag]
            try:
                f.write(f"{current_flag}={current_value}\n")
            except Exception as e:
                logger.error(f"Error writing to .env file: {e}")
                pass
    return

class JSONRPCException(Exception):
    def __init__(self, rpc_error):
        parent_args = []
        try:
            parent_args.append(rpc_error['message'])
        except Exception as e:
            logger.error(f"Error occurred in JSONRPCException: {e}")
            pass
        Exception.__init__(self, *parent_args)
        self.error = rpc_error
        self.code = rpc_error['code'] if 'code' in rpc_error else None
        self.message = rpc_error['message'] if 'message' in rpc_error else None

    def __str__(self):
        return '%d: %s' % (self.code, self.message)

    def __repr__(self):
        return '<%s \'%s\'>' % (self.__class__.__name__, self)

def EncodeDecimal(o):
    if isinstance(o, Decimal):
        return float(round(o, 8))
    raise TypeError(repr(o) + " is not JSON serializable")
    
class AsyncAuthServiceProxy:
    max_concurrent_requests = 5000
    _semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
    def __init__(self, service_url, service_name=None, reconnect_timeout=15, reconnect_amount=2, request_timeout=20):
        self.service_url = service_url
        self.service_name = service_name
        self.url = urlparse.urlparse(service_url)        
        self.client = AsyncClient(timeout=Timeout(request_timeout), limits=Limits(max_connections=200, max_keepalive_connections=10))
        self.id_count = 0
        user = self.url.username
        password = self.url.password
        authpair = f"{user}:{password}".encode('utf-8')
        self.auth_header = b'Basic ' + base64.b64encode(authpair)
        self.reconnect_timeout = reconnect_timeout
        self.reconnect_amount = reconnect_amount
        self.request_timeout = request_timeout

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        if self.service_name is not None:
            name = f"{self.service_name}.{name}"
        return AsyncAuthServiceProxy(self.service_url, name)

    async def __call__(self, *args):
        async with self._semaphore: # Acquire a semaphore
            self.id_count += 1
            postdata = json.dumps({
                'version': '1.1',
                'method': self.service_name,
                'params': args,
                'id': self.id_count
            }, default=EncodeDecimal)
            headers = {
                'Host': self.url.hostname,
                'User-Agent': "AuthServiceProxy/0.1",
                'Authorization': self.auth_header,
                'Content-type': 'application/json'
            }
            for i in range(self.reconnect_amount):
                try:
                    if i > 0:
                        logger.warning(f"Reconnect try #{i+1}")
                        sleep_time = self.reconnect_timeout * (2 ** i)
                        logger.info(f"Waiting for {sleep_time} seconds before retrying.")
                        await asyncio.sleep(sleep_time)
                    response = await self.client.post(
                        self.service_url, headers=headers, data=postdata)
                    break
                except Exception as e:
                    logger.error(f"Error occurred in __call__: {e}")
                    err_msg = f"Failed to connect to {self.url.hostname}:{self.url.port}"
                    rtm = self.reconnect_timeout
                    if rtm:
                        err_msg += f". Waiting {rtm} seconds."
                    logger.exception(err_msg)
            else:
                logger.error("Reconnect tries exceeded.")
                return
            response_json = response.json()
            if response_json['error'] is not None:
                raise JSONRPCException(response_json['error'])
            elif 'result' not in response_json:
                raise JSONRPCException({
                    'code': -343, 'message': 'missing JSON-RPC result'})
            else:
                return response_json['result']
        
async def check_masternode_top_func():
    global rpc_connection
    masternode_top_command_output = await rpc_connection.masternode('top')
    return masternode_top_command_output

async def check_supernode_list_func():
    global rpc_connection
    masternode_list_full_command_output = await rpc_connection.masternodelist('full')
    masternode_list_rank_command_output = await rpc_connection.masternodelist('rank')
    masternode_list_pubkey_command_output = await rpc_connection.masternodelist('pubkey')
    masternode_list_extra_command_output = await rpc_connection.masternodelist('extra')
    masternode_list_full_df = pd.DataFrame([masternode_list_full_command_output[x].split() for x in masternode_list_full_command_output])
    masternode_list_full_df['txid_vout'] = [x for x in masternode_list_full_command_output]
    masternode_list_full_df.columns = ['supernode_status', 'protocol_version', 'supernode_psl_address', 'lastseentime', 'activeseconds', 'lastpaidtime', 'lastpaidblock', 'ipaddress:port', 'txid_vout']
    masternode_list_full_df.index = masternode_list_full_df['txid_vout']
    masternode_list_full_df.drop(columns=['txid_vout'], inplace=True)
    for current_row in masternode_list_full_df.iterrows():
            current_row_df = pd.DataFrame(current_row[1]).T
            current_txid_vout = current_row_df.index[0]
            current_rank = masternode_list_rank_command_output[current_txid_vout]
            current_pubkey = masternode_list_pubkey_command_output[current_txid_vout]
            current_extra = masternode_list_extra_command_output[current_txid_vout]
            masternode_list_full_df.loc[current_row[0], 'rank'] = current_rank
            masternode_list_full_df.loc[current_row[0], 'pubkey'] = current_pubkey
            masternode_list_full_df.loc[current_row[0], 'extAddress'] = current_extra['extAddress']
            masternode_list_full_df.loc[current_row[0], 'extP2P'] = current_extra['extP2P']
            masternode_list_full_df.loc[current_row[0], 'extKey'] = current_extra['extKey']
    masternode_list_full_df['lastseentime'] = pd.to_numeric(masternode_list_full_df['lastseentime'], downcast='integer')            
    masternode_list_full_df['lastpaidtime'] = pd.to_numeric(masternode_list_full_df['lastpaidtime'], downcast='integer')            
    masternode_list_full_df['lastseentime'] = pd.to_datetime(masternode_list_full_df['lastseentime'], unit='s')
    masternode_list_full_df['lastpaidtime'] = pd.to_datetime(masternode_list_full_df['lastpaidtime'], unit='s')
    masternode_list_full_df['activeseconds'] = masternode_list_full_df['activeseconds'].astype(int)
    masternode_list_full_df['lastpaidblock'] = masternode_list_full_df['lastpaidblock'].astype(int)
    masternode_list_full_df['activedays'] = [float(x)/86400.0 for x in masternode_list_full_df['activeseconds'].values.tolist()]
    masternode_list_full_df['rank'] = masternode_list_full_df['rank'].astype(int)
    masternode_list_full_df = masternode_list_full_df[masternode_list_full_df['supernode_status'].isin(['ENABLED', 'PRE_ENABLED'])]
    masternode_list_full_df = masternode_list_full_df[masternode_list_full_df['ipaddress:port'] != '154.38.164.75:29933'] #TODO: Remove this
    masternode_list_full_df__json = masternode_list_full_df.to_json(orient='index')
    return masternode_list_full_df, masternode_list_full_df__json

def get_top_supernode_url(supernode_list_df):
    if not supernode_list_df.empty:
        supernode_list_df = supernode_list_df[supernode_list_df['supernode_status']=='ENABLED'] 
        top_supernode = supernode_list_df.loc[supernode_list_df['rank'] == supernode_list_df['rank'].min()]
        if not top_supernode.empty:
            ipaddress_port = top_supernode['ipaddress:port'].values[0]
            ipaddress = ipaddress_port.split(':')[0]
            supernode_url = f"http://{ipaddress}:7123"
            return supernode_url
    return None

async def get_current_pastel_block_height_func():
    global rpc_connection
    best_block_hash = await rpc_connection.getbestblockhash()
    best_block_details = await rpc_connection.getblock(best_block_hash)
    curent_block_height = best_block_details['height']
    return curent_block_height

async def get_best_block_hash_and_merkle_root_func():
    global rpc_connection
    best_block_height = await get_current_pastel_block_height_func()
    best_block_hash = await rpc_connection.getblockhash(best_block_height)
    best_block_details = await rpc_connection.getblock(best_block_hash)
    best_block_merkle_root = best_block_details['merkleroot']
    return best_block_hash, best_block_merkle_root, best_block_height

def compute_sha3_256_hexdigest(input_str):
    """Compute the SHA3-256 hash of the input string and return the hexadecimal digest."""
    return hashlib.sha3_256(input_str.encode()).hexdigest()

def get_sha256_hash_of_input_data_func(input_data_or_string):
    if isinstance(input_data_or_string, str):
        input_data_or_string = input_data_or_string.encode('utf-8')
    sha256_hash_of_input_data = hashlib.sha3_256(input_data_or_string).hexdigest()
    return sha256_hash_of_input_data

async def extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance: SQLModel) -> str:
    response_fields = {}
    last_hash_field_name = None
    last_signature_field_name = None
    # Find the last hash field and the last signature field
    for field_name in model_instance.__fields__.keys():
        if field_name.startswith("sha3_256_hash_of"):
            last_hash_field_name = field_name
        elif "_signature_on_" in field_name:
            last_signature_field_name = field_name
    # Iterate over the model fields and exclude the last hash, last signature, 'id', and fields containing '_sa_instance_state'
    for field_name, field_value in model_instance.__dict__.items():
        if field_name in [last_hash_field_name, last_signature_field_name, 'id'] or '_sa_instance_state' in field_name:
            continue
        if field_value is not None:
            if isinstance(field_value, (datetime, date)):
                response_fields[field_name] = field_value.isoformat()
            elif isinstance(field_value, (list, dict)):
                response_fields[field_name] = json.dumps(field_value, sort_keys=True)
            elif isinstance(field_value, decimal.Decimal):
                response_fields[field_name] = str(field_value)
            else:
                response_fields[field_name] = field_value
    # Sort the response fields by field name to ensure a consistent order
    sorted_response_fields = dict(sorted(response_fields.items()))
    # Convert the sorted response fields to a JSON string
    return json.dumps(sorted_response_fields, sort_keys=True)

async def compute_sha3_256_hash_of_sqlmodel_response_fields(model_instance: SQLModel) -> str:
    response_fields_json = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance)
    sha256_hash_of_response_fields = get_sha256_hash_of_input_data_func(response_fields_json)
    return sha256_hash_of_response_fields

def compare_datetimes(datetime_input1, datetime_input2):
    # Check if the inputs are datetime objects, otherwise parse them
    if not isinstance(datetime_input1, datetime):
        datetime_input1 = pd.to_datetime(datetime_input1)
    if not isinstance(datetime_input2, datetime):
        datetime_input2 = pd.to_datetime(datetime_input2)
    # Ensure both datetime objects are timezone-aware
    if datetime_input1.tzinfo is None:
        datetime_input1 = datetime_input1.replace(tzinfo=pytz.UTC)
    if datetime_input2.tzinfo is None:
        datetime_input2 = datetime_input2.replace(tzinfo=pytz.UTC)
    # Calculate the difference in seconds
    difference_in_seconds = abs((datetime_input2 - datetime_input1).total_seconds())
    # Check if the difference is within the acceptable range
    datetimes_are_close_enough_to_consider_them_matching = (
        difference_in_seconds <= MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING
    )
    return difference_in_seconds, datetimes_are_close_enough_to_consider_them_matching

async def verify_message_with_pastelid_func(pastelid, message_to_verify, pastelid_signature_on_message) -> str:
    global rpc_connection
    verification_result = await rpc_connection.pastelid('verify', message_to_verify, pastelid_signature_on_message, pastelid, 'ed448')
    return verification_result['verification']

async def validate_credit_pack_ticket_message_data_func(model_instance: SQLModel):
    validation_errors = []
    # Validate timestamp fields
    for field_name, field_value in model_instance.__dict__.items():
        if field_name.endswith("_timestamp_utc_iso_string"):
            try:
                pd.to_datetime(field_value)
            except ValueError:
                validation_errors.append(f"Invalid timestamp format for field {field_name}")
            # Check if the timestamp is within an acceptable range of the current time
            current_timestamp = pd.to_datetime(datetime.utcnow().replace(tzinfo=pytz.UTC))
            timestamp_diff, timestamps_match = compare_datetimes(field_value, current_timestamp)
            if not timestamps_match:
                validation_errors.append(f"Timestamp in field {field_name} is too far from the current time")
    # Validate pastel block height fields
    best_block_hash, best_block_merkle_root, best_block_height = await get_best_block_hash_and_merkle_root_func()
    for field_name, field_value in model_instance.__dict__.items():
        if field_name.endswith("_pastel_block_height"):
            if abs(field_value - best_block_height) > MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS:
                validation_errors.append(f"Pastel block height in field {field_name} does not match the current block height; difference is {abs(field_value - best_block_height)} blocks (local: {field_value}, remote: {best_block_height})")
    # Validate hash fields
    expected_hash = await compute_sha3_256_hash_of_sqlmodel_response_fields(model_instance)
    hash_field_name = None
    for field_name in model_instance.__fields__:
        if "sha3_256_hash_of_" in field_name and field_name.endswith("_fields"):
            hash_field_name = field_name
            break
    if hash_field_name:
        actual_hash = getattr(model_instance, hash_field_name)
        if actual_hash != expected_hash:
            validation_errors.append(f"SHA3-256 hash in field {hash_field_name} does not match the computed hash of the response fields")
    # Validate pastelid signature fields
    last_signature_field_name = None
    last_hash_field_name = None
    for field_name in model_instance.__fields__:
        if "_pastelid" in field_name:
            first_pastelid = field_name
            break
    for field_name in model_instance.__fields__:
        if "_signature_on_" in field_name:
            last_signature_field_name = field_name
        elif "sha3_256_hash_of_" in field_name and field_name.endswith("_fields"):
            last_hash_field_name = field_name
    if last_signature_field_name and last_hash_field_name:
        signature_field_name = last_signature_field_name
        hash_field_name = last_hash_field_name
        if first_pastelid == last_signature_field_name:
            first_pastelid = "NA"
        if hasattr(model_instance, first_pastelid) or first_pastelid == "NA":
            if first_pastelid == "NA":
                pastelid_and_signature_combined_field_name = last_signature_field_name
                pastelid_and_signature_combined_field_json = getattr(model_instance, pastelid_and_signature_combined_field_name)
                pastelid_and_signature_combined_field_dict = json.loads(pastelid_and_signature_combined_field_json)
                pastelid_and_signature_combined_field_dict_keys = pastelid_and_signature_combined_field_dict.keys()
                for current_key in pastelid_and_signature_combined_field_dict_keys:
                    if "pastelid" in current_key:
                        pastelid = pastelid_and_signature_combined_field_dict[current_key]
                    if "signature" in current_key:
                        signature = pastelid_and_signature_combined_field_dict[current_key]
                message_to_verify = getattr(model_instance, hash_field_name)
            else:
                pastelid = getattr(model_instance, first_pastelid)
                message_to_verify = getattr(model_instance, hash_field_name)
                signature = getattr(model_instance, signature_field_name)
            verification_result = await verify_message_with_pastelid_func(pastelid, message_to_verify, signature)
            if verification_result != 'OK':
                validation_errors.append(f"Pastelid signature in field {signature_field_name} failed verification")
        else:
            validation_errors.append(f"Corresponding pastelid field {first_pastelid} not found for signature field {signature_field_name}")
    return validation_errors

async def calculate_xor_distance(pastelid1: str, pastelid2: str) -> int:
    hash1 = hashlib.sha3_256(pastelid1.encode()).hexdigest()
    hash2 = hashlib.sha3_256(pastelid2.encode()).hexdigest()
    xor_result = int(hash1, 16) ^ int(hash2, 16)
    return xor_result

def check_if_pastelid_is_valid_func(input_string: str) -> bool:
    # Define the regex pattern to match the conditions:
    # Starts with 'jX'; Followed by characters that are only alphanumeric and are shown in the example;
    pattern = r'^jX[A-Za-z0-9]{84}$'
    if re.match(pattern, input_string):
        return True
    else:
        return False

async def get_supernode_url_from_pastelid_func(pastelid: str, supernode_list_df: pd.DataFrame) -> str:
    is_valid_pastelid = check_if_pastelid_is_valid_func(pastelid)
    if not is_valid_pastelid:
        raise ValueError(f"Invalid PastelID: {pastelid}")
    supernode_row = supernode_list_df[supernode_list_df['extKey'] == pastelid]
    if not supernode_row.empty:
        supernode_ipaddress_port = supernode_row['ipaddress:port'].values[0]
        ipaddress = supernode_ipaddress_port.split(':')[0]
        supernode_url = f"http://{ipaddress}:7123"
        return supernode_url
    else:
        raise ValueError(f"Supernode with PastelID {pastelid} not found in the supernode list")

async def get_closest_supernode_to_pastelid_url(input_pastelid: str, supernode_list_df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    if not supernode_list_df.empty:
        list_of_supernode_pastelids = supernode_list_df['extKey'].tolist()
        closest_supernode_pastelid = await get_closest_supernode_pastelid_from_list(input_pastelid, list_of_supernode_pastelids)
        supernode_url = await get_supernode_url_from_pastelid_func(closest_supernode_pastelid, supernode_list_df)
        return supernode_url, closest_supernode_pastelid
    return None, None

async def get_n_closest_supernodes_to_pastelid_urls(n: int, input_pastelid: str, supernode_list_df: pd.DataFrame) -> List[Tuple[str, str]]:
    if not supernode_list_df.empty:
        list_of_supernode_pastelids = supernode_list_df['extKey'].tolist()
        xor_distances = [(supernode_pastelid, await calculate_xor_distance(input_pastelid, supernode_pastelid)) for supernode_pastelid in list_of_supernode_pastelids]
        sorted_xor_distances = sorted(xor_distances, key=lambda x: x[1])
        closest_supernodes = sorted_xor_distances[:n]
        supernode_urls_and_pastelids = [(await get_supernode_url_from_pastelid_func(pastelid, supernode_list_df), pastelid) for pastelid, _ in closest_supernodes]
        return supernode_urls_and_pastelids
    return []

async def get_closest_supernode_pastelid_from_list(local_pastelid: str, supernode_pastelids: List[str]) -> str:
    xor_distances = [(supernode_pastelid, await calculate_xor_distance(local_pastelid, supernode_pastelid)) for supernode_pastelid in supernode_pastelids]
    closest_supernode = min(xor_distances, key=lambda x: x[1])
    return closest_supernode[0]

async def fetch_current_psl_market_price():
    async def check_prices():
        async with httpx.AsyncClient() as client:
            try:
                # Fetch data from CoinMarketCap
                response_cmc = await client.get("https://coinmarketcap.com/currencies/pastel/")
                price_cmc = float(re.search(r'price today is \$([0-9\.]+) USD', response_cmc.text).group(1))
            except (httpx.RequestError, AttributeError, ValueError):
                price_cmc = None
            try:
                # Fetch data from CoinGecko
                response_cg = await client.get("https://api.coingecko.com/api/v3/simple/price?ids=pastel&vs_currencies=usd")
                if response_cg.status_code == 200:
                    data = response_cg.json()
                    price_cg = data.get("pastel", {}).get("usd")
                else:
                    price_cg = None
            except (httpx.RequestError, AttributeError, ValueError):
                price_cg = None
        return price_cmc, price_cg
    price_cmc, price_cg = await check_prices()
    if price_cmc is None and price_cg is None:
        #Sleep for a couple seconds and try again:
        await asyncio.sleep(2)
        price_cmc, price_cg = await check_prices()
    # Calculate the average price
    prices = [price for price in [price_cmc, price_cg] if price is not None]
    if not prices:
        raise ValueError("Could not retrieve PSL price from any source.")
    average_price = sum(prices) / len(prices)
    # Validate the price
    if not 0.0000001 < average_price < 0.02:
        raise ValueError(f"Invalid PSL price: {average_price}")
    logger.info(f"The current Average PSL price is: ${average_price:.8f} based on {len(prices)} sources")
    return average_price

async def estimated_market_price_of_inference_credits_in_psl_terms() -> float:
    try:
        psl_price_usd = await fetch_current_psl_market_price()
        target_value_per_credit_usd = TARGET_VALUE_PER_CREDIT_IN_USD
        target_profit_margin = TARGET_PROFIT_MARGIN
        # Calculate the cost per credit in USD, considering the profit margin
        cost_per_credit_usd = target_value_per_credit_usd / (1 - target_profit_margin)
        # Convert the cost per credit from USD to PSL
        cost_per_credit_psl = cost_per_credit_usd / psl_price_usd
        logger.info(f"Estimated market price of 1.0 inference credit: {cost_per_credit_psl:.4f} PSL")
        return cost_per_credit_psl
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error calculating estimated market price of inference credits: {str(e)}")
        raise

def parse_and_format(value):
    try:
        # Check if the JSON string is already formatted
        if isinstance(value, str) and "\n" in value:
            return value
        # Unescape the JSON string if it's a string
        if isinstance(value, str):
            unescaped_value = json.loads(json.dumps(value))
            parsed_value = json.loads(unescaped_value)
        else:
            parsed_value = value
        return json.dumps(parsed_value, indent=4)
    except (json.JSONDecodeError, TypeError):
        return value

def pretty_json_func(data):
    if isinstance(data, SQLModel):
        data = data.dict()  # Assuming 'dict()' method provides serialization of SQLModel to dictionary
    if isinstance(data, dict):
        formatted_data = {}
        for key, value in data.items():
            if isinstance(value, uuid.UUID):  # Convert UUIDs to string
                formatted_data[key] = str(value)
            elif isinstance(value, dict):  # Recursively handle dictionary values
                formatted_data[key] = pretty_json_func(value)
            elif key.endswith("_json"):  # Handle keys that end with '_json'
                formatted_data[key] = parse_and_format(value)
            else:  # Handle other types of values
                formatted_data[key] = value
        return json.dumps(formatted_data, indent=4)
    elif isinstance(data, str):  # Handle string type data separately
        return parse_and_format(data)
    else:
        return data  # Return data as is if not a dictionary or string
    
def log_action_with_payload(action_string, payload_name, json_payload):
    logger.info(f"Now {action_string} {payload_name} with payload:\n{pretty_json_func(json_payload)}")

def transform_credit_pack_purchase_request_response(result: dict) -> dict:
    transformed_result = result.copy()
    fields_to_convert = [
        "list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms",
        "list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash",
        "list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json"
    ]
    for field in fields_to_convert:
        if field in transformed_result:
            transformed_result[field] = json.dumps(transformed_result[field])
    return transformed_result

async def send_to_address_func(address, amount, comment="", comment_to="", subtract_fee_from_amount=False):
    """
    Send an amount to a given Pastel address.

    Args:
        address (str): The Pastel address to send to.
        amount (float): The amount in PSL to send.
        comment (str, optional): A comment used to store what the transaction is for.
                                    This is not part of the transaction, just kept in your wallet.
                                    Defaults to an empty string.
        comment_to (str, optional): A comment to store the name of the person or organization
                                    to which you're sending the transaction. This is not part of
                                    the transaction, just kept in your wallet. Defaults to an empty string.
        subtract_fee_from_amount (bool, optional): Whether to deduct the fee from the amount being sent.
                                                    If True, the recipient will receive less Pastel than you enter
                                                    in the amount field. Defaults to False.

    Returns:
        str: The transaction ID if successful, None otherwise.

    Example:
        send_to_address_func("PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n", 0.1, "donation", "seans outpost", True)
    """
    global rpc_connection
    try:
        result = await rpc_connection.sendtoaddress(address, amount, comment, comment_to, subtract_fee_from_amount)
        return result
    except Exception as e:
        logger.error(f"Error in send_to_address_func: {e}")
        return None

async def send_many_func(amounts, min_conf=1, comment="", change_address=""):
    """
    Send multiple amounts to multiple recipients.

    Args:
        amounts (dict): A dictionary representing the amounts to send.
                        Each key is the Pastel address, and the corresponding value is the amount in PSL to send.
        min_conf (int, optional): The minimum number of confirmations required for the funds to be used. Defaults to 1.
        comment (str, optional): A comment to include with the transaction. Defaults to an empty string.
        change_address (str, optional): The Pastel address to receive the change from the transaction. Defaults to an empty string.

    Returns:
        str: The transaction ID if successful, None otherwise.

    Example:
        amounts = {
            "PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n": 0.01,
            "PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n": 0.02
        }
        send_many_func(amounts, min_conf=6, comment="testing", change_address="PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n")
    """
    global rpc_connection
    try:
        # Set the 'fromaccount' parameter to an empty string
        from_account = ""
        # Call the 'sendmany' RPC method
        result = await rpc_connection.sendmany(from_account, amounts, min_conf, comment, [""], change_address)
        return result
    except Exception as e:
        logger.error(f"Error in send_many_func: {e}")
        return None

async def z_send_many_with_change_to_sender_func(from_address, amounts, min_conf=1, fee=0.1):
    """
    Send multiple amounts from a given address to multiple recipients.

    Args:
        from_address (str): The taddr or zaddr to send the funds from.
        amounts (list): A list of dictionaries representing the amounts to send.
                        Each dictionary should have the following keys:
                        - "address" (str): The taddr or zaddr to send funds to.
                        - "amount" (float): The amount in PSL to send.
                        - "memo" (str, optional): If the address is a zaddr, raw data represented in hexadecimal string format.
        min_conf (int, optional): The minimum number of confirmations required for the funds to be used. Defaults to 1.
        fee (float, optional): The fee amount to attach to the transaction. Defaults to 0.1.

    Returns:
        str: The operation ID if successful, None otherwise.

    Example:
        amounts = [
            {"address": "PzSSk8QJFqjo133DoFZvn9wwcCxt5RYeeLFJZRgws6xgJ3LroqRgXKNkhkG3ENmC8oe82UTr3PHcQB9mw7DSLXhyP6atQQ5", "amount": 5.0},
            {"address": "PzXFZjHx6KzqNpAaMewvrUj8x1fvj7UZLFZYEuN8jJJuQhSfPYaVoAF1qrFSh3q2zUmCg7QkfQr4nAVrdovwKA4KDwPp5g", "amount": 10.0, "memo": "0xabcd"}
        ]
        z_send_many_with_change_to_sender_func("PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n", amounts, min_conf=2, fee=0.05)
    """
    global rpc_connection
    try:
        result = await rpc_connection.z_sendmanywithchangetosender(from_address, amounts, min_conf, fee)
        return result
    except Exception as e:
        logger.error(f"Error in z_send_many_with_change_to_sender_func: {e}")
        return None

async def z_get_operation_status_func(operation_ids=None):
    """
    Get the status of one or more operations.

    Args:
        operation_ids (list, optional): A list of operation IDs to query the status for.
                                        If not provided, all known operations will be examined.

    Returns:
        list: A list of JSON objects containing the operation status and any associated result or error data.

    Example:
        operation_ids = ["opid-1234", "opid-5678"]
        z_get_operation_status_func(operation_ids)
    """
    global rpc_connection
    try:
        if operation_ids is None:
            operation_ids = []
        result = await rpc_connection.z_getoperationstatus(operation_ids)
        return result
    except Exception as e:
        logger.error(f"Error in z_get_operation_status_func: {e}")
        return None

async def check_psl_address_balance_alternative_func(address_to_check):
    global rpc_connection
    address_amounts_dict = await rpc_connection.listaddressamounts()
    # Convert the dictionary into a list of dictionaries, each representing a row
    data = [{'address': address, 'amount': amount} for address, amount in address_amounts_dict.items()]
    # Create the DataFrame from the list of dictionaries
    address_amounts_df = pd.DataFrame(data)
    # Filter the DataFrame for the specified address
    address_amounts_df_filtered = address_amounts_df[address_amounts_df['address'] == address_to_check]
    # Calculate the sum of the 'amount' column for the filtered DataFrame
    balance_at_address = address_amounts_df_filtered['amount'].sum()
    return balance_at_address
        
async def create_and_fund_new_psl_credit_tracking_address(amount_of_psl_to_fund_address_with: float):
    global rpc_connection
    new_credit_tracking_address = await rpc_connection.getnewaddress()
    txid = await send_to_address_func(new_credit_tracking_address, amount_of_psl_to_fund_address_with, comment="Funding new credit tracking address ", comment_to="", subtract_fee_from_amount=False)
    logger.info(f"Funded new credit tracking address {new_credit_tracking_address} with {amount_of_psl_to_fund_address_with} PSL. TXID: {txid}")
    return new_credit_tracking_address, txid

async def check_psl_address_balance_func(address_to_check):
    global rpc_connection
    balance_at_address = await rpc_connection.z_getbalance(address_to_check)
    return balance_at_address

async def check_if_address_is_already_imported_in_local_wallet(address_to_check):
    global rpc_connection
    address_amounts_dict = await rpc_connection.listaddressamounts()
    # Convert the dictionary into a list of dictionaries, each representing a row
    data = [{'address': address, 'amount': amount} for address, amount in address_amounts_dict.items()]
    # Create the DataFrame from the list of dictionaries
    address_amounts_df = pd.DataFrame(data)        
    # Filter the DataFrame for the specified address
    address_amounts_df_filtered = address_amounts_df[address_amounts_df['address'] == address_to_check]
    if address_amounts_df_filtered.empty:
        return False
    return True

async def get_and_decode_raw_transaction(txid: str, blockhash: str = None) -> dict:
    """
    Retrieves and decodes detailed information about a specified transaction
    from the Pastel network using the RPC calls.

    Args:
        txid (str): The transaction id to fetch and decode.
        blockhash (str, optional): The block hash to specify which block to search for the transaction.

    Returns:
        dict: A dictionary containing detailed decoded information about the transaction.
    """
    global rpc_connection
    try:
        # Retrieve the raw transaction data
        raw_tx_data = await rpc_connection.getrawtransaction(txid, 0, blockhash)
        if not raw_tx_data:
            logger.error(f"Failed to retrieve raw transaction data for {txid}")
            return {}

        # Decode the raw transaction data
        decoded_tx_data = await rpc_connection.decoderawtransaction(raw_tx_data)
        if not decoded_tx_data:
            logger.error(f"Failed to decode raw transaction data for {txid}")
            return {}

        # Log the decoded transaction details
        logger.info(f"Decoded transaction details for {txid}: {decoded_tx_data}")

        return decoded_tx_data
    except Exception as e:
        logger.error(f"Error in get_and_decode_transaction for {txid}: {e}")
        return {}

async def get_transaction_details(txid: str, include_watchonly: bool = False) -> dict:
    """
    Fetches detailed information about a specified transaction from the Pastel network using the RPC call.

    Args:
        txid (str): The transaction id to fetch details for.
        include_watchonly (bool, optional): Whether to include watchonly addresses in the details. Defaults to False.

    Returns:
        dict: A dictionary containing detailed information about the transaction.
    """
    global rpc_connection
    try:
        # Call the 'gettransaction' RPC method with the provided txid and includeWatchonly flag
        transaction_details = await rpc_connection.gettransaction(txid, include_watchonly)
        
        # Log the retrieved transaction details
        logger.info(f"Retrieved transaction details for {txid}: {transaction_details}")

        return transaction_details
    except Exception as e:
        logger.error(f"Error retrieving transaction details for {txid}: {e}")
        return {}
    
async def send_tracking_amount_from_control_address_to_burn_address_to_confirm_inference_request(
    inference_request_id: str,
    credit_usage_tracking_psl_address: str,
    credit_usage_tracking_amount_in_psl: float,
    burn_address: str,
):
    """
    Send the tracking amount from the control address to the burn address to confirm an inference request.

    Args:
        inference_request_id (str): The ID of the inference request.
        credit_usage_tracking_psl_address (str): The control address to send the tracking amount from.
        credit_usage_tracking_amount_in_psl (float): The tracking amount in PSL to send.
        burn_address (str): The burn address to send the tracking amount to.

    Returns:
        str: The transaction ID (txid) if the transaction is successfully confirmed, None otherwise.

    Example:
        send_tracking_amount_from_control_address_to_burn_address_to_confirm_inference_request(
            inference_request_id="abc123",
            credit_usage_tracking_psl_address="PtczsZ91Bt3oDPDQotzUsrx1wjmsFVgf28n",
            credit_usage_tracking_amount_in_psl=0.5,
            burn_address="PtpasteLBurnAddressXXXXXXXXXXbJ5ndd"
        )
    """
    try:
        amounts = {
            burn_address: credit_usage_tracking_amount_in_psl
        }
        txid = await send_many_func(
            amounts=amounts,
            min_conf=0,
            comment="Confirmation tracking transaction for inference request with request_id " + inference_request_id, 
            change_address=credit_usage_tracking_psl_address
        )
        if txid is not None:
            logger.info(f"Sent {credit_usage_tracking_amount_in_psl} PSL from {credit_usage_tracking_psl_address} to {burn_address} to confirm inference request {inference_request_id}. TXID: {txid}")
            transaction_info = await rpc_connection.gettransaction(txid)
            if transaction_info:
                return txid
            else:
                logger.error(f"No transaction info found for TXID: {txid} to confirm inference request {inference_request_id}")
            return None
        else:
            logger.error(f"Failed to send {credit_usage_tracking_amount_in_psl} PSL from {credit_usage_tracking_psl_address} to {burn_address} to confirm inference request {inference_request_id}")
            return None
    except Exception as e:
        logger.error(f"Error in send_tracking_amount_from_control_address_to_burn_address_to_confirm_inference_request: {e}")
        raise    

async def import_address_func(address: str, label: str = "", rescan: bool = False) -> None:
    """
    Import an address or script (in hex) that can be watched as if it were in your wallet but cannot be used to spend.

    Args:
        address (str): The address to import.
        label (str, optional): An optional label for the address. Defaults to an empty string.
        rescan (bool, optional): Rescan the wallet for transactions. Defaults to False.

    Returns:
        None

    Raises:
        RPCError: If an error occurs during the RPC call.

    Example:
        import_address_func("myaddress", "testing", False)
    """
    global rpc_connection
    try:
        await rpc_connection.importaddress(address, label, rescan)
        logger.info(f"Imported address: {address}")
    except Exception as e:
        logger.error(f"Error importing address: {address}. Error: {e}")
    
async def compress_data_with_zstd_func(input_data):
    zstd_compression_level = 20
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    zstd_compressed_data = zstandard_compressor.compress(input_data)
    zstd_compressed_data__base64_encoded = base64.b64encode(zstd_compressed_data).decode('utf-8')
    return zstd_compressed_data, zstd_compressed_data__base64_encoded

async def decompress_data_with_zstd_func(compressed_input_data):
    zstd_decompressor = zstd.ZstdDecompressor()
    zstd_decompressed_data = zstd_decompressor.decompress(compressed_input_data)
    return zstd_decompressed_data

async def sign_message_with_pastelid_func(pastelid, message_to_sign, passphrase) -> str:
    global rpc_connection
    results_dict = await rpc_connection.pastelid('sign', message_to_sign, pastelid, passphrase, 'ed448')
    return results_dict['signature']

#____________________________________________________________________________________________________________________________
# SQLModel model classes based on the server's database_code.py

# Messaging related models:

class Message(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    receiving_sn_pastelid: str = Field(index=True)
    sending_sn_txid_vout: str = Field(index=True)
    receiving_sn_txid_vout: str = Field(index=True)
    message_type: str = Field(index=True)
    message_body: str = Field(sa_column=Column(JSON))
    signature: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    def __repr__(self):
        return f"<Message(id={self.id}, sending_sn_pastelid='{self.sending_sn_pastelid}', receiving_sn_pastelid='{self.receiving_sn_pastelid}', message_type='{self.message_type}', timestamp='{self.timestamp}')>"
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        json_schema_extra = {
            "example": {
                "sending_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "receiving_sn_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sending_sn_txid_vout": "0x1234...:0",
                "receiving_sn_txid_vout": "0x5678...:0",
                "message_type": "text",
                "message_body": "Hello, how are you?",
                "signature": "0xabcd...",
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }

class UserMessage(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    from_pastelid: str = Field(index=True)
    to_pastelid: str = Field(index=True)
    message_body: str = Field(sa_column=Column(JSON))
    message_signature: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        json_schema_extra = {
            "example": {
                "from_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "to_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "message_body": "Hey, let's meet up!",
                "message_signature": "0xdef0...",
                "timestamp": "2023-06-01T12:30:00Z"
            }
        }
        
# Credit pack purchasing/provisioning related models:        

class CreditPackPurchaseRequest(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, index=True, nullable=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    requesting_end_user_pastelid: str = Field(index=True)
    requested_initial_credits_in_credit_pack: int
    list_of_authorized_pastelids_allowed_to_use_credit_pack: str = Field(sa_column=Column(JSON))
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_timestamp_utc_iso_string: str
    request_pastel_block_height: int
    credit_purchase_request_message_version_string: str
    requesting_end_user_pastelid_signature_on_request_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "requested_initial_credits_in_credit_pack": 1000,
                "list_of_authorized_pastelids_allowed_to_use_credit_pack": ["jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk"],
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_timestamp_utc_iso_string": "2023-06-01T12:00:00Z",
                "request_pastel_block_height": 123456,
                "credit_purchase_request_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x5678...",
                "requesting_end_user_pastelid_signature_on_request_hash": "0xabcd..."
            }
        }

class CreditPackPurchaseRequestRejection(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    rejection_reason_string: str
    rejection_timestamp_utc_iso_string: str
    rejection_pastel_block_height: int
    credit_purchase_request_rejection_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_rejection_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "rejection_reason_string": "Invalid credit usage tracking PSL address",
                "rejection_timestamp_utc_iso_string": "2023-06-01T12:10:00Z",
                "rejection_pastel_block_height": 123457,
                "credit_purchase_request_rejection_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_rejection_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash": "0xdef0..."
            }
        }
        
class CreditPackPurchaseRequestPreliminaryPriceQuote(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_usage_tracking_psl_address: str = Field(index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    preliminary_quoted_price_per_credit_in_psl: float
    preliminary_total_cost_of_credit_pack_in_psl: float
    preliminary_price_quote_timestamp_utc_iso_string: str
    preliminary_price_quote_pastel_block_height: int
    preliminary_price_quote_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "preliminary_quoted_price_per_credit_in_psl": 0.1,
                "preliminary_total_cost_of_credit_pack_in_psl": 100,
                "preliminary_price_quote_timestamp_utc_iso_string": "2023-06-01T12:05:00Z",
                "preliminary_price_quote_pastel_block_height": 123456,
                "preliminary_price_quote_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash": "0xdef0..."
            }
        }

class CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    agree_with_preliminary_price_quote: bool
    credit_usage_tracking_psl_address: str = Field(index=True)
    preliminary_quoted_price_per_credit_in_psl: float
    preliminary_price_quote_response_timestamp_utc_iso_string: str
    preliminary_price_quote_response_pastel_block_height: int
    preliminary_price_quote_response_message_version_string: str
    requesting_end_user_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields": "0x5678...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "agree_with_preliminary_price_quote": True,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "preliminary_quoted_price_per_credit_in_psl": 0.1,
                "preliminary_price_quote_response_timestamp_utc_iso_string": "2023-06-01T12:10:00Z",
                "preliminary_price_quote_response_pastel_block_height": 123457,
                "preliminary_price_quote_response_message_version_string": "1.0",
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields": "0xdef0...",
                "requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash": "0x1234..."
            }
        }

class CreditPackPurchaseRequestResponseTermination(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    termination_reason_string: str
    termination_timestamp_utc_iso_string: str
    termination_pastel_block_height: int
    credit_purchase_request_termination_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_termination_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_termination_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "termination_reason_string": "Insufficient agreeing supernodes",
                "termination_timestamp_utc_iso_string": "2023-06-01T12:30:00Z",
                "termination_pastel_block_height": 123459,
                "credit_purchase_request_termination_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_termination_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_termination_hash": "0xdef0..."
            }
        }        
        
class CreditPackPurchaseRequestResponse(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    psl_cost_per_credit: float
    proposed_total_cost_of_credit_pack_in_psl: float
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_response_timestamp_utc_iso_string: str
    request_response_pastel_block_height: int
    credit_purchase_request_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms: str = Field(sa_column=Column(JSON))
    list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash: str = Field(sa_column=Column(JSON))
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk", ...}',
                "psl_cost_per_credit": 0.1,
                "proposed_total_cost_of_credit_pack_in_psl": 100,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_response_timestamp_utc_iso_string": "2023-06-01T12:15:00Z",
                "request_response_pastel_block_height": 123457,
                "credit_purchase_request_response_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms": ["jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk", "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP"],
                "list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash": ["0x1234...", "0x5678..."],
                "list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json": ["0xabcd...", "0xef01..."],
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x9abc...",
                "responding_supernode_signature_on_credit_pack_purchase_request_response_hash": "0xdef0..."
            }
        }

class CreditPackPurchaseRequestConfirmation(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    requesting_end_user_pastelid: str = Field(index=True)
    txid_of_credit_purchase_burn_transaction: str = Field(index=True)
    credit_purchase_request_confirmation_utc_iso_string: str
    credit_purchase_request_confirmation_pastel_block_height: int
    credit_purchase_request_confirmation_message_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x5678...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "txid_of_credit_purchase_burn_transaction": "0xabcd...",
                "credit_purchase_request_confirmation_utc_iso_string": "2023-06-01T12:30:00Z",
                "credit_purchase_request_confirmation_pastel_block_height": 123458,
                "credit_purchase_request_confirmation_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0xdef0...",
                "requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x1234..."
            }
        }

class CreditPackPurchaseRequestConfirmationResponse(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(foreign_key="creditpackpurchaserequestconfirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields", index=True)
    credit_pack_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str = Field(index=True)
    credit_pack_confirmation_failure_reason_if_applicable: str
    credit_purchase_request_confirmation_response_utc_iso_string: str
    credit_purchase_request_confirmation_response_pastel_block_height: int
    credit_purchase_request_confirmation_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x5678...",
                "credit_pack_confirmation_outcome_string": "success",
                "pastel_api_credit_pack_ticket_registration_txid": "0xabcd...",
                "credit_pack_confirmation_failure_reason_if_applicable": "",
                "credit_purchase_request_confirmation_response_utc_iso_string": "2023-06-01T12:45:00Z",
                "credit_purchase_request_confirmation_response_pastel_block_height": 123459,
                "credit_purchase_request_confirmation_response_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields": "0xdef0...",
                "responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash": "0x1234..."
            }
        }

class CreditPackRequestStatusCheck(SQLModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    requesting_end_user_pastelid: str
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields": "0x5678..."
            }
        }

class CreditPackPurchaseRequestStatus(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
    status: str = Field(index=True)
    status_details: str
    status_update_timestamp_utc_iso_string: str
    status_update_pastel_block_height: int
    credit_purchase_request_status_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_status_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_status_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "status": "in_progress",
                "status_details": "Waiting for price agreement responses from supernodes",
                "status_update_timestamp_utc_iso_string": "2023-06-01T12:30:00Z",
                "status_update_pastel_block_height": 123456,
                "credit_purchase_request_status_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_status_fields": "0x5678...",
                "responding_supernode_signature_on_credit_pack_purchase_request_status_hash": "0xef01..."
            }
        }

class CreditPackStorageRetryRequest(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    requesting_end_user_pastelid: str = Field(index=True)
    closest_agreeing_supernode_to_retry_storage_pastelid: str = Field(index=True)
    credit_pack_storage_retry_request_timestamp_utc_iso_string: str
    credit_pack_storage_retry_request_pastel_block_height: int
    credit_pack_storage_retry_request_message_version_string: str
    sha3_256_hash_of_credit_pack_storage_retry_request_fields: str
    requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "closest_agreeing_supernode_to_retry_storage_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "credit_pack_storage_retry_request_timestamp_utc_iso_string": "2023-06-01T12:50:00Z",
                "credit_pack_storage_retry_request_pastel_block_height": 123460,
                "credit_pack_storage_retry_request_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_storage_retry_request_fields": "0xabcd...",
                "requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash": "0xdef0..."
            }
        }

class CreditPackStorageRetryRequestResponse(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    credit_pack_storage_retry_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str
    credit_pack_storage_retry_confirmation_failure_reason_if_applicable: str
    credit_pack_storage_retry_confirmation_response_utc_iso_string: str
    credit_pack_storage_retry_confirmation_response_pastel_block_height: int
    credit_pack_storage_retry_confirmation_response_message_version_string: str
    closest_agreeing_supernode_to_retry_storage_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields: str
    closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash: str    
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x5678...",
                "credit_pack_storage_retry_confirmation_outcome_string": "success",
                "pastel_api_credit_pack_ticket_registration_txid": "0xabcd...",
                "credit_pack_storage_retry_confirmation_failure_reason_if_applicable": "",
                "credit_pack_storage_retry_confirmation_response_utc_iso_string": "2023-06-01T12:55:00Z",
                "credit_pack_storage_retry_confirmation_response_pastel_block_height": 123461,
                "credit_pack_storage_retry_confirmation_response_message_version_string": "1.0",
                "closest_agreeing_supernode_to_retry_storage_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields": "0xdef0...",
                "closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash": "0x1234..."
            }
        }

# Inference request related models:

class InferenceAPIUsageRequest(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    inference_request_id: str = Field(unique=True, index=True)
    requesting_pastelid: str = Field(index=True)
    credit_pack_ticket_pastel_txid: str = Field(index=True)
    requested_model_canonical_string: str
    model_inference_type_string: str
    model_parameters_json: str  # Store the dict serialized as a JSON string
    model_input_data_json_b64: str
    inference_request_utc_iso_string: str
    inference_request_pastel_block_height: int
    status: str = Field(index=True)
    inference_request_message_version_string: str
    sha3_256_hash_of_inference_request_fields: str
    requesting_pastelid_signature_on_request_hash: str
    @field_validator("model_parameters_json", mode="before")
    def serialize_dict_to_json(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v
    @field_validator("model_parameters_json", mode="after")
    def deserialize_json_to_dict(cls, v):
        try:
            return json.loads(v)
        except ValueError:
            return {}
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                id: "79df343b-4ad3-435c-800e-e59e616ff84d",
                "inference_request_id": "0x1234...",
                "requesting_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "credit_pack_ticket_pastel_txid": "0x5678...",
                "requested_model_canonical_string": "gpt-3.5-turbo",
                "model_inference_type_string": "text-completion",
                "model_parameters_json": '{"max_tokens": 100, "temperature": 0.7}',
                "model_input_data_json_b64": "eyJwcm9tcHQiOiAiSGVsbG8sIGhvdyBhcmUgeW91PyJ9",
                "inference_request_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_request_pastel_block_height": 123456,
                "status": "in_progress",
                "inference_request_message_version_string": "1.0",
                "sha3_256_hash_of_inference_request_fields": "0x5678...",
                "requesting_pastelid_signature_on_request_hash": "0xabcd..."
            }
        }

class InferenceAPIUsageResponse(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    inference_response_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inferenceapiusagerequest.inference_request_id", index=True)
    proposed_cost_of_request_in_inference_credits: float
    remaining_credits_in_pack_after_request_processed: float
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_confirmation_message_amount_in_patoshis: int
    max_block_height_to_include_confirmation_transaction: int
    inference_request_response_utc_iso_string: str
    inference_request_response_pastel_block_height: int
    inference_request_response_message_version_string: str    
    sha3_256_hash_of_inference_request_response_fields: str
    supernode_pastelid_and_signature_on_inference_request_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "inference_response_id": "0x1234...",
                "inference_request_id": "0x5678...",
                "proposed_cost_of_request_in_inference_credits": 10,
                "remaining_credits_in_pack_after_request_processed": 990,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_confirmation_message_amount_in_patoshis": 1000,
                "max_block_height_to_include_confirmation_transaction": 123456,
                "inference_request_response_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_request_response_pastel_block_height": 123456,
                "inference_request_response_message_version_string": "1.0",
                "sha3_256_hash_of_inference_request_response_fields": "0x5678...",
                "supernode_pastelid_and_signature_on_inference_request_response_hash": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk:0xabcd..."
            }
        }
        
class InferenceAPIOutputResult(SQLModel, table=True):
    id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
    inference_result_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inferenceapiusagerequest.inference_request_id", index=True)
    inference_response_id: str = Field(foreign_key="inferenceapiusageresponse.inference_response_id", index=True)
    responding_supernode_pastelid: str = Field(index=True)
    inference_result_json_base64: str
    inference_result_file_type_strings: str
    inference_result_utc_iso_string: str
    inference_result_pastel_block_height: int
    inference_result_message_version_string: str    
    sha3_256_hash_of_inference_result_fields: str    
    responding_supernode_signature_on_inference_result_id: str
    class Config:
        json_schema_extra = {
            "example": {
                "id": "79df343b-4ad3-435c-800e-e59e616ff84d",
                "inference_result_id": "0x1234...",
                "inference_request_id": "0x5678...",
                "inference_response_id": "0x9abc...",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "inference_result_json_base64": "eyJvdXRwdXQiOiAiSGVsbG8sIEknbSBkb2luZyBncmVhdCEgSG93IGFib3V0IHlvdT8ifQ==",
                "inference_result_file_type_strings": "json",
                "inference_result_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_result_pastel_block_height": 123456,
                "inference_result_message_version_string": "1.0",
                "sha3_256_hash_of_inference_result_fields": "0x5678...",
                "responding_supernode_signature_on_inference_result_id": "0xdef0..."
            }
        }
        
class InferenceConfirmation(SQLModel):
    inference_request_id: str
    requesting_pastelid: str
    confirmation_transaction: dict
    class Config:
        json_schema_extra = {
            "example": {
                "inference_request_id": "0x1234...",
                "requesting_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "confirmation_transaction": {
                    "txid": "0x5678...",
                    "amount": 1000,
                    "block_height": 123456
                }
            }
        }

#____________________________________________________________________________________________

# Class for handling interactions with the supernode servers:

class PastelMessagingClient:
    def __init__(self, pastelid: str, passphrase: str):
        self.pastelid = pastelid
        self.passphrase = passphrase

    async def request_and_sign_challenge(self, supernode_url: str) -> Dict[str, str]:
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.get(f"{supernode_url}/request_challenge/{self.pastelid}")
            response.raise_for_status()
            result = response.json()
            challenge = result["challenge"]
            challenge_id = result["challenge_id"]
            # Sign the challenge string using the local RPC client
            signature = await sign_message_with_pastelid_func(self.pastelid, challenge, self.passphrase)
            return {
                "challenge": challenge,
                "challenge_id": challenge_id,
                "signature": signature
            }
                    
    async def send_user_message(self, supernode_url: str, user_message: UserMessage) -> Dict[str, Any]:
        # Request and sign a challenge
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        # Prepare the user message payload
        payload = user_message.model_dump()
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/send_user_message",
                json={
                    "user_message": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            return result

    async def get_user_messages(self, supernode_url: str) -> List[UserMessage]:
        # Request and sign a challenge
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        signature = challenge_result["signature"]
        # Get the user messages
        params = {
            "pastelid": self.pastelid,
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": signature
        }
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.get(f"{supernode_url}/get_user_messages", params=params)
            response.raise_for_status()
            result = response.json()
            return [UserMessage.model_validate(message) for message in result]
        
    # Credit pack related client methods:        
    async def get_credit_pack_ticket_from_txid(self, supernode_url: str, txid: str) -> CreditPackPurchaseRequestResponse:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        params = {
            "txid": txid,
            "pastelid": self.pastelid,
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        log_action_with_payload("retrieving", "credit pack ticket from txid", params)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.get(f"{supernode_url}/get_credit_pack_ticket_from_txid", params=params)
            response.raise_for_status()
            result = response.json()
            result_transformed = transform_credit_pack_purchase_request_response(result)
            log_action_with_payload("receiving", "credit pack ticket from Supernode", result_transformed)
            return CreditPackPurchaseRequestResponse.model_validate(result_transformed)

    async def credit_pack_ticket_initial_purchase_request(self, supernode_url: str, credit_pack_request: CreditPackPurchaseRequest) -> Union[CreditPackPurchaseRequestPreliminaryPriceQuote, CreditPackPurchaseRequestRejection]:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = credit_pack_request.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}
        log_action_with_payload("requesting", "a new Pastel credit pack ticket", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/credit_purchase_initial_request",
                json={
                    "credit_pack_request": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            if "rejection_reason_string" in result:
                logger.error(f"Credit pack purchase request rejected: {result['rejection_reason_string']}")
                return CreditPackPurchaseRequestRejection.model_validate(result)
            else:
                log_action_with_payload("receiving", "response to credit pack purchase request", result)
                return CreditPackPurchaseRequestPreliminaryPriceQuote.model_validate(result)
            
    async def calculate_price_difference_percentage(self, quoted_price: float, estimated_price: float) -> float:
        if estimated_price == 0:
            raise ValueError("Estimated price cannot be zero.")
        difference_percentage = abs(quoted_price - estimated_price) / estimated_price
        return difference_percentage

    async def confirm_preliminary_price_quote(
        self,
        preliminary_price_quote: CreditPackPurchaseRequestPreliminaryPriceQuote,
        maximum_total_credit_pack_price_in_psl: Optional[float] = None,
        maximum_per_credit_price_in_psl: Optional[float] = None
    ) -> bool:
        if maximum_total_credit_pack_price_in_psl is None and maximum_per_credit_price_in_psl is None:
            maximum_per_credit_price_in_psl = MAXIMUM_PER_CREDIT_PRICE_IN_PSL_FOR_CLIENT
        # Extract the relevant fields from the preliminary price quote
        quoted_price_per_credit = preliminary_price_quote.preliminary_quoted_price_per_credit_in_psl
        quoted_total_price = preliminary_price_quote.preliminary_total_cost_of_credit_pack_in_psl
        # Parse the credit pack purchase request fields JSON
        request_fields = json.loads(preliminary_price_quote.credit_pack_purchase_request_fields_json)
        requested_credits = request_fields["requested_initial_credits_in_credit_pack"]
        # Calculate the missing maximum price parameter if not provided
        if maximum_total_credit_pack_price_in_psl is None:
            maximum_total_credit_pack_price_in_psl = maximum_per_credit_price_in_psl * requested_credits
        elif maximum_per_credit_price_in_psl is None:
            maximum_per_credit_price_in_psl = maximum_total_credit_pack_price_in_psl / requested_credits
        # Estimate the fair market price for the credits
        estimated_price_per_credit = await estimated_market_price_of_inference_credits_in_psl_terms()
        # Calculate the price difference percentage
        price_difference_percentage = await self.calculate_price_difference_percentage(quoted_price_per_credit, estimated_price_per_credit)
        # Compare the quoted prices with the maximum prices and the estimated fair price
        if (
            quoted_price_per_credit <= maximum_per_credit_price_in_psl and
            quoted_total_price <= maximum_total_credit_pack_price_in_psl and
            price_difference_percentage <= MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING
        ):
            logger.info(f"Preliminary price quote is within the acceptable range: {quoted_price_per_credit} PSL per credit, {quoted_total_price} PSL total, which is within the maximum of {maximum_per_credit_price_in_psl} PSL per credit and {maximum_total_credit_pack_price_in_psl} PSL total. The price difference from the estimated fair market price is {100*price_difference_percentage:.2f}%, which is within the allowed maximum of {100*MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING:.2f}%.")
            return True
        else:
            logger.warning(f"Preliminary price quote exceeds the maximum acceptable price or the price difference from the estimated fair price is too high! Quoted price: {quoted_price_per_credit} PSL per credit, {quoted_total_price} PSL total, maximum price: {maximum_per_credit_price_in_psl} PSL per credit, {maximum_total_credit_pack_price_in_psl} PSL total. The price difference from the estimated fair market price is {100*price_difference_percentage:.2f}%, which exceeds the allowed maximum of {100*MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING:.2f}%.")
            return False
        
    async def internal_estimate_of_credit_pack_ticket_cost_in_psl(self, desired_number_of_credits: float, price_cushion_pct: float):
        estimated_price_per_credit = await estimated_market_price_of_inference_credits_in_psl_terms()
        estimated_total_cost_of_ticket = round(desired_number_of_credits * estimated_price_per_credit * (1 + price_cushion_pct), 2)
        return estimated_total_cost_of_ticket
        
    async def credit_pack_ticket_preliminary_price_quote_response(
        self,
        supernode_url: str,
        credit_pack_request: CreditPackPurchaseRequest,
        preliminary_price_quote: Union[CreditPackPurchaseRequestPreliminaryPriceQuote, CreditPackPurchaseRequestRejection],
        maximum_total_credit_pack_price_in_psl: Optional[float] = None,
        maximum_per_credit_price_in_psl: Optional[float] = None
    ) -> Union[CreditPackPurchaseRequestResponse, CreditPackPurchaseRequestResponseTermination]:
        if isinstance(preliminary_price_quote, CreditPackPurchaseRequestRejection):
            logger.error(f"Credit pack purchase request rejected: {preliminary_price_quote.rejection_reason_string}")
            return None
        # Check if the end user agrees with the preliminary price quote
        agree_with_price_quote = await self.confirm_preliminary_price_quote(preliminary_price_quote, maximum_total_credit_pack_price_in_psl, maximum_per_credit_price_in_psl)
        if not agree_with_price_quote:
            logger.info("End user does not agree with the preliminary price quote!")
            agree_with_preliminary_price_quote = False
        else:
            agree_with_preliminary_price_quote = True        
        # Prepare the CreditPackPurchaseRequestPreliminaryPriceQuoteResponse
        price_quote_response = CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(
            sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
            sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields=preliminary_price_quote.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields,
            credit_pack_purchase_request_fields_json=preliminary_price_quote.credit_pack_purchase_request_fields_json,
            agree_with_preliminary_price_quote=agree_with_preliminary_price_quote,
            credit_usage_tracking_psl_address=preliminary_price_quote.credit_usage_tracking_psl_address,
            preliminary_quoted_price_per_credit_in_psl=preliminary_price_quote.preliminary_quoted_price_per_credit_in_psl,
            preliminary_price_quote_response_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            preliminary_price_quote_response_pastel_block_height=await get_current_pastel_block_height_func(),
            preliminary_price_quote_response_message_version_string="1.0",
            requesting_end_user_pastelid=credit_pack_request.requesting_end_user_pastelid,
            sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields="",
            requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash=""
        )
        # Generate the hash and signature fields
        price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(price_quote_response)
        price_quote_response.requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash = await sign_message_with_pastelid_func(
            credit_pack_request.requesting_end_user_pastelid,
            price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields,
            self.passphrase
        )
        # Send the CreditPackPurchaseRequestPreliminaryPriceQuoteResponse to the supernode
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = price_quote_response.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}
        log_action_with_payload("sending", "price quote response to supernode", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client:
            response = await client.post(
                f"{supernode_url}/credit_purchase_preliminary_price_quote_response",
                json={
                    "preliminary_price_quote_response": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            if "termination_reason_string" in result:
                logger.error(f"Credit pack purchase request response terminated: {result['termination_reason_string']}")
                return CreditPackPurchaseRequestResponseTermination.model_validate(result)
            else:
                transformed_result = transform_credit_pack_purchase_request_response(result)
                log_action_with_payload("receiving", "response to credit pack purchase request", transformed_result)
                return CreditPackPurchaseRequestResponse.model_validate(transformed_result)

    async def check_status_of_credit_purchase_request(self, supernode_url: str, credit_pack_purchase_request_hash: str) -> CreditPackPurchaseRequestStatus:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        # Create the CreditPackRequestStatusCheck model instance
        status_check = CreditPackRequestStatusCheck(
            sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_purchase_request_hash,
            requesting_end_user_pastelid=self.pastelid,
            requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields=await sign_message_with_pastelid_func(self.pastelid, credit_pack_purchase_request_hash, self.passphrase)
        )
        # Convert the model instance to JSON payload
        payload = status_check.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}
        log_action_with_payload("checking", "status of credit pack purchase request", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/check_status_of_credit_purchase_request",
                json={
                    "credit_pack_request_status_check": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("receiving", "credit pack purchase request response from Supernode", result)
            return CreditPackPurchaseRequestStatus.model_validate(result)

    async def confirm_credit_purchase_request(self, supernode_url: str, credit_pack_purchase_request_confirmation: CreditPackPurchaseRequestConfirmation) -> CreditPackPurchaseRequestConfirmationResponse:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = credit_pack_purchase_request_confirmation.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}
        log_action_with_payload("confirming", "credit pack purchase request", payload)        
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*30)) as client: # Need to be patient with the timeout here since it requires the transaction to be mined/confirmed
            response = await client.post(
                f"{supernode_url}/confirm_credit_purchase_request",
                json={
                    "confirmation": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("receiving", "response to credit pack purchase confirmation", result)
            return CreditPackPurchaseRequestConfirmationResponse.model_validate(result)

    async def credit_pack_purchase_completion_announcement(self, supernode_url: str, credit_pack_purchase_request_confirmation: CreditPackPurchaseRequestConfirmation) -> None:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = credit_pack_purchase_request_confirmation.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}        
        log_action_with_payload("sending", "purchase completion announcement message", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/credit_pack_purchase_completion_announcement",
                json={
                    "confirmation": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()

    async def credit_pack_storage_retry_request(self, supernode_url: str, credit_pack_storage_retry_request: CreditPackStorageRetryRequest) -> CreditPackStorageRetryRequestResponse:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = credit_pack_storage_retry_request.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}        
        log_action_with_payload("sending", "credit pack storage retry request", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/credit_pack_storage_retry_request",
                json={
                    "request": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("receiving", "response to credit pack storage retry request", result)
            return CreditPackStorageRetryRequestResponse.model_validate(result)

    async def credit_pack_storage_retry_completion_announcement(self, supernode_url: str, credit_pack_storage_retry_request_response: CreditPackStorageRetryRequestResponse) -> None:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = credit_pack_storage_retry_request_response.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}        
        log_action_with_payload("sending", "storage retry completion announcement message", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(
                f"{supernode_url}/credit_pack_storage_retry_completion_announcement",
                json={
                    "response": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()

    async def make_inference_api_usage_request(self, supernode_url: str, request_data: InferenceAPIUsageRequest) -> InferenceAPIUsageResponse:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = request_data.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}        
        log_action_with_payload("making", "inference usage request", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client:
            response = await client.post(
                f"{supernode_url}/make_inference_api_usage_request",
                json={
                    "inference_api_usage_request": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("received", "response to inference usage request", result)
            return InferenceAPIUsageResponse.model_validate(result)

    async def send_inference_confirmation(self, supernode_url: str, confirmation_data: InferenceConfirmation) -> Dict[str, Any]:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        payload = confirmation_data.model_dump()
        payload = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in payload.items()}        
        log_action_with_payload("sending", "inference confirmation", payload)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*4)) as client:
            response = await client.post(
                f"{supernode_url}/confirm_inference_request",
                json={
                    "inference_confirmation": payload,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
            )
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("receiving", "response to inference confirmation", result)
            return result

    async def check_status_of_inference_request_results(self, supernode_url: str, inference_response_id: str) -> bool:
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            try:
                logger.info(f"Checking status of inference request results for ID {inference_response_id}")
                response = await client.get(f"{supernode_url}/check_status_of_inference_request_results/{inference_response_id}")
                response.raise_for_status()
                result = response.json()
                log_action_with_payload("receiving", f"status of inference request results for ID {inference_response_id}", result)
                return result if isinstance(result, bool) else False
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error in check_status_of_inference_request_results from Supernode URL: {supernode_url}: {e}")
                return False
            except Exception as e:
                logger.error(f"Error in check_status_of_inference_request_results from Supernode URL: {supernode_url}: {e}")
                return False

    async def retrieve_inference_output_results(self, supernode_url: str, inference_request_id: str, inference_response_id: str) -> InferenceAPIOutputResult:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]
        params = {
            "inference_response_id": inference_response_id,
            "pastelid": self.pastelid,
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        log_action_with_payload("attempting", f"to retrieve inference output results for response ID {inference_response_id}", params)
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(f"{supernode_url}/retrieve_inference_output_results", params=params)
            response.raise_for_status()
            result = response.json()
            log_action_with_payload("receiving", "inference output results", result)
            return InferenceAPIOutputResult.model_validate(result)

    async def call_audit_inference_request_response(self, supernode_url: str, inference_response_id: str) -> InferenceAPIUsageResponse:
        try:
            signature = await sign_message_with_pastelid_func(self.pastelid, inference_response_id, self.passphrase)
            payload = {
                "inference_response_id": inference_response_id,
                "pastel_id": self.pastelid,
                "signature": signature
            }
            log_action_with_payload("calling", "audit inference request response", payload)
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*2)) as client:
                response = await client.post(f"{supernode_url}/audit_inference_request_response", json=payload)
                response.raise_for_status()
                result = response.json()
                log_action_with_payload("receiving", "response to audit inference request response", result)
                return InferenceAPIUsageResponse.model_validate(result)
        except Exception as e:
            logger.error(f"Error in audit_inference_request_response from Supernode URL: {supernode_url}: {e}")
            traceback.print_exc()
            raise

    async def call_audit_inference_request_result(self, supernode_url: str, inference_response_id: str) -> InferenceAPIOutputResult:
        try:
            signature = await sign_message_with_pastelid_func(self.pastelid, inference_response_id, self.passphrase)
            payload = {
                "inference_response_id": inference_response_id,
                "pastel_id": self.pastelid,
                "signature": signature
            }
            log_action_with_payload("calling", "audit inference request result", payload)
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*2)) as client:
                response = await client.post(f"{supernode_url}/audit_inference_request_result", json=payload)
                response.raise_for_status()
                result = response.json()
                log_action_with_payload("receiving", "response to audit inference request result", result)
                return InferenceAPIOutputResult.model_validate(result)
        except Exception as e:
            logger.error(f"Error in audit_inference_request_result from Supernode URL: {supernode_url}: {e}")
            raise
            
    async def audit_inference_request_response_id(self, inference_response_id: str, pastelid_of_supernode_to_audit: str):
        supernode_list_df, _ = await check_supernode_list_func()
        n = 4
        supernode_urls_and_pastelids = await get_n_closest_supernodes_to_pastelid_urls(n, self.pastelid, supernode_list_df)
        list_of_supernode_pastelids = [x[1] for x in supernode_urls_and_pastelids if x[1] != pastelid_of_supernode_to_audit]
        list_of_supernode_urls = [x[0] for x in supernode_urls_and_pastelids if x[1] != pastelid_of_supernode_to_audit]
        list_of_supernode_ips = [x.split('//')[1].split(':')[0] for x in list_of_supernode_urls]
        logger.info(f"Now attempting to audit inference request response with ID {inference_response_id} with {len(list_of_supernode_pastelids)} closest supernodes (with Supernode IPs of {list_of_supernode_ips})...")
        # Audit the inference request response
        logger.info(f"Now attempting to audit inference request response with ID {inference_response_id} by comparing information from other Supernodes to the information reported by the Responding Supernode...")
        response_audit_tasks = [self.call_audit_inference_request_response(url, inference_response_id) for url in list_of_supernode_urls]
        response_audit_results = await asyncio.gather(*response_audit_tasks)
        # Wait for 20 seconds before auditing the inference request result
        await asyncio.sleep(20)
        # Audit the inference request result
        logger.info(f"Now attempting to audit inference request result for response ID {inference_response_id} by comparing information from other Supernodes to the information reported by the Responding Supernode...")
        result_audit_tasks = [self.call_audit_inference_request_result(url, inference_response_id) for url in list_of_supernode_urls]
        result_audit_results = await asyncio.gather(*result_audit_tasks)
        # Combine the audit results
        audit_results = response_audit_results + result_audit_results
        logger.info(f"Audit results retrieved for inference response ID {inference_response_id}")
        return audit_results
    
    async def check_if_supernode_supports_desired_model(self, supernode_url: str, model_canonical_string: str, model_inference_type_string: str, model_parameters_json: str) -> bool:
        try:
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
                response = await client.get(f"{supernode_url}/get_inference_model_menu")
                response.raise_for_status()
                model_menu = response.json()
                desired_parameters = json.loads(model_parameters_json)  # Convert JSON string to dictionary
                for model in model_menu["models"]:
                    if model["model_name"] == model_canonical_string and \
                    model_inference_type_string in model["supported_inference_type_strings"]:
                        # Track unsupported parameters
                        unsupported_parameters = []
                        for desired_param, desired_value in desired_parameters.items():
                            param_found = False
                            for param in model["model_parameters"]:
                                if param["name"] == desired_param:
                                    # Check if the desired parameter value is within the valid range or options
                                    if "type" in param:
                                        if param["type"] == "int" and isinstance(desired_value, int):
                                            param_found = True
                                        elif param["type"] == "float" and isinstance(desired_value, float):
                                            param_found = True
                                        elif param["type"] == "string" and isinstance(desired_value, str):
                                            if "options" in param and desired_value in param["options"]:
                                                param_found = True
                                            elif "options" not in param:
                                                param_found = True
                                    else:
                                        param_found = True
                                    break
                            if not param_found:
                                unsupported_parameters.append(desired_param)
                        if not unsupported_parameters:
                            return True  # All desired parameters are supported
                        else:
                            # Log unsupported parameters and return False
                            unsupported_param_str = ", ".join(unsupported_parameters)
                            logger.error(f"Unsupported model parameters for {model_canonical_string}: {unsupported_param_str}")
                            return False
                return False  # Model not found or does not support the desired inference type
        except Exception as e:
            logger.error(f"Error in check_if_supernode_supports_desired_model from Supernode URL: {supernode_url}: {e}")
            return False
        
    async def get_closest_supernode_url_that_supports_desired_model(self, desired_model_canonical_string: str, desired_model_inference_type_string: str, desired_model_parameters_json: str):
        supernode_list_df, _ = await check_supernode_list_func()
        n = len(supernode_list_df)
        supernode_urls_and_pastelids = await get_n_closest_supernodes_to_pastelid_urls(n, self.pastelid, supernode_list_df)
        list_of_supernode_pastelids = [x[1] for x in supernode_urls_and_pastelids]
        list_of_supernode_urls = [x[0] for x in supernode_urls_and_pastelids]
        list_of_supernode_ips = [x.split('//')[1].split(':')[0] for x in list_of_supernode_urls]
        logger.info(f"Now attempting to check which supernodes support the desired model ({desired_model_canonical_string}) with {len(list_of_supernode_pastelids)} closest supernodes (with Supernode IPs of {list_of_supernode_ips})...")
        # Check which supernodes support the desired model
        model_support_tasks = [self.check_if_supernode_supports_desired_model(url, desired_model_canonical_string, desired_model_inference_type_string, desired_model_parameters_json) for url in list_of_supernode_urls]
        model_support_results = await asyncio.gather(*model_support_tasks)
        supernode_support_dict = {pastelid: supports for pastelid, supports in zip(list_of_supernode_pastelids, model_support_results)}
        logger.info(f"Found {sum(model_support_results)} supernodes that support the desired model ({desired_model_canonical_string}) out of {len(model_support_results)} checked.")
        closest_supporting_supernode_pastelid = list_of_supernode_pastelids[model_support_results.index(True)] if True in model_support_results else None
        closest_supporting_supernode_url = list_of_supernode_urls[model_support_results.index(True)] if True in model_support_results else None
        logger.info(f"Closest supporting supernode PastelID: {closest_supporting_supernode_pastelid} | URL: {closest_supporting_supernode_url}")
        return supernode_support_dict, closest_supporting_supernode_pastelid, closest_supporting_supernode_url


def validate_inference_response_fields(response_audit_results: List[InferenceAPIUsageResponse], usage_request_response: InferenceAPIUsageResponse) -> Dict[str, bool]:
    # Count the occurrences of each value for the relevant fields in response_audit_results
    inference_response_id_counts = {}
    inference_request_id_counts = {}
    proposed_cost_in_credits_counts = {}
    remaining_credits_after_request_counts = {}
    credit_usage_tracking_psl_address_counts = {}
    request_confirmation_message_amount_in_patoshis_counts = {}
    max_block_height_to_include_confirmation_transaction_counts = {}
    supernode_pastelid_and_signature_on_inference_response_id_counts = {}
    for result in response_audit_results:
        inference_response_id_counts[result.inference_response_id] = inference_response_id_counts.get(result.inference_response_id, 0) + 1
        inference_request_id_counts[result.inference_request_id] = inference_request_id_counts.get(result.inference_request_id, 0) + 1
        proposed_cost_in_credits_counts[result.proposed_cost_of_request_in_inference_credits] = proposed_cost_in_credits_counts.get(result.proposed_cost_of_request_in_inference_credits, 0) + 1
        remaining_credits_after_request_counts[result.remaining_credits_in_pack_after_request_processed] = remaining_credits_after_request_counts.get(result.remaining_credits_in_pack_after_request_processed, 0) + 1
        credit_usage_tracking_psl_address_counts[result.credit_usage_tracking_psl_address] = credit_usage_tracking_psl_address_counts.get(result.credit_usage_tracking_psl_address, 0) + 1
        request_confirmation_message_amount_in_patoshis_counts[result.request_confirmation_message_amount_in_patoshis] = request_confirmation_message_amount_in_patoshis_counts.get(result.request_confirmation_message_amount_in_patoshis, 0) + 1
        max_block_height_to_include_confirmation_transaction_counts[result.max_block_height_to_include_confirmation_transaction] = max_block_height_to_include_confirmation_transaction_counts.get(result.max_block_height_to_include_confirmation_transaction, 0) + 1
        supernode_pastelid_and_signature_on_inference_response_id_counts[result.supernode_pastelid_and_signature_on_inference_request_response_hash] = supernode_pastelid_and_signature_on_inference_response_id_counts.get(result.supernode_pastelid_and_signature_on_inference_request_response_hash, 0) + 1
    # Determine the majority value for each field
    majority_inference_response_id = max(inference_response_id_counts, key=inference_response_id_counts.get) if inference_response_id_counts else None
    majority_inference_request_id = max(inference_request_id_counts, key=inference_request_id_counts.get) if inference_request_id_counts else None
    majority_proposed_cost_in_credits = max(proposed_cost_in_credits_counts, key=proposed_cost_in_credits_counts.get) if proposed_cost_in_credits_counts else None
    majority_remaining_credits_after_request = max(remaining_credits_after_request_counts, key=remaining_credits_after_request_counts.get) if remaining_credits_after_request_counts else None
    majority_credit_usage_tracking_psl_address = max(credit_usage_tracking_psl_address_counts, key=credit_usage_tracking_psl_address_counts.get) if credit_usage_tracking_psl_address_counts else None
    majority_request_confirmation_message_amount_in_patoshis = max(request_confirmation_message_amount_in_patoshis_counts, key=request_confirmation_message_amount_in_patoshis_counts.get) if request_confirmation_message_amount_in_patoshis_counts else None
    majority_max_block_height_to_include_confirmation_transaction = max(max_block_height_to_include_confirmation_transaction_counts, key=max_block_height_to_include_confirmation_transaction_counts.get) if max_block_height_to_include_confirmation_transaction_counts else None
    majority_supernode_pastelid_and_signature_on_inference_response_id = max(supernode_pastelid_and_signature_on_inference_response_id_counts, key=supernode_pastelid_and_signature_on_inference_response_id_counts.get) if supernode_pastelid_and_signature_on_inference_response_id_counts else None
    # Compare the majority values with the values from the usage_request_response
    validation_results = {
        "inference_response_id": majority_inference_response_id == usage_request_response.inference_response_id,
        "inference_request_id": majority_inference_request_id == usage_request_response.inference_request_id,
        "proposed_cost_in_credits": majority_proposed_cost_in_credits == usage_request_response.proposed_cost_of_request_in_inference_credits,
        "remaining_credits_after_request": majority_remaining_credits_after_request == usage_request_response.remaining_credits_in_pack_after_request_processed,
        "credit_usage_tracking_psl_address": majority_credit_usage_tracking_psl_address == usage_request_response.credit_usage_tracking_psl_address,
        "request_confirmation_message_amount_in_patoshis": majority_request_confirmation_message_amount_in_patoshis == usage_request_response.request_confirmation_message_amount_in_patoshis,
        "max_block_height_to_include_confirmation_transaction": majority_max_block_height_to_include_confirmation_transaction == usage_request_response.max_block_height_to_include_confirmation_transaction,
        "supernode_pastelid_and_signature_on_inference_response_id": majority_supernode_pastelid_and_signature_on_inference_response_id == usage_request_response.supernode_pastelid_and_signature_on_inference_request_response_hash
    }
    return validation_results

def validate_inference_result_fields(result_audit_results: List[InferenceAPIOutputResult], usage_result: InferenceAPIOutputResult) -> Dict[str, bool]:
    # Count the occurrences of each value for the relevant fields in result_audit_results
    inference_result_id_counts = {}
    inference_request_id_counts = {}
    inference_response_id_counts = {}
    responding_supernode_pastelid_counts = {}
    inference_result_json_base64_counts = {}
    inference_result_file_type_strings_counts = {}
    responding_supernode_signature_on_inference_result_id_counts = {}
    for result in result_audit_results:
        inference_result_id_counts[result.inference_result_id] = inference_result_id_counts.get(result.inference_result_id, 0) + 1
        inference_request_id_counts[result.inference_request_id] = inference_request_id_counts.get(result.inference_request_id, 0) + 1
        inference_response_id_counts[result.inference_response_id] = inference_response_id_counts.get(result.inference_response_id, 0) + 1
        responding_supernode_pastelid_counts[result.responding_supernode_pastelid] = responding_supernode_pastelid_counts.get(result.responding_supernode_pastelid, 0) + 1
        inference_result_json_base64_counts[result.inference_result_json_base64[:32]] = inference_result_json_base64_counts.get(result.inference_result_json_base64[:32], 0) + 1
        inference_result_file_type_strings_counts[result.inference_result_file_type_strings] = inference_result_file_type_strings_counts.get(result.inference_result_file_type_strings, 0) + 1
        responding_supernode_signature_on_inference_result_id_counts[result.responding_supernode_signature_on_inference_result_id] = responding_supernode_signature_on_inference_result_id_counts.get(result.responding_supernode_signature_on_inference_result_id, 0) + 1
    # Determine the majority value for each field
    majority_inference_result_id = max(inference_result_id_counts, key=inference_result_id_counts.get) if inference_result_id_counts else None
    majority_inference_request_id = max(inference_request_id_counts, key=inference_request_id_counts.get) if inference_request_id_counts else None
    majority_inference_response_id = max(inference_response_id_counts, key=inference_response_id_counts.get) if inference_response_id_counts else None
    majority_responding_supernode_pastelid = max(responding_supernode_pastelid_counts, key=responding_supernode_pastelid_counts.get) if responding_supernode_pastelid_counts else None
    majority_inference_result_json_base64 = max(inference_result_json_base64_counts, key=inference_result_json_base64_counts.get) if inference_result_json_base64_counts else None
    majority_inference_result_file_type_strings = max(inference_result_file_type_strings_counts, key=inference_result_file_type_strings_counts.get) if inference_result_file_type_strings_counts else None
    majority_responding_supernode_signature_on_inference_result_id = max(responding_supernode_signature_on_inference_result_id_counts, key=responding_supernode_signature_on_inference_result_id_counts.get) if responding_supernode_signature_on_inference_result_id_counts else None
    # Compare the majority values with the values from the usage_result
    validation_results = {
        "inference_result_id": majority_inference_result_id == usage_result.inference_result_id,
        "inference_request_id": majority_inference_request_id == usage_result.inference_request_id,
        "inference_response_id": majority_inference_response_id == usage_result.inference_response_id,
        "responding_supernode_pastelid": majority_responding_supernode_pastelid == usage_result.responding_supernode_pastelid,
        "inference_result_json_base64": majority_inference_result_json_base64 == usage_result.inference_result_json_base64[:32],
        "inference_result_file_type_strings": majority_inference_result_file_type_strings == usage_result.inference_result_file_type_strings,
        "responding_supernode_signature_on_inference_result_id": majority_responding_supernode_signature_on_inference_result_id == usage_result.responding_supernode_signature_on_inference_result_id
    }
    return validation_results
            
def validate_inference_data(inference_result_dict: Dict[str, Any], audit_results: List[Union[InferenceAPIUsageResponse, InferenceAPIOutputResult]]) -> Dict[str, Dict[str, bool]]:
    # Extract relevant fields from inference_result_dict
    usage_request_response = InferenceAPIUsageResponse.model_validate(inference_result_dict["usage_request_response"])
    usage_result = InferenceAPIOutputResult.model_validate(inference_result_dict["output_results"])
    # Validate InferenceAPIUsageResponse fields
    response_validation_results = validate_inference_response_fields(
        [result for result in audit_results if isinstance(result, InferenceAPIUsageResponse)],
        usage_request_response
    )
    # Validate InferenceAPIOutputResult fields
    result_validation_results = validate_inference_result_fields(
        [result for result in audit_results if isinstance(result, InferenceAPIOutputResult)],
        usage_result
    )
    # Combine validation results
    validation_results = {
        "response_validation": response_validation_results,
        "result_validation": result_validation_results
    }
    return validation_results
        
async def send_message_and_check_for_new_incoming_messages(
    to_pastelid: str,
    message_body: str
):
    global MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE
    # Create messaging client to use:
    messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)    
    # Get the list of Supernodes
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    # Send a user message
    logger.info("Sending user message...")
    logger.info(f"Recipient pastelid: {to_pastelid}")
    # Lookup the 3 closest supernodes to the recipient pastelid
    closest_supernodes_to_recipient = await get_n_closest_supernodes_to_pastelid_urls(3, to_pastelid, supernode_list_df)
    logger.info(f"Closest Supernodes to recipient pastelid: {[sn[1] for sn in closest_supernodes_to_recipient]}")
    # Create a UserMessage object
    user_message = UserMessage(
        from_pastelid=MY_LOCAL_PASTELID,
        to_pastelid=to_pastelid,
        message_body=json.dumps(message_body),  # Convert message_body to JSON
        message_signature=await sign_message_with_pastelid_func(MY_LOCAL_PASTELID, message_body, MY_PASTELID_PASSPHRASE)
    )
    # Send the message to the 3 closest Supernodes concurrently
    send_tasks = []
    for supernode_url, _ in closest_supernodes_to_recipient:
        send_task = asyncio.create_task(messaging_client.send_user_message(supernode_url, user_message))
        send_tasks.append(send_task)
    send_results = await asyncio.gather(*send_tasks)
    logger.info(f"Sent user messages: {send_results}")
    # Get user messages from the 3 closest Supernodes
    logger.info("Retrieving incoming user messages...")
    logger.info(f"My local pastelid: {messaging_client.pastelid}")
    # Lookup the 3 closest supernodes to the local pastelid
    closest_supernodes_to_local = await get_n_closest_supernodes_to_pastelid_urls(3, messaging_client.pastelid, supernode_list_df)
    logger.info(f"Closest Supernodes to local pastelid: {[sn[1] for sn in closest_supernodes_to_local]}")
    # Retrieve messages from the 3 closest Supernodes concurrently
    message_retrieval_tasks = []
    for supernode_url, _ in closest_supernodes_to_local:
        message_retrieval_task = asyncio.create_task(messaging_client.get_user_messages(supernode_url))
        message_retrieval_tasks.append(message_retrieval_task)
    message_lists = await asyncio.gather(*message_retrieval_tasks)
    # Combine the message lists and remove duplicates
    unique_messages = []
    message_ids = set()
    for message_list in message_lists:
        for message in message_list:
            if message.id not in message_ids:
                unique_messages.append(message)
                message_ids.add(message.id)
    logger.info(f"Retrieved unique user messages: {unique_messages}")
    message_dict = {
        "sent_messages": send_results,
        "received_messages": unique_messages
    }        
    return message_dict

async def handle_credit_pack_ticket_end_to_end(
    number_of_credits: float,
    credit_usage_tracking_psl_address: str,
    burn_address: str,
    maximum_total_credit_pack_price_in_psl: Optional[float] = None,
    maximum_per_credit_price_in_psl: Optional[float] = None
    ):
    global MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE
    # Create messaging client to use:
    messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)    
    # Get the list of Supernodes
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    # Prepare the credit pack request
    credit_pack_request = CreditPackPurchaseRequest(
        requesting_end_user_pastelid=MY_LOCAL_PASTELID,
        requested_initial_credits_in_credit_pack=number_of_credits,
        list_of_authorized_pastelids_allowed_to_use_credit_pack=json.dumps([MY_LOCAL_PASTELID]),
        credit_usage_tracking_psl_address=credit_usage_tracking_psl_address,
        request_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
        request_pastel_block_height=await get_current_pastel_block_height_func(),
        credit_purchase_request_message_version_string="1.0",
        sha3_256_hash_of_credit_pack_purchase_request_fields="",
        requesting_end_user_pastelid_signature_on_request_hash=""
    )
    credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(credit_pack_request)
    credit_pack_request.requesting_end_user_pastelid_signature_on_request_hash = await sign_message_with_pastelid_func(MY_LOCAL_PASTELID, credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields, MY_PASTELID_PASSPHRASE)
    # print(f"Credit pack purchase request data:\n {credit_pack_request}")
    # Send the credit pack request to the highest-ranked supernode
    closest_supernodes = await get_n_closest_supernodes_to_pastelid_urls(1, MY_LOCAL_PASTELID, supernode_list_df)
    highest_ranked_supernode_url = closest_supernodes[0][0]
    preliminary_price_quote = await messaging_client.credit_pack_ticket_initial_purchase_request(highest_ranked_supernode_url, credit_pack_request)
    # Check if the end user agrees with the preliminary price quote
    signed_credit_pack_ticket_or_rejection = await messaging_client.credit_pack_ticket_preliminary_price_quote_response(highest_ranked_supernode_url, credit_pack_request, preliminary_price_quote, maximum_total_credit_pack_price_in_psl, maximum_per_credit_price_in_psl)
    if isinstance(signed_credit_pack_ticket_or_rejection, CreditPackPurchaseRequestResponseTermination):
        logger.error(f"Credit pack purchase request terminated: {signed_credit_pack_ticket_or_rejection.termination_reason_string}")
        return None
    signed_credit_pack_ticket = signed_credit_pack_ticket_or_rejection
    # Send the required PSL from the credit usage tracking address to the burn address
    burn_transaction_txid = await send_to_address_func(burn_address, round(signed_credit_pack_ticket.proposed_total_cost_of_credit_pack_in_psl, 5), "Burn transaction for credit pack ticket")
    if burn_transaction_txid is None:
        logger.error("Error sending PSL to burn address for credit pack ticket")
        return None
    # Prepare the credit pack purchase request confirmation
    credit_pack_purchase_request_confirmation = CreditPackPurchaseRequestConfirmation(
        sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
        sha3_256_hash_of_credit_pack_purchase_request_response_fields=signed_credit_pack_ticket.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
        credit_pack_purchase_request_fields_json=signed_credit_pack_ticket.credit_pack_purchase_request_fields_json,
        requesting_end_user_pastelid=MY_LOCAL_PASTELID,
        txid_of_credit_purchase_burn_transaction=burn_transaction_txid,
        credit_purchase_request_confirmation_utc_iso_string=datetime.now(dt.UTC).isoformat(),
        credit_purchase_request_confirmation_pastel_block_height=await get_current_pastel_block_height_func(),
        credit_purchase_request_confirmation_message_version_string="1.0",
        sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields="",
        requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=""
    )
    credit_pack_purchase_request_confirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(credit_pack_purchase_request_confirmation)
    credit_pack_purchase_request_confirmation.requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields = await sign_message_with_pastelid_func(MY_LOCAL_PASTELID, credit_pack_purchase_request_confirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields, MY_PASTELID_PASSPHRASE)
    credit_pack_purchase_request_confirmation_response = await messaging_client.confirm_credit_purchase_request(highest_ranked_supernode_url, credit_pack_purchase_request_confirmation)
    # Send the credit pack purchase completion announcement to the responding supernode:
    for supernode_pastelid in signed_credit_pack_ticket.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms:
        try:
            is_valid_pastelid = check_if_pastelid_is_valid_func(supernode_pastelid)
            if is_valid_pastelid:
                supernode_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                await messaging_client.credit_pack_purchase_completion_announcement(supernode_url, credit_pack_purchase_request_confirmation)
        except Exception as e:
            logger.error(f"Error sending credit_pack_purchase_completion_announcement to Supernode URL: {supernode_url}: {e}")
    # Check the status of the credit purchase request
    for i, (supernode_url, _) in enumerate(closest_supernodes):
        try:
            credit_pack_purchase_request_status = await messaging_client.check_status_of_credit_purchase_request(supernode_url, credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields)
            logger.info(f"Credit pack purchase request status: {credit_pack_purchase_request_status}")
            break
        except Exception as e:
            logger.error(f"Error checking status of credit purchase request with Supernode {i+1}: {e}")
            if i == len(closest_supernodes) - 1:
                logger.error("Failed to check status of credit purchase request with all Supernodes")
                return None
    if credit_pack_purchase_request_status.status != "completed":
        logger.error(f"Credit pack purchase request failed: {credit_pack_purchase_request_status.status}")
        # Retry the storage with the closest agreeing supernode
        closest_agreeing_supernode_pastelid = await get_closest_supernode_pastelid_from_list(MY_LOCAL_PASTELID, signed_credit_pack_ticket.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms)
        credit_pack_storage_retry_request = CreditPackStorageRetryRequest(
            sha3_256_hash_of_credit_pack_purchase_request_response_fields=signed_credit_pack_ticket.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
            credit_pack_purchase_request_fields_json=signed_credit_pack_ticket.credit_pack_purchase_request_fields_json,
            requesting_end_user_pastelid=MY_LOCAL_PASTELID,
            closest_agreeing_supernode_to_retry_storage_pastelid=closest_agreeing_supernode_pastelid,
            credit_pack_storage_retry_request_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            credit_pack_storage_retry_request_pastel_block_height=await get_current_pastel_block_height_func(),
            credit_pack_storage_retry_request_message_version_string="1.0",
            sha3_256_hash_of_credit_pack_storage_retry_request_fields="",
            requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash=""
        )
        credit_pack_storage_retry_request.sha3_256_hash_of_credit_pack_storage_retry_request_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(credit_pack_storage_retry_request)
        credit_pack_storage_retry_request.requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash = await sign_message_with_pastelid_func(MY_LOCAL_PASTELID, credit_pack_storage_retry_request.sha3_256_hash_of_credit_pack_storage_retry_request_fields, MY_PASTELID_PASSPHRASE)
        closest_agreeing_supernode_url = await get_supernode_url_from_pastelid_func(closest_agreeing_supernode_pastelid, supernode_list_df)
        credit_pack_storage_retry_request_response = await messaging_client.credit_pack_storage_retry_request(closest_agreeing_supernode_url, credit_pack_storage_retry_request)
        # Send the credit pack purchase completion announcement to the responding supernode:
        for supernode_pastelid in signed_credit_pack_ticket.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms:
            try:
                is_valid_pastelid = check_if_pastelid_is_valid_func(supernode_pastelid)
                if is_valid_pastelid:
                    supernode_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    await messaging_client.credit_pack_purchase_completion_announcement(supernode_url, credit_pack_storage_retry_request_response)        
            except Exception as e:
                logger.error(f"Error sending credit_pack_purchase_completion_announcement to Supernode URL: {supernode_url}: {e}")
        return credit_pack_storage_retry_request_response
    else:
        return credit_pack_purchase_request_confirmation_response

async def get_credit_pack_ticket_info_end_to_end(credit_pack_ticket_pastel_txid: str):
    # Create messaging client to use:
    messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)       
    # Get the closest Supernode URL
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    supernode_url, _ = await get_closest_supernode_to_pastelid_url(MY_LOCAL_PASTELID, supernode_list_df)
    logger.info(f"Getting credit pack ticket data from Supernode URL: {supernode_url}...")
    credit_pack_data_object = await messaging_client.get_credit_pack_ticket_from_txid(supernode_url, credit_pack_ticket_pastel_txid)
    return credit_pack_data_object

async def handle_inference_request_end_to_end(
    credit_pack_ticket_pastel_txid: str,
    input_prompt_to_llm: str,
    requested_model_canonical_string: str,
    model_inference_type_string: str,
    model_parameters: dict,
    maximum_inference_cost_in_credits: float,
    burn_address: str
):
    # Create messaging client to use:
    messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)       
    # Get the closest Supernode URL
    model_parameters_json = json.dumps(model_parameters)
    supernode_support_dict, closest_supporting_supernode_pastelid, closest_supporting_supernode_url = await messaging_client.get_closest_supernode_url_that_supports_desired_model(requested_model_canonical_string, model_inference_type_string, model_parameters_json) 
    supernode_url = closest_supporting_supernode_url
    supernode_pastelid = closest_supporting_supernode_pastelid
    try:
        assert(supernode_url is not None)
    except AssertionError:
        logger.error(f"Error! No supporting Supernode found for the desired model: {requested_model_canonical_string} with inference type: {model_inference_type_string}")
    input_prompt_to_llm__base64_encoded = base64.b64encode(input_prompt_to_llm.encode()).decode('utf-8')
    # Prepare the inference API usage request
    inference_request_data = InferenceAPIUsageRequest(
        inference_request_id=str(uuid.uuid4()),
        requesting_pastelid=MY_LOCAL_PASTELID,
        credit_pack_ticket_pastel_txid=credit_pack_ticket_pastel_txid,
        requested_model_canonical_string=requested_model_canonical_string,
        model_inference_type_string=model_inference_type_string,
        model_parameters_json=model_parameters_json,
        model_input_data_json_b64=input_prompt_to_llm__base64_encoded,
        inference_request_utc_iso_string=datetime.now(dt.UTC).isoformat(),
        inference_request_pastel_block_height=await get_current_pastel_block_height_func(),
        status="initiating",
        inference_request_message_version_string="1.0",
        sha3_256_hash_of_inference_request_fields="",
        requesting_pastelid_signature_on_request_hash=""
    )
    sha3_256_hash_of_inference_request_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(inference_request_data)
    inference_request_data.sha3_256_hash_of_inference_request_fields = sha3_256_hash_of_inference_request_fields
    requesting_pastelid_signature_on_request_hash = await sign_message_with_pastelid_func(MY_LOCAL_PASTELID, sha3_256_hash_of_inference_request_fields, MY_PASTELID_PASSPHRASE)
    inference_request_data.requesting_pastelid_signature_on_request_hash = requesting_pastelid_signature_on_request_hash    
    # Send the inference API usage request
    usage_request_response = await messaging_client.make_inference_api_usage_request(supernode_url, inference_request_data)
    logger.info(f"Received inference API usage request response from SN:\n {usage_request_response}")
    # Check the validity of the response
    validation_errors = await validate_credit_pack_ticket_message_data_func(usage_request_response)
    if validation_errors:
        raise ValueError(f"Invalid inference request response from Supernode URL {supernode_url}: {', '.join(validation_errors)}")    
    # Extract the relevant information from the response
    usage_request_response_dict = usage_request_response.model_dump()
    usage_request_response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in usage_request_response_dict.items()}
    inference_request_id = usage_request_response_dict["inference_request_id"]
    inference_response_id = usage_request_response_dict["inference_response_id"]
    proposed_cost_in_credits = float(usage_request_response_dict["proposed_cost_of_request_in_inference_credits"])
    credit_usage_tracking_psl_address = usage_request_response_dict["credit_usage_tracking_psl_address"]
    credit_usage_tracking_amount_in_psl = float(usage_request_response_dict["request_confirmation_message_amount_in_patoshis"])/(10**5) # Divide by number of Patoshis per PSL
    # Check if tracking address contains enough PSL to send tracking amount:
    tracking_address_balance = await check_psl_address_balance_alternative_func(credit_usage_tracking_psl_address)
    if tracking_address_balance < credit_usage_tracking_amount_in_psl:
        logger.error(f"Insufficient balance in tracking address: {credit_usage_tracking_psl_address}; amount needed: {credit_usage_tracking_amount_in_psl}; current balance: {tracking_address_balance}; shortfall: {credit_usage_tracking_amount_in_psl - tracking_address_balance}")
        return None, None, None
    if proposed_cost_in_credits <= maximum_inference_cost_in_credits: # Check if the quoted price is less than or equal to the maximum allowed cost
        # TODO: Check if the credit pack has sufficient credits based on the actual credit pack data
        # Send the required PSL coins to authorize the request:
        tracking_transaction_txid = await send_tracking_amount_from_control_address_to_burn_address_to_confirm_inference_request(inference_request_id, credit_usage_tracking_psl_address, credit_usage_tracking_amount_in_psl, burn_address)
        txid_looks_valid = bool(re.match("^[0-9a-fA-F]{64}$", tracking_transaction_txid))
        if txid_looks_valid: # Prepare the inference confirmation
            confirmation_data = InferenceConfirmation(
                inference_request_id=inference_request_id,
                requesting_pastelid=MY_LOCAL_PASTELID,
                confirmation_transaction={"txid": tracking_transaction_txid}
            )
            confirmation_result = await messaging_client.send_inference_confirmation(supernode_url, confirmation_data) # Send the inference confirmation
            logger.info(f"Sent inference confirmation: {confirmation_result}")
            max_tries_to_get_confirmation = 10
            initial_wait_time_in_seconds = 10
            wait_time_in_seconds = initial_wait_time_in_seconds
            for cnt in range(max_tries_to_get_confirmation):
                wait_time_in_seconds = wait_time_in_seconds*(1.2**cnt)
                logger.info(f"Waiting for the inference results for {round(wait_time_in_seconds, 1)} seconds... (Attempt {cnt+1}/{max_tries_to_get_confirmation}); Checking with Supernode URL: {supernode_url}")
                await asyncio.sleep(wait_time_in_seconds)
                assert(len(inference_request_id)>0)
                assert(len(inference_response_id)>0)
                results_available = await messaging_client.check_status_of_inference_request_results(supernode_url, inference_response_id) # Get the inference output results
                if results_available:
                    output_results = await messaging_client.retrieve_inference_output_results(supernode_url, inference_request_id, inference_response_id)
                    output_results_dict = output_results.model_dump()
                    output_results_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in output_results_dict.items()}
                    output_results_size = len(output_results.inference_result_json_base64)
                    max_response_size_to_log = 20000
                    if output_results_size < max_response_size_to_log:
                        logger.info(f"Retrieved inference output results: {output_results}")
                    # Create the inference_result_dict with all relevant information
                    inference_request_data_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_request_data.model_dump().items()}
                    inference_result_dict = {
                        "supernode_url": supernode_url,
                        "request_data": inference_request_data_dict,
                        "usage_request_response": usage_request_response_dict,
                        "input_prompt_to_llm": input_prompt_to_llm,
                        "output_results": output_results_dict,
                    }
                    if model_inference_type_string == "text_to_image":
                        inference_result_dict["generated_image_base64"] = output_results.inference_result_json_base64
                        inference_result_dict["generated_image_decoded"] = base64.b64decode(output_results.inference_result_json_base64)
                    else:
                        inference_result_decoded = base64.b64decode(output_results.inference_result_json_base64).decode()
                        logger.info(f"Decoded response:\n {inference_result_decoded}")
                        inference_result_dict["inference_result_decoded"] = inference_result_decoded
                    use_audit_feature = 1
                    if use_audit_feature:
                        logger.info("Waiting 5 seconds for audit results to be available...")
                        await asyncio.sleep(5) # Wait for the audit results to be available
                        audit_results = await messaging_client.audit_inference_request_response_id(inference_response_id, supernode_pastelid)
                        validation_results = validate_inference_data(inference_result_dict, audit_results)
                        logger.info(f"Validation results: {validation_results}")      
                    else:
                        audit_results = ""
                        validation_results = ""
                    return inference_result_dict, audit_results, validation_results
                else:
                    logger.info("Inference results not available yet; retrying...")
        else:
            logger.error(f"Invalid tracking transaction TXID: {tracking_transaction_txid}")
    else:
        logger.info(f"Quoted price of {proposed_cost_in_credits} credits exceeds the maximum allowed cost of {maximum_inference_cost_in_credits} credits. Inference request not confirmed.")
    return None, None, None
    
async def main():
    global rpc_connection
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
    rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    if rpc_port == '9932':
        burn_address = 'PtpasteLBurnAddressXXXXXXXXXXbJ5ndd'
    elif rpc_port == '19932':
        burn_address = 'tPpasteLBurnAddressXXXXXXXXXXX3wy7u'
    elif rpc_port == '29932':
        burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7' # https://blockchain-devel.slack.com/archives/C03Q2MCQG9K/p1705896449986459
        
    use_test_messaging_functionality = 0
    use_test_credit_pack_ticket_functionality = 0
    use_test_credit_pack_ticket_usage = 1
    use_test_inference_request_functionality = 1
    use_test_llm_text_completion = 1
    use_test_image_generation = 0

    if use_test_messaging_functionality:
        # Sample message data:
        message_body = "Hello, this is a brand 🍉 NEW test message from a regular user!"
        to_pastelid = "jXXiVgtFzLto4eYziePHjjb1hj3c6eXdABej5ndnQ62B8ouv1GYveJaD5QUMfainQM3b4MTieQuzFEmJexw8Cr"        
        message_dict = await send_message_and_check_for_new_incoming_messages(to_pastelid, message_body)
        logger.info(f"Message data: {message_dict}")

    #________________________________________________________

    if use_test_credit_pack_ticket_functionality:
        # Test credit pack ticket functionality
        desired_number_of_credits = 1500
        amount_of_psl_for_tracking_transactions = 10.0
        credit_price_cushion_percentage = 0.15
        maximum_total_amount_of_psl_to_fund_in_new_tracking_address = 100000.0
        messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)       
        estimated_total_cost_in_psl_for_credit_pack = await messaging_client.internal_estimate_of_credit_pack_ticket_cost_in_psl(desired_number_of_credits, credit_price_cushion_percentage)
        if estimated_total_cost_in_psl_for_credit_pack > maximum_total_amount_of_psl_to_fund_in_new_tracking_address:
            logger.error(f"Estimated total cost of credit pack exceeds the maximum allowed amount of {maximum_total_amount_of_psl_to_fund_in_new_tracking_address} PSL")
            raise ValueError(f"Estimated total cost of credit pack exceeds the maximum allowed amount of {maximum_total_amount_of_psl_to_fund_in_new_tracking_address} PSL")
        amount_to_fund_credit_tracking_address = round(amount_of_psl_for_tracking_transactions + estimated_total_cost_in_psl_for_credit_pack, 2)
        credit_usage_tracking_psl_address, _ = await create_and_fund_new_psl_credit_tracking_address(amount_to_fund_credit_tracking_address)
        credit_pack_purchase_request_confirmation_response = await handle_credit_pack_ticket_end_to_end(
            desired_number_of_credits,
            credit_usage_tracking_psl_address,
            burn_address,
        )
        if credit_pack_purchase_request_confirmation_response:
            logger.info(f"Credit pack ticket stored on the blockchain with transaction ID: {credit_pack_purchase_request_confirmation_response.pastel_api_credit_pack_ticket_registration_txid}")
            logger.info(f"Credit pack details: {credit_pack_purchase_request_confirmation_response}")
        else:
            logger.error("Credit pack ticket storage failed!")

    if 'credit_pack_purchase_request_confirmation_response' in globals():
        credit_pack_ticket_pastel_txid = credit_pack_purchase_request_confirmation_response.pastel_api_credit_pack_ticket_registration_txid
    else:
        credit_pack_ticket_pastel_txid = "6145722a224cc85875cd57c5cff18a136a13e655ac9483dfad6ace0b195d8cd0" # "44oZnTrqgCF8wF2AyCzqTJKRioYLq3Wosv1M"
    logger.info(f"Selected credit pack ticket transaction ID: {credit_pack_ticket_pastel_txid}; corresponding psl tracking address: {credit_usage_tracking_psl_address}") # Each credit pack ticket has a corresponding UNIQUE tracking PSL address!
    
    # TODO: Add all credit pack tickets we create to local client database and make function that can automatically select the credit pack ticket with the largest remaining balance of credits and its corresponding psl tracking address.
    
    if use_test_credit_pack_ticket_usage:
        start_time = time.time()
        credit_ticket_object = await get_credit_pack_ticket_info_end_to_end(credit_pack_ticket_pastel_txid)
        credit_pack_purchase_request_dict = json.loads(credit_ticket_object.credit_pack_purchase_request_fields_json)
        credit_usage_tracking_psl_address = credit_pack_purchase_request_dict['credit_usage_tracking_psl_address']
        initial_credit_pack_balance = credit_pack_purchase_request_dict['requested_initial_credits_in_credit_pack']
        logger.info(f"Credit pack ticket data retrieved with initial balance {initial_credit_pack_balance} and credit tracking PSL address of {credit_usage_tracking_psl_address}")
        logger.info(f"Corresponding credit pack request dict: {credit_pack_purchase_request_dict}")
        end_time = time.time()
        duration_in_seconds = (end_time - start_time)
        logger.info(f"Total time taken for credit pack ticket lookup: {round(duration_in_seconds, 2)} seconds")
                
                
    if use_test_inference_request_functionality:
        if use_test_llm_text_completion:
            start_time = time.time()
            # input_prompt_text_to_llm = "Explain to me with detailed examples what a Galois group is and how it helps understand the roots of a polynomial equation: "
            # input_prompt_text_to_llm = "What made the Battle of Salamus so important? What clever ideas were used in the battle? What mistakes were made?"
            input_prompt_text_to_llm = "how do you measure the speed of an earthquake?"
            # requested_model_canonical_string = "mistralapi-mistral-large-latest" # "groq-mixtral-8x7b-32768" # "claude3-opus" "claude3-sonnet" "mistral-7b-instruct-v0.2" # "claude3-haiku" # "phi-2" , "mistral-7b-instruct-v0.2", "groq-mixtral-8x7b-32768", "groq-llama2-70b-4096", "groq-gemma-7b-it", "mistralapi-mistral-small-latest", "mistralapi-mistral-large-latest"
            requested_model_canonical_string = "claude3-opus" # "groq-mixtral-8x7b-32768" # "groq-mixtral-8x7b-32768" # "claude3-opus" "claude3-sonnet" "mistral-7b-instruct-v0.2" # "claude3-haiku" # "phi-2" , "mistral-7b-instruct-v0.2", "groq-mixtral-8x7b-32768", "groq-llama2-70b-4096", "groq-gemma-7b-it", "mistralapi-mistral-small-latest", "mistralapi-mistral-large-latest"
            model_inference_type_string = "text_completion" # "embedding"        
            # model_parameters = {"number_of_tokens_to_generate": 200, "temperature": 0.7, "grammar_file_string": "", "number_of_completions_to_generate": 1}
            model_parameters = {"number_of_tokens_to_generate": 2000, "number_of_completions_to_generate": 1}
            max_credit_cost_to_approve_inference_request = 200.0
            
            inference_dict, audit_results, validation_results = await handle_inference_request_end_to_end(
                credit_pack_ticket_pastel_txid,
                input_prompt_text_to_llm,
                requested_model_canonical_string,
                model_inference_type_string,
                model_parameters,
                max_credit_cost_to_approve_inference_request,
                burn_address
            )
            logger.info(f"Inference result data:\n\n {inference_dict}")
            logger.info("\n_____________________________________________________________________\n") 
            logger.info(f"\n\nFinal Decoded Inference Result:\n\n {inference_dict['inference_result_decoded']}")
            end_time = time.time()
            duration_in_minutes = (end_time - start_time)/60
            logger.info(f"Total time taken for inference request: {round(duration_in_minutes, 2)} minutes")

        if use_test_image_generation:
            # Test image generation
            start_time = time.time()
            # input_prompt_text_to_llm = "A stunning house with a beautiful garden and a pool, in a photorealistic style."
            input_prompt_text_to_llm = "A picture of a clown holding a sign that says PASTEL"
            requested_model_canonical_string = "stability-core"
            model_inference_type_string = "text_to_image"
            style_strings_list = ["3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art", "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound", "neon-punk", "origami", "photographic", "pixel-art", "tile-texture"] 
            style_preset_string = style_strings_list[-3] # "photographic"
            output_format_list = ["png", "jpeg", "webp"]
            output_format_string = output_format_list[0] # "png"
            aspect_ratio_list = ["16:9", "1:1", "21:9", "2:3", "3:2", "4:5", "5:4", "9:16", "9:21"]
            aspect_ratio_string = aspect_ratio_list[0] # "16:9"
            random_seed = random.randint(0, 1000)
            if "core" in requested_model_canonical_string:
                model_parameters = {
                    "aspect_ratio": aspect_ratio_string,
                    "seed": random_seed,
                    "style_preset": style_preset_string,
                    "output_format": output_format_string,
                    "negative_prompt": "low quality, blurry, pixelated"
                }
            else:
                model_parameters = {
                    "height": 512,
                    "width": 512,
                    "steps": 50,
                    "seed": 0,
                    "num_samples": 1,
                    "negative_prompt": "low quality, blurry, pixelated",
                    "style_preset": style_preset_string
                }
            max_credit_cost_to_approve_inference_request = 200.0
            inference_dict, audit_results, validation_results = await handle_inference_request_end_to_end(
                credit_pack_ticket_pastel_txid,
                input_prompt_text_to_llm,
                requested_model_canonical_string,
                model_inference_type_string,
                model_parameters,
                max_credit_cost_to_approve_inference_request,
                burn_address
            )
            logger.info(f"Inference result data received at {datetime.now()}; decoded image size in megabytes: {round(len(inference_dict['generated_image_decoded'])/(1024*1024), 2)} MB")
            logger.info("\n_____________________________________________________________________\n")
            
            # Save the generated image to a file
            current_datetime_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            image_generation_prompt_without_whitespace_or_newlines_abbreviated_to_100_characters = re.sub(r'\s+', '_', input_prompt_text_to_llm)[:100]
            generated_image_filename = f"generated_image__prompt__{image_generation_prompt_without_whitespace_or_newlines_abbreviated_to_100_characters}__generated_on_{current_datetime_string}.{output_format_string}"
            generated_image_folder_name = 'generated_images'
            if not os.path.exists(generated_image_folder_name):
                os.makedirs(generated_image_folder_name)
            generated_image_file_path = os.path.join(generated_image_folder_name, generated_image_filename)    
            image_data = inference_dict['generated_image_decoded']
            with open(generated_image_file_path, "wb") as f:
                f.write(image_data)
            logger.info(f"Generated image saved as '{generated_image_file_path}'")
            end_time = time.time()
            duration_in_minutes = (end_time - start_time)/60
            logger.info(f"Total time taken for inference request: {round(duration_in_minutes, 2)} minutes")
            
if __name__ == "__main__":
    asyncio.run(main())