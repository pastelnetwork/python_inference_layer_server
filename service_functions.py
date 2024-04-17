import asyncio
import base64
import decimal
import hashlib
import ipaddress
import json
import os
import platform
import statistics
import time
import uuid
import random
import re
import sys
import html
import warnings
import pytz
from urllib.parse import quote_plus, unquote_plus
from datetime import datetime, timedelta, date
import datetime as dt
import pandas as pd
import httpx
from httpx import AsyncClient, Limits, Timeout
import urllib.parse as urlparse
from logger_config import setup_logger
from blockchain_ticket_storage import store_data_in_blockchain, retrieve_data_from_blockchain
import zstandard as zstd
from sqlalchemy.exc import OperationalError, InvalidRequestError
from typing import List, Tuple, Dict, Union, Optional
from decouple import Config as DecoupleConfig, RepositoryEnv
from magika import Magika
import tiktoken
import anthropic
from groq import AsyncGroq
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from cryptography.fernet import Fernet
from fuzzywuzzy import process
from transformers import AutoTokenizer, GPT2TokenizerFast, WhisperTokenizer
import database_code as db_code
from sqlmodel import select, func, SQLModel

encryption_key = None
magika = Magika()

SENSITIVE_ENV_FIELDS = ["LOCAL_PASTEL_ID_PASSPHRASE", "MY_PASTELID_PASSPHRASE", "SWISS_ARMY_LLAMA_SECURITY_TOKEN", "OPENAI_API_KEY", "CLAUDE3_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "STABILITY_API_KEY", "OPENROUTER_API_KEY"]
LOCAL_PASTEL_ID_PASSPHRASE = None
MY_PASTELID_PASSPHRASE = None
SWISS_ARMY_LLAMA_SECURITY_TOKEN = None
OPENAI_API_KEY = None
CLAUDE3_API_KEY = None
GROQ_API_KEY = None
MISTRAL_API_KEY = None
STABILITY_API_KEY = None
OPENROUTER_API_KEY = None

def get_env_value(key):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(current_dir, '.env')    
    try:
        with open(env_file_path, 'r') as env_file:
            for line in env_file:
                if line.startswith(key + '='):
                    return line.split('=', 1)[1].strip() # Split on the first '=' to allow for '=' in the value
    except FileNotFoundError:
        print(f"Error: .env file at {env_file_path} not found.")
    return None

def generate_or_load_encryption_key_sync():
    key_file_path = os.path.expanduser('~/env_encryption_key_for_supernode_inference_app')
    key = None
    if os.path.exists(key_file_path): # Check if key file exists and load it
        with open(key_file_path, 'rb') as key_file:
            key = key_file.read()
        try:
            Fernet(key)  # Validate the key
            loaded_or_generated = "loaded"
        except ValueError:
            key = None
    if key is None: # If key is invalid or doesn't exist, generate a new one
        logger.info("Invalid or no encryption key found. Generating a new one.")
        loaded_or_generated = "generated"
        key = Fernet.generate_key()
        with open(key_file_path, 'wb') as key_file:
            key_file.write(key)
        print(f"Generated new encryption key for sensitive env fields: {key}")
        encrypt_sensitive_fields(key)  # Encrypt sensitive fields if generating key for the first time
    logger.info(f"Encryption key {loaded_or_generated} successfully.")        
    return key

def encrypt_sensitive_fields(key):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    env_file_path = os.path.join(current_dir, '.env')    
    cipher_suite = Fernet(key)
    with open(env_file_path, 'r') as file: # Load existing .env file
        lines = file.readlines()
    updated_lines = [] # Encrypt and update sensitive fields
    for line in lines:
        if any(field in line for field in SENSITIVE_ENV_FIELDS):
            for field in SENSITIVE_ENV_FIELDS:
                if line.startswith(field):
                    value = line.strip().split('=')[1]
                    encrypted_data = cipher_suite.encrypt(value.encode()).decode()
                    url_encoded_encrypted_data = quote_plus(encrypted_data)
                    line = f"{field}={url_encoded_encrypted_data}\n"
                    print(f"Encrypted {field}: {url_encoded_encrypted_data}")
                    break
        updated_lines.append(line)
    with open(env_file_path, 'w') as file: # Write the updated lines back to the .env file
        file.writelines(updated_lines)
    logger.info(f"Updated {len(SENSITIVE_ENV_FIELDS)} sensitive fields in .env file with encrypted values!")

def decrypt_sensitive_data(url_encoded_encrypted_data, encryption_key):
    cipher_suite = Fernet(encryption_key)
    encrypted_data = unquote_plus(url_encoded_encrypted_data)  # URL-decode first
    decrypted_data = cipher_suite.decrypt(encrypted_data.encode()).decode()  # Ensure this is a bytes-like object
    return decrypted_data

def encrypt_sensitive_data(data, encryption_key):
    cipher_suite = Fernet(encryption_key)
    encrypted_data = cipher_suite.encrypt(data.encode()).decode()
    url_encoded_encrypted_data = quote_plus(encrypted_data)
    return url_encoded_encrypted_data

def decrypt_sensitive_fields():
    global LOCAL_PASTEL_ID_PASSPHRASE, MY_PASTELID_PASSPHRASE, SWISS_ARMY_LLAMA_SECURITY_TOKEN, OPENAI_API_KEY, CLAUDE3_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, STABILITY_API_KEY, OPENROUTER_API_KEY, encryption_key
    LOCAL_PASTEL_ID_PASSPHRASE = decrypt_sensitive_data(get_env_value("LOCAL_PASTEL_ID_PASSPHRASE"), encryption_key)
    MY_PASTELID_PASSPHRASE = decrypt_sensitive_data(get_env_value("MY_PASTELID_PASSPHRASE"), encryption_key)
    SWISS_ARMY_LLAMA_SECURITY_TOKEN = decrypt_sensitive_data(get_env_value("SWISS_ARMY_LLAMA_SECURITY_TOKEN"), encryption_key)
    OPENAI_API_KEY = decrypt_sensitive_data(get_env_value("OPENAI_API_KEY"), encryption_key)
    CLAUDE3_API_KEY = decrypt_sensitive_data(get_env_value("CLAUDE3_API_KEY"), encryption_key)
    GROQ_API_KEY = decrypt_sensitive_data(get_env_value("GROQ_API_KEY"), encryption_key)
    MISTRAL_API_KEY = decrypt_sensitive_data(get_env_value("MISTRAL_API_KEY"), encryption_key)
    STABILITY_API_KEY = decrypt_sensitive_data(get_env_value("STABILITY_API_KEY"), encryption_key)
    OPENROUTER_API_KEY = decrypt_sensitive_data(get_env_value("OPENROUTER_API_KEY"), encryption_key)
        
# Logger setup
logger = setup_logger()

number_of_cpus = os.cpu_count()
my_os = platform.system()
loop = asyncio.get_event_loop()
warnings.filterwarnings('ignore')

config = DecoupleConfig(RepositoryEnv('.env'))
TEMP_OVERRIDE_LOCALHOST_ONLY = config.get("TEMP_OVERRIDE_LOCALHOST_ONLY", default=0, cast=int)
NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE = config.get("NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE", default=3, cast=int)
GITHUB_MODEL_MENU_URL = config.get("GITHUB_MODEL_MENU_URL")
CHALLENGE_EXPIRATION_TIME_IN_SECONDS = config.get("CHALLENGE_EXPIRATION_TIME_IN_SECONDS", default=300, cast=int)
SWISS_ARMY_LLAMA_PORT = config.get("SWISS_ARMY_LLAMA_PORT", default=8089, cast=int)
CREDIT_COST_MULTIPLIER_FACTOR = config.get("CREDIT_COST_MULTIPLIER_FACTOR", default=0.1, cast=float)
BASE_TRANSACTION_AMOUNT = decimal.Decimal(config.get("BASE_TRANSACTION_AMOUNT", default=0.000001, cast=float))
FEE_PER_KB = decimal.Decimal(config.get("FEE_PER_KB", default=0.0001, cast=float))
MESSAGING_TIMEOUT_IN_SECONDS = config.get("MESSAGING_TIMEOUT_IN_SECONDS", default=60, cast=int)
API_KEY_TESTS_FILE = "api_key_tests.json"
API_KEY_TEST_VALIDITY_HOURS = config.get("API_KEY_TEST_VALIDITY_HOURS", default=72, cast=int)
TARGET_VALUE_PER_CREDIT_IN_USD = config.get("TARGET_VALUE_PER_CREDIT_IN_USD", default=0.1, cast=float)
TARGET_PROFIT_MARGIN = config.get("TARGET_PROFIT_MARGIN", default=0.1, cast=float)
MINIMUM_COST_IN_CREDITS = config.get("MINIMUM_COST_IN_CREDITS", default=0.1, cast=float)
CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER = config.get("CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER", default=10, cast=int) # Since we always round inference credits to the nearest 0.1, this gives us enough resolution using Patoshis     
MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET = config.get("MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET", default=50, cast=int)
MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING = config.get("MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING", default=0.1, cast=float)
MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS = config.get("MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS", default=15.0, cast=float)
MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS = config.get("MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS", default=1, cast=int)
MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED = config.get("MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED", default=10, cast=int)
MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION = config.get("MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION", default=3, cast=int)
MAXIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES = config.get("MAXIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES", default=10, cast=int)
SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE = config.get("SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE", default=0.51, cast=float)
SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE = config.get("SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE", default=0.85, cast=float)
SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK = 1
challenge_store = {}

def parse_timestamp(timestamp_str):
    try:
        # Attempt to parse with fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        # Fall back to parsing without fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S')

def parse_and_format(value):
    try:
        if isinstance(value, str): # Unescape the JSON string if it's a string
            unescaped_value = json.loads(json.dumps(value))
            parsed_value = json.loads(unescaped_value)
        else:
            parsed_value = value
        return json.dumps(parsed_value, indent=4)
    except (json.JSONDecodeError, TypeError):
        return value

def pretty_json_func(data):
    if isinstance(data, SQLModel):
        data = data.model_dump()
    if isinstance(data, dict):
        formatted_data = {}
        for key, value in data.items():
            if key.endswith("_json"):
                if isinstance(value, dict):
                    formatted_data[key] = parse_and_format(value)
                else:
                    formatted_data[key] = parse_and_format(value)
            elif isinstance(value, dict):
                formatted_data[key] = pretty_json_func(value)
            else:
                formatted_data[key] = value
        return json.dumps(formatted_data, indent=4)
    elif isinstance(data, str):
        return parse_and_format(data)
    else:
        return data
    
def log_action_with_payload(action_string, payload_name, json_payload):
    logger.info(f"Now {action_string} {payload_name} with payload:\n{pretty_json_func(json_payload)}")

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

def compute_sha3_256_hexdigest(input_str):
    """Compute the SHA3-256 hash of the input string and return the hexadecimal digest."""
    return hashlib.sha3_256(input_str.encode()).hexdigest()

async def calculate_xor_distance(pastelid1: str, pastelid2: str) -> int:
    hash1 = compute_sha3_256_hexdigest(pastelid1)
    hash2 = compute_sha3_256_hexdigest(pastelid2)
    xor_result = int(hash1, 16) ^ int(hash2, 16)
    return xor_result

async def get_supernode_url_from_pastelid_func(pastelid: str, supernode_list_df: pd.DataFrame) -> str:
    supernode_row = supernode_list_df[supernode_list_df['extKey'] == pastelid]
    if not supernode_row.empty:
        supernode_ipaddress_port = supernode_row['ipaddress:port'].values[0]
        ipaddress = supernode_ipaddress_port.split(':')[0]
        supernode_url = f"http://{ipaddress}:7123"
        return supernode_url
    else:
        raise ValueError(f"Supernode with PastelID {pastelid} not found in the supernode list")

async def get_closest_supernode_pastelid_from_list(local_pastelid: str, supernode_pastelids: List[str]) -> str:
    xor_distances = [(supernode_pastelid, await calculate_xor_distance(local_pastelid, supernode_pastelid)) for supernode_pastelid in supernode_pastelids]
    closest_supernode = min(xor_distances, key=lambda x: x[1])
    return closest_supernode[0]

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
    if isinstance(o, decimal.Decimal):
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

async def get_last_block_data_func():
    global rpc_connection
    current_block_height = await get_current_pastel_block_height_func()
    block_data = await rpc_connection.getblock(str(current_block_height))
    return block_data

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

async def check_psl_address_balance_func(address_to_check):
    global rpc_connection
    balance_at_address = await rpc_connection.z_getbalance(address_to_check)
    return balance_at_address

async def get_raw_transaction_func(txid):
    global rpc_connection
    raw_transaction_data = await rpc_connection.getrawtransaction(txid, 1) 
    return raw_transaction_data

async def verify_message_with_pastelid_func(pastelid, message_to_verify, pastelid_signature_on_message) -> str:
    global rpc_connection
    verification_result = await rpc_connection.pastelid('verify', message_to_verify, pastelid_signature_on_message, pastelid, 'ed448')
    return verification_result['verification']

async def generate_challenge(pastelid: str) -> Tuple[str, str]:
    """
    Generates a random challenge string and a unique challenge ID for a given PastelID.
    The challenge string is stored temporarily, associated with the challenge ID, and expires after a certain period of time.
    """
    challenge_string = str(uuid.uuid4())
    challenge_id = str(uuid.uuid4())
    expiration_time = time.time() + CHALLENGE_EXPIRATION_TIME_IN_SECONDS
    challenge_store[challenge_id] = {
        "pastelid": pastelid,
        "challenge_string": challenge_string,
        "expiration_time": expiration_time
    }
    return challenge_string, challenge_id

async def verify_challenge_signature(pastelid: str, signature: str, challenge_id: str) -> bool:
    """
    Verifies the signature of the PastelID on the challenge string associated with the provided challenge ID.
    If the signature is valid and the challenge ID exists and hasn't expired, it returns True. Otherwise, it returns False.
    """
    global rpc_connection
    if challenge_id not in challenge_store:
        return False
    challenge_data = challenge_store[challenge_id]
    stored_pastelid = challenge_data["pastelid"]
    challenge_string = challenge_data["challenge_string"]
    expiration_time = challenge_data["expiration_time"]
    if pastelid != stored_pastelid:
        return False
    current_time = time.time()
    if current_time > expiration_time:
        del challenge_store[challenge_id]
        return False
    verification_result = await rpc_connection.pastelid('verify', challenge_string, signature, pastelid, 'ed448')
    is_valid_signature = verification_result['verification'] == 'OK'
    if is_valid_signature:
        del challenge_store[challenge_id]
        return True
    else:
        return False
    
async def verify_challenge_signature_from_inference_request_id(inference_request_id: str, challenge_signature: str, challenge_id: str) -> bool:
    # Retrieve the inference API usage request from the database
    async with db_code.Session() as db:
        result = await db.exec(
            select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
        )
        inference_request = result.one_or_none()
    if inference_request:
        requesting_pastelid = inference_request.requesting_pastelid
        is_valid_signature = await verify_challenge_signature(requesting_pastelid, challenge_signature, challenge_id)
        return is_valid_signature
    else:
        return False

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
    masternode_list_full_df['lastseentime'] = pd.to_datetime(masternode_list_full_df['lastseentime'], unit='s')
    masternode_list_full_df['lastpaidtime'] = pd.to_datetime(masternode_list_full_df['lastpaidtime'], unit='s')
    masternode_list_full_df['activeseconds'] = masternode_list_full_df['activeseconds'].astype(int)
    masternode_list_full_df['lastpaidblock'] = masternode_list_full_df['lastpaidblock'].astype(int)
    masternode_list_full_df['activedays'] = [float(x)/86400.0 for x in masternode_list_full_df['activeseconds'].values.tolist()]
    masternode_list_full_df['rank'] = masternode_list_full_df['rank'].astype(int)
    masternode_list_full_df__json = masternode_list_full_df.to_json(orient='index')
    return masternode_list_full_df, masternode_list_full_df__json
    
async def get_local_machine_supernode_data_func():
    local_machine_ip = get_external_ip_func()
    supernode_list_full_df, _ = await check_supernode_list_func()
    proper_port_number = statistics.mode([x.split(':')[1] for x in supernode_list_full_df['ipaddress:port'].values.tolist()])
    local_machine_ip_with_proper_port = local_machine_ip + ':' + proper_port_number
    local_machine_supernode_data = supernode_list_full_df[supernode_list_full_df['ipaddress:port'] == local_machine_ip_with_proper_port]
    if len(local_machine_supernode_data) == 0:
        logger.error('Local machine is not a supernode!')
        return 0, 0, 0, 0
    else:
        local_sn_rank = local_machine_supernode_data['rank'].values[0]
        local_sn_pastelid = local_machine_supernode_data['extKey'].values[0]
    return local_machine_supernode_data, local_sn_rank, local_sn_pastelid, local_machine_ip_with_proper_port

async def get_my_local_pastelid_func():
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    my_local_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]    
    return my_local_pastelid

async def get_sn_data_from_pastelid_func(specified_pastelid):
    supernode_list_full_df, _ = await check_supernode_list_func()
    specified_machine_supernode_data = supernode_list_full_df[supernode_list_full_df['extKey'] == specified_pastelid]
    if len(specified_machine_supernode_data) == 0:
        logger.error('Specified machine is not a supernode!')
        return pd.DataFrame()
    else:
        return specified_machine_supernode_data

async def get_sn_data_from_sn_pubkey_func(specified_sn_pubkey):
    supernode_list_full_df, _ = await check_supernode_list_func()
    specified_machine_supernode_data = supernode_list_full_df[supernode_list_full_df['pubkey'] == specified_sn_pubkey]
    if len(specified_machine_supernode_data) == 0:
        logger.error('Specified machine is not a supernode!')
        return pd.DataFrame()
    else:
        return specified_machine_supernode_data

async def compress_data_with_zstd_func(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    zstd_compression_level = 22
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    zstd_compressed_data = zstandard_compressor.compress(input_data)
    zstd_compressed_data__base64_encoded = base64.b64encode(zstd_compressed_data).decode('utf-8')
    return zstd_compressed_data, zstd_compressed_data__base64_encoded

async def decompress_data_with_zstd_func(compressed_input_data):
    zstd_decompressor = zstd.ZstdDecompressor()
    zstd_decompressed_data = zstd_decompressor.decompress(compressed_input_data)
    return zstd_decompressed_data

async def list_sn_messages_func():
    global rpc_connection
    datetime_cutoff_to_ignore_obsolete_messages = pd.to_datetime(datetime.now() - timedelta(days=NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE))
    supernode_list_df, _ = await check_supernode_list_func()
    txid_vout_to_pastelid_dict = dict(zip(supernode_list_df.index, supernode_list_df['extKey']))
    async with db_code.Session() as db:
        # Retrieve messages from the database that meet the timestamp criteria
        db_messages = db.exec(
            select(db_code.Message)
            .where(db_code.Message.timestamp >= datetime_cutoff_to_ignore_obsolete_messages)
            .order_by(db_code.Message.timestamp.desc())
        ).all()
        existing_messages = {(message.sending_sn_pastelid, message.receiving_sn_pastelid, message.timestamp) for message in db_messages}
    # Retrieve new messages from the RPC interface
    new_messages = await rpc_connection.masternode('message', 'list')
    new_messages_data = []
    for message in new_messages:
        message_key = list(message.keys())[0]
        message = message[message_key]
        sending_sn_txid_vout = message['From']
        receiving_sn_txid_vout = message['To']
        sending_pastelid = txid_vout_to_pastelid_dict.get(sending_sn_txid_vout)
        receiving_pastelid = txid_vout_to_pastelid_dict.get(receiving_sn_txid_vout)
        if sending_pastelid is None or receiving_pastelid is None:
            logger.warning(f"Skipping message due to missing PastelID for txid_vout: {sending_sn_txid_vout} or {receiving_sn_txid_vout}")
            continue
        message_timestamp = parse_timestamp(datetime.fromtimestamp(message['Timestamp']).isoformat())
        # Check if the message already exists in the database
        if (sending_pastelid, receiving_pastelid, message_timestamp) in existing_messages:
            logger.debug("Message already exists in the database. Skipping...")
            continue
        message_body = base64.b64decode(message['Message'].encode('utf-8'))
        verification_status = await verify_received_message_using_pastelid_func(message_body, sending_pastelid)
        decompressed_message = await decompress_data_with_zstd_func(message_body)
        decompressed_message = decompressed_message.decode('utf-8')
        try:
            message_dict = json.loads(decompressed_message)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON: {e}")
            logger.error(f"Decompressed message: {decompressed_message}")
            continue
        if verification_status == 'OK':
            new_message = {
                'sending_sn_pastelid': sending_pastelid,
                'receiving_sn_pastelid': receiving_pastelid,
                'message_type': message_dict['message_type'],
                'message_body': decompressed_message,
                'signature': message_dict['signature'],
                'timestamp': message_timestamp,
                'sending_sn_txid_vout': sending_sn_txid_vout,
                'receiving_sn_txid_vout': receiving_sn_txid_vout
            }
            new_messages_data.append(new_message)
    combined_messages_df = pd.DataFrame(new_messages_data)
    if not combined_messages_df.empty:
        combined_messages_df = combined_messages_df[combined_messages_df['timestamp'] >= datetime_cutoff_to_ignore_obsolete_messages]
        combined_messages_df = combined_messages_df.sort_values('timestamp', ascending=False)
    return combined_messages_df

async def sign_message_with_pastelid_func(pastelid, message_to_sign, passphrase) -> str:
    global rpc_connection
    results_dict = await rpc_connection.pastelid('sign', message_to_sign, pastelid, passphrase, 'ed448')
    return results_dict['signature']

async def parse_sn_messages_from_last_k_minutes_func(k=10, message_type='all'):
    messages_list_df = await list_sn_messages_func()
    messages_list_df__recent = messages_list_df[messages_list_df['timestamp'] > (datetime.now() - timedelta(minutes=k))]
    if message_type == 'all':
        list_of_message_dicts = messages_list_df__recent[['message_body', 'message_type', 'sending_sn_pastelid', 'timestamp']].to_dict(orient='records')
    else:
        list_of_message_dicts = messages_list_df__recent[messages_list_df__recent['message_type'] == message_type][['message_body', 'message_type', 'sending_sn_pastelid', 'timestamp']].to_dict(orient='records')
    if len(list_of_message_dicts) > 0:
        return [
            {
                'message': json.loads(msg['message_body'])['message'],  # Extract the 'message' field as a string
                'message_type': msg['message_type'],
                'sending_sn_pastelid': msg['sending_sn_pastelid'],
                'timestamp': msg['timestamp'].isoformat()  # Convert timestamp to ISO format
            }
            for msg in list_of_message_dicts
        ]
    else:
        return []

async def verify_received_message_using_pastelid_func(message_received, sending_sn_pastelid):
    try:
        decompressed_message = await decompress_data_with_zstd_func(message_received)
        message_received_dict = json.loads(decompressed_message)
        raw_message = message_received_dict['message']
        signature = message_received_dict['signature']
        verification_status = await verify_message_with_pastelid_func(sending_sn_pastelid, raw_message, signature)
    except Exception as e:
        logger.error(f"Error verifying message: {e}")
        verification_status = f"Message verification failed: {str(e)}"
    return verification_status

async def send_message_to_sn_using_pastelid_func(message_to_send, message_type, receiving_sn_pastelid, pastelid_passphrase):
    global rpc_connection
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    sending_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    sending_sn_pubkey = local_machine_supernode_data['pubkey'].values.tolist()[0]
    pastelid_signature_on_message = await sign_message_with_pastelid_func(sending_sn_pastelid, message_to_send, pastelid_passphrase)
    signed_message_to_send = json.dumps({
        'message': message_to_send,
        'message_type': message_type,
        'signature': pastelid_signature_on_message,
        'sending_sn_pubkey': sending_sn_pubkey
    }, ensure_ascii=False)    
    compressed_message, _ = await compress_data_with_zstd_func(signed_message_to_send.encode('utf-8'))
    compressed_message_base64 = base64.b64encode(compressed_message).decode('utf-8')
    specified_machine_supernode_data = await get_sn_data_from_pastelid_func(receiving_sn_pastelid)
    receiving_sn_pubkey = specified_machine_supernode_data['pubkey'].values.tolist()[0]
    logger.info(f"Now sending message to SN with PastelID: {receiving_sn_pastelid} and SN pubkey: {receiving_sn_pubkey}: {message_to_send}")
    await rpc_connection.masternode('message','send', receiving_sn_pubkey, compressed_message_base64)
    return signed_message_to_send, pastelid_signature_on_message

async def broadcast_message_to_list_of_sns_using_pastelid_func(message_to_send, message_type, list_of_receiving_sn_pastelids, pastelid_passphrase, verbose=0):
    global rpc_connection
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    sending_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    sending_sn_pubkey = local_machine_supernode_data['pubkey'].values.tolist()[0]
    pastelid_signature_on_message = await sign_message_with_pastelid_func(sending_sn_pastelid, message_to_send, pastelid_passphrase)
    signed_message_to_send = json.dumps({
        'message': message_to_send,
        'message_type': message_type,
        'signature': pastelid_signature_on_message,
        'sending_sn_pubkey': sending_sn_pubkey
    }, ensure_ascii=False)
    compressed_message, _ = await compress_data_with_zstd_func(signed_message_to_send.encode('utf-8'))
    compressed_message_base64 = base64.b64encode(compressed_message).decode('utf-8')
    if verbose:
        logger.info(f"Now sending message to list of {len(list_of_receiving_sn_pastelids)} SNs: `{message_to_send}`")        
    async def send_message(receiving_sn_pastelid):
        current_receiving_sn_pubkey = (await get_sn_data_from_pastelid_func(receiving_sn_pastelid))['pubkey'].values.tolist()[0]
        await rpc_connection.masternode('message','send', current_receiving_sn_pubkey, compressed_message_base64)
    await asyncio.gather(*[send_message(pastelid) for pastelid in list_of_receiving_sn_pastelids])
    return signed_message_to_send

async def broadcast_message_to_all_sns_using_pastelid_func(message_to_send, message_type, pastelid_passphrase, verbose=0):
    global rpc_connection
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    sending_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    sending_sn_pubkey = local_machine_supernode_data['pubkey'].values.tolist()[0]
    pastelid_signature_on_message = await sign_message_with_pastelid_func(sending_sn_pastelid, message_to_send, pastelid_passphrase)
    signed_message_to_send = json.dumps({
        'message': message_to_send,
        'message_type': message_type,
        'signature': pastelid_signature_on_message,
        'sending_sn_pubkey': sending_sn_pubkey
    }, ensure_ascii=False)
    compressed_message, _ = await compress_data_with_zstd_func(signed_message_to_send.encode('utf-8'))
    compressed_message_base64 = base64.b64encode(compressed_message).decode('utf-8')
    list_of_receiving_sn_pastelids = (await check_supernode_list_func())[0]['extKey'].values.tolist()
    if verbose:
        logger.info(f"Now sending message to ALL {len(list_of_receiving_sn_pastelids)} SNs: `{message_to_send}`")        
    async def send_message(receiving_sn_pastelid):
        current_receiving_sn_pubkey = (await get_sn_data_from_pastelid_func(receiving_sn_pastelid))['pubkey'].values.tolist()[0]
        await rpc_connection.masternode('message','send', current_receiving_sn_pubkey, compressed_message_base64)
    await asyncio.gather(*[send_message(pastelid) for pastelid in list_of_receiving_sn_pastelids])
    return signed_message_to_send

async def get_supernode_model_menu(supernode_url):
    try:
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.get(f"{supernode_url}/get_inference_model_menu")
            response.raise_for_status()
            model_menu = response.json()
            return model_menu
    except Exception as e:
        logger.error(f"Error retrieving model menu from Supernode URL: {supernode_url}: {e}")
        return None

def is_model_supported(model_menu, desired_model_canonical_string, desired_model_inference_type_string, desired_model_parameters_json):
    if model_menu:
        desired_parameters = json.loads(desired_model_parameters_json)
        # Use fuzzy string matching to find the closest matching model name
        model_names = [model["model_name"] for model in model_menu["models"]]
        best_match = process.extractOne(desired_model_canonical_string, model_names)
        if best_match is not None and best_match[1] >= 95:  # Adjust the threshold as needed
            matched_model = next(model for model in model_menu["models"] if model["model_name"] == best_match[0])
            if "supported_inference_type_strings" in matched_model.keys() and "model_parameters" in matched_model.keys():
                if desired_model_inference_type_string in matched_model["supported_inference_type_strings"]:
                    # Check if all desired parameters are supported
                    for desired_param, desired_value in desired_parameters.items():
                        param_found = False
                        for param in matched_model["model_parameters"]:
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
                            return False
                    return True
            return False
    return False
        
async def broadcast_message_to_n_closest_supernodes_to_given_pastelid(input_pastelid, message_body, message_type):
    supernode_list_df, _ = await check_supernode_list_func()
    n = 4
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    local_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    message_body_dict = json.loads(message_body) # Extract the desired model information from the message_body
    desired_model_canonical_string = message_body_dict.get('requested_model_canonical_string')
    desired_model_inference_type_string = message_body_dict.get('model_inference_type_string')
    desired_model_parameters_json = message_body_dict.get('model_parameters_json')
    async def is_model_supported_async(supernode_ip_and_port, model_canonical_string, model_inference_type_string, model_parameters_json): # Filter supernodes based on model support
        supernode_ip = supernode_ip_and_port.split(':')[0]
        supernode_url = f"http://{supernode_ip}:7123"
        model_menu = await get_supernode_model_menu(supernode_url)
        return is_model_supported(model_menu, model_canonical_string, model_inference_type_string, model_parameters_json)
    supported_supernodes_coroutines = [is_model_supported_async(row['ipaddress:port'], desired_model_canonical_string, desired_model_inference_type_string, desired_model_parameters_json) for _, row in supernode_list_df.iterrows()]
    supported_supernodes_mask = await asyncio.gather(*supported_supernodes_coroutines)
    supported_supernodes = supernode_list_df[supported_supernodes_mask]
    supported_supernodes_minus_this_supernode = supported_supernodes[supported_supernodes['extKey'] != local_sn_pastelid]
    if len(supported_supernodes_minus_this_supernode) == 0:
        logger.error(f"No other supported supernodes found for the desired model: {desired_model_canonical_string} with inference type: {desired_model_inference_type_string} and parameters: {desired_model_parameters_json}")
        supported_supernodes_minus_this_supernode = supernode_list_df[supernode_list_df['extKey'] != local_sn_pastelid]
        logger.info("We had to choose audit supernodes which cannot process the request themselves if needed!")
    # Get the closest supernodes from the supported supernodes
    closest_supernodes = await get_n_closest_supernodes_to_pastelid_urls(n, input_pastelid, supported_supernodes_minus_this_supernode)
    list_of_supernode_pastelids = [x[1] for x in closest_supernodes]
    list_of_supernode_urls = [x[0] for x in closest_supernodes]
    list_of_supernode_ips = [x.split('//')[1].split(':')[0] for x in list_of_supernode_urls]
    signed_message = await broadcast_message_to_list_of_sns_using_pastelid_func(message_body, message_type, list_of_supernode_pastelids, LOCAL_PASTEL_ID_PASSPHRASE)
    logger.info(f"Broadcasted a {message_type} to {len(list_of_supernode_pastelids)} closest supernodes to PastelID: {input_pastelid} (with Supernode IPs of {list_of_supernode_ips}): {message_body}")
    return signed_message

async def retry_on_database_locked(func, *args, max_retries=5, initial_delay=1, backoff_factor=2, jitter_factor=0.1, **kwargs):
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                jitter = random.uniform(1 - jitter_factor, 1 + jitter_factor)
                delay *= backoff_factor * jitter
                logger.warning(f"Database locked. Retrying in {delay:.2f} second(s)...")
                await asyncio.sleep(delay)
            else:
                raise
        except InvalidRequestError as e:
            if "This Session's transaction has been rolled back due to a previous exception during flush" in str(e):
                logger.warning("Session transaction has been rolled back. Retrying...")
                db_session = args[0]  # Assuming the first argument is the db_session
                await db_session.rollback()
            else:
                raise

async def process_broadcast_messages(message, db_session):
    message_body = json.loads(message.message_body)
    if message.message_type == 'inference_request_response_announcement_message':
        response_data = json.loads(message_body['message'])
        usage_request = db_code.InferenceAPIUsageRequest(
            inference_request_id=response_data['inference_request_id'],
            requesting_pastelid=response_data['requesting_pastelid'],
            credit_pack_ticket_pastel_txid=response_data['credit_pack_ticket_pastel_txid'],
            requested_model_canonical_string=response_data['requested_model_canonical_string'],
            model_inference_type_string=response_data['model_inference_type_string'],
            model_parameters_json=response_data['model_parameters_json'],
            model_input_data_json_b64=response_data['model_input_data_json_b64'],
        )
        usage_response = db_code.InferenceAPIUsageResponse(
            inference_response_id=response_data['inference_response_id'],
            inference_request_id=response_data['inference_request_id'],
            proposed_cost_of_request_in_inference_credits=response_data['proposed_cost_of_request_in_inference_credits'],
            remaining_credits_in_pack_after_request_processed=response_data['remaining_credits_in_pack_after_request_processed'],
            credit_usage_tracking_psl_address=response_data['credit_usage_tracking_psl_address'],
            request_confirmation_message_amount_in_patoshis=response_data['request_confirmation_message_amount_in_patoshis'],
            max_block_height_to_include_confirmation_transaction=response_data['max_block_height_to_include_confirmation_transaction'],
            supernode_pastelid_and_signature_on_inference_response_id=response_data['supernode_pastelid_and_signature_on_inference_response_id']
        )
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before adding and committing
        await retry_on_database_locked(db_session.add, usage_request)
        await retry_on_database_locked(db_session.add, usage_response)
        await retry_on_database_locked(db_session.commit)
        await retry_on_database_locked(db_session.refresh, usage_request)
        await retry_on_database_locked(db_session.refresh, usage_response)
    elif message.message_type == 'inference_request_result_announcement_message':
        result_data = json.loads(message_body['message'])
        output_result = db_code.InferenceAPIOutputResult(
            inference_result_id=result_data['inference_result_id'],
            inference_request_id=result_data['inference_request_id'],
            inference_response_id=result_data['inference_response_id'],
            responding_supernode_pastelid=result_data['responding_supernode_pastelid'],
            inference_result_json_base64=result_data['inference_result_json_base64'],
            inference_result_file_type_strings=result_data['inference_result_file_type_strings'],
            responding_supernode_signature_on_inference_result_id=result_data['responding_supernode_signature_on_inference_result_id']
        )
        await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before adding and committing
        await retry_on_database_locked(db_session.add, output_result)
        await retry_on_database_locked(db_session.commit)
        await retry_on_database_locked(db_session.refresh, output_result)
        
async def monitor_new_messages():
    last_processed_timestamp = None
    while True:
        try:
            async with db_code.Session() as db:
                if last_processed_timestamp is None:
                    result = await db.exec(select(db_code.Message.timestamp).order_by(db_code.Message.timestamp.desc()).limit(1))
                    last_processed_timestamp_raw = result.one_or_none()
                    if last_processed_timestamp_raw is None:
                        last_processed_timestamp = pd.Timestamp.min
                    else:
                        last_processed_timestamp = pd.Timestamp(last_processed_timestamp_raw[0])
                new_messages_df = await list_sn_messages_func()
                if new_messages_df is not None and not new_messages_df.empty:
                    new_messages_df = new_messages_df[new_messages_df['timestamp'] > last_processed_timestamp]
                    if not new_messages_df.empty:
                        for _, message in new_messages_df.iterrows():
                            result = await db.exec(
                                select(db_code.Message).where(
                                    db_code.Message.sending_sn_pastelid == message['sending_sn_pastelid'],
                                    db_code.Message.receiving_sn_pastelid == message['receiving_sn_pastelid'],
                                    db_code.Message.timestamp == message['timestamp']
                                )
                            )
                            existing_message = result.one_or_none()
                            if existing_message:
                                continue
                            log_action_with_payload("Received new", "message", result)
                            last_processed_timestamp = message['timestamp']
                            sending_sn_pastelid = message['sending_sn_pastelid']
                            receiving_sn_pastelid = message['receiving_sn_pastelid']
                            message_size_bytes = len(message['message_body'].encode('utf-8'))
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before updating metadata                            
                            # Update MessageSenderMetadata
                            result = await db.exec(
                                select(db_code.MessageSenderMetadata).where(db_code.MessageSenderMetadata.sending_sn_pastelid == sending_sn_pastelid)
                            )
                            sender_metadata = result.one_or_none()
                            if sender_metadata:
                                sender_metadata.total_messages_sent += 1
                                sender_metadata.total_data_sent_bytes += message_size_bytes
                                sender_metadata.sending_sn_txid_vout = message['sending_sn_txid_vout']
                                sender_metadata.sending_sn_pubkey = message['signature']
                            else:
                                sender_metadata = db_code.MessageSenderMetadata(
                                    sending_sn_pastelid=sending_sn_pastelid,
                                    total_messages_sent=1,
                                    total_data_sent_bytes=message_size_bytes,
                                    sending_sn_txid_vout=message['sending_sn_txid_vout'],
                                    sending_sn_pubkey=message['signature']
                                )
                                db.add(sender_metadata)
                            # Update MessageReceiverMetadata
                            result = await db.exec(
                                select(db_code.MessageReceiverMetadata).where(db_code.MessageReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid)
                            )
                            receiver_metadata = result.one_or_none()
                            if receiver_metadata:
                                receiver_metadata.total_messages_received += 1
                                receiver_metadata.total_data_received_bytes += message_size_bytes
                                receiver_metadata.receiving_sn_txid_vout = message['receiving_sn_txid_vout']
                            else:
                                receiver_metadata = db_code.MessageReceiverMetadata(
                                    receiving_sn_pastelid=receiving_sn_pastelid,
                                    total_messages_received=1,
                                    total_data_received_bytes=message_size_bytes,
                                    receiving_sn_txid_vout=message['receiving_sn_txid_vout']
                                )
                                db.add(receiver_metadata)
                            # Update MessageSenderReceiverMetadata
                            result = await db.exec(
                                select(db_code.MessageSenderReceiverMetadata).where(
                                    db_code.MessageSenderReceiverMetadata.sending_sn_pastelid == sending_sn_pastelid,
                                    db_code.MessageSenderReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid
                                )
                            )
                            sender_receiver_metadata = result.one_or_none()
                            if sender_receiver_metadata:
                                sender_receiver_metadata.total_messages += 1
                                sender_receiver_metadata.total_data_bytes += message_size_bytes
                            else:
                                sender_receiver_metadata = db_code.MessageSenderReceiverMetadata(
                                    sending_sn_pastelid=sending_sn_pastelid,
                                    receiving_sn_pastelid=receiving_sn_pastelid,
                                    total_messages=1,
                                    total_data_bytes=message_size_bytes
                                )
                                db.add(sender_receiver_metadata)
                            new_messages = [
                                db_code.Message(
                                    sending_sn_pastelid=row['sending_sn_pastelid'],
                                    receiving_sn_pastelid=row['receiving_sn_pastelid'],
                                    message_type=row['message_type'],
                                    message_body=row['message_body'],
                                    signature=row['signature'],
                                    timestamp=row['timestamp'],
                                    sending_sn_txid_vout=row['sending_sn_txid_vout'],
                                    receiving_sn_txid_vout=row['receiving_sn_txid_vout']
                                )
                                for _, row in new_messages_df.iterrows()
                            ]
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before adding messages                            
                            await retry_on_database_locked(db.add_all, new_messages)
                            await retry_on_database_locked(db.commit)  # Commit the transaction for adding new messages
                            # Update overall MessageMetadata
                            result = await db.exec(
                                select(
                                    func.count(db_code.Message.id),
                                    func.count(func.distinct(db_code.Message.sending_sn_pastelid)),
                                    func.count(func.distinct(db_code.Message.receiving_sn_pastelid))
                                )
                            )
                            total_messages, total_senders, total_receivers = result.one()
                            result = await db.exec(select(db_code.MessageMetadata).order_by(db_code.MessageMetadata.timestamp.desc()).limit(1))
                            message_metadata = result.one_or_none()
                            if message_metadata:
                                message_metadata.total_messages = total_messages
                                message_metadata.total_senders = total_senders
                                message_metadata.total_receivers = total_receivers
                            else:
                                message_metadata = db_code.MessageMetadata(
                                    total_messages=total_messages,
                                    total_senders=total_senders,
                                    total_receivers=total_receivers
                                )
                                db.add(message_metadata)
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before updating message metadata                                
                            await retry_on_database_locked(db.commit)
                            # Process broadcast messages concurrently
                            processing_tasks = [
                                process_broadcast_messages(message, db)
                                for message in new_messages
                            ]
                            await asyncio.gather(*processing_tasks)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while monitoring new messages: {str(e)}")
            await asyncio.sleep(5)
        finally:
            await asyncio.sleep(5)            
            
async def create_user_message(from_pastelid: str, to_pastelid: str, message_body: str, message_signature: str) -> dict:
    async with db_code.Session() as db:
        user_message = db_code.UserMessage(from_pastelid=from_pastelid, to_pastelid=to_pastelid, message_body=message_body, message_signature=message_signature)
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        return user_message.model_dump()

async def create_supernode_user_message(sending_sn_pastelid: str, receiving_sn_pastelid: str, user_message_data: dict) -> dict:
    async with db_code.Session() as db:
        supernode_user_message = db_code.SupernodeUserMessage(
            message_body=user_message_data['message_body'],
            message_type="user_message",
            sending_sn_pastelid=sending_sn_pastelid,
            receiving_sn_pastelid=receiving_sn_pastelid,
            signature=user_message_data['message_signature'],
            timestamp=datetime.utcnow(),
            user_message_id=user_message_data['id']
        )
        db.add(supernode_user_message)
        db.commit()
        db.refresh(supernode_user_message)
        return supernode_user_message.model_dump()

async def send_user_message_via_supernodes(from_pastelid: str, to_pastelid: str, message_body: str, message_signature: str) -> dict:
    user_message_data = await create_user_message(from_pastelid, to_pastelid, message_body, message_signature)
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    sending_sn_pastelid = local_machine_supernode_data['extKey'][0]  # Assuming this is a list
    # Find the 3 closest Supernodes to the receiving end user's PastelID
    supernode_list_df, _ = await check_supernode_list_func()
    closest_supernodes = await get_n_closest_supernodes_to_pastelid_urls(3, to_pastelid, supernode_list_df)
    if not closest_supernodes:
        raise ValueError(f"No Supernodes found for PastelID: {to_pastelid}.")
    # Create a list to store the message_dicts for each Supernode
    message_dicts = []
    # Send the message to the 3 closest Supernodes in parallel using asyncio.gather
    send_tasks = []
    for closest_supernode_url, receiving_sn_pastelid in closest_supernodes:
        # Now that we have user_message_data and the receiving Supernode PastelID, let's create the supernode user message.
        supernode_user_message_data = await create_supernode_user_message(sending_sn_pastelid, receiving_sn_pastelid, user_message_data)
        # Preparing the message to be sent to the supernode.
        signed_message_to_send = json.dumps({
            'message': user_message_data['message_body'],
            'message_type': 'user_message',
            'signature': user_message_data['message_signature'],
            'from_pastelid': user_message_data['from_pastelid'],
            'to_pastelid': user_message_data['to_pastelid']
        }, ensure_ascii=False)
        # Send the message to the receiving Supernode.
        send_task = asyncio.create_task(send_message_to_sn_using_pastelid_func(signed_message_to_send, 'user_message', receiving_sn_pastelid, LOCAL_PASTEL_ID_PASSPHRASE))
        send_tasks.append(send_task)
    # Wait for all send tasks to complete
    send_results = await asyncio.gather(*send_tasks)
    # Create the message_dict for each Supernode
    for send_result in send_results:
        signed_message, pastelid_signature_on_message = send_result
        message_dict = {
            "message": user_message_data['message_body'],  # The content of the message
            "message_type": "user_message",  # Static type as per your design
            "sending_sn_pastelid": sending_sn_pastelid,  # From local machine supernode data
            "timestamp": datetime.now(dt.UTC).isoformat(),  # Current UTC timestamp
            "id": supernode_user_message_data['id'],  # ID from the supernode user message record
            "signature": pastelid_signature_on_message,
            "user_message": {
                # Details from the user_message_data
                "from_pastelid": from_pastelid,
                "to_pastelid": to_pastelid,
                "message_body": message_body,
                "message_signature": user_message_data['message_signature'],
                "id": user_message_data['id'],  # Assuming these fields are included in your dictionary
                "timestamp": user_message_data['timestamp']
            }
        }
        message_dicts.append(message_dict)
    return message_dicts

async def process_received_user_message(supernode_user_message: db_code.SupernodeUserMessage):
    async with db_code.Session() as db:
        user_message = db.exec(select(db_code.UserMessage).where(db_code.UserMessage.id == supernode_user_message.user_message_id)).one_or_none()
        if user_message:
            verification_status = await verify_message_with_pastelid_func(user_message.from_pastelid, user_message.message_body, user_message.message_signature)
            if verification_status == 'OK':
                # Process the user message (e.g., store it, forward it to the recipient, etc.)
                logger.info(f"Received and verified user message from {user_message.from_pastelid} to {user_message.to_pastelid}")
            else:
                logger.warning(f"Received user message from {user_message.from_pastelid} to {user_message.to_pastelid}, but verification failed")
        else:
            logger.warning(f"Received SupernodeUserMessage (id: {supernode_user_message.id}), but the associated UserMessage was not found")

async def get_user_messages_for_pastelid(pastelid: str) -> List[db_code.UserMessage]:
    async with db_code.Session() as db:
        user_messages = db.exec(select(db_code.UserMessage).where((db_code.UserMessage.from_pastelid == pastelid) | (db_code.UserMessage.to_pastelid == pastelid))).all()
        return user_messages
            
#________________________________________________________________________________________________________________            
# Credit pack related service functions:

async def get_credit_pack_purchase_request(sha3_256_hash_of_credit_pack_purchase_request_fields: str) -> db_code.CreditPackPurchaseRequest:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequest).where(db_code.CreditPackPurchaseRequest.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request: {str(e)}")
        raise

async def save_credit_pack_purchase_request(credit_pack_purchase_request: db_code.CreditPackPurchaseRequest) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request: {str(e)}")
        raise

async def get_credit_pack_purchase_request_response(sha3_256_hash_of_credit_pack_purchase_request_response_fields: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponse).where(db_code.CreditPackPurchaseRequestResponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields == sha3_256_hash_of_credit_pack_purchase_request_response_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request response: {str(e)}")
        raise

async def get_credit_pack_purchase_request_from_response(response: db_code.CreditPackPurchaseRequestResponse) -> Optional[db_code.CreditPackPurchaseRequest]:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequest).where(db_code.CreditPackPurchaseRequest.sha3_256_hash_of_credit_pack_purchase_request_fields == response.sha3_256_hash_of_credit_pack_purchase_request_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request response: {str(e)}")
        raise    
    
async def save_credit_pack_purchase_request_response(credit_pack_purchase_request_response: db_code.CreditPackPurchaseRequestResponse) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_response)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request response: {str(e)}")
        raise

async def get_credit_pack_purchase_request_rejection(sha3_256_hash_of_credit_pack_purchase_request_fields: str) -> db_code.CreditPackPurchaseRequestRejection:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestRejection).where(db_code.CreditPackPurchaseRequestRejection.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request rejection: {str(e)}")
        raise

async def save_credit_pack_purchase_request_rejection(credit_pack_purchase_request_rejection: db_code.CreditPackPurchaseRequestRejection) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_rejection)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request rejection: {str(e)}")
        raise

async def get_credit_pack_purchase_request_response_termination(sha3_256_hash_of_credit_pack_purchase_request_fields: str) -> db_code.CreditPackPurchaseRequestResponseTermination:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponseTermination).where(db_code.CreditPackPurchaseRequestResponseTermination.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request response termination: {str(e)}")
        raise

async def save_credit_pack_purchase_request_response_termination(credit_pack_purchase_request_response_termination: db_code.CreditPackPurchaseRequestResponseTermination) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_response_termination)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request response termination: {str(e)}")
        raise

async def get_credit_pack_purchase_request_confirmation(sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str) -> db_code.CreditPackPurchaseRequestConfirmation:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation).where(db_code.CreditPackPurchaseRequestConfirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields == sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request confirmation: {str(e)}")
        raise

async def save_credit_pack_purchase_request_confirmation(credit_pack_purchase_request_confirmation: db_code.CreditPackPurchaseRequestConfirmation) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_confirmation)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request confirmation: {str(e)}")
        raise

async def get_credit_pack_purchase_request_confirmation_response(sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields: str) -> db_code.CreditPackPurchaseRequestConfirmationResponse:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmationResponse).where(db_code.CreditPackPurchaseRequestConfirmationResponse.sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields == sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields)
            )
            return result.one_or_none()
    except Exception as e:
        logger.error(f"Error getting credit pack purchase request confirmation response: {str(e)}")
        raise

async def save_credit_pack_purchase_request_confirmation_response(credit_pack_purchase_request_confirmation_response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_confirmation_response)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request confirmation response: {str(e)}")
        raise
    
async def save_credit_pack_purchase_preliminary_price_quote(credit_pack_purchase_preliminary_price_quote: db_code.CreditPackPurchaseRequestPreliminaryPriceQuote) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_preliminary_price_quote)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase preliminary price quote: {str(e)}")
        raise

async def save_credit_pack_purchase_price_agreement_request(credit_pack_purchase_price_agreement_request: db_code.CreditPackPurchasePriceAgreementRequest) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_price_agreement_request)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase price agreement request: {str(e)}")
        raise

async def save_credit_pack_purchase_price_agreement_request_response(credit_pack_purchase_price_agreement_request_response: db_code.CreditPackPurchasePriceAgreementRequestResponse) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_price_agreement_request_response)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase price agreement request response: {str(e)}")
        raise    
    
async def save_credit_pack_purchase_request_status_check(credit_pack_purchase_request_status_check: db_code.CreditPackPurchaseRequestStatus) -> None:
    try:
        async with db_code.Session() as db_session:
            db_session.add(credit_pack_purchase_request_status_check)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request status check: {str(e)}")
        raise    

async def save_credit_pack_purchase_request_final_response(response: db_code.CreditPackPurchaseRequestResponse) -> None:
    try:
        async with db_code.Session() as db:
            db.add(response)
            await db.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request final response: {str(e)}")
        raise

async def save_credit_pack_storage_completion_announcement(response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        async with db_code.Session() as db:
            db.add(response)
            await db.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack storage completion announcement: {str(e)}")
        raise
        
async def save_credit_pack_purchase_completion_announcement(confirmation: db_code.CreditPackPurchaseRequestConfirmation) -> None:
    try:
        async with db_code.Session() as db:
            db.add(confirmation)
            await db.commit()
    except Exception as e:
        logger.error(f"Error storing credit pack purchase completion announcement: {str(e)}")
        raise
    
async def save_credit_pack_storage_retry_request(response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        async with db_code.Session() as db:
            db.add(response)
            await db.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack storage retry request: {str(e)}")
        raise
    
async def save_credit_pack_storage_retry_request_response(response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        async with db_code.Session() as db:
            db.add(response)
            await db.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack storage retry request response: {str(e)}")
        raise

async def save_credit_pack_storage_retry_completion_announcement(response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        async with db_code.Session() as db:
            db.add(response)
            await db.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack storage retry completion announcement: {str(e)}")
        raise
            
async def check_credit_pack_purchase_request_status(credit_pack_purchase_request: db_code.CreditPackPurchaseRequest) -> str:
    async with db_code.Session() as db:
        response = await db.exec(
            select(db_code.CreditPackPurchaseRequestResponse).where(db_code.CreditPackPurchaseRequestResponse.sha3_256_hash_of_credit_pack_purchase_request_fields == credit_pack_purchase_request.sha3_256_hash_of_credit_pack_purchase_request_fields)
        )
        response = response.one_or_none()
        if response is None:
            return "pending"
        else:
            confirmation = await db.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation).where(db_code.CreditPackPurchaseRequestConfirmation.sha3_256_hash_of_credit_pack_purchase_request_response_fields == response.sha3_256_hash_of_credit_pack_purchase_request_response_fields)
            )
            confirmation = confirmation.one_or_none()
            if confirmation is None:
                return "approved"
            else:
                confirmation_response = await db.exec(
                    select(db_code.CreditPackPurchaseRequestConfirmationResponse).where(db_code.CreditPackPurchaseRequestConfirmationResponse.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields == confirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields)
                )
                confirmation_response = confirmation_response.one_or_none()
                if confirmation_response is None:
                    return "confirmed"
                else:
                    if confirmation_response.credit_pack_confirmation_outcome_string == "success":
                        return "completed"
                    else:
                        return "failed"
                    
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

async def calculate_preliminary_psl_price_per_credit():
    try:
        # Get the current PSL market price in USD
        psl_price_usd = await fetch_current_psl_market_price()
        # Calculate the cost per credit in USD, considering the profit margin
        cost_per_credit_usd = TARGET_VALUE_PER_CREDIT_IN_USD / (1 - TARGET_PROFIT_MARGIN)
        # Convert the cost per credit from USD to PSL
        cost_per_credit_psl = cost_per_credit_usd / psl_price_usd
        # Round the cost per credit to the nearest 0.1
        rounded_cost_per_credit_psl = round(cost_per_credit_psl, 1)
        logger.info(f"Calculated preliminary price per credit: {rounded_cost_per_credit_psl:.1f} PSL")
        return rounded_cost_per_credit_psl
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error calculating preliminary price per credit: {str(e)}")
        raise

async def determine_agreement_with_proposed_price(proposed_psl_price_per_credit: float) -> bool:
    try:
        # Calculate the local preliminary price per credit
        local_price_per_credit = await calculate_preliminary_psl_price_per_credit()
        # Calculate the acceptable price range (within 10% of the local price)
        min_acceptable_price = local_price_per_credit * (1.0 - MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING)
        max_acceptable_price = local_price_per_credit * (1.0 + MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING)
        # Determine if the proposed price is within the acceptable range
        agree_with_proposed_price = min_acceptable_price <= proposed_psl_price_per_credit <= max_acceptable_price
        logger.info(f"Proposed price per credit: {proposed_psl_price_per_credit:.1f} PSL")
        logger.info(f"Local price per credit: {local_price_per_credit:.1f} PSL")
        logger.info(f"Acceptable price range: [{min_acceptable_price:.1f}, {max_acceptable_price:.1f}] PSL")
        logger.info(f"Agreement with proposed price: {agree_with_proposed_price}")
        return agree_with_proposed_price
    except Exception as e:
        logger.error(f"Error determining agreement with proposed price: {str(e)}")
        raise
    
async def check_burn_transaction(txid: str, credit_usage_tracking_psl_address: str, total_cost_in_psl: float, request_response_pastel_block_height: int) -> Tuple[bool, int]:
    try:
        max_block_height = request_response_pastel_block_height + MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET
        max_retries = 30
        initial_retry_delay_in_seconds = 30
        total_cost_in_psl = round(total_cost_in_psl,5)
        matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_address_for_tracking_transaction(
            burn_address,
            credit_usage_tracking_psl_address, 
            total_cost_in_psl,
            txid,
            max_block_height,
            max_retries,
            initial_retry_delay_in_seconds
        )
        return matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address
    except Exception as e:
        logger.error(f"Error checking burn transaction: {str(e)}")
        raise
    
async def save_credit_pack_purchase_request_response_txid_mapping(credit_pack_purchase_request_response: db_code.CreditPackPurchaseRequestResponse, txid: str) -> None:
    try:
        mapping = db_code.CreditPackPurchaseRequestResponseTxidMapping(
            sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
            pastel_api_credit_pack_ticket_registration_txid=txid
        )
        async with db_code.Session() as db_session:
            db_session.add(mapping)
            await db_session.commit()
    except Exception as e:
        logger.error(f"Error saving credit pack purchase request response txid mapping: {str(e)}")
        raise
        
async def retrieve_credit_pack_ticket_from_blockchain_using_txid(txid: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        retrieved_data = await retrieve_data_from_blockchain(txid)
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse.parse_raw(retrieved_data)
        return credit_pack_purchase_request_response
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket from blockchain: {str(e)}")
        raise
    
async def retrieve_credit_pack_ticket_using_txid(txid: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        # Try to retrieve the credit pack ticket from the local database using the txid mapping
        async with db_code.Session() as db_session:
            mapping = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponseTxidMapping).where(db_code.CreditPackPurchaseRequestResponseTxidMapping.pastel_api_credit_pack_ticket_registration_txid == txid)
            ).one_or_none()
            if mapping is not None:
                credit_pack_purchase_request_response = await db_session.get(db_code.CreditPackPurchaseRequestResponse, mapping.credit_pack_purchase_request_response_id)
                return credit_pack_purchase_request_response
        # If the ticket is not found in the local database, retrieve it from the blockchain
        credit_pack_purchase_request_response = await retrieve_credit_pack_ticket_from_blockchain_using_txid(txid)
        if credit_pack_purchase_request_response is not None:
            # Save the retrieved ticket to the local database for future reference
            try:
                async with db_code.Session() as db_session:
                    db_session.add(credit_pack_purchase_request_response)
                    await db_session.commit()
                    await db_session.refresh(credit_pack_purchase_request_response)
            except Exception as e:
                logger.error(f"Error saving retrieved credit pack ticket to the local database: {str(e)}")
            # Save the txid mapping for the retrieved ticket
            try:
                await save_credit_pack_purchase_request_response_txid_mapping(credit_pack_purchase_request_response, txid)
            except Exception as e:
                logger.error(f"Error saving txid mapping for retrieved credit pack ticket: {str(e)}")
        else:
            raise ValueError(f"Credit pack ticket not found for txid: {txid}")
        return credit_pack_purchase_request_response
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket using txid: {txid}. Error: {str(e)}")
        raise

async def store_credit_pack_ticket_in_blockchain(credit_pack_purchase_request_response_json: str) -> str:
    try:
        logger.info("Now attempting to write the ticket data to the blockchain...")
        credit_pack_ticket_txid, total_bytes_used = await store_data_in_blockchain(credit_pack_purchase_request_response_json)
        logger.info(f"Received back pastel txid of {credit_pack_ticket_txid} for the stored blockchain ticket data; total bytes used to store the data was {total_bytes_used:,}; now waiting for the transaction to be confirmed...")
        max_retries = 20
        retry_delay = 20
        try_count = 0
        num_confirmations = 0
        storage_validation_error_string = ""
        while try_count < max_retries and num_confirmations == 0:
            # Retrieve the transaction details using gettransaction RPC method
            tx_info = await rpc_connection.gettransaction(credit_pack_ticket_txid)
            if tx_info:
                num_confirmations = tx_info.get("confirmations", 0)
                if num_confirmations > 0:
                    logger.info(f"Transaction {credit_pack_ticket_txid} has been confirmed with {num_confirmations} confirmations.")
                    break
                else:
                    logger.info(f"Transaction {credit_pack_ticket_txid} is not yet confirmed. Waiting for {retry_delay:.2f} seconds before checking again.")
                    await asyncio.sleep(retry_delay)
                    try_count += 1
                    retry_delay *= 1.15  # Optional: increase delay between retries
            else:
                logger.warning(f"Transaction {credit_pack_ticket_txid} not found. Waiting for {retry_delay} seconds before checking again.")
                await asyncio.sleep(retry_delay)
                try_count += 1
                retry_delay *= 1.15  # Optional: increase delay between retries
        if num_confirmations > 0:
            logger.info("Now verifying that we can reconstruct the original file written exactly...")
            reconstructed_file_data = await retrieve_data_from_blockchain(credit_pack_ticket_txid)
            decoded_reconstructed_file_data = reconstructed_file_data.decode('utf-8')
            if decoded_reconstructed_file_data == credit_pack_purchase_request_response_json:
                logger.info("Successfully verified that the stored blockchain ticket data can be reconstructed exactly!")
                use_test_reconstruction_of_object_from_json = 0
                if use_test_reconstruction_of_object_from_json:
                    credit_pack_purchase_request_response_json_transformed = transform_credit_pack_purchase_request_response(json.loads(credit_pack_purchase_request_response_json))
                    credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse(**credit_pack_purchase_request_response_json_transformed)
                    logger.info(f"Reconstructed credit pack ticket data: {credit_pack_purchase_request_response}")
            else:
                logger.error("Failed to verify that the stored blockchain ticket data can be reconstructed exactly!")
                storage_validation_error_string = "Failed to verify that the stored blockchain ticket data can be reconstructed exactly! Difference: " + str(set(decoded_reconstructed_file_data).symmetric_difference(set(credit_pack_purchase_request_response_json)))
                logger.error(storage_validation_error_string)
                return credit_pack_ticket_txid, storage_validation_error_string
        else:
            storage_validation_error_string = f"Transaction {credit_pack_ticket_txid} was not confirmed after {max_retries} attempts."
            logger.error(storage_validation_error_string)
            raise TimeoutError(f"Transaction {credit_pack_ticket_txid} was not confirmed after {max_retries} attempts.")
        return credit_pack_ticket_txid, storage_validation_error_string
    except Exception as e:
        storage_validation_error_string = f"Error storing credit pack ticket: {str(e)}"
        logger.error(storage_validation_error_string)
        return credit_pack_ticket_txid, storage_validation_error_string

async def check_original_supernode_storage_confirmation(sha3_256_hash_of_credit_pack_purchase_request_response_fields: str) -> bool:
    async with db_code.Session() as db:
        result = await db.exec(
            select(db_code.CreditPackPurchaseRequestConfirmationResponse).where(db_code.CreditPackPurchaseRequestConfirmationResponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields == sha3_256_hash_of_credit_pack_purchase_request_response_fields)
        )
        return result.one_or_none() is not None

async def process_credit_purchase_initial_request(credit_pack_purchase_request: db_code.CreditPackPurchaseRequest) -> db_code.CreditPackPurchaseRequestPreliminaryPriceQuote:
    try:
        # Validate the request fields
        request_validation_errors = await validate_credit_pack_ticket_message_data_func(credit_pack_purchase_request)
        if request_validation_errors:
            rejection_message = await generate_credit_pack_request_rejection_message(credit_pack_purchase_request, request_validation_errors)
            logger.error(f"Invalid credit purchase request: {', '.join(request_validation_errors)}")
            return rejection_message
        # Determine the preliminary price quote
        preliminary_quoted_price_per_credit_in_psl = await calculate_preliminary_psl_price_per_credit()
        preliminary_total_cost_of_credit_pack_in_psl = round(preliminary_quoted_price_per_credit_in_psl * credit_pack_purchase_request.requested_initial_credits_in_credit_pack, 5)
        # Create the response without the hash and signature fields
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestPreliminaryPriceQuote(
            sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_purchase_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
            credit_usage_tracking_psl_address=credit_pack_purchase_request.credit_usage_tracking_psl_address,
            credit_pack_purchase_request_fields_json=await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(credit_pack_purchase_request),
            preliminary_quoted_price_per_credit_in_psl=preliminary_quoted_price_per_credit_in_psl,
            preliminary_total_cost_of_credit_pack_in_psl=preliminary_total_cost_of_credit_pack_in_psl,
            preliminary_price_quote_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            preliminary_price_quote_pastel_block_height=await get_current_pastel_block_height_func(),
            preliminary_price_quote_message_version_string="1.0",
            responding_supernode_pastelid=MY_PASTELID,
            sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields="",
            responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash=""
        )
        # Generate the hash and signature fields
        credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(credit_pack_purchase_request_response)
        credit_pack_purchase_request_response.responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash = await sign_message_with_pastelid_func(
            MY_PASTELID,
            credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields,
            LOCAL_PASTEL_ID_PASSPHRASE
        )
        # Validate the response
        response_validation_errors = await validate_credit_pack_ticket_message_data_func(credit_pack_purchase_request_response)
        if response_validation_errors:
            logger.error(f"Invalid credit purchase request preliminary price quote: {', '.join(response_validation_errors)}")
            raise ValueError(f"Invalid credit purchase request preliminary price quote: {', '.join(response_validation_errors)}")
        await save_credit_pack_purchase_request(credit_pack_purchase_request)
        await save_credit_pack_purchase_request_response(credit_pack_purchase_request_response)        
        return credit_pack_purchase_request_response
    except Exception as e:
        logger.error(f"Error processing credit purchase initial request: {str(e)}")
        raise
    
async def generate_credit_pack_request_rejection_message(credit_pack_request: db_code.CreditPackPurchaseRequest, validation_errors: List[str]) -> db_code.CreditPackPurchaseRequestRejection:
    rejection_message = db_code.CreditPackPurchaseRequestRejection(
        sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
        credit_pack_purchase_request_fields_json=await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(credit_pack_request),
        rejection_reason_string=", ".join(validation_errors),
        rejection_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
        rejection_pastel_block_height=await get_current_pastel_block_height_func(),
        credit_purchase_request_rejection_message_version_string="1.0",
        responding_supernode_pastelid=MY_PASTELID,
        sha3_256_hash_of_credit_pack_purchase_request_rejection_fields="",
        responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash=""
    )
    rejection_message.sha3_256_hash_of_credit_pack_purchase_request_rejection_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(rejection_message)
    rejection_message.responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash = await sign_message_with_pastelid_func(
        MY_PASTELID,
        rejection_message.sha3_256_hash_of_credit_pack_purchase_request_rejection_fields,
        LOCAL_PASTEL_ID_PASSPHRASE
    )
    rejection_validation_errors = await validate_credit_pack_ticket_message_data_func(rejection_message)
    if rejection_validation_errors:
        logger.error(f"Invalid credit purchase request rejection message: {', '.join(rejection_validation_errors)}")
        raise ValueError(f"Invalid credit purchase request rejection message: {', '.join(rejection_validation_errors)}")
    await save_credit_pack_purchase_request_rejection(rejection_message)
    return rejection_message
    
async def select_potentially_agreeing_supernodes() -> List[str]:
    try:
        # Get the best block hash and merkle root
        best_block_hash, best_block_merkle_root, _ = await get_best_block_hash_and_merkle_root_func()
        # Get the list of all supernodes
        supernode_list_df, _ = await check_supernode_list_func()
        number_of_supernodes_found = len(supernode_list_df) - 1
        if number_of_supernodes_found < MAXIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES:
            logger.warning(f"Fewer than {MAXIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES} supernodes available. Using all {number_of_supernodes_found} available supernodes.")
            all_other_supernode_pastelids = [x for x in supernode_list_df['extKey'].tolist() if x != MY_PASTELID]
            return all_other_supernode_pastelids
        # Compute the XOR distance between each supernode's hash(pastelid) and the best block's merkle root
        xor_distances = []
        for _, row in supernode_list_df.iterrows():
            supernode_pastelid = row['extKey']
            supernode_pastelid_hash = get_sha256_hash_of_input_data_func(supernode_pastelid)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            merkle_root_int = int(best_block_merkle_root, 16)
            xor_distance = supernode_pastelid_int ^ merkle_root_int
            xor_distances.append((supernode_pastelid, xor_distance))
        # Sort the supernodes based on their XOR distances in ascending order
        sorted_supernodes = sorted(xor_distances, key=lambda x: x[1])
        # Select the supernodes with the closest XOR distances
        potentially_agreeing_supernodes = [supernode[0] for supernode in sorted_supernodes[:MAXIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES] if supernode[0] != MY_PASTELID]
        return potentially_agreeing_supernodes
    except Exception as e:
        logger.error(f"Error selecting potentially agreeing supernodes: {str(e)}")
        raise

async def send_price_agreement_request_to_supernodes(request: db_code.CreditPackPurchasePriceAgreementRequest, supernodes: List[str]) -> List[db_code.CreditPackPurchasePriceAgreementRequestResponse]:
    try:
        async with httpx.AsyncClient() as client:
            tasks = []
            supernode_list_df, _ = await check_supernode_list_func()
            for supernode_pastelid in supernodes:
                payload = {}
                try:
                    supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    url = f"{supernode_base_url}/credit_pack_price_agreement_request"
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                    challenge = challenge_dict["challenge"]
                    challenge_id = challenge_dict["challenge_id"]
                    challenge_signature = challenge_dict["challenge_signature"]
                    payload = {
                        "credit_pack_price_agreement_request": request.model_dump(),
                        "challenge": challenge,
                        "challenge_id": challenge_id,
                        "challenge_signature": challenge_signature
                    }     
                except Exception as e:
                    logger.warning(f"Error getting challenge from supernode {supernode_pastelid}: {str(e)}")        
                if len(payload) > 0:
                    task = asyncio.create_task(client.post(url, json=payload))
                    tasks.append(task)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            price_agreement_request_responses = []
            for response in responses:
                if isinstance(response, httpx.Response):
                    if response.status_code == 200:
                        price_agreement_request_response = db_code.CreditPackPurchasePriceAgreementRequestResponse(**response.json())
                        price_agreement_request_responses.append(price_agreement_request_response)
                    else:
                        logger.warning(f"Error sending price agreement request to supernode {response.url}: {response.text}")
                else:
                    logger.error(f"Error sending price agreement request to supernode: {str(response)}")
            return price_agreement_request_responses
    except Exception as e:
        logger.error(f"Error sending price agreement request to supernodes: {str(e)}")
        raise
    
async def request_and_sign_challenge(supernode_url: str) -> Dict[str, str]:
    async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
        response = await client.get(f"{supernode_url}/request_challenge/{MY_PASTELID}")
        response.raise_for_status()
        result = response.json()
        challenge = result["challenge"]
        challenge_id = result["challenge_id"]
        # Sign the challenge string using the local RPC client
        challenge_signature = await sign_message_with_pastelid_func(MY_PASTELID, challenge, LOCAL_PASTEL_ID_PASSPHRASE)
        return {
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }    
        
async def send_credit_pack_purchase_request_final_response_to_supernodes(response: db_code.CreditPackPurchaseRequestResponse, supernodes: List[str]) -> List[httpx.Response]:
    try:
        async with httpx.AsyncClient() as client:
            tasks = []
            supernode_list_df, _ = await check_supernode_list_func()
            for supernode_pastelid in supernodes:
                payload = {}
                try:
                    supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    url = f"{supernode_base_url}/credit_pack_purchase_request_final_response_announcement"
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                    challenge = challenge_dict["challenge"]
                    challenge_id = challenge_dict["challenge_id"]
                    challenge_signature = challenge_dict["challenge_signature"]
                    payload = {
                        "response": response.model_dump(),
                        "challenge": challenge,
                        "challenge_id": challenge_id,
                        "challenge_signature": challenge_signature
                    }
                except Exception as e:
                    logger.warning(f"Error getting challenge from supernode {supernode_pastelid}: {str(e)}")
                if len(payload) > 0:
                    task = asyncio.create_task(client.post(url, json=payload))
                    tasks.append(task)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            valid_responses = []
            for response in responses:
                if isinstance(response, httpx.Response):
                    if response.status_code == 200:
                        valid_responses.append(response)
                    else:
                        logger.warning(f"Error sending final response announcement to supernode {response.url}: {response.text}")
                else:
                    logger.error(f"Error sending final response announcement to supernode: {str(response)}")
            return valid_responses
    except Exception as e:
        logger.error(f"Error sending final response announcement to supernodes: {str(e)}")
        raise    

async def process_credit_pack_price_agreement_request(price_agreement_request: db_code.CreditPackPurchasePriceAgreementRequest) -> Union[db_code.CreditPackPurchasePriceAgreementRequestResponse, str]:
    try:
        # Validate the request fields
        request_validation_errors = await validate_credit_pack_ticket_message_data_func(price_agreement_request)
        if request_validation_errors:
            logger.error(f"Invalid price agreement request: {', '.join(request_validation_errors)}")
            return f"Invalid price agreement request: {', '.join(request_validation_errors)}"
        # Determine if the supernode agrees with the proposed price
        agree_with_proposed_price = await determine_agreement_with_proposed_price(price_agreement_request.proposed_psl_price_per_credit)
        # Create the response without the hash and signature fields
        response = db_code.CreditPackPurchasePriceAgreementRequestResponse(
            sha3_256_hash_of_price_agreement_request_fields=price_agreement_request.sha3_256_hash_of_price_agreement_request_fields,
            credit_pack_purchase_request_fields_json=price_agreement_request.credit_pack_purchase_request_fields_json,
            agree_with_proposed_price=agree_with_proposed_price,
            credit_usage_tracking_psl_address=price_agreement_request.credit_usage_tracking_psl_address,
            proposed_psl_price_per_credit=price_agreement_request.proposed_psl_price_per_credit,
            proposed_price_agreement_response_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            proposed_price_agreement_response_pastel_block_height=await get_current_pastel_block_height_func(),
            proposed_price_agreement_response_message_version_string="1.0",
            responding_supernode_signature_on_credit_pack_purchase_request_fields_json=await sign_message_with_pastelid_func(MY_PASTELID, price_agreement_request.credit_pack_purchase_request_fields_json, LOCAL_PASTEL_ID_PASSPHRASE),
            responding_supernode_pastelid=MY_PASTELID,
            sha3_256_hash_of_price_agreement_request_response_fields="",
            responding_supernode_signature_on_price_agreement_request_response_hash=""
        )
        # Generate the hash and signature fields
        response.sha3_256_hash_of_price_agreement_request_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(response)
        response.responding_supernode_signature_on_price_agreement_request_response_hash = await sign_message_with_pastelid_func(
            MY_PASTELID,
            response.sha3_256_hash_of_price_agreement_request_response_fields,
            LOCAL_PASTEL_ID_PASSPHRASE
        )
        # Validate the response
        response_validation_errors = await validate_credit_pack_ticket_message_data_func(response)
        if response_validation_errors:
            logger.error(f"Invalid price agreement request response: {', '.join(response_validation_errors)}")
            return f"Invalid price agreement request response: {', '.join(response_validation_errors)}"
        await save_credit_pack_purchase_price_agreement_request_response(response)        
        return response
    except Exception as e:
        logger.error(f"Error processing credit pack price agreement request: {str(e)}")
        raise
            
async def process_credit_purchase_preliminary_price_quote_response(preliminary_price_quote_response: db_code.CreditPackPurchaseRequestPreliminaryPriceQuoteResponse) -> Union[db_code.CreditPackPurchaseRequestResponse, db_code.CreditPackPurchaseRequestResponseTermination]:
    try:
        # Validate the response fields
        if not preliminary_price_quote_response.agree_with_preliminary_price_quote:
            logger.warning("End user does not agree with preliminary price quote! Unable to proceed with credit pack purchase.")
            raise ValueError("End user does not agree with preliminary price quote")
        response_validation_errors = await validate_credit_pack_ticket_message_data_func(preliminary_price_quote_response)
        if response_validation_errors:
            logger.error(f"Invalid preliminary price quote response: {', '.join(response_validation_errors)}")
            raise ValueError(f"Invalid preliminary price quote response: {', '.join(response_validation_errors)}")
        # Select the potentially agreeing supernodes
        logger.info("Now selecting potentially agreeing supernodes to sign off on the proposed credit pricing for the credit pack purchase request...")
        potentially_agreeing_supernodes = await select_potentially_agreeing_supernodes()
        logger.info(f"Selected {len(potentially_agreeing_supernodes)} potentially agreeing supernodes: {potentially_agreeing_supernodes}")
        # Create the price agreement request without the hash and signature fields
        price_agreement_request = db_code.CreditPackPurchasePriceAgreementRequest(
            sha3_256_hash_of_credit_pack_purchase_request_response_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields,
            supernode_requesting_price_agreement_pastelid=MY_PASTELID,
            credit_pack_purchase_request_fields_json=await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(preliminary_price_quote_response),
            credit_usage_tracking_psl_address=preliminary_price_quote_response.credit_usage_tracking_psl_address,
            proposed_psl_price_per_credit=preliminary_price_quote_response.preliminary_quoted_price_per_credit_in_psl,
            price_agreement_request_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            price_agreement_request_pastel_block_height=await get_current_pastel_block_height_func(),
            price_agreement_request_message_version_string="1.0",
            sha3_256_hash_of_price_agreement_request_fields="",
            supernode_requesting_price_agreement_pastelid_signature_on_request_hash=""
        )
        # Generate the hash and signature fields
        price_agreement_request.sha3_256_hash_of_price_agreement_request_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(price_agreement_request)
        price_agreement_request.supernode_requesting_price_agreement_pastelid_signature_on_request_hash = await sign_message_with_pastelid_func(
            MY_PASTELID,
            price_agreement_request.sha3_256_hash_of_price_agreement_request_fields,
            LOCAL_PASTEL_ID_PASSPHRASE
        )
        # Validate the price_agreement_request
        request_validation_errors = await validate_credit_pack_ticket_message_data_func(price_agreement_request)
        if request_validation_errors:
            logger.error(f"Invalid price agreement request: {', '.join(request_validation_errors)}")
            raise ValueError(f"Invalid price agreement request: {', '.join(request_validation_errors)}") 
        await save_credit_pack_purchase_price_agreement_request(price_agreement_request)
        # Send the price agreement request to the potentially agreeing supernodes
        logger.info(f"Now sending price agreement request to {len(potentially_agreeing_supernodes)} potentially agreeing supernodes...")
        price_agreement_request_responses = await send_price_agreement_request_to_supernodes(price_agreement_request, potentially_agreeing_supernodes)
        # Process the price agreement request responses
        valid_price_agreement_request_responses = []
        for current_price_agreement_response in price_agreement_request_responses:
            response_validation_errors = await validate_credit_pack_ticket_message_data_func(current_price_agreement_response)
            if not response_validation_errors:
                valid_price_agreement_request_responses.append(current_price_agreement_response)
        logger.info(f"Received {len(valid_price_agreement_request_responses)} valid price agreement responses from potentially agreeing supernodes out of {len(potentially_agreeing_supernodes)} asked.")                
        # Check if enough supernodes responded with valid responses
        supernode_price_agreement_response_percentage_achieved = len(valid_price_agreement_request_responses) / len(potentially_agreeing_supernodes)
        if supernode_price_agreement_response_percentage_achieved <= SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:
            logger.warning(f"Not enough supernodes responded with valid price agreement responses; only {supernode_price_agreement_response_percentage_achieved:.2%} of the supernodes responded, less than the required quorum percentage of {SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:.2%}")
            logger.info("Responding to end user with termination message...")
            termination_message = db_code.CreditPackPurchaseRequestResponseTermination(
                sha3_256_hash_of_credit_pack_purchase_request_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
                credit_pack_purchase_request_fields_json=preliminary_price_quote_response.credit_pack_purchase_request_fields_json,
                termination_reason_string="Not enough supernodes responded with valid price agreement responses",
                termination_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                termination_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_purchase_request_termination_message_version_string="1.0",
                responding_supernode_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_purchase_request_termination_fields="",
                responding_supernode_signature_on_credit_pack_purchase_request_termination_hash="",
            )
            termination_message.sha3_256_hash_of_credit_pack_purchase_request_termination_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(termination_message)
            termination_message.responding_supernode_signature_on_credit_pack_purchase_request_termination_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                termination_message.sha3_256_hash_of_credit_pack_purchase_request_termination_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
            termination_validation_errors = await validate_credit_pack_ticket_message_data_func(termination_message)
            if termination_validation_errors:
                logger.error(f"Invalid credit purchase request termination message: {', '.join(termination_validation_errors)}")
                raise ValueError(f"Invalid credit purchase request termination message: {', '.join(termination_validation_errors)}")
            return termination_message
        # Tally the agreeing supernodes
        list_of_agreeing_supernodes = [response.responding_supernode_pastelid for response in valid_price_agreement_request_responses if response.agree_with_proposed_price]
        supernode_price_agreement_voting_percentage = len(list_of_agreeing_supernodes) / len(valid_price_agreement_request_responses)
        logger.info(f"Of the {len(valid_price_agreement_request_responses)} valid price agreement responses, {len(list_of_agreeing_supernodes)} supernodes agreed to the proposed pricing, achieving a voting percentage of {supernode_price_agreement_voting_percentage:.2%}")
        # Check if enough supernodes agreed to the proposed pricing
        if supernode_price_agreement_voting_percentage <= SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE:
            logger.warning(f"Not enough supernodes agreed to the proposed pricing; only {supernode_price_agreement_voting_percentage:.2%} of the supernodes agreed, less than the required majority percentage of {SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE:.2%}")
            logger.info("Responding to end user with termination message...")
            termination_message = db_code.CreditPackPurchaseRequestResponseTermination(
                sha3_256_hash_of_credit_pack_purchase_request_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
                credit_pack_purchase_request_fields_json=preliminary_price_quote_response.credit_pack_purchase_request_fields_json,
                termination_reason_string="Not enough supernodes agreed to the proposed pricing",
                termination_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                termination_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_purchase_request_termination_message_version_string="1.0",
                responding_supernode_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_purchase_request_termination_fields="",
                responding_supernode_signature_on_credit_pack_purchase_request_termination_hash=""
            )
            termination_message.sha3_256_hash_of_credit_pack_purchase_request_termination_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(termination_message)
            termination_message.responding_supernode_signature_on_credit_pack_purchase_request_termination_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                termination_message.sha3_256_hash_of_credit_pack_purchase_request_termination_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
            termination_validation_errors = await validate_credit_pack_ticket_message_data_func(termination_message)
            if termination_validation_errors:
                logger.error(f"Invalid credit purchase request termination message: {', '.join(termination_validation_errors)}")
                raise ValueError(f"Invalid credit purchase request termination message: {', '.join(termination_validation_errors)}")            
            return termination_message
        logger.info(f"Enough supernodes agreed to the proposed pricing; {len(list_of_agreeing_supernodes)} supernodes agreed to the proposed pricing, achieving a voting percentage of {supernode_price_agreement_voting_percentage:.2%}, more than the required minimum percentage of {SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE:.2%}")
        # Aggregate the signatures from the agreeing supernodes:
        list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash = []
        list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json = []
        for response in valid_price_agreement_request_responses:
            if response.agree_with_proposed_price:
                list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash.append(
                    response.responding_supernode_signature_on_price_agreement_request_response_hash
                )
                list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json.append(
                    response.responding_supernode_signature_on_credit_pack_purchase_request_fields_json
                )
        credit_request_response_dict = json.loads(preliminary_price_quote_response.credit_pack_purchase_request_fields_json)
        requested_initial_credits_in_credit_pack = credit_request_response_dict['requested_initial_credits_in_credit_pack']
        # Create the credit pack purchase request response
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse(
            sha3_256_hash_of_credit_pack_purchase_request_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
            credit_pack_purchase_request_fields_json=preliminary_price_quote_response.credit_pack_purchase_request_fields_json,
            psl_cost_per_credit=preliminary_price_quote_response.preliminary_quoted_price_per_credit_in_psl,
            proposed_total_cost_of_credit_pack_in_psl=requested_initial_credits_in_credit_pack*preliminary_price_quote_response.preliminary_quoted_price_per_credit_in_psl,
            credit_usage_tracking_psl_address=preliminary_price_quote_response.credit_usage_tracking_psl_address,
            request_response_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            request_response_pastel_block_height=await get_current_pastel_block_height_func(),
            credit_purchase_request_response_message_version_string="1.0",
            responding_supernode_pastelid=MY_PASTELID,
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms=list_of_agreeing_supernodes,
            list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash=list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash,
            list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json=list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json,
            sha3_256_hash_of_credit_pack_purchase_request_response_fields="",
            responding_supernode_signature_on_credit_pack_purchase_request_response_hash=""
        )
        logger.info(f"Now generating the final credit pack purchase request response and assembling the {len(list_of_agreeing_supernodes)} agreeing supernode signatures...")
        # Generate the hash and signature fields
        credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(credit_pack_purchase_request_response)
        credit_pack_purchase_request_response.responding_supernode_signature_on_credit_pack_purchase_request_response_hash = await sign_message_with_pastelid_func(
            MY_PASTELID,
            credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
            LOCAL_PASTEL_ID_PASSPHRASE
        )
        # Validate the credit_pack_purchase_request_response
        request_response_validation_errors = await validate_credit_pack_ticket_message_data_func(credit_pack_purchase_request_response)
        if request_response_validation_errors:
            logger.error(f"Invalid credit pack purchase request response: {', '.join(request_response_validation_errors)}")
            raise ValueError(f"Invalid credit pack purchase request response: {', '.join(request_response_validation_errors)}")          
        await save_credit_pack_purchase_request_response(credit_pack_purchase_request_response)
        # Send the final credit pack purchase request response to the agreeing supernodes
        logger.info(f"Now sending the final credit pack purchase request response to the list of {len(list_of_agreeing_supernodes)} agreeing supernodes: {list_of_agreeing_supernodes}")
        announcement_responses = await send_credit_pack_purchase_request_final_response_to_supernodes(credit_pack_purchase_request_response, list_of_agreeing_supernodes)
        logger.info(f"Received {len(announcement_responses)} responses to the final credit pack purchase request response announcement, of which {len([response for response in announcement_responses if response.status_code == 200])} were successful")
        return credit_pack_purchase_request_response
    except Exception as e:
        logger.error(f"Error processing credit purchase preliminary price quote response: {str(e)}")
        raise

async def get_credit_purchase_request_status(status_request: db_code.CreditPackRequestStatusCheck) -> db_code.CreditPackPurchaseRequestStatus:
    try:
        # Validate the request fields
        if not status_request.sha3_256_hash_of_credit_pack_purchase_request_fields or not status_request.requesting_end_user_pastelid:
            raise ValueError("Invalid status check request")
        status_request_validation_errors = await validate_credit_pack_ticket_message_data_func(status_request)
        if status_request_validation_errors:
            logger.error(f"Invalid status check request: {', '.join(status_request_validation_errors)}")
            raise ValueError(f"Invalid status check request: {', '.join(status_request_validation_errors)}")
        await save_credit_pack_purchase_request_status_check(status_request)
        # Retrieve the credit pack purchase request
        credit_pack_purchase_request = await get_credit_pack_purchase_request(status_request.sha3_256_hash_of_credit_pack_purchase_request_fields)
        # Check the status of the credit pack purchase request
        status = await check_credit_pack_purchase_request_status(credit_pack_purchase_request)
        # Create the response
        response = db_code.CreditPackPurchaseRequestStatus(
            sha3_256_hash_of_credit_pack_purchase_request_fields=status_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
            status=status
        )
        return response
    except Exception as e:
        logger.error(f"Error getting credit purchase request status: {str(e)}")
        raise
    
async def process_credit_pack_purchase_request_final_response_announcement(response: db_code.CreditPackPurchaseRequestResponse) -> None:
    try:
        # Validate the response fields
        if not response.sha3_256_hash_of_credit_pack_purchase_request_fields or not response.credit_pack_purchase_request_fields_json:
            raise ValueError("Invalid final response announcement")
        # Save the final response to the db
        await save_credit_pack_purchase_request_final_response(response)
    except Exception as e:
        logger.error(f"Error processing credit pack purchase request final response announcement: {str(e)}")
        raise
    
async def get_block_height_for_credit_pack_purchase_request_confirmation(sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str) -> int:
    try:
        # Retrieve the CreditPackPurchaseRequestConfirmation from the database using the hash
        confirmation = await db_code.CreditPackPurchaseRequestConfirmation.get(sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields)
        return confirmation.credit_purchase_request_confirmation_pastel_block_height
    except Exception as e:
        logger.error(f"Error retrieving block height for CreditPackPurchaseRequestConfirmation: {str(e)}")
        raise
    
async def get_block_height_from_block_hash(pastel_block_hash: str):
    global rpc_connection
    if not pastel_block_hash:
        raise ValueError("Invalid block hash provided.")
    try:
        block_details = await rpc_connection.getblock(pastel_block_hash)
        block_height = block_details.get('height')
        if block_height is not None:
            return block_height
        else:
            raise ValueError("Block height could not be retrieved.")
    except Exception as e:
        raise Exception(f"Error retrieving block height: {str(e)}")

async def get_closest_agreeing_supernode_pastelid(end_user_pastelid: str, agreeing_supernode_pastelids: List[str]) -> str:
    try:
        end_user_pastelid_hash = compute_sha3_256_hexdigest(end_user_pastelid)
        end_user_pastelid_int = int(end_user_pastelid_hash, 16)
        xor_distances = []
        for supernode_pastelid in agreeing_supernode_pastelids:
            supernode_pastelid_hash = compute_sha3_256_hexdigest(supernode_pastelid)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            distance = end_user_pastelid_int ^ supernode_pastelid_int
            xor_distances.append((supernode_pastelid, distance))
        closest_supernode_pastelid = min(xor_distances, key=lambda x: x[1])[0]
        return closest_supernode_pastelid
    except Exception as e:
        logger.error(f"Error getting closest agreeing supernode pastelid: {str(e)}")
        raise    
    
async def send_credit_pack_storage_completion_announcement_to_supernodes(response: db_code.CreditPackPurchaseRequestConfirmationResponse, agreeing_supernode_pastelids: List[str]) -> List[httpx.Response]:
    try:
        async with httpx.AsyncClient() as client:
            tasks = []
            supernode_list_df, _ = await check_supernode_list_func()
            for supernode_pastelid in agreeing_supernode_pastelids:
                payload = {}
                try:
                    supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    url = f"{supernode_base_url}/credit_pack_storage_completion_announcement"
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                    challenge = challenge_dict["challenge"]
                    challenge_id = challenge_dict["challenge_id"]
                    challenge_signature = challenge_dict["challenge_signature"]
                    payload = {
                        "storage_completion_announcement": response.model_dump(),
                        "challenge": challenge,
                        "challenge_id": challenge_id,
                        "challenge_signature": challenge_signature
                    }     
                except Exception as e:
                    logger.warning(f"Error getting challenge from supernode {supernode_pastelid}: {str(e)}")
                if len(payload) > 0:
                    task = asyncio.create_task(client.post(url, json=payload))
                    tasks.append(task)
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            valid_responses = []
            for response in responses:
                if isinstance(response, httpx.Response):
                    if response.status_code == 200:
                        valid_responses.append(response)
                    else:
                        logger.warning(f"Error sending storage completion announcement to supernode {response.url}: {response.text}")
                else:
                    logger.error(f"Error sending storage completion announcement to supernode: {str(response)}")
            return valid_responses
    except Exception as e:
        logger.error(f"Error sending storage completion announcement to supernodes: {str(e)}")
        raise
    
async def process_credit_purchase_request_confirmation(confirmation: db_code.CreditPackPurchaseRequestConfirmation) -> db_code.CreditPackPurchaseRequestConfirmationResponse:
    try:
        # Validate the confirmation fields
        validation_errors = await validate_credit_pack_ticket_message_data_func(confirmation)
        if validation_errors:
            logger.error(f"Invalid credit purchase request confirmation: {', '.join(validation_errors)}")
            raise ValueError(f"Invalid credit purchase request confirmation: {', '.join(validation_errors)}")
        await save_credit_pack_purchase_request_confirmation(confirmation)
        # Retrieve the credit pack purchase request response
        credit_pack_purchase_request_response = await get_credit_pack_purchase_request_response(confirmation.sha3_256_hash_of_credit_pack_purchase_request_response_fields)
        # Check the burn transaction
        logger.info("Now checking the burn transaction for the credit pack purchase request...")
        matching_transaction_found = False
        exceeding_transaction_found = False
        num_confirmations = 0
        current_block_height = await get_current_pastel_block_height_func()
        initial_transaction_check_sleep_time_in_seconds = 30 
        current_transaction_check_sleep_time_in_seconds = initial_transaction_check_sleep_time_in_seconds
        while current_block_height <= credit_pack_purchase_request_response.request_response_pastel_block_height + MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET:
            matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_transaction(
                confirmation.txid_of_credit_purchase_burn_transaction,
                credit_pack_purchase_request_response.credit_usage_tracking_psl_address,
                credit_pack_purchase_request_response.proposed_total_cost_of_credit_pack_in_psl,
                credit_pack_purchase_request_response.request_response_pastel_block_height
            )
            if matching_transaction_found or exceeding_transaction_found:
                if (num_confirmations >= MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION) or SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK:
                    break
                else:
                    logger.info(f"Waiting for {current_transaction_check_sleep_time_in_seconds} seconds to check again if burn transaction is confirmed...")
                    await asyncio.sleep(current_transaction_check_sleep_time_in_seconds) 
            else:
                logger.info(f"Waiting for {current_transaction_check_sleep_time_in_seconds} seconds to check again if burn transaction is confirmed...")
                await asyncio.sleep(current_transaction_check_sleep_time_in_seconds)
            current_transaction_check_sleep_time_in_seconds = round(current_transaction_check_sleep_time_in_seconds*1.05, 2)
            current_block_height = await get_current_pastel_block_height_func()
        if matching_transaction_found or exceeding_transaction_found:
            # Store the credit pack ticket on the blockchain
            credit_pack_purchase_request_response_json = json.dumps(credit_pack_purchase_request_response.model_dump())
            credit_pack_ticket_bytes_before_compression = sys.getsizeof(credit_pack_purchase_request_response_json)
            compressed_credit_pack_ticket, _ = await compress_data_with_zstd_func(credit_pack_purchase_request_response_json)
            credit_pack_ticket_bytes_after_compression = sys.getsizeof(compressed_credit_pack_ticket)
            compression_ratio = credit_pack_ticket_bytes_before_compression / credit_pack_ticket_bytes_after_compression
            logger.info(f"Achieved a compression ratio of {compression_ratio:.2f} on credit pack ticket data!")
            logger.info(f"Required burn transaction confirmed with {num_confirmations} confirmations; now attempting to write the credit pack ticket to the blockchain (a total of {credit_pack_ticket_bytes_before_compression:,} bytes before compression and {credit_pack_ticket_bytes_after_compression:,} bytes after compression)...")
            log_action_with_payload("Writing", "the credit pack ticket to the blockchain", credit_pack_purchase_request_response_json)
            pastel_api_credit_pack_ticket_registration_txid, storage_validation_error_string = await store_credit_pack_ticket_in_blockchain(credit_pack_purchase_request_response_json)
            if storage_validation_error_string=="":
                credit_pack_confirmation_outcome_string = "success"
                await save_credit_pack_purchase_request_response_txid_mapping(credit_pack_purchase_request_response, pastel_api_credit_pack_ticket_registration_txid)
            else:
                credit_pack_confirmation_outcome_string = "failed"
            # Create the confirmation response without the hash and signature fields
            confirmation_response = db_code.CreditPackPurchaseRequestConfirmationResponse(
                sha3_256_hash_of_credit_pack_purchase_request_fields=confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=confirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields,
                credit_pack_confirmation_outcome_string=credit_pack_confirmation_outcome_string,
                pastel_api_credit_pack_ticket_registration_txid=pastel_api_credit_pack_ticket_registration_txid,
                credit_pack_confirmation_failure_reason_if_applicable="",
                credit_purchase_request_confirmation_response_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                credit_purchase_request_confirmation_response_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_purchase_request_confirmation_response_message_version_string="1.0",
                responding_supernode_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields="",
                responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash=""
            )
            # Generate the hash and signature fields
            confirmation_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(confirmation_response)
            confirmation_response.responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                confirmation_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
            confirmation_response_validation_errors = await validate_credit_pack_ticket_message_data_func(confirmation_response)
            # Send the CreditPackPurchaseRequestConfirmationResponse to the agreeing supernodes
            announcement_responses = await send_credit_pack_storage_completion_announcement_to_supernodes(confirmation_response, credit_pack_purchase_request_response.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms)
            logger.info(f"Received {len(announcement_responses)} responses to the credit pack storage completion announcement, of which {len([response for response in announcement_responses if response.status_code == 200])} were successful")
        else:
            # Create the confirmation response with failure without the hash and signature fields
            confirmation_response = db_code.CreditPackPurchaseRequestConfirmationResponse(
                sha3_256_hash_of_credit_pack_purchase_request_fields=confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=confirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields,
                credit_pack_confirmation_outcome_string="failure",
                pastel_api_credit_pack_ticket_registration_txid="",
                credit_pack_confirmation_failure_reason_if_applicable="Burn transaction not confirmed within the required number of blocks",
                credit_purchase_request_confirmation_response_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                credit_purchase_request_confirmation_response_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_purchase_request_confirmation_response_message_version_string="1.0",
                responding_supernode_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields="",
                responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash=""
            )
            # Generate the hash and signature fields
            confirmation_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(confirmation_response)
            confirmation_response.responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                confirmation_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
        confirmation_response_validation_errors = await validate_credit_pack_ticket_message_data_func(confirmation_response)
        if confirmation_response_validation_errors:
            logger.error(f"Invalid credit purchase request confirmation response: {', '.join(confirmation_response_validation_errors)}")
            raise ValueError(f"Invalid credit purchase request confirmation response: {', '.join(confirmation_response_validation_errors)}")
        await save_credit_pack_purchase_request_confirmation_response(confirmation_response)            
        # Validate the response
        confirmation_response_validation_errors = await validate_credit_pack_ticket_message_data_func(confirmation_response)
        if confirmation_response_validation_errors:
            logger.error(f"Invalid credit pack purchase completion announcement: {', '.join(confirmation_response_validation_errors)}")
            raise ValueError(f"Invalid credit purchase request confirmation response: {', '.join(validation_errors)}")
        await save_credit_pack_purchase_request_confirmation_response(confirmation_response)
        return confirmation_response
    except Exception as e:
        logger.error(f"Error processing credit purchase request confirmation: {str(e)}")
        raise    
    
async def process_credit_pack_purchase_completion_announcement(confirmation: db_code.CreditPackPurchaseRequestConfirmation) -> None:
    try:
        # Validate the confirmation fields
        if not confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields or not confirmation.sha3_256_hash_of_credit_pack_purchase_request_response_fields:
            raise ValueError("Invalid credit pack purchase completion announcement")
        confirmation_validation_errors = await validate_credit_pack_ticket_message_data_func(confirmation)
        if confirmation_validation_errors:
            logger.error(f"Invalid credit pack purchase completion announcement: {', '.join(confirmation_validation_errors)}")
            raise ValueError(f"Invalid credit pack purchase completion announcement: {', '.join(confirmation_validation_errors)}")
        # Store the completion announcement
        await save_credit_pack_purchase_completion_announcement(confirmation)
    except Exception as e:
        logger.error(f"Error processing credit pack purchase completion announcement: {str(e)}")
        raise

async def process_credit_pack_storage_completion_announcement(completion_response: db_code.CreditPackPurchaseRequestConfirmationResponse) -> None:
    try:
        # Validate the response fields
        if not completion_response.sha3_256_hash_of_credit_pack_purchase_request_fields or not completion_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields:
            raise ValueError("Invalid credit pack storage completion announcement")
        completion_response_validation_errors = await validate_credit_pack_ticket_message_data_func(completion_response)
        if completion_response_validation_errors:
            logger.error(f"Invalid credit pack storage completion announcement: {', '.join(completion_response_validation_errors)}")
            raise ValueError(f"Invalid credit pack storage completion announcement: {', '.join(completion_response_validation_errors)}")        
        # Store the storage completion announcement
        await save_credit_pack_storage_completion_announcement(completion_response)
    except Exception as e:
        logger.error(f"Error processing credit pack storage completion announcement: {str(e)}")
        raise

async def process_credit_pack_storage_retry_request(storage_retry_request: db_code.CreditPackStorageRetryRequest) -> db_code.CreditPackStorageRetryRequestResponse:
    try:
        # Validate the request fields
        storage_retry_request_validation_errors = await validate_credit_pack_ticket_message_data_func(storage_retry_request)
        if storage_retry_request_validation_errors:
            logger.error(f"Invalid credit pack storage retry request: {', '.join(storage_retry_request_validation_errors)}")
            raise ValueError(f"Invalid credit pack storage retry request: {', '.join(storage_retry_request_validation_errors)}")
        await save_credit_pack_storage_retry_request(storage_retry_request)
        # Check if the supernode receiving the retry request is the closest agreeing supernode to the end user's pastelid
        closest_agreeing_supernode_pastelid = await get_closest_agreeing_supernode_pastelid(storage_retry_request.requesting_end_user_pastelid, json.loads(storage_retry_request.credit_pack_purchase_request_response_json)["list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms"])
        if closest_agreeing_supernode_pastelid != MY_PASTELID:
            error_message = f"The supernode receiving the retry request is not the closest agreeing supernode to the end user's pastelid! Closest agreeing supernode to end user's pastelid ({storage_retry_request.requesting_end_user_pastelid}) is {closest_agreeing_supernode_pastelid}, whereas our local pastelid is {MY_PASTELID}"
            logger.error(error_message)
            raise ValueError(error_message)
        # Check if more than MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED Pastel blocks have elapsed since the CreditPackPurchaseRequestConfirmation was sent
        confirmation_block_height = await get_block_height_for_credit_pack_purchase_request_confirmation(storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields)
        current_block_height = await get_current_pastel_block_height_func()
        if current_block_height - confirmation_block_height <= MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED:
            error_message = f"Insufficient time has elapsed since the CreditPackPurchaseRequestConfirmation was sent (current block height: {current_block_height}, confirmation block height: {confirmation_block_height}, so the elapsed block count of {current_block_height - confirmation_block_height} is less than the minimum required block count of {MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED})"
            logger.error(error_message)
            raise ValueError(error_message)
        # Check if the original responding supernode has confirmed the storage
        original_supernode_confirmed_storage = await check_original_supernode_storage_confirmation(storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields)
        if not original_supernode_confirmed_storage:
            # Store the credit pack ticket on the blockchain
            pastel_api_credit_pack_ticket_registration_txid, storage_validation_error_string = await store_credit_pack_ticket_in_blockchain(json.loads(storage_retry_request.credit_pack_purchase_request_response_json))
            if storage_validation_error_string=="":
                credit_pack_confirmation_outcome_string = "success"
            else:
                credit_pack_confirmation_outcome_string = "failed"            
            # Create the retry request response without the hash and signature fields
            retry_request_response = db_code.CreditPackStorageRetryRequestResponse(
                sha3_256_hash_of_credit_pack_purchase_request_fields=storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
                credit_pack_storage_retry_confirmation_outcome_string=credit_pack_confirmation_outcome_string,
                pastel_api_credit_pack_ticket_registration_txid=pastel_api_credit_pack_ticket_registration_txid,
                credit_pack_storage_retry_confirmation_failure_reason_if_applicable="",
                credit_pack_storage_retry_confirmation_response_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                credit_pack_storage_retry_confirmation_response_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_pack_storage_retry_confirmation_response_message_version_string="1.0",
                closest_agreeing_supernode_to_retry_storage_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields="",
                closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash=""
            )
            # Generate the hash and signature fields
            retry_request_response.sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(retry_request_response)
            retry_request_response.closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                retry_request_response.sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
        else:
            # Create the retry request response with failure without the hash and signature fields
            retry_request_response = db_code.CreditPackStorageRetryRequestResponse(
                sha3_256_hash_of_credit_pack_purchase_request_fields=storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
                sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields=storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields,
                credit_pack_storage_retry_confirmation_outcome_string="failure",
                pastel_api_credit_pack_ticket_registration_txid="",
                credit_pack_storage_retry_confirmation_failure_reason_if_applicable="Original responding supernode already confirmed storage",
                credit_pack_storage_retry_confirmation_response_utc_iso_string=datetime.now(dt.UTC).isoformat(),
                credit_pack_storage_retry_confirmation_response_pastel_block_height=await get_current_pastel_block_height_func(),
                credit_pack_storage_retry_confirmation_response_message_version_string="1.0",
                closest_agreeing_supernode_to_retry_storage_pastelid=MY_PASTELID,
                sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields="",
                closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash=""
            )
            # Generate the hash and signature fields
            retry_request_response.sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(retry_request_response)
            retry_request_response.closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash = await sign_message_with_pastelid_func(
                MY_PASTELID,
                retry_request_response.sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields,
                LOCAL_PASTEL_ID_PASSPHRASE
            )
        # Validate the response
        retry_request_response_validation_errors = await validate_credit_pack_ticket_message_data_func(retry_request_response)
        if retry_request_response_validation_errors:
            logger.error(f"Invalid credit pack storage retry request response: {', '.join(retry_request_response_validation_errors)}")
            raise ValueError(f"Invalid credit pack storage retry request response: {', '.join(retry_request_response_validation_errors)}")
        await save_credit_pack_storage_retry_request_response(retry_request_response)
        return retry_request_response_validation_errors
    except Exception as e:
        logger.error(f"Error processing credit pack storage retry request: {str(e)}")
        raise
                    
async def process_credit_pack_storage_retry_completion_announcement(retry_completion_response: db_code.CreditPackStorageRetryRequestResponse) -> None:
    try:
        # Validate the response fields
        if not retry_completion_response.sha3_256_hash_of_credit_pack_purchase_request_fields or not retry_completion_response.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields:
            raise ValueError("Invalid credit pack storage retry completion announcement")
        retry_completion_response_validation_errors = await validate_credit_pack_ticket_message_data_func(retry_completion_response)
        if retry_completion_response_validation_errors:
            logger.error(f"Invalid credit pack storage retry completion announcement: {', '.join(retry_completion_response_validation_errors)}")
            raise ValueError(f"Invalid credit pack storage retry completion announcement: {', '.join(retry_completion_response_validation_errors)}")        
        # Store the storage retry completion announcement
        await save_credit_pack_storage_retry_completion_announcement(retry_completion_response)
    except Exception as e:
        logger.error(f"Error processing credit pack storage retry completion announcement: {str(e)}")
        raise
    
    
#________________________________________________________________________________________________________________            
                
# Inference request related service functions:
                
async def get_inference_model_menu(use_verbose=0):
    try:
        # Load the API key test results from the file
        api_key_tests = load_api_key_tests()
        # Fetch the latest model menu from GitHub
        async with httpx.AsyncClient() as client:
            response = await client.get(GITHUB_MODEL_MENU_URL)
            response.raise_for_status()
            github_model_menu = response.json()
        # Test API keys and filter out models based on the availability of valid API keys
        filtered_model_menu = {"models": []}
        for model in github_model_menu["models"]:
            if model["model_name"].startswith("stability-"):
                if STABILITY_API_KEY and await is_api_key_valid("stability", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added Stability model: {model['model_name']} to the filtered model menu.")
            elif model["model_name"].startswith("openai-"):
                if OPENAI_API_KEY and await is_api_key_valid("openai", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added OpenAImodel: {model['model_name']} to the filtered model menu.")
            elif model["model_name"].startswith("mistralapi-"):
                if MISTRAL_API_KEY and await is_api_key_valid("mistral", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added MistralAPI model: {model['model_name']} to the filtered model menu.")
            elif model["model_name"].startswith("groq-"):
                if GROQ_API_KEY and await is_api_key_valid("groq", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added Groq API model: {model['model_name']} to the filtered model menu.")
            elif "claude" in model["model_name"].lower():
                if CLAUDE3_API_KEY and await is_api_key_valid("claude", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added Anthropic API model: {model['model_name']} to the filtered model menu.")
            elif "openrouter" in model["model_name"].lower():
                if OPENROUTER_API_KEY and await is_api_key_valid("openrouter", api_key_tests):
                    filtered_model_menu["models"].append(model)
                    if use_verbose:
                        logger.info(f"Added OpenRouter model: {model['model_name']} to the filtered model menu.")
            else:
                # Models that don't require API keys can be automatically included
                filtered_model_menu["models"].append(model)
        # Save the filtered model menu locally
        with open("model_menu.json", "w") as file:
            json.dump(filtered_model_menu, file, indent=2)
        save_api_key_tests(api_key_tests) # Save the updated API key test results to file
        return filtered_model_menu
    except Exception as e:
        logger.error(f"Error retrieving inference model menu: {str(e)}")
        raise

def load_api_key_tests():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_test_json_file_path = os.path.join(current_dir, API_KEY_TESTS_FILE) 
    try:
        with open(api_key_test_json_file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_api_key_tests(api_key_tests):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_key_test_json_file_path = os.path.join(current_dir, API_KEY_TESTS_FILE)     
    with open(api_key_test_json_file_path, "w") as file:
        json.dump(api_key_tests, file, indent=2)

async def is_api_key_valid(api_name, api_key_tests):
    if api_name not in api_key_tests or not is_test_result_valid(api_key_tests[api_name]["timestamp"]):
        test_passed = await run_api_key_test(api_name)
        api_key_tests[api_name] = {
            "passed": test_passed,
            "timestamp": datetime.now().isoformat()
        }
        return test_passed
    else:
        return api_key_tests[api_name]["passed"]

def is_test_result_valid(test_timestamp):
    test_datetime = datetime.fromisoformat(test_timestamp)
    return (datetime.now() - test_datetime) < timedelta(hours=API_KEY_TEST_VALIDITY_HOURS)

async def run_api_key_test(api_name):
    if api_name == "stability":
        return await test_stability_api_key()
    elif api_name == "openai":
        return await test_openai_api_key()
    elif api_name == "mistral":
        return await test_mistral_api_key()
    elif api_name == "groq":
        return await test_groq_api_key()
    elif api_name == "claude":
        return await test_claude_api_key()
    elif api_name == "openrouter":
        return await test_openrouter_api_key()    
    else:
        return False

async def test_openai_api_key():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test; just reply with the word yes if you're working!"}]
                }
            )
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"OpenAI API key test failed: {str(e)}")
        return False
    
async def test_stability_api_key():
    try:
        engine_id = "stable-diffusion-v1-6"
        api_host = "https://api.stability.ai"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {STABILITY_API_KEY}"
                },
                json={
                    "text_prompts": [
                        {
                            "text": "A lighthouse on a cliff"
                        }
                    ],
                    "cfg_scale": 7,
                    "height": 512,
                    "width": 512,
                    "samples": 1,
                    "steps": 10,
                },
            )
            if response.status_code != 200:
                raise Exception("Non-200 response: " + str(response.text))
            data = response.json()
            if "artifacts" in data and len(data["artifacts"]) > 0:
                for artifact in data["artifacts"]:  # Directly iterating over the list of dictionaries
                    if artifact.get("finishReason") == "SUCCESS":
                        logger.info("Stability API test passed.")
                        return True
            logger.info("Stability API test failed!")                    
            return False
    except Exception as e:
        logger.warning(f"Stability API key test failed: {str(e)}")
        return False
    
async def test_openrouter_api_key():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "openai/gpt-3.5-turbo",
                    "messages": [{"role": "user", "content": "Test; just reply with the word yes if you're working!"}]
                }
            )
            return response.status_code == 200
    except Exception as e:
        logger.warning(f"OpenRouter API key test failed: {str(e)}")
        return False    

async def test_mistral_api_key():
    try:
        client = MistralAsyncClient(api_key=MISTRAL_API_KEY)
        async_response = client.chat_stream(
            model="mistral-small-latest",
            messages=[ChatMessage(role="user", content="Test; just reply with the word yes if you're working!")],
            max_tokens=10,
            temperature=0.7,
        )
        completion_text = ""
        async for chunk in async_response:
            if chunk.choices[0].delta.content:
                completion_text += chunk.choices[0].delta.content
        logger.info(f"Mistral API test response: {completion_text}")                
        return len(completion_text) > 0
    except Exception as e:
        logger.warning(f"Mistral API key test failed: {str(e)}")
        return False

async def test_groq_api_key():
    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Test; just reply with the word yes if you're working!"}],
            model="mixtral-8x7b-32768",
            max_tokens=10,
            temperature=0.7,
        )
        response_string = chat_completion.choices[0].message.content.strip()
        logger.info(f"Groq API test response: {response_string}")
        if response_string is not None:
            test_passed = len(response_string) > 0
        else:
            test_passed = False
        return test_passed
    except Exception as e:
        logger.warning(f"Groq API key test failed: {str(e)}")
        return False

async def test_claude_api_key():
    try:
        client = anthropic.AsyncAnthropic(api_key=CLAUDE3_API_KEY)
        async with client.messages.stream(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.7,
            messages=[{"role": "user", "content": "Test; just reply with the word yes if you're working!"}],
        ) as stream:
            message = await stream.get_final_message()
            response_string = message.content[0].text.strip()
            logger.info(f"Anthropic API test response: {response_string}")
            if response_string is not None:
                test_passed = len(response_string) > 0
            else:
                test_passed = False
            return test_passed
    except Exception as e:
        logger.warning(f"Claude API key test failed: {str(e)}")
        return False

async def save_inference_api_usage_request(inference_request_model: db_code.InferenceAPIUsageRequest) -> db_code.InferenceAPIUsageRequest:
    async with db_code.Session() as db_session:
        db_session.add(inference_request_model)
        db_session.commit()
        db_session.refresh(inference_request_model)
    return inference_request_model

def get_tokenizer(model_name: str):
    model_to_tokenizer_mapping = {
        "claude3": "Xenova/claude-tokenizer",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
        "mistral-7b-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
        "phi": "TheBloke/phi-2-GGUF",
        "openai": "cl100k_base",
        "groq-llama2": "TheBloke/Yarn-Llama-2-7B-128K-GGUF",
        "groq-mixtral": "EleutherAI/gpt-neox-20b",
        "groq-gemma": "google/flan-ul2",
        "mistralapi": "mistralai/Mistral-7B-Instruct-v0.2",
        "stability": "openai/clip-vit-large-patch14",
        "whisper": "openai/whisper-large-v2",
        "clip-interrogator": "openai/clip-vit-large-patch14",
        "videocap-transformer": "ArhanK005/videocap-transformer",
        "openrouter/google": "google/flan-t5-xl",
        "openrouter/anthropic": "Xenova/claude-tokenizer",
        "openrouter/meta-llama": "huggyllama/llama-7b",
        "openrouter/distilgpt2": "distilgpt2",
        "openrouter/bigscience/bloom": "bigscience/bloom-1b7",
        "openrouter/databricks/dolly-v2-12b": "databricks/dolly-v2-12b",
        "openrouter/EleutherAI/gpt-j-6b": "EleutherAI/gpt-j-6B",
        "openrouter/OpenAssistant/oasst-sft-6-llama-30b-xor": "OpenAssistant/oasst-sft-6-llama-30b-xor",
        "openrouter/stabilityai/stablelm-tuned-alpha-7b": "stabilityai/stablelm-base-alpha-7b",
        "openrouter/togethercomputer/GPT-JT-6B-v1": "togethercomputer/GPT-JT-6B-v1",
        "openrouter/tiiuae/falcon-7b-instruct": "tiiuae/falcon-7b-instruct"
    }
    best_match = process.extractOne(model_name.lower(), model_to_tokenizer_mapping.keys())
    return model_to_tokenizer_mapping.get(best_match[0], "gpt2")  # Default to "gpt2" if no match found

def count_tokens(model_name: str, input_data: str) -> int:
    tokenizer_name = get_tokenizer(model_name)
    logger.info(f"Selected tokenizer {tokenizer_name} for model {model_name}")
    if 'claude' in model_name.lower():
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
    elif 'whisper' in model_name.lower():
        tokenizer = WhisperTokenizer.from_pretrained(tokenizer_name)
    elif 'clip-interrogator' in model_name.lower() or 'stability' in model_name.lower():
        # CLIP and Stability models don't require tokenization for input data
        return 0
    elif 'videocap-transformer' in model_name.lower():
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    elif 'openai' in model_name.lower():
        encoding = tiktoken.get_encoding(tokenizer_name)
        input_tokens = encoding.encode(input_data)
        return len(input_tokens)
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if hasattr(tokenizer, "encode"):  # For tokenizers with an "encode" method (e.g., GPT-2, GPT-Neo)
        input_tokens = tokenizer.encode(input_data)
    else:  # For tokenizers without an "encode" method (e.g., BERT, RoBERTa)
        input_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_data))
    return len(input_tokens)

def calculate_api_cost(model_name: str, input_data: str, model_parameters: Dict) -> float:
    # Define the pricing data for each API service and model
    logger.info(f"Evaluating API cost for model: {model_name}")
    pricing_data = {
        "claude-instant": {"input_cost": 0.0008, "output_cost": 0.0024, "per_call_cost": 0.0013},
        "claude-2.1": {"input_cost": 0.008, "output_cost": 0.024, "per_call_cost": 0.0128},
        "claude3-haiku": {"input_cost": 0.00025, "output_cost": 0.00125, "per_call_cost": 0.0006},
        "claude3-sonnet": {"input_cost": 0.003, "output_cost": 0.015, "per_call_cost": 0.0078},
        "claude3-opus": {"input_cost": 0.015, "output_cost": 0.075, "per_call_cost": 0.0390},
        "mistralapi-mistral-small-latest": {"input_cost": 0.002, "output_cost": 0.006, "per_call_cost": 0.0032},
        "mistralapi-mistral-medium-latest": {"input_cost": 0.0027, "output_cost": 0.0081, "per_call_cost": 0},
        "mistralapi-mistral-large-latest": {"input_cost": 0.008, "output_cost": 0.024, "per_call_cost": 0.0128},
        "mistralapi-mistral-7b": {"input_cost": 0.00025, "output_cost": 0.00025, "per_call_cost": 0},
        "mistralapi-open-mistral-7b": {"input_cost": 0.00025, "output_cost": 0.00025, "per_call_cost": 0},
        "mistralapi-open-mixtral-8x7b": {"input_cost": 0.0007, "output_cost": 0.0007, "per_call_cost": 0},
        "mistralapi-mistral-embed": {"input_cost": 0.0001, "output_cost": 0, "per_call_cost": 0},
        "openai-gpt-4-0125-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openai-gpt-4-1106-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openai-gpt-4-1106-vision-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openai-gpt-4": {"input_cost": 0.03, "output_cost": 0.06, "per_call_cost": 0},
        "openai-gpt-4-32k": {"input_cost": 0.06, "output_cost": 0.12, "per_call_cost": 0},
        "openai-gpt-3.5-turbo-0125": {"input_cost": 0.0005, "output_cost": 0.0015, "per_call_cost": 0},
        "openai-gpt-3.5-turbo-instruct": {"input_cost": 0.0015, "output_cost": 0.002, "per_call_cost": 0},
        "openai-text-embedding-ada-002": {"input_cost": 0.0004, "output_cost": 0, "per_call_cost": 0},
        "groq-llama2-70b-4096": {"input_cost": 0.0007, "output_cost": 0.0008, "per_call_cost": 0},
        "groq-llama2-7b-2048": {"input_cost": 0.0001, "output_cost": 0.0001, "per_call_cost": 0},
        "groq-mixtral-8x7b-32768": {"input_cost": 0.00027, "output_cost": 0.00027, "per_call_cost": 0},
        "groq-gemma-7b-it": {"input_cost": 0.0001, "output_cost": 0.0001, "per_call_cost": 0},
        "stability-core": {"credits_per_call": 3},
        "stability-sdxl-1.0": {"credits_per_call": 0.4},  # Average of 0.2-0.6
        "stability-sd-1.6": {"credits_per_call": 0.6},  # Average of 0.2-1.0
        "stability-creative-upscaler": {"credits_per_call": 25},
        "stability-esrgan": {"credits_per_call": 0.2},
        "stability-search-and-replace": {"credits_per_call": 4},
        "stability-inpaint": {"credits_per_call": 3},
        "stability-outpaint": {"credits_per_call": 4},
        "stability-remove-background": {"credits_per_call": 2},
        "openrouter/auto": {"input_cost": 0.0005, "output_cost": 0.0015, "per_call_cost": 0},
        "openrouter/nousresearch/nous-capybara-7b:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/mistralai/mistral-7b-instruct:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/gryphe/mythomist-7b:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/undi95/toppy-m-7b:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/cinematika-7b:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/google/gemma-7b-it:free": {"input_cost": 0, "output_cost": 0, "per_call_cost": 0},
        "openrouter/jebcarter/psyfighter-13b": {"input_cost": 0.001, "output_cost": 0.001, "per_call_cost": 0},
        "openrouter/koboldai/psyfighter-13b-2": {"input_cost": 0.001, "output_cost": 0.001, "per_call_cost": 0},
        "openrouter/intel/neural-chat-7b": {"input_cost": 0.005, "output_cost": 0.005, "per_call_cost": 0},
        "openrouter/haotian-liu/llava-13b": {"input_cost": 0.005, "output_cost": 0.005, "per_call_cost": 0},
        "openrouter/nousresearch/nous-hermes-2-vision-7b": {"input_cost": 0.005, "output_cost": 0.005, "per_call_cost": 0},
        "openrouter/meta-llama/llama-2-13b-chat": {"input_cost": 0.0001474, "output_cost": 0.0001474, "per_call_cost": 0},
        "openrouter/migtissera/synthia-70b": {"input_cost": 0.00375, "output_cost": 0.00375, "per_call_cost": 0},
        "openrouter/pygmalionai/mythalion-13b": {"input_cost": 0.001125, "output_cost": 0.001125, "per_call_cost": 0},
        "openrouter/xwin-lm/xwin-lm-70b": {"input_cost": 0.00375, "output_cost": 0.00375, "per_call_cost": 0},
        "openrouter/alpindale/goliath-120b": {"input_cost": 0.009375, "output_cost": 0.009375, "per_call_cost": 0},
        "openrouter/neversleep/noromaid-20b": {"input_cost": 0.00225, "output_cost": 0.00225, "per_call_cost": 0},
        "openrouter/gryphe/mythomist-7b": {"input_cost": 0.000375, "output_cost": 0.000375, "per_call_cost": 0},
        "openrouter/sophosympatheia/midnight-rose-70b": {"input_cost": 0.009, "output_cost": 0.009, "per_call_cost": 0},
        "openrouter/undi95/remm-slerp-l2-13b:extended": {"input_cost": 0.001125, "output_cost": 0.001125, "per_call_cost": 0},
        "openrouter/gryphe/mythomax-l2-13b:extended": {"input_cost": 0.001125, "output_cost": 0.001125, "per_call_cost": 0},
        "openrouter/mancer/weaver": {"input_cost": 0.003375, "output_cost": 0.003375, "per_call_cost": 0},
        "openrouter/nousresearch/nous-hermes-llama2-13b": {"input_cost": 0.00027, "output_cost": 0.00027, "per_call_cost": 0},
        "openrouter/nousresearch/nous-capybara-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/meta-llama/codellama-34b-instruct": {"input_cost": 0.00072, "output_cost": 0.00072, "per_call_cost": 0},
        "openrouter/codellama/codellama-70b-instruct": {"input_cost": 0.00081, "output_cost": 0.00081, "per_call_cost": 0},
        "openrouter/phind/phind-codellama-34b": {"input_cost": 0.00072, "output_cost": 0.00072, "per_call_cost": 0},
        "openrouter/teknium/openhermes-2-mistral-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/teknium/openhermes-2.5-mistral-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/undi95/remm-slerp-l2-13b": {"input_cost": 0.00027, "output_cost": 0.00027, "per_call_cost": 0},
        "openrouter/undi95/toppy-m-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/cinematika-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/01-ai/yi-34b-chat": {"input_cost": 0.00072, "output_cost": 0.00072, "per_call_cost": 0},
        "openrouter/01-ai/yi-34b": {"input_cost": 0.00072, "output_cost": 0.00072, "per_call_cost": 0},
        "openrouter/01-ai/yi-6b": {"input_cost": 0.000126, "output_cost": 0.000126, "per_call_cost": 0},
        "openrouter/togethercomputer/stripedhyena-nous-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/togethercomputer/stripedhyena-hessian-7b": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/mistralai/mixtral-8x7b": {"input_cost": 0.00054, "output_cost": 0.00054, "per_call_cost": 0},
        "openrouter/nousresearch/nous-hermes-yi-34b": {"input_cost": 0.00072, "output_cost": 0.00072, "per_call_cost": 0},
        "openrouter/nousresearch/nous-hermes-2-mixtral-8x7b-sft": {"input_cost": 0.00054, "output_cost": 0.00054, "per_call_cost": 0},
        "openrouter/nousresearch/nous-hermes-2-mistral-7b-dpo": {"input_cost": 0.00018, "output_cost": 0.00018, "per_call_cost": 0},
        "openrouter/mistralai/mixtral-8x22b": {"input_cost": 0.00108, "output_cost": 0.00108, "per_call_cost": 0},
        "openrouter/open-orca/mistral-7b-openorca": {"input_cost": 0.0001425, "output_cost": 0.0001425, "per_call_cost": 0},
        "openrouter/huggingfaceh4/zephyr-7b-beta": {"input_cost": 0.0001425, "output_cost": 0.0001425, "per_call_cost": 0},
        "openrouter/openai/gpt-3.5-turbo": {"input_cost": 0.0005, "output_cost": 0.0015, "per_call_cost": 0},
        "openrouter/openai/gpt-3.5-turbo-0125": {"input_cost": 0.0005, "output_cost": 0.0015, "per_call_cost": 0},
        "openrouter/openai/gpt-3.5-turbo-16k": {"input_cost": 0.003, "output_cost": 0.004, "per_call_cost": 0},
        "openrouter/openai/gpt-4-turbo": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openrouter/openai/gpt-4-turbo-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openrouter/openai/gpt-4": {"input_cost": 0.03, "output_cost": 0.06, "per_call_cost": 0},
        "openrouter/openai/gpt-4-32k": {"input_cost": 0.06, "output_cost": 0.12, "per_call_cost": 0},
        "openrouter/openai/gpt-4-vision-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0},
        "openrouter/openai/gpt-3.5-turbo-instruct": {"input_cost": 0.0015, "output_cost": 0.002, "per_call_cost": 0},
        "openrouter/google/palm-2-chat-bison": {"input_cost": 0.00025, "output_cost": 0.0005, "per_call_cost": 0},
        "openrouter/google/palm-2-codechat-bison": {"input_cost": 0.00025, "output_cost": 0.0005, "per_call_cost": 0},
        "openrouter/google/palm-2-chat-bison-32k": {"input_cost": 0.00025, "output_cost": 0.0005, "per_call_cost": 0},
        "openrouter/google/palm-2-codechat-bison-32k": {"input_cost": 0.00025, "output_cost": 0.0005, "per_call_cost": 0},
        "openrouter/google/gemini-pro": {"input_cost": 0.000125, "output_cost": 0.000375, "per_call_cost": 0},
        "openrouter/google/gemini-pro-vision": {"input_cost": 0.000125, "output_cost": 0.000375, "per_call_cost": 0},
        "openrouter/google/gemini-pro-1.5": {"input_cost": 0.0025, "output_cost": 0.0075, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-opus": {"input_cost": 0.015, "output_cost": 0.075, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-sonnet": {"input_cost": 0.003, "output_cost": 0.015, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-haiku": {"input_cost": 0.00025, "output_cost": 0.00125, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-opus:beta": {"input_cost": 0.015, "output_cost": 0.075, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-sonnet:beta": {"input_cost": 0.003, "output_cost": 0.015, "per_call_cost": 0},
        "openrouter/anthropic/claude-3-haiku:beta": {"input_cost": 0.00025, "output_cost": 0.00125, "per_call_cost": 0},
        "openrouter/meta-llama/llama-2-70b-chat": {"input_cost": 0.0007, "output_cost": 0.0009, "per_call_cost": 0},
        "openrouter/nousresearch/nous-capybara-34b": {"input_cost": 0.0009, "output_cost": 0.0009, "per_call_cost": 0},
        "openrouter/jondurbin/airoboros-l2-70b": {"input_cost": 0.0007, "output_cost": 0.0009, "per_call_cost": 0},
        "openrouter/jondurbin/bagel-34b": {"input_cost": 0.00575, "output_cost": 0.00575, "per_call_cost": 0},
        "openrouter/austism/chronos-hermes-13b": {"input_cost": 0.00022, "output_cost": 0.00022, "per_call_cost": 0},
        "openrouter/mistralai/mistral-7b-instruct": {"input_cost": 0.00013, "output_cost": 0.00013, "per_call_cost": 0},
        "openrouter/gryphe/mythomax-l2-13b": {"input_cost": 0.0002, "output_cost": 0.0002, "per_call_cost": 0},
        "openrouter/openchat/openchat-7b": {"input_cost": 0.00013, "output_cost": 0.00013, "per_call_cost": 0},
        "openrouter/lizpreciatior/lzlv-70b-fp16-hf": {"input_cost": 0.0007, "output_cost": 0.0008, "per_call_cost": 0},
        "openrouter/mistralai/mixtral-8x7b-instruct": {"input_cost": 0.00027, "output_cost": 0.00027, "per_call_cost": 0},
        "openrouter/cognitivecomputations/dolphin-mixtral-8x7b": {"input_cost": 0.0005, "output_cost": 0.0005, "per_call_cost": 0},
        "openrouter/neversleep/noromaid-mixtral-8x7b-instruct": {"input_cost": 0.008, "output_cost": 0.008, "per_call_cost": 0},        
    }
    # Find the best match for the model name using fuzzy string matching
    best_match = process.extractOne(model_name.lower(), pricing_data.keys())
    if best_match is None or best_match[1] < 60:  # Adjust the threshold as needed
        logger.warning(f"No pricing data found for model: {model_name}")
        return 0.0
    model_pricing = pricing_data[best_match[0]]
    if model_name.startswith("stability-"):
        # For Stability models, calculate the cost based on the credits per call
        number_of_completions_to_generate = model_parameters.get("number_of_completions_to_generate", 1)
        credits_cost = model_pricing["credits_per_call"] * number_of_completions_to_generate
        estimated_cost = credits_cost * 10 / 1000  # Convert credits to dollars ($10 per 1,000 credits)
    else:
        # For other models, calculate the cost based on input/output tokens and per-call cost
        input_data_tokens = count_tokens(model_name, input_data)
        logger.info(f"Total input data tokens: {input_data_tokens}")
        number_of_tokens_to_generate = model_parameters.get("number_of_tokens_to_generate", 1000)
        number_of_completions_to_generate = model_parameters.get("number_of_completions_to_generate", 1)
        input_cost = model_pricing["input_cost"] * input_data_tokens / 1000
        output_cost = model_pricing["output_cost"] * number_of_tokens_to_generate / 1000
        per_call_cost = model_pricing["per_call_cost"] * number_of_completions_to_generate
        estimated_cost = input_cost + output_cost + per_call_cost
    logger.info(f"Estimated cost: ${estimated_cost:.4f}")
    return estimated_cost

async def calculate_proposed_inference_cost_in_credits(requested_model_data: Dict, model_parameters: Dict, input_data: str) -> float:
    model_name = requested_model_data["model_name"]
    # Calculate the API cost if applicable
    api_cost = calculate_api_cost(model_name, input_data, model_parameters)
    if api_cost > 0.0:
        # If it's an API service-based inference request, convert the API cost to inference credits
        target_value_per_credit = TARGET_VALUE_PER_CREDIT_IN_USD
        target_profit_margin = TARGET_PROFIT_MARGIN
        # Calculate the proposed cost in inference credits based on the API cost
        proposed_cost_in_credits = api_cost / (target_value_per_credit * (1 - target_profit_margin))
        final_proposed_cost_in_credits = max([MINIMUM_COST_IN_CREDITS, round(proposed_cost_in_credits, 1)])
        logger.info(f"Proposed cost in credits (API-based): {final_proposed_cost_in_credits}")
        return final_proposed_cost_in_credits
    # If it's a local LLM inference request, calculate the cost based on the model's credit costs
    input_token_cost = requested_model_data["credit_costs"]["input_tokens"]
    output_token_cost = requested_model_data["credit_costs"]["output_tokens"]
    compute_cost = requested_model_data["credit_costs"]["compute_cost"]
    memory_cost = requested_model_data["credit_costs"]["memory_cost"]
    # Extract relevant information from the model_parameters
    number_of_tokens_to_generate = model_parameters.get("number_of_tokens_to_generate", 1000)
    number_of_completions_to_generate = model_parameters.get("number_of_completions_to_generate", 1)
    # Calculate the input data size in tokens using the appropriate tokenizer
    input_data_tokens = count_tokens(model_name, input_data)
    logger.info(f"Total input data tokens: {input_data_tokens}")
    # Estimate the output data size in tokens (assuming output tokens <= number_of_tokens_to_generate)
    estimated_output_tokens = number_of_tokens_to_generate
    # Calculate the proposed cost based on the extracted information
    proposed_cost_in_credits = number_of_completions_to_generate * (
        (input_data_tokens * input_token_cost) +
        (estimated_output_tokens * output_token_cost) +
        compute_cost
    ) + memory_cost
    final_proposed_cost_in_credits = round(proposed_cost_in_credits*CREDIT_COST_MULTIPLIER_FACTOR, 1)
    final_proposed_cost_in_credits = max([MINIMUM_COST_IN_CREDITS, final_proposed_cost_in_credits])
    logger.info(f"Proposed cost in credits (local LLM): {final_proposed_cost_in_credits}")
    return final_proposed_cost_in_credits

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

def is_swiss_army_llama_responding():
    try:
        url = f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/get_list_of_available_model_names/"
        params = {'token': SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        response = httpx.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error: {e}")
        return False

def normalize_string(s):
    """Remove non-alphanumeric characters and convert to lowercase."""
    return re.sub(r'\W+', '', s).lower()

def validate_pastel_txid_string(input_string: str):
    # Sample txid: 625694b632a05f5df8d70904b9b3ff03d144aec0352b2290a275379586daf8db
    return re.match(r'^[0-9a-fA-F]{64}$', input_string) is not None

async def validate_inference_api_usage_request(inference_api_usage_request: db_code.InferenceAPIUsageRequest) -> Tuple[bool, float, float]:
    try:
        validation_errors = await validate_inference_request_message_data_func()
        if validation_errors:
            raise ValueError(f"Invalid inference request message: {', '.join(validation_errors)}")        
        requesting_pastelid = inference_api_usage_request.requesting_pastelid
        credit_pack_ticket_pastel_txid = inference_api_usage_request.credit_pack_ticket_pastel_txid
        requested_model = inference_api_usage_request.requested_model_canonical_string
        model_inference_type_string = inference_api_usage_request.model_inference_type_string
        model_parameters = inference_api_usage_request.model_parameters_json
        input_data = inference_api_usage_request.model_input_data_json_b64
        if not validate_pastel_txid_string(credit_pack_ticket_pastel_txid):
            logger.error(f"Invalid PastelID: {credit_pack_ticket_pastel_txid}")
            return False, 0, 0
        credit_pack_purchase_request_response_object = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_pastel_txid)
        credit_pack_purchase_request_object = await get_credit_pack_purchase_request_from_response(credit_pack_purchase_request_response_object)
        if credit_pack_purchase_request_object:
            list_of_authorized_pastelids_allowed_to_use_credit_pack = json.dumps(credit_pack_purchase_request_object.list_of_authorized_pastelids_allowed_to_use_credit_pack)
            # Check if the requesting PastelID is authorized to use the credit pack
            if requesting_pastelid not in list_of_authorized_pastelids_allowed_to_use_credit_pack:
                logger.warning(f"Unauthorized PastelID: {requesting_pastelid}")
                return False, 0, 0
        # # Retrieve the model menu
        model_menu = await get_inference_model_menu()
        # Check if the requested model exists in the model menu
        requested_model_data = next((model for model in model_menu["models"] if normalize_string(model["model_name"]) == normalize_string(requested_model)), None)
        if requested_model_data is None:
            logger.warning(f"Invalid model requested: {requested_model}")
            return False, 0, 0
        if "api_based_pricing" in requested_model_data['credit_costs']:
            is_api_based_model = 1
        else:
            is_api_based_model = 0
        # Check if the requested inference type is supported by the model
        if model_inference_type_string not in requested_model_data["supported_inference_type_strings"]:
            logger.warning(f"Unsupported inference type '{model_inference_type_string}' for model '{requested_model}'")
            return False, 0, 0
        if not is_api_based_model:
            if is_swiss_army_llama_responding():
                # Validate the requested model against the Swiss Army Llama API
                async with httpx.AsyncClient() as client:
                    params = {"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
                    response = await client.get(f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/get_list_of_available_model_names/", params=params)
                    if response.status_code == 200:
                        available_models = response.json()["model_names"]
                        if requested_model not in available_models:
                            # Add the new model to Swiss Army Llama
                            add_model_response = await client.post(f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/add_new_model/", params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN, "model_url": requested_model_data["model_url"]})
                            if add_model_response.status_code != 200:
                                logger.warning(f"Failed to add new model to Swiss Army Llama: {requested_model}")
                                return False, 0, 0
                    else:
                        logger.warning("Failed to retrieve available models from Swiss Army Llama API")
                        return False, 0, 0
            else:
                logger.error(f"Error! Swiss Army Llama is not running on port {SWISS_ARMY_LLAMA_PORT}")
        # Calculate the proposed cost in credits based on the requested model and input data
        model_parameters_dict = json.loads(model_parameters)
        input_data_binary = base64.b64decode(input_data)
        result = magika.identify_bytes(input_data_binary)
        detected_data_type = result.output.ct_label
        if detected_data_type == "txt":
            input_data = input_data_binary.decode("utf-8")
        proposed_cost_in_credits = await calculate_proposed_inference_cost_in_credits(requested_model_data, model_parameters_dict, input_data)
        # Check if the credit pack has sufficient credits for the request
        current_credit_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address = await determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack_purchase_request_response_object, burn_address)
        if not proposed_cost_in_credits >= current_credit_balance:
            logger.warning(f"Insufficient credits for the request. Required: {proposed_cost_in_credits}, Available: {current_credit_balance}")
            return False, proposed_cost_in_credits, current_credit_balance
        # Calculate the remaining credits after the request
        remaining_credits_after_request = current_credit_balance - proposed_cost_in_credits
        return True, proposed_cost_in_credits, remaining_credits_after_request
    except Exception as e:
        logger.error(f"Error validating inference API usage request: {str(e)}")
        raise
    
async def process_inference_api_usage_request(inference_api_usage_request: db_code.InferenceAPIUsageRequest) -> db_code.InferenceAPIUsageResponse: 
    # Validate the inference API usage request
    is_valid_request, proposed_cost_in_credits, remaining_credits_after_request = await validate_inference_api_usage_request(inference_api_usage_request) 
    inference_api_usage_request_dict = inference_api_usage_request.model_dump()
    if not is_valid_request:
        logger.error("Invalid inference API usage request received!")
        raise ValueError(f"Error! Received invalid inference API usage request: {inference_api_usage_request_dict}")
    else:
        log_action_with_payload("Received", "inference API usage request", inference_api_usage_request_dict)
    # Save the inference API usage request
    saved_request = await save_inference_api_usage_request(inference_api_usage_request)
    credit_pack_ticket_pastel_txid = inference_api_usage_request.credit_pack_ticket_pastel_txid
    credit_pack_purchase_request_response_object = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_pastel_txid)
    credit_usage_tracking_psl_address = credit_pack_purchase_request_response_object.credit_usage_tracking_psl_address
    # Create and save the InferenceAPIUsageResponse
    inference_response = await create_and_save_inference_api_usage_response(saved_request, proposed_cost_in_credits, remaining_credits_after_request, credit_usage_tracking_psl_address)
    return inference_response

async def create_and_save_inference_api_usage_response(saved_request: db_code.InferenceAPIUsageRequest, proposed_cost_in_credits: float, remaining_credits_after_request: float, credit_usage_tracking_psl_address: str) -> db_code.InferenceAPIUsageResponse:
    # Generate a unique identifier for the inference response
    inference_response_id = str(uuid.uuid4())
    # Create an InferenceAPIUsageResponse instance without the hash and signature fields
    _, _, local_supernode_pastelid, _ = await get_local_machine_supernode_data_func()
    inference_response = db_code.InferenceAPIUsageResponse(
        inference_response_id=inference_response_id,
        inference_request_id=saved_request.inference_request_id,
        proposed_cost_of_request_in_inference_credits=proposed_cost_in_credits,
        remaining_credits_in_pack_after_request_processed=remaining_credits_after_request,
        credit_usage_tracking_psl_address=credit_usage_tracking_psl_address,
        request_confirmation_message_amount_in_patoshis=int(proposed_cost_in_credits * CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER),
        max_block_height_to_include_confirmation_transaction=await get_current_pastel_block_height_func() + 10,  # Adjust as needed
        inference_request_response_utc_iso_string=datetime.now(dt.UTC).isoformat(),
        inference_request_response_pastel_block_height=await get_current_pastel_block_height_func(),
        inference_request_response_message_version_string="1.0"
    )
    # Generate the hash and signature fields
    inference_response.sha3_256_hash_of_inference_request_response_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(inference_response)
    confirmation_signature = await sign_message_with_pastelid_func(local_supernode_pastelid, inference_response.sha3_256_hash_of_inference_request_response_fields, LOCAL_PASTEL_ID_PASSPHRASE)
    inference_response.supernode_pastelid_and_signature_on_inference_request_response_hash = json.dumps({'signing_sn_pastelid': local_supernode_pastelid, 'sn_signature_on_response_hash': confirmation_signature})
    # Save the InferenceAPIUsageResponse to the database
    async with db_code.Session() as db_session:
        db_session.add(inference_response)
        db_session.commit()
        db_session.refresh(inference_response)
    return inference_response

async def check_burn_address_for_tracking_transaction(
    burn_address: str,
    tracking_address: str,
    expected_amount: float,
    txid: str,
    max_block_height: int,
    max_retries: int = 10,
    initial_retry_delay: int = 25
) -> Tuple[bool, int, int]:
    global rpc_connection
    try_count = 0
    retry_delay = initial_retry_delay
    total_amount_to_burn_address = 0.0
    while try_count < max_retries:
        # Retrieve and decode the transaction details using the txid
        if len(txid) > 0:
            decoded_tx_data = await get_and_decode_raw_transaction(txid)
            if decoded_tx_data:
                # Check if the transaction matches the specified criteria
                if any(vout["scriptPubKey"].get("addresses", [None])[0] == burn_address for vout in decoded_tx_data["vout"]):
                    # Calculate the total amount sent to the burn address in the transaction
                    total_amount_to_burn_address = sum(
                        vout["value"] for vout in decoded_tx_data["vout"]
                        if vout["scriptPubKey"].get("addresses", [None])[0] == burn_address
                    )
                    if total_amount_to_burn_address == expected_amount:
                        # Retrieve the transaction details using gettransaction RPC method
                        tx_info = await rpc_connection.gettransaction(txid)
                        if tx_info:
                            num_confirmations = tx_info.get("confirmations", 0)
                            transaction_block_hash = tx_info.get("blockhash", None)
                            if transaction_block_hash:
                                transaction_block_height = await get_block_height_from_block_hash(transaction_block_hash)
                            else:
                                transaction_block_height = 0
                            if ((num_confirmations >= MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION) or SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK) and transaction_block_height <= max_block_height:
                                logger.info(f"Matching confirmed transaction found with {num_confirmations} confirmation blocks, greater than or equal to the required confirmation blocks of {MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION}!")
                                return True, False, transaction_block_height, num_confirmations, total_amount_to_burn_address
                            else:
                                logger.info(f"Matching unconfirmed transaction found! Waiting for it to be mined in a block with at least {MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION} confirmation blocks! (Currently it has only {num_confirmations} confirmation blocks)")
                                return True, False, transaction_block_height, num_confirmations, total_amount_to_burn_address
                    elif total_amount_to_burn_address >= expected_amount:
                        # Retrieve the transaction details using gettransaction RPC method
                        tx_info = await rpc_connection.gettransaction(txid)
                        if tx_info:
                            num_confirmations = tx_info.get("confirmations", 0)
                            transaction_block_hash = tx_info.get("blockhash", None)
                            if transaction_block_hash:
                                transaction_block_height = await get_block_height_from_block_hash(transaction_block_hash)
                            else:
                                transaction_block_height = 0                                                            
                            if ((num_confirmations >= MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION) or SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK) and transaction_block_height <= max_block_height:
                                logger.info(f"Matching confirmed transaction was not found, but we did find a confirmed (with {num_confirmations} confirmation blocks) burn transaction with more than the expected amount ({total_amount_to_burn_address} sent versus the expected amount of {expected_amount})")
                                return False, True, transaction_block_height, num_confirmations, total_amount_to_burn_address
                            else:
                                logger.info(f"Matching unconfirmed transaction was not found, but we did find an unconfirmed burn transaction with more than the expected amount ({total_amount_to_burn_address} sent versus the expected amount of {expected_amount})")
                                return False, True, transaction_block_height, num_confirmations, total_amount_to_burn_address                        
                    else:
                        logger.warning(f"Transaction {txid} found, but the amount sent to the burn address ({total_amount_to_burn_address}) is less than the expected amount ({expected_amount})")
                else:
                    logger.warning(f"Transaction {txid} does not send funds to the specified burn address")
            # If the transaction is not found or does not match the criteria, wait before retrying
            logger.info(f"WAITING {retry_delay} seconds before checking transaction {txid} status again...")
            await asyncio.sleep(retry_delay)
            try_count += 1
            retry_delay *= 1.15  # Optional: increase delay between retries
        else:
            logger.error(f"Invalid txid for tracking transaction: {txid}")
    logger.info(f"Transaction not found or did not match the criteria after {max_retries} attempts.")
    return False, False, None, None, total_amount_to_burn_address

async def process_inference_confirmation(inference_request_id: str, confirmation_transaction: db_code.InferenceConfirmation) -> bool:
    try:
        # Retrieve the inference API usage request from the database
        async with db_code.Session() as db:
            inference_request = await db.exec(
                select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
            )
            inference_request = inference_request.one_or_none()
        if inference_request is None:
            logger.warning(f"Invalid inference request ID: {inference_request_id}")
            return False
        # Retrieve the inference API usage request response from the database
        async with db_code.Session() as db:
            inference_response = await db.exec(
                select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_request_id == inference_request_id)
            )
            inference_response = inference_response.one_or_none()
        # Ensure burn address is tracked by local wallet:
        burn_address_already_imported = await check_if_address_is_already_imported_in_local_wallet(burn_address)
        if not burn_address_already_imported:
            await import_address_func(burn_address, "burn_address", True)        
        # Check burn address for tracking transaction:
        confirmation_transaction_txid = confirmation_transaction.confirmation_transaction['txid']
        credit_usage_tracking_amount_in_psl = float(inference_response.request_confirmation_message_amount_in_patoshis)/(10**5) # Divide by number of Patoshis per PSL
        matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_address_for_tracking_transaction(burn_address, inference_response.credit_usage_tracking_psl_address, credit_usage_tracking_amount_in_psl, confirmation_transaction_txid, inference_response.max_block_height_to_include_confirmation_transaction)
        if matching_transaction_found:
            logger.info(f"Found correct inference request confirmation tracking transaction in burn address (with {num_confirmations} confirmation blocks so far)! TXID: {confirmation_transaction_txid}; Tracking Amount in PSL: {credit_usage_tracking_amount_in_psl};") 
            credit_pack = ""
            computed_current_credit_pack_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address = await determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack, burn_address)
            logger.info(f"Computed current credit pack balance: {computed_current_credit_pack_balance} based on {number_of_confirmation_transactions_from_tracking_address_to_burn_address} tracking transactions from tracking address to burn address.")       
            # Update the inference request status to "confirmed"
            inference_request.status = "confirmed"
            async with db_code.Session() as db:
                db.add(inference_request)
                await db.commit()
                await db.refresh(inference_request)
            # Trigger the inference request processing
            asyncio.create_task(execute_inference_request(inference_request_id))
            return True
        else:
            logger.error(f"Did not find correct inference request confirmation tracking transaction in burn address! TXID: {confirmation_transaction_txid}; Tracking Amount in PSL: {credit_usage_tracking_amount_in_psl};") 
    except Exception as e:
        logger.error(f"Error processing inference confirmation: {str(e)}")
        raise

async def save_inference_output_results(inference_request_id: str, inference_response_id: str, output_results: dict, output_results_file_type_strings: dict) -> None:
    try:
        _, _, local_supernode_pastelid, _ = await get_local_machine_supernode_data_func()
        # Generate a unique identifier for the inference result
        inference_result_id = str(uuid.uuid4())
        # Create an inference output result record without the hash and signature fields
        inference_output_result = db_code.InferenceAPIOutputResult(
            inference_result_id=inference_result_id,
            inference_request_id=inference_request_id,
            inference_response_id=inference_response_id,
            responding_supernode_pastelid=local_supernode_pastelid,
            inference_result_json_base64=base64.b64encode(json.dumps(output_results).encode("utf-8")).decode("utf-8"),
            inference_result_file_type_strings=json.dumps(output_results_file_type_strings),
            inference_result_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            inference_result_pastel_block_height=await get_current_pastel_block_height_func(),
            inference_result_message_version_string="1.0"
        )
        # Generate the hash and signature fields
        inference_output_result.sha3_256_hash_of_inference_result_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(inference_output_result)
        result_id_signature = await sign_message_with_pastelid_func(local_supernode_pastelid, inference_output_result.sha3_256_hash_of_inference_result_fields, LOCAL_PASTEL_ID_PASSPHRASE)
        inference_output_result.responding_supernode_signature_on_inference_result_id = result_id_signature
        # Save the inference output result to the database
        async with db_code.Session() as db:
            db.add(inference_output_result)
            await db.commit()
            await db.refresh(inference_output_result)
    except Exception as e:
        logger.error(f"Error saving inference output results: {str(e)}")
        raise

def get_claude3_model_name(model_name: str) -> str:
    model_mapping = {
        "claude3-haiku": "claude-3-haiku-20240307",
        "claude3-opus": "claude-3-opus-20240229",
        "claude3-sonnet": "claude-3-sonnet-20240229"
    }
    return model_mapping.get(model_name, "")

async def submit_inference_request_to_stability_api(inference_request):
    # Integrate with the Stability API to perform the image generation task
    if inference_request.model_inference_type_string == "text_to_image":
        model_parameters = json.loads(inference_request.model_parameters_json)
        prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        if "stability-core" in inference_request.requested_model_canonical_string:
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client:
                response = await client.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/core",
                    headers={
                        "authorization": f"Bearer {STABILITY_API_KEY}",
                        "accept": "image/*"
                    },
                    files={"none": ''},
                    data={
                        "prompt": prompt,
                        "aspect_ratio": model_parameters.get("aspect_ratio", "1:1"),
                        "negative_prompt": model_parameters.get("negative_prompt", ""),
                        "seed": model_parameters.get("seed", 0),
                        "style_preset": model_parameters.get("style_preset", ""),
                        "output_format": model_parameters.get("output_format", "png"),
                    },
                )
                if response.status_code == 200:
                    output_results = base64.b64encode(response.content).decode("utf-8")
                    output_results_file_type_strings = {
                        "output_text": "base64_image",
                        "output_files": ["NA"]
                    }
                    return output_results, output_results_file_type_strings
                else:
                    logger.error(f"Error generating image from Stability API: {response.text}")
                    return None, None
        else:
            model_parameters = json.loads(inference_request.model_parameters_json)
            prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
            engine_id = inference_request.requested_model_canonical_string
            api_host = "https://api.stability.ai"
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client:
                response = await client.post(
                    f"{api_host}/v1/generation/{engine_id}/text-to-image",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {STABILITY_API_KEY}"
                    },
                    json={
                        "text_prompts": [{"text": prompt}],
                        "cfg_scale": model_parameters.get("cfg_scale", 7),
                        "height": model_parameters.get("height", 512),
                        "width": model_parameters.get("width", 512),
                        "samples": model_parameters.get("num_samples", 1),
                        "steps": model_parameters.get("steps", 50),
                        "style_preset": model_parameters.get("style_preset", None),
                    },
                )
                if response.status_code == 200:
                    data = response.json()
                    if "artifacts" in data and len(data["artifacts"]) > 0:
                        for artifact in data["artifacts"]:
                            if artifact.get("finishReason") == "SUCCESS":
                                output_results = artifact["base64"]
                                output_results_file_type_strings = {
                                    "output_text": "base64_image",
                                    "output_files": ["NA"]
                                }
                                return output_results, output_results_file_type_strings
                        else:
                            logger.warning("No successful artifact found in the Stability API response.")
                            return None, None
                    else:
                        logger.error("No artifacts found in the Stability API response.")
                        return None, None
                else:
                    logger.error(f"Error generating image from Stability API: {response.text}")
                    return None, None
    elif inference_request.model_inference_type_string == "creative_upscale" and "stability-core" in inference_request.requested_model_canonical_string:
        model_parameters = json.loads(inference_request.model_parameters_json)
        input_image = base64.b64decode(inference_request.model_input_data_json_b64)
        prompt = model_parameters.get("prompt", "")
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client:
            response = await client.post(
                "https://api.stability.ai/v2beta/stable-image/upscale/creative",
                headers={
                    "authorization": f"Bearer {STABILITY_API_KEY}",
                    "accept": "application/json"
                },
                files={"image": input_image},
                data={
                    "prompt": prompt,
                    "output_format": model_parameters.get("output_format", "png"),
                    "seed": model_parameters.get("seed", 0),
                    "negative_prompt": model_parameters.get("negative_prompt", ""),
                    "creativity": model_parameters.get("creativity", 0.3),
                },
            )
            if response.status_code == 200:
                generation_id = response.json().get("id")
                while True:
                    await asyncio.sleep(10)  # Wait for 10 seconds before polling for result
                    result_response = await client.get(
                        f"https://api.stability.ai/v2beta/stable-image/upscale/creative/result/{generation_id}",
                        headers={
                            "authorization": f"Bearer {STABILITY_API_KEY}",
                            "accept": "image/*"
                        },
                    )
                    if result_response.status_code == 200:
                        output_results = base64.b64encode(result_response.content).decode("utf-8")
                        output_results_file_type_strings = {
                            "output_text": "base64_image",
                            "output_files": ["NA"]
                        }
                        return output_results, output_results_file_type_strings
                    elif result_response.status_code != 202:
                        logger.error(f"Error retrieving upscaled image: {result_response.text}")
                        return None, None
            else:
                logger.error(f"Error initiating upscale request: {response.text}")
                return None, None
    else:
        logger.warning(f"Unsupported inference type for Stability model: {inference_request.model_inference_type_string}")
        return None, None

async def submit_inference_request_to_openai_api(inference_request):
    # Integrate with the OpenAI API to perform the inference task
    logger.info("Now accessing OpenAI API...")
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(inference_request.model_parameters_json)
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = model_parameters.get("number_of_completions_to_generate", 1)
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": inference_request.requested_model_canonical_string.replace("openai-", ""),
                        "messages": [{"role": "user", "content": input_prompt}],
                        "max_tokens": model_parameters.get("number_of_tokens_to_generate", 1000),
                        "temperature": model_parameters.get("temperature", 0.7),
                        "n": 1
                    }
                )
                if response.status_code == 200:
                    response_json = response.json()
                    output_results.append(response_json["choices"][0]["message"]["content"])
                    total_input_tokens += response_json["usage"]["prompt_tokens"]
                    total_output_tokens += response_json["usage"]["completion_tokens"]
                else:
                    logger.error(f"Error generating text from OpenAI API: {response.text}")
                    return None, None
        logger.info(f"Total input tokens used with {inference_request.requested_model_canonical_string} model: {total_input_tokens}")
        logger.info(f"Total output tokens used with {inference_request.requested_model_canonical_string} model: {total_output_tokens}")
        if num_completions == 1:
            output_text = output_results[0]
        else:
            output_text = json.dumps({f"completion_{i+1:02}": result for i, result in enumerate(output_results)})
        logger.info(f"Generated the following output text using {inference_request.requested_model_canonical_string}: {output_text[:100]} <abbreviated>...")
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    elif inference_request.model_inference_type_string == "embedding":
        input_text = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": inference_request.requested_model_canonical_string.replace("openai-", ""),
                "input": input_text
            }
        )
        if response.status_code == 200:
            output_results = response.json()["data"][0]["embedding"]
            output_results_file_type_strings = {
                "output_text": "embedding",
                "output_files": ["NA"]
            }
            return output_results, output_results_file_type_strings
        else:
            logger.error(f"Error generating embedding from OpenAI API: {response.text}")
            return None, None
    else:
        logger.warning(f"Unsupported inference type for OpenAI model: {inference_request.model_inference_type_string}")
        return None, None
    
async def submit_inference_request_to_openrouter(inference_request):
    logger.info("Now accessing OpenRouter...")
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(inference_request.model_parameters_json)
        messages = [{"role": "user", "content": base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")}]
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": inference_request.requested_model_canonical_string,
                    "messages": messages,
                    "max_tokens": model_parameters.get("number_of_tokens_to_generate", 1000),
                    "temperature": model_parameters.get("temperature", 0.7),
                }
            )
            if response.status_code == 200:
                output_results = response.json()["choices"][0]["message"]["content"]
                result = magika.identify_bytes(output_results.encode("utf-8")) 
                detected_data_type = result.output.ct_label
                output_results_file_type_strings = {
                    "output_text": detected_data_type,
                    "output_files": ["NA"]
                }
                return output_results, output_results_file_type_strings
            else:
                logger.error(f"Error generating text from OpenRouter: {response.text}")
                return None, None
    else:
        logger.warning(f"Unsupported inference type for OpenRouter model: {inference_request.model_inference_type_string}")
        return None, None

async def submit_inference_request_to_mistral_api(inference_request):
    # Integrate with the Mistral API to perform the inference task
    logger.info("Now accessing Mistral API...")
    client = MistralAsyncClient(api_key=MISTRAL_API_KEY)
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(inference_request.model_parameters_json)
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = model_parameters.get("number_of_completions_to_generate", 1)
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            messages = [ChatMessage(role="user", content=input_prompt)]
            async_response = client.chat_stream(
                model=inference_request.requested_model_canonical_string.replace("mistralapi-",""),
                messages=messages,
                max_tokens=model_parameters.get("number_of_tokens_to_generate", 1000),
                temperature=model_parameters.get("temperature", 0.7),
            )
            completion_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            async for chunk in async_response:
                if chunk.choices[0].delta.content:
                    completion_text += chunk.choices[0].delta.content
                    completion_tokens += 1
                else:
                    prompt_tokens += 1
            output_results.append(completion_text)
            total_input_tokens += prompt_tokens
            total_output_tokens += completion_tokens
        logger.info(f"Total input tokens used with {inference_request.requested_model_canonical_string} model: {total_input_tokens}")
        logger.info(f"Total output tokens used with {inference_request.requested_model_canonical_string} model: {total_output_tokens}")
        if num_completions == 1:
            output_text = output_results[0]
        else:
            output_text = json.dumps({f"completion_{i+1:02}": result for i, result in enumerate(output_results)})
        logger.info(f"Generated the following output text using {inference_request.requested_model_canonical_string}: {output_text[:100]} <abbreviated>...")
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    elif inference_request.model_inference_type_string == "embedding":
        input_text = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        embeddings_batch_response = client.embeddings(
            model=inference_request.requested_model_canonical_string,
            input=[input_text],
        )
        output_results = embeddings_batch_response.data[0].embedding
        output_results_file_type_strings = {
            "output_text": "embedding",
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    else:
        logger.warning(f"Unsupported inference type for Mistral model: {inference_request.model_inference_type_string}")
        return None, None
    
async def submit_inference_request_to_groq_api(inference_request):
    # Integrate with the Groq API to perform the inference task
    logger.info("Now accessing Groq API...")
    client = AsyncGroq(api_key=GROQ_API_KEY)
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(inference_request.model_parameters_json)
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = model_parameters.get("number_of_completions_to_generate", 1)
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": input_prompt}],
                model=inference_request.requested_model_canonical_string.replace("groq-",""),
                max_tokens=model_parameters.get("number_of_tokens_to_generate", 1000),
                temperature=model_parameters.get("temperature", 0.7),
            )
            output_results.append(chat_completion.choices[0].message.content)
            total_input_tokens += chat_completion.usage.prompt_tokens
            total_output_tokens += chat_completion.usage.completion_tokens
        logger.info(f"Total input tokens used with {inference_request.requested_model_canonical_string} model: {total_input_tokens}")
        logger.info(f"Total output tokens used with {inference_request.requested_model_canonical_string} model: {total_output_tokens}")
        if num_completions == 1:
            output_text = output_results[0]
        else:
            output_text = json.dumps({f"completion_{i+1:02}": result for i, result in enumerate(output_results)})
        logger.info(f"Generated the following output text using {inference_request.requested_model_canonical_string}: {output_text[:100]} <abbreviated>...")
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    elif inference_request.model_inference_type_string == "embedding":
        input_text = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        response = await client.embed(
            input=input_text,
            model=inference_request.requested_model_canonical_string[5:],
        )
        output_results = response.embedding
        output_results_file_type_strings = {
            "output_text": "embedding",
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    else:
        logger.warning(f"Unsupported inference type for Groq model: {inference_request.model_inference_type_string}")
        return None, None

async def submit_inference_request_to_claude_api(inference_request):
    # Integrate with the Claude API to perform the inference task
    logger.info("Now accessing Claude (Anthropic) API...")
    client = anthropic.AsyncAnthropic(api_key=CLAUDE3_API_KEY)
    claude3_model_id_string = get_claude3_model_name(inference_request.requested_model_canonical_string)
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(inference_request.model_parameters_json)
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = model_parameters.get("number_of_completions_to_generate", 1)
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            async with client.messages.stream(
                model=claude3_model_id_string,
                max_tokens=model_parameters.get("number_of_tokens_to_generate", 1000),
                temperature=model_parameters.get("temperature", 0.7),
                messages=[{"role": "user", "content": input_prompt}],
            ) as stream:
                message = await stream.get_final_message()
                output_results.append(message.content[0].text)
                total_input_tokens += message.usage.input_tokens
                total_output_tokens += message.usage.output_tokens
        logger.info(f"Total input tokens used with {claude3_model_id_string} model: {total_input_tokens}")
        logger.info(f"Total output tokens used with {claude3_model_id_string} model: {total_output_tokens}")
        if num_completions == 1:
            output_text = output_results[0]
        else:
            output_text = json.dumps({f"completion_{i+1:02}": result for i, result in enumerate(output_results)})
        logger.info(f"Generated the following output text using {claude3_model_id_string}: {output_text[:100]} <abbreviated>...")
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    elif inference_request.model_inference_type_string == "embedding":
        input_text = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        response = await client.embed(
            text=input_text,
            model=claude3_model_id_string
        )
        output_results = response.embedding
        output_results_file_type_strings = {
            "output_text": "embedding",
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    else:
        logger.warning(f"Unsupported inference type for Claude3 Haiku: {inference_request.model_inference_type_string}")
        return None, None

async def submit_inference_request_to_swiss_army_llama(inference_request):
    logger.info(f"Now calling Swiss Army Llama with model {inference_request.requested_model_canonical_string}")
    model_parameters = json.loads(inference_request.model_parameters_json)
    async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*12)) as client:
        if inference_request.model_inference_type_string == "text_completion":
            payload = {
                "input_prompt": base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8"),
                "llm_model_name": inference_request.requested_model_canonical_string,
                "temperature": model_parameters.get("temperature", 0.7),
                "number_of_tokens_to_generate": model_parameters.get("number_of_tokens_to_generate", 1000),
                "number_of_completions_to_generate": model_parameters.get("number_of_completions_to_generate", 1),
                "grammar_file_string": model_parameters.get("grammar_file_string", ""),
            }
            response = await client.post(
                f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/get_text_completions_from_input_prompt/",
                json=payload,
                params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
            )
            if response.status_code == 200:
                output_results = response.json()
                output_text = output_results[0]["generated_text"]
                result = magika.identify_bytes(output_text.encode("utf-8"))
                detected_data_type = result.output.ct_label
                output_results_file_type_strings = {
                    "output_text": detected_data_type,
                    "output_files": ["NA"]
                }
                return output_results, output_results_file_type_strings
            else:
                logger.error(f"Failed to execute text completion inference request: {response.text}")
                return None, None
        elif inference_request.model_inference_type_string == "embedding":
            payload = {
                "text": base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8"),
                "llm_model_name": inference_request.requested_model_canonical_string
            }
            response = await client.post(
                f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/get_embedding_vector_for_string/",
                json=payload,
                params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
            )
            if response.status_code == 200:
                output_results = response.json()
                output_results_file_type_strings = {
                    "output_text": "embedding",
                    "output_files": ["NA"]
                }
                return output_results, output_results_file_type_strings
            else:
                logger.error(f"Failed to execute embedding inference request: {response.text}")
                return None, None
        elif inference_request.model_inference_type_string == "token_level_embedding":
            payload = {
                "text": base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8"),
                "llm_model_name": inference_request.requested_model_canonical_string
            }
            response = await client.post(
                f"http://localhost:{SWISS_ARMY_LLAMA_PORT}/get_token_level_embeddings_matrix_and_combined_feature_vector_for_string/",
                json=payload,
                params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN, "send_back_json_or_zip_file": "json"}
            )
            if response.status_code == 200:
                output_results = response.json()
                output_results_file_type_strings = {
                    "output_text": "token_level_embedding",
                    "output_files": ["NA"]
                }
                return output_results, output_results_file_type_strings
            else:
                logger.error(f"Failed to execute token level embedding inference request: {response.text}")
                return None, None
        else:
            logger.warning(f"Unsupported inference type: {inference_request.model_inference_type_string}")
            return None, None    

async def execute_inference_request(inference_request_id: str) -> None:
    try:
        # Retrieve the inference API usage request from the database
        async with db_code.Session() as db:
            inference_request = db.exec(
                select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
            ).one_or_none()
        if inference_request is None:
            logger.warning(f"Invalid inference request ID: {inference_request_id}")
            return
        # Retrieve the inference API usage request response from the database
        async with db_code.Session() as db:
            inference_response = db.exec(
                select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_request_id == inference_request_id)
            ).one_or_none()

        if inference_request.requested_model_canonical_string.startswith("stability-"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_stability_api(inference_request)
        elif inference_request.requested_model_canonical_string.startswith("openai-"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_openai_api(inference_request)
        elif inference_request.requested_model_canonical_string.startswith("mistralapi-"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_mistral_api(inference_request)
        elif inference_request.requested_model_canonical_string.startswith("groq-"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_groq_api(inference_request)
        elif "claude" in inference_request.requested_model_canonical_string.lower():
            output_results, output_results_file_type_strings = await submit_inference_request_to_claude_api(inference_request)
        elif inference_request.requested_model_canonical_string.startswith("openrouter/"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_openrouter(inference_request)
        else:
            output_results, output_results_file_type_strings = await submit_inference_request_to_swiss_army_llama(inference_request)

        if output_results is not None and output_results_file_type_strings is not None:
            # Save the inference output results to the database
            await save_inference_output_results(inference_request_id, inference_response.inference_response_id, output_results, output_results_file_type_strings)
    except Exception as e:
        logger.error(f"Error executing inference request: {str(e)}")
        raise

async def check_status_of_inference_request_results(inference_response_id: str) -> bool:
    async with db_code.Session() as db_session:
        # Retrieve the inference output result
        inference_output_result = db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_response_id)
        ).one_or_none()
        if inference_output_result is None:
            return False
        else:
            return True
    
async def get_inference_output_results_and_verify_authorization(inference_response_id: str, requesting_pastelid: str) -> db_code.InferenceAPIOutputResult:
    async with db_code.Session() as db_session:
        # Retrieve the inference output result
        inference_output_result = db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_response_id)
        ).one_or_none()
        if inference_output_result is None:
            raise ValueError("Inference output results not found")
        # Retrieve the inference request to verify requesting PastelID
        inference_request = db_session.exec(
            select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_output_result.inference_request_id)
        ).one_or_none()
        if inference_request is None or inference_request.requesting_pastelid != requesting_pastelid:
            raise ValueError("Unauthorized access to inference output results")
        return inference_output_result

async def determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack: db_code.CreditPackPurchaseRequestResponse, burn_address: str):
    """
    Determines the current credit balance of an inference credit pack based on the tracking transactions.

    Args:
        credit_pack: The inference credit pack object containing the initial credit balance and tracking PSL address.
        burn_address: The burn address to which the tracking transactions are sent.

    Returns:
        float: The current credit balance of the inference credit pack.
        int: The number of confirmation transactions from the tracking address to the burn address.
    """
    global rpc_connection
    credit_pack_purchase_request_object = await get_credit_pack_purchase_request_from_response(credit_pack)
    initial_credit_balance = credit_pack_purchase_request_object.requested_initial_credits_in_credit_pack
    credit_usage_tracking_psl_address = credit_pack.credit_usage_tracking_psl_address
    # Get all transactions sent to the burn address
    transactions = await rpc_connection.listtransactions("*", 100000, 0, True)
    burn_address_transactions = [
        tx for tx in transactions
        if tx.get("address") == burn_address and tx.get("category") == "receive" and tx.get("amount") < 1.0
    ]
    burn_address_txids = [tx.get("txid") for tx in burn_address_transactions]
    number_of_burn_address_txids = len(burn_address_txids)
    logger.info(f"Number of transactions sent to the burn address with amount < 1.0 PSL: {number_of_burn_address_txids}")
    # Fetch and decode raw transactions in parallel using asyncio.gather
    decoded_tx_data_list = await asyncio.gather(*[get_and_decode_raw_transaction(txid) for txid in burn_address_txids])
    # Filter the tracking transactions to include only those sent from the credit_usage_tracking_psl_address
    tracking_transactions = []
    for decoded_tx_data in decoded_tx_data_list:
        decoded_tx_data_as_string = json.dumps(decoded_tx_data)
        if credit_usage_tracking_psl_address in decoded_tx_data_as_string:
            tracking_transactions.append(decoded_tx_data)
    number_of_confirmation_transactions_from_tracking_address_to_burn_address = len(tracking_transactions)
    # Calculate the total number of inference credits consumed
    total_credits_consumed = 0
    for tx in tracking_transactions:
        for vout in tx.get("vout", []):
            if vout.get("scriptPubKey", {}).get("addresses", [None])[0] == burn_address and vout["value"] < 1.0:
                total_credits_consumed += float(vout["valuePat"]) / CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER
    # Calculate the current credit balance
    current_credit_balance = initial_credit_balance - total_credits_consumed
    logger.info(f"Initial credit balance: {initial_credit_balance}")
    logger.info(f"Total credits consumed: {total_credits_consumed}")
    logger.info(f"Current credit balance: {current_credit_balance}")
    return current_credit_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address

async def update_inference_sn_reputation_score(supernode_pastelid: str, reputation_score: float) -> bool:
    try:
        # TODO: Implement the logic to update the inference SN reputation score
        # This could involve storing the reputation score in a database or broadcasting it to other supernodes
        # For now, let's assume the reputation score is updated successfully
        return True
    except Exception as e:
        logger.error(f"Error updating inference SN reputation score: {str(e)}")
        raise
    
async def get_inference_api_usage_request_for_audit(inference_request_id: str) -> db_code.InferenceAPIUsageRequest:
    async with db_code.Session() as db_session:
        result = db_session.exec(
            select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
        ).one_or_none()
        return result
        
async def get_inference_api_usage_response_for_audit(inference_response_id: str) -> db_code.InferenceAPIUsageResponse:
    async with db_code.Session() as db_session:
        result = db_session.exec(
            select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_response_id == inference_response_id)
        ).one_or_none()
        return result

async def get_inference_api_usage_result_for_audit(inference_response_id: str) -> db_code.InferenceAPIOutputResult:
    async with db_code.Session() as db_session:
        result = db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_response_id)
        ).one_or_none()
        return result
# ________________________________________________________________________________________________________________________________

# Blockchain ticket related functions:

def check_if_transparent_psl_address_is_valid_func(pastel_address_string):
    if len(pastel_address_string) == 35 and (pastel_address_string[0:2] == 'Pt'):
        pastel_address_is_valid = 1
    else:
        pastel_address_is_valid = 0
    return pastel_address_is_valid

def check_if_transparent_lsp_address_is_valid_func(pastel_address_string):
    if len(pastel_address_string) == 35 and (pastel_address_string[0:2] == 'tP'):
        pastel_address_is_valid = 1
    else:
        pastel_address_is_valid = 0
    return pastel_address_is_valid

async def get_df_json_from_tickets_list_rpc_response_func(rpc_response):
    tickets_df = pd.DataFrame.from_records([rpc_response[idx]['ticket'] for idx, x in enumerate(rpc_response)])
    tickets_df['txid'] = [rpc_response[idx]['txid'] for idx, x in enumerate(rpc_response)]
    tickets_df['height'] = [rpc_response[idx]['height'] for idx, x in enumerate(rpc_response)]
    tickets_df_json = tickets_df.to_json(orient='index')
    return tickets_df_json

async def get_pastel_blockchain_ticket_func(txid):
    global rpc_connection
    response_json = await rpc_connection.tickets('get', txid )
    if len(response_json) > 0:
        ticket_type_string = response_json['ticket']['type']
        corresponding_reg_ticket_block_height = response_json['height']
        latest_block_height = await get_current_pastel_block_height_func()
        if int(corresponding_reg_ticket_block_height) < 0:
            logger.warning(f'The corresponding reg ticket block height of {corresponding_reg_ticket_block_height} is less than 0!')
        if int(corresponding_reg_ticket_block_height) > int(latest_block_height):
            logger.info(f'The corresponding reg ticket block height of {corresponding_reg_ticket_block_height} is greater than the latest block height of {latest_block_height}!')
        corresponding_reg_ticket_block_info = await rpc_connection.getblock(str(corresponding_reg_ticket_block_height))
        corresponding_reg_ticket_block_timestamp = corresponding_reg_ticket_block_info['time']
        corresponding_reg_ticket_block_timestamp_utc_iso = datetime.utcfromtimestamp(corresponding_reg_ticket_block_timestamp).isoformat()
        response_json['reg_ticket_block_timestamp_utc_iso'] = corresponding_reg_ticket_block_timestamp_utc_iso
        if ticket_type_string == 'nft-reg':
            activation_response_json = await rpc_connection.tickets('find', 'act', txid )
        elif ticket_type_string == 'action-reg':
            activation_response_json = await rpc_connection.tickets('find', 'action-act', txid )
        elif ticket_type_string == 'collection-reg':
            activation_response_json = await rpc_connection.tickets('find', 'collection-act', txid )
        else:
            activation_response_json = f'No activation ticket needed for this ticket type ({ticket_type_string})'
        if len(activation_response_json) > 0:
            response_json['activation_ticket'] = activation_response_json
        else:
            response_json['activation_ticket'] = 'No activation ticket found for this ticket-- check again soon'
        return response_json
    else:
        response_json = 'No ticket found for this txid'
    return response_json

async def get_all_pastel_blockchain_tickets_func(verbose=0):
    if verbose:
        logger.info('Now retrieving all Pastel blockchain tickets...')
    tickets_obj = {}
    list_of_ticket_types = ['id', 'nft', 'offer', 'accept', 'transfer', 'royalty', 'username', 'ethereumaddress', 'action', 'action-act'] # 'collection', 'collection-act'
    for current_ticket_type in list_of_ticket_types:
        if verbose:
            logger.info('Getting ' + current_ticket_type + ' tickets...')
        response = await rpc_connection.tickets('list', current_ticket_type)
        if response is not None and len(response) > 0:
            tickets_obj[current_ticket_type] = await get_df_json_from_tickets_list_rpc_response_func(response)
    return tickets_obj

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

async def get_and_decode_raw_transaction(txid: str) -> dict:
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
        raw_tx_data = await rpc_connection.getrawtransaction(txid)
        if not raw_tx_data:
            logger.error(f"Failed to retrieve raw transaction data for {txid}")
            return {}
        # Decode the raw transaction data
        decoded_tx_data = await rpc_connection.decoderawtransaction(raw_tx_data)
        if not decoded_tx_data:
            logger.error(f"Failed to decode raw transaction data for {txid}")
            return {}
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

#Misc helper functions:
class MyTimer():
    def __init__(self):
        self.start = time.time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        end = time.time()
        runtime = end - self.start
        msg = '({time} seconds to complete)'
        logger.info(msg.format(time=round(runtime, 2)))

def compute_elapsed_time_in_minutes_between_two_datetimes_func(start_datetime, end_datetime):
    time_delta = (end_datetime - start_datetime)
    total_seconds_elapsed = time_delta.total_seconds()
    total_minutes_elapsed = total_seconds_elapsed / 60
    return total_minutes_elapsed

def compute_elapsed_time_in_minutes_since_start_datetime_func(start_datetime):
    end_datetime = datetime.utcnow()
    total_minutes_elapsed = compute_elapsed_time_in_minutes_between_two_datetimes_func(start_datetime, end_datetime)
    return total_minutes_elapsed

def check_if_ip_address_is_valid_func(ip_address_string):
    try:
        _ = ipaddress.ip_address(ip_address_string)
        ip_address_is_valid = 1
    except Exception as e:
        logger.error('Validation Error: ' + str(e))
        ip_address_is_valid = 0
    return ip_address_is_valid

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
        difference_in_seconds <= MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS
    )
    if not datetimes_are_close_enough_to_consider_them_matching:
        logger.warning(f"Timestamps are too far apart: {difference_in_seconds} seconds")
    return difference_in_seconds, datetimes_are_close_enough_to_consider_them_matching

def get_sha256_hash_of_input_data_func(input_data_or_string):
    if isinstance(input_data_or_string, str):
        input_data_or_string = input_data_or_string.encode('utf-8')
    sha256_hash_of_input_data = hashlib.sha3_256(input_data_or_string).hexdigest()
    return sha256_hash_of_input_data

async def extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance: SQLModel) -> str:
    response_fields = {}
    last_hash_field_name = None
    last_signature_field_names = []
    # Find the last hash field and all signature fields
    for field_name in model_instance.__fields__.keys():
        if field_name.startswith("sha3_256_hash_of"):
            last_hash_field_name = field_name
        elif "_signature_on_" in field_name:
            last_signature_field_names.append(field_name)
    # Determine which fields to exclude based on the model type
    if isinstance(model_instance, db_code.CreditPackPurchasePriceAgreementRequestResponse):
        fields_to_exclude = [last_hash_field_name, 'id'] + last_signature_field_names
    else:
        fields_to_exclude = [last_hash_field_name, last_signature_field_names[-1], 'id']
    # Iterate over the model fields and exclude the specified fields and those containing '_sa_instance_state'
    for field_name, field_value in model_instance.__dict__.items():
        if field_name in fields_to_exclude or '_sa_instance_state' in field_name:
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
    last_hash_field_name = None
    for field_name in model_instance.__fields__:
        if field_name.startswith("sha3_256_hash_of") and field_name.endswith("_fields"):
            last_hash_field_name = field_name
    if last_hash_field_name:
        actual_hash = getattr(model_instance, last_hash_field_name)
        if actual_hash != expected_hash:
            validation_errors.append(f"SHA3-256 hash in field {last_hash_field_name} does not match the computed hash of the response fields")
    # Validate pastelid signature fields
    if isinstance(model_instance, db_code.CreditPackPurchasePriceAgreementRequestResponse):
        signature_field_names = []
        hash_field_name = None
        for field_name in model_instance.__fields__:
            if "_pastelid" in field_name:
                first_pastelid = field_name
                break
        for field_name in model_instance.__fields__:
            if "_signature_on_" in field_name:
                signature_field_names.append(field_name)
            elif "sha3_256_hash_of_" in field_name and field_name.endswith("_fields"):
                hash_field_name = field_name
        if signature_field_names and hash_field_name:
            if hasattr(model_instance, first_pastelid):
                pastelid = getattr(model_instance, first_pastelid)
                for signature_field_name in signature_field_names:
                    if signature_field_name == "responding_supernode_signature_on_price_agreement_request_response_hash":
                        message_to_verify = getattr(model_instance, "sha3_256_hash_of_price_agreement_request_response_fields")
                    elif signature_field_name == "responding_supernode_signature_on_credit_pack_purchase_request_fields_json":
                        message_to_verify = getattr(model_instance, "credit_pack_purchase_request_fields_json")
                    else:
                        continue
                    signature = getattr(model_instance, signature_field_name)
                    verification_result = await verify_message_with_pastelid_func(pastelid, message_to_verify, signature)
                    if verification_result != 'OK':
                        validation_errors.append(f"Pastelid signature in field {signature_field_name} failed verification")
            else:
                validation_errors.append(f"Corresponding pastelid field {first_pastelid} not found for signature fields {signature_field_names}")
    else:
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
            if hasattr(model_instance, first_pastelid):
                pastelid = getattr(model_instance, first_pastelid)
                message_to_verify = getattr(model_instance, hash_field_name)
                signature = getattr(model_instance, signature_field_name)
                verification_result = await verify_message_with_pastelid_func(pastelid, message_to_verify, signature)
                if verification_result != 'OK':
                    validation_errors.append(f"Pastelid signature in field {signature_field_name} failed verification")
            else:
                validation_errors.append(f"Corresponding pastelid field {first_pastelid} not found for signature field {signature_field_name}")
    return validation_errors

async def validate_inference_request_message_data_func(model_instance: SQLModel):
    validation_errors = await validate_credit_pack_ticket_message_data_func(model_instance)
    return validation_errors

def get_external_ip_func():
    response = httpx.get("https://ipinfo.io/ip")
    response.raise_for_status()
    return response.text.strip()

def safe_highlight_func(text, pattern, replacement):
    try:
        return re.sub(pattern, replacement, text)
    except Exception as e:
        logger.warning(f"Failed to apply highlight rule: {e}")
        return text

def highlight_rules_func(text):
    rules = [
        (re.compile(r"\b(success\w*)\b", re.IGNORECASE), '#COLOR1_OPEN#', '#COLOR1_CLOSE#'),
        (re.compile(r"\b(error|fail\w*)\b", re.IGNORECASE), '#COLOR2_OPEN#', '#COLOR2_CLOSE#'),
        (re.compile(r"\b(pending)\b", re.IGNORECASE), '#COLOR3_OPEN#', '#COLOR3_CLOSE#'),
        (re.compile(r"\b(response)\b", re.IGNORECASE), '#COLOR4_OPEN#', '#COLOR4_CLOSE#'),
        (re.compile(r'\"(.*?)\"', re.IGNORECASE), '#COLOR5_OPEN#', '#COLOR5_CLOSE#'),
        (re.compile(r"\'(.*?)\'", re.IGNORECASE), "#COLOR6_OPEN#", '#COLOR6_CLOSE#'),
        (re.compile(r"\`(.*?)\`", re.IGNORECASE), '#COLOR7_OPEN#', '#COLOR7_CLOSE#'),
        (re.compile(r"\b(https?://\S+)\b", re.IGNORECASE), '#COLOR8_OPEN#', '#COLOR8_CLOSE#'),
        (re.compile(r"\b(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2},\d{3})\b", re.IGNORECASE), '#COLOR9_OPEN#', '#COLOR9_CLOSE#'),
        (re.compile(r"\b(_{100,})\b", re.IGNORECASE), '#COLOR10_OPEN#', '#COLOR10_CLOSE#'),
        (re.compile(r"\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)\b", re.IGNORECASE), '#COLOR11_OPEN#', '#COLOR11_CLOSE#'),
        (re.compile(r"\b([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\b", re.IGNORECASE), '#COLOR12_OPEN#', '#COLOR12_CLOSE#'),
        (re.compile(r"\b([a-f0-9]{64})\b", re.IGNORECASE), '#COLOR13_OPEN#', '#COLOR13_CLOSE#')                                
    ]
    for pattern, replacement_open, replacement_close in rules:
        text = pattern.sub(f"{replacement_open}\\1{replacement_close}", text)
    text = html.escape(text)
    text = text.replace('#COLOR1_OPEN#', '<span style="color: #baffc9;">').replace('#COLOR1_CLOSE#', '</span>')
    text = text.replace('#COLOR2_OPEN#', '<span style="color: #ffb3ba;">').replace('#COLOR2_CLOSE#', '</span>')
    text = text.replace('#COLOR3_OPEN#', '<span style="color: #ffdfba;">').replace('#COLOR3_CLOSE#', '</span>')
    text = text.replace('#COLOR4_OPEN#', '<span style="color: #ffffba;">').replace('#COLOR4_CLOSE#', '</span>')
    text = text.replace('#COLOR5_OPEN#', '<span style="color: #bdc7e7;">').replace('#COLOR5_CLOSE#', '</span>')
    text = text.replace('#COLOR6_OPEN#', "<span style='color: #d5db9c;'>").replace('#COLOR6_CLOSE#', '</span>')
    text = text.replace('#COLOR7_OPEN#', '<span style="color: #a8d8ea;">').replace('#COLOR7_CLOSE#', '</span>')
    text = text.replace('#COLOR8_OPEN#', '<span style="color: #e2a8a8;">').replace('#COLOR8_CLOSE#', '</span>')
    text = text.replace('#COLOR9_OPEN#', '<span style="color: #ece2d0;">').replace('#COLOR9_CLOSE#', '</span>')
    text = text.replace('#COLOR10_OPEN#', '<span style="color: #d6e0f0;">').replace('#COLOR10_CLOSE#', '</span>')
    text = text.replace('#COLOR11_OPEN#', '<span style="color: #f2d2e2;">').replace('#COLOR11_CLOSE#', '</span>')
    text = text.replace('#COLOR12_OPEN#', '<span style="color: #d5f2ea;">').replace('#COLOR12_CLOSE#', '</span>')
    text = text.replace('#COLOR13_OPEN#', '<span style="color: #f2ebd3;">').replace('#COLOR13_CLOSE#', '</span>')
    return text


#_______________________________________________________________


rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")

if rpc_port == '9932':
    burn_address = 'PtpasteLBurnAddressXXXXXXXXXXbJ5ndd'
elif rpc_port == '19932':
    burn_address = 'tPpasteLBurnAddressXXXXXXXXXXX3wy7u'
elif rpc_port == '29932':
    burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7' # https://blockchain-devel.slack.com/archives/C03Q2MCQG9K/p1705896449986459

users_credit_tracking_psl_address = '44oSueBgdMaAnGxrbTNZwQeDnxvPJg4dGAR3'

encryption_key = generate_or_load_encryption_key_sync()  # Generate or load the encryption key synchronously    
decrypt_sensitive_fields()
MY_PASTELID = asyncio.run(get_my_local_pastelid_func())
logger.info(f"Using local PastelID: {MY_PASTELID}")

use_encrypt_new_secrets = 0
if use_encrypt_new_secrets:
    encrypted_openai_key = encrypt_sensitive_data("abc123", encryption_key)
    print(f"Encrypted OpenAI key: {encrypted_openai_key}")
    
    encrypted_groq_key = encrypt_sensitive_data("abc123", encryption_key)
    print(f"Encrypted groq key: {encrypted_groq_key}")

    encrypted_mistral_key = encrypt_sensitive_data("abc123", encryption_key)
    print(f"Encrypted mistral key: {encrypted_mistral_key}")
    
    encrypted_stability_key = encrypt_sensitive_data("abc123", encryption_key)
    print(f"Encrypted stability key: {encrypted_stability_key}")    
    
    encrypted_openrouter_key = encrypt_sensitive_data("abc123", encryption_key)
    print(f"Encrypted openrouter key: {encrypted_openrouter_key}")
    
use_test_market_price_data = 0
if use_test_market_price_data:
    current_psl_price = asyncio.run(fetch_current_psl_market_price())
    
use_get_inference_model_menu_on_start = 0
if use_get_inference_model_menu_on_start:
    random_async_wait_duration_in_seconds = round(random.random()*10.0, 3)
    logger.info(f"Checking API keys and getting inference model menu (but first waiting for a random period of {random_async_wait_duration_in_seconds} seconds to not overwhelm the APIs)...")
    asyncio.run(asyncio.sleep(random_async_wait_duration_in_seconds))
    use_verbose=1
    asyncio.run(get_inference_model_menu(use_verbose))