import asyncio
import base64
import decimal
import hashlib
import ipaddress
import json
import os
import io
import platform
import math
import statistics
import time
import uuid
import socket
import subprocess
import random
import re
import sys
import traceback
import html
import tempfile
import warnings
import pytz
from pathlib import Path
from collections.abc import Iterable
from urllib.parse import quote_plus, unquote_plus
from datetime import datetime, timedelta, date
import datetime as dt
import pandas as pd
import httpx
from httpx import Timeout
from urllib.parse import urlparse
from logger_config import logger
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
from sqlmodel import select, delete, update, func, or_, SQLModel
from sqlalchemy.exc import IntegrityError
from sshtunnel import SSHTunnelForwarder, BaseSSHTunnelForwarderError
from mutagen import File as MutagenFile
from PIL import Image

encryption_key = None
magika = Magika()

SENSITIVE_ENV_FIELDS = ["LOCAL_PASTEL_ID_PASSPHRASE", "SWISS_ARMY_LLAMA_SECURITY_TOKEN", "OPENAI_API_KEY", "CLAUDE3_API_KEY", "GROQ_API_KEY", "MISTRAL_API_KEY", "STABILITY_API_KEY", "OPENROUTER_API_KEY"]
LOCAL_PASTEL_ID_PASSPHRASE = None
SWISS_ARMY_LLAMA_SECURITY_TOKEN = None
OPENAI_API_KEY = None
CLAUDE3_API_KEY = None
GROQ_API_KEY = None
MISTRAL_API_KEY = None
STABILITY_API_KEY = None
OPENROUTER_API_KEY = None

def get_local_ip():
    hostname = socket.gethostname()
    return socket.gethostbyname(hostname)

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
    global LOCAL_PASTEL_ID_PASSPHRASE, SWISS_ARMY_LLAMA_SECURITY_TOKEN, OPENAI_API_KEY, CLAUDE3_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, STABILITY_API_KEY, OPENROUTER_API_KEY, encryption_key
    LOCAL_PASTEL_ID_PASSPHRASE = decrypt_sensitive_data(get_env_value("LOCAL_PASTEL_ID_PASSPHRASE"), encryption_key)
    SWISS_ARMY_LLAMA_SECURITY_TOKEN = decrypt_sensitive_data(get_env_value("SWISS_ARMY_LLAMA_SECURITY_TOKEN"), encryption_key)
    OPENAI_API_KEY = decrypt_sensitive_data(get_env_value("OPENAI_API_KEY"), encryption_key)
    CLAUDE3_API_KEY = decrypt_sensitive_data(get_env_value("CLAUDE3_API_KEY"), encryption_key)
    GROQ_API_KEY = decrypt_sensitive_data(get_env_value("GROQ_API_KEY"), encryption_key)
    MISTRAL_API_KEY = decrypt_sensitive_data(get_env_value("MISTRAL_API_KEY"), encryption_key)
    STABILITY_API_KEY = decrypt_sensitive_data(get_env_value("STABILITY_API_KEY"), encryption_key)
    OPENROUTER_API_KEY = decrypt_sensitive_data(get_env_value("OPENROUTER_API_KEY"), encryption_key)
        
number_of_cpus = os.cpu_count()
my_os = platform.system()
loop = asyncio.get_event_loop()
warnings.filterwarnings('ignore')
local_ip = get_local_ip()

config = DecoupleConfig(RepositoryEnv('.env'))
TEMP_OVERRIDE_LOCALHOST_ONLY = config.get("TEMP_OVERRIDE_LOCALHOST_ONLY", default=0, cast=int)
NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE = config.get("NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE", default=3, cast=int)
GITHUB_MODEL_MENU_URL = config.get("GITHUB_MODEL_MENU_URL")
CHALLENGE_EXPIRATION_TIME_IN_SECONDS = config.get("CHALLENGE_EXPIRATION_TIME_IN_SECONDS", default=300, cast=int)
SWISS_ARMY_LLAMA_PORT = config.get("SWISS_ARMY_LLAMA_PORT", default=8089, cast=int)
USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE = config.get("USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE", default=0, cast=int)
REMOTE_SWISS_ARMY_LLAMA_INSTANCE_SSH_KEY_PATH = config.get("REMOTE_SWISS_ARMY_LLAMA_INSTANCE_SSH_KEY_PATH", default="/home/ubuntu/vastai_privkey")
REMOTE_SWISS_ARMY_LLAMA_INSTANCE_IP_ADDRESS = config.get("REMOTE_SWISS_ARMY_LLAMA_INSTANCE_IP_ADDRESS", default="50.20.127.188")
REMOTE_SWISS_ARMY_LLAMA_INSTANCE_PORT = config.get("REMOTE_SWISS_ARMY_LLAMA_INSTANCE_PORT", default=40396, cast=int)
REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT = config.get("REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT", default=8087, cast=int)
REMOTE_SWISS_ARMY_LLAMA_EXPOSED_PORT = config.get("REMOTE_SWISS_ARMY_LLAMA_EXPOSED_PORT", default=8089, cast=int)
CREDIT_COST_MULTIPLIER_FACTOR = config.get("CREDIT_COST_MULTIPLIER_FACTOR", default=0.1, cast=float)
BASE_TRANSACTION_AMOUNT = decimal.Decimal(config.get("BASE_TRANSACTION_AMOUNT", default=0.000001, cast=float))
FEE_PER_KB = decimal.Decimal(config.get("FEE_PER_KB", default=0.0001, cast=float))
MESSAGING_TIMEOUT_IN_SECONDS = config.get("MESSAGING_TIMEOUT_IN_SECONDS", default=60, cast=int)
API_KEY_TESTS_FILE = "api_key_tests.json"
API_KEY_TEST_VALIDITY_HOURS = config.get("API_KEY_TEST_VALIDITY_HOURS", default=72, cast=int)
TARGET_VALUE_PER_CREDIT_IN_USD = config.get("TARGET_VALUE_PER_CREDIT_IN_USD", default=0.1, cast=float)
TARGET_PROFIT_MARGIN = config.get("TARGET_PROFIT_MARGIN", default=0.1, cast=float)
MINIMUM_COST_IN_CREDITS = config.get("MINIMUM_COST_IN_CREDITS", default=0.1, cast=float)
MINUTES_BETWEEN_REFRESHING_SUPERNODE_PING_AND_PORT_RESPONSE_DATA = config.get("MINUTES_BETWEEN_REFRESHING_SUPERNODE_PING_AND_PORT_RESPONSE_DATA", default=3, cast=int)
CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER = config.get("CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER", default=10, cast=int) # Since we always round inference credits to the nearest 0.1, this gives us enough resolution using Patoshis     
MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET = config.get("MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET", default=50, cast=int)
MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING = config.get("MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING", default=0.1, cast=float)
MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS = config.get("MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS", default=15.0, cast=float)
MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS = config.get("MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS", default=1, cast=int)
MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED = config.get("MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED", default=10, cast=int)
MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION = config.get("MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION", default=3, cast=int)
MINIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES = config.get("MINIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES", default=10, cast=int)
MAXIMUM_NUMBER_OF_CONCURRENT_RPC_REQUESTS = config.get("MAXIMUM_NUMBER_OF_CONCURRENT_RPC_REQUESTS", default=30, cast=int)
INDIVIDUAL_SUPERNODE_PRICE_AGREEMENT_REQUEST_TIMEOUT_PERIOD_IN_SECONDS = config.get("INDIVIDUAL_SUPERNODE_PRICE_AGREEMENT_REQUEST_TIMEOUT_PERIOD_IN_SECONDS", default=12, cast=int)
INDIVIDUAL_SUPERNODE_MODEL_MENU_REQUEST_TIMEOUT_PERIOD_IN_SECONDS = config.get("INDIVIDUAL_SUPERNODE_MODEL_MENU_REQUEST_TIMEOUT_PERIOD_IN_SECONDS", default=3, cast=int)
BURN_TRANSACTION_MAXIMUM_AGE_IN_DAYS = config.get("BURN_TRANSACTION_MAXIMUM_AGE_IN_DAYS", default=3, cast=float)
SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE = config.get("SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE", default=0.51, cast=float)
SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE = config.get("SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE", default=0.65, cast=float)
MAXIMUM_CREDITS_PER_CREDIT_PACK = config.get("MAXIMUM_CREDITS_PER_CREDIT_PACK", default=1000000, cast=int)
SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK = 1
UVICORN_PORT = config.get("UVICORN_PORT", default=7123, cast=int)
COIN = 100000 # patoshis in 1 PSL
challenge_store = {}
file_store = {} # In-memory store for files with expiration times

def parse_timestamp(timestamp_str):
    try:
        # Attempt to parse with fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        # Fall back to parsing without fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S')

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

def normalize_data(data):
    if isinstance(data, dict):
        return {key: normalize_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [normalize_data(item) for item in data]
    elif isinstance(data, uuid.UUID):
        return str(data)
    elif isinstance(data, datetime):
        if data.tzinfo is None:
            # Make the datetime offset-aware with UTC timezone
            return data.replace(tzinfo=pytz.UTC)
        else:
            # Convert to UTC timezone
            return data.astimezone(pytz.UTC)
    else:
        return data
    
def format_list(input_list):
    def json_serialize(item):
        if isinstance(item, uuid.UUID):
            return json.dumps(str(item), indent=4)
        elif isinstance(item, dict):
            return json.dumps(pretty_json_func(item), indent=4)
        elif isinstance(item, list):
            return format_list(item)
        else:
            return json.dumps(item, indent=4)
    formatted_list = "[\n" + ",\n".join("    " + json_serialize(item).replace("\n", "\n    ") for item in input_list) + "\n]"
    return formatted_list

def pretty_json_func(data):
    if isinstance(data, SQLModel):
        data = data.dict()  # Convert SQLModel instance to dictionary
    if isinstance(data, dict):
        formatted_data = {}
        for key, value in data.items():
            if isinstance(value, uuid.UUID):  # Convert UUIDs to string
                formatted_data[key] = str(value)
            elif isinstance(value, dict):  # Recursively handle dictionary values
                formatted_data[key] = pretty_json_func(value)
            elif isinstance(value, list):  # Special handling for lists
                formatted_data[key] = format_list(value)
            elif key.endswith("_json"):  # Handle keys that end with '_json'
                formatted_data[key] = parse_and_format(value)
            else:  # Handle other types of values
                formatted_data[key] = value
        return json.dumps(formatted_data, indent=4)
    elif isinstance(data, list):  # Top-level list handling
        return format_list(data)
    elif isinstance(data, str):  # Handle string type data separately
        return parse_and_format(data)
    else:
        return data  # Return data as is if not a dictionary or string
    
def abbreviated_pretty_json_func(data):
    max_payload_length_in_characters = 10000
    formatted_payload = pretty_json_func(data)
    if len(formatted_payload) > max_payload_length_in_characters:
        abbreviated_payload = formatted_payload[:max_payload_length_in_characters] + "..."
        closing_brackets = "]" * (formatted_payload.count("[") - formatted_payload[:max_payload_length_in_characters].count("["))
        closing_brackets += "}" * (formatted_payload.count("{") - formatted_payload[:max_payload_length_in_characters].count("{"))
        abbreviated_payload += closing_brackets
        formatted_payload = abbreviated_payload
    return formatted_payload    
    
def log_action_with_payload(action_string, payload_name, json_payload):
    formatted_payload = abbreviated_pretty_json_func(json_payload)
    logger.info(f"Now {action_string} {payload_name} with payload:\n{formatted_payload}")
    
def get_local_rpc_settings_func(directory_with_pastel_conf=os.path.expanduser("~/.pastel/")):
    with open(os.path.join(directory_with_pastel_conf, "pastel.conf"), 'r') as f:
        lines = f.readlines()
    other_flags = {}
    rpchost = '127.0.0.1'
    rpcport = '19932'
    rpcuser = None
    rpcpassword = None
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):  # Ignore blank lines and comments
            continue
        if '=' in line:
            key, value = line.split('=', 1)  # Split only on the first '='
            key = key.strip()
            value = value.strip()
            if key == 'rpcport':
                rpcport = value
            elif key == 'rpcuser':
                rpcuser = value
            elif key == 'rpcpassword':
                rpcpassword = value
            elif key == 'rpchost':
                rpchost = value
            else:
                other_flags[key] = value
    return rpchost, rpcport, rpcuser, rpcpassword, other_flags

def get_network_info(rpc_port):
    if rpc_port == '9932':
        network = 'mainnet'
        burn_address = 'PtpasteLBurnAddressXXXXXXXXXXbJ5ndd'
    elif rpc_port == '19932':
        network = 'testnet'
        burn_address = 'tPpasteLBurnAddressXXXXXXXXXXX3wy7u'
    elif rpc_port == '29932':
        network = 'devnet'
        burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7'
    else:
        raise ValueError(f"Unknown RPC port: {rpc_port}")
    return network, burn_address

def required_collateral(network):
    if network == 'mainnet':
        return 5000000  # 5 million PSL for mainnet
    else:
        return 1000000  # 1 million PSL for testnet/devnet

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

def is_base64_encoded(data):
    if not isinstance(data, str):
        return False
    if len(data) % 4 != 0:
        return False
    base64_pattern = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')
    if not base64_pattern.match(data):
        return False
    try:
        base64.b64decode(data, validate=True)
        return True
    except Exception:
        return False

def kill_open_ssh_tunnels(local_port):
    try:
        lsof_command = [
            "lsof", "-i", f"TCP:{local_port}", "-sTCP:LISTEN"
        ]
        result = subprocess.run(lsof_command, capture_output=True, text=True)
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            pids = []
            for line in lines[1:]:
                columns = line.split()
                if len(columns) > 1:
                    pids.append(columns[1])
            if pids:
                for pid in pids:
                    kill_command = ["kill", "-9", pid]
                    subprocess.run(kill_command)
                logger.info("Killed SSH tunnels with PIDs: {}".format(', '.join(pids)))
            else:
                logger.info("No SSH tunnels found to kill.")
        else:
            logger.info("No SSH tunnels found.")
    except Exception as e:
        logger.error("Error while killing SSH tunnels: {}".format(e))
        
def establish_ssh_tunnel():
    if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        kill_open_ssh_tunnels(REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT)
        key_path = REMOTE_SWISS_ARMY_LLAMA_INSTANCE_SSH_KEY_PATH
        if not os.access(key_path, os.R_OK):
            raise PermissionError("SSH key file at {} is not readable.".format(key_path))
        current_permissions = os.stat(key_path).st_mode & 0o777
        if current_permissions != 0o600:
            os.chmod(key_path, 0o600)
            logger.info("Permissions for SSH key file set to 600.")
        try:
            tunnel = SSHTunnelForwarder(
                (REMOTE_SWISS_ARMY_LLAMA_INSTANCE_IP_ADDRESS, REMOTE_SWISS_ARMY_LLAMA_INSTANCE_PORT),
                ssh_username="root",
                ssh_pkey=key_path,
                remote_bind_address=("localhost", REMOTE_SWISS_ARMY_LLAMA_EXPOSED_PORT),
                local_bind_address=("localhost", REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT),
                host_pkey_directories=[]  # Disable host key checking
            )
            tunnel.start()
            logger.info("SSH tunnel established: {}".format(tunnel.local_bind_address))
            while True:
                time.sleep(10)
        except BaseSSHTunnelForwarderError as e:
            logger.error("SSH tunnel error: {}".format(e))
        except Exception as e:
            logger.error("Error establishing SSH tunnel: {}".format(e))
                
def get_audio_length(audio_input) -> float:
    if isinstance(audio_input, bytes):
        audio_file = io.BytesIO(audio_input)
        audio = MutagenFile(audio_file)
    elif isinstance(audio_input, str):
        audio = MutagenFile(audio_input)
    else:
        raise ValueError("audio_input must be either bytes or a file path string.")
    if audio is None or not hasattr(audio.info, 'length'):
        raise ValueError("Could not determine the length of the audio file.")
    return audio.info.length
        
def convert_uuids_to_strings(data):
    if isinstance(data, dict):
        return {key: convert_uuids_to_strings(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_uuids_to_strings(item) for item in data]
    elif isinstance(data, uuid.UUID):
        return str(data)
    else:
        return data

def compute_sha3_256_hexdigest(input_str: str):
    """Compute the SHA3-256 hash of the input string and return the hexadecimal digest."""
    return hashlib.sha3_256(input_str.encode('utf-8')).hexdigest()

def compute_sha3_256_hexdigest_of_file(file_data: bytes):
    return hashlib.sha3_256(file_data).hexdigest()

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)
    if path in file_store:
        del file_store[path]

async def save_file(file_content: bytes, filename: str):
    file_location = os.path.join(tempfile.gettempdir(), filename)
    with open(file_location, "wb") as buffer:
        buffer.write(file_content)
    file_hash = compute_sha3_256_hexdigest_of_file(file_content)
    file_size = os.path.getsize(file_location) # Calculate file size
    expire_at = datetime.utcnow() + timedelta(hours=24) # Set expiration time (24 hours)
    file_store[file_location] = expire_at
    return file_location, file_hash, file_size

async def upload_and_get_file_metadata(file_content: bytes, file_prefix: str = "document") -> Dict:
    file_name = f"{file_prefix}_{compute_sha3_256_hexdigest_of_file(file_content)[:8]}.{magika.identify_bytes(file_content).output.ct_label}"
    file_location, file_hash, file_size = await save_file(file_content, file_name)
    external_ip = get_external_ip_func()
    file_url = f"http://{external_ip}:{UVICORN_PORT}/download/{file_name}"    
    return {
        "file_location": file_location,
        "file_hash": file_hash,
        "file_size": file_size,
        "file_url": file_url
    }
    
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
    _semaphore = asyncio.BoundedSemaphore(MAXIMUM_NUMBER_OF_CONCURRENT_RPC_REQUESTS)

    def __init__(self, service_url, service_name=None, reconnect_timeout=2, reconnect_amount=3, request_timeout=20):
        self.service_url = service_url
        self.service_name = service_name
        self.url = urlparse(service_url)
        self.id_count = 0
        user = self.url.username
        password = self.url.password
        authpair = f"{user}:{password}".encode('utf-8')
        self.auth_header = b'Basic ' + base64.b64encode(authpair)
        self.reconnect_timeout = reconnect_timeout
        self.reconnect_amount = reconnect_amount
        self.request_timeout = request_timeout
        self.client = httpx.AsyncClient(timeout=request_timeout, http2=True)

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        if self.service_name is not None:
            name = f"{self.service_name}.{name}"
        return AsyncAuthServiceProxy(self.service_url, name)

    async def __call__(self, *args):
        async with self._semaphore:
            self.id_count += 1
            postdata = json.dumps({
                'version': '2.0',
                'method': self.service_name,
                'params': args,
                'id': self.id_count
            })
            headers = {
                'Host': self.url.hostname,
                'User-Agent': "AuthServiceProxy/0.1",
                'Authorization': self.auth_header.decode(),
                'Content-type': 'application/json',
                'Connection': 'keep-alive'
            }
            for i in range(self.reconnect_amount):
                try:
                    if i > 0:
                        logger.warning(f"Reconnect try #{i+1}")
                        sleep_time = self.reconnect_timeout * (2 ** i)
                        logger.info(f"Waiting for {sleep_time} seconds before retrying.")
                        await asyncio.sleep(sleep_time)
                    response = await self.client.post(
                        self.service_url,
                        headers=headers,
                        content=postdata
                    )
                    response.raise_for_status()
                    response_json = response.json()
                    break
                except httpx.HTTPStatusError as e:
                    logger.error(f"HTTP error occurred in __call__: {e}")
                except httpx.RequestError as e:
                    logger.error(f"Request error occurred in __call__: {e}")
                except Exception as e:
                    logger.error(f"Unexpected error occurred in __call__: {e}")
            else:
                logger.error("Reconnect tries exceeded.")
                return
            if 'error' in response_json and response_json['error'] is not None:
                raise JSONRPCException(response_json['error'])
            elif 'result' not in response_json:
                raise JSONRPCException({
                    'code': -343, 'message': 'missing JSON-RPC result'})
            else:
                return response_json['result']

    async def close(self):
        await self.client.aclose()
                    
async def get_current_pastel_block_height_func():
    best_block_hash = await rpc_connection.getbestblockhash()
    best_block_details = await rpc_connection.getblock(best_block_hash)
    curent_block_height = best_block_details['height']
    return curent_block_height

async def get_best_block_hash_and_merkle_root_func():
    best_block_height = await get_current_pastel_block_height_func()
    best_block_hash = await rpc_connection.getblockhash(best_block_height)
    best_block_details = await rpc_connection.getblock(best_block_hash)
    best_block_merkle_root = best_block_details['merkleroot']
    return best_block_hash, best_block_merkle_root, best_block_height

async def get_last_block_data_func():
    current_block_height = await get_current_pastel_block_height_func()
    block_data = await rpc_connection.getblock(str(current_block_height))
    return block_data

async def check_psl_address_balance_alternative_func(address_to_check):
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
    balance_at_address = await rpc_connection.z_getbalance(address_to_check)
    return balance_at_address

async def get_raw_transaction_func(txid):
    raw_transaction_data = await rpc_connection.getrawtransaction(txid, 1) 
    return raw_transaction_data

async def verify_message_with_pastelid_func(pastelid, message_to_verify, pastelid_signature_on_message) -> str:
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
    masternode_top_command_output = await rpc_connection.masternode('top')
    return masternode_top_command_output

async def generate_supernode_inference_ip_blacklist(max_response_time_in_milliseconds=1200):
    blacklist_path = Path('supernode_inference_ip_blacklist.txt')
    logger.info("Now compiling Supernode IP blacklist based on Supernode responses to port checks...")
    if not blacklist_path.exists():
        blacklist_path.touch()
    full_supernode_list_df, _ = await check_supernode_list_func()
    port_7123_failures = 0
    async def check_port(supernode):
        nonlocal port_7123_failures
        ip_address_port = supernode.get('ipaddress:port')
        if not ip_address_port or ip_address_port.startswith(local_ip):
            return None
        ip_address = ip_address_port.split(":")[0]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f'http://{ip_address}:7123/liveness_ping', timeout=max_response_time_in_milliseconds / 1000)
                if response.status_code != 200:
                    port_7123_failures += 1
                    return None
            return supernode
        except httpx.RequestError:
            port_7123_failures += 1
            return None
    check_results = await asyncio.gather(*(check_port(supernode) for supernode in full_supernode_list_df.to_dict(orient='records')))
    filtered_supernodes = [supernode['extKey'] for supernode in check_results if supernode is not None]
    successful_nodes = {supernode['extKey'] for supernode in check_results if supernode is not None}
    failed_nodes = {supernode['ipaddress:port'] for supernode in full_supernode_list_df.to_dict(orient='records') if supernode['extKey'] not in successful_nodes}
    logger.info(f"Port 7123 failures: {port_7123_failures}")
    logger.info(f"There were {len(failed_nodes)} failed Supernodes out of {len(full_supernode_list_df)} total Supernodes, a failure rate of {len(failed_nodes) / len(full_supernode_list_df) * 100:.2f}%")
    with blacklist_path.open('w') as blacklist_file:
        for failed_node in failed_nodes:
            ip_address = failed_node.split(':')[0]
            blacklist_file.write(f"{ip_address}\n")
    return filtered_supernodes

async def check_supernode_list_func():
    masternode_list_full_command_output = await rpc_connection.masternodelist('full')
    masternode_list_rank_command_output = await rpc_connection.masternodelist('rank')
    masternode_list_pubkey_command_output = await rpc_connection.masternodelist('pubkey')
    masternode_list_extra_command_output = await rpc_connection.masternodelist('extra')
    if masternode_list_full_command_output:
        masternode_list_full_df = pd.DataFrame([masternode_list_full_command_output[x].split() for x in masternode_list_full_command_output])
        masternode_list_full_df['txid_vout'] = [x for x in masternode_list_full_command_output]
        masternode_list_full_df.columns = ['supernode_status', 'protocol_version', 'supernode_psl_address', 'lastseentime', 'activeseconds', 'lastpaidtime', 'lastpaidblock', 'ipaddress:port', 'txid_vout']
        masternode_list_full_df.index = masternode_list_full_df['txid_vout']
        masternode_list_full_df.drop(columns=['txid_vout'], inplace=True)
        for txid_vout in masternode_list_full_df.index:
            rank = masternode_list_rank_command_output.get(txid_vout)
            pubkey = masternode_list_pubkey_command_output.get(txid_vout)
            extra = masternode_list_extra_command_output.get(txid_vout, {})
            masternode_list_full_df.at[txid_vout, 'rank'] = rank if rank is not None else 'Unknown'
            masternode_list_full_df.at[txid_vout, 'pubkey'] = pubkey if pubkey is not None else 'Unknown'
            masternode_list_full_df.at[txid_vout, 'extAddress'] = extra.get('extAddress', 'Unknown')
            masternode_list_full_df.at[txid_vout, 'extP2P'] = extra.get('extP2P', 'Unknown')
            masternode_list_full_df.at[txid_vout, 'extKey'] = extra.get('extKey', 'Unknown')
        masternode_list_full_df['lastseentime'] = pd.to_numeric(masternode_list_full_df['lastseentime'], downcast='integer')
        masternode_list_full_df['lastpaidtime'] = pd.to_numeric(masternode_list_full_df['lastpaidtime'], downcast='integer')
        masternode_list_full_df['lastseentime'] = pd.to_datetime(masternode_list_full_df['lastseentime'], unit='s')
        masternode_list_full_df['lastpaidtime'] = pd.to_datetime(masternode_list_full_df['lastpaidtime'], unit='s')
        masternode_list_full_df['activeseconds'] = masternode_list_full_df['activeseconds'].astype(int)
        masternode_list_full_df['lastpaidblock'] = masternode_list_full_df['lastpaidblock'].astype(int)
        masternode_list_full_df['activedays'] = masternode_list_full_df['activeseconds'].apply(lambda x: float(x) / 86400.0)
        masternode_list_full_df['rank'] = masternode_list_full_df['rank'].astype(int, errors='ignore')
        masternode_list_full_df = masternode_list_full_df[masternode_list_full_df['supernode_status'].isin(['ENABLED', 'PRE_ENABLED'])]
        masternode_list_full_df__json = masternode_list_full_df.to_json(orient='index')
        return masternode_list_full_df, masternode_list_full_df__json
    else:
        error_message = "Masternode list command returning nothing-- pasteld probably just started and hasn't yet finished the mnsync process!"
        logger.error(error_message)
        raise ValueError(error_message) 
    
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
    datetime_cutoff_to_ignore_obsolete_messages = pd.to_datetime(datetime.now() - timedelta(days=NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE))
    supernode_list_df, _ = await check_supernode_list_func()
    txid_vout_to_pastelid_dict = dict(zip(supernode_list_df.index, supernode_list_df['extKey']))
    async with db_code.Session() as db:
        # Retrieve messages from the database that meet the timestamp criteria
        query = await db.exec(
            select(db_code.Message)
            .where(db_code.Message.timestamp >= datetime_cutoff_to_ignore_obsolete_messages)
            .order_by(db_code.Message.timestamp.desc())
        )
        db_messages = query.all()
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
            # logger.warning(f"Skipping message due to missing PastelID for txid_vout: {sending_sn_txid_vout} or {receiving_sn_txid_vout}")
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
        async with httpx.AsyncClient(timeout=Timeout(INDIVIDUAL_SUPERNODE_MODEL_MENU_REQUEST_TIMEOUT_PERIOD_IN_SECONDS)) as client:
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
        model_names = [model["model_name"] for model in model_menu["models"]]
        best_match = process.extractOne(desired_model_canonical_string, model_names)
        if best_match is not None and best_match[1] >= 95:
            matched_model = next(model for model in model_menu["models"] if model["model_name"] == best_match[0])
            if "supported_inference_type_strings" in matched_model and "model_parameters" in matched_model:
                if desired_model_inference_type_string in matched_model["supported_inference_type_strings"]:
                    for desired_param, desired_value in desired_parameters.items():
                        param_found = False
                        for param in matched_model["model_parameters"]:
                            if param["name"] == desired_param:
                                if "type" in param:
                                    if param["type"] == "int" and isinstance(int(desired_value), int):
                                        param_found = True
                                    elif param["type"] == "float" and isinstance(float(desired_value), float):
                                        param_found = True
                                    elif param["type"] == "string" and isinstance(str(desired_value), str):
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
    blacklist_path = Path('supernode_inference_ip_blacklist.txt')
    blacklisted_ips = set()
    if blacklist_path.exists():
        with blacklist_path.open('r') as blacklist_file:
            blacklisted_ips = {line.strip() for line in blacklist_file if line.strip()}
    else:
        logger.info("Blacklist file not found. Proceeding without blacklist filtering.")
    supernode_list_df, _ = await check_supernode_list_func()
    n = 4
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    local_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    message_body_dict = json.loads(message_body)
    desired_model_canonical_string = message_body_dict.get('requested_model_canonical_string')
    desired_model_inference_type_string = message_body_dict.get('model_inference_type_string')
    desired_model_parameters_json = base64.b64decode(message_body_dict.get('model_parameters_json_b64')).decode('utf-8')
    async def is_model_supported_async(supernode_ip_and_port, model_canonical_string, model_inference_type_string, model_parameters_json):
        supernode_ip = supernode_ip_and_port.split(':')[0]
        supernode_url = f"http://{supernode_ip}:7123"
        if supernode_ip in blacklisted_ips:
            return False
        model_menu = await get_supernode_model_menu(supernode_url)
        return is_model_supported(model_menu, model_canonical_string, model_inference_type_string, model_parameters_json)
    supported_supernodes_coroutines = [
        is_model_supported_async(row['ipaddress:port'], desired_model_canonical_string, desired_model_inference_type_string, desired_model_parameters_json)
        for _, row in supernode_list_df.iterrows()
    ]
    supported_supernodes_mask = await asyncio.gather(*supported_supernodes_coroutines)
    supported_supernodes = supernode_list_df[supported_supernodes_mask]
    supported_supernodes_minus_this_supernode = supported_supernodes[supported_supernodes['extKey'] != local_sn_pastelid]
    if len(supported_supernodes_minus_this_supernode) == 0:
        logger.error(f"No other supported supernodes found for the desired model: {desired_model_canonical_string} with inference type: {desired_model_inference_type_string} and parameters: {desired_model_parameters_json}")
        supported_supernodes_minus_this_supernode = supernode_list_df[supernode_list_df['extKey'] != local_sn_pastelid]
        logger.info("We had to choose audit supernodes which cannot process the request themselves if needed!")
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
            
async def check_if_record_exists(db_session, model, **kwargs):
    """ Check if a record exists in the database matching the given criteria. """
    existing_record = await db_session.execute(
        select(model).filter_by(**kwargs)
    )
    return existing_record.scalars().first()
        
async def process_broadcast_messages(message, db_session):
    try:
        message_body = json.loads(message.message_body)
        if message.message_type == 'inference_request_response_announcement_message':
            response_data = json.loads(message_body['message'])
            # Check if the record already exists
            existing_request = await check_if_record_exists(
                db_session, db_code.InferenceAPIUsageRequest,
                sha3_256_hash_of_inference_request_fields=response_data['sha3_256_hash_of_inference_request_fields']
            )
            existing_response = await check_if_record_exists(
                db_session, db_code.InferenceAPIUsageResponse,
                sha3_256_hash_of_inference_request_response_fields=response_data['sha3_256_hash_of_inference_request_response_fields']
            )
            if not existing_request and not existing_response:
                # If neither record exists, proceed to create new entries
                usage_request = db_code.InferenceAPIUsageRequest(**response_data)
                usage_response = db_code.InferenceAPIUsageResponse(**response_data)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Random sleep before DB operations
                await retry_on_database_locked(db_session.add, usage_request)
                await retry_on_database_locked(db_session.add, usage_response)
                await retry_on_database_locked(db_session.commit)
                await retry_on_database_locked(db_session.refresh, usage_request)
                await retry_on_database_locked(db_session.refresh, usage_response)
            else:
                logger.info("Skipping insertion as the record already exists.")
        elif message.message_type == 'inference_request_result_announcement_message':
            result_data = json.loads(message_body['message'])
            existing_result = await check_if_record_exists(
                db_session, db_code.InferenceAPIOutputResult,
                sha3_256_hash_of_inference_result_fields=result_data['sha3_256_hash_of_inference_result_fields']
            )
            if not existing_result:
                output_result = db_code.InferenceAPIOutputResult(**result_data)
                await asyncio.sleep(random.uniform(0.1, 0.5))  # Random sleep before DB operations
                await retry_on_database_locked(db_session.add, output_result)
                await retry_on_database_locked(db_session.commit)
                await retry_on_database_locked(db_session.refresh, output_result)
            else:
                logger.info("Skipping insertion as the result record already exists.")
    except Exception as e: # noqa: F841
        traceback.print_exc()
        
async def monitor_new_messages():
    last_processed_timestamp = None
    while True:
        try:
            async with db_code.Session() as db:
                if last_processed_timestamp is None:
                    query = await db.exec(select(db_code.Message.timestamp).order_by(db_code.Message.timestamp.desc()).limit(1))
                    last_processed_timestamp_raw = query.one_or_none()
                    if last_processed_timestamp_raw is None:
                        last_processed_timestamp = pd.Timestamp.min
                    else:
                        last_processed_timestamp = pd.Timestamp(last_processed_timestamp_raw)
                new_messages_df = await list_sn_messages_func()
                if new_messages_df is not None and not new_messages_df.empty:
                    new_messages_df = new_messages_df[new_messages_df['timestamp'] > last_processed_timestamp]
                    if not new_messages_df.empty:
                        for _, message in new_messages_df.iterrows():
                            query = await db.exec(
                                select(db_code.Message).where(
                                    db_code.Message.sending_sn_pastelid == message['sending_sn_pastelid'],
                                    db_code.Message.receiving_sn_pastelid == message['receiving_sn_pastelid'],
                                    db_code.Message.timestamp == message['timestamp']
                                )
                            )
                            existing_message = query.one_or_none()
                            if existing_message:
                                continue
                            log_action_with_payload("received new", "message", message)
                            last_processed_timestamp = message['timestamp']
                            sending_sn_pastelid = message['sending_sn_pastelid']
                            receiving_sn_pastelid = message['receiving_sn_pastelid']
                            message_size_bytes = len(message['message_body'].encode('utf-8'))
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before updating metadata                            
                            # Update MessageSenderMetadata
                            query = await db.exec(
                                select(db_code.MessageSenderMetadata).where(db_code.MessageSenderMetadata.sending_sn_pastelid == sending_sn_pastelid)
                            )
                            sender_metadata = query.one_or_none()
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
                            query = await db.exec(
                                select(db_code.MessageReceiverMetadata).where(db_code.MessageReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid)
                            )
                            receiver_metadata = query.one_or_none()
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
                            query = await db.exec(
                                select(db_code.MessageSenderReceiverMetadata).where(
                                    db_code.MessageSenderReceiverMetadata.sending_sn_pastelid == sending_sn_pastelid,
                                    db_code.MessageSenderReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid
                                )
                            )
                            sender_receiver_metadata = query.one_or_none()
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
                            query = await db.exec(
                                select(
                                    func.count(db_code.Message.id),
                                    func.count(func.distinct(db_code.Message.sending_sn_pastelid)),
                                    func.count(func.distinct(db_code.Message.receiving_sn_pastelid))
                                )
                            )
                            total_messages, total_senders, total_receivers = query.one()
                            query = await db.exec(select(db_code.MessageMetadata).order_by(db_code.MessageMetadata.timestamp.desc()).limit(1))
                            message_metadata = query.one_or_none()
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
            traceback.print_exc()
            await asyncio.sleep(5)
        finally:
            await asyncio.sleep(5)            
            
async def create_user_message(from_pastelid: str, to_pastelid: str, message_body: str, message_signature: str) -> dict:
    async with db_code.Session() as db:
        user_message = db_code.UserMessage(from_pastelid=from_pastelid, to_pastelid=to_pastelid, message_body=message_body, message_signature=message_signature)
        db.add(user_message)
        db.commit()
        db.refresh(user_message)
        user_message_dict = user_message.model_dump()
        user_message_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in user_message_dict.items()}
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
        supernode_user_message_dict = supernode_user_message.model_dump()
        supernode_user_message_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in supernode_user_message_dict.items()}
        return supernode_user_message_dict

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
        query = await db.exec(select(db_code.UserMessage).where(db_code.UserMessage.id == supernode_user_message.user_message_id))
        user_message = query.one_or_none()
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
        query = await db.exec(select(db_code.UserMessage).where((db_code.UserMessage.from_pastelid == pastelid) | (db_code.UserMessage.to_pastelid == pastelid)))
        user_messages = query.all()
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
    
async def get_credit_pack_purchase_request_response_from_request_hash(sha3_256_hash_of_credit_pack_purchase_request_fields: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponse).where(db_code.CreditPackPurchaseRequestResponse.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
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

async def get_credit_pack_purchase_request_confirmation_from_request_hash(sha3_256_hash_of_credit_pack_purchase_request_fields: str) -> db_code.CreditPackPurchaseRequestConfirmation:
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation).where(db_code.CreditPackPurchaseRequestConfirmation.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
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
        logger.info(f"Calculated preliminary price per credit: {rounded_cost_per_credit_psl:,.1f} PSL")
        return rounded_cost_per_credit_psl
    except (ValueError, ZeroDivisionError) as e:
        logger.error(f"Error calculating preliminary price per credit: {str(e)}")
        traceback.print_exc()
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
        logger.info(f"Proposed price per credit: {proposed_psl_price_per_credit:,.1f} PSL")
        logger.info(f"Local price per credit: {local_price_per_credit:,.1f} PSL")
        logger.info(f"Acceptable price range: [{min_acceptable_price:,.1f}, {max_acceptable_price:,.1f}] PSL")
        logger.info(f"Agreement with proposed price: {agree_with_proposed_price}")
        return agree_with_proposed_price
    except Exception as e:
        logger.error(f"Error determining agreement with proposed price: {str(e)}")
        traceback.print_exc()        
        raise
    
async def check_burn_transaction(txid: str, credit_usage_tracking_psl_address: str, total_cost_in_psl: float, request_response_pastel_block_height: int) -> Tuple[bool, int]:
    try:
        max_block_height = request_response_pastel_block_height + MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET
        max_retries = 30
        initial_retry_delay_in_seconds = 30
        total_cost_in_psl = round(total_cost_in_psl,5)
        matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_address_for_tracking_transaction(
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
        traceback.print_exc()
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

def turn_lists_and_dicts_into_strings_func(data: bytes) -> str:
    # Decode the bytes to a string
    data_str = data.decode('utf-8')
    # Parse the string as JSON
    data_dict = json.loads(data_str)
    def replace_lists_and_dicts_with_strings(obj):
        if isinstance(obj, dict):
            # Convert the dictionary to a JSON string if it's not the top-level object
            return json.dumps({k: replace_lists_and_dicts_with_strings(v) for k, v in obj.items()})
        elif isinstance(obj, list):
            # Convert the list to a JSON string
            return json.dumps(obj)
        else:
            return obj
    # Process the input data
    processed_data = replace_lists_and_dicts_with_strings(data_dict)
    # Return the processed data
    return processed_data

def transform_sqlmodel_list_and_dict_fields_into_strings_func(model_instance: SQLModel):
    for field_name, value in model_instance.__dict__.items():
        if isinstance(value, (list, dict)):
            # Convert the list or dictionary to a JSON string
            setattr(model_instance, field_name, json.dumps(value))
    return model_instance    
        
def parse_sqlmodel_strings_into_lists_and_dicts_func(model_instance: SQLModel) -> SQLModel:
    for field_name, value in model_instance.__dict__.items():
        if isinstance(value, str):  # Check if the field value is a string
            try:
                parsed_value = json.loads(value)  # Try to parse the string as JSON
                # Ensure the parsed value is a list or dictionary before setting it
                if isinstance(parsed_value, (list, dict)):
                    setattr(model_instance, field_name, parsed_value)  # Replace the string with a list or dictionary
            except json.JSONDecodeError:
                # If parsing fails, skip this field and continue
                continue
    return model_instance
        
async def retrieve_credit_pack_ticket_from_blockchain_using_txid(txid: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        credit_pack_combined_blockchain_ticket_data_json = await retrieve_generic_ticket_data_from_blockchain(txid)
        credit_pack_combined_blockchain_ticket_data_dict = json.loads(credit_pack_combined_blockchain_ticket_data_json)
        if credit_pack_combined_blockchain_ticket_data_dict is None:
            return None, None, None
        credit_pack_purchase_request_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_dict']
        credit_pack_purchase_request_response_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_response_dict']
        credit_pack_purchase_request_confirmation_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_confirmation_dict']
        credit_pack_purchase_request = db_code.CreditPackPurchaseRequest(**credit_pack_purchase_request_dict)
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse(**credit_pack_purchase_request_response_dict)
        credit_pack_purchase_request_confirmation = db_code.CreditPackPurchaseRequestConfirmation(**credit_pack_purchase_request_confirmation_dict)
        return credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket from blockchain: {str(e)}")
        raise
    
async def retrieve_credit_pack_ticket_using_txid(txid: str) -> db_code.CreditPackPurchaseRequestResponse:
    try:
        # Try to retrieve the credit pack ticket from the local database using the txid mapping
        async with db_code.Session() as db_session:
            mapping = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponseTxidMapping).where(db_code.CreditPackPurchaseRequestResponseTxidMapping.pastel_api_credit_pack_ticket_registration_txid == txid)
            )
            mapping_result = mapping.one_or_none()
            if mapping_result is not None:
                credit_pack_purchase_request = await get_credit_pack_purchase_request(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                credit_pack_purchase_request_response = await get_credit_pack_purchase_request_response_from_request_hash(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                credit_pack_purchase_request_confirmation = await get_credit_pack_purchase_request_confirmation_from_request_hash(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                return credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation
        # If the ticket is not found in the local database, retrieve it from the blockchain
        logger.info(f"Requested credit pack ticket with TXID {txid} not found in local database, attempting to get it from blockchain now...")
        credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation = await retrieve_credit_pack_ticket_from_blockchain_using_txid(txid)
        if all((credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation)):
            # Save the retrieved data to the local database for future reference
            logger.info(f"Now saving credit pack ticket data for TXID {txid} to database for future reference...")
            try:
                async with db_code.Session() as db_session:
                    db_session.add(credit_pack_purchase_request_response)
                    db_session.add(credit_pack_purchase_request_confirmation)
                    await db_session.commit()
                    await db_session.refresh(credit_pack_purchase_request_response)
                    await db_session.refresh(credit_pack_purchase_request_confirmation)                    
            except Exception as e:
                logger.error(f"Error saving retrieved credit pack ticket to the local database: {str(e)}")
            # Save the txid mapping for the retrieved ticket
            try:
                await save_credit_pack_purchase_request_response_txid_mapping(credit_pack_purchase_request_response, txid)
            except Exception as e:
                logger.error(f"Error saving txid mapping for retrieved credit pack ticket: {str(e)}")
            await db_code.consolidate_wal_data() # Consolidate WAL
        else:
            raise ValueError(f"Credit pack ticket not found for txid: {txid}")
        return credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket using txid: {txid}. Error: {str(e)}")
        traceback.print_exc()        
        raise

async def retrieve_credit_pack_ticket_from_purchase_burn_txid(purchase_burn_txid: str):
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation).where(db_code.CreditPackPurchaseRequestConfirmation.txid_of_credit_purchase_burn_transaction == purchase_burn_txid)
            )
            credit_pack_request_confirmation = result.one_or_none()
            sha3_256_hash_of_credit_pack_purchase_request_fields = credit_pack_request_confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields
        # Try to retrieve the credit pack ticket from the local database using the txid mapping
        async with db_code.Session() as db_session:
            mapping = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponseTxidMapping).where(db_code.CreditPackPurchaseRequestResponseTxidMapping.sha3_256_hash_of_credit_pack_purchase_request_fields == sha3_256_hash_of_credit_pack_purchase_request_fields)
            )
            mapping_result = mapping.one_or_none()
            if mapping_result is not None:
                logger.info(f"Found credit pack data for purchase burn TXID {purchase_burn_txid} in the local database, so returning it immediately!")
                credit_pack_purchase_request = await get_credit_pack_purchase_request(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                credit_pack_purchase_request_response = await get_credit_pack_purchase_request_response_from_request_hash(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                credit_pack_purchase_request_confirmation = await get_credit_pack_purchase_request_confirmation_from_request_hash(mapping_result.sha3_256_hash_of_credit_pack_purchase_request_fields)
                return credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation            
    except Exception as e:
        logger.error(f"Error occurred while retrieving credit pack ticket data: {e}")
        traceback.print_exc()
        return None, None, None        

async def get_final_credit_pack_registration_txid_from_credit_purchase_burn_txid(purchase_burn_txid: str):
    try:
        async with db_code.Session() as db_session:
            result = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation).where(db_code.CreditPackPurchaseRequestConfirmation.txid_of_credit_purchase_burn_transaction == purchase_burn_txid)
            )
            credit_pack_request_confirmation = result.one_or_none()
            sha3_256_hash_of_credit_pack_purchase_request_fields = credit_pack_request_confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields    
            return sha3_256_hash_of_credit_pack_purchase_request_fields
    except Exception as e:
        logger.error(f"Error occurred while retrieving credit pack ticket data: {e}")
        traceback.print_exc()
        return None      
    
def recursively_parse_json(data):
    # Helper function to handle recursive parsing
    if isinstance(data, str):
        try:
            parsed_data = json.loads(data)
            # After parsing, check if parsed_data is a dict or list, otherwise return the original string
            if isinstance(parsed_data, (dict, list)):
                return recursively_parse_json(parsed_data)
            else:
                return data  # Do not convert numbers, booleans, or nulls
        except json.JSONDecodeError:
            return data  # Return the original string if it's not JSON
    elif isinstance(data, dict):
        # Recursively process each key-value pair in the dictionary with sorted keys
        return {k: recursively_parse_json(data[k]) for k in sorted(data.keys())}
    elif isinstance(data, list):
        # Recursively process each element in the list
        return [recursively_parse_json(item) for item in data]
    else:
        # Return the data as is if it's neither a string, dict, nor list
        return data

def collect_leaf_nodes(data, parent_key=''):
    # Helper function to collect leaf nodes, concatenating key-value pairs
    if isinstance(data, dict):
        return '|'.join(f"{k} : {collect_leaf_nodes(v, k)}" for k, v in sorted(data.items()))
    elif isinstance(data, list):
        # Use parent key for elements of a list to keep the context
        return '|'.join(f"{parent_key} : {collect_leaf_nodes(item, parent_key)}" for item in data)
    elif isinstance(data, str) or not isinstance(data, Iterable):
        return str(data)
    return ''

def compute_fully_parsed_json_sha3_256_hash(input_json):
    # Parse the input JSON string
    parsed_data = json.loads(input_json)
    # Recursively parse nested JSON and other data structures
    fully_parsed_data = recursively_parse_json(parsed_data)
    # Collect all leaf nodes and concatenate them with key names
    concatenated_data = collect_leaf_nodes(fully_parsed_data)
    # Compute SHA3-256 hash
    sha3_hash = hashlib.sha3_256(concatenated_data.encode('utf-8')).hexdigest()
    return sha3_hash, concatenated_data

async def store_generic_ticket_data_in_blockchain(ticket_input_data_json: str, ticket_type_identifier: str):
    try:    
        if not isinstance(ticket_input_data_json, str):
            error_message = "Ticket data must be a valid string!"
            logger.error(error_message)
            raise ValueError(error_message)
        if not isinstance(ticket_type_identifier, str):
            error_message = "Ticket type identifier data must be a valid string!"
            logger.error(error_message)
            raise ValueError(error_message)
        ticket_type_identifier = ticket_type_identifier.upper()
        ticket_input_data_fully_parsed_sha3_256_hash, concatenated_data = compute_fully_parsed_json_sha3_256_hash(ticket_input_data_json)
        ticket_input_data_dict = recursively_parse_json(ticket_input_data_json)
        ticket_uncompressed_size_in_bytes = len(ticket_input_data_json.encode('utf-8'))
        ticket_dict = {"ticket_identifier_string": ticket_type_identifier,
                        "ticket_input_data_fully_parsed_sha3_256_hash": ticket_input_data_fully_parsed_sha3_256_hash,
                        "ticket_uncompressed_size_in_bytes": ticket_uncompressed_size_in_bytes,
                        "ticket_input_data_dict": ticket_input_data_dict}
        ticket_json = json.dumps(ticket_dict, ensure_ascii=False)
        ticket_json_b64 = base64.b64encode(ticket_json.encode('utf-8')).decode('utf-8')
        ticket_txid = ""
        logger.info("Now attempting to write data to blockchain using 'tickets register contract' command...")
        ticket_register_command_response = await rpc_connection.tickets('register', 'contract', ticket_json_b64, ticket_type_identifier, ticket_input_data_fully_parsed_sha3_256_hash)
        logger.info("Done with 'tickets register contract' command!")
        asyncio.sleep(2)
        if len(ticket_register_command_response) > 0:
            if 'txid' in ticket_register_command_response.keys():
                ticket_txid = ticket_register_command_response['txid']
                ticket_get_command_response = await rpc_connection.tickets('get', ticket_txid , 1)
                retrieved_ticket_data = ticket_get_command_response['ticket']['contract_ticket']
                ticket_tx_info = ticket_get_command_response['tx_info']
                if ticket_tx_info is None:
                    logger.error(f"Ticket was not processed correctly for registration txid {ticket_txid}")
                    return None
                uncompressed_ticket_size_in_bytes = ticket_tx_info['uncompressed_size']
                compressed_ticket_size_in_bytes = ticket_tx_info['compressed_size']
                retrieved_ticket_input_data_fully_parsed_sha3_256_hash = retrieved_ticket_data['ticket_input_data_fully_parsed_sha3_256_hash']
                retrieved_ticket_input_data_dict = retrieved_ticket_data['ticket_input_data_dict']
                retrieved_ticket_input_data_dict_json = json.dumps(retrieved_ticket_input_data_dict)
                computed_fully_parsed_json_sha3_256_hash, concatenated_data = compute_fully_parsed_json_sha3_256_hash(retrieved_ticket_input_data_dict_json)
                assert(computed_fully_parsed_json_sha3_256_hash==retrieved_ticket_input_data_fully_parsed_sha3_256_hash)
                assert(computed_fully_parsed_json_sha3_256_hash==ticket_input_data_fully_parsed_sha3_256_hash)
                logger.info(f"Generic blockchain ticket of sub-type {ticket_type_identifier} was successfully stored in the blockchain with TXID {ticket_txid}")
                logger.info(f"Original Data length: {uncompressed_ticket_size_in_bytes:,} bytes; Compressed data length: {compressed_ticket_size_in_bytes:,} bytes;") #Number of multisig outputs: {len(txouts):,}; Total size of multisig outputs in bytes: {sum(len(txout[1]) for txout in txouts):,}") 
        else:
            logger.error("Error storing ticket data in the blockchain! Could not retrieve and perfectly reconstruct original ticket data.")
        return ticket_txid, ticket_dict, ticket_json_b64
    except Exception as e:
        logger.error(f"Error occurred while storing ticket data in the blockchain: {e}")
        traceback.print_exc()
        return None

async def retrieve_generic_ticket_data_from_blockchain(ticket_txid: str):
    try:    
        ticket_get_command_response = await rpc_connection.tickets('get', ticket_txid , 1)
        retrieved_ticket_data = ticket_get_command_response['ticket']['contract_ticket']
        if retrieved_ticket_data is None:
            logger.error(f"Error: no ticket data returned for TXID {ticket_txid}")
            return None
        retrieved_ticket_input_data_fully_parsed_sha3_256_hash = retrieved_ticket_data['ticket_input_data_fully_parsed_sha3_256_hash']
        retrieved_ticket_input_data_dict = retrieved_ticket_data['ticket_input_data_dict']
        retrieved_ticket_input_data_dict_json = json.dumps(retrieved_ticket_input_data_dict)
        computed_fully_parsed_json_sha3_256_hash, concatenated_data = compute_fully_parsed_json_sha3_256_hash(retrieved_ticket_input_data_dict_json)
        assert(computed_fully_parsed_json_sha3_256_hash==retrieved_ticket_input_data_fully_parsed_sha3_256_hash)
        credit_pack_combined_blockchain_ticket_data_json = json.dumps(retrieved_ticket_input_data_dict)
        return credit_pack_combined_blockchain_ticket_data_json
    except Exception as e:
        logger.error(f"Error occurred while retrieving ticket data in the blockchain: {e}")
        traceback.print_exc()
        return None
    
async def get_list_of_credit_pack_ticket_txids_already_in_db():
    async with db_code.Session() as db_session:
        mappings = await db_session.exec(select(db_code.CreditPackPurchaseRequestResponseTxidMapping))
        mapping_results = mappings.all()
        if mapping_results is not None:
            logger.info(f"Found {len(mapping_results):,} credit pack TXIDs already stored in database...")
            list_of_already_stored_credit_pack_txids = list(set([x.pastel_api_credit_pack_ticket_registration_txid for x in mapping_results]))          
            return list_of_already_stored_credit_pack_txids
        else:
            return []

async def list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them(ticket_type_identifier: str = "INFERENCE_API_CREDIT_PACK_TICKET",  starting_block_height: int = 0, force_revalidate_all_tickets: int = 0):
    try:
        if not isinstance(ticket_type_identifier, str):
            error_message = "Ticket type identifier data must be a valid string!"
            logger.error(error_message)
            raise ValueError(error_message)
        ticket_type_identifier = ticket_type_identifier.upper()    
        logger.info(f"Getting all blockchain tickets of type {ticket_type_identifier} starting from block height {starting_block_height:,}...")
        list_of_ticket_data_dicts = await rpc_connection.tickets('list', 'contract', ticket_type_identifier, starting_block_height)
        list_of_ticket_internal_data_dicts = [x['ticket']['contract_ticket'] for x in list_of_ticket_data_dicts]
        logger.info(f"Found {len(list_of_ticket_internal_data_dicts):,} total tickets in the blockchain since block height {starting_block_height:,} of type {ticket_type_identifier}; Now checking if they are internally consistent...")
        list_of_retrieved_ticket_input_data_fully_parsed_sha3_256_hashes = [x['ticket_input_data_fully_parsed_sha3_256_hash'] for x in list_of_ticket_internal_data_dicts]
        list_of_retrieved_ticket_input_data_dicts = [x['ticket_input_data_dict'] for x in list_of_ticket_internal_data_dicts]
        list_of_retrieved_ticket_input_data_dicts_json = [json.dumps(x) for x in list_of_retrieved_ticket_input_data_dicts]
        list_of_computed_fully_parsed_json_sha3_256_hashes = [compute_fully_parsed_json_sha3_256_hash(x)[0] for x in list_of_retrieved_ticket_input_data_dicts_json]
        # Sort hashes before comparison to handle differences in order
        list_of_computed_fully_parsed_json_sha3_256_hashes_sorted = sorted(list_of_computed_fully_parsed_json_sha3_256_hashes)
        list_of_retrieved_ticket_input_data_fully_parsed_sha3_256_hashes_sorted = sorted(list_of_retrieved_ticket_input_data_fully_parsed_sha3_256_hashes)
        # Collect mismatches instead of asserting
        mismatches = [(i, x) for i, x in enumerate(list_of_retrieved_ticket_input_data_fully_parsed_sha3_256_hashes_sorted) if list_of_computed_fully_parsed_json_sha3_256_hashes_sorted[i] != x]
        if mismatches:
            logger.error(f"Mismatches found in hashes: {mismatches}")
        else:
            logger.info("All retrieved tickets were internally consistent and match the computed SHA3-256 hash of all fields!")
        list_of_already_stored_credit_pack_txids = await get_list_of_credit_pack_ticket_txids_already_in_db()
        list_of_known_bad_credit_pack_txids = await get_list_of_all_known_bad_credit_pack_ticket_txids_from_db()
        list_of_ticket_txids = [x['txid'] for x in list_of_ticket_data_dicts]
        if force_revalidate_all_tickets:
            list_of_ticket_txids_not_already_stored = list_of_ticket_txids
            if len(list_of_ticket_txids) > 0:
                logger.info(f"Now attempting to perform in-depth validation of all aspects of the {len(list_of_ticket_txids):,} retrieved {ticket_type_identifier} tickets (Forcing full revalidation of ALL tickets, even those already stored in the local database or which are in the known bad list of TXIDs!)...")
        else:
            list_of_ticket_txids_not_already_stored = [x for x in list_of_ticket_txids if x not in list_of_already_stored_credit_pack_txids and x not in list_of_known_bad_credit_pack_txids]
            if len(list_of_ticket_txids_not_already_stored) > 0:
                logger.info(f"Now attempting to perform in-depth validation of all aspects of the {len(list_of_ticket_txids_not_already_stored):,} retrieved {ticket_type_identifier} tickets that are not already in the database or in already in the known bad list of TXIDs...")
        list_of_fully_validated_ticket_txids = []
        for idx, current_txid in enumerate(list_of_ticket_txids_not_already_stored):
            validation_results = await validate_existing_credit_pack_ticket(current_txid)
            asyncio.sleep(0.5)
            if not validation_results["credit_pack_ticket_is_valid"]:
                logger.warning(f"[Ticket {idx+1}/{len(list_of_ticket_txids_not_already_stored)}] Blockchain ticket of type {ticket_type_identifier} and TXID {current_txid} was unable to be fully validated, so skipping it! Reasons for validation failure:\n{validation_results['validation_failure_reasons_list']}")
            else:
                logger.info(f"[Ticket {idx+1}/{len(list_of_ticket_txids_not_already_stored)}] Blockchain ticket of type {ticket_type_identifier} and TXID {current_txid} was fully validated! Now saving to database if not already there...")
                list_of_fully_validated_ticket_txids.append(current_txid)
                _ = await retrieve_credit_pack_ticket_using_txid(current_txid) # This will save the valid tickets to the database automatically for future reference
        if len(list_of_ticket_txids_not_already_stored) > 0:
            logger.info(f"We were able to fully validate {len(list_of_fully_validated_ticket_txids):,} of the {len(list_of_ticket_txids_not_already_stored):,} retrieved {ticket_type_identifier} tickets!")
        return list_of_retrieved_ticket_input_data_dicts_json, list_of_fully_validated_ticket_txids
    except Exception as e:
        logger.error(f"Error occurred while listing generic blockchain tickets of type {str(ticket_type_identifier)}: {e}")
        traceback.print_exc()
        return None    
    
async def periodic_ticket_listing_and_validation():
    while True:
        try:
            await list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them()
            await asyncio.sleep(30*60)  # Sleep for 30 minutes
        except Exception as e:
            logger.error(f"Error in periodic ticket listing and validation: {str(e)}")
            traceback.print_exc()
                
async def store_credit_pack_ticket_in_blockchain(credit_pack_combined_blockchain_ticket_data_json: str) -> str:
    try:
        logger.info("Now attempting to write the ticket data to the blockchain...")
        ticket_type_identifier = "INFERENCE_API_CREDIT_PACK_TICKET"
        credit_pack_ticket_txid, ticket_dict, ticket_json_b64 = await store_generic_ticket_data_in_blockchain(credit_pack_combined_blockchain_ticket_data_json, ticket_type_identifier)
        total_bytes_used = 0
        logger.info(f"Received back pastel txid of {credit_pack_ticket_txid} for the stored blockchain ticket data; total bytes used to store the data in the blockchain was {total_bytes_used:,}; now waiting for the transaction to be confirmed...")
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
                if (num_confirmations > 0) or SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK:
                    if num_confirmations > 0:
                        logger.info(f"Transaction {credit_pack_ticket_txid} has been confirmed with {num_confirmations} confirmations.")
                    else:
                        logger.info(f"Transaction {credit_pack_ticket_txid} has not yet been confirmed, but we are skipping confirmation check to speed things up.")
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
        if (num_confirmations > 0) or SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK:
            logger.info("Now verifying that we can reconstruct the original file written exactly...")
            reconstructed_file_data = await retrieve_generic_ticket_data_from_blockchain(credit_pack_ticket_txid)
            retrieved_data_fully_parsed_sha3_256_hash, _ = compute_fully_parsed_json_sha3_256_hash(reconstructed_file_data)
            original_data_fully_parsed_sha3_256_hash, _ = compute_fully_parsed_json_sha3_256_hash(credit_pack_combined_blockchain_ticket_data_json)
            if retrieved_data_fully_parsed_sha3_256_hash == original_data_fully_parsed_sha3_256_hash:
                logger.info("Successfully verified that the stored blockchain ticket data can be reconstructed exactly!")
                use_test_reconstruction_of_object_from_json = 1
                if use_test_reconstruction_of_object_from_json:
                    credit_pack_combined_blockchain_ticket_data_dict = json.loads(reconstructed_file_data)
                    credit_pack_purchase_request_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_dict']
                    credit_pack_purchase_request_response_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_response_dict']
                    credit_pack_purchase_request_confirmation_dict = credit_pack_combined_blockchain_ticket_data_dict['credit_pack_purchase_request_confirmation_dict']
                    credit_pack_purchase_request = db_code.CreditPackPurchaseRequest(**credit_pack_purchase_request_dict)  # noqa: F841
                    credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse(**credit_pack_purchase_request_response_dict)  # noqa: F841
                    credit_pack_purchase_request_confirmation = db_code.CreditPackPurchaseRequestConfirmation(**credit_pack_purchase_request_confirmation_dict)  # noqa: F841
                    logger.info(f"Reconstructed credit pack ticket data:\n Purchase Request: {abbreviated_pretty_json_func(credit_pack_purchase_request_dict)}\nPurchase Request Response: {abbreviated_pretty_json_func(credit_pack_purchase_request_response_dict)}\nPurchase Request Confirmation: {abbreviated_pretty_json_func(credit_pack_purchase_request_confirmation_dict)}")
            else:
                logger.error("Failed to verify that the stored blockchain ticket data can be reconstructed exactly!")
                storage_validation_error_string = "Failed to verify that the stored blockchain ticket data can be reconstructed exactly! Difference: " + str(set(reconstructed_file_data).symmetric_difference(set(credit_pack_combined_blockchain_ticket_data_json)))
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
        traceback.print_exc()
        return credit_pack_ticket_txid, storage_validation_error_string

async def check_original_supernode_storage_confirmation(sha3_256_hash_of_credit_pack_purchase_request_response_fields: str) -> bool:
    async with db_code.Session() as db:
        result = await db.exec(
            select(db_code.CreditPackPurchaseRequestConfirmationResponse).where(db_code.CreditPackPurchaseRequestConfirmationResponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields == sha3_256_hash_of_credit_pack_purchase_request_response_fields)
        )
        return result.one_or_none() is not None
    
async def check_if_credit_usage_tracking_psl_address_has_already_been_used_for_a_credit_pack(credit_usage_tracking_psl_address: str):
    async with db_code.Session() as db:
        result = await db.exec(
            select(db_code.CreditPackPurchaseRequestResponse).where(db_code.CreditPackPurchaseRequestResponse.credit_usage_tracking_psl_address == credit_usage_tracking_psl_address)
        )
        credit_pack_purchase_request_response = result.one_or_none()
        if credit_pack_purchase_request_response is not None:
            credit_tracking_address_already_used = True
        else:
            credit_tracking_address_already_used = False
        return credit_tracking_address_already_used, credit_pack_purchase_request_response

async def process_credit_purchase_initial_request(credit_pack_purchase_request: db_code.CreditPackPurchaseRequest) -> db_code.CreditPackPurchaseRequestPreliminaryPriceQuote:
    try:
        # Validate the request fields
        request_validation_errors = await validate_credit_pack_ticket_message_data_func(credit_pack_purchase_request)
        if request_validation_errors:
            rejection_message = await generate_credit_pack_request_rejection_message(credit_pack_purchase_request, request_validation_errors)
            logger.error(f"Invalid credit purchase request: {', '.join(request_validation_errors)}")
            return rejection_message
        if credit_pack_purchase_request.requested_initial_credits_in_credit_pack > MAXIMUM_CREDITS_PER_CREDIT_PACK:
            rejection_message = f"Requested initial credits in credit pack exceeds the maximum of {MAXIMUM_CREDITS_PER_CREDIT_PACK} credits allowed in a single credit pack!"
            logger.error(rejection_message)
            return rejection_message
        # Check if credit_usage_tracking_psl_address has already been used for an existing credit pack at any time:
        credit_tracking_address_already_used, credit_pack_purchase_request_response = await check_if_credit_usage_tracking_psl_address_has_already_been_used_for_a_credit_pack(credit_pack_purchase_request.credit_usage_tracking_psl_address)
        if credit_tracking_address_already_used:
            rejection_message = f"The specified credit tracking address of {credit_pack_purchase_request.credit_usage_tracking_psl_address} has already been used for a credit pack purchase (with sha3-256 hash of credit request fields of {credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_fields})"
            logger.error(rejection_message)
            return rejection_message
        # Determine the preliminary price quote
        preliminary_quoted_price_per_credit_in_psl = await calculate_preliminary_psl_price_per_credit()
        preliminary_total_cost_of_credit_pack_in_psl = round(preliminary_quoted_price_per_credit_in_psl * credit_pack_purchase_request.requested_initial_credits_in_credit_pack, 5)
        # Create the response without the hash and signature fields
        credit_pack_purchase_request_fields_json = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(credit_pack_purchase_request)
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestPreliminaryPriceQuote(
            sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_purchase_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
            credit_usage_tracking_psl_address=credit_pack_purchase_request.credit_usage_tracking_psl_address,
            credit_pack_purchase_request_fields_json_b64=base64.b64encode(credit_pack_purchase_request_fields_json.encode('utf-8')).decode('utf-8'),
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
        traceback.print_exc()
        raise
    
async def generate_credit_pack_request_rejection_message(credit_pack_request: db_code.CreditPackPurchaseRequest, validation_errors: List[str]) -> db_code.CreditPackPurchaseRequestRejection:
    credit_pack_purchase_request_fields_json = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(credit_pack_request)
    rejection_message = db_code.CreditPackPurchaseRequestRejection(
        sha3_256_hash_of_credit_pack_purchase_request_fields=credit_pack_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
        credit_pack_purchase_request_fields_json_b64=base64.b64encode(credit_pack_purchase_request_fields_json.encode('utf-8')).decode('utf-8'),
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
        blacklist_path = Path('supernode_inference_ip_blacklist.txt')
        blacklisted_ips = set()
        if blacklist_path.exists(): # Read the blacklist file if it exists
            with blacklist_path.open('r') as blacklist_file:
                blacklisted_ips = {line.strip() for line in blacklist_file if line.strip()}
        else:
            logger.info("Blacklist file not found. Proceeding without blacklist filtering.")
        total_number_of_blacklisted_sns = len(blacklisted_ips)
        best_block_hash, best_block_merkle_root, _ = await get_best_block_hash_and_merkle_root_func() # Get the best block hash and merkle root
        supernode_list_df, _ = await check_supernode_list_func() # Get the list of all supernodes
        number_of_supernodes_found = len(supernode_list_df) - 1
        if number_of_supernodes_found < MINIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES:
            logger.warning(f"Fewer than {MINIMUM_NUMBER_OF_POTENTIALLY_AGREEING_SUPERNODES} supernodes available. Using all {number_of_supernodes_found} available supernodes.")
            all_other_supernode_pastelids = [x for x in supernode_list_df['extKey'].tolist() if x != MY_PASTELID]
            return all_other_supernode_pastelids
        # Compute the XOR distance between each supernode's hash(pastelid) and the best block's merkle root
        xor_distances = []
        list_of_blacklisted_supernode_pastelids = []
        for _, row in supernode_list_df.iterrows():
            ip_address_port = row['ipaddress:port']
            ip_address = ip_address_port.split(":")[0]
            if ip_address in blacklisted_ips:
                current_supernode_pastelid = row['extKey']
                list_of_blacklisted_supernode_pastelids.append(current_supernode_pastelid)
                continue
            supernode_pastelid = row['extKey']
            supernode_pastelid_hash = get_sha256_hash_of_input_data_func(supernode_pastelid)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            merkle_root_int = int(best_block_merkle_root, 16)
            xor_distance = supernode_pastelid_int ^ merkle_root_int
            xor_distances.append((supernode_pastelid, xor_distance))
        # Sort the supernodes based on their XOR distances in ascending order
        sorted_supernodes = sorted(xor_distances, key=lambda x: x[1])
        # Select the supernodes with the closest XOR distances
        potentially_agreeing_supernodes = [supernode[0] for supernode in sorted_supernodes if supernode[0] != MY_PASTELID]
        return potentially_agreeing_supernodes, total_number_of_blacklisted_sns, list_of_blacklisted_supernode_pastelids
    except Exception as e:
        logger.error(f"Error selecting potentially agreeing supernodes: {str(e)}")
        traceback.print_exc()
        raise    

async def select_top_n_closest_supernodes_to_best_block_merkle_root(supernode_pastelids: List[str], n: int = 10, best_block_merkle_root: Optional[str] = None) -> List[str]:
    try:
        if best_block_merkle_root is None:
            _, best_block_merkle_root, _ = await get_best_block_hash_and_merkle_root_func()  # Get the best block hash and merkle root if not supplied
        best_block_merkle_root_hash = get_sha256_hash_of_input_data_func(best_block_merkle_root)
        merkle_root_hash_int = int(best_block_merkle_root_hash, 16)
        # Compute the XOR distance between each supernode's hash(pastelid) and the best block's merkle root
        xor_distances = []
        for pastelid in supernode_pastelids:
            supernode_pastelid_hash = get_sha256_hash_of_input_data_func(pastelid)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            xor_distance = supernode_pastelid_int ^ merkle_root_hash_int
            xor_distances.append((pastelid, xor_distance))
        # Sort the supernodes based on their XOR distances in ascending order
        sorted_supernodes = sorted(xor_distances, key=lambda x: x[1])
        # Select the top N supernodes with the closest XOR distances
        top_n_supernodes = [supernode[0] for supernode in sorted_supernodes[:n]]
        return top_n_supernodes
    except Exception as e:
        logger.error(f"Error selecting top N closest supernodes to best block merkle root: {str(e)}")
        traceback.print_exc()
        raise

async def check_liveness(supernode_base_url: str) -> bool:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{supernode_base_url}/liveness_ping", timeout=2)
            return response.status_code == 200
    except httpx.RequestError:
        return False
    
async def get_supernode_url_and_check_liveness(supernode_pastelid, supernode_list_df, client):
    try:
        supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
        is_alive = await check_liveness(supernode_base_url)
        return supernode_base_url, is_alive
    except Exception as e:
        logger.warning(f"Error getting supernode URL or checking liveness for supernode {supernode_pastelid}: {str(e)}")
        return None, False
    
async def send_price_agreement_request_to_supernodes(request: db_code.CreditPackPurchasePriceAgreementRequest, supernodes: List[str]) -> List[db_code.CreditPackPurchasePriceAgreementRequestResponse]:
    blacklist_path = Path('supernode_inference_ip_blacklist.txt')
    blacklisted_ips = set()
    if blacklist_path.exists():
        with blacklist_path.open('r') as blacklist_file:
            blacklisted_ips = {line.strip() for line in blacklist_file if line.strip()}
    else:
        logger.info("Blacklist file not found. Proceeding without blacklist filtering.")
    async with httpx.AsyncClient(timeout=Timeout(None)) as client:
        supernode_list_df, _ = await check_supernode_list_func()
        # Prepare tasks for supernode liveness check
        supernode_liveness_tasks = [
            asyncio.create_task(get_supernode_url_and_check_liveness(supernode_pastelid, supernode_list_df, client))
            for supernode_pastelid in supernodes
        ]
        supernode_urls_and_statuses = await asyncio.gather(*supernode_liveness_tasks)
        price_agreement_semaphore = asyncio.Semaphore(MAXIMUM_NUMBER_OF_CONCURRENT_RPC_REQUESTS)
        async def send_request(supernode_base_url, payload, timeout):
            async with price_agreement_semaphore:
                try:
                    response = await client.post(supernode_base_url, json=payload, timeout=timeout)
                    return response
                except httpx.RequestError as e:
                    logger.error(f"Error sending request to {supernode_base_url}: {str(e)}")
                    return None
        async def process_supernode(supernode_base_url, is_alive):
            if is_alive:
                async with price_agreement_semaphore:
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                challenge = challenge_dict["challenge"]
                challenge_id = challenge_dict["challenge_id"]
                challenge_signature = challenge_dict["challenge_signature"]
                request_dict = request.model_dump()
                request_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in request_dict.items()}
                payload = {
                    "credit_pack_price_agreement_request": request_dict,
                    "challenge": challenge,
                    "challenge_id": challenge_id,
                    "challenge_signature": challenge_signature
                }
                url = f"{supernode_base_url}/credit_pack_price_agreement_request"
                timeout = Timeout(INDIVIDUAL_SUPERNODE_PRICE_AGREEMENT_REQUEST_TIMEOUT_PERIOD_IN_SECONDS)  # Timeout for individual supernode request
                return await send_request(url, payload, timeout)
            else:
                logger.warning(f"Supernode {supernode_base_url} is not responding on port 7123")
                return None
        request_tasks = [
            process_supernode(supernode_base_url, is_alive)
            for supernode_base_url, is_alive in supernode_urls_and_statuses
            if supernode_base_url.split(":")[1].replace("//", "").split("/")[0] not in blacklisted_ips
        ]
        logger.info(f"Now sending out {len(request_tasks):,} price agreement requests to potentially agreeing supernodes...")
        datetime_start = datetime.now()
        responses = await asyncio.gather(*request_tasks, return_exceptions=True)
        datetime_end = datetime.now()
        duration = datetime_end - datetime_start
        logger.info(f"Finished sending price agreement requests to supernodes in {duration.total_seconds():.2f} seconds!")
        price_agreement_request_responses = [
            db_code.CreditPackPurchasePriceAgreementRequestResponse(**response.json())
            for response in responses
            if isinstance(response, httpx.Response) and response.status_code == 200
        ]
        logger.info(f"Received a total of {len(price_agreement_request_responses):,} valid price agreement responses from supernodes out of {len(request_tasks):,} total requests sent, a success rate of {len(price_agreement_request_responses)/len(request_tasks):.2%}")
        return price_agreement_request_responses

async def request_and_sign_challenge(supernode_url: str) -> Dict[str, str]:
    async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS/5)) as client:
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
        # Read the blacklist file if it exists
        blacklist_path = Path('supernode_inference_ip_blacklist.txt')
        blacklisted_ips = set()
        if blacklist_path.exists():
            with blacklist_path.open('r') as blacklist_file:
                blacklisted_ips = {line.strip() for line in blacklist_file if line.strip()}
        else:
            logger.info("Blacklist file not found. Proceeding without blacklist filtering.")
        async with httpx.AsyncClient() as client:
            tasks = []
            supernode_list_df, _ = await check_supernode_list_func()
            for supernode_pastelid in supernodes:
                payload = {}
                try:
                    supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    ip_address_port = supernode_base_url.split("//")[1].split(":")[0]
                    if ip_address_port in blacklisted_ips:
                        logger.info(f"Skipping blacklisted supernode {supernode_pastelid}")
                        continue
                    url = f"{supernode_base_url}/credit_pack_purchase_request_final_response_announcement"
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                    challenge = challenge_dict["challenge"]
                    challenge_id = challenge_dict["challenge_id"]
                    challenge_signature = challenge_dict["challenge_signature"]
                    response_dict = response.model_dump()
                    response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in response_dict.items()}
                    payload = {
                        "response": response_dict,
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
        traceback.print_exc()
        raise    

def transform_json(input_string):
    # Parse the JSON string into a Python dictionary
    data = json.loads(input_string)
    # Serialize the dictionary back to a JSON string
    # Ensure that special characters are escaped correctly
    transformed_string = json.dumps(data, ensure_ascii=False)
    return transformed_string

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
            credit_pack_purchase_request_fields_json_b64=price_agreement_request.credit_pack_purchase_request_fields_json_b64,
            agree_with_proposed_price=agree_with_proposed_price,
            credit_usage_tracking_psl_address=price_agreement_request.credit_usage_tracking_psl_address,
            proposed_psl_price_per_credit=price_agreement_request.proposed_psl_price_per_credit,
            proposed_price_agreement_response_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            proposed_price_agreement_response_pastel_block_height=await get_current_pastel_block_height_func(),
            proposed_price_agreement_response_message_version_string="1.0",
            responding_supernode_signature_on_credit_pack_purchase_request_fields_json_b64=await sign_message_with_pastelid_func(MY_PASTELID, price_agreement_request.credit_pack_purchase_request_fields_json_b64, LOCAL_PASTEL_ID_PASSPHRASE),
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
        traceback.print_exc()
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
        potentially_agreeing_supernodes, total_number_of_blacklisted_sns, list_of_blacklisted_supernode_pastelids = await select_potentially_agreeing_supernodes()
        logger.info(f"Selected {len(potentially_agreeing_supernodes)} potentially agreeing supernodes: {potentially_agreeing_supernodes}")
        # Create the price agreement request without the hash and signature fields
        price_agreement_request = db_code.CreditPackPurchasePriceAgreementRequest(
            sha3_256_hash_of_credit_pack_purchase_request_response_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields,
            supernode_requesting_price_agreement_pastelid=MY_PASTELID,
            credit_pack_purchase_request_fields_json_b64=preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64,
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
        use_manual_signature_validation = 0
        for current_price_agreement_response in price_agreement_request_responses:
            if use_manual_signature_validation:
                agreeing_supernode_pastelid = current_price_agreement_response.responding_supernode_pastelid
                sha3_256_hash_of_price_agreement_request_response_fields = current_price_agreement_response.sha3_256_hash_of_price_agreement_request_response_fields
                responding_supernode_signature_on_price_agreement_request_response_hash = current_price_agreement_response.responding_supernode_signature_on_price_agreement_request_response_hash
                is_signature_valid = await verify_message_with_pastelid_func(agreeing_supernode_pastelid, sha3_256_hash_of_price_agreement_request_response_fields, responding_supernode_signature_on_price_agreement_request_response_hash)
                if not is_signature_valid:
                    logger.warning(f"Error validating price agreement request response signature from supernode {current_price_agreement_response.responding_supernode_pastelid}")
                    continue
            response_validation_errors = await validate_credit_pack_ticket_message_data_func(current_price_agreement_response)
            if not response_validation_errors:
                valid_price_agreement_request_responses.append(current_price_agreement_response)
        supernode_price_agreement_response_percentage_achieved = len(valid_price_agreement_request_responses) / len(potentially_agreeing_supernodes)
        logger.info(f"Received {len(valid_price_agreement_request_responses)} valid price agreement responses from potentially agreeing supernodes out of {len(potentially_agreeing_supernodes)} asked (excluding {total_number_of_blacklisted_sns} blacklisted supernodes), for a quorum percentage of {supernode_price_agreement_response_percentage_achieved:.2%} (Required minimum quorum percentage is {SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:.2%})")              
        # Check if enough supernodes responded with valid responses
        if supernode_price_agreement_response_percentage_achieved <= SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:
            logger.warning(f"Not enough supernodes responded with valid price agreement responses; only {supernode_price_agreement_response_percentage_achieved:.2%} of the supernodes responded, less than the required quorum percentage of {SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:.2%}")
            logger.info("Responding to end user with termination message...")
            termination_message = db_code.CreditPackPurchaseRequestResponseTermination(
                sha3_256_hash_of_credit_pack_purchase_request_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
                credit_pack_purchase_request_fields_json_b64=preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64,
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
                credit_pack_purchase_request_fields_json_b64=preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64,
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
        # Select top N closest supernodes to the best block merkle root for inclusion in the response
        _, best_block_merkle_root, best_block_height = await get_best_block_hash_and_merkle_root_func()
        list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion = await select_top_n_closest_supernodes_to_best_block_merkle_root(list_of_agreeing_supernodes, n=10, best_block_merkle_root=best_block_merkle_root)
        # Aggregate the signatures from the selected agreeing supernodes
        selected_agreeing_supernodes_signatures_dict = {}
        for response in valid_price_agreement_request_responses:
            if response.responding_supernode_pastelid in list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion:
                selected_agreeing_supernodes_signatures_dict[response.responding_supernode_pastelid] = {
                    "price_agreement_request_response_hash_signature": response.responding_supernode_signature_on_price_agreement_request_response_hash,
                    "credit_pack_purchase_request_fields_json_b64_signature": response.responding_supernode_signature_on_credit_pack_purchase_request_fields_json_b64
                }
        if isinstance(preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64, str):
            credit_pack_purchase_request_fields_json = base64.b64decode(preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64).decode('utf-8')
            credit_request_response_dict = json.loads(credit_pack_purchase_request_fields_json)
        requested_initial_credits_in_credit_pack = credit_request_response_dict['requested_initial_credits_in_credit_pack']
        # Create the credit pack purchase request response
        credit_pack_purchase_request_response = db_code.CreditPackPurchaseRequestResponse(
            sha3_256_hash_of_credit_pack_purchase_request_fields=preliminary_price_quote_response.sha3_256_hash_of_credit_pack_purchase_request_fields,
            credit_pack_purchase_request_fields_json_b64=preliminary_price_quote_response.credit_pack_purchase_request_fields_json_b64,
            psl_cost_per_credit=preliminary_price_quote_response.preliminary_quoted_price_per_credit_in_psl,
            proposed_total_cost_of_credit_pack_in_psl=requested_initial_credits_in_credit_pack*preliminary_price_quote_response.preliminary_quoted_price_per_credit_in_psl,
            credit_usage_tracking_psl_address=preliminary_price_quote_response.credit_usage_tracking_psl_address,
            request_response_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            request_response_pastel_block_height=await get_current_pastel_block_height_func(),
            best_block_merkle_root=best_block_merkle_root,
            best_block_height=best_block_height,
            credit_purchase_request_response_message_version_string="1.0",
            responding_supernode_pastelid=MY_PASTELID,
            list_of_blacklisted_supernode_pastelids=list_of_blacklisted_supernode_pastelids,
            list_of_potentially_agreeing_supernodes=potentially_agreeing_supernodes,
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms=list_of_agreeing_supernodes,
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion=list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion,
            selected_agreeing_supernodes_signatures_dict=selected_agreeing_supernodes_signatures_dict,
            sha3_256_hash_of_credit_pack_purchase_request_response_fields="",
            responding_supernode_signature_on_credit_pack_purchase_request_response_hash=""
        )
        logger.info(f"Now generating the final credit pack purchase request response and assembling the {len(list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion)} selected agreeing supernode signatures...")
        # Generate the hash and signature fields
        response_fields_that_are_hashed = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(credit_pack_purchase_request_response)  # noqa: F841
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
        logger.info(f"Now sending the final credit pack purchase request response to the list of {len(list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion)} selected agreeing supernodes: {list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion}")
        announcement_responses = await send_credit_pack_purchase_request_final_response_to_supernodes(credit_pack_purchase_request_response, list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion)
        logger.info(f"Received {len(announcement_responses)} responses to the final credit pack purchase request response announcement, of which {len([response for response in announcement_responses if response.status_code == 200])} were successful")
        return credit_pack_purchase_request_response
    except Exception as e:
        logger.error(f"Error processing credit purchase preliminary price quote response: {str(e)}")
        traceback.print_exc()
        raise

async def get_credit_purchase_request_status(status_request: db_code.CreditPackRequestStatusCheck) -> db_code.CreditPackPurchaseRequestStatus:
    try:
        # Validate the request fields
        if not status_request.sha3_256_hash_of_credit_pack_purchase_request_fields or not status_request.requesting_end_user_pastelid:
            raise ValueError("Invalid status check request")
        await save_credit_pack_purchase_request_status_check(status_request)
        # Retrieve the credit pack purchase request
        credit_pack_purchase_request = await get_credit_pack_purchase_request(status_request.sha3_256_hash_of_credit_pack_purchase_request_fields)
        # Retrieve the credit pack purchase request response
        credit_pack_purchase_request_response = await get_credit_pack_purchase_request_response_from_request_hash(status_request.sha3_256_hash_of_credit_pack_purchase_request_fields)
        # Check the status of the credit pack purchase request
        status = await check_credit_pack_purchase_request_status(credit_pack_purchase_request)
        # Determine the status details based on the status
        if status == "pending":
            status_details = "Waiting for the credit pack purchase request to be processed"
        elif status == "approved":
            status_details = "Credit pack purchase request has been approved, waiting for confirmation"
        elif status == "confirmed":
            status_details = "Credit pack purchase has been confirmed, waiting for completion"
        elif status == "completed":
            status_details = "Credit pack purchase has been completed successfully"
        elif status == "failed":
            status_details = "Credit pack purchase has failed"
        else:
            status_details = "Unknown status"
        # Create the response
        response = db_code.CreditPackPurchaseRequestStatus(
            sha3_256_hash_of_credit_pack_purchase_request_fields=status_request.sha3_256_hash_of_credit_pack_purchase_request_fields,
            sha3_256_hash_of_credit_pack_purchase_request_response_fields=credit_pack_purchase_request_response.sha3_256_hash_of_credit_pack_purchase_request_response_fields if credit_pack_purchase_request_response else "",
            status=status,
            status_details=status_details,
            status_update_timestamp_utc_iso_string=datetime.now(dt.UTC).isoformat(),
            status_update_pastel_block_height=await get_current_pastel_block_height_func(),
            credit_purchase_request_status_message_version_string="1.0",
            responding_supernode_pastelid=credit_pack_purchase_request_response.responding_supernode_pastelid if credit_pack_purchase_request_response else MY_PASTELID,
            sha3_256_hash_of_credit_pack_purchase_request_status_fields="",
            responding_supernode_signature_on_credit_pack_purchase_request_status_hash=""
        )
        # Generate the hash and signature fields
        response.sha3_256_hash_of_credit_pack_purchase_request_status_fields = await compute_sha3_256_hash_of_sqlmodel_response_fields(response)
        response.responding_supernode_signature_on_credit_pack_purchase_request_status_hash = await sign_message_with_pastelid_func(
            MY_PASTELID,
            response.sha3_256_hash_of_credit_pack_purchase_request_status_fields,
            LOCAL_PASTEL_ID_PASSPHRASE
        )
        await save_credit_pack_purchase_request_status_check(response)
        return response
    except Exception as e:
        logger.error(f"Error getting credit purchase request status: {str(e)}")
        traceback.print_exc()
        raise
    
async def process_credit_pack_purchase_request_final_response_announcement(response: db_code.CreditPackPurchaseRequestResponse) -> None:
    try:
        # Validate the response fields
        if not response.sha3_256_hash_of_credit_pack_purchase_request_fields or not response.credit_pack_purchase_request_fields_json_b64:
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
        traceback.print_exc()
        raise    
    
async def send_credit_pack_storage_completion_announcement_to_supernodes(response: db_code.CreditPackPurchaseRequestConfirmationResponse, agreeing_supernode_pastelids: List[str]) -> List[httpx.Response]:
    try:
        # Read the blacklist file if it exists
        blacklist_path = Path('supernode_inference_ip_blacklist.txt')
        blacklisted_ips = set()
        if blacklist_path.exists():
            with blacklist_path.open('r') as blacklist_file:
                blacklisted_ips = {line.strip() for line in blacklist_file if line.strip()}
        else:
            logger.info("Blacklist file not found. Proceeding without blacklist filtering.")
        
        async with httpx.AsyncClient() as client:
            tasks = []
            supernode_list_df, _ = await check_supernode_list_func()
            for supernode_pastelid in agreeing_supernode_pastelids:
                payload = {}
                try:
                    supernode_base_url = await get_supernode_url_from_pastelid_func(supernode_pastelid, supernode_list_df)
                    ip_address_port = supernode_base_url.split("//")[1].split(":")[0]
                    if ip_address_port in blacklisted_ips:
                        logger.info(f"Skipping blacklisted supernode {supernode_pastelid}")
                        continue
                    url = f"{supernode_base_url}/credit_pack_storage_completion_announcement"
                    challenge_dict = await request_and_sign_challenge(supernode_base_url)
                    challenge = challenge_dict["challenge"]
                    challenge_id = challenge_dict["challenge_id"]
                    challenge_signature = challenge_dict["challenge_signature"]
                    response_dict = response.model_dump()
                    response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in response_dict.items()}
                    payload = {
                        "storage_completion_announcement": response_dict,
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
        traceback.print_exc()
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
            credit_pack_purchase_request_json = base64.b64decode(credit_pack_purchase_request_response.credit_pack_purchase_request_fields_json_b64).decode('utf-8')
            credit_pack_purchase_request_dict = json.loads(credit_pack_purchase_request_json)
            credit_pack_purchase_request_response_dict = credit_pack_purchase_request_response.model_dump()
            credit_pack_purchase_request_response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in credit_pack_purchase_request_response_dict.items()}
            credit_pack_purchase_request_confirmation_dict = confirmation.model_dump()
            credit_pack_purchase_request_confirmation_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in credit_pack_purchase_request_confirmation_dict.items()}
            credit_pack_combined_blockchain_ticket_data = {"credit_pack_purchase_request_dict": credit_pack_purchase_request_dict, "credit_pack_purchase_request_response_dict": credit_pack_purchase_request_response_dict, "credit_pack_purchase_request_confirmation_dict": credit_pack_purchase_request_confirmation_dict}
            credit_pack_combined_blockchain_ticket_data_json = json.dumps(credit_pack_combined_blockchain_ticket_data)
            credit_pack_ticket_bytes_before_compression = sys.getsizeof(credit_pack_combined_blockchain_ticket_data_json)
            compressed_credit_pack_ticket, _ = await compress_data_with_zstd_func(credit_pack_combined_blockchain_ticket_data_json)
            credit_pack_ticket_bytes_after_compression = sys.getsizeof(compressed_credit_pack_ticket)
            compression_ratio = credit_pack_ticket_bytes_before_compression / credit_pack_ticket_bytes_after_compression
            logger.info(f"Achieved a compression ratio of {compression_ratio:.2f} on credit pack ticket data!")
            logger.info(f"Required burn transaction confirmed with {num_confirmations} confirmations; now attempting to write the credit pack ticket to the blockchain (a total of {credit_pack_ticket_bytes_before_compression:,} bytes before compression and {credit_pack_ticket_bytes_after_compression:,} bytes after compression)...")
            log_action_with_payload("Writing", "the credit pack ticket to the blockchain", credit_pack_combined_blockchain_ticket_data_json)
            pastel_api_credit_pack_ticket_registration_txid, storage_validation_error_string = await store_credit_pack_ticket_in_blockchain(credit_pack_combined_blockchain_ticket_data_json)
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
            logger.info(f"Now attempting to send the credit pack storage completion announcement to the {len(credit_pack_purchase_request_response.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms)} agreeing Supernodes...")
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
        traceback.print_exc()
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
        traceback.print_exc()
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
        traceback.print_exc()
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
            credit_pack_purchase_request_response = await get_credit_pack_purchase_request_response(storage_retry_request.sha3_256_hash_of_credit_pack_purchase_request_response_fields)
            credit_pack_purchase_request_response_dict = credit_pack_purchase_request_response.model_dump()
            credit_pack_purchase_request_json = base64.b64decode(credit_pack_purchase_request_response_dict['credit_pack_purchase_request_fields_json_b64']).decode('utf-8')
            credit_pack_purchase_request_dict = json.loads(credit_pack_purchase_request_json)
            credit_pack_purchase_request_confirmation = await get_credit_pack_purchase_request_confirmation_from_request_hash(credit_pack_purchase_request_response_dict['sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields'])
            credit_pack_purchase_request_confirmation_dict = credit_pack_purchase_request_confirmation.model_dump()
            credit_pack_purchase_request_confirmation_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in credit_pack_purchase_request_confirmation_dict.items()}
            credit_pack_combined_blockchain_ticket_data = {"credit_pack_purchase_request_dict": credit_pack_purchase_request_dict, "credit_pack_purchase_request_response_dict": credit_pack_purchase_request_response_dict, "credit_pack_purchase_request_confirmation_dict": credit_pack_purchase_request_confirmation_dict}
            credit_pack_combined_blockchain_ticket_data_json = json.dumps(credit_pack_combined_blockchain_ticket_data)
            pastel_api_credit_pack_ticket_registration_txid, storage_validation_error_string = await store_credit_pack_ticket_in_blockchain(credit_pack_combined_blockchain_ticket_data_json)
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
        traceback.print_exc()
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
        traceback.print_exc()
        raise
    
async def insert_credit_pack_ticket_txid_into_known_bad_table_in_db(credit_pack_ticket_txid: str, list_of_reasons_it_is_known_bad: str):
    known_bad_txid_object = db_code.CreditPackKnownBadTXID(credit_pack_ticket_txid=credit_pack_ticket_txid, list_of_reasons_it_is_known_bad=list_of_reasons_it_is_known_bad)
    async with db_code.Session() as db_session:
        db_session.add(known_bad_txid_object)
        await db_session.commit()
        await db_session.refresh(known_bad_txid_object)
    return known_bad_txid_object

async def get_list_of_all_known_bad_credit_pack_ticket_txids_from_db():
    async with db_code.Session() as db_session:
        known_bad_txids = await db_session.exec(select(db_code.CreditPackKnownBadTXID))
        known_bad_txid_results = known_bad_txids.all()
        if known_bad_txid_results is not None:
            list_of_all_known_bad_credit_pack_txids = list(set([x.credit_pack_ticket_txid for x in known_bad_txid_results]))          
            return list_of_all_known_bad_credit_pack_txids
        else:
            return []        

async def check_if_credit_pack_ticket_txid_in_list_of_known_bad_txids_in_db(credit_pack_ticket_txid: str):
    async with db_code.Session() as db_session:
        # Query to check if the TXID exists in the known bad TXIDs table
        result = await db_session.execute(
            select(db_code.CreditPackKnownBadTXID)
            .where(db_code.CreditPackKnownBadTXID.credit_pack_ticket_txid == credit_pack_ticket_txid)
        )
        known_bad_txid = result.scalar_one_or_none()
        # Return True if the TXID is found in the known bad list, False otherwise
        return known_bad_txid is not None
    
async def validate_merkle_root_at_block_height(merkle_root_to_check: str, block_height: int) -> bool:
    block_hash = await rpc_connection.getblockhash(block_height)
    block_details = await rpc_connection.getblock(block_hash)
    if block_details['merkleroot'] == merkle_root_to_check:
        return True
    else:
        return False    

async def validate_existing_credit_pack_ticket(credit_pack_ticket_txid: str) -> dict:
    try:
        use_verbose_validation = 0
        logger.info(f"Validating credit pack ticket with TXID: {credit_pack_ticket_txid}")
        # Retrieve the credit pack ticket data from the blockchain
        credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation = await retrieve_credit_pack_ticket_from_blockchain_using_txid(credit_pack_ticket_txid)
        logger.info(f"Credit pack ticket data for credit pack with TXID {credit_pack_ticket_txid}:\n\nTicket Request Response:\n\n {abbreviated_pretty_json_func(credit_pack_purchase_request_response.model_dump())} \n\nTicket Request Confirmation:\n\n {abbreviated_pretty_json_func(credit_pack_purchase_request_confirmation.model_dump())}")
        validation_results = {
            "credit_pack_ticket_is_valid": True,
            "validation_checks": [],
            "validation_failure_reasons_list": []
        }
        # Validate the payment
        matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_transaction(
            credit_pack_purchase_request_confirmation.txid_of_credit_purchase_burn_transaction,
            credit_pack_purchase_request_response.credit_usage_tracking_psl_address,
            credit_pack_purchase_request_response.proposed_total_cost_of_credit_pack_in_psl,
            credit_pack_purchase_request_response.request_response_pastel_block_height
        )
        validation_results["validation_checks"].append({
            "check_name": f"Ticket Payment Burn Transaction validation (Burn payment TXID: {credit_pack_purchase_request_confirmation.txid_of_credit_purchase_burn_transaction} sent with {amount_received_at_burn_address} PSL",
            "is_valid": matching_transaction_found or exceeding_transaction_found
        })
        if not (matching_transaction_found or exceeding_transaction_found):
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Invalid burn transaction for credit pack ticket with TXID: {credit_pack_ticket_txid}")
        active_supernodes_count_at_the_time, active_supernodes_at_the_time = await fetch_active_supernodes_count_and_details(credit_pack_purchase_request_response.request_response_pastel_block_height)
        list_of_active_supernode_pastelids_at_the_time = [x["pastel_id"] for x in active_supernodes_at_the_time]
        list_of_blacklisted_supernode_pastelids = credit_pack_purchase_request_response.list_of_blacklisted_supernode_pastelids
        list_of_potentially_agreeing_supernodes = credit_pack_purchase_request_response.list_of_potentially_agreeing_supernodes
        # Check for new fields
        try:
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms = credit_pack_purchase_request_response.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion = credit_pack_purchase_request_response.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion
            selected_agreeing_supernodes_signatures_dict = credit_pack_purchase_request_response.selected_agreeing_supernodes_signatures_dict
            best_block_merkle_root = credit_pack_purchase_request_response.best_block_merkle_root
            best_block_height = credit_pack_purchase_request_response.best_block_height
            if list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion is None:
                validation_results["credit_pack_ticket_is_valid"] = False
                validation_results["validation_failure_reasons_list"].append("Required field 'list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion' is missing in credit pack purchase request response")
                return validation_results                
        except AttributeError as e:
            logger.error(f"Required field missing in credit pack purchase request response: {str(e)}")
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Required field missing in credit pack purchase request response: {str(e)}")
            return validation_results
        # First check if all included pastelids were valid supernodes at the time:
        for potentially_agreeing_supernode_pastelid in list_of_potentially_agreeing_supernodes:
            potentially_agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height = potentially_agreeing_supernode_pastelid in list_of_active_supernode_pastelids_at_the_time
            validation_results["validation_checks"].append({
                "check_name": f"Potentially agreeing supernode with pastelid {potentially_agreeing_supernode_pastelid} was in the list of active supernodes as of block height {credit_pack_purchase_request_response.request_response_pastel_block_height:,}",
                "is_valid": potentially_agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height
            })    
            if not potentially_agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height:
                validation_results["credit_pack_ticket_is_valid"] = False
                validation_results["validation_failure_reasons_list"].append(f"Potentially agreeing supernode with pastelid {potentially_agreeing_supernode_pastelid} was NOT in the list of active supernodes as of block height {credit_pack_purchase_request_response.request_response_pastel_block_height:,}")
        for agreeing_supernode_pastelid in list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms:
            agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height = agreeing_supernode_pastelid in list_of_active_supernode_pastelids_at_the_time
            validation_results["validation_checks"].append({
                "check_name": f"Agreeing supernode with pastelid {agreeing_supernode_pastelid} was in the list of active supernodes as of block height {credit_pack_purchase_request_response.request_response_pastel_block_height:,}",
                "is_valid": agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height
            })
            if not agreeing_supernode_pastelid_in_list_of_active_supernodes_at_block_height:
                validation_results["credit_pack_ticket_is_valid"] = False
                validation_results["validation_failure_reasons_list"].append(f"Agreeing supernode with pastelid {agreeing_supernode_pastelid} was NOT in the list of active supernodes as of block height {credit_pack_purchase_request_response.request_response_pastel_block_height:,}")            
        # Validate the ticket response hashes:
        credit_pack_purchase_request_response_transformed = parse_sqlmodel_strings_into_lists_and_dicts_func(credit_pack_purchase_request_response)
        credit_pack_purchase_request_confirmation_transformed = parse_sqlmodel_strings_into_lists_and_dicts_func(credit_pack_purchase_request_confirmation)
        validation_errors_in_credit_pack_purchase_request_response = await validate_credit_pack_blockchain_ticket_data_field_hashes(credit_pack_purchase_request_response_transformed)
        validation_errors_in_credit_pack_purchase_request_confirmation = await validate_credit_pack_blockchain_ticket_data_field_hashes(credit_pack_purchase_request_confirmation_transformed) 
        if len(validation_errors_in_credit_pack_purchase_request_response) > 0:
            logger.warning(f"Warning! Computed hash does not match for ticket request response object for credit pack ticket with txid {credit_pack_ticket_txid}; Validation errors detected:\n{validation_errors_in_credit_pack_purchase_request_response}")
            validation_results["validation_checks"].append({
                "check_name": f"Computed hash does not match for ticket request response object for credit pack ticket with txid: {validation_errors_in_credit_pack_purchase_request_response}",
                "is_valid": False
            })            
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append("Hash of credit pack request response object stored in blockchain does not match the hash included in the object.")
        if len(validation_errors_in_credit_pack_purchase_request_confirmation) > 0:
            logger.warning(f"Warning! Computed hash does not match for ticket request confirmation object for credit pack ticket with txid {credit_pack_ticket_txid}; Validation errors detected:\n{validation_errors_in_credit_pack_purchase_request_confirmation}")
            validation_results["validation_checks"].append({
                "check_name": f"Computed hash does not match for ticket request confirmation object for credit pack ticket with txid: {validation_errors_in_credit_pack_purchase_request_confirmation}",
                "is_valid": False
            })            
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append("Hash of credit pack request confirmation object stored in blockchain does not match the hash included in the object.")
        # Validate the signatures
        for agreeing_supernode_pastelid in list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion:
            signatures = selected_agreeing_supernodes_signatures_dict[agreeing_supernode_pastelid]
            if use_verbose_validation:
                logger.info(f"Verifying signature for selected agreeing supernode {agreeing_supernode_pastelid}")
                logger.info(f"Message to verify (decoded from b64): {credit_pack_purchase_request.model_dump()}")
                logger.info(f"Signature: {signatures['credit_pack_purchase_request_fields_json_b64_signature']}")
            is_fields_json_b64_signature_valid = await verify_message_with_pastelid_func(agreeing_supernode_pastelid, 
                                                                                        credit_pack_purchase_request_response.credit_pack_purchase_request_fields_json_b64,
                                                                                        signatures['credit_pack_purchase_request_fields_json_b64_signature'])
            if not is_fields_json_b64_signature_valid:
                logger.warning(f"Warning! Signature failed for SN {agreeing_supernode_pastelid} for credit pack with txid {credit_pack_ticket_txid}")
            if use_verbose_validation:
                logger.info(f"Signature validation result: {is_fields_json_b64_signature_valid}")
            validation_results["validation_checks"].append({
                "check_name": f"Signature validation for selected agreeing supernode {agreeing_supernode_pastelid} on credit pack purchase request fields json",
                "is_valid": is_fields_json_b64_signature_valid
            })
            if not is_fields_json_b64_signature_valid:
                validation_results["credit_pack_ticket_is_valid"] = False
                validation_results["validation_failure_reasons_list"].append(f"Signature failed for SN {agreeing_supernode_pastelid} for credit pack with txid {credit_pack_ticket_txid}")
        # Validate the selected agreeing supernodes
        selected_agreeing_supernodes = await select_top_n_closest_supernodes_to_best_block_merkle_root(
            list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms,
            n=10,
            best_block_merkle_root=best_block_merkle_root
        )
        best_block_merkle_root_matches = await validate_merkle_root_at_block_height(best_block_merkle_root, best_block_height)
        if not best_block_merkle_root_matches:
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Best block merkle root {best_block_merkle_root} does not match actual merkle root as of block height {best_block_height} for credit pack with txid {credit_pack_ticket_txid}")
        max_block_height_difference_between_best_block_height_and_request_response_pastel_block_height = 10
        actual_block_height_difference_between_best_block_height_and_request_response_pastel_block_height = abs(credit_pack_purchase_request_response.request_response_pastel_block_height - best_block_height)
        if actual_block_height_difference_between_best_block_height_and_request_response_pastel_block_height > max_block_height_difference_between_best_block_height_and_request_response_pastel_block_height:
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Block height difference between specified best block height {best_block_height} and request response pastel block height {credit_pack_purchase_request_response.request_response_pastel_block_height} exceeds the maximum allowed difference of {max_block_height_difference_between_best_block_height_and_request_response_pastel_block_height} for credit pack with txid {credit_pack_ticket_txid}")
        selected_agreeing_supernodes_set = set(list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms_selected_for_signature_inclusion)
        computed_selected_agreeing_supernodes_set = set(selected_agreeing_supernodes)
        is_selected_agreeing_supernodes_valid = selected_agreeing_supernodes_set == computed_selected_agreeing_supernodes_set
        validation_results["validation_checks"].append({
            "check_name": "Validation of selected agreeing supernodes for signature inclusion",
            "is_valid": is_selected_agreeing_supernodes_valid,
            "expected_selected_agreeing_supernodes": list(computed_selected_agreeing_supernodes_set),
            "actual_selected_agreeing_supernodes": list(selected_agreeing_supernodes_set)
        })
        if not is_selected_agreeing_supernodes_valid:
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append("Selected agreeing supernodes for signature inclusion do not match the expected set based on XOR distance to the best block merkle root.")
        num_potentially_agreeing_supernodes = len(list_of_potentially_agreeing_supernodes)
        num_agreeing_supernodes = len(credit_pack_purchase_request_response.list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms)
        number_of_blacklisted_supernodes_at_the_time = len(list_of_blacklisted_supernode_pastelids)
        masternode_list_full_df, _ = await check_supernode_list_func()   
        current_block_height = await get_current_pastel_block_height_func()        
        active_supernodes_count_now, _ = await fetch_active_supernodes_count_and_details(current_block_height)
        adjustment_number_of_supernodes = max([0, active_supernodes_count_now - len(masternode_list_full_df)])
        quorum_percentage = num_potentially_agreeing_supernodes / (active_supernodes_count_at_the_time - number_of_blacklisted_supernodes_at_the_time - adjustment_number_of_supernodes)
        agreeing_percentage = num_agreeing_supernodes / num_potentially_agreeing_supernodes
        is_quorum_valid = quorum_percentage >= SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE
        is_agreeing_percentage_valid = agreeing_percentage >= SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE
        validation_results["validation_checks"].append({
            "check_name": "Agreeing supernodes percentage validation",
            "is_valid": is_agreeing_percentage_valid and is_quorum_valid,
            "quorum_percentage": round(quorum_percentage, 5),
            "agreeing_percentage": round(agreeing_percentage, 5)
        })
        if not is_quorum_valid:
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Agreeing supernodes quorum percentage validation failed for credit pack with txid {credit_pack_ticket_txid}; quorum % was only {quorum_percentage:.3f} (i.e., {num_potentially_agreeing_supernodes} out of {(active_supernodes_count_at_the_time - number_of_blacklisted_supernodes_at_the_time)}) and the minimum required % is {SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE:.3f}")
        if not is_agreeing_percentage_valid:
            validation_results["credit_pack_ticket_is_valid"] = False
            validation_results["validation_failure_reasons_list"].append(f"Agreeing supernodes agreement percentage validation failed for credit pack with txid {credit_pack_ticket_txid}; agreement % was only {agreeing_percentage:.3f} (i.e., {num_agreeing_supernodes} out of {num_potentially_agreeing_supernodes}) and the minimum required % is {SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE:.3f}")
        if len(validation_results["validation_failure_reasons_list"]) > 0: # Construct final validation results response taking all tests into account: 
            logger.info(f"Validation failures for credit pack ticket with TXID {credit_pack_ticket_txid}: {validation_results['validation_failure_reasons_list']}")
            list_of_reasons_it_is_known_bad = json.dumps(validation_results['validation_failure_reasons_list'])
            known_bad_txid_object = await insert_credit_pack_ticket_txid_into_known_bad_table_in_db(credit_pack_ticket_txid, list_of_reasons_it_is_known_bad)
            if known_bad_txid_object:
                logger.info(f"Added invalid credit pack ticket TXID {credit_pack_ticket_txid} to known bad list in database!")
        else:
            logger.info(f"All validation checks passed for credit pack ticket with TXID {credit_pack_ticket_txid}")
        return validation_results
    except Exception as e:
        logger.error(f"Error validating credit pack ticket with TXID {credit_pack_ticket_txid}: {str(e)}")
        traceback.print_exc()
        raise

async def get_valid_credit_pack_tickets_for_pastelid(pastelid: str) -> List[dict]:
    async def process_request_confirmation(request_confirmation):
        hash_of_request_fields = request_confirmation.sha3_256_hash_of_credit_pack_purchase_request_fields
        async with db_code.Session() as db_session:
            txid_mapping = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestResponseTxidMapping)
                .where(db_code.CreditPackPurchaseRequestResponseTxidMapping.sha3_256_hash_of_credit_pack_purchase_request_fields == hash_of_request_fields)
            )
            txid_mapping = txid_mapping.one_or_none()
        if txid_mapping:
            txid = txid_mapping.pastel_api_credit_pack_ticket_registration_txid
            async with db_code.Session() as db_session:
                existing_data_result = await db_session.exec(
                    select(db_code.CreditPackCompleteTicketWithBalance)
                    .where(db_code.CreditPackCompleteTicketWithBalance.credit_pack_ticket_registration_txid == txid)
                )
                existing_data = existing_data_result.one_or_none()
            if existing_data:
                complete_ticket = json.loads(existing_data.complete_credit_pack_data_json)
                current_credit_balance, number_of_confirmation_transactions = await determine_current_credit_pack_balance_based_on_tracking_transactions(txid)
                complete_ticket['credit_pack_current_credit_balance'] = current_credit_balance
                complete_ticket['balance_as_of_datetime'] = datetime.now(dt.UTC).isoformat()
                complete_ticket_json = json.dumps(complete_ticket)
                async with db_code.Session() as db_session:
                    existing_data.complete_credit_pack_data_json = complete_ticket_json
                    existing_data.datetime_last_updated = datetime.now(dt.UTC)
                    db_session.add(existing_data)
                    await db_session.commit()
            else:
                current_credit_balance, number_of_confirmation_transactions = await determine_current_credit_pack_balance_based_on_tracking_transactions(txid)
                _, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation = await retrieve_credit_pack_ticket_using_txid(txid)
                if all((credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation)):
                    credit_pack_purchase_request_fields_json = base64.b64decode(credit_pack_purchase_request_response.credit_pack_purchase_request_fields_json_b64).decode('utf-8')
                    credit_pack_purchase_request = json.loads(credit_pack_purchase_request_fields_json)
                    complete_ticket = {
                        "credit_pack_purchase_request": credit_pack_purchase_request,
                        "credit_pack_purchase_request_response": credit_pack_purchase_request_response.model_dump(),
                        "credit_pack_purchase_request_confirmation": credit_pack_purchase_request_confirmation.model_dump(),
                        "credit_pack_registration_txid": txid,
                        "credit_pack_current_credit_balance": current_credit_balance,
                        "balance_as_of_datetime": datetime.now(dt.UTC).isoformat()
                    }
                    complete_ticket = convert_uuids_to_strings(complete_ticket)
                    complete_ticket = normalize_data(complete_ticket)
                    complete_ticket_json = json.dumps(complete_ticket)
                    async with db_code.Session() as db_session:
                        if existing_data:
                            existing_data.complete_credit_pack_data_json = complete_ticket_json
                            existing_data.datetime_last_updated = datetime.now(dt.UTC)
                            db_session.add(existing_data)
                        else:
                            new_complete_ticket = db_code.CreditPackCompleteTicketWithBalance(
                                credit_pack_ticket_registration_txid=txid,
                                complete_credit_pack_data_json=complete_ticket_json,
                                datetime_last_updated=datetime.now(dt.UTC)
                            )
                            db_session.add(new_complete_ticket)
                        await db_session.commit()
            return complete_ticket
        return None
    try:
        async with db_code.Session() as db_session:
            credit_pack_request_confirmations_results = await db_session.exec(
                select(db_code.CreditPackPurchaseRequestConfirmation)
                .where(db_code.CreditPackPurchaseRequestConfirmation.requesting_end_user_pastelid == pastelid)
            )
            credit_pack_request_confirmations = credit_pack_request_confirmations_results.all()
        tasks = [process_request_confirmation(request_confirmation) for request_confirmation in credit_pack_request_confirmations]
        complete_tickets = await asyncio.gather(*tasks)
        # Filter out None values (if any)
        complete_tickets = [ticket for ticket in complete_tickets if ticket]
        return complete_tickets
    except Exception as e:
        logger.error(f"Error retrieving credit pack tickets for PastelID {pastelid}: {str(e)}")
        traceback.print_exc()
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
        traceback.print_exc()
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
        await db_session.commit()
        await db_session.refresh(inference_request_model)
    return inference_request_model

def get_fallback_tokenizer():
    return "gpt2"  # Default to "gpt2" tokenizer as a fallback

def get_tokenizer(model_name: str):
    model_name = model_name.replace('swiss_army_llama-', '')
    model_to_tokenizer_mapping = {
        "claude3": "Xenova/claude-tokenizer",
        "phi": "TheBloke/phi-2-GGUF",
        "openai": "cl100k_base",
        "groq-llama3": "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
        "groq-mixtral": "EleutherAI/gpt-neox-20b",
        "groq-gemma": "google/flan-ul2",
        "bge-m3": "Shitao/bge-m3",
        "mistralapi": "mistralai/Mistral-7B-Instruct-v0.2",
        "stability": "openai/clip-vit-large-patch14",
        "Lexi-Llama-3-8B-Uncensored_Q5_K_M": "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
        "Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M": "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
        "Meta-Llama-3-8B-Instruct.Q3_K_S": "gradientai/Llama-3-70B-Instruct-Gradient-1048k",
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
        "openrouter/stabilityai/stablelm-tuned-alpha-7b": "stabilityai/stablelm-base-alpha-7b",
        "openrouter/togethercomputer/GPT-JT-6B-v1": "togethercomputer/GPT-JT-6B-v1",
        "openrouter/tiiuae/falcon-7b-instruct": "tiiuae/falcon-7b-instruct"
    }
    best_match = process.extractOne(model_name.lower(), model_to_tokenizer_mapping.keys())
    return model_to_tokenizer_mapping.get(best_match[0], get_fallback_tokenizer())  # Default to fallback tokenizer if no match found

def count_tokens(model_name: str, input_data: str) -> int:
    tokenizer_name = get_tokenizer(model_name)
    logger.info(f"Selected tokenizer {tokenizer_name} for model {model_name}")
    try:
        if 'claude' in model_name.lower():
            tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_name)
        elif 'whisper' in model_name.lower():
            tokenizer = WhisperTokenizer.from_pretrained(tokenizer_name)
        elif 'clip-interrogator' in model_name.lower() or 'stability' in model_name.lower():
            return 0
        elif 'videocap-transformer' in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        elif 'openai' in model_name.lower():
            encoding = tiktoken.get_encoding(tokenizer_name)
            input_tokens = encoding.encode(input_data)
            return len(input_tokens)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if hasattr(tokenizer, "encode"):
            input_tokens = tokenizer.encode(input_data)
        else:
            input_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_data))
        return len(input_tokens)
    except Exception as e:
        logger.error(f"Failed to load tokenizer {tokenizer_name} for model {model_name}: {e}")
        fallback_tokenizer_name = get_fallback_tokenizer()
        logger.info(f"Falling back to tokenizer {fallback_tokenizer_name}")
        if 'openai' in fallback_tokenizer_name.lower():
            encoding = tiktoken.get_encoding(fallback_tokenizer_name)
            input_tokens = encoding.encode(input_data)
        else:
            tokenizer = AutoTokenizer.from_pretrained(fallback_tokenizer_name)
            if hasattr(tokenizer, "encode"):
                input_tokens = tokenizer.encode(input_data)
            else:
                input_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_data))
        return len(input_tokens)

def calculate_api_cost(model_name: str, input_data: str, model_parameters: Dict) -> float:
    # Define the pricing data for each API service and model
    logger.info(f"Evaluating API cost for model: {model_name}")
    pricing_data = {
        "claude-2.1": {"input_cost": 0.008, "output_cost": 0.024, "per_call_cost": 0.0128},
        "claude3-haiku": {"input_cost": 0.00025, "output_cost": 0.00125, "per_call_cost": 0.0006},
        "claude3.5-sonnet": {"input_cost": 0.0003, "output_cost": 0.0015, "per_call_cost": 0.0},
        "claude3-opus": {"input_cost": 0.0015, "output_cost": 0.0075, "per_call_cost": 0.0},
        "mistralapi-mistral-small-latest": {"input_cost": 0.002, "output_cost": 0.006, "per_call_cost": 0.0032},
        "mistralapi-mistral-medium-latest": {"input_cost": 0.0027, "output_cost": 0.0081, "per_call_cost": 0},
        "mistralapi-mistral-large-latest": {"input_cost": 0.008, "output_cost": 0.024, "per_call_cost": 0.0128},
        "mistralapi-mistral-7b": {"input_cost": 0.00025, "output_cost": 0.00025, "per_call_cost": 0},
        "mistralapi-open-mistral-7b": {"input_cost": 0.00025, "output_cost": 0.00025, "per_call_cost": 0},
        "mistralapi-open-mixtral-8x7b": {"input_cost": 0.0007, "output_cost": 0.0007, "per_call_cost": 0},
        "mistralapi-mistral-embed": {"input_cost": 0.0001, "output_cost": 0, "per_call_cost": 0},
        "openai-gpt-4o": {"input_cost": 0.005, "output_cost": 0.015, "per_call_cost": 0.0},
        "openai-gpt-4o-2024-05-13": {"input_cost": 0.005, "output_cost": 0.015, "per_call_cost": 0.0},
        "openai-gpt-4o-vision": {"input_cost": 0.005, "output_cost": 0.015, "per_call_cost": 0.0},
        "openai-gpt-4-turbo": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0.0},
        "openai-gpt-4-turbo-2024-04-09": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0.0},
        "openai-gpt-4": {"input_cost": 0.03, "output_cost": 0.06, "per_call_cost": 0.0},
        "openai-gpt-4-32k": {"input_cost": 0.06, "output_cost": 0.12, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-0125": {"input_cost": 0.0005, "output_cost": 0.0015, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-instruct": {"input_cost": 0.0015, "output_cost": 0.002, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-16k": {"input_cost": 0.003, "output_cost": 0.004, "per_call_cost": 0.0},
        "openai-gpt-4-0125-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0.0},
        "openai-gpt-4-1106-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0.0},
        "openai-gpt-4-vision-preview": {"input_cost": 0.01, "output_cost": 0.03, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-1106": {"input_cost": 0.001, "output_cost": 0.002, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-0613": {"input_cost": 0.0015, "output_cost": 0.002, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-16k-0613": {"input_cost": 0.003, "output_cost": 0.004, "per_call_cost": 0.0},
        "openai-gpt-3.5-turbo-0301": {"input_cost": 0.0015, "output_cost": 0.002, "per_call_cost": 0.0},
        "openai-text-embedding-ada-002": {"input_cost": 0.0004, "output_cost": 0.0, "per_call_cost": 0.0},
        "groq-llama3-70b-8192": {"input_cost": 0.0007, "output_cost": 0.0008, "per_call_cost": 0},
        "groq-llama3-8b-8192": {"input_cost": 0.0001, "output_cost": 0.0001, "per_call_cost": 0},
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
    elif model_name.endswith("4o-vision"):
        if input_data is None:
            logger.error("Input data is empty!")
        input_data_dict = json.loads(input_data)
        image_data_binary = base64.b64decode(input_data_dict["image"]) # Decode image data and question from input_data
        question = input_data_dict["question"]
        # Calculate image resolution
        image = Image.open(io.BytesIO(image_data_binary))
        width, height = image.size
        image_file_size_in_mb = len(image_data_binary) / (1024 * 1024)  # Calculate image file size in MB
        logger.info(f"Submitted image file is {image_file_size_in_mb:.2f}MB and has resolution of {width} x {height}")
        # Resize logic to fit the larger dimension to 1286 pixels while maintaining aspect ratio
        target_larger_dimension = 1286
        aspect_ratio = width / height
        if width > height:
            resized_width = target_larger_dimension
            resized_height = int(target_larger_dimension / aspect_ratio)
        else:
            resized_height = target_larger_dimension
            resized_width = int(target_larger_dimension * aspect_ratio)
        logger.info(f"Image will be resized automatically by OpenAI to a resolution of {resized_width} x {resized_height}")
        # Calculate number of tiles and tokens
        tiles_x = -(-resized_width // 512)
        tiles_y = -(-resized_height // 512)
        total_tiles = tiles_x * tiles_y
        base_tokens = 85
        tile_tokens = 170 * total_tiles
        total_tokens = base_tokens + tile_tokens
        # Count tokens in the question
        question_tokens = count_tokens(model_name, question)
        input_data_tokens = total_tokens + question_tokens
        logger.info(f"Total input data tokens: {input_data_tokens}")
        number_of_tokens_to_generate = model_parameters.get("number_of_tokens_to_generate", 1000)
        number_of_completions_to_generate = model_parameters.get("number_of_completions_to_generate", 1)
        input_cost = float(model_pricing["input_cost"]) * float(input_data_tokens) / 1000.0
        output_cost = float(model_pricing["output_cost"]) * float(number_of_tokens_to_generate) / 1000.0
        per_call_cost = float(model_pricing["per_call_cost"]) * float(number_of_completions_to_generate)
        estimated_cost = input_cost + output_cost + per_call_cost
    else:
        # For other models, calculate the cost based on input/output tokens and per-call cost
        input_data_tokens = count_tokens(model_name, input_data)
        logger.info(f"Total input data tokens: {input_data_tokens}")
        number_of_tokens_to_generate = model_parameters.get("number_of_tokens_to_generate", 1000)
        number_of_completions_to_generate = model_parameters.get("number_of_completions_to_generate", 1)
        input_cost = float(model_pricing["input_cost"]) * float(input_data_tokens) / 1000.0
        output_cost = float(model_pricing["output_cost"]) * float(number_of_tokens_to_generate) / 1000.0
        per_call_cost = float(model_pricing["per_call_cost"]) * float(number_of_completions_to_generate)
        estimated_cost = input_cost + output_cost + per_call_cost
    logger.info(f"Estimated cost: ${estimated_cost:.4f}")
    return estimated_cost

async def convert_document_to_sentences(file_content: bytes, tried_local=False) -> Dict:
    logger.info("Now calling Swiss Army Llama to convert document to sentences.")
    local_swiss_army_llama_responding = is_swiss_army_llama_responding(local=True)
    if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        remote_swiss_army_llama_responding = is_swiss_army_llama_responding(local=False)
    else:
        remote_swiss_army_llama_responding = 0
    if remote_swiss_army_llama_responding and USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        port = REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT
    elif local_swiss_army_llama_responding:
        port = SWISS_ARMY_LLAMA_PORT
    else:
        logger.error(f"Neither the local Swiss Army Llama (supposed to be running on port {SWISS_ARMY_LLAMA_PORT}) nor the remote Swiss Army Llama (supposed to be running, if enabled, on mapped port {REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT}) is responding!")
        raise ValueError("Swiss Army Llama is not responding.")
    metadata = await upload_and_get_file_metadata(file_content, "document")
    file_url = metadata["file_url"]
    file_hash = metadata["file_hash"]
    file_size = metadata["file_size"]
    url = f"http://localhost:{port}/convert_document_to_sentences/"
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            response = await client.post(url, data={
                "url": file_url,
                "hash": file_hash,
                "size": file_size,
                "token": SWISS_ARMY_LLAMA_SECURITY_TOKEN
            })
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to convert document to sentences: {e}")
            if port == REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT and not tried_local:
                logger.info("Falling back to local Swiss Army Llama.")
                return await convert_document_to_sentences(file_content, tried_local=True)
            raise ValueError("Error converting document to sentences")
            
async def calculate_proposed_inference_cost_in_credits(requested_model_data: Dict, model_parameters: Dict, model_inference_type_string: str, input_data: str) -> float:
    model_name = requested_model_data["model_name"]
    if 'swiss_army_llama-' not in model_name:
        api_cost = calculate_api_cost(model_name, input_data, model_parameters)
        if api_cost > 0.0:
            target_value_per_credit = TARGET_VALUE_PER_CREDIT_IN_USD
            target_profit_margin = TARGET_PROFIT_MARGIN
            proposed_cost_in_credits = api_cost / (target_value_per_credit * (1 - target_profit_margin))
            final_proposed_cost_in_credits = max([MINIMUM_COST_IN_CREDITS, round(proposed_cost_in_credits, 1)])
            logger.info(f"Proposed cost in credits (API-based): {final_proposed_cost_in_credits}")
            return final_proposed_cost_in_credits
    else:
        credit_costs = requested_model_data["credit_costs"][model_inference_type_string]
        if model_inference_type_string == "ask_question_about_an_image":
            input_tokens = 3000
        else:
            input_tokens = count_tokens(model_name, input_data) if model_inference_type_string != "embedding_document" else 0
        compute_cost = float(credit_costs["compute_cost"])
        memory_cost = float(credit_costs["memory_cost"])
        if model_inference_type_string == "text_completion" or model_inference_type_string == "ask_question_about_an_image":
            output_token_cost = float(credit_costs["output_tokens"])
            number_of_tokens_to_generate = int(model_parameters.get("number_of_tokens_to_generate", 1000))
            number_of_completions_to_generate = int(model_parameters.get("number_of_completions_to_generate", 1))
            estimated_output_tokens = number_of_tokens_to_generate
            proposed_cost_in_credits = number_of_completions_to_generate * (
                (input_tokens * float(credit_costs["input_tokens"])) +
                (estimated_output_tokens * output_token_cost) +
                compute_cost
            ) + memory_cost
        elif model_inference_type_string == "embedding_document":
            if is_base64_encoded(input_data):
                input_data = base64.b64decode(input_data)
                input_data = input_data.decode('utf-8')
            try:
                input_data_dict = json.loads(input_data)
                document_file_data = input_data_dict['document']
                if is_base64_encoded(document_file_data):
                    document_file_data = base64.b64decode(document_file_data)
                document_stats = await convert_document_to_sentences(document_file_data)
                sentences = document_stats["individual_sentences"]
                total_sentences = document_stats["total_number_of_sentences"]
                concatenated_sentences = " ".join(sentences)
                total_tokens = count_tokens(model_name, concatenated_sentences)
                proposed_cost_in_credits = (
                    (total_tokens * float(credit_costs["average_tokens_per_sentence"])) +
                    (total_sentences * float(credit_costs["total_sentences"])) +
                    (1 if model_parameters.get("query_string") else 0) * float(credit_costs["query_string_included"]) +
                    compute_cost +
                    memory_cost
                )
            except Exception as e:
                logger.error(f"Error parsing document data from input: {str(e)}")
                traceback.print_exc()
                raise
        elif model_inference_type_string == "embedding_audio":
            if is_base64_encoded(input_data):
                input_data = base64.b64decode(input_data)
                input_data = input_data.decode('utf-8')
            try:
                input_data_dict = json.loads(input_data)
                audio_file_data = input_data_dict['audio']
                if is_base64_encoded(audio_file_data):
                    audio_file_data = base64.b64decode(audio_file_data)            
                audio_length_seconds = get_audio_length(audio_file_data)
                proposed_cost_in_credits = (
                    (1 if model_parameters.get("query_string") else 0) * float(credit_costs["query_string_included"]) +
                    (audio_length_seconds * float(credit_costs["audio_file_length_in_seconds"])) +
                    compute_cost +
                    memory_cost
                )
            except Exception as e:
                logger.error(f"Error parsing document data from input: {str(e)}")
                traceback.print_exc()
                raise
        final_proposed_cost_in_credits = round(proposed_cost_in_credits * CREDIT_COST_MULTIPLIER_FACTOR, 1)
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

def normalize_string(s):
    """Remove non-alphanumeric characters and convert to lowercase."""
    return re.sub(r'\W+', '', s).lower()

def validate_pastel_txid_string(input_string: str):
    # Sample txid: 625694b632a05f5df8d70904b9b3ff03d144aec0352b2290a275379586daf8db
    return re.match(r'^[0-9a-fA-F]{64}$', input_string) is not None

def is_swiss_army_llama_responding(local=True):
    port = SWISS_ARMY_LLAMA_PORT if local else REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT
    try:
        url = f"http://localhost:{port}/get_list_of_available_model_names/"
        params = {'token': SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        response = httpx.get(url, params=params)
        return response.status_code == 200
    except Exception as e:
        logger.error("Error: {}".format(e))
        return False

async def check_if_input_text_would_get_rejected_from_api_services(input_text: str) -> bool:
    inference_request_allowed = True
    local_swiss_army_llama_responding = is_swiss_army_llama_responding(local=True)
    if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        remote_swiss_army_llama_responding = is_swiss_army_llama_responding(local=False)
    else:
        remote_swiss_army_llama_responding = 0
    if remote_swiss_army_llama_responding and USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        port = REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT
    elif local_swiss_army_llama_responding:
        port = SWISS_ARMY_LLAMA_PORT
    else:
        logger.error(f"Neither the local Swiss Army Llama (supposed to be running on port {SWISS_ARMY_LLAMA_PORT}) nor the remote Swiss Army Llama (supposed to be running, if enabled, on mapped port {REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT}) is responding! Unable to check if request is allowed, so just saying request status simply marked as 'ACCEPT'!")
        return None
    start_time = time.time()
    logger.info("Now checking suitability of requested inference for API services to avoid banning (using Swiss Army Llama with model TinyLlama-1.1B-Chat)...")
    input_prompt = "A user wants to do inference using the following prompt text using the OpenAI API; is the content or nature of the request problematic or offensive in any way that makes it likely that OpenAI will reject the request for safety reasons and potentially ban the API key for submitting it? Respond ONLY with a one word answer-- `REJECT` if the request will cause problems with OpenAI, or `ACCEPT` if it's unlikely to pose a problem. Here is the inference request submitted by the user:\n\n{}".format(input_text)
    url = f"http://localhost:{port}/get_text_completions_from_input_prompt/"
    try:
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS * 3)) as client:
            payload = {
                "input_prompt": input_prompt,
                "llm_model_name": "Meta-Llama-3-8B-Instruct.Q3_K_S",
                "temperature": 0.7,
                "number_of_tokens_to_generate": 3,
                "number_of_completions_to_generate": 1,
                "grammar_file_string": "accept_or_reject",
            }
            response = await client.post(
                url,
                json=payload,
                params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
            )
            response.raise_for_status()
            output_results = response.json()
            output_text = output_results[0]["generated_text"]
            end_time = time.time()
            duration_in_seconds = (end_time - start_time)
            logger.info("Total time taken for inference request content check: {} seconds".format(round(duration_in_seconds, 2)))
            if output_text == "ACCEPT":
                logger.info("Inference request is not problematic, passing on to API service now...")
                return inference_request_allowed
            if output_text == "REJECT":
                logger.error("Error! Inference request was determined to be problematic and likely to result in a rejection or ban from the API service!")
                inference_request_allowed = False
                return inference_request_allowed
            logger.warning("Warning! Inference check was supposed to result in either 'ACCEPT' or 'REJECT', but it instead returned: '{}'".format(output_text))
            return inference_request_allowed
    except Exception as e:
        logger.error("Failed to execute inference check request with Swiss Army Llama: {}".format(e))
        if port == REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT:
            logger.info("Falling back to local Swiss Army Llama.")
            return await check_if_input_text_would_get_rejected_from_api_services(input_text)
    return inference_request_allowed

async def validate_inference_api_usage_request(inference_api_usage_request: db_code.InferenceAPIUsageRequest) -> Tuple[bool, float, float]:
    try:
        validation_errors = await validate_inference_request_message_data_func(inference_api_usage_request)
        if validation_errors:
            raise ValueError(f"Invalid inference request message: {', '.join(validation_errors)}")
        requesting_pastelid = inference_api_usage_request.requesting_pastelid
        credit_pack_ticket_pastel_txid = inference_api_usage_request.credit_pack_ticket_pastel_txid
        requested_model = inference_api_usage_request.requested_model_canonical_string
        model_inference_type_string = inference_api_usage_request.model_inference_type_string
        model_parameters = base64.b64decode(inference_api_usage_request.model_parameters_json_b64).decode('utf-8')
        input_data = inference_api_usage_request.model_input_data_json_b64
        if not validate_pastel_txid_string(credit_pack_ticket_pastel_txid):
            logger.error(f"Invalid Pastel TXID: {credit_pack_ticket_pastel_txid}")
            return False, 0, 0
        _, credit_pack_purchase_request_response, _ = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_pastel_txid)
        credit_pack_purchase_request_object = await get_credit_pack_purchase_request_from_response(credit_pack_purchase_request_response)
        if credit_pack_purchase_request_object:
            list_of_authorized_pastelids_allowed_to_use_credit_pack = json.dumps(credit_pack_purchase_request_object.list_of_authorized_pastelids_allowed_to_use_credit_pack)
            if requesting_pastelid not in list_of_authorized_pastelids_allowed_to_use_credit_pack:
                logger.warning(f"Unauthorized PastelID: {requesting_pastelid}")
                return False, 0, 0
        model_menu = await get_inference_model_menu()
        requested_model_data = next((model for model in model_menu["models"] if normalize_string(model["model_name"]) == normalize_string(requested_model)), None)
        if requested_model_data is None:
            logger.warning(f"Invalid model requested: {requested_model}")
            return False, 0, 0
        if "api_based_pricing" in requested_model_data['credit_costs']:
            is_api_based_model = 1
        else:
            is_api_based_model = 0
        if model_inference_type_string not in requested_model_data["supported_inference_type_strings"]:
            logger.warning(f"Unsupported inference type '{model_inference_type_string}' for model '{requested_model}'")
            return False, 0, 0
        if not is_api_based_model:
            logger.info("Inference request is for Swiss Army Llama model, so checking if the service is available...")
            local_swiss_army_llama_responding = is_swiss_army_llama_responding(local=True)
            if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
                remote_swiss_army_llama_responding = is_swiss_army_llama_responding(local=False)
            else:
                remote_swiss_army_llama_responding = 0
            if remote_swiss_army_llama_responding and USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
                port = REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT
            elif local_swiss_army_llama_responding:
                port = SWISS_ARMY_LLAMA_PORT
            else:
                logger.error(f"Neither the local Swiss Army Llama (supposed to be running on port {SWISS_ARMY_LLAMA_PORT}) nor the remote Swiss Army Llama (supposed to be running, if enabled, on mapped port {REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT}) is responding!")
                return False, 0, 0                        
            async with httpx.AsyncClient() as client:
                params = {"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
                response = await client.get(f"http://localhost:{port}/get_list_of_available_model_names/", params=params)
                if response.status_code == 200:
                    available_models = response.json()["model_names"]
                    if requested_model not in available_models:
                        add_model_response = await client.post(f"http://localhost:{port}/add_new_model/", params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN, "model_url": requested_model_data["model_url"]})
                        if add_model_response.status_code != 200:
                            logger.warning(f"Failed to add new model to Swiss Army Llama: {requested_model}")
                            return False, 0, 0
                else:
                    logger.warning(f"Failed to retrieve available models from Swiss Army Llama API on port {port}")
                    return False, 0, 0
        model_parameters_dict = json.loads(model_parameters)
        input_data_binary = base64.b64decode(input_data)
        result = magika.identify_bytes(input_data_binary)
        detected_data_type = result.output.ct_label
        use_check_inference_requests_locally_before_sending_to_api_service = 0
        if detected_data_type == "txt":
            input_data = input_data_binary.decode("utf-8")
            if is_api_based_model and use_check_inference_requests_locally_before_sending_to_api_service:
                inference_request_allowed = await check_if_input_text_would_get_rejected_from_api_services(input_data)
                if not inference_request_allowed:
                    logger.error(f"Cannot proceed with inference request to model {requested_model} because of risk that it will be rejected and lead to banning!")
                    return False, 0, 0
        if is_base64_encoded(input_data):
            input_data = base64.b64decode(input_data)
            input_data = input_data.decode('utf-8')     
        proposed_cost_in_credits = await calculate_proposed_inference_cost_in_credits(requested_model_data, model_parameters_dict, model_inference_type_string, input_data)
        validation_results = await validate_existing_credit_pack_ticket(credit_pack_ticket_pastel_txid)
        if not validation_results["credit_pack_ticket_is_valid"]:
            logger.warning(f"Invalid credit pack ticket: {validation_results['validation_failure_reasons_list']}")
            return False, 0, 0
        else:
            logger.info(f"Credit pack ticket with txid {credit_pack_ticket_pastel_txid} passed all validation checks: {abbreviated_pretty_json_func(validation_results['validation_checks'])}")
        current_credit_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address = await determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack_ticket_pastel_txid)
        if proposed_cost_in_credits >= current_credit_balance:
            logger.warning(f"Insufficient credits for the request. Required: {proposed_cost_in_credits:,}, Available: {current_credit_balance:,}")
            return False, proposed_cost_in_credits, current_credit_balance
        else:
            logger.info(f"Credit pack ticket has sufficient credits for the request. Required: {proposed_cost_in_credits:,}, Available: {current_credit_balance:,}")
        remaining_credits_after_request = current_credit_balance - proposed_cost_in_credits
        return True, proposed_cost_in_credits, remaining_credits_after_request
    except Exception as e:
        logger.error(f"Error validating inference API usage request: {str(e)}")
        traceback.print_exc()
        raise
    
async def process_inference_api_usage_request(inference_api_usage_request: db_code.InferenceAPIUsageRequest) -> db_code.InferenceAPIUsageResponse: 
    # Validate the inference API usage request
    is_valid_request, proposed_cost_in_credits, remaining_credits_after_request = await validate_inference_api_usage_request(inference_api_usage_request) 
    inference_api_usage_request_dict = inference_api_usage_request.model_dump()
    inference_api_usage_request_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_api_usage_request_dict.items()}
    if not is_valid_request:
        logger.error("Invalid inference API usage request received!")
        raise ValueError(f"Error! Received invalid inference API usage request: {inference_api_usage_request_dict}")
    else:
        log_action_with_payload("Received", "inference API usage request", inference_api_usage_request_dict)
    # Save the inference API usage request
    saved_request = await save_inference_api_usage_request(inference_api_usage_request)
    credit_pack_ticket_pastel_txid = inference_api_usage_request.credit_pack_ticket_pastel_txid
    _, credit_pack_purchase_request_response, _ = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_pastel_txid)
    credit_usage_tracking_psl_address = credit_pack_purchase_request_response.credit_usage_tracking_psl_address
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
        remaining_credits_in_pack_after_request_processed=round(remaining_credits_after_request,1),
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
        await db_session.commit()
        await db_session.refresh(inference_response)
    return inference_response

async def check_burn_address_for_tracking_transaction(
    tracking_address: str,
    expected_amount: float,
    txid: Optional[str],
    max_block_height: int,
    max_retries: int = 10,
    initial_retry_delay: int = 25
) -> Tuple[bool, int, int]:
    global rpc_connection
    global burn_address
    try_count = 0
    retry_delay = initial_retry_delay
    total_amount_to_burn_address = 0.0
    while try_count < max_retries:
        if txid is None: # If txid is not provided, search for transactions using listsinceblock RPC method
            start_block_hash = await rpc_connection.getblockhash(0)
            listsinceblock_output = await rpc_connection.listsinceblock(start_block_hash, 1, True)
            all_transactions = listsinceblock_output["transactions"]
            all_burn_transactions = [
                tx for tx in all_transactions
                if tx.get("address") == burn_address and tx.get("category") == "receive"
            ]
            all_burn_transactions_df = pd.DataFrame.from_records(all_burn_transactions)
            all_burn_transactions_df_filtered = all_burn_transactions_df[all_burn_transactions_df['amount']==expected_amount]
            if len(all_burn_transactions_df_filtered)==1:
                txid = all_burn_transactions_df_filtered['txid'].values[0]
            else:
                latest_block_height = await get_current_pastel_block_height_func()
                min_confirmations = latest_block_height - max_block_height
                max_confirmations = latest_block_height - (max_block_height - MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET)
                all_burn_transactions_df_filtered2 = all_burn_transactions_df_filtered[all_burn_transactions_df_filtered['confirmations']<=max_confirmations]
                all_burn_transactions_df_filtered3 = all_burn_transactions_df_filtered2[all_burn_transactions_df_filtered2['confirmations']>=min_confirmations]
                if len(all_burn_transactions_df_filtered3) > 1:
                    logger.warning(f"Multiple transactions found with the same amount and confirmations, but the most recent one is {all_burn_transactions_df_filtered3['txid'].values[0]}")
                txid = all_burn_transactions_df_filtered3['txid'].values[0]
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
                            if SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK:
                                logger.info(f"Matching confirmed transaction found with {num_confirmations:,} confirmation blocks, which is acceptable because the 'SKIP_BURN_TRANSACTION_BLOCK_CONFIRMATION_CHECK' flag is set to TRUE...")
                            else:
                                logger.info(f"Matching confirmed transaction found with {num_confirmations:,} confirmation blocks, greater than or equal to the required confirmation blocks of {MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION}!")
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
        else:
            logger.warning(f"Transaction {txid} not found.")
        # If the transaction is not found or does not match the criteria, wait before retrying
        logger.info(f"WAITING {retry_delay} seconds before checking transaction status again...")
        await asyncio.sleep(retry_delay)
        try_count += 1
        retry_delay *= 1.15  # Optional: increase delay between retries
    logger.info(f"Transaction not found or did not match the criteria after {max_retries} attempts.")
    return False, False, None, None, total_amount_to_burn_address

async def process_inference_confirmation(inference_request_id: str, inference_confirmation: db_code.InferenceConfirmation) -> bool:
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
        confirmation_transaction_txid = inference_confirmation.confirmation_transaction['txid']
        credit_usage_tracking_amount_in_psl = float(inference_response.request_confirmation_message_amount_in_patoshis)/(10**5) # Divide by number of Patoshis per PSL
        matching_transaction_found, exceeding_transaction_found, transaction_block_height, num_confirmations, amount_received_at_burn_address = await check_burn_address_for_tracking_transaction(inference_response.credit_usage_tracking_psl_address, credit_usage_tracking_amount_in_psl, confirmation_transaction_txid, inference_response.max_block_height_to_include_confirmation_transaction)
        if matching_transaction_found:
            logger.info(f"Found correct inference request confirmation tracking transaction in burn address (with {num_confirmations} confirmation blocks so far)! TXID: {confirmation_transaction_txid}; Tracking Amount in PSL: {credit_usage_tracking_amount_in_psl};")
            computed_current_credit_pack_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address = await determine_current_credit_pack_balance_based_on_tracking_transactions(inference_request.credit_pack_ticket_pastel_txid)
            logger.info(f"Computed current credit pack balance: {computed_current_credit_pack_balance:,.1f} based on {number_of_confirmation_transactions_from_tracking_address_to_burn_address:,} tracking transactions from tracking address to burn address.")       
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
        traceback.print_exc()
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
        traceback.print_exc()
        raise

def get_claude3_model_name(model_name: str) -> str:
    model_mapping = {
        "claude3-haiku": "claude-3-haiku-20240307",
        "claude3-opus": "claude-3-opus-20240229",
        "claude3.5-sonnet": "claude-3-5-sonnet-20240620"
    }
    return model_mapping.get(model_name, "")

async def submit_inference_request_to_stability_api(inference_request):
    # Integrate with the Stability API to perform the image generation task
    if inference_request.model_inference_type_string == "text_to_image":
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        if "stability-core" in inference_request.requested_model_canonical_string:
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS * 3)) as client:
                data = {
                    "prompt": prompt,
                    "aspect_ratio": model_parameters.get("aspect_ratio", "1:1"),
                    "output_format": model_parameters.get("output_format", "jpeg")
                }
                if model_parameters.get("negative_prompt"):
                    data["negative_prompt"] = model_parameters["negative_prompt"]
                if model_parameters.get("seed") is not None:
                    data["seed"] = model_parameters["seed"]
                if model_parameters.get("style_preset"):
                    data["style_preset"] = model_parameters["style_preset"]
                response = await client.post(
                    "https://api.stability.ai/v2beta/stable-image/generate/core",
                    headers={
                        "authorization": f"Bearer {STABILITY_API_KEY}",
                        "accept": "application/json"
                    },
                    files={"none": ''},
                    data=data,
                )
                if response.status_code == 200:
                    response_json = response.content
                    output_results = json.loads(response_json)
                    output_results_file_type_strings = {
                        "output_text": "base64_image",
                        "output_files": ["NA"]
                    }
                    return output_results, output_results_file_type_strings
                else:
                    logger.error(f"Error generating image from Stability API: {response.text}")
                    return None, None
        else:
            engine_id = inference_request.requested_model_canonical_string
            api_host = "https://api.stability.ai"
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS * 3)) as client:
                json_data = {
                    "text_prompts": [{"text": prompt}],
                    "cfg_scale": model_parameters.get("cfg_scale", 7),
                    "height": model_parameters.get("height", 512),
                    "width": model_parameters.get("width", 512),
                    "samples": model_parameters.get("num_samples", 1),
                    "steps": model_parameters.get("steps", 50),
                }
                if model_parameters.get("style_preset"):
                    json_data["style_preset"] = model_parameters["style_preset"]
                response = await client.post(
                    f"{api_host}/v1/generation/{engine_id}/text-to-image",
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                        "Authorization": f"Bearer {STABILITY_API_KEY}"
                    },
                    json=json_data,
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
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        input_image = base64.b64decode(inference_request.model_input_data_json_b64)
        prompt = model_parameters.get("prompt", "")
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS * 3)) as client:
            data = {
                "prompt": prompt,
                "output_format": model_parameters.get("output_format", "png"),
                "creativity": model_parameters.get("creativity", 0.3),
            }
            if model_parameters.get("seed") is not None:
                data["seed"] = model_parameters["seed"]
            if model_parameters.get("negative_prompt"):
                data["negative_prompt"] = model_parameters["negative_prompt"]
            response = await client.post(
                "https://api.stability.ai/v2beta/stable-image/upscale/creative",
                headers={
                    "authorization": f"Bearer {STABILITY_API_KEY}",
                    "accept": "application/json"
                },
                files={"image": input_image},
                data=data,
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
    logger.info("Now accessing OpenAI API...")
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = int(model_parameters.get("number_of_completions_to_generate", 1))
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        openai_text_completion_response_timeout_seconds = 60
        async with httpx.AsyncClient(timeout=openai_text_completion_response_timeout_seconds) as client:
            for i in range(num_completions):
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": inference_request.requested_model_canonical_string.replace("openai-", ""),
                        "messages": [{"role": "user", "content": input_prompt}],
                        "max_tokens": int(model_parameters.get("number_of_tokens_to_generate", 1000)),
                        "temperature": float(model_parameters.get("temperature", 0.7)),
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
        async with httpx.AsyncClient() as client:
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
    elif inference_request.model_inference_type_string == "ask_question_about_an_image":
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        num_completions = int(model_parameters.get("number_of_completions_to_generate", 1))
        input_data = json.loads(base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8"))
        image_data_binary = base64.b64decode(input_data["image"])
        question = input_data["question"]
        base64_image = base64.b64encode(image_data_binary).decode('utf-8')
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0        
        openai_vision_timeout_seconds = 90
        async with httpx.AsyncClient(timeout=openai_vision_timeout_seconds) as client:
            for i in range(num_completions):
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {OPENAI_API_KEY}"
                    },
                    json={
                        "model": inference_request.requested_model_canonical_string.replace("openai-", "").replace("-vision", ""),
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": question
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{base64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        "max_tokens": int(json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8")).get("number_of_tokens_to_generate", 300))
                    }
                )
                if response.status_code == 200:
                    response_json = response.json()
                    output_results.append(response_json["choices"][0]["message"]["content"])
                    total_input_tokens += response_json["usage"]["prompt_tokens"]
                    total_output_tokens += response_json["usage"]["completion_tokens"]                    
                else:
                    logger.error(f"Error generating response from OpenAI API: {response.text}")
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
    else:
        logger.warning(f"Unsupported inference type for OpenAI model: {inference_request.model_inference_type_string}")
        return None, None
    
async def submit_inference_request_to_openrouter(inference_request):
    logger.info("Now accessing OpenRouter...")
    if inference_request.model_inference_type_string == "text_completion":
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))       
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
                    "max_tokens": int(model_parameters.get("number_of_tokens_to_generate", 1000)),
                    "temperature": float(model_parameters.get("temperature", 0.7)),
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
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))        
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = int(model_parameters.get("number_of_completions_to_generate", 1))
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            messages = [ChatMessage(role="user", content=input_prompt)]
            async_response = client.chat_stream(
                model=inference_request.requested_model_canonical_string.replace("mistralapi-",""),
                messages=messages,
                max_tokens=int(model_parameters.get("number_of_tokens_to_generate", 1000)),
                temperature=float(model_parameters.get("temperature", 0.7)),
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
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = int(model_parameters.get("number_of_completions_to_generate", 1))
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": input_prompt}],
                model=inference_request.requested_model_canonical_string.replace("groq-",""),
                max_tokens=int(model_parameters.get("number_of_tokens_to_generate", 1000)),
                temperature=float(model_parameters.get("temperature", 0.7)),
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
        model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
        input_prompt = base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8")
        num_completions = int(model_parameters.get("number_of_completions_to_generate", 1))
        output_results = []
        total_input_tokens = 0
        total_output_tokens = 0
        for i in range(num_completions):
            async with client.messages.stream(
                model=claude3_model_id_string,
                max_tokens=int(model_parameters.get("number_of_tokens_to_generate", 1000)),
                temperature=float(model_parameters.get("temperature", 0.7)),
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

# Swiss Army Llama related functions:

def determine_swiss_army_llama_port():
    local_swiss_army_llama_responding = is_swiss_army_llama_responding(local=True)
    if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        remote_swiss_army_llama_responding = is_swiss_army_llama_responding(local=False)
    else:
        remote_swiss_army_llama_responding = 0
    
    if remote_swiss_army_llama_responding and USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        return REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT
    elif local_swiss_army_llama_responding:
        return SWISS_ARMY_LLAMA_PORT
    return None

async def handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handler_function):
    logger.error("Failed to execute inference request: {}".format(e))
    if port == REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT and not is_fallback:
        logger.info("Falling back to local Swiss Army Llama.")
        return await handler_function(client, inference_request, model_parameters, SWISS_ARMY_LLAMA_PORT, True)
    else:
        return None, None
    
async def handle_swiss_army_llama_text_completion(client, inference_request, model_parameters, port, is_fallback):
    payload = {
        "input_prompt": base64.b64decode(inference_request.model_input_data_json_b64).decode("utf-8"),
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "temperature": model_parameters.get("temperature", 0.7),
        "number_of_tokens_to_generate": model_parameters.get("number_of_tokens_to_generate", 1000),
        "number_of_completions_to_generate": model_parameters.get("number_of_completions_to_generate", 1),
        "grammar_file_string": model_parameters.get("grammar_file_string", ""),
    }
    try:
        response = await client.post(
            f"http://localhost:{port}/get_text_completions_from_input_prompt/",
            json=payload,
            params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        )
        response.raise_for_status()
        output_results = response.json()
        output_text = output_results[0]["generated_text"]
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_text, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_text_completion)

async def handle_swiss_army_llama_image_question(client, inference_request, model_parameters, port, is_fallback):
    input_data = json.loads(base64.b64decode(inference_request.model_input_data_json_b64).decode())
    image_data_binary = base64.b64decode(input_data["image"])
    question = input_data["question"]
    payload = {
        "question": question,
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "temperature": model_parameters.get("temperature", 0.7),
        "number_of_tokens_to_generate": model_parameters.get("number_of_tokens_to_generate", 256),
        "number_of_completions_to_generate": model_parameters.get("number_of_completions_to_generate", 1)
    }
    files = {"image": ("image.png", image_data_binary, "image/png")}
    try:
        response = await client.post(
            f"http://localhost:{port}/ask_question_about_image/",
            data=payload,
            files=files,
            params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        )
        response.raise_for_status()
        output_results = response.json()
        output_text = output_results[0]["generated_text"]
        result = magika.identify_bytes(output_text.encode("utf-8"))
        detected_data_type = result.output.ct_label
        output_results_file_type_strings = {
            "output_text": detected_data_type,
            "output_files": ["NA"]
        }
        return output_text, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_image_question)

async def handle_swiss_army_llama_embedding_document(client, inference_request, model_parameters, port, is_fallback):
    input_data = inference_request.model_input_data_json_b64
    if is_base64_encoded(input_data):
        input_data = base64.b64decode(input_data)
        input_data = input_data.decode('utf-8')
    try:
        input_data_dict = json.loads(input_data)
        document_file_data = input_data_dict['document']
        if is_base64_encoded(document_file_data):
            document_file_data = base64.b64decode(document_file_data)
    except Exception as e:
        logger.error(f"Error decoding audio data: {str(e)}")
        traceback.print_exc()
        raise("Error decoding audio data")
    try:
        file_metadata = await upload_and_get_file_metadata(document_file_data, file_prefix="document")
        file_url = file_metadata["file_url"]
        file_hash = file_metadata["file_hash"]
        file_size = file_metadata["file_size"]
        query_text = model_parameters.get("query_text", None)
    except Exception as e:
        logger.error(f"Error parsing document data from input: {str(e)}")
        traceback.print_exc()
        raise
    params = {
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "embedding_pooling_method": model_parameters.get("embedding_pooling_method", "svd"),
        "corpus_identifier_string": model_parameters.get("corpus_identifier_string", ""),
        "json_format": model_parameters.get("json_format", "records"),
        "send_back_json_or_zip_file": model_parameters.get("send_back_json_or_zip_file", "zip"),
        "query_text": query_text
    }
    files = {
        'url': (None, file_url),
        'hash': (None, file_hash),
        'size': (None, str(file_size))
    }
    try:
        response = await client.post(
            f"http://localhost:{port}/get_all_embedding_vectors_for_document/",
            params=params,
            files=files,
            headers={"accept": "application/json"}
        )
        response.raise_for_status()
        if model_parameters.get("send_back_json_or_zip_file", "zip") == "json":
            output_results = response.json()
            output_results_file_type_strings = {
                "output_text": "embedding_document",
                "output_files": ["NA"]
            }            
        else:
            zip_file_content = response.read()
            output_results = base64.b64encode(zip_file_content).decode('utf-8')
            output_results_file_type_strings = {
                "output_text": "NA",
                "output_files": ["zip"]
            }
        return output_results, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_embedding_document)
    
async def handle_swiss_army_llama_embedding_audio(client, inference_request, model_parameters, port, is_fallback):
    input_data = inference_request.model_input_data_json_b64
    if is_base64_encoded(input_data):
        input_data = base64.b64decode(input_data)
        input_data = input_data.decode('utf-8')
    try:
        input_data_dict = json.loads(input_data)
        audio_file_data = input_data_dict['audio']
        if is_base64_encoded(audio_file_data):
            audio_file_data = base64.b64decode(audio_file_data) 
    except Exception as e:
        logger.error(f"Error decoding audio data: {str(e)}")
        traceback.print_exc()
        raise("Error decoding audio data")
    try:
        file_metadata = await upload_and_get_file_metadata(audio_file_data, file_prefix="audio")
        file_url = file_metadata["file_url"]
        file_hash = file_metadata["file_hash"]
        file_size = file_metadata["file_size"]
        query_text = model_parameters.get("query_text", "")
        corpus_identifier_string = model_parameters.get("corpus_identifier_string", "")
    except Exception as e:
        logger.error(f"Error parsing audio data from input: {str(e)}")
        traceback.print_exc()
        raise
    params = {
        "compute_embeddings_for_resulting_transcript_document": model_parameters.get("compute_embeddings_for_resulting_transcript_document", True),
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "embedding_pooling_method": model_parameters.get("embedding_pooling_method", "svd"),
        "corpus_identifier_string": corpus_identifier_string
    }
    files = {
        'url': (None, file_url),
        'hash': (None, file_hash),
        'size': (None, str(file_size))
    }
    try:
        response = await client.post(
            f"http://localhost:{port}/compute_transcript_with_whisper_from_audio/",
            params=params,
            files=files,
            headers={"accept": "application/json"}
        )
        response.raise_for_status()
        output_results = response.json()
        output_results_file_type_strings = {
            "output_text": "embedding_audio",
            "output_files": ["NA"]
        }
        if query_text:
            search_payload = {
                "query_text": query_text,
                "number_of_most_similar_strings_to_return": model_parameters.get("number_of_most_similar_strings_to_return", 10),
                "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
                "embedding_pooling_method": model_parameters.get("embedding_pooling_method", "svd"),
                "corpus_identifier_string": corpus_identifier_string
            }
            search_response = await client.post(
                f"http://localhost:{port}/search_stored_embeddings_with_query_string_for_semantic_similarity/",
                json=search_payload,
                headers={"accept": "application/json", "Content-Type": "application/json"}
            )
            search_response.raise_for_status()
            search_results = search_response.json()
            output_results["search_results"] = search_results
        return output_results, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_embedding_audio)

async def handle_swiss_army_llama_semantic_search(client, inference_request, model_parameters, port, is_fallback):
    payload = {
        "query_text": model_parameters.get("query_text", ""),
        "number_of_most_similar_strings_to_return": model_parameters.get("number_of_most_similar_strings_to_return", 10),
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "embedding_pooling_method": model_parameters.get("embedding_pooling_method", "svd_first_four"),
        "corpus_identifier_string": model_parameters.get("corpus_identifier_string", "")
    }
    try:
        response = await client.post(
            f"http://localhost:{port}/search_stored_embeddings_with_query_string_for_semantic_similarity/",
            json=payload,
            params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        )
        response.raise_for_status()
        output_results = response.json()
        output_results_file_type_strings = {
            "output_text": "semantic_search",
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_semantic_search)

async def handle_swiss_army_llama_advanced_semantic_search(client, inference_request, model_parameters, port, is_fallback):
    payload = {
        "query_text": model_parameters.get("query_text", ""),
        "llm_model_name": inference_request.requested_model_canonical_string.replace("swiss_army_llama-", ""),
        "embedding_pooling_method": model_parameters.get("embedding_pooling_method", "svd"),
        "corpus_identifier_string": model_parameters.get("corpus_identifier_string", "string"),
        "similarity_filter_percentage": model_parameters.get("similarity_filter_percentage", 0.01),
        "number_of_most_similar_strings_to_return": model_parameters.get("number_of_most_similar_strings_to_return", 0),
        "result_sorting_metric": model_parameters.get("result_sorting_metric", "hoeffding_d")
    }
    try:
        response = await client.post(
            f"http://localhost:{port}/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/",
            json=payload,
            params={"token": SWISS_ARMY_LLAMA_SECURITY_TOKEN}
        )
        response.raise_for_status()
        output_results = response.json()
        output_results_file_type_strings = {
            "output_text": "advanced_semantic_search",
            "output_files": ["NA"]
        }
        return output_results, output_results_file_type_strings
    except Exception as e:
        return await handle_swiss_army_llama_exception(e, client, inference_request, model_parameters, port, is_fallback, handle_swiss_army_llama_advanced_semantic_search)

async def submit_inference_request_to_swiss_army_llama(inference_request, is_fallback=False):
    logger.info("Now calling Swiss Army Llama with model {}".format(inference_request.requested_model_canonical_string))
    model_parameters = json.loads(base64.b64decode(inference_request.model_parameters_json_b64).decode("utf-8"))
    port = determine_swiss_army_llama_port()
    if not port:
        logger.error(f"Neither the local (port {SWISS_ARMY_LLAMA_PORT}) nor the remote (port {REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT}) Swiss Army Llama is responding!")
        return None, None
    async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS * 12)) as client:
        if inference_request.model_inference_type_string == "text_completion":
            return await handle_swiss_army_llama_text_completion(client, inference_request, model_parameters, port, is_fallback)
        elif inference_request.model_inference_type_string == "embedding_document":
            return await handle_swiss_army_llama_embedding_document(client, inference_request, model_parameters, port, is_fallback)
        elif inference_request.model_inference_type_string == "embedding_audio":
            return await handle_swiss_army_llama_embedding_audio(client, inference_request, model_parameters, port, is_fallback)
        elif inference_request.model_inference_type_string == "ask_question_about_an_image":
            return await handle_swiss_army_llama_image_question(client, inference_request, model_parameters, port, is_fallback)
        elif inference_request.model_inference_type_string == "semantic_search":
            return await handle_swiss_army_llama_semantic_search(client, inference_request, model_parameters, port, is_fallback)
        elif inference_request.model_inference_type_string == "advanced_semantic_search":
            return await handle_swiss_army_llama_advanced_semantic_search(client, inference_request, model_parameters, port, is_fallback)
        else:
            logger.warning("Unsupported inference type: {}".format(inference_request.model_inference_type_string))
            return None, None

async def execute_inference_request(inference_request_id: str) -> None:
    try:
        # Retrieve the inference API usage request from the database
        async with db_code.Session() as db:
            query = await db.exec(
                select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
            )
            inference_request = query.one_or_none()
        if inference_request is None:
            logger.warning(f"Invalid inference request ID: {inference_request_id}")
            return
        # Retrieve the inference API usage request response from the database
        async with db_code.Session() as db:
            query = await db.exec(
                select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_request_id == inference_request_id)
            )
            inference_response = query.one_or_none()
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
        elif inference_request.requested_model_canonical_string.startswith("swiss_army_llama-"):
            output_results, output_results_file_type_strings = await submit_inference_request_to_swiss_army_llama(inference_request)
        else:
            error_message = f"Unsupported provider or model selected for {inference_request.requested_model_canonical_string}: {inference_request.model_inference_type_string}"
            logger.error(error_message)
            raise ValueError(error_message)
        if output_results is not None and output_results_file_type_strings is not None:
            # Save the inference output results to the database
            await save_inference_output_results(inference_request_id, inference_response.inference_response_id, output_results, output_results_file_type_strings)
    except Exception as e:
        logger.error(f"Error executing inference request: {str(e)}")
        traceback.print_exc()
        raise

async def check_status_of_inference_request_results(inference_response_id: str) -> bool:
    try:
        async with db_code.Session() as db_session:
            # Retrieve the inference output result
            result = await db_session.exec(
                select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_response_id)
            )
            inference_output_result = result.one_or_none()
            if inference_output_result is None:
                return False
            else:
                return True
    except Exception as e:
        logger.error(f"Error checking status of inference request results: {str(e)}")
        raise

async def get_inference_output_results_and_verify_authorization(inference_response_id: str, requesting_pastelid: str) -> db_code.InferenceAPIOutputResult:
    async with db_code.Session() as db_session:
        # Retrieve the inference output result
        query = await db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_response_id)
        )
        inference_output_result = query.one_or_none()
        if inference_output_result is None:
            raise ValueError("Inference output results not found")
        # Retrieve the inference request to verify requesting PastelID
        query = await db_session.exec(
            select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_output_result.inference_request_id)
        )
        inference_request = query.one_or_none()
        if inference_request is None or inference_request.requesting_pastelid != requesting_pastelid:
            raise ValueError("Unauthorized access to inference output results")
        return inference_output_result

async def process_transactions_in_chunks_old(transactions, chunk_size):
    decoded_tx_data_list = []
    total_chunk_count = math.ceil(len(transactions) / chunk_size)
    chunk_count = 0
    if transactions:
        for i in range(0, len(transactions), chunk_size):
            chunk = transactions[i:i + chunk_size]
            chunk_count += 1
            if total_chunk_count > 1:
                logger.info(f"Processing burn transactions chunk {chunk_count} of {total_chunk_count} (total transaction count of {len(transactions):,})...")
            # Create tasks for each transaction that needs processing
            tasks = [create_transaction_task(transaction) for transaction in chunk if transaction['category'] == 'receive' and transaction['amount'] > 0]
            transactions_to_insert = await asyncio.gather(*tasks)
            # Filter out None values and prepare to bulk insert
            transactions_to_insert = [txn for txn in transactions_to_insert if txn]
            if transactions_to_insert:
                async with db_code.Session() as db:
                    db.add_all(transactions_to_insert)
                    await db.commit()
                    logger.info(f"Added {len(transactions_to_insert):,} new burn transactions to the database successfully.")
            await db_code.consolidate_wal_data()  # Consolidate WAL after processing each chunk
            decoded_tx_data_list.extend(chunk)  # Optional: Keep track of processed chunks
        return decoded_tx_data_list
    else:
        return []

async def process_transactions_in_chunks(transactions, chunk_size, ignore_unconfirmed_transactions=0):
    decoded_tx_data_list = []
    total_chunk_count = math.ceil(len(transactions) / chunk_size)
    chunk_count = 0
    all_seen_txids = set()  # Track all txids processed in this session
    for i in range(0, len(transactions), chunk_size):
        chunk = transactions[i:i + chunk_size]
        chunk_count += 1
        if total_chunk_count > 1:       
            logger.info(f"Processing burn transactions chunk {chunk_count} of {total_chunk_count} (total transaction count of {len(transactions):,})...")
        if ignore_unconfirmed_transactions:
            tasks = [create_transaction_task(transaction) for transaction in chunk if transaction['amount'] > 0]
        else:
            tasks = [create_transaction_task_old(transaction) for transaction in chunk if transaction['amount'] > 0]
        transactions_to_insert = await asyncio.gather(*tasks)
        transactions_to_insert = [txn for txn in transactions_to_insert if txn]
        # Ensure no duplicates are inserted within the same batch
        filtered_transactions = []
        for txn in transactions_to_insert:
            if txn.txid not in all_seen_txids:
                filtered_transactions.append(txn)
                all_seen_txids.add(txn.txid)
        if filtered_transactions:
            async with db_code.Session() as db:
                try:
                    db.add_all(filtered_transactions)
                    await db.commit()
                    logger.info(f"Added {len(filtered_transactions):,} new burn transactions to the database successfully.")
                except IntegrityError:
                    await db.rollback()  # Rollback the session in case of an IntegrityError
                    for txn in filtered_transactions:
                        try:
                            db.add(txn)
                            await db.commit()  # Try to commit each transaction individually
                        except IntegrityError:
                            await db.rollback()  # Skip the transaction if it still fails
        await db_code.consolidate_wal_data()  # Consolidate WAL after processing each chunk
        decoded_tx_data_list.extend(transactions_to_insert)  # Optional: Keep track of processed chunks
    return decoded_tx_data_list
    
async def create_transaction_task(transaction):
    txid = transaction['txid']
    async with db_code.Session() as db:
        # Check if transaction already exists; if it does, return early without doing anything
        stmt = select(db_code.BurnAddressTransaction).where(db_code.BurnAddressTransaction.txid == txid)
        stmt_output = await db.execute(stmt)
        if stmt_output.scalars().first():
            return None  # Skip if transaction already exists in the database
        return db_code.BurnAddressTransaction(
            txid=txid,
            block_height=transaction['blockindex'],
            burn_address=burn_address,
            tracking_address=transaction['from_address'],
            amount=transaction['amount'],
            pending=transaction['blockindex'] is None
        )

async def create_transaction_task_old(transaction):
    txid = transaction['txid']
    async with db_code.Session() as db:
        # Check if transaction already exists
        stmt = select(db_code.BurnAddressTransaction).where(db_code.BurnAddressTransaction.txid == txid)
        stmt_output = await db.execute(stmt)
        if stmt_output.scalars().first():
            return None  # Skip if transaction already exists in the database
        raw_tx = await rpc_connection.getrawtransaction(txid, 1)
        block_height = raw_tx.get('height', None)
        input_addresses = await gather_input_addresses(raw_tx['vin'])
        if len(input_addresses) == 1:
            tracking_address = next(iter(input_addresses))
            amount = sum(vout['value'] for vout in raw_tx['vout'] if burn_address in vout['scriptPubKey'].get('addresses', []))
            return db_code.BurnAddressTransaction(
                txid=txid,
                block_height=block_height,
                burn_address=burn_address,
                tracking_address=tracking_address,
                amount=amount,
                pending=block_height is None
            )
        return None
    
async def gather_input_addresses(vins):
    input_addresses = set()
    tasks = [fetch_input_address(vin) for vin in vins if 'txid' in vin]
    vin_results = await asyncio.gather(*tasks)
    for addresses in vin_results:
        input_addresses.update(addresses)
    return input_addresses

async def fetch_input_address(vin):
    prev_tx_details = await rpc_connection.getrawtransaction(vin['txid'], 1)
    prev_output = prev_tx_details['vout'][vin['vout']]
    return prev_output['scriptPubKey'].get('addresses', []) if 'scriptPubKey' in prev_output and 'addresses' in prev_output['scriptPubKey'] else []

async def full_rescan_burn_transactions_old():
    async with db_code.Session() as db: # Check if there are any records in BurnAddressTransaction or BlockHash tables
        burn_address_query_results = await db.execute(select(db_code.BurnAddressTransaction).limit(1))
        burn_tx_exists = burn_address_query_results.first()
        block_hash_query_results = await db.execute(select(db_code.BlockHash).limit(1))
        block_hash_exists = block_hash_query_results.first()
    if not burn_tx_exists:
        logger.info("No burn transaction records found in database, proceeding with full rescan...")
        current_block_height = await get_current_pastel_block_height_func()
        first_block_hash = await rpc_connection.getblockhash(0)
        listsinceblock_output = await rpc_connection.listsinceblock(first_block_hash, 1, True)
        all_transactions = listsinceblock_output["transactions"]        
        burn_transactions = [tx for tx in all_transactions if tx.get("address") == burn_address]
        chunk_size = 100  # Adjust the chunk size as needed
        decoded_tx_data_list = await process_transactions_in_chunks(burn_transactions, chunk_size)
        logger.info(f"Decoded {len(decoded_tx_data_list):,} new burn transactions in total!")
    if not block_hash_exists:
        logger.info("No block hash records found in database, proceeding with full rescan...")
        current_block_height = await get_current_pastel_block_height_func()
        logger.info(f"Now getting block hashes for {current_block_height:,} blocks...")
        await fetch_and_insert_block_hashes(0, current_block_height, 5000)
        logger.info("Block hashes updated successfully.")
    else:
        logger.info("Existing records found. Skipping full rescan.")

async def full_rescan_burn_transactions():
    global burn_address
    async with db_code.Session() as db:  # Check if there are any records in BurnAddressTransaction or BlockHash tables
        burn_address_query_results = await db.execute(select(db_code.BurnAddressTransaction).limit(1))
        burn_tx_exists = burn_address_query_results.first()
        block_hash_query_results = await db.execute(select(db_code.BlockHash).limit(1))
        block_hash_exists = block_hash_query_results.first()
    if not burn_tx_exists:
        logger.info("No burn transaction records found in database, proceeding with full rescan...")
        logger.info("Please wait, retrieving ALL burn transactions from ANY address starting with the genesis block (may take a while and cause high CPU usage...)")
        burn_transactions = await rpc_connection.scanburntransactions("*")
        chunk_size = 1000  # Adjust the chunk size as needed
        ignore_unconfirmed_transactions = 1
        if burn_transactions:
            decoded_tx_data_list = await process_transactions_in_chunks(burn_transactions, chunk_size, ignore_unconfirmed_transactions)
            logger.info(f"Decoded {len(decoded_tx_data_list):,} new burn transactions in total!")
    if not block_hash_exists:
        logger.info("No block hash records found in database, proceeding with full rescan...")
        current_block_height = await get_current_pastel_block_height_func()
        logger.info(f"Now getting block hashes for {current_block_height:,} blocks...")
        chunk_size = 1000
        await fetch_and_insert_block_hashes(0, current_block_height, chunk_size)
        logger.info("Block hashes updated successfully.")
    else:
        logger.info("Existing records found. Skipping full rescan.")

async def fetch_all_mnid_tickets_details():
    mnid_response = await rpc_connection.tickets('list', 'id', 'mn')
    if mnid_response is None or len(mnid_response) == 0:
        return []
    tickets_data = {ticket['txid']: ticket for ticket in mnid_response}
    async with db_code.Session() as session:
        async with session.begin():
            result = await session.execute(select(db_code.MNIDTicketDetails.txid).where(db_code.MNIDTicketDetails.txid.in_(tickets_data.keys())))
            existing_txids = result.scalars().all()  # Correct method to fetch scalar results directly
            existing_txids_set = set(existing_txids)
            new_tickets_to_insert = []
            for txid, ticket in tickets_data.items():
                if txid not in existing_txids_set:
                    new_ticket = db_code.MNIDTicketDetails(
                        txid=txid,
                        pastel_id=ticket['ticket']['pastelID'],
                        address=ticket['ticket']['address'],
                        pq_key=ticket['ticket']['pq_key'],
                        outpoint=ticket['ticket']['outpoint'],
                        block_height=ticket['height'],
                        timestamp=datetime.utcfromtimestamp(int(ticket['ticket']['timeStamp']))
                    )
                    new_tickets_to_insert.append(new_ticket)
            if new_tickets_to_insert:
                try:
                    session.add_all(new_tickets_to_insert)
                    await session.commit()
                except IntegrityError as e:
                    await session.rollback()
                    logger.error(f"Error inserting new tickets due to a unique constraint failure: {e}")
    return new_tickets_to_insert
                    
async def fetch_active_supernodes_count_and_details(block_height: int):
    async with db_code.Session() as session:
        async with session.begin():  # Fetch all mnid tickets created up to the specified block height
            result = await session.execute(
                select(db_code.MNIDTicketDetails)
                .where(db_code.MNIDTicketDetails.block_height <= block_height)
            )
            mnid_tickets = result.scalars().all()
            active_supernodes = []
            for ticket in mnid_tickets:
                try:
                    # Check if outpoint is valid
                    if ticket.outpoint and isinstance(ticket.outpoint, str) and '-' in ticket.outpoint:
                        txid, vout_str = ticket.outpoint.split('-')
                        # Check if vout is a valid integer
                        if not vout_str.isdigit():
                            continue
                        vout = int(vout_str)
                        tx_info = await rpc_connection.getrawtransaction(txid, 1)
                        # Ensure tx_info is a dictionary and contains 'vout'
                        if tx_info and isinstance(tx_info, dict) and 'vout' in tx_info:
                            # Ensure vout is within the valid range of tx_info['vout']
                            if vout >= 0 and vout < len(tx_info['vout']):
                                vout_data = tx_info['vout'][vout]
                                # Ensure vout_data is a dictionary and contains necessary keys
                                if vout_data and isinstance(vout_data, dict) and 'n' in vout_data and 'value' in vout_data:
                                    # Check if the outpoint is still unspent and meets collateral requirements
                                    if vout_data['n'] == vout and vout_data['value'] >= masternode_collateral_amount:
                                        # Ensure timestamp is valid
                                        if isinstance(ticket.timestamp, (datetime, int)):
                                            timestamp = datetime.utcfromtimestamp(ticket.timestamp.timestamp() if isinstance(ticket.timestamp, datetime) else int(ticket.timestamp))
                                            supernode_details = {
                                                "txid": ticket.txid,
                                                "pastel_id": ticket.pastel_id,
                                                "address": ticket.address,
                                                "pq_key": ticket.pq_key,
                                                "outpoint": ticket.outpoint,
                                                "block_height": ticket.block_height,
                                                "timestamp": timestamp
                                            }
                                            active_supernodes.append(supernode_details)
                except (KeyError, ValueError, TypeError) as e:  # noqa: F841
                    pass
    active_supernodes_count = len(active_supernodes)
    return active_supernodes_count, active_supernodes

async def detect_chain_reorg_and_rescan():
    while True:
        async with db_code.Session() as db:
            query = await db.exec(select(db_code.BlockHash).order_by(db_code.BlockHash.block_height.desc()).limit(1))
            latest_stored_block_hash = query.one_or_none()
        if latest_stored_block_hash:
            latest_stored_block_height = latest_stored_block_hash.block_height
            latest_stored_block_hash = latest_stored_block_hash.block_hash
            latest_blockchain_block_hash = await rpc_connection.getblockhash(latest_stored_block_height)
            if latest_blockchain_block_hash != latest_stored_block_hash:
                logger.warning("Chain reorg detected! Triggering a full rescan of burn transactions.")
                async with db_code.Session() as db:
                    await db.exec(delete(db_code.BurnAddressTransaction))
                    await db.exec(delete(db_code.BlockHash))
                    await db.commit()
                await full_rescan_burn_transactions()
        await asyncio.sleep(6000)  # Check for chain reorg every 6000 seconds
                
async def fetch_and_insert_block_hashes_old(start_height, end_height, batch_size=300):
    current_height = max(start_height - 1, 0)
    total_inserted = 0
    batch_counter = 0
    while current_height <= end_height:
        tasks = [asyncio.create_task(rpc_connection.getblock(h, 1)) for h in range(current_height, min(current_height + batch_size, end_height + 1), 3)]
        results = await asyncio.gather(*tasks)
        blocks_to_insert = []
        # Collect potential blocks to insert
        for result in results:
            if result:
                if 'previousblockhash' in result and result['height'] > start_height:
                    blocks_to_insert.append((result['height'] - 1, result['previousblockhash']))
                blocks_to_insert.append((result['height'], result['hash']))
                if 'nextblockhash' in result and result['height'] < end_height:
                    blocks_to_insert.append((result['height'] + 1, result['nextblockhash']))
        # Filter out duplicate heights from the batch itself
        unique_blocks = {block[0]: block for block in blocks_to_insert}.values()
        # Check database for existing heights from the refined list
        unique_heights = {height for height, _ in unique_blocks}
        async with db_code.Session() as db:
            existing_query = await db.execute(select(db_code.BlockHash.block_height).where(db_code.BlockHash.block_height.in_(unique_heights)))
            existing_heights = set(existing_query.scalars().all())
        # Filter out existing block heights
        final_blocks_to_insert = [block for block in unique_blocks if block[0] not in existing_heights]
        # Bulk insert new blocks
        if final_blocks_to_insert:
            await bulk_insert_block_hashes(final_blocks_to_insert)
            inserted_count = len(final_blocks_to_insert)
            total_inserted += inserted_count
            batch_counter += inserted_count
        # Log progress for every 500 inserts or as needed
        if batch_counter >= 500:
            blocks_left = end_height - current_height + 1
            logger.info(f"Fetched and inserted {batch_counter:,} block hashes; {total_inserted:,} total block hashes in total, {blocks_left:,} blocks left to process...")
            batch_counter = 0
        # Move to the next range, skipping ahead to optimize RPC calls
        current_height += batch_size * 3

async def fetch_and_insert_block_hashes(start_height, end_height, batch_size=300):
    current_height = max(start_height - 1, 0)
    total_inserted = 0
    batch_counter = 0
    while current_height <= end_height:
        # Calculate the number of hashes to fetch in this batch
        num_hashes = min((end_height + 1) - current_height, batch_size)
        result = await rpc_connection.getblockhash(current_height, num_hashes)
        blocks_to_insert = []
        previous_height = None
        previous_hash = None
        # Loop through the results to construct the potential block insert tuples
        for block in result:
            block_height = block['height']
            block_hash = block['hash']
            # Insert the previous block hash if available and in the correct range
            if previous_height is not None and previous_height >= start_height:
                blocks_to_insert.append((previous_height, previous_hash))
            blocks_to_insert.append((block_height, block_hash))
            previous_height = block_height + 1
            previous_hash = block_hash
        # Filter out duplicate heights from the batch itself
        unique_blocks = {block[0]: block for block in blocks_to_insert}.values()
        # Check database for existing heights from the refined list
        unique_heights = {height for height, _ in unique_blocks}
        async with db_code.Session() as db:
            existing_query = await db.execute(select(db_code.BlockHash.block_height).where(db_code.BlockHash.block_height.in_(unique_heights)))
            existing_heights = set(existing_query.scalars().all())
        # Filter out existing block heights
        final_blocks_to_insert = [block for block in unique_blocks if block[0] not in existing_heights]
        # Bulk insert new blocks
        if final_blocks_to_insert:
            await bulk_insert_block_hashes(final_blocks_to_insert)
            inserted_count = len(final_blocks_to_insert)
            total_inserted += inserted_count
            batch_counter += inserted_count
        # Log progress for every 500 inserts or as needed
        if batch_counter >= 500:
            blocks_left = end_height - current_height + 1
            logger.info(f"Fetched and inserted {batch_counter:,} block hashes; {total_inserted:,} total block hashes in total, {blocks_left:,} blocks left to process...")
            batch_counter = 0
        # Move to the next range, accounting for the batch size
        current_height += num_hashes

async def bulk_insert_block_hashes(block_hashes):
    async with db_code.Session() as db:
        # Recheck existing heights just before inserting to handle race conditions
        heights_to_check = [block[0] for block in block_hashes]
        existing_query = await db.execute(select(db_code.BlockHash.block_height).where(db_code.BlockHash.block_height.in_(heights_to_check)))
        existing_heights = set(existing_query.scalars().all())
        new_blocks = [db_code.BlockHash(block_height=height, block_hash=hash) for height, hash in block_hashes if height not in existing_heights]
        if new_blocks:
            db.add_all(new_blocks)
            try:
                await db.commit()
            except Exception as e:
                await db.rollback()
                logger.error(f"Error during bulk insert of block hashes: {str(e)}")

async def determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack_ticket_txid: str):
    logger.info(f"Retrieving credit pack ticket data for txid: {credit_pack_ticket_txid}")
    _, credit_pack_purchase_request_response, _ = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_txid)
    credit_pack_purchase_request_fields_json = base64.b64decode(credit_pack_purchase_request_response.credit_pack_purchase_request_fields_json_b64).decode('utf-8')
    credit_pack_purchase_request_dict = json.loads(credit_pack_purchase_request_fields_json)
    initial_credit_balance = credit_pack_purchase_request_dict['requested_initial_credits_in_credit_pack']
    credit_usage_tracking_psl_address = credit_pack_purchase_request_response.credit_usage_tracking_psl_address
    logger.info(f"Credit pack ticket data retrieved. Initial credit balance: {initial_credit_balance:,.1f}, Tracking address: {credit_usage_tracking_psl_address}")
    async with db_code.Session() as db:
        db_transactions_result = await db.exec(
            select(db_code.BurnAddressTransaction)
            .where(db_code.BurnAddressTransaction.burn_address == burn_address)
            .where(db_code.BurnAddressTransaction.tracking_address == credit_usage_tracking_psl_address)
            .where(db_code.BurnAddressTransaction.amount <= 1.0)
        )
        db_transactions = db_transactions_result.all()
    total_credits_consumed = sum(COIN*tx.amount / CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER for tx in db_transactions)
    async with db_code.Session() as db:
        query_results = await db.execute(select(db_code.BlockHash.block_height))
        latest_db_block_height = max(query_results.scalars().all())
    current_block_height = await get_current_pastel_block_height_func()
    if current_block_height > latest_db_block_height:
        async with db_code.Session() as db:
            existing_transactions_query = select(db_code.BurnAddressTransaction)
            existing_transactions_result = await db.execute(existing_transactions_query)
            existing_txids = [tx.txid for tx in existing_transactions_result.scalars().all()]
        latest_db_block_hash = await rpc_connection.getblockhash(latest_db_block_height)
        listsinceblock_output = await rpc_connection.listsinceblock(latest_db_block_hash, 1, True)
        new_transactions = listsinceblock_output["transactions"]
        new_burn_transactions = [
            tx for tx in new_transactions
            if tx.get("address") == burn_address and tx.get("category") == "receive" and tx.get("confirmations", 0) <= (current_block_height - latest_db_block_height) and abs(tx.get("amount")) < 1.0
        ]
        filtered_new_burn_transactions = [x for x in new_burn_transactions if x['txid'] not in existing_txids]
        chunk_size = 500
        ignore_unconfirmed_transactions = 0
        new_decoded_tx_data_list = await process_transactions_in_chunks(filtered_new_burn_transactions, chunk_size, ignore_unconfirmed_transactions)
        logger.info(f"Decoded {len(new_decoded_tx_data_list):,} new burn transactions in total!")
        async with db_code.Session() as db:
            new_tracking_transactions_results = await db.exec(
                select(db_code.BurnAddressTransaction)
                .where(db_code.BurnAddressTransaction.burn_address == burn_address)
                .where(db_code.BurnAddressTransaction.tracking_address == credit_usage_tracking_psl_address)
                .where(
                    or_(
                        db_code.BurnAddressTransaction.block_height > latest_db_block_height,
                        db_code.BurnAddressTransaction.pending == True  # noqa: E712
                    )
                )
            )
            new_tracking_transactions = new_tracking_transactions_results.all()
        total_credits_consumed += sum(COIN*tx.amount / CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER for tx in new_tracking_transactions)
        use_update_block_hashes = 0
        if use_update_block_hashes:
            async with db_code.Session() as db:
                existing_block_hash_results = await db.exec( select(db_code.BlockHash) )
                existing_block_hash_heights = [block_hash.block_height for block_hash in existing_block_hash_results.all()]
            new_block_heights_to_get_block_hash_for = [height for height in range(1, current_block_height) if height not in existing_block_hash_heights]
            logger.info(f"Now getting block hashes for {len(new_block_heights_to_get_block_hash_for):,} new block heights (from {new_block_heights_to_get_block_hash_for[0]:,} to {current_block_height:,}) and storing in the DB...")
            await fetch_and_insert_block_hashes(new_block_heights_to_get_block_hash_for[0], current_block_height)
            logger.info(f"Finished getting and storing {len(new_block_heights_to_get_block_hash_for):,} block hashes in the DB!")
    current_credit_balance = initial_credit_balance - total_credits_consumed
    number_of_confirmation_transactions_from_tracking_address_to_burn_address = len(db_transactions) + len(new_tracking_transactions)
    logger.info(f"Calculation completed. Initial credit balance: {initial_credit_balance:,.1f}; Total credits consumed: {total_credits_consumed:,.1f} across {number_of_confirmation_transactions_from_tracking_address_to_burn_address:,} inference requests; Current credit balance: {current_credit_balance:,.1f}")
    return current_credit_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address

async def determine_current_credit_pack_balance_based_on_tracking_transactions_broken(credit_pack_ticket_txid: str): # This is faster but does not work correctly in all cases-- it misses a lot of transactions for some reason. 
    logger.info(f"Retrieving credit pack ticket data for txid: {credit_pack_ticket_txid}")
    _, credit_pack_purchase_request_response, _ = await retrieve_credit_pack_ticket_using_txid(credit_pack_ticket_txid)
    credit_pack_purchase_request_fields_json = base64.b64decode(credit_pack_purchase_request_response.credit_pack_purchase_request_fields_json_b64).decode('utf-8')
    credit_pack_purchase_request_dict = json.loads(credit_pack_purchase_request_fields_json)
    initial_credit_balance = credit_pack_purchase_request_dict['requested_initial_credits_in_credit_pack']
    credit_usage_tracking_psl_address = credit_pack_purchase_request_response.credit_usage_tracking_psl_address
    logger.info(f"Credit pack ticket data retrieved. Initial credit balance: {initial_credit_balance:,.1f}, Tracking address: {credit_usage_tracking_psl_address}")
    async with db_code.Session() as db:
        db_transactions_result = await db.exec(
            select(db_code.BurnAddressTransaction)
            .where(db_code.BurnAddressTransaction.burn_address == burn_address)
            .where(db_code.BurnAddressTransaction.tracking_address == credit_usage_tracking_psl_address)
        )
        db_transactions = db_transactions_result.all()
    total_credits_consumed = sum(COIN*tx.amount / CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER for tx in db_transactions)
    async with db_code.Session() as db:
        query_results = await db.execute(select(db_code.BlockHash.block_height))
        latest_db_block_height = max(query_results.scalars().all())
    current_block_height = await get_current_pastel_block_height_func()
    if current_block_height > latest_db_block_height:
        async with db_code.Session() as db:
            existing_transactions_query = select(db_code.BurnAddressTransaction)
            existing_transactions_result = await db.execute(existing_transactions_query)
            existing_txids = [tx.txid for tx in existing_transactions_result.scalars().all()]
        chunk_size = 5000
        new_burn_transactions = await rpc_connection.scanburntransactions(credit_usage_tracking_psl_address, latest_db_block_height - chunk_size*3)  # second argument to this function was just "latest_db_block_height" but now trying to subtract an additional offset to make it work.
        filtered_new_burn_transactions = [x for x in new_burn_transactions if x['txid'] not in existing_txids]
        ignore_unconfirmed_transactions = 0
        new_decoded_tx_data_list = await process_transactions_in_chunks(filtered_new_burn_transactions, chunk_size, ignore_unconfirmed_transactions)
        if len(new_decoded_tx_data_list) > 0:
            logger.info(f"Decoded {len(new_decoded_tx_data_list):,} new burn transactions in total!")
        async with db_code.Session() as db:
            new_tracking_transactions_results = await db.exec(
                select(db_code.BurnAddressTransaction)
                .where(db_code.BurnAddressTransaction.burn_address == burn_address)
                .where(db_code.BurnAddressTransaction.tracking_address == credit_usage_tracking_psl_address)
                .where(
                    or_(
                        db_code.BurnAddressTransaction.block_height > latest_db_block_height,
                        db_code.BurnAddressTransaction.pending == True  # noqa: E712
                    )
                )
            )
            new_tracking_transactions = new_tracking_transactions_results.all()
        total_credits_consumed += sum(COIN*tx.amount / CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER for tx in new_tracking_transactions)
        use_update_block_hashes = 0
        if use_update_block_hashes:
            async with db_code.Session() as db:
                existing_block_hash_results = await db.exec( select(db_code.BlockHash) )
                existing_block_hash_heights = [block_hash.block_height for block_hash in existing_block_hash_results.all()]
            new_block_heights_to_get_block_hash_for = [height for height in range(1, current_block_height) if height not in existing_block_hash_heights]
            logger.info(f"Now getting block hashes for {len(new_block_heights_to_get_block_hash_for):,} new block heights (from {new_block_heights_to_get_block_hash_for[0]:,} to {current_block_height:,}) and storing in the DB...")
            await fetch_and_insert_block_hashes(new_block_heights_to_get_block_hash_for[0], current_block_height)
            logger.info(f"Finished getting and storing {len(new_block_heights_to_get_block_hash_for):,} block hashes in the DB!")
    current_credit_balance = initial_credit_balance - total_credits_consumed
    number_of_confirmation_transactions_from_tracking_address_to_burn_address = len(db_transactions) + len(new_tracking_transactions)
    logger.info(f"Calculation completed. Initial credit balance: {initial_credit_balance:,.1f}; Total credits consumed: {total_credits_consumed:,.1f} across {number_of_confirmation_transactions_from_tracking_address_to_burn_address:,} inference requests; Current credit balance: {current_credit_balance:,.1f}")
    return current_credit_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address

async def update_pending_transactions():
    while True:
        async with db_code.Session() as db: # Query for pending transactions
            pending_transactions_result = await db.exec(
                select(db_code.BurnAddressTransaction)
                .where(db_code.BurnAddressTransaction.pending == True)  # noqa: E712
            )
            pending_transactions = pending_transactions_result.all()
        for transaction in pending_transactions:
            try: # Retrieve the raw transaction data from the RPC
                raw_tx = await rpc_connection.getrawtransaction(transaction.txid, 1)
                block_height = raw_tx.get('height', None)
                if block_height is not None:
                    async with db_code.Session() as db: # Update the transaction's block height and pending status
                        stmt = (
                            update(db_code.BurnAddressTransaction)
                            .where(db_code.BurnAddressTransaction.txid == transaction.txid)
                            .values(block_height=block_height, pending=False)
                        )
                        await db.exec(stmt)
                        await db.commit()
            except Exception as e:
                logger.error(f"Error updating pending transaction {transaction.txid}: {str(e)}")
        await asyncio.sleep(60)  # Sleep for a certain interval before the next update
        
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
        query = await db_session.exec(
            select(db_code.InferenceAPIUsageRequest).where(db_code.InferenceAPIUsageRequest.inference_request_id == inference_request_id)
        )
        result = query.one_or_none()
        return result
        
async def get_inference_api_usage_response_for_audit(inference_request_or_response_id: str) -> db_code.InferenceAPIUsageResponse:
    async with db_code.Session() as db_session:
        query1 = await db_session.exec(
            select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_request_id == inference_request_or_response_id)
        )
        result1 = query1.one_or_none()
        query2 = await db_session.exec(
            select(db_code.InferenceAPIUsageResponse).where(db_code.InferenceAPIUsageResponse.inference_response_id == inference_request_or_response_id)
        )
        result2 = query2.one_or_none()        
        if result1 is not None:
            result = result1
        elif result2 is not None:
            result = result2
        return result

async def get_inference_api_usage_result_for_audit(inference_request_or_response_id: str) -> db_code.InferenceAPIOutputResult:
    async with db_code.Session() as db_session:
        query1 = await db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_request_id == inference_request_or_response_id)
        )
        result1 = query1.one_or_none()
        query2 = await db_session.exec(
            select(db_code.InferenceAPIOutputResult).where(db_code.InferenceAPIOutputResult.inference_response_id == inference_request_or_response_id)
        )
        result2 = query2.one_or_none()        
        if result1 is not None:
            result = result1
        elif result2 is not None:
            result = result2        
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

def check_if_transparent_address_is_valid_func(pastel_address_string):
    pastel_address_is_valid = 0
    if rpc_port == '9932':
        if len(pastel_address_string) == 35 and (pastel_address_string[0:2] == 'Pt'):
            pastel_address_is_valid = 1
    elif rpc_port == '19932':
        if len(pastel_address_string) == 35 and (pastel_address_string[0:2] == 'tP'):
            pastel_address_is_valid = 1
    elif rpc_port == '29932':
        if len(pastel_address_string) == 36 and (pastel_address_string[0:2] == '44'):
            pastel_address_is_valid = 1
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

def sort_dict_by_keys(input_dict):
    # Sort the main dictionary by keys and then sort each nested dictionary by keys
    sorted_dict = {key: dict(sorted(value.items())) for key, value in sorted(input_dict.items())}
    return json.dumps(sorted_dict, indent=4)  # Convert the dictionary to a JSON string for output

async def extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance: SQLModel) -> str:
    response_fields = {}
    last_hash_field_name = None
    last_signature_field_names = []
    for field_name in model_instance.__fields__.keys():
        if field_name.startswith("sha3_256_hash_of"):
            last_hash_field_name = field_name
        elif "_signature_on_" in field_name:
            last_signature_field_names.append(field_name)
    if isinstance(model_instance, db_code.CreditPackPurchasePriceAgreementRequestResponse):
        fields_to_exclude = [last_hash_field_name, 'id'] + last_signature_field_names
    else:
        fields_to_exclude = [last_hash_field_name, last_signature_field_names[-1], 'id']
    for field_name, field_value in model_instance.__dict__.items():
        if field_name in fields_to_exclude or '_sa_instance_state' in field_name:
            continue
        if field_value is not None:
            if isinstance(field_value, (datetime, date)):
                response_fields[field_name] = field_value.isoformat()
            elif isinstance(field_value, list):
                response_fields[field_name] = json.dumps(field_value, ensure_ascii=False, sort_keys=True)
            elif isinstance(field_value, dict):
                response_fields[field_name] = sort_dict_by_keys(field_value)
            elif isinstance(field_value, decimal.Decimal):
                response_fields[field_name] = str(field_value)
            elif isinstance(field_value, bool):
                response_fields[field_name] = int(field_value)
            else:
                response_fields[field_name] = field_value
    sorted_response_fields = dict(sorted(response_fields.items()))
    return json.dumps(sorted_response_fields, ensure_ascii=False, sort_keys=True)

async def compute_sha3_256_hash_of_sqlmodel_response_fields(model_instance: SQLModel) -> str:
    response_fields_json = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance)
    sha256_hash_of_response_fields = get_sha256_hash_of_input_data_func(response_fields_json)
    return sha256_hash_of_response_fields

async def validate_credit_pack_blockchain_ticket_data_field_hashes(model_instance: SQLModel):
    validation_errors = []
    # model_instance.credit_purchase_request_response_message_version_string = str(model_instance.credit_purchase_request_response_message_version_string)
    response_fields_json = await extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance)
    expected_hash = get_sha256_hash_of_input_data_func(response_fields_json)
    last_hash_field_name = None
    for field_name in model_instance.__fields__:
        if field_name.startswith("sha3_256_hash_of") and field_name.endswith("_fields"):
            last_hash_field_name = field_name
    if last_hash_field_name:
        actual_hash = getattr(model_instance, last_hash_field_name)
        if actual_hash != expected_hash:
            # print('Skipping hash validation check for now...') # TODO: Fix this!
            validation_errors.append(f"SHA3-256 hash in field {last_hash_field_name} does not match the computed hash of the response fields")    
    return validation_errors

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
                    elif signature_field_name == "responding_supernode_signature_on_credit_pack_purchase_request_fields_json_b64":
                        message_to_verify = getattr(model_instance, "credit_pack_purchase_request_fields_json_b64")
                    else:
                        continue
                    signature = getattr(model_instance, signature_field_name)
                    verification_result = await verify_message_with_pastelid_func(pastelid, message_to_verify, signature)
                    if verification_result != 'OK':
                        validation_errors.append(f"Pastelid signature in field {signature_field_name} failed verification")
            else:
                validation_errors.append(f"Corresponding pastelid field {first_pastelid} not found for signature fields {signature_field_names}")
    else:
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

async def validate_inference_request_message_data_func(model_instance: SQLModel):
    validation_errors = await validate_credit_pack_ticket_message_data_func(model_instance)
    return validation_errors

def get_external_ip_func() -> str:
    urls = [
        "https://ipinfo.io/ip",
        "https://api.ipify.org",
        "https://ifconfig.me"
    ]
    for url in urls:
        try:
            response = httpx.get(url)
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            logger.error(f"Failed to get external IP from {url}: {e}")
    raise RuntimeError("Unable to get external IP address from all fallback options.")

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

async def ensure_burn_address_imported_as_watch_address_in_local_wallet():
    burn_address_already_imported = await check_if_address_is_already_imported_in_local_wallet(burn_address)
    if not burn_address_already_imported:
        logger.info(f"Burn address is NOT yet imported! Now attempting to import burn address {burn_address} as a watch address in the local wallet...")
        await import_address_func(burn_address, "burn_address", True)
    else:
        logger.info(f"Burn address {burn_address} is already imported as a watch address in the local wallet!")
    

#_______________________________________________________________


rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
network, burn_address = get_network_info(rpc_port)
masternode_collateral_amount = required_collateral(network)
rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")

if rpc_port == '9932':
    burn_address = 'PtpasteLBurnAddressXXXXXXXXXXbJ5ndd'
elif rpc_port == '19932':
    burn_address = 'tPpasteLBurnAddressXXXXXXXXXXX3wy7u'
elif rpc_port == '29932':
    burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7' # https://blockchain-devel.slack.com/archives/C03Q2MCQG9K/p1705896449986459

encryption_key = generate_or_load_encryption_key_sync()  # Generate or load the encryption key synchronously    
decrypt_sensitive_fields()
MY_PASTELID = asyncio.run(get_my_local_pastelid_func())
logger.info(f"Using local PastelID: {MY_PASTELID}")
if 'genpassphrase' in other_flags.keys():
    LOCAL_PASTEL_ID_PASSPHRASE = other_flags['genpassphrase']

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