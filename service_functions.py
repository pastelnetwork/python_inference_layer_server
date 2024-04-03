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
import html
import warnings
from datetime import datetime, timedelta
import pandas as pd
import httpx
from httpx import AsyncClient, Limits, Timeout
import urllib.parse as urlparse
from logger_config import setup_logger
import zstandard as zstd
from database_code import AsyncSessionLocal, Message, MessageMetadata, MessageSenderMetadata, MessageReceiverMetadata, MessageSenderReceiverMetadata
from database_code import InferenceAPIUsageRequest, InferenceAPIUsageResponse, InferenceAPIOutputResult, UserMessage, SupernodeUserMessage, InferenceAPIUsageRequestModel, InferenceConfirmationModel, InferenceOutputResultsModel
from sqlalchemy import select, func
from sqlalchemy.exc import OperationalError, InvalidRequestError
from typing import List, Tuple
from decouple import Config as DecoupleConfig, RepositoryEnv
from magika import Magika
magika = Magika()

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
LOCAL_PASTEL_ID_PASSPHRASE = config.get("LOCAL_PASTEL_ID_PASSPHRASE")
credit_usage_to_tracking_amount_multiplier = 10 # Since we always round inference credits to the nearest 0.1, this gives us enough resolution using Patoshis

challenge_store = {}

def parse_timestamp(timestamp_str):
    try:
        # Attempt to parse with fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S.%f')
    except ValueError:
        # Fall back to parsing without fractional seconds
        return pd.to_datetime(timestamp_str, format='%Y-%m-%dT%H:%M:%S')

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

def get_closest_supernode_to_pastelid_url(input_pastelid, supernode_list_df):
    if not supernode_list_df.empty:
        # Compute SHA3-256 hash of the input_pastelid
        input_pastelid_hash = compute_sha3_256_hexdigest(input_pastelid)
        input_pastelid_int = int(input_pastelid_hash, 16)
        list_of_supernode_pastelids = supernode_list_df['extKey'].values.tolist()
        xor_distance_to_supernodes = []
        for x in list_of_supernode_pastelids:
            # Compute SHA3-256 hash of each supernode_pastelid
            supernode_pastelid_hash = compute_sha3_256_hexdigest(x)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            # Compute XOR distance
            distance = input_pastelid_int ^ supernode_pastelid_int
            xor_distance_to_supernodes.append(distance)
        closest_supernode_index = xor_distance_to_supernodes.index(min(xor_distance_to_supernodes))
        closest_supernode_pastelid = list_of_supernode_pastelids[closest_supernode_index]
        closest_supernode_ipaddress_port = supernode_list_df.loc[supernode_list_df['extKey'] == closest_supernode_pastelid]['ipaddress:port'].values[0]
        ipaddress = closest_supernode_ipaddress_port.split(':')[0]
        supernode_url = f"http://{ipaddress}:7123"
        return supernode_url, closest_supernode_pastelid
    return None, None

async def get_n_closest_supernodes_to_pastelid_urls(n, input_pastelid, supernode_list_df):
    if not supernode_list_df.empty:
        # Compute SHA3-256 hash of the input_pastelid
        input_pastelid_hash = compute_sha3_256_hexdigest(input_pastelid)
        input_pastelid_int = int(input_pastelid_hash, 16)

        list_of_supernode_pastelids = supernode_list_df['extKey'].values.tolist()
        xor_distances = []

        # Compute XOR distances for each supernode PastelID
        for supernode_pastelid in list_of_supernode_pastelids:
            supernode_pastelid_hash = compute_sha3_256_hexdigest(supernode_pastelid)
            supernode_pastelid_int = int(supernode_pastelid_hash, 16)
            distance = input_pastelid_int ^ supernode_pastelid_int
            xor_distances.append((supernode_pastelid, distance))

        # Sort the XOR distances in ascending order
        sorted_xor_distances = sorted(xor_distances, key=lambda x: x[1])

        # Get the N closest supernodes
        closest_supernodes = sorted_xor_distances[:n]

        # Retrieve the URLs and PastelIDs of the N closest supernodes
        supernode_urls_and_pastelids = []
        for supernode_pastelid, _ in closest_supernodes:
            supernode_ipaddress_port = supernode_list_df.loc[supernode_list_df['extKey'] == supernode_pastelid]['ipaddress:port'].values[0]
            ipaddress = supernode_ipaddress_port.split(':')[0]
            supernode_url = f"http://{ipaddress}:7123"
            supernode_urls_and_pastelids.append((supernode_url, supernode_pastelid))
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
            
class InferenceCreditPackMockup:
    def __init__(self, credit_pack_identifier: str, authorized_pastelids: List[str], psl_cost_per_credit: float, total_psl_cost_for_pack: float, initial_credit_balance: float, credit_usage_tracking_psl_address: str):
        self.credit_pack_identifier = credit_pack_identifier
        self.authorized_pastelids = authorized_pastelids
        self.psl_cost_per_credit = psl_cost_per_credit
        self.total_psl_cost_for_pack = total_psl_cost_for_pack
        self.current_credit_balance = initial_credit_balance
        self.initial_credit_balance = initial_credit_balance
        self.credit_usage_tracking_psl_address = credit_usage_tracking_psl_address
        self.version = 1
        self.purchase_height = 0
        self.timestamp = datetime.utcnow()

    def is_authorized(self, pastelid: str) -> bool:
        return pastelid in self.authorized_pastelids

    def has_sufficient_credits(self, requested_credits: float) -> bool:
        return self.current_credit_balance >= requested_credits

    def deduct_credits(self, credits_to_deduct: float):
        if self.has_sufficient_credits(credits_to_deduct):
            self.current_credit_balance -= credits_to_deduct
        else:
            raise ValueError("Insufficient credits in the pack.")

    def to_dict(self):
        return {
            "credit_pack_identifier": self.credit_pack_identifier,
            "authorized_pastelids": self.authorized_pastelids,
            "psl_cost_per_credit": self.psl_cost_per_credit,
            "total_psl_cost_for_pack": self.total_psl_cost_for_pack,
            "initial_credit_balance": self.initial_credit_balance,
            "current_credit_balance": self.current_credit_balance,
            "credit_usage_tracking_psl_address": self.credit_usage_tracking_psl_address,
            "version": self.version,
            "purchase_height": self.purchase_height,
            "timestamp": self.timestamp.isoformat()
        }
        
    @classmethod
    def load_from_json(cls, file_path):
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                return cls(
                    credit_pack_identifier=data["credit_pack_identifier"],
                    authorized_pastelids=data["authorized_pastelids"],
                    psl_cost_per_credit=data["psl_cost_per_credit"],
                    total_psl_cost_for_pack=data["total_psl_cost_for_pack"],
                    initial_credit_balance=data["initial_credit_balance"],
                    credit_usage_tracking_psl_address=data["credit_usage_tracking_psl_address"]
                )
        except FileNotFoundError:
            return None

    def save_to_json(self, file_path):
        data = self.to_dict()
        with open(file_path, "w") as file:
            json.dump(data, file, indent=2)        
                
async def get_current_pastel_block_height_func():
    global rpc_connection
    best_block_hash = await rpc_connection.getbestblockhash()
    best_block_details = await rpc_connection.getblock(best_block_hash)
    curent_block_height = best_block_details['height']
    return curent_block_height

async def get_previous_block_hash_and_merkle_root_func():
    global rpc_connection
    previous_block_height = await get_current_pastel_block_height_func()
    previous_block_hash = await rpc_connection.getblockhash(previous_block_height)
    previous_block_details = await rpc_connection.getblock(previous_block_hash)
    previous_block_merkle_root = previous_block_details['merkleroot']
    return previous_block_hash, previous_block_merkle_root, previous_block_height

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
    async with AsyncSessionLocal() as db:
        inference_request = await db.execute(
            select(InferenceAPIUsageRequest).where(InferenceAPIUsageRequest.inference_request_id == inference_request_id)
        )
        inference_request = inference_request.scalar_one_or_none()    
    requesting_pastelid = inference_request.requesting_pastelid
    is_valid_signature = await verify_challenge_signature(requesting_pastelid, challenge_signature, challenge_id)
    return is_valid_signature

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
        # logger.info('Local machine is a supernode!')
        local_sn_rank = local_machine_supernode_data['rank'].values[0]
        local_sn_pastelid = local_machine_supernode_data['extKey'].values[0]
    return local_machine_supernode_data, local_sn_rank, local_sn_pastelid, local_machine_ip_with_proper_port

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
    zstd_compression_level = 20
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

    async with AsyncSessionLocal() as db:
        # Retrieve messages from the database that meet the timestamp criteria
        result = await db.execute(select(Message).where(Message.timestamp >= datetime_cutoff_to_ignore_obsolete_messages).order_by(Message.timestamp.desc()))
        db_messages = result.scalars().all()
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

async def broadcast_message_to_n_closest_supernodes_to_given_pastelid(input_pastelid, message_body, message_type):
    supernode_list_df, _ = await check_supernode_list_func()
    n = 4
    supernode_urls_and_pastelids = await get_n_closest_supernodes_to_pastelid_urls(n, input_pastelid, supernode_list_df)
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    local_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]    
    list_of_supernode_pastelids = [x[1] for x in supernode_urls_and_pastelids if x[1]!=local_sn_pastelid]
    list_of_supernode_urls = [x[0] for x in supernode_urls_and_pastelids if x[1]!=local_sn_pastelid]
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
            
async def process_broadcast_messages(message):
    async with AsyncSessionLocal() as db_session:
        message_body = json.loads(message.message_body)
        if message.message_type == 'inference_request_response_announcement_message':
            response_data = json.loads(message_body['message'])
            usage_response = InferenceAPIUsageResponse(
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
            await retry_on_database_locked(db_session.add, usage_response)
            await retry_on_database_locked(db_session.commit)
            await retry_on_database_locked(db_session.refresh, usage_response)
        elif message.message_type == 'inference_request_result_announcement_message':
            result_data = json.loads(message_body['message'])
            output_result = InferenceAPIOutputResult(
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
            async with AsyncSessionLocal() as db:
                if last_processed_timestamp is None:
                    result = await db.execute(select(Message.timestamp).order_by(Message.timestamp.desc()).limit(1))
                    last_processed_timestamp_raw = result.scalar_one_or_none()
                    if last_processed_timestamp_raw is None:
                        last_processed_timestamp = pd.Timestamp.min
                    else:
                        last_processed_timestamp = pd.Timestamp(last_processed_timestamp_raw)
                new_messages_df = await list_sn_messages_func()
                if new_messages_df is not None and not new_messages_df.empty:
                    new_messages_df = new_messages_df[new_messages_df['timestamp'] > last_processed_timestamp]
                    if not new_messages_df.empty:
                        for _, message in new_messages_df.iterrows():
                            result = await db.execute(
                                select(Message).where(
                                    Message.sending_sn_pastelid == message['sending_sn_pastelid'],
                                    Message.receiving_sn_pastelid == message['receiving_sn_pastelid'],
                                    Message.timestamp == message['timestamp']
                                )
                            )
                            existing_message = result.scalar_one_or_none()
                            if existing_message:
                                continue
                            logger.info(f"New message received: {message['message_body']}")
                            last_processed_timestamp = message['timestamp']
                            sending_sn_pastelid = message['sending_sn_pastelid']
                            receiving_sn_pastelid = message['receiving_sn_pastelid']
                            message_size_bytes = len(message['message_body'].encode('utf-8'))
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before updating metadata                            
                            # Update MessageSenderMetadata
                            result = await db.execute(
                                select(MessageSenderMetadata).where(MessageSenderMetadata.sending_sn_pastelid == sending_sn_pastelid)
                            )
                            sender_metadata = result.scalar_one_or_none()
                            if sender_metadata:
                                sender_metadata.total_messages_sent += 1
                                sender_metadata.total_data_sent_bytes += message_size_bytes
                                sender_metadata.sending_sn_txid_vout = message['sending_sn_txid_vout']
                                sender_metadata.sending_sn_pubkey = message['signature']
                            else:
                                sender_metadata = MessageSenderMetadata(
                                    sending_sn_pastelid=sending_sn_pastelid,
                                    total_messages_sent=1,
                                    total_data_sent_bytes=message_size_bytes,
                                    sending_sn_txid_vout=message['sending_sn_txid_vout'],
                                    sending_sn_pubkey=message['signature']
                                )
                                db.add(sender_metadata)
                            # Update MessageReceiverMetadata
                            result = await db.execute(
                                select(MessageReceiverMetadata).where(MessageReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid)
                            )
                            receiver_metadata = result.scalar_one_or_none()
                            if receiver_metadata:
                                receiver_metadata.total_messages_received += 1
                                receiver_metadata.total_data_received_bytes += message_size_bytes
                                receiver_metadata.receiving_sn_txid_vout = message['receiving_sn_txid_vout']
                            else:
                                receiver_metadata = MessageReceiverMetadata(
                                    receiving_sn_pastelid=receiving_sn_pastelid,
                                    total_messages_received=1,
                                    total_data_received_bytes=message_size_bytes,
                                    receiving_sn_txid_vout=message['receiving_sn_txid_vout']
                                )
                                db.add(receiver_metadata)
                            # Update MessageSenderReceiverMetadata
                            result = await db.execute(
                                select(MessageSenderReceiverMetadata).where(
                                    MessageSenderReceiverMetadata.sending_sn_pastelid == sending_sn_pastelid,
                                    MessageSenderReceiverMetadata.receiving_sn_pastelid == receiving_sn_pastelid
                                )
                            )
                            sender_receiver_metadata = result.scalar_one_or_none()
                            if sender_receiver_metadata:
                                sender_receiver_metadata.total_messages += 1
                                sender_receiver_metadata.total_data_bytes += message_size_bytes
                            else:
                                sender_receiver_metadata = MessageSenderReceiverMetadata(
                                    sending_sn_pastelid=sending_sn_pastelid,
                                    receiving_sn_pastelid=receiving_sn_pastelid,
                                    total_messages=1,
                                    total_data_bytes=message_size_bytes
                                )
                                db.add(sender_receiver_metadata)
                            new_messages = [
                                Message(
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
                            # Process broadcast messages concurrently
                            processing_tasks = [
                                process_broadcast_messages(message, db)
                                for message in new_messages
                            ]
                            await asyncio.gather(*processing_tasks)              
                            # Update overall MessageMetadata
                            result = await db.execute(
                                select(
                                    func.count(Message.id),
                                    func.count(func.distinct(Message.sending_sn_pastelid)),
                                    func.count(func.distinct(Message.receiving_sn_pastelid))
                                )
                            )
                            total_messages, total_senders, total_receivers = (result.first())
                            result = await db.execute(select(MessageMetadata).order_by(MessageMetadata.timestamp.desc()).limit(1))
                            message_metadata = result.scalar_one_or_none()
                            if message_metadata:
                                message_metadata.total_messages = total_messages
                                message_metadata.total_senders = total_senders
                                message_metadata.total_receivers = total_receivers
                            else:
                                message_metadata = MessageMetadata(
                                    total_messages=total_messages,
                                    total_senders=total_senders,
                                    total_receivers=total_receivers
                                )
                                db.add(message_metadata)
                            await asyncio.sleep(random.uniform(0.1, 0.5))  # Add a short random sleep before updating message metadata                                
                            await retry_on_database_locked(db.commit)
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while monitoring new messages: {str(e)}")
            await asyncio.sleep(5)
        finally:
            await db.close()  # Close the session explicitly
            await asyncio.sleep(5)            
            
async def create_user_message(from_pastelid: str, to_pastelid: str, message_body: str, message_signature: str) -> dict:
    async with AsyncSessionLocal() as db:
        user_message = UserMessage(from_pastelid=from_pastelid, to_pastelid=to_pastelid, message_body=message_body, message_signature=message_signature)
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)
        return user_message.to_dict()  # Assuming you have a to_dict method or similar to serialize your model

async def create_supernode_user_message(sending_sn_pastelid: str, receiving_sn_pastelid: str, user_message_data: dict) -> dict:
    async with AsyncSessionLocal() as db:
        supernode_user_message = SupernodeUserMessage(
            message_body=user_message_data['message_body'],
            message_type="user_message",
            sending_sn_pastelid=sending_sn_pastelid,
            receiving_sn_pastelid=receiving_sn_pastelid,
            signature=user_message_data['message_signature'],
            timestamp=datetime.utcnow(),
            user_message_id=user_message_data['id']
        )
        db.add(supernode_user_message)
        await db.commit()
        await db.refresh(supernode_user_message)
        return supernode_user_message.to_dict() 

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
            "timestamp": datetime.utcnow().isoformat(),  # Current UTC timestamp
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

async def process_received_user_message(supernode_user_message: SupernodeUserMessage):
    async with AsyncSessionLocal() as db:
        user_message = await db.get(UserMessage, supernode_user_message.user_message_id)
        if user_message:
            verification_status = await verify_message_with_pastelid_func(user_message.from_pastelid, user_message.message_body, user_message.signature)
            if verification_status == 'OK':
                # Process the user message (e.g., store it, forward it to the recipient, etc.)
                logger.info(f"Received and verified user message from {user_message.from_pastelid} to {user_message.to_pastelid}")
            else:
                logger.warning(f"Received user message from {user_message.from_pastelid} to {user_message.to_pastelid}, but verification failed")
        else:
            logger.warning(f"Received SupernodeUserMessage (id: {supernode_user_message.id}), but the associated UserMessage was not found")

async def get_user_messages_for_pastelid(pastelid: str) -> List[UserMessage]:
    async with AsyncSessionLocal() as db:
        user_messages = await db.execute(select(UserMessage).where((UserMessage.from_pastelid == pastelid) | (UserMessage.to_pastelid == pastelid)))
        return user_messages.scalars().all()
            
            
#________________________________________________________________________________________________________________            


async def get_inference_model_menu():
    try:
        # Check if the model menu file exists locally
        if os.path.exists("model_menu.json"):
            with open("model_menu.json", "r") as file:
                local_model_menu = json.load(file)
        else:
            local_model_menu = None
        # Fetch the latest model menu from GitHub
        async with httpx.AsyncClient() as client:
            response = await client.get(GITHUB_MODEL_MENU_URL)
            response.raise_for_status()
            github_model_menu = response.json()
        # Compare the local and GitHub model menus
        if local_model_menu != github_model_menu:
            # Update the local model menu file
            with open("model_menu.json", "w") as file:
                json.dump(github_model_menu, file, indent=2)
        # Use the updated model menu
        model_menu = github_model_menu
        return model_menu
    except Exception as e:
        logger.error(f"Error retrieving inference model menu: {str(e)}")
        # Fallback to the local model menu if available
        if local_model_menu:
            return local_model_menu
        else:
            raise

async def save_inference_api_usage_request(inference_request_model: InferenceAPIUsageRequestModel) -> InferenceAPIUsageRequest:
    db_inference_api_usage_request = InferenceAPIUsageRequest(
        requesting_pastelid=inference_request_model.requesting_pastelid,
        credit_pack_identifier=inference_request_model.credit_pack_identifier,
        requested_model_canonical_string=inference_request_model.requested_model_canonical_string,
        model_parameters_json=inference_request_model.model_parameters_json,
        model_input_data_json_b64=inference_request_model.model_input_data_json_b64,
        inference_request_id=str(uuid.uuid4()),  # Generate a unique inference request ID
        total_psl_cost_for_pack=credit_pack.total_psl_cost_for_pack,
        initial_credit_balance=credit_pack.initial_credit_balance,
        requesting_pastelid_signature="placeholder_signature"  # Set a placeholder signature for now
    )
    async with AsyncSessionLocal() as db_session:
        db_session.add(db_inference_api_usage_request)
        await db_session.commit()
        await db_session.refresh(db_inference_api_usage_request)
    return db_inference_api_usage_request

async def calculate_proposed_cost(requested_model_data: dict, model_parameters: dict, input_data: str) -> float:
    # Extract relevant information from the requested_model_data
    base_cost = requested_model_data.get("base_cost", 1.0)
    cost_per_token = requested_model_data.get("cost_per_token", 0.001)
    # Extract relevant information from the model_parameters
    max_tokens = model_parameters.get("max_tokens", 100)
    # Calculate the input data size
    input_data_size = len(input_data) 
    # Calculate the proposed cost based on the extracted information
    proposed_cost_in_credits = round(base_cost + (cost_per_token * max_tokens) + (input_data_size * 0.0001), 1)
    return proposed_cost_in_credits

async def validate_inference_api_usage_request(request_data: dict) -> Tuple[bool, float, float]:
    try:
        requesting_pastelid = request_data["requesting_pastelid"]
        credit_pack_identifier = request_data["credit_pack_identifier"]
        requested_model = request_data["requested_model_canonical_string"]
        model_parameters = request_data["model_parameters_json"]
        input_data = request_data["model_input_data_json_b64"]
        # Check if the credit pack identifier matches the global credit pack
        if credit_pack_identifier != credit_pack.credit_pack_identifier:
            logger.warning(f"Invalid credit pack identifier: {credit_pack_identifier}")
            return False, 0, credit_pack.current_credit_balance
        # Check if the requesting PastelID is authorized to use the credit pack
        if not credit_pack.is_authorized(requesting_pastelid):
            logger.warning(f"Unauthorized PastelID: {requesting_pastelid}")
            return False, 0, credit_pack.current_credit_balance
        # Validate the requested model and parameters against the model menu
        model_menu = await get_inference_model_menu()
        requested_model_data = next(
            (model for model in model_menu["models"] if model["model_name"] == requested_model),
            None
        )
        if requested_model_data is None:
            logger.warning(f"Invalid model requested: {requested_model}")
            return False, 0, credit_pack.current_credit_balance
        # TODO: Validate the model parameters against the model's supported parameters
        # Calculate the proposed cost in credits based on the requested model and input data
        model_parameters_dict = json.loads(model_parameters)
        input_data_binary = base64.b64decode(input_data)
        result = magika.identify_bytes(input_data_binary)
        detected_data_type = result.output.ct_label
        if detected_data_type == "txt":
            input_data = input_data_binary.decode("utf-8")
        proposed_cost_in_credits = await calculate_proposed_cost(requested_model_data, model_parameters_dict, input_data)
        # Check if the credit pack has sufficient credits for the request
        if not credit_pack.has_sufficient_credits(proposed_cost_in_credits):
            logger.warning(f"Insufficient credits for the request. Required: {proposed_cost_in_credits}, Available: {credit_pack.current_credit_balance}")
            return False, proposed_cost_in_credits, credit_pack.current_credit_balance
        # Calculate the remaining credits after the request
        remaining_credits_after_request = credit_pack.current_credit_balance - proposed_cost_in_credits
        if remaining_credits_after_request < 0:
            logger.warning(f"Insufficient credits for the request. Required: {proposed_cost_in_credits}, Available: {credit_pack.current_credit_balance}")
            return False, proposed_cost_in_credits, credit_pack.current_credit_balance
        return True, proposed_cost_in_credits, remaining_credits_after_request
    except Exception as e:
        logger.error(f"Error validating inference API usage request: {str(e)}")
        raise    

async def process_inference_api_usage_request(inference_api_usage_request: InferenceAPIUsageRequestModel) -> InferenceAPIUsageResponse: 
    # Validate the inference API usage request
    request_data = inference_api_usage_request.dict()
    is_valid_request, proposed_cost_in_credits, remaining_credits_after_request = await validate_inference_api_usage_request(request_data)        
    logger.info(f"Received inference API usage request: {request_data}")
    # Save the inference API usage request
    saved_request = await save_inference_api_usage_request(inference_api_usage_request)
    credit_pack_identifier = inference_api_usage_request.credit_pack_identifier
    credit_pack = InferenceCreditPackMockup.load_from_json(CREDIT_PACK_FILE)
    if credit_pack.credit_pack_identifier == credit_pack_identifier:
        credit_usage_tracking_psl_address = credit_pack.credit_usage_tracking_psl_address
    # Create and save the InferenceAPIUsageResponse
    inference_response = await create_and_save_inference_api_usage_response(saved_request, proposed_cost_in_credits, remaining_credits_after_request, credit_usage_tracking_psl_address)
    return inference_response

async def create_and_save_inference_api_usage_response(saved_request: InferenceAPIUsageRequest, proposed_cost_in_credits: float, remaining_credits_after_request: float, credit_usage_tracking_psl_address: str) -> InferenceAPIUsageResponse:
    # Generate a unique identifier for the inference response
    inference_response_id = str(uuid.uuid4())
    # Create an InferenceAPIUsageResponse instance
    _, _, local_supernode_pastelid, _ = await get_local_machine_supernode_data_func()
    confirmation_signature = await sign_message_with_pastelid_func(local_supernode_pastelid, inference_response_id, LOCAL_PASTEL_ID_PASSPHRASE)
    supernode_pastelid_and_signature_on_inference_response_id = json.dumps({'signing_sn_pastelid': local_supernode_pastelid, 'sn_signature_on_response_id': confirmation_signature})
    inference_response = InferenceAPIUsageResponse(
        inference_response_id=inference_response_id,
        inference_request_id=saved_request.inference_request_id,
        proposed_cost_of_request_in_inference_credits=proposed_cost_in_credits,
        remaining_credits_in_pack_after_request_processed=remaining_credits_after_request,
        credit_usage_tracking_psl_address=credit_usage_tracking_psl_address,
        request_confirmation_message_amount_in_patoshis=int(proposed_cost_in_credits * credit_usage_to_tracking_amount_multiplier),
        max_block_height_to_include_confirmation_transaction=await get_current_pastel_block_height_func() + 10,  # Adjust as needed
        supernode_pastelid_and_signature_on_inference_response_id=supernode_pastelid_and_signature_on_inference_response_id
    )
    # Save the InferenceAPIUsageResponse to the database
    async with AsyncSessionLocal() as db_session:
        db_session.add(inference_response)
        await db_session.commit()
        await db_session.refresh(inference_response)
    return inference_response

async def check_burn_address_for_tracking_transaction(
    burn_address: str,
    tracking_address: str,
    expected_amount: float,
    txid: str,
    max_block_height: int,
    max_retries: int = 10,
    initial_retry_delay: int = 25
) -> bool:
    """
    Repeatedly checks the burn address for a transaction with the correct identifier, amount,
    and originating address before a specified block height using the get_and_decode_raw_transaction function
    and also checks the mempool using the getrawmempool RPC method.

    Args:
        burn_address (str): The burn address to check for the transaction.
        tracking_address (str): The address expected to have sent the transaction.
        expected_amount (float): The exact amount of PSL expected in the transaction.
        txid (str): The transaction identifier to look for.
        max_block_height (int): The maximum block height to include the confirmation transaction.
        max_retries (int, optional): Maximum number of retries for checking the transaction. Defaults to 10.
        initial_retry_delay (int, optional): Delay between retries in seconds. Defaults to 25. Increases by 15% each retry.

    Returns:
        bool: True if a matching transaction is found, False otherwise.
    """
    try_count = 0
    retry_delay = initial_retry_delay
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
                        if decoded_tx_data.get("confirmations", 0) >= 0 and decoded_tx_data.get("blockheight", max_block_height + 1) <= max_block_height:
                            logger.info("Matching confirmed transaction found!")
                            return True
                        else:
                            logger.info("Matching unconfirmed transaction found!") 
                            return True
                    else:
                        logger.warning(f"Transaction {txid} found, but the amount sent to the burn address ({total_amount_to_burn_address}) does not match the expected amount ({expected_amount})")
                else:
                    logger.warning(f"Transaction {txid} does not send funds to the specified burn address")
            # If the transaction is not found or does not match the criteria, wait before retrying
            await asyncio.sleep(retry_delay)
            try_count += 1
            retry_delay *= 1.15  # Optional: increase delay between retries
        else:
            logger.error(f"Invalid txid for tracking transaction: {txid}")
    logger.info(f"Transaction not found or did not match the criteria after {max_retries} attempts.")
    return False

async def process_inference_confirmation(inference_request_id: str, confirmation_transaction: InferenceConfirmationModel) -> bool:
    try:
        # Retrieve the inference API usage request from the database
        async with AsyncSessionLocal() as db:
            inference_request = await db.execute(
                select(InferenceAPIUsageRequest).where(InferenceAPIUsageRequest.inference_request_id == inference_request_id)
            )
            inference_request = inference_request.scalar_one_or_none()
        if inference_request is None:
            logger.warning(f"Invalid inference request ID: {inference_request_id}")
            return False
        # Retrieve the inference API usage request response from the database
        async with AsyncSessionLocal() as db:
            inference_response = await db.execute(
                select(InferenceAPIUsageResponse).where(InferenceAPIUsageResponse.inference_request_id == inference_request_id)
            )
            inference_response = inference_response.scalar_one_or_none()
        # Ensure burn address is tracked by local wallet:
        burn_address_already_imported = await check_if_address_is_already_imported_in_local_wallet(burn_address)
        if not burn_address_already_imported:
            await import_address_func(burn_address, "burn_address", True)        
        # Check burn address for tracking transaction:
        confirmation_transaction_txid = confirmation_transaction['txid']
        credit_usage_tracking_amount_in_psl = float(inference_response.request_confirmation_message_amount_in_patoshis)/(10**5) # Divide by number of Patoshis per PSL
        transaction_found = await check_burn_address_for_tracking_transaction(burn_address, inference_response.credit_usage_tracking_psl_address, credit_usage_tracking_amount_in_psl, confirmation_transaction_txid, inference_response.max_block_height_to_include_confirmation_transaction)
        if transaction_found:
            logger.info(f"Found correct inference request confirmation tracking transaction in burn address! TXID: {confirmation_transaction_txid}; Tracking Amount in PSL: {credit_usage_tracking_amount_in_psl};") 
            computed_current_credit_pack_balance, number_of_confirmation_transactions_from_tracking_address_to_burn_address = await determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack, burn_address)
            logger.info(f"Computed current credit pack balance: {computed_current_credit_pack_balance} based on {number_of_confirmation_transactions_from_tracking_address_to_burn_address} tracking transactions from tracking address to burn address.")       
            # Update the inference request status to "confirmed"
            inference_request.status = "confirmed"
            async with AsyncSessionLocal() as db:
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
        # Sign the inference result ID with the local Supernode's PastelID
        result_id_signature = await sign_message_with_pastelid_func(local_supernode_pastelid, inference_result_id, LOCAL_PASTEL_ID_PASSPHRASE)
        supernode_pastelid_and_signature_on_inference_result_id = json.dumps({
            'signing_sn_pastelid': local_supernode_pastelid,
            'sn_signature_on_result_id': result_id_signature
        })
        # Create an inference output result record
        inference_output_result = InferenceAPIOutputResult(
            inference_result_id=inference_result_id,
            inference_request_id=inference_request_id,
            inference_response_id=inference_response_id,
            responding_supernode_pastelid=local_supernode_pastelid,
            inference_result_json_base64=base64.b64encode(json.dumps(output_results).encode("utf-8")).decode("utf-8"),
            inference_result_file_type_strings=json.dumps(output_results_file_type_strings),
            responding_supernode_signature_on_inference_result_id=supernode_pastelid_and_signature_on_inference_result_id
        )
        async with AsyncSessionLocal() as db:
            db.add(inference_output_result)
            await db.commit()
            await db.refresh(inference_output_result)
    except Exception as e:
        logger.error(f"Error saving inference output results: {str(e)}")
        raise
        
async def execute_inference_request(inference_request_id: str) -> None:
    try:
        # Retrieve the inference API usage request from the database
        async with AsyncSessionLocal() as db:
            inference_request = await db.execute(
                select(InferenceAPIUsageRequest).where(InferenceAPIUsageRequest.inference_request_id == inference_request_id)
            )
            inference_request = inference_request.scalar_one_or_none()
        if inference_request is None:
            logger.warning(f"Invalid inference request ID: {inference_request_id}")
            return
        # Retrieve the inference API usage request response from the database
        async with AsyncSessionLocal() as db:
            inference_response = await db.execute(
                select(InferenceAPIUsageResponse).where(InferenceAPIUsageResponse.inference_request_id == inference_request_id)
            )
            inference_response = inference_response.scalar_one_or_none()
        # TODO: Integrate with the Swiss Army Llama project or other inference libraries to perform the inference task
        # For now, let's assume the inference is executed successfully and the output results are generated
        output_results = {
            "output_text": "This is the generated output text.",
            "output_files": []
        }
        # Will use magika library to identify all types of any binary file data, which will be base64 encoded and zstd compressed.
        output_results_file_type_strings = {
            "output_text": "text",  
            "output_files": ["NA"]
        }
        # Save the inference output results to the database
        await save_inference_output_results(inference_request_id, inference_response.inference_response_id, output_results, output_results_file_type_strings)
    except Exception as e:
        logger.error(f"Error executing inference request: {str(e)}")
        raise

async def get_inference_output_results_and_verify_authorization(inference_response_id: str, requesting_pastelid: str) -> InferenceOutputResultsModel:
    async with AsyncSessionLocal() as db_session:
        # Retrieve the inference output result
        inference_output_result = await db_session.execute(
            select(InferenceAPIOutputResult).where(InferenceAPIOutputResult.inference_response_id == inference_response_id)
        )
        inference_output_result = inference_output_result.scalar_one_or_none()
        if inference_output_result is None:
            raise ValueError("Inference output results not found")
        # Retrieve the inference request to verify requesting PastelID
        inference_request = await db_session.execute(
            select(InferenceAPIUsageRequest).where(InferenceAPIUsageRequest.inference_request_id == inference_output_result.inference_request_id)
        )
        inference_request = inference_request.scalar_one_or_none()
        if inference_request is None or inference_request.requesting_pastelid != requesting_pastelid:
            raise ValueError("Unauthorized access to inference output results")
        # Create an InferenceOutputResultsModel instance
        inference_output_results_model = InferenceOutputResultsModel(
            inference_result_id=inference_output_result.inference_result_id,
            inference_request_id=inference_output_result.inference_request_id,
            inference_response_id=inference_output_result.inference_response_id,
            responding_supernode_pastelid=inference_output_result.responding_supernode_pastelid,
            inference_result_json_base64=inference_output_result.inference_result_json_base64,
            inference_result_file_type_strings=inference_output_result.inference_result_file_type_strings,
            responding_supernode_signature_on_inference_result_id=inference_output_result.responding_supernode_signature_on_inference_result_id
        )
        return inference_output_results_model

async def determine_current_credit_pack_balance_based_on_tracking_transactions(credit_pack, burn_address):
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
    initial_credit_balance = credit_pack.initial_credit_balance
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
    credit_usage_to_tracking_amount_multiplier = 10
    total_credits_consumed = 0
    for tx in tracking_transactions:
        for vout in tx.get("vout", []):
            if vout.get("scriptPubKey", {}).get("addresses", [None])[0] == burn_address and vout["value"] < 1.0:
                total_credits_consumed += float(vout["valuePat"]) / credit_usage_to_tracking_amount_multiplier
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
        
async def get_inference_api_usage_response_for_audit(inference_response_id: str) -> InferenceAPIUsageResponse:
    async with AsyncSessionLocal() as db_session:
        result = await db_session.execute(
            select(InferenceAPIUsageResponse).where(InferenceAPIUsageResponse.inference_response_id == inference_response_id)
        )
        return result.scalar_one_or_none()

async def get_inference_api_usage_result_for_audit(inference_response_id: str) -> InferenceAPIOutputResult:
    async with AsyncSessionLocal() as db_session:
        result = await db_session.execute(
            select(InferenceAPIOutputResult).where(InferenceAPIOutputResult.inference_response_id == inference_response_id)
        )
        return result.scalar_one_or_none()            
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


def get_sha256_hash_of_input_data_func(input_data_or_string):
    if isinstance(input_data_or_string, str):
        input_data_or_string = input_data_or_string.encode('utf-8')
    sha256_hash_of_input_data = hashlib.sha3_256(input_data_or_string).hexdigest()
    return sha256_hash_of_input_data


def check_if_ip_address_is_valid_func(ip_address_string):
    try:
        _ = ipaddress.ip_address(ip_address_string)
        ip_address_is_valid = 1
    except Exception as e:
        logger.error('Validation Error: ' + str(e))
        ip_address_is_valid = 0
    return ip_address_is_valid


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




# Load or create the global InferenceCreditPackMockup instance
CREDIT_PACK_FILE = "credit_pack.json"
credit_pack = InferenceCreditPackMockup.load_from_json(CREDIT_PACK_FILE)
users_credit_tracking_psl_address = '44oSueBgdMaAnGxrbTNZwQeDnxvPJg4dGAR3'

if credit_pack is None:
    # Create a new credit pack if the file doesn't exist or is invalid
    credit_pack = InferenceCreditPackMockup(
        credit_pack_identifier="credit_pack_123",
        authorized_pastelids=["jXYdog1FfN1YBphHrrRuMVsXT76gdfMTvDBo2aJyjQnLdz2HWtHUdE376imdgeVjQNK93drAmwWoc7A3G4t2Pj"],
        psl_cost_per_credit=10.0,
        total_psl_cost_for_pack=10000.0,
        initial_credit_balance=100000.0,
        credit_usage_tracking_psl_address=users_credit_tracking_psl_address
    )
    credit_pack.save_to_json(CREDIT_PACK_FILE)


