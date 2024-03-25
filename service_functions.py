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
import re
import html
import warnings
from datetime import datetime, timedelta
import pandas as pd
from httpx import AsyncClient, Limits, Timeout
import urllib.parse as urlparse
from logger_config import setup_logger
import zstandard as zstd
from database_code import AsyncSessionLocal, Message, MessageMetadata, MessageSenderMetadata, MessageReceiverMetadata, MessageSenderReceiverMetadata
from database_code import InferenceCreditPack, InferenceAPIUsageRequest, InferenceAPIUsageResponse, InferenceAPIOutputResult, UserMessage, SupernodeUserMessage
from sqlalchemy import select, func
from typing import List, Tuple
from decouple import Config as DecoupleConfig, RepositoryEnv

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
challenge_store = {}


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
        logger.info('Local machine is a supernode!')
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
    pastelid_to_txid_vout_dict = dict(zip(supernode_list_df['extKey'], supernode_list_df.index))
    txid_vout_to_pastelid_dict = dict(zip(supernode_list_df.index, supernode_list_df['extKey']))
    async with AsyncSessionLocal() as db:
        # Retrieve messages from the database
        result = await db.execute(select(Message).where(Message.timestamp >= datetime_cutoff_to_ignore_obsolete_messages).order_by(Message.timestamp.desc()))
        db_messages_df = pd.DataFrame([message.to_dict() for message in result.scalars().all()])
        if not db_messages_df.empty:
            db_messages_df['timestamp'] = pd.to_datetime(db_messages_df['timestamp'])
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
            message_timestamp = pd.to_datetime(datetime.fromtimestamp(message['Timestamp']))
            # Check if the message already exists in the database
            if not db_messages_df.empty:
                existing_message = db_messages_df[
                    (db_messages_df['sending_sn_pastelid'] == sending_pastelid) &
                    (db_messages_df['receiving_sn_pastelid'] == receiving_pastelid) &
                    (db_messages_df['timestamp'] == message_timestamp)
                ]
                if not existing_message.empty:
                    logger.debug(f"Message already exists in the database. Skipping...")
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
        new_messages_df = pd.DataFrame(new_messages_data)
        combined_messages_df = pd.concat([db_messages_df, new_messages_df], ignore_index=True)
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
    return signed_message_to_send

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

async def monitor_new_messages():
    last_processed_timestamp = None
    while True:
        try:
            async with AsyncSessionLocal() as db:
                if last_processed_timestamp is None:
                    result = await db.execute(select(Message.timestamp).order_by(Message.timestamp.desc()).limit(1))
                    last_processed_timestamp = result.scalar_one_or_none()
                    if last_processed_timestamp is None:
                        last_processed_timestamp = pd.Timestamp.min
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
                            db.add_all(new_messages)
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
                            await db.commit()
                await asyncio.sleep(5)
        except Exception as e:
            logger.error(f"Error while monitoring new messages: {str(e)}")
            await asyncio.sleep(5)
            
async def create_user_message(from_pastelid: str, to_pastelid: str, message_body: str, signature: str) -> UserMessage:
    user_message = UserMessage(from_pastelid=from_pastelid, to_pastelid=to_pastelid, message_body=message_body, signature=signature)
    async with AsyncSessionLocal() as db:
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)
    return user_message

async def create_supernode_user_message(sending_sn_pastelid: str, receiving_sn_pastelid: str, user_message: UserMessage) -> SupernodeUserMessage:
    supernode_user_message = SupernodeUserMessage(
        sending_sn_pastelid=sending_sn_pastelid,
        receiving_sn_pastelid=receiving_sn_pastelid,
        user_message_id=user_message.id
    )
    async with AsyncSessionLocal() as db:
        db.add(supernode_user_message)
        await db.commit()
        await db.refresh(supernode_user_message)
    return supernode_user_message

async def send_user_message_via_supernodes(from_pastelid: str, to_pastelid: str, message_body: str, message_signature: str):
    user_message = await create_user_message(from_pastelid, to_pastelid, message_body, message_signature)
    local_machine_supernode_data, _, _, _ = await get_local_machine_supernode_data_func()
    sending_sn_pastelid = local_machine_supernode_data['extKey'].values.tolist()[0]
    receiving_sn_data = await get_sn_data_from_pastelid_func(to_pastelid)
    if receiving_sn_data.empty:
        raise ValueError(f"No Supernode found for PastelID: {to_pastelid}")
    receiving_sn_pastelid = receiving_sn_data['extKey'].values.tolist()[0]
    supernode_user_message = await create_supernode_user_message(sending_sn_pastelid, receiving_sn_pastelid, user_message)
    signed_message_to_send = json.dumps({
        'message': user_message.message_body,
        'message_type': 'user_message',
        'signature': user_message.signature,
        'from_pastelid': user_message.from_pastelid,
        'to_pastelid': user_message.to_pastelid
    }, ensure_ascii=False)
    await send_message_to_sn_using_pastelid_func(signed_message_to_send, 'user_message', receiving_sn_pastelid, pastelid_passphrase)
    return supernode_user_message

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

async def get_inference_model_menu():
    try:
        # TODO: Implement the logic to retrieve the latest inference model menu from the configured source
        # For now, let's assume the model menu is a static dictionary
        model_menu = {
            "models": [
                {
                    "model_name": "llama-7b",
                    "model_url": "https://huggingface.co/decapoda-research/llama-7b-hf",
                    "input_fields": ["text"],
                    "output_fields": ["text"],
                    "model_parameters": {
                        "max_length": 100,
                        "temperature": 0.7
                    },
                    "credit_costs": {
                        "input_tokens": 1.5,
                        "output_tokens": 1.1
                    }
                },
                # Add more models as needed
            ]
        }
        return model_menu
    except Exception as e:
        logger.error(f"Error retrieving inference model menu: {str(e)}")
        raise

async def validate_inference_api_usage_request(request_data: dict):
    try:
        # TODO: Implement the validation logic for the inference API usage request
        # Check the credit pack balance, verify the user's PastelID signature, and ensure that the requested model and parameters are valid based on the model menu
        # For now, let's assume the request is valid
        return True
    except Exception as e:
        logger.error(f"Error validating inference API usage request: {str(e)}")
        raise

async def process_inference_confirmation(inference_request_id: str, confirmation_transaction: dict):
    try:
        # TODO: Implement the logic to process the inference confirmation
        # Check if the confirmation transaction is valid and trigger the inference request processing
        # For now, let's assume the confirmation is valid and the inference request is processed
        return True
    except Exception as e:
        logger.error(f"Error processing inference confirmation: {str(e)}")
        raise

async def execute_inference_request(inference_request_id: str):
    try:
        # TODO: Implement the logic to execute the inference request
        # Integrate with the Swiss Army Llama project or other inference libraries to perform the inference task and generate the output results
        # For now, let's assume the inference is executed successfully and the output results are generated
        output_results = {
            "output_text": "This is the generated output text.",
            "output_files": []
        }
        return output_results
    except Exception as e:
        logger.error(f"Error executing inference request: {str(e)}")
        raise

async def send_inference_output_results(inference_request_id: str, inference_response_id: str, output_results: dict):
    try:
        async with AsyncSessionLocal() as db:
            inference_output_result = InferenceAPIOutputResult(
                inference_request_id=inference_request_id,
                inference_response_id=inference_response_id,
                responding_supernode_pastelid="your_supernode_pastelid",
                inference_result_json_base64="base64_encoded_output_results",
                responding_supernode_signature_on_inference_result_id="your_signature"
            )
            db.add(inference_output_result)
            await db.commit()
            await db.refresh(inference_output_result)
            return inference_output_result
    except Exception as e:
        logger.error(f"Error sending inference output results: {str(e)}")
        raise

async def update_inference_sn_reputation_score(supernode_pastelid: str, reputation_score: float):
    try:
        # TODO: Implement the logic to update the inference SN reputation score
        # Update the reputation score of the supernode based on its performance in the inference request process
        # For now, let's assume the reputation score is updated successfully
        return True
    except Exception as e:
        logger.error(f"Error updating inference SN reputation score: {str(e)}")
        raise
    
        
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
    output = os.popen('curl ifconfig.me')
    ip_address = output.read()
    return ip_address


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


