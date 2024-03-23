import asyncio
import json
import httpx
import os
import logging
import shutil
import queue
import zstandard as zstd
import base64
import urllib.parse as urlparse
import decimal
import pandas as pd
from typing import List, Dict, Any
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from httpx import AsyncClient, Limits, Timeout

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

def get_top_supernode_url(supernode_list_df):
    if not supernode_list_df.empty:
        top_supernode = supernode_list_df.loc[supernode_list_df['rank'] == supernode_list_df['rank'].min()]
        if not top_supernode.empty:
            ipaddress_port = top_supernode['ipaddress:port'].values[0]
            ipaddress = ipaddress_port.split(':')[0]
            supernode_url = f"http://{ipaddress}:7123"
            return supernode_url
    return None

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

class PastelMessagingClient:
    def __init__(self, pastelid: str, passphrase: str):
        self.pastelid = pastelid
        self.passphrase = passphrase

    async def request_challenge(self, supernode_url: str) -> Dict[str, str]:
        payload = {"pastelid": self.pastelid}
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{supernode_url}/request_challenge", json=payload)
            response.raise_for_status()
            result = response.json()
            return result

    async def sign_challenge(self, challenge: str) -> str:
        # Use the local RPC client to sign the challenge string
        signature = await sign_message_with_pastelid_func(self.pastelid, challenge, self.passphrase)
        return signature

    async def send_user_message(self, supernode_url: str, user_message: Dict[str, Any], challenge: str, challenge_id: str) -> Dict[str, Any]:
        signature = await self.sign_challenge(challenge)
        payload = {
            "user_message": user_message,
            "pastelid_signature": signature,
            "challenge_id": challenge_id
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{supernode_url}/send_user_message", json=payload)
            response.raise_for_status()
            result = response.json()
            return result

    async def get_user_messages(self, supernode_url: str, challenge: str, challenge_id: str) -> List[Dict[str, Any]]:
        signature = await self.sign_challenge(challenge)
        params = {
            "pastelid_signature": signature,
            "challenge_id": challenge_id
        }
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{supernode_url}/get_user_messages/{self.pastelid}", params=params)
            response.raise_for_status()
            result = response.json()
            return result
        
async def send_user_message(self, supernode_url: str, to_pastelid: str, message_body: str, challenge: str, challenge_id: str) -> Dict[str, Any]:
    user_message = {
        "from_pastelid": self.pastelid,
        "to_pastelid": to_pastelid,
        "message_body": message_body,
    }

    # Sign the message with the user's PastelID
    message_signature = await sign_message_with_pastelid_func(self.pastelid, json.dumps(user_message), self.passphrase)
    user_message["signature"] = message_signature

    # Sign the challenge with the user's PastelID
    challenge_signature = await self.sign_challenge(challenge)

    payload = {
        "user_message": user_message,
        "pastelid_signature": challenge_signature,
        "challenge_id": challenge_id
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{supernode_url}/send_user_message", json=payload)
        response.raise_for_status()
        result = response.json()
        return result
    
async def main():
    global rpc_connection
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
    rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    
    # Replace with your own values
    pastelid = "your_pastelid"
    passphrase = "your_passphrase"

    messaging_client = PastelMessagingClient(pastelid, passphrase)

    # Get the list of Supernodes
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    logger.info(f"Supernode list: {supernode_list_df}")

    # Request a challenge from a Supernode
    supernode_url = get_top_supernode_url(supernode_list_df)
    challenge_result = await messaging_client.request_challenge(supernode_url)
    challenge = challenge_result["challenge"]
    challenge_id = challenge_result["challenge_id"]
    logger.info(f"Received challenge: {challenge}")

    # Send a user message
    to_pastelid = "recipient_pastelid"
    message_body = "Hello, this is a test message!"
    send_result = await messaging_client.send_user_message(supernode_url, to_pastelid, message_body, challenge, challenge_id)
    logger.info(f"Sent user message: {send_result}")

    # Get user messages
    messages = await messaging_client.get_user_messages(supernode_url, challenge, challenge_id)
    logger.info(f"Retrieved user messages: {messages}")

if __name__ == "__main__":
    asyncio.run(main())