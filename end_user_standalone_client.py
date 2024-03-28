import asyncio
import json
import httpx
import os
import logging
import shutil
import queue
import zstandard as zstd
import base64
import hashlib
import urllib.parse as urlparse
import decimal
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from httpx import AsyncClient, Limits, Timeout
from database_code import InferenceAPIUsageRequestModel, InferenceConfirmationModel

logger = logging.getLogger("pastel_supernode_messaging_client")
MESSAGING_TIMEOUT_IN_SECONDS = 30

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
    masternode_list_full_df['lastseentime'] = pd.to_numeric(masternode_list_full_df['lastseentime'], downcast='integer')            
    masternode_list_full_df['lastpaidtime'] = pd.to_numeric(masternode_list_full_df['lastpaidtime'], downcast='integer')            
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
                    
    async def send_user_message(self, supernode_url: str, to_pastelid: str, message_body: str) -> Dict[str, Any]:
        # Request and sign a challenge
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]

        # Sign the message body using the local RPC client
        message_signature = await sign_message_with_pastelid_func(self.pastelid, message_body, self.passphrase)

        # Prepare the user message
        user_message = {
            "from_pastelid": self.pastelid,
            "to_pastelid": to_pastelid,
            "message_body": message_body,
            "message_signature": message_signature
        }

        # Send the user message
        payload = {
            "user_message": user_message,
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(f"{supernode_url}/send_user_message", json=payload)
            response.raise_for_status()
            result = response.json()
            return result

    async def get_user_messages(self, supernode_url: str) -> List[Dict[str, Any]]:
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
            return result

    async def make_inference_api_usage_request(self, supernode_url: str, request_data: InferenceAPIUsageRequestModel) -> Dict[str, Any]:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]

        payload = {
            "inference_api_usage_request": request_data.dict(),
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(f"{supernode_url}/make_inference_api_usage_request", json=payload)
            response.raise_for_status()
            result = response.json()
            return {
                "inference_response_id": result.get("inference_response_id"),
                "inference_request_id": result.get("inference_request_id"),
                "proposed_cost_of_request_in_inference_credits": result.get("proposed_cost_of_request_in_inference_credits"),
                "remaining_credits_in_pack_after_request_processed": result.get("remaining_credits_in_pack_after_request_processed"),
                "credit_usage_tracking_psl_address": result.get("credit_usage_tracking_psl_address"),
                "request_confirmation_message_amount_in_patoshis": result.get("request_confirmation_message_amount_in_patoshis"),
                "max_block_height_to_include_confirmation_transaction": result.get("max_block_height_to_include_confirmation_transaction"),
                "supernode_pastelids_and_signatures_pack_on_inference_response_id": result.get("supernode_pastelids_and_signatures_pack_on_inference_response_id")
            }

    async def send_inference_confirmation(self, supernode_url: str, confirmation_data: InferenceConfirmationModel) -> Dict[str, Any]:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]

        payload = {
            "inference_confirmation": confirmation_data.dict(),
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(f"{supernode_url}/send_inference_confirmation", json=payload)
            response.raise_for_status()
            result = response.json()
            return result

    async def get_inference_output_results(self, supernode_url: str, inference_request_id: str, inference_response_id: str) -> Dict[str, Any]:
        challenge_result = await self.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        challenge_id = challenge_result["challenge_id"]
        challenge_signature = challenge_result["signature"]

        params = {
            "inference_request_id": inference_request_id,
            "inference_response_id": inference_response_id,
            "challenge": challenge,
            "challenge_id": challenge_id,
            "challenge_signature": challenge_signature
        }
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.get(f"{supernode_url}/get_inference_output_results", params=params)
            response.raise_for_status()
            result = response.json()
            return result
        
async def main():
    global rpc_connection
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
    rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    
    use_test_messaging_functionality = 0
        
    # Replace with your own values
    my_local_pastelid = "jXYdog1FfN1YBphHrrRuMVsXT76gdfMTvDBo2aJyjQnLdz2HWtHUdE376imdgeVjQNK93drAmwWoc7A3G4t2Pj"
    my_passphrase = "5QcX9nX67buxyeC"

    messaging_client = PastelMessagingClient(my_local_pastelid, my_passphrase)

    # Get the list of Supernodes
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    supernode_url = get_top_supernode_url(supernode_list_df)
    supernode_url = 'http://154.38.164.75:7123' #Temporary override for debugging
    
    if use_test_messaging_functionality:
        # Request a challenge from a Supernode
        challenge_result = await messaging_client.request_and_sign_challenge(supernode_url)
        challenge = challenge_result["challenge"]
        signature = challenge_result["signature"]
        logger.info(f"Received challenge: {challenge}")
        logger.info(f"Signed challenge with signature: {signature}")

        # Send a user message
        logger.info("Sending user message...")
        to_pastelid = "jXXiVgtFzLto4eYziePHjjb1hj3c6eXdABej5ndnQ62B8ouv1GYveJaD5QUMfainQM3b4MTieQuzFEmJexw8Cr"
        logger.info(f"Recipient pastelid: {to_pastelid}")
        #Lookup the closest supernode to the recipient pastelid and use it as the supernode_url; this Supernode will act as the "mail server" for the recipient since it is closest to the recipient's pastelid:
        supernode_url, supernode_pastelid = get_closest_supernode_to_pastelid_url(to_pastelid, supernode_list_df)
        supernode_url = 'http://154.38.164.75:7123' #Temporary override for debugging
        logger.info(f"Closest Supernode to recipient pastelid: {supernode_pastelid}")
        
        message_body = "Hello, this is a brand 🍉 NEW test message from a regular user!"
        send_result = await messaging_client.send_user_message(supernode_url, to_pastelid, message_body)
        logger.info(f"Sent user message: {send_result}")

        # Get user messages
        logger.info("Retrieving user messages...")
        logger.info(f"My local pastelid: {my_local_pastelid}")
        # Lookup the closest supernode to the local pastelid and use it as the supernode_url; this Supernode will act as the "mail server" for the local user since it is closest to the local pastelid:
        supernode_url, supernode_pastelid = get_closest_supernode_to_pastelid_url(my_local_pastelid, supernode_list_df)
        supernode_url = 'http://154.38.164.75:7123' #Temporary override for debugging
        logger.info(f"Closest Supernode to local pastelid: {supernode_pastelid}")
        messages = await messaging_client.get_user_messages(supernode_url)
        logger.info(f"Retrieved user messages: {messages}")

    #________________________________________________________

    local_credit_tracking_psl_address = '44oVQrU5Hda9gLWR2ZGrLMiwsb31wkQFNajb'
    burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7' # https://blockchain-devel.slack.com/archives/C03Q2MCQG9K/p1705896449986459
    
    # Load or create the global InferenceCreditPackMockup instance
    CREDIT_PACK_FILE = "credit_pack.json"
    credit_pack = InferenceCreditPackMockup.load_from_json(CREDIT_PACK_FILE)

    if credit_pack is None:
        # Create a new credit pack if the file doesn't exist or is invalid
        credit_pack = InferenceCreditPackMockup(
            credit_pack_identifier="credit_pack_123",
            authorized_pastelids=["jXYdog1FfN1YBphHrrRuMVsXT76gdfMTvDBo2aJyjQnLdz2HWtHUdE376imdgeVjQNK93drAmwWoc7A3G4t2Pj"],
            psl_cost_per_credit=10.0,
            total_psl_cost_for_pack=10000.0,
            initial_credit_balance=100000.0,
            credit_usage_tracking_psl_address=local_credit_tracking_psl_address
        )
        credit_pack.save_to_json(CREDIT_PACK_FILE)

    sample_text_completion = "Explain to me with detailed examples what a Galois group is and how it helps understand the roots of a polynomial equation: "
    sample_text_completion__base64_encoded = base64.b64encode(sample_text_completion.encode()).decode('utf-8')
    model_parameters = {"max_length": 100, "temperature": 0.7}

    # Prepare the inference API usage request
    request_data = InferenceAPIUsageRequestModel(
        requesting_pastelid=my_local_pastelid,
        credit_pack_identifier=credit_pack.credit_pack_identifier,
        requested_model_canonical_string="llama-7b",
        model_parameters_json=json.dumps(model_parameters),
        model_input_data_json_b64=sample_text_completion__base64_encoded
    )

    # Send the inference API usage request
    usage_request_response = await messaging_client.make_inference_api_usage_request(supernode_url, request_data)
    logger.info(f"Received inference API usage request response: {usage_request_response}")

    # Extract the relevant information from the response
    inference_request_id = usage_request_response["inference_request_id"]
    proposed_cost_in_credits = usage_request_response["proposed_cost_of_request_in_inference_credits"]
    credit_usage_tracking_psl_address = usage_request_response["credit_usage_tracking_psl_address"]
    credit_usage_to_tracking_amount_multiplier = 0.0001
    credit_usage_tracking_amount = proposed_cost_in_credits * credit_usage_to_tracking_amount_multiplier
    request_confirmation_message_amount_in_patoshis = usage_request_response["request_confirmation_message_amount_in_patoshis"]

    # Check if the credit pack has sufficient credits
    if credit_pack.has_sufficient_credits(proposed_cost_in_credits):
        # Deduct the credits from the credit pack
        credit_pack.deduct_credits(proposed_cost_in_credits)
        logger.info(f"Credits deducted from the credit pack. Remaining balance: {credit_pack.current_credit_balance}")
        
        # Save the updated credit pack state to the JSON file
        credit_pack.save_to_json(CREDIT_PACK_FILE)
        
        # Send the required PSL coins to authorize the request
        # TODO: Implement the logic to send the required PSL coins from the tracking address to the burn address
        # You can use the `request_confirmation_message_amount_in_patoshis` and `credit_usage_tracking_psl_address` from the response
        # This step may require interaction with the Pastel blockchain and wallet

        # Prepare the inference confirmation
        confirmation_data = InferenceConfirmationModel(
            inference_request_id=inference_request_id,
            confirmation_transaction={"transaction_details": "details_of_the_confirmation_transaction"}
        )

        # Send the inference confirmation
        confirmation_result = await messaging_client.send_inference_confirmation(supernode_url, confirmation_data)
        logger.info(f"Sent inference confirmation: {confirmation_result}")

        # Get the inference output results
        inference_response_id = confirmation_result["inference_response_id"]
        output_results = await messaging_client.get_inference_output_results(supernode_url, inference_request_id, inference_response_id)
        logger.info(f"Retrieved inference output results: {output_results}")
    else:
        logger.warning("Insufficient credits in the credit pack.")

if __name__ == "__main__":
    asyncio.run(main())