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
import re
import pandas as pd
from pydantic import BaseModel
from datetime import datetime, timezone
from typing import List, Dict, Any
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from httpx import AsyncClient, Limits, Timeout
from decouple import Config as DecoupleConfig, RepositoryEnv

# Note: you must have `minrelaytxfee=0.00001` in your pastel.conf to allow "dust" transactions for the inference request confirmation transactions to work!

logger = logging.getLogger("pastel_supernode_messaging_client")

config = DecoupleConfig(RepositoryEnv('.env'))
MESSAGING_TIMEOUT_IN_SECONDS = config.get("MESSAGING_TIMEOUT_IN_SECONDS", default=60, cast=int)
MY_LOCAL_PASTELID = config.get("MY_LOCAL_PASTELID", cast=str)
MY_PASTELID_PASSPHRASE = config.get("MY_PASTELID_PASSPHRASE", cast=str)
LOCAL_CREDIT_TRACKING_PSL_ADDRESS = config.get("LOCAL_CREDIT_TRACKING_PSL_ADDRESS", cast=str)
CREDIT_PACK_FILE = config.get("CREDIT_PACK_FILE", cast=str)

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
        supernode_list_df = supernode_list_df[supernode_list_df['supernode_status']=='ENABLED'] 
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

class InferenceAPIUsageRequestModel(BaseModel):
    requesting_pastelid: str
    credit_pack_identifier: str
    requested_model_canonical_string: str
    model_inference_type_string: str
    model_parameters_json: str
    model_input_data_json_b64: str

    class Config:
        protected_namespaces = ()

class InferenceConfirmationModel(BaseModel):
    inference_request_id: str
    confirmation_transaction: dict

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
        self.timestamp = datetime.now(timezone.utc)

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
                "supernode_pastelid_and_signature_on_inference_response_id": result.get("supernode_pastelid_and_signature_on_inference_response_id")
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
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS*3)) as client: # Added extra time for the checks to complete
            response = await client.post(f"{supernode_url}/confirm_inference_request", json=payload)
            response.raise_for_status()
            result = response.json()
            return result

    async def retrieve_inference_output_results(self, supernode_url: str, inference_request_id: str, inference_response_id: str) -> Dict[str, Any]:
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
        async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
            response = await client.post(f"{supernode_url}/retrieve_inference_output_results", params=params)
            response.raise_for_status()
            result = response.json()
            return result

    async def call_audit_inference_request_response(self, supernode_url: str, inference_response_id: str) -> Dict[str, Any]:
        try:
            signature = await sign_message_with_pastelid_func(self.pastelid, inference_response_id, self.passphrase)
            payload = {
                "inference_response_id": inference_response_id,
                "pastel_id": self.pastelid,
                "signature": signature
            }
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
                response = await client.post(f"{supernode_url}/audit_inference_request_response", json=payload)
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
                    "supernode_pastelid_and_signature_on_inference_response_id": result.get("supernode_pastelid_and_signature_on_inference_response_id")
                }
        except Exception as e:
            logger.error(f"Error in audit_inference_request_response from Supernode URL: {supernode_url}: {e}")
            return {}
        
    async def call_audit_inference_request_result(self, supernode_url: str, inference_response_id: str) -> Dict[str, Any]:
        try:
            signature = await sign_message_with_pastelid_func(self.pastelid, inference_response_id, self.passphrase)
            payload = {
                "inference_response_id": inference_response_id,
                "pastel_id": self.pastelid,
                "signature": signature
            }
            async with httpx.AsyncClient(timeout=Timeout(MESSAGING_TIMEOUT_IN_SECONDS)) as client:
                response = await client.post(f"{supernode_url}/audit_inference_request_result", json=payload)
                response.raise_for_status()
                result = response.json()
                return {
                    "inference_result_id": result.get("inference_result_id"),
                    "inference_request_id": result.get("inference_request_id"),
                    "inference_response_id": result.get("inference_response_id"),
                    "responding_supernode_pastelid": result.get("responding_supernode_pastelid"),
                    "inference_result_json_base64": result.get("inference_result_json_base64"),
                    "inference_result_file_type_strings": result.get("inference_result_file_type_strings"),
                    "responding_supernode_signature_on_inference_result_id": result.get("responding_supernode_signature_on_inference_result_id")
                }
        except Exception as e:
            logger.error(f"Error in audit_inference_request_result from Supernode URL: {supernode_url}: {e}")
            return {}
            
    async def audit_inference_request_response_id(self, inference_response_id: str, pastelid_of_supernode_to_audit: str):
        supernode_list_df, _ = await check_supernode_list_func()
        n = 4
        supernode_urls_and_pastelids = await get_n_closest_supernodes_to_pastelid_urls(n, self.pastelid, supernode_list_df)
        list_of_supernode_pastelids = [x[1] for x in supernode_urls_and_pastelids if x[1] != pastelid_of_supernode_to_audit]
        list_of_supernode_urls = [x[0] for x in supernode_urls_and_pastelids if x[1] != pastelid_of_supernode_to_audit]
        list_of_supernode_ips = [x.split('//')[1].split(':')[0] for x in list_of_supernode_urls]
        logger.info(f"Now attempting to audit inference request response with ID {inference_response_id} with {len(list_of_supernode_pastelids)} closest supernodes (with Supernode IPs of {list_of_supernode_ips})...")
        # Audit the inference request response
        response_audit_tasks = [self.call_audit_inference_request_response(url, inference_response_id) for url in list_of_supernode_urls]
        response_audit_results = await asyncio.gather(*response_audit_tasks)
        # Wait for 20 seconds before auditing the inference request result
        await asyncio.sleep(20)
        # Audit the inference request result
        result_audit_tasks = [self.call_audit_inference_request_result(url, inference_response_id) for url in list_of_supernode_urls]
        result_audit_results = await asyncio.gather(*result_audit_tasks)
        # Combine the audit results
        audit_results = response_audit_results + result_audit_results
        logger.info(f"Audit results retrieved for inference response ID {inference_response_id}")
        return audit_results

def validate_inference_response_fields(
    response_audit_results,
    inference_response_id,
    inference_request_id,
    proposed_cost_in_credits,
    remaining_credits_after_request,
    credit_usage_tracking_psl_address,
    request_confirmation_message_amount_in_patoshis,
    max_block_height_to_include_confirmation_transaction,
    supernode_pastelid_and_signature_on_inference_response_id
):
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
        if not result:
            continue
        if result.get("inference_response_id"):
            inference_response_id_counts[result["inference_response_id"]] = inference_response_id_counts.get(result["inference_response_id"], 0) + 1
        if result.get("inference_request_id"):
            inference_request_id_counts[result["inference_request_id"]] = inference_request_id_counts.get(result["inference_request_id"], 0) + 1
        if result.get("proposed_cost_of_request_in_inference_credits"):
            proposed_cost_in_credits_counts[float(result["proposed_cost_of_request_in_inference_credits"])] = proposed_cost_in_credits_counts.get(float(result["proposed_cost_of_request_in_inference_credits"]), 0) + 1
        if result.get("remaining_credits_in_pack_after_request_processed"):
            remaining_credits_after_request_counts[float(result["remaining_credits_in_pack_after_request_processed"])] = remaining_credits_after_request_counts.get(float(result["remaining_credits_in_pack_after_request_processed"]), 0) + 1
        if result.get("credit_usage_tracking_psl_address"):
            credit_usage_tracking_psl_address_counts[result["credit_usage_tracking_psl_address"]] = credit_usage_tracking_psl_address_counts.get(result["credit_usage_tracking_psl_address"], 0) + 1
        if result.get("request_confirmation_message_amount_in_patoshis"):
            request_confirmation_message_amount_in_patoshis_counts[int(result["request_confirmation_message_amount_in_patoshis"])] = request_confirmation_message_amount_in_patoshis_counts.get(int(result["request_confirmation_message_amount_in_patoshis"]), 0) + 1
        if result.get("max_block_height_to_include_confirmation_transaction"):
            max_block_height_to_include_confirmation_transaction_counts[int(result["max_block_height_to_include_confirmation_transaction"])] = max_block_height_to_include_confirmation_transaction_counts.get(int(result["max_block_height_to_include_confirmation_transaction"]), 0) + 1
        if result.get("supernode_pastelid_and_signature_on_inference_response_id"):
            supernode_pastelid_and_signature_on_inference_response_id_counts[result["supernode_pastelid_and_signature_on_inference_response_id"]] = supernode_pastelid_and_signature_on_inference_response_id_counts.get(result["supernode_pastelid_and_signature_on_inference_response_id"], 0) + 1
    # Determine the majority value for each field
    majority_inference_response_id = max(inference_response_id_counts, key=inference_response_id_counts.get) if inference_response_id_counts else None
    majority_inference_request_id = max(inference_request_id_counts, key=inference_request_id_counts.get) if inference_request_id_counts else None
    majority_proposed_cost_in_credits = max(proposed_cost_in_credits_counts, key=proposed_cost_in_credits_counts.get) if proposed_cost_in_credits_counts else None
    majority_remaining_credits_after_request = max(remaining_credits_after_request_counts, key=remaining_credits_after_request_counts.get) if remaining_credits_after_request_counts else None
    majority_credit_usage_tracking_psl_address = max(credit_usage_tracking_psl_address_counts, key=credit_usage_tracking_psl_address_counts.get) if credit_usage_tracking_psl_address_counts else None
    majority_request_confirmation_message_amount_in_patoshis = max(request_confirmation_message_amount_in_patoshis_counts, key=request_confirmation_message_amount_in_patoshis_counts.get) if request_confirmation_message_amount_in_patoshis_counts else None
    majority_max_block_height_to_include_confirmation_transaction = max(max_block_height_to_include_confirmation_transaction_counts, key=max_block_height_to_include_confirmation_transaction_counts.get) if max_block_height_to_include_confirmation_transaction_counts else None
    majority_supernode_pastelid_and_signature_on_inference_response_id = max(supernode_pastelid_and_signature_on_inference_response_id_counts, key=supernode_pastelid_and_signature_on_inference_response_id_counts.get) if supernode_pastelid_and_signature_on_inference_response_id_counts else None
    # Compare the majority values with the values from the responding supernode
    validation_results = {
        "inference_response_id": majority_inference_response_id == inference_response_id,
        "inference_request_id": majority_inference_request_id == inference_request_id,
        "proposed_cost_in_credits": majority_proposed_cost_in_credits == proposed_cost_in_credits,
        "remaining_credits_after_request": majority_remaining_credits_after_request == remaining_credits_after_request,
        "credit_usage_tracking_psl_address": majority_credit_usage_tracking_psl_address == credit_usage_tracking_psl_address,
        "request_confirmation_message_amount_in_patoshis": majority_request_confirmation_message_amount_in_patoshis == request_confirmation_message_amount_in_patoshis,
        "max_block_height_to_include_confirmation_transaction": majority_max_block_height_to_include_confirmation_transaction == max_block_height_to_include_confirmation_transaction,
        "supernode_pastelid_and_signature_on_inference_response_id": majority_supernode_pastelid_and_signature_on_inference_response_id == supernode_pastelid_and_signature_on_inference_response_id
    }
    return validation_results

def validate_inference_result_fields(
    result_audit_results,
    inference_result_id,
    inference_request_id,
    inference_response_id,
    responding_supernode_pastelid,
    inference_result_json_base64,
    inference_result_file_type_strings,
    responding_supernode_signature_on_inference_result_id
):
    # Count the occurrences of each value for the relevant fields in result_audit_results
    inference_result_id_counts = {}
    inference_request_id_counts = {}
    inference_response_id_counts = {}
    responding_supernode_pastelid_counts = {}
    inference_result_json_base64_counts = {}
    inference_result_file_type_strings_counts = {}
    responding_supernode_signature_on_inference_result_id_counts = {}
    for result in result_audit_results:
        if not result:
            continue
        if result.get("inference_result_id"):
            inference_result_id_counts[result["inference_result_id"]] = inference_result_id_counts.get(result["inference_result_id"], 0) + 1
        if result.get("inference_request_id"):
            inference_request_id_counts[result["inference_request_id"]] = inference_request_id_counts.get(result["inference_request_id"], 0) + 1
        if result.get("inference_response_id"):
            inference_response_id_counts[result["inference_response_id"]] = inference_response_id_counts.get(result["inference_response_id"], 0) + 1
        if result.get("responding_supernode_pastelid"):
            responding_supernode_pastelid_counts[result["responding_supernode_pastelid"]] = responding_supernode_pastelid_counts.get(result["responding_supernode_pastelid"], 0) + 1
        if result.get("inference_result_json_base64"):
            inference_result_json_base64_counts[result["inference_result_json_base64"][:32]] = inference_result_json_base64_counts.get(result["inference_result_json_base64"][:32], 0) + 1
        if result.get("inference_result_file_type_strings"):
            inference_result_file_type_strings_counts[result["inference_result_file_type_strings"]] = inference_result_file_type_strings_counts.get(result["inference_result_file_type_strings"], 0) + 1
        if result.get("responding_supernode_signature_on_inference_result_id"):
            responding_supernode_signature_on_inference_result_id_counts[result["responding_supernode_signature_on_inference_result_id"]] = responding_supernode_signature_on_inference_result_id_counts.get(result["responding_supernode_signature_on_inference_result_id"], 0) + 1
    # Determine the majority value for each field
    majority_inference_result_id = max(inference_result_id_counts, key=inference_result_id_counts.get) if inference_result_id_counts else None
    majority_inference_request_id = max(inference_request_id_counts, key=inference_request_id_counts.get) if inference_request_id_counts else None
    majority_inference_response_id = max(inference_response_id_counts, key=inference_response_id_counts.get) if inference_response_id_counts else None
    majority_responding_supernode_pastelid = max(responding_supernode_pastelid_counts, key=responding_supernode_pastelid_counts.get) if responding_supernode_pastelid_counts else None
    majority_inference_result_json_base64 = max(inference_result_json_base64_counts, key=inference_result_json_base64_counts.get) if inference_result_json_base64_counts else None
    majority_inference_result_file_type_strings = max(inference_result_file_type_strings_counts, key=inference_result_file_type_strings_counts.get) if inference_result_file_type_strings_counts else None
    majority_responding_supernode_signature_on_inference_result_id = max(responding_supernode_signature_on_inference_result_id_counts, key=responding_supernode_signature_on_inference_result_id_counts.get) if responding_supernode_signature_on_inference_result_id_counts else None
    # Compare the majority values with the values from the responding supernode
    validation_results = {
        "inference_result_id": majority_inference_result_id == inference_result_id,
        "inference_request_id": majority_inference_request_id == inference_request_id,
        "inference_response_id": majority_inference_response_id == inference_response_id,
        "responding_supernode_pastelid": majority_responding_supernode_pastelid == responding_supernode_pastelid,
        "inference_result_json_base64": majority_inference_result_json_base64 == inference_result_json_base64[:32],
        "inference_result_file_type_strings": majority_inference_result_file_type_strings == inference_result_file_type_strings,
        "responding_supernode_signature_on_inference_result_id": majority_responding_supernode_signature_on_inference_result_id == responding_supernode_signature_on_inference_result_id
    }
    return validation_results
            
def validate_inference_data(inference_result_dict, audit_results):
    # Split audit_results into response_audit_results and result_audit_results
    response_audit_results = audit_results[:len(audit_results)//2]
    result_audit_results = audit_results[len(audit_results)//2:]
    # Extract relevant fields from inference_result_dict
    usage_request_response = inference_result_dict["usage_request_response"]
    inference_response_id = usage_request_response["inference_response_id"]
    inference_request_id = usage_request_response["inference_request_id"]
    proposed_cost_in_credits = float(usage_request_response["proposed_cost_of_request_in_inference_credits"])
    remaining_credits_after_request = float(usage_request_response["remaining_credits_in_pack_after_request_processed"])
    credit_usage_tracking_psl_address = usage_request_response["credit_usage_tracking_psl_address"]
    request_confirmation_message_amount_in_patoshis = int(usage_request_response["request_confirmation_message_amount_in_patoshis"])
    max_block_height_to_include_confirmation_transaction = int(usage_request_response["max_block_height_to_include_confirmation_transaction"])
    supernode_pastelid_and_signature_on_inference_response_id = usage_request_response["supernode_pastelid_and_signature_on_inference_response_id"]
    # Validate InferenceAPIUsageResponse fields
    response_validation_results = validate_inference_response_fields(
        response_audit_results,
        inference_response_id,
        inference_request_id,
        proposed_cost_in_credits,
        remaining_credits_after_request,
        credit_usage_tracking_psl_address,
        request_confirmation_message_amount_in_patoshis,
        max_block_height_to_include_confirmation_transaction,
        supernode_pastelid_and_signature_on_inference_response_id
    )
    # Extract relevant fields from inference_result_dict for InferenceAPIOutputResult
    usage_result = inference_result_dict["output_results"]
    inference_result_id = usage_result["inference_result_id"]
    responding_supernode_pastelid = usage_result["responding_supernode_pastelid"]
    inference_result_json_base64 = usage_result["inference_result_json_base64"]
    inference_result_file_type_strings = usage_result["inference_result_file_type_strings"]
    responding_supernode_signature_on_inference_result_id = usage_result["responding_supernode_signature_on_inference_result_id"]
    # Validate InferenceAPIOutputResult fields
    result_validation_results = validate_inference_result_fields(
        result_audit_results,
        inference_result_id,
        inference_request_id,
        inference_response_id,
        responding_supernode_pastelid,
        inference_result_json_base64,
        inference_result_file_type_strings,
        responding_supernode_signature_on_inference_result_id
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
    # Send the message to the 3 closest Supernodes concurrently
    send_tasks = []
    for supernode_url, _ in closest_supernodes_to_recipient:
        send_task = asyncio.create_task(messaging_client.send_user_message(supernode_url, to_pastelid, message_body))
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
            if message["id"] not in message_ids:
                unique_messages.append(message)
                message_ids.add(message["id"])
    logger.info(f"Retrieved unique user messages: {unique_messages}")
    message_dict = {
        "sent_messages": send_results,
        "received_messages": unique_messages
    }        
    return message_dict
        

async def handle_inference_request_end_to_end(
    input_prompt_text_to_llm: str,
    model_parameters: dict,
    maximum_inference_cost_in_credits: float,
    burn_address: str
):
    global MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE, LOCAL_CREDIT_TRACKING_PSL_ADDRESS, CREDIT_PACK_FILE
    burn_address_already_imported = await check_if_address_is_already_imported_in_local_wallet(burn_address)
    if not burn_address_already_imported:
        logger.info(f"Please wait, importing burn address {burn_address} into local wallet, which requires a reindexing...")
        await import_address_func(burn_address, "burn_address", True)
    local_tracking_address_already_imported = await check_if_address_is_already_imported_in_local_wallet(LOCAL_CREDIT_TRACKING_PSL_ADDRESS)
    if not local_tracking_address_already_imported:
        logger.error(f"Error: Local tracking address does not exist in local wallet: {LOCAL_CREDIT_TRACKING_PSL_ADDRESS}")
    # Create messaging client to use:
    messaging_client = PastelMessagingClient(MY_LOCAL_PASTELID, MY_PASTELID_PASSPHRASE)
    # Load or create the global InferenceCreditPackMockup instance
    credit_pack = InferenceCreditPackMockup.load_from_json(CREDIT_PACK_FILE)
    if credit_pack is None:
        # Create a new credit pack if the file doesn't exist or is invalid
        credit_pack = InferenceCreditPackMockup(
            credit_pack_identifier="credit_pack_123",
            authorized_pastelids=["jXYdog1FfN1YBphHrrRuMVsXT76gdfMTvDBo2aJyjQnLdz2HWtHUdE376imdgeVjQNK93drAmwWoc7A3G4t2Pj"],
            psl_cost_per_credit=10.0,
            total_psl_cost_for_pack=10000.0,
            initial_credit_balance=100000.0,
            credit_usage_tracking_psl_address=LOCAL_CREDIT_TRACKING_PSL_ADDRESS
        )
        credit_pack.save_to_json(CREDIT_PACK_FILE)
    # Get the list of Supernodes
    supernode_list_df, supernode_list_json = await check_supernode_list_func()
    # Get the closest Supernode URL
    supernode_url, supernode_pastelid = get_closest_supernode_to_pastelid_url(MY_LOCAL_PASTELID, supernode_list_df)
    # supernode_url = 'http://154.38.164.75:7123'  # Temporary override for debugging
    logger.info(f"Selected Supernode URL: {supernode_url} for inference request!")
    input_prompt_text_to_llm__base64_encoded = base64.b64encode(input_prompt_text_to_llm.encode()).decode('utf-8')
    # Prepare the inference API usage request
    request_data = InferenceAPIUsageRequestModel(
        requesting_pastelid=MY_LOCAL_PASTELID,
        credit_pack_identifier=credit_pack.credit_pack_identifier,
        requested_model_canonical_string="mistral-7b-instruct-v0.2",
        model_inference_type_string="text_completion",
        model_parameters_json=json.dumps(model_parameters),
        model_input_data_json_b64=input_prompt_text_to_llm__base64_encoded
    )
    # Send the inference API usage request
    usage_request_response = await messaging_client.make_inference_api_usage_request(supernode_url, request_data)
    logger.info(f"Received inference API usage request response from SN:\n {usage_request_response}")
    # Extract the relevant information from the response
    inference_request_id = usage_request_response["inference_request_id"]
    inference_response_id = usage_request_response["inference_response_id"]
    proposed_cost_in_credits = float(usage_request_response["proposed_cost_of_request_in_inference_credits"])
    credit_usage_tracking_psl_address = usage_request_response["credit_usage_tracking_psl_address"]
    try:
        assert(credit_usage_tracking_psl_address == LOCAL_CREDIT_TRACKING_PSL_ADDRESS)
    except AssertionError:
        logger.error(f"Error! Inference request response has a different tracking address than the local tracking address used in the request: {credit_usage_tracking_psl_address} vs {LOCAL_CREDIT_TRACKING_PSL_ADDRESS}")
        return None
    credit_usage_tracking_amount_in_psl = float(usage_request_response["request_confirmation_message_amount_in_patoshis"])/(10**5) # Divide by number of Patoshis per PSL
    # Check if tracking address contains enough PSL to send tracking amount:
    tracking_address_balance = await check_psl_address_balance_alternative_func(credit_usage_tracking_psl_address)
    if tracking_address_balance < credit_usage_tracking_amount_in_psl:
        logger.error(f"Insufficient balance in tracking address: {credit_usage_tracking_psl_address}; amount needed: {credit_usage_tracking_amount_in_psl}; current balance: {tracking_address_balance}; shortfall: {credit_usage_tracking_amount_in_psl - tracking_address_balance}")
        return None
    # Check if the quoted price is less than or equal to the maximum allowed cost
    if proposed_cost_in_credits <= maximum_inference_cost_in_credits:
        # Check if the credit pack has sufficient credits
        if credit_pack.has_sufficient_credits(proposed_cost_in_credits):
            # Deduct the credits from the credit pack
            credit_pack.deduct_credits(proposed_cost_in_credits)
            logger.info(f"Credits deducted from the credit pack. Remaining balance: {credit_pack.current_credit_balance}")
            # Save the updated credit pack state to the JSON file
            credit_pack.save_to_json(CREDIT_PACK_FILE)
            # Send the required PSL coins to authorize the request
            tracking_transaction_txid = await send_tracking_amount_from_control_address_to_burn_address_to_confirm_inference_request(inference_request_id, credit_usage_tracking_psl_address, credit_usage_tracking_amount_in_psl, burn_address)
            txid_looks_valid = bool(re.match("^[0-9a-fA-F]{64}$", tracking_transaction_txid))
            if txid_looks_valid:
                # Prepare the inference confirmation
                confirmation_data = InferenceConfirmationModel(
                    inference_request_id=inference_request_id,
                    confirmation_transaction={"txid": tracking_transaction_txid}
                )
                # Send the inference confirmation
                confirmation_result = await messaging_client.send_inference_confirmation(supernode_url, confirmation_data)
                logger.info(f"Sent inference confirmation: {confirmation_result}")
                max_tries_to_get_confirmation = 10
                initial_wait_time_in_seconds = 5
                wait_time_in_seconds = initial_wait_time_in_seconds
                for cnt in range(max_tries_to_get_confirmation):
                    wait_time_in_seconds = wait_time_in_seconds*(1.15**cnt)
                    logger.info(f"Waiting for the inference results for {round(wait_time_in_seconds, 1)} seconds... (Attempt {cnt+1}/{max_tries_to_get_confirmation}); Checking with Supernode URL: {supernode_url}")
                    await asyncio.sleep(wait_time_in_seconds)
                    # Get the inference output results
                    assert(len(inference_request_id)>0)
                    assert(len(inference_response_id)>0)
                    output_results = await messaging_client.retrieve_inference_output_results(supernode_url, inference_request_id, inference_response_id)
                    logger.info(f"Retrieved inference output results: {output_results}")
                    # Create the inference_result_dict with all relevant information
                    inference_result_dict = {
                        "supernode_url": supernode_url,
                        "request_data": request_data.dict(),
                        "usage_request_response": usage_request_response,
                        "input_prompt_text_to_llm": input_prompt_text_to_llm,
                        "output_results": output_results
                    }
                    audit_results = await messaging_client.audit_inference_request_response_id(inference_response_id, supernode_pastelid)
                    validation_results = validate_inference_data(inference_result_dict, audit_results)
                    logger.info(f"Validation results: {validation_results}")                    
                    return inference_result_dict, audit_results, validation_results
            else:
                logger.error(f"Invalid tracking transaction TXID: {tracking_transaction_txid}")
        else:
            logger.error("Insufficient credits in the credit pack; request cannot be authorized.")
            return None
    else:
        logger.info(f"Quoted price of {proposed_cost_in_credits} credits exceeds the maximum allowed cost of {maximum_inference_cost_in_credits} credits. Inference request not confirmed.")
        return None
            
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
    use_test_inference_request_functionality = 1

    if use_test_messaging_functionality:
        # Sample message data:
        message_body = "Hello, this is a brand 🍉 NEW test message from a regular user!"
        to_pastelid = "jXXiVgtFzLto4eYziePHjjb1hj3c6eXdABej5ndnQ62B8ouv1GYveJaD5QUMfainQM3b4MTieQuzFEmJexw8Cr"        
        message_dict = await send_message_and_check_for_new_incoming_messages(to_pastelid, message_body)
        logger.info(f"Message data: {message_dict}")

    #________________________________________________________

    if use_test_inference_request_functionality:
        input_prompt_text_to_llm = "Explain to me with detailed examples what a Galois group is and how it helps understand the roots of a polynomial equation: "
        model_parameters = {"number_of_tokens_to_generate": 1000, "temperature": 0.7, "grammar_file_string": "", "number_of_completions_to_generate": 1}
        max_credit_cost_to_approve_inference_request = 100.0
        inference_dict, audit_results, validation_results = await handle_inference_request_end_to_end(input_prompt_text_to_llm, model_parameters, max_credit_cost_to_approve_inference_request, burn_address)
        logger.info(f"Inference result data: {inference_dict}")

if __name__ == "__main__":
    asyncio.run(main())