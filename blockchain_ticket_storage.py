import binascii
import hashlib
import os
import asyncio
import base64
import base58
import json
import random
import time
import traceback
import urllib.parse as urlparse
from decimal import Decimal
from binascii import hexlify
import zstandard as zstd
from httpx import AsyncClient, Limits, Timeout
from logger_config import setup_logger
logger = setup_logger()

max_storage_tasks_in_parallel = 20
max_retrieval_tasks_in_parallel = 20
max_concurrent_requests = 5
storage_task_semaphore = asyncio.BoundedSemaphore(max_storage_tasks_in_parallel)
retrieval_task_semaphore = asyncio.BoundedSemaphore(max_retrieval_tasks_in_parallel)  # Adjust the number as needed
transaction_semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
use_parallel = 0
fee_per_kb = Decimal(0.0001)
base_amount = Decimal(0.000001)
psl_to_patoshis_ratio = 100000
    
def get_network_info(rpc_port):
    if rpc_port == '9932':
        network = 'mainnet'
        burn_address = 'PtpasteLBurnAddressXXXXXXXXXXbJ5ndd'
        address_prefix_bytes = b'\x19'  # Mainnet address prefix byte
    elif rpc_port == '19932':
        network = 'testnet'
        burn_address = 'tPpasteLBurnAddressXXXXXXXXXXX3wy7u'
        address_prefix_bytes = b'\x7f'  # Testnet address prefix byte
    elif rpc_port == '29932':
        network = 'devnet'
        burn_address = '44oUgmZSL997veFEQDq569wv5tsT6KXf9QY7'
        address_prefix_bytes = b'\x1c'  # Devnet address prefix byte
    else:
        raise ValueError(f"Unknown RPC port: {rpc_port}")
    return network, burn_address, address_prefix_bytes

class JSONRPCException(Exception):
    def __init__(self, rpc_error):
        parent_args = []
        try:
            parent_args.append(rpc_error['message'])
        except Exception as e:
            logger.info(f"Error occurred in JSONRPCException: {e}")
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
    max_concurrent_requests = 1000
    _semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
    def __init__(self, service_url, service_name=None, reconnect_timeout=25, max_retries=3, request_timeout=120, fallback_url=None):
        self.service_url = service_url
        self.service_name = service_name
        self.url = urlparse.urlparse(service_url)        
        self.client = AsyncClient(timeout=Timeout(request_timeout), limits=Limits(max_connections=max_concurrent_requests, max_keepalive_connections=100))
        self.id_count = 0
        user = self.url.username
        password = self.url.password
        authpair = f"{user}:{password}".encode('utf-8')
        self.auth_header = b'Basic ' + base64.b64encode(authpair)
        self.reconnect_timeout = reconnect_timeout
        self.max_retries = max_retries
        self.request_timeout = request_timeout
        self.circuit_breaker_open = False
        self.circuit_breaker_timeout = 60
        self.circuit_breaker_failure_threshold = 5
        self.circuit_breaker_failure_count = 0
        self.fallback_url = fallback_url
        self.max_backoff_time = 120
        self.health_check_endpoint = "/health"
        self.health_check_interval = 60
        self.use_health_check = 0

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        if self.service_name is not None:
            name = f"{self.service_name}.{name}"
        return AsyncAuthServiceProxy(self.service_url, name)

    async def __call__(self, *args):
        async with self._semaphore:  # Acquire a semaphore
            if self.circuit_breaker_open:
                if self.circuit_breaker_timeout > 0:
                    logger.warning("Circuit breaker is open. Waiting for timeout...")
                    await asyncio.sleep(self.circuit_breaker_timeout)
                    self.circuit_breaker_timeout = 0
                else:
                    logger.info("Testing circuit breaker with a request...")
                    self.circuit_breaker_failure_count = 0
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
            start_time = time.time()
            for i in range(self.max_retries):
                try:
                    if i > 0:
                        logger.warning(f"Retry attempt #{i+1}")
                        sleep_time = min(self.reconnect_timeout * (2 ** i) + random.uniform(0, self.reconnect_timeout), self.max_backoff_time)
                        logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying.")
                        await asyncio.sleep(sleep_time)
                    if self.use_health_check:
                        await self.health_check()
                    response = await self.client.post(
                        self.service_url, headers=headers, data=postdata)
                    self.circuit_breaker_failure_count = 0
                    self.circuit_breaker_open = False
                    elapsed_time = time.time() - start_time
                    self.adapt_circuit_breaker_timeout(elapsed_time)
                    break
                except Exception as e:
                    logger.error(f"Error occurred in __call__: {e}")
                    logger.exception("Full stack trace:")
                    self.circuit_breaker_failure_count += 1
                    if self.circuit_breaker_failure_count >= self.circuit_breaker_failure_threshold:
                        logger.warning("Circuit breaker threshold reached. Opening circuit.")
                        self.circuit_breaker_open = True
                        self.circuit_breaker_timeout = 60
                        if self.fallback_url:
                            logger.info("Switching to fallback URL.")
                            self.service_url = self.fallback_url
                            self.url = urlparse.urlparse(self.service_url)
            else:
                logger.error("Max retries exceeded.")
                return
            response_json = response.json()
            if response_json['error'] is not None:
                raise JSONRPCException(response_json['error'])
            elif 'result' not in response_json:
                raise JSONRPCException({
                    'code': -343, 'message': 'missing JSON-RPC result'})
            else:
                return response_json['result']

    async def health_check(self):
        try:
            health_check_url = self.service_url + self.health_check_endpoint
            response = await self.client.get(health_check_url)
            if response.status_code != 200:
                raise Exception("Health check failed.")
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            raise

    def adapt_circuit_breaker_timeout(self, elapsed_time):
        if elapsed_time > self.circuit_breaker_timeout:
            self.circuit_breaker_timeout = min(self.circuit_breaker_timeout * 1.5, 300)
        elif elapsed_time < self.circuit_breaker_timeout / 2:
            self.circuit_breaker_timeout = max(self.circuit_breaker_timeout * 0.8, 60)

def get_sha256_hash(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    return hashlib.sha3_256(input_data).hexdigest()
    
def get_raw_sha256_hash(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    return hashlib.sha3_256(input_data).digest()
        
def compress_data(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    zstd_compression_level = 22
    zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
    zstd_compressed_data = zstandard_compressor.compress(input_data)
    return zstd_compressed_data

def decompress_data(compressed_data):
    return zstd.decompress(compressed_data)

def unhexstr(str):
    return binascii.unhexlify(str.encode('utf8'))

async def get_unspent_transactions():
    global rpc_connection
    unspent_transactions = await rpc_connection.listunspent()
    return unspent_transactions

async def select_txins(value, number_of_utxos_to_review=10):
    global rpc_connection
    unspent = await get_unspent_transactions()
    random.shuffle(unspent)
    valid_unspent = []
    reviewed_utxos = 0
    for tx in unspent:
        if not tx['spendable'] or tx['address'] == burn_address or tx['generated'] or tx['amount'] < value:
            continue  # Skip UTXOs that are not spendable or belong to the burn address or which are coinbase transactions
        valid_unspent.append(tx)
        reviewed_utxos += 1
        if reviewed_utxos >= number_of_utxos_to_review:
            break  # Stop reviewing UTXOs if the limit is reached
    # Sort the valid UTXOs by the amount to prioritize smaller amounts
    valid_unspent.sort(key=lambda x: x['amount'])
    selected_txins = []
    total_amount = 0
    for tx in valid_unspent:
        # Check if the wallet has the private key for the UTXO's address
        address_info = await rpc_connection.validateaddress(tx['address'])
        if not address_info['ismine'] and not address_info['iswatchonly']:
            continue  # Skip this UTXO if the wallet doesn't have the private key
        # Check if the UTXO is still valid and unspent
        txout = await rpc_connection.gettxout(tx['txid'], tx['vout'])
        if txout is None:
            continue  # Skip UTXOs that are not found or already spent        
        selected_txins.append(tx)
        total_amount += tx['amount']
        if total_amount >= value:
            break
    if total_amount < value:
        raise Exception("Insufficient funds")
    else:
        return selected_txins, Decimal(total_amount)

async def p2fms_script(data_segments):
    global rpc_connection
    if network == 'mainnet':
        address_prefix_bytes = b'\x38\x87'  # Prefix for mainnet
    elif network == 'testnet':
        address_prefix_bytes = b'\x78\xcd'  # Prefix for testnet
    elif network == 'devnet':
        address_prefix_bytes = b'\x0f\x21'  # Prefix for devnet
    else:
        raise ValueError("Invalid network type specified")
    keys_required_to_spend = 1
    if len(data_segments) > 15:
        raise ValueError("Maximum of 15 data payloads allowed.")
    valid_address = await rpc_connection.getnewaddress()  # Get a valid address from the wallet
    keys = [valid_address]
    for segment in data_segments:
        if len(segment) > 19:
            raise ValueError("Data segment size must be 19 bytes or less.")
        fake_address_bytes = address_prefix_bytes + segment.ljust(19, b'\0')
        checksum = hashlib.sha256(hashlib.sha256(fake_address_bytes).digest()).digest()[:4]
        fake_address = base58.b58encode_check(fake_address_bytes + checksum).decode()
        validate_address_check = await rpc_connection.validateaddress(fake_address)
        if validate_address_check['isvalid']:
            keys.append(fake_address)
        else:
            raise ValueError("Invalid fake address generated.")
    result = await rpc_connection.createmultisig(keys_required_to_spend, keys)
    multisig_address = result["address"]
    hex_encoded_redemption_script = result["redeemScript"]
    decoded_redemption_script = bytes.fromhex(hex_encoded_redemption_script).decode('utf-8')
    logger.info(f"Created P2FMS address: {multisig_address} with redeemScript: {decoded_redemption_script}")
    return multisig_address, decoded_redemption_script

def segment_data(data):
    segment_size_in_bytes = 19  # The size of arbitrary data that can be stored in a fake address
    data_segments = []
    remaining_data = data
    while len(remaining_data) > 0:
        segment = remaining_data[:segment_size_in_bytes]
        data_segments.append(segment)
        remaining_data = remaining_data[segment_size_in_bytes:]
    return data_segments
    
async def create_p2fms_transaction(inputs, outputs):
    global rpc_connection
    # Create the transaction inputs
    transaction_inputs = []
    for utxo in inputs:
        transaction_inputs.append({
            'txid': utxo['txid'],
            'vout': utxo['vout']
        })
    transaction_outputs = {} 
    for output in outputs: # Create the transaction outputs
        if isinstance(output, tuple):
            address, amount = output
            transaction_outputs[address] = float(amount)
        else:
            transaction_outputs[output] = float(base_amount)
    # Calculate the change amount
    total_input_amount = sum(utxo['amount'] for utxo in inputs)
    total_output_amount = sum(transaction_outputs.values())
    change_amount = total_input_amount - total_output_amount - fee_per_kb
    if change_amount > 0:
        change_address = await rpc_connection.getrawchangeaddress()
        transaction_outputs[change_address] = float(change_amount)
    # Create the raw transaction
    raw_transaction = await rpc_connection.createrawtransaction(transaction_inputs, transaction_outputs)
    return raw_transaction

async def send_transaction(signed_tx):
    global rpc_connection
    try:
        txid = await rpc_connection.sendrawtransaction(signed_tx)
        return txid
    except Exception as e:
        logger.error(f"Error occurred while sending transaction: {e}")
        raise

async def create_and_send_transaction(txins, txouts, use_parallel=True):
    global rpc_connection
    logger.info(f"Now creating transaction with inputs:\n {txins}; \n and outputs:\n {txouts}")
    hex_transaction = await create_p2fms_transaction(txins, txouts)
    logger.info(f"Hex raw transaction created before signing: {hex_transaction}")
    assert isinstance(hex_transaction, str)
    signed_tx = await rpc_connection.signrawtransaction(hex_transaction)
    if 'errors' in signed_tx.keys():
        logger.error(f"Error occurred while signing transaction: {signed_tx['errors']}")
        return None
    if not signed_tx['complete']:
        logger.error("Failed to sign all transaction inputs")
        return None
    decoded_signed_raw_transaction = await rpc_connection.decoderawtransaction(signed_tx['hex'])
    logger.info(f"Created signed raw transaction with fields:\n {decoded_signed_raw_transaction}")
    hex_signed_transaction = signed_tx['hex']
    try:
        if use_parallel:
            async with transaction_semaphore:
                send_raw_transaction_result = await send_transaction(hex_signed_transaction)
        else:
            send_raw_transaction_result = await send_transaction(hex_signed_transaction)
        return send_raw_transaction_result
    except JSONRPCException as e:
        if e.code == -25 or e.code == -26:  # -25 indicates missing inputs, -26 indicates insufficient funds
            logger.error(f"Error occurred while sending transaction: {e}")
            return None
        else:
            raise
    except Exception as e:
        logger.error(f"Error occurred while sending transaction: {e}")
        raise

def calculate_chunks(data):
    max_segments_per_chunk = 15  # Max signatures in a multisig is 16, but the first one has to be a real valid address, leaving 15 slots for arbitrary data
    segment_size_in_bytes = 19  # Each of the 15 available slots for arbitrary data can be up to 19 bytes long (for fake addresses)
    max_chunk_size = max_segments_per_chunk * segment_size_in_bytes - 2  # 283 bytes, but we use 2 bytes for an index number for each chunk
    num_chunks = (len(data) + max_chunk_size - 1) // max_chunk_size
    chunks = []
    for i in range(num_chunks):
        start_index = i * max_chunk_size
        end_index = min((i + 1) * max_chunk_size, len(data))
        chunk = data[start_index:end_index]
        chunks.append(chunk)
    return chunks

async def store_data_chunks(chunks):
    chunk_storage_txids = []
    for i, chunk in enumerate(chunks):
        index = i.to_bytes(2, 'big')  # 2-byte index
        chunk_with_index = index + chunk
        txid = await store_data_chunk(chunk_with_index)
        chunk_storage_txids.append(txid)
    return chunk_storage_txids

async def store_data_chunk(chunk):
    global rpc_connection
    async with storage_task_semaphore:
        data_segments = segment_data(chunk)
        multisig_address, redemption_script = await p2fms_script(data_segments)  # Create script with embedded data
        txouts = [
            (multisig_address, base_amount),
            (redemption_script, 0)  # Include the redemption script as a separate output with 0 value
        ]
        selected_utxos, total_amount = await select_txins(base_amount)
        txins = [{'txid': utxo['txid'], 'vout': utxo['vout']} for utxo in selected_utxos]
        change_address = await rpc_connection.getrawchangeaddress()  # Get a change address from the wallet
        change_amount = total_amount - base_amount - fee_per_kb  # Calculate the change amount
        if change_amount > 0:
            txouts.append((change_address, change_amount))  # Add the change output to the transaction
        transaction_id = await create_and_send_transaction(txins, txouts, use_parallel)
        if transaction_id:
            return transaction_id
        return None

async def store_chunk_txids(chunk_txids):
    serialized_txids = b''.join(unhexstr(txid) for txid in chunk_txids)
    transaction_id = await store_data_chunk(serialized_txids)
    return transaction_id
    
async def store_data_in_blockchain(input_data):
    global rpc_connection
    try:    
        # await rpc_connection.lockunspent(True) # Unlock all previously locked UTXOs before starting a new transaction
        compressed_data = compress_data(input_data)
        uncompressed_data_hash = get_raw_sha256_hash(input_data)
        compressed_data_hash = get_raw_sha256_hash(compressed_data)
        uncompressed_data_length = len(input_data)
        header = uncompressed_data_length.to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash
        data_with_header = header + compressed_data
        chunks = calculate_chunks(data_with_header)
        logger.info(f"Total size of compressed data: {len(compressed_data)} bytes; Data will be stored in {len(chunks)} chunks")
        try:
            chunk_txids = await store_data_chunks(chunks)
            if chunk_txids is None:
                logger.error("Error occurred while storing data chunks")
                return None
            else:
                logger.info(f"Data chunks stored successfully in the blockchain. Chunk TXIDs: {chunk_txids}")
                final_txid = await store_chunk_txids(chunk_txids)
                if final_txid is None:
                    return None
                else:
                    logger.info(f"Data stored successfully in the blockchain. Final TXID containing all chunk txids: {final_txid}")
                    return final_txid
        except Exception as e:
            logger.error(f"Error occurred while storing data in the blockchain: {e}")
            logger.error(traceback.format_exc())
            raise        
    except Exception as e:
        logger.error(f"Error occurred while storing data in the blockchain: {e}")
        logger.error(traceback.format_exc())
        raise

async def retrieve_data_from_blockchain(txid):
    global rpc_connection
    raw_transaction = await rpc_connection.getrawtransaction(txid)
    decoded_transaction = await rpc_connection.decoderawtransaction(raw_transaction)
    for output in decoded_transaction['vout']:
        script_pub_key = output['scriptPubKey']['hex']
        script = unhexstr(script_pub_key)
        if script.startswith(b'\x51'):  # Check if the script starts with OP_1
            data_segments = []
            offset = 1  # Skip OP_1
            num_keys = script[offset]
            offset += 1
            for _ in range(num_keys):
                key_length = script[offset]
                offset += 1
                if key_length == 33:  # Skip the valid pubkey (33 bytes)
                    offset += 33
                else:  # Extract the data segment (65 bytes)
                    data_start = offset
                    data_end = data_start + 65
                    data_segments.append(script[data_start:data_end])
                    offset = data_end
            combined_data = b''.join(data_segments)
            if len(combined_data) > 32:  # Check if the data is a chunk or chunk TXIDs
                try:
                    chunk_txids = [hexlify(combined_data[i:i+32]).decode() for i in range(0, len(combined_data), 32)]
                    data_chunks = []
                    for txid in chunk_txids:
                        chunk = await retrieve_chunk(txid)
                        data_chunks.append(chunk)
                    sorted_chunks = sorted(data_chunks, key=lambda x: int.from_bytes(x[:2], 'big'))
                    combined_data = b''.join(chunk[2:] for chunk in sorted_chunks)
                except Exception as e:  # noqa: F841
                    pass  # Treat the data as a single chunk
            uncompressed_data_length = int.from_bytes(combined_data[:2], 'big')
            uncompressed_data_hash = combined_data[2:34]
            compressed_data_hash = combined_data[34:66]
            compressed_data = combined_data[66:]
            if get_raw_sha256_hash(compressed_data) != compressed_data_hash:
                logger.error("Compressed data hash verification failed")
                return None
            decompressed_data = decompress_data(compressed_data)
            if get_raw_sha256_hash(decompressed_data) != uncompressed_data_hash:
                logger.error("Uncompressed data hash verification failed")
                return None
            if len(decompressed_data) != uncompressed_data_length:
                logger.error("Uncompressed data length verification failed")
                return None
            logger.info(f"Data retrieved successfully from the blockchain. Length: {len(decompressed_data)} bytes")
            return decompressed_data
    return None

async def retrieve_chunk(txid):
    global rpc_connection
    raw_transaction = await rpc_connection.getrawtransaction(txid)
    decoded_transaction = await rpc_connection.decoderawtransaction(raw_transaction)
    for output in decoded_transaction['vout']:
        script_pub_key = output['scriptPubKey']['hex']
        script = unhexstr(script_pub_key)
        if script.startswith(b'\x51'):  # Check if the script starts with OP_1
            data_segments = []
            offset = 1  # Skip OP_1
            num_keys = script[offset]
            offset += 1
            for _ in range(num_keys):
                key_length = script[offset]
                offset += 1
                if key_length == 33:  # Skip the valid pubkey (33 bytes)
                    offset += 33
                else:  # Extract the data segment (65 bytes)
                    data_start = offset
                    data_end = data_start + 65
                    data_segments.append(script[data_start:data_end])
                    offset = data_end
            chunk_data = b''.join(data_segments)
            return chunk_data
    return None
            
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

rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
network, burn_address, address_prefix_bytes = get_network_info(rpc_port)
rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
