import hashlib
import struct
import os
import io
import asyncio
import base64
import json
import random
import time
import traceback
from binascii import unhexlify, hexlify, a2b_hex
import urllib.parse as urlparse
from decimal import Decimal
import zstandard as zstd
from httpx import AsyncClient, Limits, Timeout
from logger_config import setup_logger
logger = setup_logger()

base_transaction_amount = Decimal(0.1)
max_concurrent_requests = 1000
FEEPERKB = Decimal(0.1)
COIN = 100000 # patoshis in 1 PSL
OP_CHECKSIG = b'\xac'
OP_CHECKMULTISIG = b'\xae'
OP_PUSHDATA1 = b'\x4c'
OP_DUP = b'\x76'
OP_HASH160 = b'\xa9'
OP_EQUALVERIFY = b'\x88'
    
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

def get_sha3_256_hash(input_data):
    if isinstance(input_data, str):
        input_data = input_data.encode('utf-8')
    return hashlib.sha3_256(input_data).hexdigest()
    
def get_raw_sha3_256_hash(input_data):
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
    
def pushint(n):
    assert 0 < n <= 16
    return bytes([0x51 + n-1])
    
def checkmultisig_scriptpubkey_dump(fd):
    data = fd.read(65*3)
    if not data:
        return None
    r = pushint(1)
    n = 0
    while data:
        chunk = data[0:65]
        data = data[65:]
        if len(chunk) < 33:
            chunk += b'\x00'*(33-len(chunk))
        elif len(chunk) < 65:
            chunk += b'\x00'*(65-len(chunk))
        r += pushdata(chunk)
        n += 1
    r += pushint(n) + OP_CHECKMULTISIG
    return r
    
def addr2bytes(s):
    digits58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    n = 0
    for c in s:
        n *= 58
        if c not in digits58:
            raise ValueError
        n += digits58.index(c)
    h = '%x' % n
    if len(h) % 2:
        h = '0' + h
    for c in s:
        if c == digits58[0]:
            h = '00' + h
        else:
            break
    decoded_address = unhexstr(h)
    prefix_length = 2
    return decoded_address[prefix_length:-4]

def unhexstr(str):
    return unhexlify(str.encode('utf8'))

def pushdata(data):
    length = len(data)
    if length < 0x4c:
        return bytes([length]) + data
    elif length <= 0xff:
        return b'\x4c' + bytes([length]) + data
    elif length <= 0xffff:
        return b'\x4d' + struct.pack('<H', length) + data
    else:
        return b'\x4e' + struct.pack('<I', length) + data

def varint(n):
    if n < 0xfd:
        return bytes([n])
    elif n <= 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n <= 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    else:
        return b'\xff' + struct.pack('<Q', n)
            
class CTxIn:
    def __init__(self, prevout_hash, prevout_n, script_sig=b'', sequence=0xffffffff):
        self.prevout_hash = prevout_hash
        self.prevout_n = prevout_n
        self.script_sig = script_sig
        self.sequence = sequence
    
class CMutableTransaction:
    def __init__(self):
        version = 4  #  SAPLING_TX_VERSION
        overwinter_flag = 1 << 31  # Set the "overwintered" flag
        version |= overwinter_flag        
        self.version = version 
        self.version_group_id = 0x892f2085  # SAPLING_VERSION_GROUP_ID
        self.vin = []
        self.vout = []
        self.lock_time = 0
        self.expiry_height = 0
        self.value_balance = 0
        self.vShieldedSpend = []
        self.vShieldedOutput = []
        self.binding_sig = b'\x00' * 64  # Placeholder binding signature

def packtx(tx):
    # Serialize transaction fields
    tx_data = struct.pack('<I', tx.version)  # Transaction version (4 bytes)
    tx_data += struct.pack('<I', tx.version_group_id)  # Version group ID (4 bytes)
    # Serialize transaction inputs
    tx_data += varint(len(tx.vin))  # Number of inputs (varint)
    for txin in tx.vin:
        tx_data += txin.prevout_hash[::-1]  # Transaction ID (32 bytes) in little-endian
        tx_data += struct.pack('<I', txin.prevout_n)  # Output index (4 bytes)
        tx_data += varint(len(txin.script_sig))  # scriptSig length (varint)
        tx_data += txin.script_sig  # scriptSig (variable length)        
        tx_data += struct.pack('<I', txin.sequence)  # Sequence number (4 bytes)
    # Serialize transaction outputs
    tx_data += varint(len(tx.vout))  # Number of outputs (varint)
    for txout in tx.vout:
        tx_data += struct.pack('<Q', int(txout[0]*COIN))  # Transaction amount in patoshis (8 bytes)
        tx_data += varint(len(txout[1]))  # scriptPubKey length (varint)
        tx_data += txout[1]  # scriptPubKey (variable length)
    tx_data += struct.pack('<I', tx.lock_time)  # Locktime (4 bytes)
    tx_data += struct.pack('<I', tx.expiry_height)  # Expiry height (4 bytes)
    tx_data += struct.pack('<Q', tx.value_balance)  # Value balance (8 bytes)
    # Serialize Sapling-specific fields
    tx_data += varint(len(tx.vShieldedSpend))  # Number of shielded spends (varint)
    tx_data += varint(len(tx.vShieldedOutput))  # Number of shielded outputs (varint)
    if tx.vShieldedOutput:
        tx_data += varint(len(tx.vShieldedOutput))  # Number of shielded outputs (varint)
        # Serialize shielded outputs
    else:
        tx_data += varint(0)  # No shielded outputs    
    if tx.vShieldedSpend or tx.vShieldedOutput:
        consensus_branch_id = 0x5efaaeef  # Vermeer consensus branch ID
        tx_data += struct.pack('<I', consensus_branch_id)  # Consensus branch ID (4 bytes)
        tx_data += tx.binding_sig  # Binding signature (64 bytes)
    return tx_data
    
async def store_data_in_blockchain(input_data):
    global rpc_connection
    try:    
        compressed_data = compress_data(input_data)
        uncompressed_data_hash = get_raw_sha3_256_hash(input_data)
        compressed_data_hash = get_raw_sha3_256_hash(compressed_data)
        compressed_data_length = len(compressed_data)
        identifier = "CREDIT_PACK_STORAGE_TICKET"
        identifier_padded = identifier.encode('utf-8').ljust(32, b'\x00')  # Pad the identifier to 32 bytes
        header = identifier_padded + compressed_data_length.to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash
        data_with_header = header + compressed_data
        combined_data_hex = hexlify(data_with_header)
        txins, change = await select_txins(0.00001)
        raw_transaction = CMutableTransaction()
        raw_transaction.vin = [CTxIn(unhexlify(txin['txid']), txin['vout']) for txin in txins]
        txouts = []
        fd = io.BytesIO(combined_data_hex)
        while True:
            script_pubkey = checkmultisig_scriptpubkey_dump(fd)
            if script_pubkey is None:
                break
            value = round(Decimal(100/COIN), 5)            
            txouts.append((value, script_pubkey))
            change -= value   
        out_value = round(base_transaction_amount, 5)
        change -= out_value
        receiving_address = await rpc_connection.getnewaddress()
        txouts.append((out_value, OP_DUP + OP_HASH160 + pushdata(addr2bytes(receiving_address)) + OP_EQUALVERIFY + OP_CHECKSIG))
        change_address = await rpc_connection.getnewaddress() # change output
        txouts.append([change, OP_DUP + OP_HASH160 + pushdata(addr2bytes(change_address)) + OP_EQUALVERIFY + OP_CHECKSIG])
        logger.info(f"Original Data length: {len(input_data):,} bytes; Compressed data length: {len(compressed_data):,} bytes; Number of multisig outputs: {len(txouts):,}; Total size of multisig outputs in bytes: {sum(len(txout[1]) for txout in txouts):,}") 
        raw_transaction.vout = txouts        
        unsigned_tx = packtx(raw_transaction)
        signed_tx_before_fees = await rpc_connection.signrawtransaction(hexlify(unsigned_tx).decode('utf-8'))
        fee = round(Decimal(len(signed_tx_before_fees)/1000) * FEEPERKB, 5)
        change -= fee
        txouts[-1][0] = change
        final_tx = packtx(raw_transaction)
        signed_tx_after_fees = await rpc_connection.signrawtransaction(hexlify(final_tx).decode('utf-8'))
        assert(signed_tx_after_fees['complete'])
        hex_signed_transaction = signed_tx_after_fees['hex']
        final_signed_transaction_size_in_bytes = len(hex_signed_transaction)/2
        logger.info(f"Final signed transaction size: {final_signed_transaction_size_in_bytes:,} bytes; Overall expansion factor versus compressed data size: {final_signed_transaction_size_in_bytes/len(compressed_data):.2f}; Total transaction fee: {fee:.5f} PSL")
        logger.info(f"Sending data transaction to address: {receiving_address}")
        txid = await rpc_connection.sendrawtransaction(hex_signed_transaction)
        logger.info(f"TXID of Data Transaction: {txid}")
        return txid, final_signed_transaction_size_in_bytes
    except Exception as e:
        logger.error(f"Error occurred while storing data in the blockchain: {e}")
        traceback.print_exc()
        return None

async def retrieve_data_from_blockchain(txid):
    global rpc_connection
    try:
        # Get the raw transaction from the blockchain using the transaction ID
        raw_transaction = await rpc_connection.getrawtransaction(txid, 1)  # Verbose output includes decoded data
        outputs = raw_transaction['vout']  # Extract outputs from the transaction
        # Concatenate all scriptPubKey hex strings excluding the last two (change and receiving address outputs)
        encoded_hex_data = ''.join(output['scriptPubKey']['hex'][4:-4] for output in outputs[:-2])
        # Decode the hex data to bytes and clean up extraneous characters and padding
        reconstructed_combined_data = unhexlify(encoded_hex_data)
        reconstructed_combined_data_cleaned = reconstructed_combined_data.decode('utf-8').replace("A", "").rstrip("\x00")
        data_buffer = unhexlify(reconstructed_combined_data_cleaned)
        # Extract the identifier
        identifier_padded = data_buffer[:32]
        identifier = identifier_padded.rstrip(b'\x00').decode('utf-8')
        data_buffer = data_buffer[32:]  # Remove the identifier from the buffer
        # Extract compressed data length
        compressed_data_length = int.from_bytes(data_buffer[:2], 'big')
        data_buffer = data_buffer[2:]  # Remove the compressed data length from the buffer
        # Extract hashes
        uncompressed_data_hash = data_buffer[:32]
        compressed_data_hash = data_buffer[32:64]
        data_buffer = data_buffer[64:]  # Remove the hashes from the buffer
        # Extract compressed data
        compressed_data = data_buffer[:compressed_data_length]
        data_buffer = data_buffer[compressed_data_length:]  # Remove the compressed data from the buffer
        # Validate the compressed data hash
        if get_raw_sha3_256_hash(compressed_data) != compressed_data_hash:
            logger.error("Compressed data hash verification failed")
            return None
        # Decompress the data and validate the uncompressed data hash and length
        decompressed_data = decompress_data(compressed_data)
        if get_raw_sha3_256_hash(decompressed_data) != uncompressed_data_hash:
            logger.error("Uncompressed data hash verification failed")
            return None
        # Log successful retrieval and return the decompressed data
        logger.info(f"Data retrieved successfully from the blockchain. Identifier: {identifier}, Length: {len(decompressed_data)} bytes")
        return decompressed_data
    except Exception as e:
        logger.error(f"Error occurred while retrieving data from the blockchain: {e}")
        traceback.print_exc()
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
network, burn_address = get_network_info(rpc_port)
rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
