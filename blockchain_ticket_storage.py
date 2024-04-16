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
from binascii import unhexlify, hexlify
import urllib.parse as urlparse
from decimal import Decimal
import zstandard as zstd
from httpx import AsyncClient, Limits, Timeout
from logger_config import setup_logger
logger = setup_logger()

max_concurrent_requests = 5
use_parallel = 0
FEEPERKB = Decimal(0.1)
base_transaction_amount = Decimal(0.1)
COIN = 100000 # patoshis in 1 PSL
    
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
    
class OpCodes:
    OP_0 = 0x00
    OP_PUSHDATA1 = 0x4c
    OP_PUSHDATA2 = 0x4d
    OP_PUSHDATA4 = 0x4e
    OP_1NEGATE = 0x4f
    OP_RESERVED = 0x50
    OP_1 = 0x51
    OP_DUP = 0x76
    OP_HASH160 = 0xa9
    OP_EQUAL = 0x87
    OP_EQUALVERIFY = 0x88
    OP_CHECKSIG = 0xac
    OP_CHECKMULTISIG = 0xae
    OP_RETURN = 0x6a
    
opcodes = OpCodes()
    
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
    r += pushint(n) + bytes([opcodes.OP_CHECKMULTISIG])
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
    return unhexstr(h)[1:-4] # skip version and checksum

def unhexstr(str):
    return unhexlify(str.encode('utf8'))

def pushdata(data):
    if len(data) <= 75:
        return bytes([len(data)]) + data
    elif len(data) <= 255:
        return bytes([opcodes.OP_PUSHDATA1, len(data)]) + data
    elif len(data) <= 65535:
        return bytes([opcodes.OP_PUSHDATA2]) + struct.pack('<H', len(data)) + data
    else:
        return bytes([opcodes.OP_PUSHDATA4]) + struct.pack('<I', len(data)) + data

def varint(n):
    if n < 0xfd:
        return bytes([n])
    elif n < 0xffff:
        return b'\xfd' + struct.pack('<H', n)
    elif n < 0xffffffff:
        return b'\xfe' + struct.pack('<I', n)
    else:
        return b'\xff' + struct.pack('<Q', n)
    
class CTxIn:
    def __init__(self, prevout_hash, prevout_n, script_sig=b'', sequence=0xffffffff):
        self.prevout_hash = prevout_hash
        self.prevout_n = prevout_n
        self.script_sig = script_sig if isinstance(script_sig, bytes) else script_sig.encode('utf-8')
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
        tx_data += txin.prevout_hash  # Transaction ID (32 bytes) in little-endian
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
    tx_data += struct.pack('<q', tx.value_balance)  # Value balance (8 bytes)
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
        uncompressed_data_hash = get_raw_sha256_hash(input_data)
        compressed_data_hash = get_raw_sha256_hash(compressed_data)
        uncompressed_data_length = len(input_data)
        header = uncompressed_data_length.to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash
        data_with_header = header + compressed_data
        combined_data_hex = hexlify(data_with_header)
        fd = io.BytesIO(combined_data_hex)
        txins, change = await select_txins(0.00001)
        raw_transaction = CMutableTransaction()
        raw_transaction.vin = [CTxIn(unhexlify(txin['txid'][::-1]), txin['vout']) for txin in txins]
        txouts = []
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
        txouts.append((out_value, bytes([opcodes.OP_DUP]) + bytes([opcodes.OP_HASH160]) + pushdata(addr2bytes(receiving_address)) + bytes([opcodes.OP_EQUALVERIFY]) + bytes([opcodes.OP_CHECKSIG])))
        change_address = await rpc_connection.getnewaddress() # change output
        txouts.append([change, bytes([opcodes.OP_DUP]) + bytes([opcodes.OP_HASH160]) + pushdata(addr2bytes(change_address)) + bytes([opcodes.OP_EQUALVERIFY]) + bytes([opcodes.OP_CHECKSIG])])
        logger.info(f"Data length: {len(input_data)} bytes; Compressed data length: {len(compressed_data):,} bytes; Number of multisig outputs: {len(txouts):,}; Total size of multisig outputs in bytes: {sum(len(txout[1]) for txout in txouts):,}")
        raw_transaction.vout = txouts        
        # Sign the transaction
        unsigned_tx = packtx(raw_transaction)
        signed_tx = await rpc_connection.signrawtransaction(hexlify(unsigned_tx).decode('utf-8'))
        # Extract the signed transaction hex
        signed_tx_hex = signed_tx['hex']
        # Decode the signed transaction hex to get the updated transaction object
        decoded_tx = await rpc_connection.decoderawtransaction(signed_tx_hex)
        # Update the script_sig for each input in the raw_transaction
        for i, txin in enumerate(decoded_tx['vin']):
            script_sig = unhexlify(txin['scriptSig']['hex'])
            raw_transaction.vin[i].script_sig = script_sig
        final_tx = packtx(raw_transaction)
        final_signed_transaction_size_in_bytes = len(final_tx) / 2
        logger.info(f"Final signed transaction size: {final_signed_transaction_size_in_bytes:,} bytes; Overall expansion factor versus compressed data size: {final_signed_transaction_size_in_bytes/len(compressed_data):.2f}")
        fee = round(Decimal(len(final_tx)/1000) * FEEPERKB, 5)
        assert(signed_tx['complete'])
        hex_signed_transaction = hexlify(final_tx).decode('utf-8')
        logger.info(f"Sending data transaction to address: {receiving_address}")
        logger.info(f"Size: {len(final_tx)/2}  Fee: {fee}")
        send_raw_transaction_result = await rpc_connection.sendrawtransaction(hex_signed_transaction)
        blockchain_transaction_id = send_raw_transaction_result
        logger.info(f"Transaction ID: {blockchain_transaction_id}")
        return blockchain_transaction_id
    except Exception as e:
        logger.error(f"Error occurred while storing data in the blockchain: {e}")
        traceback.print_exc()
        return None

async def retrieve_data_from_blockchain(txid):
    global rpc_connection
    raw_transaction = await rpc_connection.getrawtransaction(txid)
    outputs = raw_transaction.split('0100000000000000')
    encoded_hex_data = ''
    for output in outputs[1:-2]:  # there are 3 65-byte parts in this that we need
        cur = 6
        encoded_hex_data += output[cur:cur+130]
        cur += 132
        encoded_hex_data += output[cur:cur+130]
        cur += 132
        encoded_hex_data += output[cur:cur+130]
    encoded_hex_data += outputs[-2][6:-4]
    reconstructed_combined_data = unhexstr(encoded_hex_data)
    uncompressed_data_length = int.from_bytes(reconstructed_combined_data[:2], 'big')
    uncompressed_data_hash = reconstructed_combined_data[2:34]
    compressed_data_hash = reconstructed_combined_data[34:66]
    compressed_data = reconstructed_combined_data[66:]
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
