import hashlib
import struct
import os
import io
import asyncio
import base64
import json
import pickle
import random
import time
import traceback
from datetime import datetime
from binascii import unhexlify, hexlify
import urllib.parse as urlparse
from decimal import Decimal
import plyvel
import shutil
import tempfile
import zstandard as zstd
from httpx import AsyncClient, Limits, Timeout
from logger_config import setup_logger
from sqlmodel import select, delete
import database_code as db_code

logger = setup_logger()
base_transaction_amount = Decimal(0.1)
max_concurrent_requests = 1000
FEEPERKB = Decimal(0.00001)
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
    # Sort the valid UTXOs by the amount to prioritize larger amounts
    valid_unspent.sort(key=lambda x: x['amount'], reverse=True)
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
        output_value = int(txout[0]*COIN)
        if 0 <= output_value <= 0xffffffffffffffff:
            tx_data += struct.pack('<Q', output_value)  # Transaction amount in patoshis (8 bytes)
        else:
            logger.error(f"Invalid output value: {output_value}. Skipping this output.")        
        tx_data += varint(len(txout[1]))  # scriptPubKey length (varint)
        tx_data += txout[1]  # scriptPubKey (variable length)
    tx_data += struct.pack('<I', tx.lock_time)  # Locktime (4 bytes)
    tx_data += struct.pack('<I', tx.expiry_height)  # Expiry height (4 bytes)
    if 0 <= tx.value_balance <= 0xffffffffffffffff:
        tx_data += struct.pack('<Q', tx.value_balance)  # Value balance (8 bytes)
    else:
        logger.error(f"Invalid value balance: {tx.value_balance}. Setting it to 0.")
        tx_data += struct.pack('<Q', 0)  # Set value balance to 0 if it's outside the valid range    
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
        txouts.append((float(out_value), OP_DUP + OP_HASH160 + pushdata(addr2bytes(receiving_address)) + OP_EQUALVERIFY + OP_CHECKSIG))
        change_address = await rpc_connection.getnewaddress() # change output
        txouts.append([round(float(change), 5), OP_DUP + OP_HASH160 + pushdata(addr2bytes(change_address)) + OP_EQUALVERIFY + OP_CHECKSIG])
        logger.info(f"Original Data length: {len(input_data):,} bytes; Compressed data length: {len(compressed_data):,} bytes; Number of multisig outputs: {len(txouts):,}; Total size of multisig outputs in bytes: {sum(len(txout[1]) for txout in txouts):,}") 
        raw_transaction.vout = txouts        
        unsigned_tx = packtx(raw_transaction)
        signed_tx_before_fees = await rpc_connection.signrawtransaction(hexlify(unsigned_tx).decode('utf-8'))
        fee = round(Decimal(len(signed_tx_before_fees)/1000) * FEEPERKB, 5)
        if fee > change:
            logger.error(f"Transaction fee exceeds change amount. Fee: {fee:.5f} PSL; Change: {change:.5f} PSL")
            change = 0
        else:
            change -= fee
        txouts[-1][0] = round(float(max(change, 0)), 5)
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
        if len(reconstructed_combined_data_cleaned) % 2 != 0:
            reconstructed_combined_data_cleaned = reconstructed_combined_data_cleaned.replace("!","")
        if len(reconstructed_combined_data_cleaned) % 2 != 0:
            reconstructed_combined_data_cleaned = reconstructed_combined_data_cleaned[:-1]  # Remove any trailing characters
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

async def process_blocks_for_masternode_transactions(force_recheck_from_scratch=False):
    global rpc_connection
    if not os.path.exists('masternode_transactions_dicts'):
        os.makedirs('masternode_transactions_dicts')
    if force_recheck_from_scratch:
        logger.info("Forcing a recheck from scratch. Deleting existing pickle files and database records.")
        for file in os.listdir('masternode_transactions_dicts'):
            if file.startswith('masternode_txids__block_') and file.endswith('.pickle'):
                os.remove(os.path.join('masternode_transactions_dicts', file))
        if os.path.exists('masternode_master_file.pickle'):
            os.remove('masternode_master_file.pickle')
        async with db_code.Session() as db:
            await db.exec(delete(db_code.MasternodeTransaction))
            await db.commit()
    masternode_transactions_dict = {}
    masternode_master_file = {}
    latest_block_height = await rpc_connection.getblockcount()
    last_processed_block = 0
    for file in os.listdir('masternode_transactions_dicts'):
        if file.startswith('masternode_txids__block_') and file.endswith('.pickle'):
            block_range = file[len('masternode_txids__block_'):-len('.pickle')]
            start_block, end_block = map(int, block_range.split('_to_block_'))
            if end_block > last_processed_block:
                last_processed_block = end_block
    logger.info(f"Starting from block {last_processed_block + 1:,} of {latest_block_height:,}")
    batch_size = 100
    for start_block in range(last_processed_block + 1, latest_block_height + 1, batch_size):
        end_block = min(start_block + batch_size - 1, latest_block_height)
        logger.info(f"Processing blocks {start_block:,} to {end_block:,}")
        tasks = []
        for block_height in range(start_block, end_block + 1):
            tasks.append(asyncio.create_task(process_block_for_masternode_transactions(block_height, masternode_transactions_dict)))
        await asyncio.gather(*tasks)
        if end_block % 10000 == 0:
            save_path = os.path.join('masternode_transactions_dicts', f"masternode_txids__block_{start_block}_to_block_{end_block}.pickle")
            with open(save_path, 'wb') as f:
                pickle.dump(masternode_transactions_dict, f)
            logger.info(f"Saved masternode transactions from block {start_block:,} to {end_block:,}")
            masternode_transactions_dict = {}
    if masternode_transactions_dict:
        save_path = os.path.join('masternode_transactions_dicts', f"masternode_txids__block_{end_block - (end_block % 10000) + 1}_to_block_{end_block}.pickle")
        with open(save_path, 'wb') as f:
            pickle.dump(masternode_transactions_dict, f)
        logger.info(f"Saved remaining masternode transactions from block {end_block - (end_block % 10000) + 1:,} to {end_block:,}")
    logger.info("Checking if any masternode txids are moved elsewhere...")
    for key in masternode_transactions_dict:
        tx_id, vout = key.split('-')
        try:
            tx = await rpc_connection.getrawtransaction(tx_id, 1)
            if tx['vout'][int(vout)]['scriptPubKey']['addresses'][0] != masternode_transactions_dict[key]['receiving_address']:
                masternode_master_file[key] = masternode_transactions_dict[key]
                masternode_master_file[key]['moved'] = True
                async with db_code.Session() as db:
                    transaction = await db.exec(select(db_code.MasternodeTransaction).where(db_code.MasternodeTransaction.id == key))
                    transaction = transaction.one_or_none()
                    if transaction:
                        transaction.moved = True
                        await db.commit()
        except JSONRPCException as e:
            if e.code == -5:
                masternode_master_file[key] = masternode_transactions_dict[key]
                masternode_master_file[key]['moved'] = True
                async with db_code.Session() as db:
                    transaction = await db.exec(select(db_code.MasternodeTransaction).where(db_code.MasternodeTransaction.id == key))
                    transaction = transaction.one_or_none()
                    if transaction:
                        transaction.moved = True
                        await db.commit()
    with open('masternode_master_file.pickle', 'wb') as f:
        pickle.dump(masternode_master_file, f)
    logger.info("Masternode transaction processing completed")

async def process_block_for_masternode_transactions(block_height, masternode_transactions_dict):
    global rpc_connection
    block_hash = await rpc_connection.getblockhash(block_height)
    block = await rpc_connection.getblock(block_hash)
    for tx_id in block['tx']:
        tx = await rpc_connection.getrawtransaction(tx_id, 1)
        for vout in tx['vout']:
            if vout['value'] == 5000000:
                receiving_address = vout['scriptPubKey']['addresses'][0]
                key = f"{tx_id}-{vout['n']}"
                masternode_transactions_dict[key] = {
                    'block_height': block_height,
                    'block_datetime': datetime.utcfromtimestamp(block['time']).isoformat(),
                    'receiving_address': receiving_address
                }
                logger.info(f"Found valid masternode transaction with txid {tx_id} and vout {vout['n']} at block {block_height}")
                async with db_code.Session() as db:
                    transaction = db_code.MasternodeTransaction(
                        id=key,
                        txid=tx_id,
                        vout=vout['n'],
                        receiving_address=receiving_address,
                        block_height=block_height,
                        block_datetime=datetime.utcfromtimestamp(block['time'])
                    )
                    db.add(transaction)
                    await db.commit()

async def get_masternode_transactions_by_block():
    masternode_transactions_by_block = {}
    try:
        async with db_code.Session() as db:
            transactions = await db.exec(select(db_code.MasternodeTransaction).where(db_code.MasternodeTransaction.moved == False))
            for transaction in transactions:
                block_height = transaction.block_height
                if block_height not in masternode_transactions_by_block:
                    masternode_transactions_by_block[block_height] = {
                        'count': 0,
                        'transactions': []
                    }
                masternode_transactions_by_block[block_height]['count'] += 1
                masternode_transactions_by_block[block_height]['transactions'].append({
                    'txid_vout': f"{transaction.txid}-{transaction.vout}",
                    'receiving_address': transaction.receiving_address
                })
        logger.info("Masternode transactions retrieved from the database.")
    except Exception as e:
        logger.warning(f"Failed to retrieve masternode transactions from the database: {e}")
        logger.info("Falling back to using dict/pickle files.")
        for file in os.listdir('masternode_transactions_dicts'):
            if file.startswith('masternode_txids__block_') and file.endswith('.pickle'):
                with open(os.path.join('masternode_transactions_dicts', file), 'rb') as f:
                    masternode_transactions_dict = pickle.load(f)
                for key, value in masternode_transactions_dict.items():
                    block_height = value['block_height']
                    if block_height not in masternode_transactions_by_block:
                        masternode_transactions_by_block[block_height] = {
                            'count': 0,
                            'transactions': []
                        }
                    masternode_transactions_by_block[block_height]['count'] += 1
                    masternode_transactions_by_block[block_height]['transactions'].append({
                        'txid_vout': key,
                        'receiving_address': value['receiving_address']
                    })
        with open('masternode_master_file.pickle', 'rb') as f:
            masternode_master_file = pickle.load(f)
        for key, value in masternode_master_file.items():
            if not value.get('moved', False):
                block_height = value['block_height']
                if block_height not in masternode_transactions_by_block:
                    masternode_transactions_by_block[block_height] = {
                        'count': 0,
                        'transactions': []
                    }
                masternode_transactions_by_block[block_height]['count'] += 1
                masternode_transactions_by_block[block_height]['transactions'].append({
                    'txid_vout': key,
                    'receiving_address': value['receiving_address']
                })
    return masternode_transactions_by_block

rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
network, burn_address = get_network_info(rpc_port)
rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
use_direct_ticket_scanning = 0

#_________________________________________________________________________________________________________

#Code for getting ticket data directly from the blockchain files without using the slow RPC methods

if use_direct_ticket_scanning:
    def deserialize_block(raw_block_bytes):
        f = io.BytesIO(raw_block_bytes)
        def read_bytes(n):
            data = f.read(n)
            if len(data) < n:
                raise ValueError(f"Not enough data to read {n} bytes")
            return data
        def read_varint():
            b = f.read(1)
            if not b:
                raise ValueError("Not enough data to read varint")
            n = int.from_bytes(b, 'little')
            if n < 0xfd:
                return n
            elif n == 0xfd:
                return int.from_bytes(read_bytes(2), 'little')
            elif n == 0xfe:
                return int.from_bytes(read_bytes(4), 'little')
            else:
                return int.from_bytes(read_bytes(8), 'little')
        try:
            version = struct.unpack('<I', read_bytes(4))[0]
            prev_block_hash = read_bytes(32)[::-1].hex()
            merkle_root = read_bytes(32)[::-1].hex()
            final_sapling_root = read_bytes(32)[::-1].hex()
            timestamp = struct.unpack('<I', read_bytes(4))[0]
            bits = struct.unpack('<I', read_bytes(4))[0]
            pastel_id_length = read_varint()
            pastel_id = read_bytes(pastel_id_length).decode('utf-8')
            signature_length = read_varint()
            signature = read_bytes(signature_length)
            nonce = read_bytes(32).hex()
            solution_length = len(raw_block_bytes) - f.tell()
            solution = read_bytes(solution_length).hex()
            num_transactions = read_varint()
            transactions = []
            for _ in range(num_transactions):
                transaction = deserialize_transaction(f)
                transactions.append(transaction)
            return {
                'version': version,
                'prev_block_hash': prev_block_hash,
                'merkle_root': merkle_root,
                'final_sapling_root': final_sapling_root,
                'timestamp': timestamp,
                'bits': bits,
                'pastel_id': pastel_id,
                'signature': signature,
                'nonce': nonce,
                'solution': solution,
                'num_transactions': num_transactions,
                'transactions': transactions
            }
        except (struct.error, ValueError) as e:
            print(f"Error deserializing block: {e}")
            return None
        
    def deserialize_transaction(f):
        def read_bytes(n):
            data = f.read(n)
            if len(data) < n:
                raise ValueError(f"Not enough data to read {n} bytes")
            return data
        def read_varint():
            b = f.read(1)
            if not b:
                raise ValueError("Not enough data to read varint")
            n = int.from_bytes(b, 'little')
            if n < 0xfd:
                return n
            elif n == 0xfd:
                return int.from_bytes(read_bytes(2), 'little')
            elif n == 0xfe:
                return int.from_bytes(read_bytes(4), 'little')
            else:
                return int.from_bytes(read_bytes(8), 'little')
        try:
            version = struct.unpack('<I', read_bytes(4))[0]
            version_group_id = struct.unpack('<I', read_bytes(4))[0]
            num_inputs = read_varint()
            vin = []
            for _ in range(num_inputs):
                prevout_hash = read_bytes(32)[::-1]
                prevout_n = struct.unpack('<I', read_bytes(4))[0]
                script_sig_length = read_varint()
                script_sig = read_bytes(script_sig_length)
                sequence = struct.unpack('<I', read_bytes(4))[0]
                vin.append((prevout_hash, prevout_n, script_sig, sequence))
            num_outputs = read_varint()
            vout = []
            for _ in range(num_outputs):
                value = struct.unpack('<Q', read_bytes(8))[0]
                script_pubkey_length = read_varint()
                script_pubkey = read_bytes(script_pubkey_length)
                vout.append((value, script_pubkey))
            lock_time = struct.unpack('<I', read_bytes(4))[0]
            expiry_height = struct.unpack('<I', read_bytes(4))[0]
            value_balance = struct.unpack('<Q', read_bytes(8))[0]
            num_shielded_spends = read_varint()
            num_shielded_outputs = read_varint()
            if num_shielded_outputs > 0:
                num_shielded_outputs = read_varint()
            return {
                'version': version,
                'version_group_id': version_group_id,
                'vin': vin,
                'vout': vout,
                'lock_time': lock_time,
                'expiry_height': expiry_height,
                'value_balance': value_balance,
                'num_shielded_spends': num_shielded_spends,
                'num_shielded_outputs': num_shielded_outputs,
            }
        except (struct.error, ValueError) as e:
            print(f"Error deserializing transaction: {e}")
            return None
        
    def get_block_files(network):
        pastel_dir = os.path.expanduser("~/.pastel")
        if network == "mainnet":
            blocks_dir = os.path.join(pastel_dir, "blocks")
        else:
            blocks_dir = os.path.join(pastel_dir, network, "blocks")
        return [os.path.join(blocks_dir, f) for f in os.listdir(blocks_dir) if f.startswith("blk") and f.endswith(".dat")], blocks_dir

    async def process_block_files(block_files):
        try:
            for block_file in block_files:
                with open(block_file, "rb") as f:
                    magic = f.read(4)
                    while magic == b'\xf9\xbe\xb4\xd9':
                        block_size = struct.unpack("<I", f.read(4))[0]
                        raw_block_bytes = f.read(block_size)
                        block = deserialize_block(raw_block_bytes)
                        if block is not None:
                            for transaction in block['transactions']:
                                txid = transaction['txid']
                                await process_transaction(transaction, txid)
                        magic = f.read(4)
        except Exception as e:
            logger.error(f"Error occurred while processing block files: {e}")
            traceback.print_exc()

    class CDiskTxPos:
        def __init__(self, nFile=0, nPos=0, nTxSize=0):
            self.nFile = nFile
            self.nPos = nPos
            self.nTxSize = nTxSize

        def serialize(self):
            return struct.pack("<iii", self.nFile, self.nPos, self.nTxSize)

        @classmethod
        def deserialize(cls, raw_data):
            nFile, nPos, nTxSize = struct.unpack("<iii", raw_data)
            return cls(nFile, nPos, nTxSize)

        def __repr__(self):
            return f"CDiskTxPos(nFile={self.nFile}, nPos={self.nPos}, nTxSize={self.nTxSize})"

    def decode_compactsize(raw_hex):
        """Decodes the compactsize encoding used in the UTXO data"""
        if len(raw_hex) == 0:
            return 0, 0
        first_byte = raw_hex[0]
        if first_byte < 253:
            return first_byte, 1
        elif first_byte == 253:
            return struct.unpack("<H", raw_hex[1:3])[0], 3
        elif first_byte == 254:
            return struct.unpack("<I", raw_hex[1:5])[0], 5
        else:
            return struct.unpack("<Q", raw_hex[1:9])[0], 9

    def extract_tx_pos(raw_utxo_data):
        try:
            # Extract the transaction position from the UTXO data
            # Assuming the UTXO data has the following structure:
            # [value (8 bytes)][script length (varint)][script (variable length)]
            value_hex = raw_utxo_data[:8]
            script_length, varint_size = decode_compactsize(raw_utxo_data[8:])
            script_start = 8 + varint_size
            script_hex = raw_utxo_data[script_start:script_start+script_length]
            return value_hex, script_hex
        except Exception as e:
            logger.error(f"Error occurred while extracting UTXO data: {e}")
            traceback.print_exc()
            return None, None
        
    def get_block_file_path(block_file_index, blocks_dir):
        # Generate the block file path based on the block file index
        # Assuming the block files are named in the format "blk00000.dat", "blk00001.dat", etc.
        block_file_name = f"blk{block_file_index:05d}.dat"
        block_file_path = os.path.join(blocks_dir, block_file_name)
        return block_file_path

    def is_data_storage_script(script_hex):
        # Check if the script matches the pattern used for storing arbitrary data
        # Assuming the data storage script follows the format:
        # OP_RETURN <data> (where <data> contains the identifier "CREDIT_PACK_STORAGE_TICKET")
        if script_hex.startswith('6a'):  # Check if the script starts with OP_RETURN (0x6a)
            data = script_hex[2:]  # Extract the data part of the script
            if 'CREDIT_PACK_STORAGE_TICKET' in bytes.fromhex(data).decode('utf-8', errors='ignore'):
                return True
        return False
            
    def retrieve_raw_transaction_bytes(txid, raw_utxo_data):
        try:
            # Extract the transaction position from the UTXO data
            tx_pos = extract_tx_pos(raw_utxo_data)
            if tx_pos is None:
                logger.error(f"Failed to extract transaction position for UTXO {txid}")
                return None
            # Open the block file and seek to the transaction position
            block_file_path = get_block_file_path(tx_pos.nFile)
            with open(block_file_path, "rb") as block_file:
                block_file.seek(tx_pos.nPos)
                # Read the block header
                block_header_bytes = block_file.read(80)  # Assuming a fixed block header size of 80 bytes
                # Extract the transaction offset from the block header
                tx_offset = struct.unpack("<I", block_header_bytes[-4:])[0]
                # Seek to the transaction position within the block
                block_file.seek(tx_pos.nPos + tx_offset)
                # Read the transaction bytes
                raw_transaction_bytes = block_file.read(tx_pos.nTxSize)
            return raw_transaction_bytes
        except Exception as e:
            logger.error(f"Error occurred while retrieving raw transaction bytes for UTXO {txid}: {e}")
            traceback.print_exc()
            return None

    async def process_utxo(txid, raw_utxo_data):
        try:
            raw_transaction_bytes = await retrieve_raw_transaction_bytes(txid, raw_utxo_data)
            if raw_transaction_bytes is not None:
                transaction = deserialize_transaction(raw_transaction_bytes)
                if transaction is not None:
                    await attempt_to_reconstruct_data_from_raw_transaction_data(transaction)
        except Exception as e:
            logger.error(f"Error occurred while processing UTXO {txid}: {e}")
            traceback.print_exc()
            
    async def process_transaction(transaction, txid):
        try:
            for output in transaction['vout']:
                script_pubkey = output['scriptPubKey']
                if is_data_storage_script(script_pubkey):
                    await process_utxo(txid, script_pubkey)
        except Exception as e:
            logger.error(f"Error occurred while processing transaction {txid}: {e}")
            traceback.print_exc()

    async def process_leveldb_files(leveldb_dir):
        try:
            db = plyvel.DB(leveldb_dir)
            for txid_raw, raw_utxo_data in db.iterator():
                txid_hex = txid_raw.hex()[::-1]
                await process_utxo(txid_hex, raw_utxo_data)
        except Exception as e:
            logger.error(f"Error occurred while processing LevelDB files: {e}")
            traceback.print_exc()
        finally:
            db.close()
        
    def get_leveldb_files(network):
        pastel_dir = os.path.expanduser("~/.pastel")
        if network == "mainnet":
            chainstate_dir = os.path.join(pastel_dir, "chainstate")
        else:
            chainstate_dir = os.path.join(pastel_dir, network, "chainstate")
        # Create a temporary directory to store the copied LevelDB files
        temp_dir = tempfile.mkdtemp()
        # Copy the entire chainstate directory to the temporary directory
        dst_chainstate_dir = os.path.join(temp_dir, "chainstate")
        logger.info(f"Making a copy of the leveldb files so we can read them; about to copy {sum(os.path.getsize(os.path.join(chainstate_dir, f)) for f in os.listdir(chainstate_dir))/1024/1024} mb of data")
        shutil.copytree(chainstate_dir, dst_chainstate_dir)
        # Remove the LOCK file from the copied directory
        logger.info("Done copying leveldb files to temp directory; now removing copied LOCK file...")
        lock_file_path = os.path.join(dst_chainstate_dir, "LOCK")
        if os.path.exists(lock_file_path):
            os.remove(lock_file_path)
        return dst_chainstate_dir

    async def attempt_to_reconstruct_data_from_raw_transaction_data(raw_transaction_bytes):
        try:
            transaction = deserialize_transaction(raw_transaction_bytes)
            outputs = transaction['vout']  # Extract outputs from the transaction
            # Concatenate all scriptPubKey hex strings excluding the last two (change and receiving address outputs)
            encoded_hex_data = ''.join(output['scriptPubKey']['hex'][4:-4] for output in outputs[:-2])
            # Decode the hex data to bytes and clean up extraneous characters and padding
            reconstructed_combined_data = bytes.fromhex(encoded_hex_data)
            reconstructed_combined_data_cleaned = reconstructed_combined_data.decode('utf-8').replace("A", "").rstrip("\x00")
            data_buffer = bytes.fromhex(reconstructed_combined_data_cleaned)
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

    async def save_credit_pack_purchase_request_response(credit_pack_purchase_request_response: db_code.CreditPackPurchaseRequestResponse) -> None:
        try:
            async with db_code.Session() as db_session:
                db_session.add(credit_pack_purchase_request_response)
                await db_session.commit()
        except Exception as e:
            logger.error(f"Error saving credit pack purchase request response: {str(e)}")
            raise
                
    async def background_monitor(block_files, leveldb_dir):
        tasks = [
            asyncio.create_task(process_block_files(block_files)),
            asyncio.create_task(process_leveldb_files(leveldb_dir))
        ]
        await asyncio.gather(*tasks)

    def main():
        block_files, blocks_dir = get_block_files(network)
        leveldb_dir = get_leveldb_files(network)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(background_monitor(block_files, leveldb_dir))
        finally:
            if os.path.exists(leveldb_dir):
                shutil.rmtree(leveldb_dir)

    if __name__ == "__main__":
        main()