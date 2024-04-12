import binascii
import struct
import hashlib
import os
import io
import asyncio
import base64
import json
from functools import lru_cache
import urllib.parse as urlparse
import heapq
from decimal import Decimal
from binascii import hexlify
import zstandard as zstd
import httpx
from logger_config import setup_logger
logger = setup_logger()

def unhexstr(str):
    return binascii.unhexlify(str.encode('utf8'))

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
    max_concurrent_requests = 5000
    _semaphore = asyncio.BoundedSemaphore(max_concurrent_requests)
    def __init__(self, service_url, service_name=None, reconnect_timeout=15, reconnect_amount=2, request_timeout=20):
        self.service_url = service_url
        self.service_name = service_name
        self.url = urlparse.urlparse(service_url)        
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
                        logger.info(f"Reconnect try #{i+1}")
                        sleep_time = self.reconnect_timeout * (2 ** i)
                        logger.info(f"Waiting for {sleep_time} seconds before retrying.")
                        await asyncio.sleep(sleep_time)
                    async with httpx.AsyncClient() as client:
                        response = await client.post(self.service_url.replace("http://", "https://"), headers=headers, data=postdata)
                    break
                except Exception as e:
                    logger.info(f"Error occurred in __call__: {e}")
                    err_msg = f"Failed to connect to {self.url.hostname}:{self.url.port}"
                    rtm = self.reconnect_timeout
                    if rtm:
                        err_msg += f". Waiting {rtm} seconds."
                    logger.error(err_msg)
            else:
                logger.error("Reconnect tries exceeded.")
                raise Exception("Failed to connect to the server")
            response_json = response.json()
            if response_json['error'] is not None:
                raise JSONRPCException(response_json['error'])
            elif 'result' not in response_json:
                raise JSONRPCException({
                    'code': -343, 'message': 'missing JSON-RPC result'})
            else:
                return response_json['result']

class TxWrapper:
    def __init__(self, tx):
        self.tx = tx
        
    def __lt__(self, other):
        if self.tx['amount'] != other.tx['amount']:
            return self.tx['amount'] > other.tx['amount']
        else:
            return self.tx['confirmations'] < other.tx['confirmations']
        
class BlockchainUTXOStorage:
    def __init__(self, rpc_user, rpc_password, rpc_port, base_transaction_amount, fee_per_kb):
        self.rpc_user = rpc_user
        self.rpc_password = rpc_password
        self.rpc_port = rpc_port
        self.base_transaction_amount = base_transaction_amount
        self.fee_per_kb = fee_per_kb
        self.coin = 100000000       
        self.op_checksig = b'\xac'
        self.op_checkmultisig = b'\xae'
        self.op_pushdata1 = b'\x4c'
        self.op_dup = b'\x76'
        self.op_hash160 = b'\xa9'
        self.op_equalverify = b'\x88'
        self.op_return = b'\x6a'
        self.rpc_connection_string = f'https://{self.rpc_user}:{self.rpc_password}@127.0.0.1:{self.rpc_port}'
        self.rpc_connection = AsyncAuthServiceProxy(self.rpc_connection_string)


    def get_sha256_hash(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        return hashlib.sha3_256(input_data).hexdigest()

    def get_raw_sha256_hash(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        return hashlib.sha3_256(input_data).digest()
    
    async def send_transaction(self, signed_tx):
        try:
            txid = await self.rpc_connection.sendrawtransaction(signed_tx)
            return txid
        except Exception as e:
            logger.error(f"Error occurred while sending transaction: {e}")
            raise
    
    @lru_cache(maxsize=128)
    async def get_unspent_transactions(self):
        return await self.rpc_connection.listunspent()
    
    async def select_txins(self, value):
        unspent = await self.get_unspent_transactions()
        spendable_unspent = [tx for tx in unspent if tx['spendable']]
        if value == 0:
            total_amount = sum(tx['amount'] for tx in spendable_unspent)
            return spendable_unspent, total_amount
        else:
            heap = [TxWrapper(tx) for tx in spendable_unspent]
            heapq.heapify(heap)
            selected_txins = []
            total_amount = 0
            while heap and total_amount < value:
                tx_wrapper = heapq.heappop(heap)
                selected_txins.append(tx_wrapper.tx)
                total_amount += tx_wrapper.tx['amount']
            if total_amount < value:
                raise Exception("Insufficient funds")
            else:
                return selected_txins, total_amount
        
    def varint(self, n):
        if n < 0xfd:
            return bytes([n])
        elif n < 0xffff:
            return b'\xfd' + struct.pack('<H', n)
        else:
            assert False

    def packtxin(self, prevout, scriptSig, seq=0xffffffff):
        return prevout[0][::-1] + struct.pack('<L', prevout[1]) + self.varint(len(scriptSig)) + scriptSig + struct.pack('<L', seq)

    def packtxout(self, value, scriptPubKey):
        return struct.pack('<Q', int(value * self.coin)) + self.varint(len(scriptPubKey)) + scriptPubKey
    
    def packtx(self, txins, txouts, locktime=0, version=1):
        r = struct.pack('<L', version)  # Transaction version (4 bytes)
        r += self.varint(len(txins))
        for txin in txins:
            r += self.packtxin((unhexstr(txin['txid']), txin['vout']), b'', 0xffffffff)  # Updated sequence number to 4 bytes
        r += self.varint(len(txouts))
        for (value, scriptPubKey) in txouts:
            r += self.packtxout(value, scriptPubKey)
        r += struct.pack('<L', locktime)  # Lock time (4 bytes)
        return r    

    def pushdata(self, data):
        assert len(data) < self.op_pushdata1[0]
        return bytes([len(data)]) + data

    def pushint(self, n):
        assert 0 < n <= 16
        return bytes([0x51 + n - 1])

    def addr2bytes(self, s):
        if len(s) < 26 or len(s) > 35:
            raise ValueError("Invalid address length")
        allowed_chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
        if any(c not in allowed_chars for c in s):
            raise ValueError("Invalid characters in address")
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
        return unhexstr(h)[1:-4]

    def checkmultisig_scriptPubKey_dump(self, fd):
        data = fd.read(65 * 3)
        if not data:
            return None
        r = self.pushint(1)
        n = 0
        while data:
            chunk = data[0:65]
            data = data[65:]
            if len(chunk) < 33:
                chunk += b'\x00' * (33 - len(chunk))
            elif len(chunk) < 65:
                chunk += b'\x00' * (65 - len(chunk))
            r += self.pushdata(chunk)
            n += 1
        r += self.pushint(n) + self.op_checkmultisig
        return r

    def compress_data(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        zstd_compression_level = 22
        zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
        zstd_compressed_data = zstandard_compressor.compress(input_data)
        return zstd_compressed_data
        
    def decompress_data(self, compressed_data):
        return zstd.decompress(compressed_data)

    async def get_max_data_size(self):
        # Define the transaction and output overheads
        TX_OVERHEAD = 30  # Transaction overhead in bytes (version, locktime, masternode payment)
        INPUT_OVERHEAD = 180  # Input overhead in bytes (txid, vout, scriptSig, sequence)
        OUTPUT_OVERHEAD = 34  # Output overhead in bytes (value, scriptPubKey)
        OP_RETURN_OVERHEAD = 8  # OP_RETURN output overhead in bytes (value, scriptPubKey)
        FAKE_MULTISIG_OVERHEAD = 33  # Overhead for each fake multisig output (pubkey size)
        # Legacy transactions have a hard cap on size at 100kb; to be safe, we don't exceed 50kb
        MAX_TX_SIZE = 50000
        # Calculate the maximum data size based on the transaction structure
        max_inputs = 1  # Assuming a single input transaction
        # Calculate the available space for data in the transaction
        available_space = MAX_TX_SIZE - TX_OVERHEAD - (max_inputs * INPUT_OVERHEAD) - OP_RETURN_OVERHEAD
        # Calculate the number of fake multisig outputs that can fit in the available space
        max_fake_multisig_outputs = (available_space - OUTPUT_OVERHEAD) // (FAKE_MULTISIG_OVERHEAD + OUTPUT_OVERHEAD)
        # Calculate the total size of the fake multisig outputs
        fake_multisig_data_size = max_fake_multisig_outputs * FAKE_MULTISIG_OVERHEAD
        # Reserve space for the OP_RETURN output
        op_return_data_size = available_space - (max_fake_multisig_outputs * OUTPUT_OVERHEAD) - len(self.op_return)
        # Reserve space for the uncompressed data hash and compressed data hash
        hash_size = len(self.get_raw_sha256_hash(b''))
        op_return_data_size -= 2 * hash_size
        # Reserve space for the data length prefix (2 bytes)
        op_return_data_size -= 2
        total_max_size = op_return_data_size + fake_multisig_data_size
        return total_max_size
    
    def calculate_transaction_fee(self, signed_tx):
        # Calculate the actual virtual size of the transaction
        tx_size = len(signed_tx['hex']) / 2  # Convert bytes to virtual size
        fee = Decimal(tx_size) * self.fee_per_kb / 1000  # Calculate fee based on virtual size
        return fee    
    
    async def create_and_send_transaction(self, txins, txouts, fee=None):
        tx = self.packtx(txins, txouts)
        hex_transaction = hexlify(tx).decode('utf-8')
        signed_tx = await self.rpc_connection.signrawtransaction(hex_transaction)
        if not signed_tx['complete']:
            logger.error("Failed to sign all transaction inputs")
            return None
        if fee is None:
            fee = self.calculate_transaction_fee(signed_tx)
        txouts[-1][0] -= fee
        final_tx = self.packtx(txins, txouts)
        signed_tx = await self.rpc_connection.signrawtransaction(hexlify(final_tx).decode('utf-8'))
        assert signed_tx['complete']
        hex_signed_transaction = signed_tx['hex']
        await self.rpc_connection.lockunspent(False, [{"txid": txin["txid"], "vout": txin["vout"]} for txin in txins])
        try:
            send_raw_transaction_result = await self.send_transaction(hex_signed_transaction)
            return send_raw_transaction_result, fee
        except Exception as e:
            logger.error(f"Error occurred while sending transaction: {e}")
            await self.rpc_connection.lockunspent(True, [{"txid": txin["txid"], "vout": txin["vout"]} for txin in txins])
            return None
        finally:
            await self.rpc_connection.lockunspent(True, [{"txid": txin["txid"], "vout": txin["vout"]} for txin in txins])

    async def add_output_transactions(self, change, txouts):
        out_value = self.base_transaction_amount
        change -= out_value
        receiving_address = await self.rpc_connection.getnewaddress()
        txouts.append((out_value, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(receiving_address)) + self.op_equalverify + self.op_checksig))
        change_address = await self.rpc_connection.getnewaddress()
        txouts.append((change, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(change_address)) + self.op_equalverify + self.op_checksig))
        return change
    
    async def prepare_txins_and_change(self):
        txins, change = await self.select_txins(0)
        if txins is None or change is None:
            logger.error("Insufficient funds to store the data")
            return None, None
        return txins, Decimal(change)

    async def prepare_txouts_and_change(self, txins, change, combined_data, txouts):
        if txins is None or change is None:
            return None
        fd = io.BytesIO(combined_data)
        while True:
            scriptPubKey = self.checkmultisig_scriptPubKey_dump(fd)
            if scriptPubKey is None:
                break
            value = self.base_transaction_amount
            txouts.append((value, scriptPubKey))
            change -= value
        return await self.add_output_transactions(change, txouts)

    async def process_chunk(self, chunk, uncompressed_data_hash, compressed_data_hash):
        txins, change = await self.prepare_txins_and_change()
        if txins is None or change is None:
            return None
        txouts = []
        combined_data = len(chunk).to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash + chunk
        change = await self.prepare_txouts_and_change(txins, change, combined_data, txouts)
        return await self.create_and_send_transaction(txins, txouts)

    async def store_data_chunks(self, chunks, uncompressed_data_hash, compressed_data_hash):
        tasks = [self.process_chunk(chunk, uncompressed_data_hash, compressed_data_hash) for chunk in chunks]
        results = await asyncio.gather(*tasks)
        transaction_ids = [result[0] for result in results if result]
        total_fee = sum(result[1] for result in results if result)
        if None in results:
            return None
        return transaction_ids, total_fee

    async def store_first_chunk(self, first_chunk, transaction_ids):
        txins, change = await self.prepare_txins_and_change()
        if txins is None or change is None:
            return None
        txouts = []
        op_return_data = '|'.join(transaction_ids).encode()
        script_pubkey = self.op_return + self.pushdata(op_return_data) + self.pushdata(first_chunk)
        txouts.append((0, script_pubkey))
        change = await self.prepare_txouts_and_change(txins, change, b'', txouts)
        return await self.create_and_send_tx(txins, txouts)

    async def store_single_chunk(self, chunk, uncompressed_data_hash, compressed_data_hash):
        txins, change = await self.prepare_txins_and_change()
        if txins is None or change is None:
            return None
        txouts = []
        combined_data = len(chunk).to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash + chunk
        change = await self.prepare_txouts_and_change(txins, change, combined_data, txouts)
        return await self.create_and_send_tx(txins, txouts)

    async def store_data(self, input_data):
        compressed_data = self.compress_data(input_data)
        uncompressed_data_hash = self.get_raw_sha256_hash(input_data)
        compressed_data_hash = self.get_raw_sha256_hash(compressed_data)
        max_data_size = await self.get_max_data_size()
        if len(compressed_data) <= max_data_size:
            return await self.store_single_chunk(compressed_data, uncompressed_data_hash, compressed_data_hash)
        else:
            chunk_size = max_data_size
            chunks = [compressed_data[i:i + chunk_size] for i in range(0, len(compressed_data), chunk_size)]
            transaction_ids, total_fee = await self.store_data_chunks(chunks[1:], uncompressed_data_hash, compressed_data_hash)
            if transaction_ids is None:
                return None
            first_txid, first_chunk_fee = await self.store_first_chunk(chunks[0], transaction_ids)
            if first_txid is None:
                return None
            total_fee += first_chunk_fee
            logger.info(f"Data stored successfully in the blockchain. First transaction ID: {first_txid}. Total fee: {total_fee} PSL")
            return first_txid
                
    async def retrieve_data_chunk(self, txid):
        try: 
            try:
                raw_transaction = await self.rpc_connection.getrawtransaction(txid)
            except JSONRPCException as e:
                if e.code == -5:  # -5 indicates "No information available about transaction"
                    logger.error(f"Transaction {txid} not found")
                    return None
                else:
                    raise            
            decoded_transaction = await self.rpc_connection.decoderawtransaction(raw_transaction) # Decode the raw transaction
            data_chunks = [] # Extract the data from the transaction outputs
            for output in decoded_transaction['vout']:
                if 'asm' in output['scriptPubKey']:
                    script_asm = output['scriptPubKey']['asm']
                    if 'OP_CHECKMULTISIG' in script_asm:
                        data_chunk = unhexstr(output['scriptPubKey']['hex'][2:-2])
                        data_chunks.append(data_chunk)
            if not data_chunks:
                logger.error(f"No data chunks found in transaction {txid}")
                return None
            combined_data = b''.join(data_chunks) # Combine the data chunks
            return combined_data
        except JSONRPCException as e:
            logger.error(f"Error occurred while retrieving data chunk: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return None
                            
    async def retrieve_data(self, txid):
        try:
            try:
                raw_transaction = await self.rpc_connection.getrawtransaction(txid)
            except JSONRPCException as e:
                if e.code == -5:  # -5 indicates "No information available about transaction"
                    logger.error(f"Transaction {txid} not found")
                    return None
                else:
                    raise
            decoded_transaction = await self.rpc_connection.decoderawtransaction(raw_transaction)
            op_return_output = None
            for output in decoded_transaction['vout']:
                if 'OP_RETURN' in output['scriptPubKey']['asm']:
                    op_return_output = output
                    break
            if op_return_output is None:
                logger.error("No OP_RETURN output found in the transaction")
                return None
            op_return_data = unhexstr(op_return_output['scriptPubKey']['hex'][4:])
            if b'|' in op_return_data:
                transaction_ids = op_return_data.split(b'|')
                first_chunk = op_return_data[op_return_data.rfind(b'|') + 1:]
                data_chunks = [first_chunk]
                async def retrieve_chunk(tx_id):
                    chunk = await self.retrieve_data_chunk(tx_id.decode('utf-8'))
                    if chunk is None:
                        logger.error(f"Failed to retrieve data chunk from transaction {tx_id.decode('utf-8')}")
                    return chunk
                tasks = [retrieve_chunk(tx_id) for tx_id in transaction_ids]
                results = await asyncio.gather(*tasks)
                if None in results:
                    return None
                data_chunks.extend(results)
                combined_data = b''.join(data_chunks)
            else:
                # The data is stored in a single transaction
                data_chunks = []
                for output in decoded_transaction['vout']:
                    if 'asm' in output['scriptPubKey'] and 'OP_CHECKMULTISIG' in output['scriptPubKey']['asm']:
                        data_chunk = unhexstr(output['scriptPubKey']['hex'][2:-2])
                        data_chunks.append(data_chunk)
                if not data_chunks:
                    combined_data = op_return_data
                else:
                    combined_data = b''.join(data_chunks)
            # Extract the uncompressed data hash, compressed data hash, and compressed data
            uncompressed_data_length = int.from_bytes(combined_data[:2], 'big')
            uncompressed_data_hash = combined_data[2:34]
            compressed_data_hash = combined_data[34:66]
            compressed_data = combined_data[66:]
            # Verify the integrity of the compressed data
            if self.get_raw_sha256_hash(compressed_data) != compressed_data_hash:
                logger.error("Compressed data hash verification failed")
                return None
            decompressed_data = self.decompress_data(compressed_data)
            # Verify the integrity of the decompressed data
            if self.get_raw_sha256_hash(decompressed_data) != uncompressed_data_hash:
                logger.error("Uncompressed data hash verification failed")
                return None
            logger.info(f"Data retrieved successfully from transaction {txid}; Length: {len(decompressed_data)} bytes, which matches the original data size ({uncompressed_data_length} bytes); SHA3-256 hash of the data ({self.get_sha256_hash(decompressed_data)}) also matches the original data hash")
            return decompressed_data
        except JSONRPCException as e:
            logger.error(f"Error occurred while retrieving data: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
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

use_demonstrate_blockchain_data_storage = 0
if use_demonstrate_blockchain_data_storage:                
    # Usage example:
    async def main():
        rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
        base_transaction_amount = Decimal(0.000001)
        fee_per_kb = Decimal(0.0001)
        storage = BlockchainUTXOStorage(rpc_user, rpc_password, rpc_port, base_transaction_amount, fee_per_kb)
        input_data = 'Your data to store in the blockchain'
        transaction_id = await storage.store_data(input_data)
        retrieved_data = await storage.retrieve_data(transaction_id)
        logger.info('Retrieved data:', retrieved_data.decode('utf-8'))

    if __name__ == '__main__':
        asyncio.run(main())