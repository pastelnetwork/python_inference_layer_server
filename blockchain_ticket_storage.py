import binascii
import struct
import hashlib
import os
import asyncio
import base64
import json
from functools import lru_cache
import urllib.parse as urlparse
import heapq
from decimal import Decimal
from binascii import hexlify
import zstandard as zstd
from httpx import AsyncClient, Limits, Timeout
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
    def __init__(self, service_url, service_name=None, reconnect_timeout=25, reconnect_amount=2, request_timeout=20):
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

class TxWrapper:
    def __init__(self, tx):
        self.tx = tx
        
    def __lt__(self, other):
        if self.tx['amount'] != other.tx['amount']:
            return self.tx['amount'] > other.tx['amount']
        else:
            return self.tx['confirmations'] < other.tx['confirmations']

@lru_cache(maxsize=128)
async def get_unspent_transactions(rpc_connection):
    return await rpc_connection.listunspent()

async def select_txins(value, rpc_connection):
    unspent = await get_unspent_transactions(rpc_connection)
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
        
class BlockchainUTXOStorage:
    def __init__(self, rpc_user: str, rpc_password: str, rpc_port: int, fee_per_kb: Decimal):
        self.rpc_user = rpc_user
        self.rpc_password = rpc_password
        self.rpc_port = rpc_port
        self.fee_per_kb = fee_per_kb
        self.coin = 100000
        self.op_return = b'\x6a'
        self.rpc_connection_string = f'http://{self.rpc_user}:{self.rpc_password}@127.0.0.1:{self.rpc_port}'
        self.rpc_connection = AsyncAuthServiceProxy(self.rpc_connection_string)
        self.op_return_max_size = 80  # Maximum size of OP_RETURN data in bytes

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
        assert len(data) < 76  # OP_PUSHDATA1
        return bytes([len(data)]) + data

    def compress_data(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        zstd_compression_level = 22
        zstandard_compressor = zstd.ZstdCompressor(level=zstd_compression_level, write_content_size=True, write_checksum=True)
        zstd_compressed_data = zstandard_compressor.compress(input_data)
        return zstd_compressed_data

    def decompress_data(self, compressed_data):
        return zstd.decompress(compressed_data)

    def calculate_transaction_fee(self, signed_tx):
        # Calculate the actual virtual size of the transaction
        tx_size = len(signed_tx['hex']) / 2  # Convert bytes to virtual size
        fee = Decimal(tx_size) * self.fee_per_kb / 1000  # Calculate fee based on virtual size
        return Decimal(fee)

    async def create_and_send_transaction(self, txins, txouts, fee=None):
        tx = self.packtx(txins, txouts)
        hex_transaction = hexlify(tx).decode('utf-8')
        signed_tx = await self.rpc_connection.signrawtransaction(hex_transaction)
        if not signed_tx['complete']:
            logger.error("Failed to sign all transaction inputs")
            return None
        if fee is None:
            fee = self.calculate_transaction_fee(signed_tx)
        txouts[-1] = (txouts[-1][0] - fee, txouts[-1][1])  # Subtract fee from the change output
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
        change_address = await self.rpc_connection.getnewaddress()
        txouts.append((change, self.pushdata(self.get_raw_sha256_hash(change_address.encode())) + b'\x88\xac'))
        return change

    async def prepare_txins_and_change(self):
        txins, change = await select_txins(0, self.rpc_connection)
        if txins is None or change is None:
            logger.error("Insufficient funds to store the data")
            return None, None
        return txins, Decimal(change)

    async def store_data_chunk(self, chunk):
        txins, change = await self.prepare_txins_and_change()
        if txins is None or change is None:
            logger.error("Insufficient funds to store the data")
            return None
        txouts = [(0, self.op_return + self.pushdata(chunk))]
        change = await self.add_output_transactions(change, txouts)
        result = await self.create_and_send_transaction(txins, txouts)
        if result:
            transaction_id, fee = result
            return transaction_id
        return None

    async def store_data_chunks(self, chunks):
        tasks = [asyncio.ensure_future(self.store_data_chunk(chunk)) for chunk in chunks]
        chunk_storage_txids = await asyncio.gather(*tasks)
        return chunk_storage_txids

    async def store_indexing_txids(self, indexing_txids):
        head_indexing_txid = None
        tasks = []
        for i in range(len(indexing_txids)):
            txid_chunk = indexing_txids[i].encode()
            next_indexing_txid = indexing_txids[i+1] if i+1 < len(indexing_txids) else None
            metadata = struct.pack('>HH', len(indexing_txids), i)  # Pack total chunks and current index as metadata
            txid_chunk += metadata
            task = asyncio.ensure_future(self.store_indexing_txid(txid_chunk, next_indexing_txid))
            tasks.append(task)
            if not head_indexing_txid:
                head_indexing_txid = await task
        await asyncio.gather(*tasks)
        return head_indexing_txid

    async def store_data(self, input_data):
        compressed_data = self.compress_data(input_data)
        uncompressed_data_hash = self.get_raw_sha256_hash(input_data)
        compressed_data_hash = self.get_raw_sha256_hash(compressed_data)
        uncompressed_data_length = len(input_data)
        chunk_size = self.op_return_max_size - 66  # Subtract space for hashes and length
        chunks = [
            uncompressed_data_length.to_bytes(2, 'big') +
            uncompressed_data_hash +
            compressed_data_hash +
            compressed_data[i:i+chunk_size]
            for i in range(0, len(compressed_data), chunk_size)
        ]
        chunk_storage_task = asyncio.ensure_future(self.store_data_chunks(chunks))
        chunk_storage_txids = await chunk_storage_task
        if None in chunk_storage_txids:
            return None
        indexing_task = asyncio.ensure_future(self.store_indexing_txids(chunk_storage_txids))
        head_indexing_txid = await indexing_task
        if head_indexing_txid is None:
            return None
        logger.info(f"Data stored successfully in the blockchain. Head indexing TXID: {head_indexing_txid}")
        return head_indexing_txid

    async def retrieve_data(self, head_indexing_txid):
        indexing_txids = []
        current_txid = head_indexing_txid
        while True:
            raw_transaction = await self.rpc_connection.getrawtransaction(current_txid)
            decoded_transaction = await self.rpc_connection.decoderawtransaction(raw_transaction)
            found_next_txid = False
            for output in decoded_transaction['vout']:
                if 'OP_RETURN' in output['scriptPubKey']['asm']:
                    op_return_data = unhexstr(output['scriptPubKey']['hex'][4:])
                    if len(op_return_data) <= 40:  # Indexing TXID chunk
                        total_chunks, current_index = struct.unpack('>HH', op_return_data[-4:])  # Unpack metadata
                        indexing_txids.append(op_return_data[:-4].decode())  # Append TXID without metadata
                    else:  # Next indexing TXID
                        current_txid = op_return_data.decode()
                        found_next_txid = True
                        break
            if not found_next_txid:
                break
        chunk_storage_txids = indexing_txids
        tasks = [asyncio.ensure_future(self.retrieve_chunk(txid)) for txid in chunk_storage_txids]
        data_chunks = await asyncio.gather(*tasks)
        combined_data = b''.join(data_chunks)
        uncompressed_data_hash = combined_data[2:34]
        compressed_data_hash = combined_data[34:66]
        compressed_data = combined_data[66:]
        if self.get_raw_sha256_hash(compressed_data) != compressed_data_hash:
            logger.error("Compressed data hash verification failed")
            return None
        decompressed_data = self.decompress_data(compressed_data)
        if self.get_raw_sha256_hash(decompressed_data) != uncompressed_data_hash:
            logger.error("Uncompressed data hash verification failed")
            return None
        logger.info(f"Data retrieved successfully from the blockchain. Length: {len(decompressed_data)} bytes")
        return decompressed_data

    async def retrieve_chunk(self, txid):
        raw_transaction = await self.rpc_connection.getrawtransaction(txid)
        decoded_transaction = await self.rpc_connection.decoderawtransaction(raw_transaction)
        for output in decoded_transaction['vout']:
            if 'OP_RETURN' in output['scriptPubKey']['asm']:
                op_return_data = unhexstr(output['scriptPubKey']['hex'][4:])
                if len(op_return_data) > 40:  # Data chunk
                    return op_return_data[66:]  # Skip hashes and length
                
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
        fee_per_kb = Decimal(0.0001)
        storage = BlockchainUTXOStorage(rpc_user, rpc_password, rpc_port, fee_per_kb)
        input_data = 'Your data to store in the blockchain'
        store_task = asyncio.ensure_future(storage.store_data(input_data))
        transaction_id = await store_task
        retrieve_task = asyncio.ensure_future(storage.retrieve_data(transaction_id))
        retrieved_data = await retrieve_task
        logger.info('Retrieved data:', retrieved_data.decode('utf-8'))

    if __name__ == '__main__':
        asyncio.run(main())