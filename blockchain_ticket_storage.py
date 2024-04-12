import binascii
import struct
import hashlib
import sys
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
                        response = await client.post(self.service_url, headers=headers, data=postdata)
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
        self.rpc_connection_string = f'http://{self.rpc_user}:{self.rpc_password}@127.0.0.1:{self.rpc_port}'
        self.rpc_connection = AsyncAuthServiceProxy(self.rpc_connection_string)

    def get_sha256_hash(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        return hashlib.sha3_256(input_data).hexdigest()

    def get_raw_sha256_hash(self, input_data):
        if isinstance(input_data, str):
            input_data = input_data.encode('utf-8')
        return hashlib.sha3_256(input_data).digest()

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
                return None
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

    def packtx(self, txins, txouts, locktime=0):
        r = b'\x01\x00\x00\x00'
        r += self.varint(len(txins))
        for txin in txins:
            r += self.packtxin((unhexstr(txin['txid']), txin['vout']), b'')
        r += self.varint(len(txouts))
        for (value, scriptPubKey) in txouts:
            r += self.packtxout(value, scriptPubKey)
        r += struct.pack('<L', locktime)
        return r

    def pushdata(self, data):
        assert len(data) < self.op_pushdata1[0]
        return bytes([len(data)]) + data

    def pushint(self, n):
        assert 0 < n <= 16
        return bytes([0x51 + n - 1])

    def addr2bytes(self, s):
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

async def store_data(self, input_data, chunk_size=50000):
    try:
        compressed_data = self.compress_data(input_data)
        uncompressed_data_hash = self.get_raw_sha256_hash(input_data)
        compressed_data_hash = self.get_raw_sha256_hash(compressed_data)
        chunks = [compressed_data[i:i+chunk_size] for i in range(0, len(compressed_data), chunk_size)] # Split the compressed data into chunks
        num_chunks = len(chunks)
        logger.info(f"Storing the ticket data in chunks with maximum size of {chunk_size} bytes; totaly compressed size is {len(compressed_data)} bytes, resulting in {num_chunks} chunks")
        transaction_ids, other_chunks_fee = await self.store_data_chunks(chunks[1:], uncompressed_data_hash, compressed_data_hash)
        first_txid, first_chunk_fee = await self.store_first_chunk(chunks[0], transaction_ids)
        combined_total_fee = first_chunk_fee + other_chunks_fee
        logger.info(f"Total Pastel transaction fee for storing the data: {combined_total_fee} PSL")
        logger.info('First Transaction ID:', first_txid)
        total_tx_size_in_kb = sum(len(self.packtx(txins, txouts)) for txins, txouts in zip(self.select_txins(0), self.create_txouts(chunks, transaction_ids))) / 1000
        logger.info(f"Total transaction size of compressed ticket data as stored in the blockchain: {total_tx_size_in_kb:,} KB")
        total_hex_transaction_length = sum(len(hexlify(self.packtx(txins, txouts)).decode('utf-8')) for txins, txouts in zip(self.select_txins(0), self.create_txouts(chunks, transaction_ids)))
        logger.info(f"Total length of hex transactions in characters: {total_hex_transaction_length}")
        return first_txid
    except JSONRPCException as e:
        logger.error(f"Error occurred while storing data: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}")
        return None

    async def store_data_chunks(self, chunks, uncompressed_data_hash, compressed_data_hash):
        transaction_ids = []
        total_fee = Decimal(0)
        for chunk in chunks:
            (txins, change) = await self.select_txins(0)
            if txins is None or change is None:
                logger.error("Insufficient funds to store the data")
                return None
            change = Decimal(change)
            txouts = []
            # Use the original approach for subsequent transactions
            combined_data = len(chunk).to_bytes(2, 'big') + uncompressed_data_hash + compressed_data_hash + chunk
            fd = io.BytesIO(combined_data)
            while True:
                scriptPubKey = self.checkmultisig_scriptPubKey_dump(fd)
                if scriptPubKey is None:
                    break
                value = Decimal(1) / self.coin  # Convert to Decimal
                txouts.append((value, scriptPubKey))
                change -= value
            out_value = Decimal(self.base_transaction_amount)  # Convert to Decimal
            change -= out_value
            receiving_address = await self.rpc_connection.getnewaddress()
            txouts.append((out_value, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(receiving_address)) + self.op_equalverify + self.op_checksig))
            change_address = await self.rpc_connection.getnewaddress()
            txouts.append([change, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(change_address)) + self.op_equalverify + self.op_checksig])
            tx = self.packtx(txins, txouts)
            hex_transaction = hexlify(tx).decode('utf-8')
            signed_tx = await self.rpc_connection.signrawtransaction(hex_transaction)
            fee = Decimal(len(signed_tx['hex']) / 1000) * self.fee_per_kb  # Convert to Decimal
            change -= fee
            txouts[-1][0] = change
            final_tx = self.packtx(txins, txouts)
            signed_tx = await self.rpc_connection.signrawtransaction(hexlify(final_tx).decode('utf-8'))
            assert signed_tx['complete']
            hex_signed_transaction = signed_tx['hex']
            send_raw_transaction_result = await self.rpc_connection.sendrawtransaction(hex_signed_transaction)
            blockchain_transaction_id = send_raw_transaction_result
            transaction_ids.append(blockchain_transaction_id)
            total_fee += fee
        return transaction_ids, total_fee

    async def store_first_chunk(self, first_chunk, transaction_ids):
        (txins, change) = await self.select_txins(0)
        if txins is None or change is None:
            logger.error("Insufficient funds to store the data")
            return None
        change = Decimal(change)
        txouts = []
        op_return_data = '|'.join(transaction_ids).encode()
        script_pubkey = self.op_return + self.pushdata(op_return_data) + self.pushdata(first_chunk)
        txouts.append((0, script_pubkey))
        out_value = Decimal(self.base_transaction_amount)  # Convert to Decimal
        change -= out_value
        receiving_address = await self.rpc_connection.getnewaddress()
        txouts.append((out_value, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(receiving_address)) + self.op_equalverify + self.op_checksig))
        change_address = await self.rpc_connection.getnewaddress()
        txouts.append([change, self.op_dup + self.op_hash160 + self.pushdata(self.addr2bytes(change_address)) + self.op_equalverify + self.op_checksig])
        tx = self.packtx(txins, txouts)
        hex_transaction = hexlify(tx).decode('utf-8')
        signed_tx = await self.rpc_connection.signrawtransaction(hex_transaction)
        fee = Decimal(len(signed_tx['hex']) / 1000) * self.fee_per_kb  # Convert to Decimal
        change -= fee
        txouts[-1][0] = change
        final_tx = self.packtx(txins, txouts)
        signed_tx = await self.rpc_connection.signrawtransaction(hexlify(final_tx).decode('utf-8'))
        assert signed_tx['complete']
        hex_signed_transaction = signed_tx['hex']
        send_raw_transaction_result = await self.rpc_connection.sendrawtransaction(hex_signed_transaction)
        first_txid = send_raw_transaction_result
        return first_txid, fee

    async def retrieve_data(self, first_txid):
        try:
            # Retrieve the first transaction
            first_tx = await self.rpc_connection.getrawtransaction(first_txid, True)
            outputs = first_tx['vout']
            # Extract the subsequent transaction IDs from the OP_RETURN output
            op_return_data = None
            for output in outputs:
                if 'scriptPubKey' in output and 'type' in output['scriptPubKey'] and output['scriptPubKey']['type'] == 'nulldata':
                    script_hex = output['scriptPubKey']['hex']
                    op_return_data = bytes.fromhex(script_hex[4:]).decode()
                    break
            if op_return_data is None:
                raise ValueError("No OP_RETURN output found in the first transaction")
            subsequent_txids = op_return_data.split('|')
            # Retrieve the data from the first transaction
            first_chunk_data = None
            for output in outputs:
                if 'scriptPubKey' in output and 'type' in output['scriptPubKey'] and output['scriptPubKey']['type'] == 'nulldata':
                    script_hex = output['scriptPubKey']['hex']
                    first_chunk_data = bytes.fromhex(script_hex[len(op_return_data)+6:])
                    break
            if first_chunk_data is None:
                raise ValueError("No data found in the first transaction")
            # Retrieve the data from the subsequent transactions
            compressed_data = first_chunk_data
            for txid in subsequent_txids:
                tx = await self.rpc_connection.getrawtransaction(txid, True)
                outputs = tx['vout']
                for output in outputs[1:-2]:
                    script_hex = output['scriptPubKey']['hex']
                    compressed_data += bytes.fromhex(script_hex[6:])
                compressed_data += bytes.fromhex(outputs[-2]['scriptPubKey']['hex'][6:-4])
            if len(compressed_data) < 34:
                raise ValueError("Insufficient data retrieved from the blockchain")
            reconstructed_length_of_compressed_data = int.from_bytes(compressed_data[:2], 'big')
            reconstructed_uncompressed_data_hash = compressed_data[2:34]
            reconstructed_compressed_data_hash = compressed_data[34:66]
            reconstructed_compressed_data = compressed_data[66:66+reconstructed_length_of_compressed_data]
            hash_of_reconstructed_compressed_data = self.get_raw_sha256_hash(reconstructed_compressed_data)
            assert hash_of_reconstructed_compressed_data == reconstructed_compressed_data_hash, "Compressed data hash mismatch"
            reconstructed_uncompressed_data = self.decompress_data(reconstructed_compressed_data)
            hash_of_reconstructed_uncompressed_data = self.get_raw_sha256_hash(reconstructed_uncompressed_data)
            assert hash_of_reconstructed_uncompressed_data == reconstructed_uncompressed_data_hash, "Uncompressed data hash mismatch"
            logger.info('Successfully reconstructed and decompressed data!')
            return reconstructed_uncompressed_data
        except JSONRPCException as e:
            logger.error(f"Error occurred while retrieving data: {e}")
            return None
        except (ValueError, AssertionError) as e:
            logger.error(f"Error occurred while reconstructing data: {e}")
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
        base_transaction_amount = 0.000001
        fee_per_kb = Decimal(0.0001)
        storage = BlockchainUTXOStorage(rpc_user, rpc_password, rpc_port, base_transaction_amount, fee_per_kb)
        input_data = 'Your data to store in the blockchain'
        transaction_id = await storage.store_data(input_data)
        retrieved_data = await storage.retrieve_data(transaction_id)
        logger.info('Retrieved data:', retrieved_data.decode('utf-8'))

    if __name__ == '__main__':
        asyncio.run(main())