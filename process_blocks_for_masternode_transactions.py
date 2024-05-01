import asyncio
import os
from sqlmodel import delete
from database_code import Session, MasternodeTransaction, initialize_db
from main import process_blocks_for_masternode_transactions, get_local_rpc_settings_func
from logger_config import setup_logger

logger = setup_logger()

async def main():
    await initialize_db()
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = get_local_rpc_settings_func()
    network, burn_address = get_network_info(rpc_port)
    rpc_connection = AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")
    use_direct_ticket_scanning = 0
    while True:
        try:
            await process_blocks_for_masternode_transactions()
            await asyncio.sleep(60)  # Adjust the delay as needed
        except Exception as e:
            logger.exception(f"Error processing blocks for masternode transactions: {e}")
            await asyncio.sleep(60)  # Delay before retrying

if __name__ == "__main__":
    asyncio.run(main())