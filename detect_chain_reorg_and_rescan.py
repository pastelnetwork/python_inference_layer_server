import asyncio
from main import detect_chain_reorg_and_rescan
from database_code import initialize_db
from logger_config import setup_logger

logger = setup_logger()

async def main():
    await initialize_db()
    while True:
        try:
            await detect_chain_reorg_and_rescan()
        except Exception as e:
            logger.exception(f"Error detecting chain reorg and rescanning: {e}")
        await asyncio.sleep(6000)  # Check for chain reorg every 6000 seconds

if __name__ == "__main__":
    asyncio.run(main())