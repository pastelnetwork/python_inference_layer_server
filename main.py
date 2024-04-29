from logger_config import setup_logger
from endpoint_functions import router
import asyncio
import os
import fastapi
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvloop
from uvicorn import Config, Server
from decouple import Config as DecoupleConfig, RepositoryEnv
from database_code import initialize_db
from service_functions import monitor_new_messages, generate_or_load_encryption_key_sync, decrypt_sensitive_data, get_env_value
from setup_swiss_army_llama import check_and_setup_swiss_army_llama
from blockchain_ticket_storage import process_blocks_for_masternode_transactions

config = DecoupleConfig(RepositoryEnv('.env'))
UVICORN_PORT = config.get("UVICORN_PORT", cast=int)
SWISS_ARMY_LLAMA_SECURITY_TOKEN = config.get("SWISS_ARMY_LLAMA_SECURITY_TOKEN", cast=str)
os.environ['TZ'] = 'UTC' # Set timezone to UTC for the current session
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = setup_logger()

app = fastapi.FastAPI(
    title="Pastel-Supernode-Inference-Layer",
    description="Pastel Supernode Inference Layer API",
    docs_url="/",
    redoc_url="/redoc"
)

app.include_router(router, prefix='', tags=['main'])

# Custom Exception Handling Middleware
@app.middleware("http")
async def custom_exception_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except RequestValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(status_code=ve.status_code, content={"detail": ve.error_msg})
    except RequestValidationError as re:
        logger.error(f"Request validation error: {re}")
        return JSONResponse(status_code=422, content={"detail": re.errors()})
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        return JSONResponse(status_code=500, content={"detail": str(e)})

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
    expose_headers=["Authorization"]
)

def decrypt_sensitive_fields():
    global LOCAL_PASTEL_ID_PASSPHRASE, MY_PASTELID_PASSPHRASE, SWISS_ARMY_LLAMA_SECURITY_TOKEN, OPENAI_API_KEY, CLAUDE3_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, STABILITY_API_KEY, OPENROUTER_API_KEY, encryption_key
    LOCAL_PASTEL_ID_PASSPHRASE = decrypt_sensitive_data(get_env_value("LOCAL_PASTEL_ID_PASSPHRASE"), encryption_key)
    MY_PASTELID_PASSPHRASE = decrypt_sensitive_data(get_env_value("MY_PASTELID_PASSPHRASE"), encryption_key)
    SWISS_ARMY_LLAMA_SECURITY_TOKEN = decrypt_sensitive_data(get_env_value("SWISS_ARMY_LLAMA_SECURITY_TOKEN"), encryption_key)
    OPENAI_API_KEY = decrypt_sensitive_data(get_env_value("OPENAI_API_KEY"), encryption_key)
    CLAUDE3_API_KEY = decrypt_sensitive_data(get_env_value("CLAUDE3_API_KEY"), encryption_key)
    GROQ_API_KEY = decrypt_sensitive_data(get_env_value("GROQ_API_KEY"), encryption_key)
    MISTRAL_API_KEY = decrypt_sensitive_data(get_env_value("MISTRAL_API_KEY"), encryption_key)
    STABILITY_API_KEY = decrypt_sensitive_data(get_env_value("STABILITY_API_KEY"), encryption_key)
    OPENROUTER_API_KEY = decrypt_sensitive_data(get_env_value("OPENROUTER_API_KEY"), encryption_key)
    
async def startup():
    global encryption_key  # Declare encryption_key as global
    try:
        db_init_complete = await initialize_db()
        logger.info(f"Database initialization complete: {db_init_complete}")
        encryption_key = generate_or_load_encryption_key_sync()  # Generate or load the encryption key synchronously    
        decrypt_sensitive_fields() # Now decrypt sensitive fields        
        asyncio.create_task(monitor_new_messages())  # Create a background task
        asyncio.create_task(process_blocks_for_masternode_transactions())  # Create a background task
        asyncio.create_task(asyncio.to_thread(check_and_setup_swiss_army_llama, SWISS_ARMY_LLAMA_SECURITY_TOKEN)) # Check and setup Swiss Army Llama asynchronously
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        
@app.on_event("startup")
async def startup_event():
    await startup()

async def main():
    uvicorn_config = Config(
        "main:app",
        host="0.0.0.0",
        port=UVICORN_PORT,
        loop="uvloop",
    )
    server = Server(uvicorn_config)
    await server.serve()

if __name__ == "__main__":
    generate_or_load_encryption_key_sync()
    config = DecoupleConfig(RepositoryEnv('.env'))
    asyncio.run(main())