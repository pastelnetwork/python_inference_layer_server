import warnings
from cryptography.utils import CryptographyDeprecationWarning
from logger_config import setup_logger
from endpoint_functions import router
import asyncio
import os
import random
import traceback
import fastapi
import threading
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.base import RequestResponseEndpoint
from starlette.responses import Response
import uvloop
from uvicorn import Config, Server
from decouple import Config as DecoupleConfig, RepositoryEnv
from database_code import initialize_db
from setup_swiss_army_llama import check_and_setup_swiss_army_llama
from service_functions import (monitor_new_messages, generate_or_load_encryption_key_sync, decrypt_sensitive_data, get_env_value, fetch_all_mnid_tickets_details, establish_ssh_tunnel, schedule_micro_benchmark_periodically,
                                list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them, generate_supernode_inference_ip_blacklist)
warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)
config = DecoupleConfig(RepositoryEnv('.env'))
UVICORN_PORT = config.get("UVICORN_PORT", cast=int)
USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE = config.get("USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE", default=0, cast=int)
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

class LimitRequestSizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_request_size: int):
        super().__init__(app)
        self.max_request_size = max_request_size

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_size = int(request.headers.get('content-length', 0))
        if request_size > self.max_request_size:
            return Response("Request size exceeds the limit", status_code=413)
        return await call_next(request)
    
app.add_middleware(LimitRequestSizeMiddleware, max_request_size=50 * 1024 * 1024)
app.include_router(router, prefix='', tags=['main'])

# Custom Exception Handling Middleware
@app.middleware("http")
async def custom_exception_handling(request: Request, call_next):
    try:
        return await call_next(request)
    except RequestValidationError as ve:
        logger.error(f"Validation error: {ve}")
        return JSONResponse(status_code=ve.status_code, content={"detail": ve.error_msg})
    except Exception as e:
        tb = traceback.format_exc()  # Get the full traceback
        logger.error(f"Unhandled exception: {e}\n{tb}")  # Log the exception with traceback
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
    global LOCAL_PASTEL_ID_PASSPHRASE, SWISS_ARMY_LLAMA_SECURITY_TOKEN, OPENAI_API_KEY, CLAUDE3_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, STABILITY_API_KEY, OPENROUTER_API_KEY, encryption_key
    LOCAL_PASTEL_ID_PASSPHRASE = decrypt_sensitive_data(get_env_value("LOCAL_PASTEL_ID_PASSPHRASE"), encryption_key)
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
        asyncio.create_task(fetch_all_mnid_tickets_details())
        asyncio.create_task(list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them())
        # asyncio.create_task(periodic_ticket_listing_and_validation())
        asyncio.create_task(asyncio.to_thread(check_and_setup_swiss_army_llama, SWISS_ARMY_LLAMA_SECURITY_TOKEN)) # Check and setup Swiss Army Llama asynchronously
        await generate_supernode_inference_ip_blacklist()  # Compile IP blacklist text file of unresponsive Supernodes for inference tasks
        asyncio.create_task(schedule_generate_supernode_inference_ip_blacklist())  # Schedule the task
        asyncio.create_task(schedule_micro_benchmark_periodically())  # Schedule the task
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        
@app.on_event("startup")
async def startup_event():
    await startup()

async def schedule_generate_supernode_inference_ip_blacklist():
    while True:
        jitter = random.randint(-180, 180)  # Jitter of up to 3 minutes (180 seconds)
        interval_seconds = 300 + jitter  # 300 seconds = 5 minutes
        await asyncio.sleep(interval_seconds)
        await generate_supernode_inference_ip_blacklist()
        
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
    if USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE:
        ssh_thread = threading.Thread(target=establish_ssh_tunnel, daemon=True)
        ssh_thread.start()
    generate_or_load_encryption_key_sync()
    config = DecoupleConfig(RepositoryEnv('.env'))
    asyncio.run(main())
