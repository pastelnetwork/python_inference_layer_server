from logger_config import setup_logger
from endpoint_functions import router
import asyncio
import fastapi
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvloop
from uvicorn import Config, Server
from decouple import Config as DecoupleConfig, RepositoryEnv
from database_code import initialize_db
from service_functions import monitor_new_messages
from setup_swiss_army_llama import check_and_setup_swiss_army_llama

config = DecoupleConfig(RepositoryEnv('.env'))
UVICORN_PORT = config.get("UVICORN_PORT", cast=int)
SWISS_ARMY_LLAMA_SECURITY_TOKEN = config.get("SWISS_ARMY_LLAMA_SECURITY_TOKEN", cast=str)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
logger = setup_logger()

app = fastapi.FastAPI(
    title="Pastel-Supernode-Messaging-and-Control-Layer",
    description="Pastel Supernode Messaging and Control Layer API",
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

async def startup():
    try:
        db_init_complete = await initialize_db()
        logger.info(f"Database initialization complete: {db_init_complete}")
        asyncio.create_task(monitor_new_messages())  # Create a background task
        # Check and setup Swiss Army Llama
        check_and_setup_swiss_army_llama(SWISS_ARMY_LLAMA_SECURITY_TOKEN)
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
    asyncio.run(main())