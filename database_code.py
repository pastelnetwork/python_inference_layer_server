from logger_config import setup_logger
from datetime import datetime
from decimal import Decimal
import traceback
from sqlalchemy import Column, String, DateTime, Integer, Numeric, LargeBinary, Text, text as sql_text, ForeignKey, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from decouple import Config as DecoupleConfig, RepositoryEnv
from pydantic import BaseModel, Field
from typing import Optional, List
        
config = DecoupleConfig(RepositoryEnv('.env'))
DATABASE_URL = config.get("DATABASE_URL", cast=str, default="sqlite+aiosqlite:///super_node_messaging_and_control_layer.sqlite")
logger = setup_logger()
Base = declarative_base()

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    sending_sn_pastelid = Column(String, index=True)
    receiving_sn_pastelid = Column(String, index=True)
    sending_sn_txid_vout = Column(String, index=True)
    receiving_sn_txid_vout = Column(String, index=True)    
    message_type = Column(String, index=True)
    message_body = Column(Text)
    signature = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<Message(id={self.id}, sending_sn_pastelid='{self.sending_sn_pastelid}', receiving_sn_pastelid='{self.receiving_sn_pastelid}', message_type='{self.message_type}', timestamp='{self.timestamp}')>"

class MessageModel(BaseModel):
    message: str
    message_type: str
    sending_sn_pastelid: str
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class UserMessage(Base):
    __tablename__ = "user_messages"

    id = Column(Integer, primary_key=True, index=True)
    from_pastelid = Column(String, index=True)
    to_pastelid = Column(String, index=True)
    message_body = Column(Text)
    message_signature = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class SupernodeUserMessage(Message):
    __tablename__ = "supernode_user_messages"

    id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    user_message_id = Column(Integer, ForeignKey("user_messages.id"), index=True)
    user_message = relationship("UserMessage", backref="supernode_user_messages")
    
    __mapper_args__ = {
        "polymorphic_identity": "supernode_user_message",
    }
    
class SendMessageResponse(BaseModel):
    status: str
    message: str

class UserMessageCreate(BaseModel):
    from_pastelid: str
    to_pastelid: str
    message_body: str
    message_signature: str

class UserMessageModel(UserMessageCreate):
    id: int
    timestamp: datetime

    class Config:
        from_attributes = True

class SupernodeUserMessageCreate(MessageModel):
    user_message: UserMessageCreate

class SupernodeUserMessageModel(MessageModel):
    id: int
    user_message: UserMessageModel

    class Config:
        from_attributes = True

class SupernodeData(BaseModel):
    supernode_status: str
    protocol_version: str
    supernode_psl_address: str
    lastseentime: datetime
    activeseconds: int
    lastpaidtime: datetime
    lastpaidblock: int
    ipaddress_port: str = Field(..., alias='ipaddress:port')
    rank: int
    pubkey: str
    extAddress: Optional[str]
    extP2P: Optional[str]
    extKey: Optional[str]
    activedays: float
    
    class Config:
        from_attributes = True
        populate_by_name = True

class LocalMachineSupernodeInfo(BaseModel):
    local_machine_supernode_data: SupernodeData
    local_sn_rank: int
    local_sn_pastelid: str
    local_machine_ip_with_proper_port: str

    class Config:
        from_attributes = True
        
class InferenceCreditPackRequest(BaseModel):
    authorized_pastelids_to_use_credits: List[str]
    psl_cost_per_credit: float
    total_psl_cost_for_pack: float
    initial_credit_balance: float
    credit_usage_tracking_psl_address: str

class InferenceConfirmationModel(BaseModel):
    inference_request_id: str
    confirmation_transaction: dict

class InferenceOutputResultsModel(BaseModel):
    inference_request_id: str
    inference_response_id: str
    output_results: dict
    
class ReputationScoreUpdateModel(BaseModel):
    supernode_pastelid: str
    reputation_score: float    

class InferenceAPIUsageRequestModel(BaseModel):
    requesting_pastelid: str
    credit_pack_identifier: str
    requested_model_canonical_string: str
    model_parameters_json: str
    model_input_data_json_b64: str

    class Config:
        protected_namespaces = ()

class InferenceAPIUsageRequest(Message):
    __tablename__ = "inference_api_usage_requests"

    id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    inference_request_id = Column(String, unique=True, index=True)
    requesting_pastelid = Column(String, index=True)
    credit_pack_identifier = Column(String, index=True)
    requested_model_canonical_string = Column(String)
    model_parameters_json = Column(JSON)
    model_input_data_json_b64 = Column(String)
    total_psl_cost_for_pack = Column(Numeric(precision=20, scale=8))
    initial_credit_balance = Column(Numeric(precision=20, scale=8))
    requesting_pastelid_signature = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "inference_api_usage_request",
    }

    class Config:
        protected_namespaces = ()
        
class InferenceAPIUsageResponse(Message):
    __tablename__ = "inference_api_usage_responses"

    id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    inference_response_id = Column(String, unique=True, index=True)
    inference_request_id = Column(String, ForeignKey("inference_api_usage_requests.inference_request_id"), index=True)
    proposed_cost_of_request_in_inference_credits = Column(Numeric(precision=20, scale=8))
    remaining_credits_in_pack_after_request_processed = Column(Numeric(precision=20, scale=8))
    credit_usage_tracking_psl_address = Column(String, index=True)
    request_confirmation_message_amount_in_patoshis = Column(Integer)
    max_block_height_to_include_confirmation_transaction = Column(Integer)
    supernode_pastelids_and_signatures_pack_on_inference_response_id = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "inference_api_usage_response",
    }

class InferenceAPIOutputResult(Message):
    __tablename__ = "inference_api_output_results"

    id = Column(Integer, ForeignKey("messages.id"), primary_key=True)
    inference_result_id = Column(String, unique=True, index=True)
    inference_request_id = Column(String, ForeignKey("inference_api_usage_requests.inference_request_id"), index=True)
    inference_response_id = Column(String, ForeignKey("inference_api_usage_responses.inference_response_id"), index=True)
    responding_supernode_pastelid = Column(String, index=True)
    inference_result_json_base64 = Column(String)
    responding_supernode_signature_on_inference_result_id = Column(String)

    __mapper_args__ = {
        "polymorphic_identity": "inference_api_output_result",
    }

# Add a new model for storing inference credit packs
class InferenceCreditPack(Base):
    __tablename__ = "inference_credit_packs"

    id = Column(Integer, primary_key=True, index=True)
    credit_pack_identifier = Column(String, unique=True, index=True)
    authorized_pastelids = Column(JSON)
    psl_cost_per_credit = Column(Numeric(precision=20, scale=8))
    total_psl_cost_for_pack = Column(Numeric(precision=20, scale=8))
    initial_credit_balance = Column(Numeric(precision=20, scale=8))
    current_credit_balance = Column(Numeric(precision=20, scale=8))
    credit_usage_tracking_psl_address = Column(String, index=True)
    version = Column(Integer)
    purchase_height = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class MessageMetadata(Base):
    __tablename__ = "message_metadata"

    id = Column(Integer, primary_key=True, index=True)
    total_messages = Column(Integer)
    total_senders = Column(Integer)
    total_receivers = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
class MessageSenderMetadata(Base):
    __tablename__ = "message_sender_metadata"

    id = Column(Integer, primary_key=True, index=True)
    sending_sn_pastelid = Column(String, index=True)
    sending_sn_txid_vout = Column(String, index=True)
    sending_sn_pubkey = Column(String, index=True)    
    total_messages_sent = Column(Integer)
    total_data_sent_bytes = Column(Numeric(precision=20, scale=2))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class MessageReceiverMetadata(Base):
    __tablename__ = "message_receiver_metadata"

    id = Column(Integer, primary_key=True, index=True)
    receiving_sn_pastelid = Column(String, index=True)
    receiving_sn_txid_vout = Column(String, index=True)
    total_messages_received = Column(Integer)
    total_data_received_bytes = Column(Numeric(precision=20, scale=2))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

class MessageSenderReceiverMetadata(Base):
    __tablename__ = "message_sender_receiver_metadata"

    id = Column(Integer, primary_key=True, index=True)
    sending_sn_pastelid = Column(String, index=True)
    receiving_sn_pastelid = Column(String, index=True)
    total_messages = Column(Integer)
    total_data_bytes = Column(Numeric(precision=20, scale=2))
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
def to_serializable(val):
    if isinstance(val, datetime):
        return val.isoformat()
    elif isinstance(val, Decimal):
        return float(val)
    else:
        return str(val)

def to_dict(self):
    d = {}
    for column in self.__table__.columns:
        if not isinstance(column.type, LargeBinary):
            value = getattr(self, column.name)
            if value is not None:
                serialized_value = to_serializable(value)
                d[column.name] = serialized_value if serialized_value is not None else value
    return d

Message.to_dict = to_dict
UserMessage.to_dict = to_dict
SupernodeUserMessage.to_dict = to_dict

async def get_db():
    db = AsyncSessionLocal()
    try:
        yield db
        await db.commit()
    except Exception as e:
        tb_str = traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        tb_str = "".join(tb_str)        
        logger.error(f"Database Error: {e}\nFull Traceback:\n{tb_str}")
        await db.rollback()
        raise
    finally:
        await db.close()

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
    
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

async def initialize_db():
    list_of_sqlite_pragma_strings = [
        "PRAGMA journal_mode=WAL;", 
        "PRAGMA synchronous = NORMAL;", 
        "PRAGMA cache_size = -262144;", 
        "PRAGMA busy_timeout = 2000;", 
        "PRAGMA wal_autocheckpoint = 100;"
    ]
    list_of_sqlite_pragma_justification_strings = [
        "Set SQLite to use Write-Ahead Logging (WAL) mode (from default DELETE mode) so that reads and writes can occur simultaneously",
        "Set synchronous mode to NORMAL (from FULL) so that writes are not blocked by reads",
        "Set cache size to 1GB (from default 2MB) so that more data can be cached in memory and not read from disk; to make this 256MB, set it to -262144 instead",
        "Increase the busy timeout to 2 seconds so that the database waits",
        "Set the WAL autocheckpoint to 100 (from default 1000) so that the WAL file is checkpointed more frequently"
    ]
    assert(len(list_of_sqlite_pragma_strings) == len(list_of_sqlite_pragma_justification_strings))

    try:
        async with engine.begin() as conn:
            for pragma_string in list_of_sqlite_pragma_strings:
                await conn.execute(sql_text(pragma_string))
            await conn.run_sync(Base.metadata.create_all)  # Create tables if they don't exist
        await engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Database Initialization Error: {e}")
        return False
    