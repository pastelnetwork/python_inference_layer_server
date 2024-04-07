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

#SQLAlchemy ORM Models:

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

class InferenceAPIUsageRequest(Base):
    __tablename__ = "inference_api_usage_requests"
    id = Column(Integer, primary_key=True, index=True)
    inference_request_id = Column(String, unique=True, index=True)
    requesting_pastelid = Column(String, index=True)
    credit_pack_identifier = Column(String, index=True)
    requested_model_canonical_string = Column(String)
    model_inference_type_string = Column(String)
    model_parameters_json = Column(JSON)
    model_input_data_json_b64 = Column(String)
    total_psl_cost_for_pack = Column(Numeric(precision=20, scale=8))
    initial_credit_balance = Column(Numeric(precision=20, scale=8))
    requesting_pastelid_signature = Column(String)
    class Config:
        protected_namespaces = ()
        
class InferenceAPIUsageResponse(Base):
    __tablename__ = "inference_api_usage_responses"
    id = Column(Integer, primary_key=True, index=True)
    inference_response_id = Column(String, unique=True, index=True)
    inference_request_id = Column(String, ForeignKey("inference_api_usage_requests.inference_request_id"), index=True)
    proposed_cost_of_request_in_inference_credits = Column(Numeric(precision=20, scale=8))
    remaining_credits_in_pack_after_request_processed = Column(Numeric(precision=20, scale=8))
    credit_usage_tracking_psl_address = Column(String, index=True)
    request_confirmation_message_amount_in_patoshis = Column(Integer)
    max_block_height_to_include_confirmation_transaction = Column(Integer)
    supernode_pastelid_and_signature_on_inference_response_id = Column(String)
    
class InferenceAPIOutputResult(Base):
    __tablename__ = "inference_api_output_results"
    id = Column(Integer, primary_key=True, index=True)
    inference_result_id = Column(String, unique=True, index=True)
    inference_request_id = Column(String, ForeignKey("inference_api_usage_requests.inference_request_id"), index=True)
    inference_response_id = Column(String, ForeignKey("inference_api_usage_responses.inference_response_id"), index=True)
    responding_supernode_pastelid = Column(String, index=True)
    inference_result_json_base64 = Column(String)
    inference_result_file_type_strings = Column(String)
    responding_supernode_signature_on_inference_result_id = Column(String)
        
# Legacy mockup data model class; TODO: delete once we change code that depends on it    
# class InferenceCreditPack(Base):
#     __tablename__ = "inference_credit_packs"
#     id = Column(Integer, primary_key=True, index=True)
#     credit_pack_identifier = Column(String, unique=True, index=True)
#     authorized_pastelids = Column(JSON)
#     psl_cost_per_credit = Column(Numeric(precision=20, scale=8))
#     total_psl_cost_for_pack = Column(Numeric(precision=20, scale=8))
#     initial_credit_balance = Column(Numeric(precision=20, scale=8))
#     current_credit_balance = Column(Numeric(precision=20, scale=8))
#     credit_usage_tracking_psl_address = Column(String, index=True)
#     version = Column(Integer)
#     purchase_height = Column(Integer)
#     timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        
class CreditPackPurchaseRequest(Base):
    __tablename__ = "credit_pack_purchase_requests"
    id = Column(Integer, primary_key=True, index=True)
    requesting_end_user_pastelid = Column(String, index=True)
    requested_initial_credits_in_credit_pack = Column(Integer)
    list_of_authorized_pastelids_allowed_to_use_credit_pack = Column(JSON)
    credit_usage_tracking_psl_address = Column(String, index=True)
    request_timestamp_utc_iso_string = Column(String)
    request_pastel_block_height = Column(Integer)
    request_pastel_block_hash = Column(String)
    credit_purchase_request_message_version_string = Column(String)
    credit_pack_purchase_request_version_string = Column(String)
    sha3_256_hash_of_credit_pack_purchase_request_fields = Column(String, unique=True, index=True)
    requesting_end_user_pastelid_signature_on_request_hash = Column(String)

class CreditPackPurchaseRequestResponse(Base):
    __tablename__ = "credit_pack_purchase_request_responses"
    id = Column(Integer, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields = Column(String, ForeignKey("credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields"), index=True)
    credit_pack_purchase_request_json = Column(String)
    psl_cost_per_credit = Column(Numeric(precision=20, scale=8))
    proposed_total_cost_of_credit_pack_in_psl = Column(Numeric(precision=20, scale=8))
    credit_usage_tracking_psl_address = Column(String, index=True)
    request_response_timestamp_utc_iso_string = Column(String)
    request_response_pastel_block_height = Column(Integer)
    request_response_pastel_block_hash = Column(String)
    credit_purchase_request_response_message_version_string = Column(String)
    responding_supernode_pastelid = Column(String, index=True)
    list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms = Column(JSON)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields = Column(String, unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_response_hash = Column(String)
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_hash = Column(JSON)
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_fields_json = Column(JSON)

class CreditPackPurchaseRequestConfirmation(Base):
    __tablename__ = "credit_pack_purchase_request_confirmations"
    id = Column(Integer, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields = Column(String, ForeignKey("credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields"), index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields = Column(String, ForeignKey("credit_pack_purchase_request_responses.sha3_256_hash_of_credit_pack_purchase_request_response_fields"), index=True)
    credit_pack_purchase_request_response_json = Column(String)
    requesting_end_user_pastelid = Column(String, index=True)
    txid_of_credit_purchase_burn_transaction = Column(String, index=True)
    credit_purchase_request_confirmation_utc_iso_string = Column(String)
    credit_purchase_request_confirmation_pastel_block_height = Column(Integer)
    credit_purchase_request_confirmation_pastel_block_hash = Column(String)
    credit_purchase_request_confirmation_message_version_string = Column(String)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields = Column(String, unique=True, index=True)
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields = Column(String)

class CreditPackPurchaseRequestConfirmationResponse(Base):
    __tablename__ = "credit_pack_purchase_request_confirmation_responses"
    id = Column(Integer, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields = Column(String, ForeignKey("credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields"), index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields = Column(String, ForeignKey("credit_pack_purchase_request_confirmations.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields"), index=True)
    credit_pack_confirmation_outcome_string = Column(String)
    pastel_api_credit_pack_ticket_registration_txid = Column(String, index=True)
    credit_pack_confirmation_failure_reason_if_applicable = Column(String)
    credit_purchase_request_confirmation_response_utc_iso_string = Column(String)
    credit_purchase_request_confirmation_response_pastel_block_height = Column(Integer)
    credit_purchase_request_confirmation_response_pastel_block_hash = Column(String)
    credit_purchase_request_confirmation_response_message_version_string = Column(String)
    responding_supernode_pastelid = Column(String, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields = Column(String, unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash = Column(String)

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
InferenceAPIUsageRequest.to_dict = to_dict
InferenceAPIUsageResponse.to_dict = to_dict

#_____________________________________________________________________________
# Pydantic Response Models:

class MessageModel(BaseModel):
    message: str
    message_type: str
    sending_sn_pastelid: str
    timestamp: datetime

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
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

class ReputationScoreUpdateModel(BaseModel):
    supernode_pastelid: str
    reputation_score: float    

class InferenceAPIUsageRequestModel(BaseModel):
    requesting_pastelid: str
    credit_pack_identifier: str
    requested_model_canonical_string: str
    model_inference_type_string: str
    model_parameters_json: str
    model_input_data_json_b64: str
    class Config:
        protected_namespaces = ()
    def dict(self, *args, **kwargs):
        return super().dict(*args, **kwargs)
    
class InferenceAPIUsageResponseModel(BaseModel):
    inference_response_id: str
    inference_request_id: str
    proposed_cost_of_request_in_inference_credits: Decimal
    remaining_credits_in_pack_after_request_processed: Decimal
    credit_usage_tracking_psl_address: str
    request_confirmation_message_amount_in_patoshis: int
    max_block_height_to_include_confirmation_transaction: int
    supernode_pastelid_and_signature_on_inference_response_id: str


class InferenceOutputResultsModel(BaseModel):
    inference_result_id: str
    inference_request_id: str
    inference_response_id: str
    responding_supernode_pastelid: str
    inference_result_json_base64: str
    inference_result_file_type_strings: str
    responding_supernode_signature_on_inference_result_id: str

class CreditPackPurchaseRequestModel(BaseModel):
    requesting_end_user_pastelid: str
    requested_initial_credits_in_credit_pack: int
    list_of_authorized_pastelids_allowed_to_use_credit_pack: List[str]
    credit_usage_tracking_psl_address: str
    request_timestamp_utc_iso_string: str
    request_pastel_block_height: int
    request_pastel_block_hash: str
    credit_purchase_request_message_version_string: str
    credit_pack_purchase_request_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    requesting_end_user_pastelid_signature_on_request_hash: str

class CreditPackPurchaseRequestRejectionModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    credit_pack_purchase_request_response_fields_json: str
    rejection_reason_string: str
    rejection_timestamp_utc_iso_string: str
    rejection_pastel_block_height: int
    rejection_pastel_block_hash: str
    credit_purchase_request_rejection_message_version_string: str
    responding_supernode_pastelid: str
    sha3_256_hash_of_credit_pack_purchase_request_rejection_fields: str
    responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash: str
    
class CreditPackPurchaseRequestPreliminaryPriceQuote(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    credit_pack_purchase_request_response_fields_json: str
    preliminary_quoted_price_per_credit_in_psl: float
    preliminary_total_cost_of_credit_pack_in_psl: float
    preliminary_price_quote_timestamp_utc_iso_string: str
    preliminary_price_quote_pastel_block_height: int
    preliminary_price_quote_pastel_block_hash: str
    preliminary_price_quote_message_version_string: str
    responding_supernode_pastelid: str
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str
    responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash: str
    
class CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str
    credit_pack_purchase_request_response_fields_json: str
    agree_with_preliminary_price_quote: bool
    preliminary_price_quote_response_timestamp_utc_iso_string: str
    preliminary_price_quote_response_pastel_block_height: int
    preliminary_price_quote_response_pastel_block_hash: str
    preliminary_price_quote_response_message_version_string: str
    requesting_end_user_pastelid: str
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields: str
    requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash: str
    
class CreditPackPurchasePriceAgreementRequestModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str
    supernode_requesting_price_agreement_pastelid: str 
    credit_pack_purchase_request_response_fields_json: str
    price_agreement_request_timestamp_utc_iso_string: str
    price_agreement_request_pastel_block_height: int
    price_agreement_request_pastel_block_hash: str
    price_agreement_request_message_version_string: str
    sha3_256_hash_of_price_agreement_request_fields: str
    supernode_requesting_price_agreement_pastelid_signature_on_request_hash: str

class CreditPackPurchasePriceAgreementRequestResponseModel(BaseModel):
    sha3_256_hash_of_price_agreement_request_fields: str
    credit_pack_purchase_request_response_fields_json: str
    agree_with_proposed_price: bool
    proposed_price_agreement_response_timestamp_utc_iso_string: str
    proposed_price_agreement_response_pastel_block_height: int
    proposed_price_agreement_response_pastel_block_hash: str
    proposed_price_agreement_response_message_version_string: str
    responding_supernode_pastelid: str
    sha3_256_hash_of_price_agreement_request_response_fields: str
    responding_supernode_signature_on_price_agreement_request_response_hash: str
    responding_supernode_signature_on_credit_pack_purchase_request_response_fields_json: str
    
class CreditPackRequestStatusCheckModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    requesting_end_user_pastelid: str
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields: str
    
class CreditPackPurchaseRequestResponseTerminationModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    credit_pack_purchase_request_json: str
    termination_reason_string: str
    termination_timestamp_utc_iso_string: str
    termination_pastel_block_height: int
    termination_pastel_block_hash: str
    credit_purchase_request_termination_message_version_string: str
    responding_supernode_pastelid: str
    sha3_256_hash_of_credit_pack_purchase_request_termination_fields: str
    responding_supernode_signature_on_credit_pack_purchase_request_termination_hash: str    
    
class CreditPackPurchaseRequestResponseModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    credit_pack_purchase_request_json: str
    psl_cost_per_credit: float
    proposed_total_cost_of_credit_pack_in_psl: float
    credit_usage_tracking_psl_address: str
    request_response_timestamp_utc_iso_string: str
    request_response_pastel_block_height: int
    request_response_pastel_block_hash: str
    credit_purchase_request_response_message_version_string: str
    responding_supernode_pastelid: str
    list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms: List[str]
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str
    responding_supernode_signature_on_credit_pack_purchase_request_response_hash: str
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_hash: List[str]
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_fields_json: List[str]

class CreditPackPurchaseRequestConfirmationModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str
    credit_pack_purchase_request_response_json: str
    requesting_end_user_pastelid: str
    txid_of_credit_purchase_burn_transaction: str
    credit_purchase_request_confirmation_utc_iso_string: str
    credit_purchase_request_confirmation_pastel_block_height: int
    credit_purchase_request_confirmation_pastel_block_hash: str
    credit_purchase_request_confirmation_message_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    
class CreditPackPurchaseRequestConfirmationResponseModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    credit_pack_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str
    credit_pack_confirmation_failure_reason_if_applicable: str
    credit_purchase_request_confirmation_response_utc_iso_string: str
    credit_purchase_request_confirmation_response_pastel_block_height: int
    credit_purchase_request_confirmation_response_pastel_block_hash: str
    credit_purchase_request_confirmation_response_message_version_string: str
    responding_supernode_pastelid: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields: str
    responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash: str

class CreditPackStorageRetryRequestModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str
    credit_pack_purchase_request_response_json: str
    requesting_end_user_pastelid: str
    closest_agreeing_supernode_to_retry_storage_pastelid: str
    credit_pack_storage_retry_request_timestamp_utc_iso_string: str
    credit_pack_storage_retry_request_pastel_block_height: int
    credit_pack_storage_retry_request_pastel_block_hash: str
    credit_pack_storage_retry_request_message_version_string: str
    sha3_256_hash_of_credit_pack_storage_retry_request_fields: str
    requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash: str

class CreditPackStorageRetryRequestResponseModel(BaseModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    credit_pack_storage_retry_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str
    credit_pack_storage_retry_confirmation_failure_reason_if_applicable: str
    credit_pack_storage_retry_confirmation_response_utc_iso_string: str
    credit_pack_storage_retry_confirmation_response_pastel_block_height: int
    credit_pack_storage_retry_confirmation_response_pastel_block_hash: str
    credit_pack_storage_retry_confirmation_response_message_version_string: str
    closest_agreeing_supernode_to_retry_storage_pastelid: str
    sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields: str
    closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash: str

#_____________________________________________________________________________


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

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False}, execution_options={"isolation_level": "SERIALIZABLE"})    

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
        "PRAGMA wal_autocheckpoint = 100;",
        "PRAGMA mmap_size = 30000000000;",
        "PRAGMA threads = 4;",
        "PRAGMA optimize;",
        "PRAGMA secure_delete = OFF;"
    ]
    list_of_sqlite_pragma_justification_strings = [
        "Set SQLite to use Write-Ahead Logging (WAL) mode (from default DELETE mode) so that reads and writes can occur simultaneously",
        "Set synchronous mode to NORMAL (from FULL) so that writes are not blocked by reads",
        "Set cache size to 1GB (from default 2MB) so that more data can be cached in memory and not read from disk; to make this 256MB, set it to -262144 instead",
        "Increase the busy timeout to 2 seconds so that the database waits",
        "Set the WAL autocheckpoint to 100 (from default 1000) so that the WAL file is checkpointed more frequently",
        "Set the maximum size of the memory-mapped I/O cache to 30GB to improve performance by accessing the database file directly from memory",
        "Enable multi-threaded mode in SQLite and set the number of worker threads to 4 to allow concurrent access to the database",
        "Optimize the database by running a set of optimization steps to improve query performance",
        "Disable the secure delete feature to improve deletion performance at the cost of potentially leaving deleted data recoverable"
    ]
    assert(len(list_of_sqlite_pragma_strings) == len(list_of_sqlite_pragma_justification_strings))
    try:
        async with engine.begin() as conn:
            try:
                for pragma_string in list_of_sqlite_pragma_strings:
                    await conn.execute(sql_text(pragma_string))
                await conn.run_sync(Base.metadata.create_all)  # Create tables if they don't exist
            finally:
                await conn.close()  # Close the connection explicitly
        await engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Database Initialization Error: {e}")
        return False