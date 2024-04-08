from logger_config import setup_logger
from datetime import datetime
from decimal import Decimal
from sqlalchemy import text as sql_text
from sqlmodel import Session as SQLModelSession
from contextlib import contextmanager
from sqlmodel import Field, SQLModel, create_async_engine, Relationship
from decouple import Config as DecoupleConfig, RepositoryEnv
from typing import Optional, List
        
config = DecoupleConfig(RepositoryEnv('.env'))
DATABASE_URL = config.get("DATABASE_URL", cast=str, default="sqlite+aiosqlite:///super_node_messaging_and_control_layer.sqlite")
logger = setup_logger()

#SQLModel Models (combined SQLalchemy ORM models with Pydantic response models)

#_________________________________________________________________________________________
# Messaging related models:

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    receiving_sn_pastelid: str = Field(index=True)
    sending_sn_txid_vout: str = Field(index=True)
    receiving_sn_txid_vout: str = Field(index=True)
    message_type: str = Field(index=True)
    message_body: str
    signature: str
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)
    def __repr__(self):
        return f"<Message(id={self.id}, sending_sn_pastelid='{self.sending_sn_pastelid}', receiving_sn_pastelid='{self.receiving_sn_pastelid}', message_type='{self.message_type}', timestamp='{self.timestamp}')>"

class UserMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    from_pastelid: str = Field(index=True)
    to_pastelid: str = Field(index=True)
    message_body: str
    message_signature: str
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

class SupernodeUserMessage(Message, table=True):
    user_message_id: Optional[int] = Field(default=None, foreign_key="user_messages.id", index=True)
    user_message: Optional[UserMessage] = Relationship(back_populates="supernode_user_messages")

class MessageMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    total_messages: int
    total_senders: int
    total_receivers: int
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

class MessageSenderMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    sending_sn_txid_vout: str = Field(index=True)
    sending_sn_pubkey: str = Field(index=True)
    total_messages_sent: int
    total_data_sent_bytes: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

class MessageReceiverMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    receiving_sn_pastelid: str = Field(index=True)
    receiving_sn_txid_vout: str = Field(index=True)
    total_messages_received: int
    total_data_received_bytes: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

class MessageSenderReceiverMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    receiving_sn_pastelid: str = Field(index=True)
    total_messages: int
    total_data_bytes: Decimal
    timestamp: datetime = Field(default_factory=datetime.utcnow, index=True)

class SendMessageResponse(SQLModel):
    status: str
    message: str    

#_________________________________________________________________________________________
# Misc. data models:

class SupernodeData(SQLModel):
    supernode_status: str
    protocol_version: str
    supernode_psl_address: str
    lastseentime: datetime
    activeseconds: int
    lastpaidtime: datetime
    lastpaidblock: int
    ipaddress_port: str
    rank: int
    pubkey: str
    extAddress: Optional[str]
    extP2P: Optional[str]
    extKey: Optional[str]
    activedays: float

class LocalMachineSupernodeInfo(SQLModel):
    local_machine_supernode_data: SupernodeData
    local_sn_rank: int
    local_sn_pastelid: str
    local_machine_ip_with_proper_port: str

class ReputationScoreUpdate(SQLModel):
    supernode_pastelid: str
    reputation_score: float
    
#_________________________________________________________________________________________
# Credit pack related models:

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

class CreditPackPurchaseRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    requesting_end_user_pastelid: str = Field(index=True)
    requested_initial_credits_in_credit_pack: int
    list_of_authorized_pastelids_allowed_to_use_credit_pack: List[str]
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_timestamp_utc_iso_string: str
    request_pastel_block_height: int
    request_pastel_block_hash: str
    credit_purchase_request_message_version_string: str
    credit_pack_purchase_request_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_request_hash: str

class CreditPackPurchaseRequestResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    credit_pack_purchase_request_json: str
    psl_cost_per_credit: Decimal
    proposed_total_cost_of_credit_pack_in_psl: Decimal
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_response_timestamp_utc_iso_string: str
    request_response_pastel_block_height: int
    request_response_pastel_block_hash: str
    credit_purchase_request_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms: List[str]
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_response_hash: str
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_hash: List[str]
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_response_fields_json: List[str]

class CreditPackPurchaseRequestConfirmation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="credit_pack_purchase_request_responses.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
    credit_pack_purchase_request_response_json: str
    requesting_end_user_pastelid: str = Field(index=True)
    txid_of_credit_purchase_burn_transaction: str = Field(index=True)
    credit_purchase_request_confirmation_utc_iso_string: str
    credit_purchase_request_confirmation_pastel_block_height: int
    credit_purchase_request_confirmation_pastel_block_hash: str
    credit_purchase_request_confirmation_message_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str

class CreditPackPurchaseRequestConfirmationResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="credit_pack_purchase_requests.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(foreign_key="credit_pack_purchase_request_confirmations.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields", index=True)
    credit_pack_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str = Field(index=True)
    credit_pack_confirmation_failure_reason_if_applicable: str
    credit_purchase_request_confirmation_response_utc_iso_string: str
    credit_purchase_request_confirmation_response_pastel_block_height: int
    credit_purchase_request_confirmation_response_pastel_block_hash: str
    credit_purchase_request_confirmation_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash: str

class CreditPackPurchaseRequestRejection(SQLModel):
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

class CreditPackPurchaseRequestPreliminaryPriceQuote(SQLModel):
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

class CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(SQLModel):
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

class CreditPackPurchasePriceAgreementRequest(SQLModel):
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str
    supernode_requesting_price_agreement_pastelid: str
    credit_pack_purchase_request_response_fields_json: str
    price_agreement_request_timestamp_utc_iso_string: str
    price_agreement_request_pastel_block_height: int
    price_agreement_request_pastel_block_hash: str
    price_agreement_request_message_version_string: str
    sha3_256_hash_of_price_agreement_request_fields: str
    supernode_requesting_price_agreement_pastelid_signature_on_request_hash: str

class CreditPackPurchasePriceAgreementRequestResponse(SQLModel):
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

class CreditPackRequestStatusCheck(SQLModel):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str
    requesting_end_user_pastelid: str
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields: str

class CreditPackPurchaseRequestResponseTermination(SQLModel):
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

class CreditPackStorageRetryRequest(SQLModel):
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

class CreditPackStorageRetryRequestResponse(SQLModel):
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
    
#_________________________________________________________________________________________
# Inference request related models (i.e., using the credit packs to do inferences):
    
class InferenceAPIUsageRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_request_id: str = Field(unique=True, index=True)
    requesting_pastelid: str = Field(index=True)
    credit_pack_identifier: str = Field(index=True)
    requested_model_canonical_string: str
    model_inference_type_string: str
    model_parameters_json: dict
    model_input_data_json_b64: str
    total_psl_cost_for_pack: Decimal
    initial_credit_balance: Decimal
    requesting_pastelid_signature: str

class InferenceAPIUsageResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_response_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inference_api_usage_requests.inference_request_id", index=True)
    proposed_cost_of_request_in_inference_credits: Decimal
    remaining_credits_in_pack_after_request_processed: Decimal
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_confirmation_message_amount_in_patoshis: int
    max_block_height_to_include_confirmation_transaction: int
    supernode_pastelid_and_signature_on_inference_response_id: str

class InferenceAPIOutputResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_result_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inference_api_usage_requests.inference_request_id", index=True)
    inference_response_id: str = Field(foreign_key="inference_api_usage_responses.inference_response_id", index=True)
    responding_supernode_pastelid: str = Field(index=True)
    inference_result_json_base64: str
    inference_result_file_type_strings: str
    responding_supernode_signature_on_inference_result_id: str

class InferenceCreditPackRequest(SQLModel):
    authorized_pastelids_to_use_credits: List[str]
    psl_cost_per_credit: float
    total_psl_cost_for_pack: float
    initial_credit_balance: float
    credit_usage_tracking_psl_address: str

class InferenceConfirmation(SQLModel):
    inference_request_id: str
    confirmation_transaction: dict

# class InferenceAPIUsageRequestModel(BaseModel):
#     requesting_pastelid: str
#     credit_pack_identifier: str
#     requested_model_canonical_string: str
#     model_inference_type_string: str
#     model_parameters_json: str
#     model_input_data_json_b64: str
#     class Config:
#         protected_namespaces = ()
#     def dict(self, *args, **kwargs):
#         return super().dict(*args, **kwargs)

#_____________________________________________________________________________

engine = create_async_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False}, execution_options={"isolation_level": "SERIALIZABLE"})    

@contextmanager
def Session():
    session = SQLModelSession(engine)
    try:
        yield session
    finally:
        session.close()

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
                await conn.run_sync(SQLModel.metadata.create_all)  # Create tables if they don't exist
            finally:
                await conn.close()  # Close the connection explicitly
        await engine.dispose()
        return True
    except Exception as e:
        logger.error(f"Database Initialization Error: {e}")
        return False