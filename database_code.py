import json
import warnings
from logger_config import setup_logger
from datetime import datetime
import datetime as dt
from typing import Optional
from contextlib import asynccontextmanager
from sqlmodel import Field, SQLModel, Relationship, Column, JSON
from sqlmodel.ext.asyncio.session import AsyncSession as SQLModelSession
from sqlalchemy.orm import sessionmaker
from pydantic import field_validator
from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import create_async_engine
from decouple import Config as DecoupleConfig, RepositoryEnv
        
config = DecoupleConfig(RepositoryEnv('.env'))
DATABASE_URL = config.get("DATABASE_URL", cast=str, default="sqlite+aiosqlite:///super_node_messaging_and_control_layer.sqlite")
logger = setup_logger()
# Ignore specific warnings related to shadowing of fields in SQLModel or Pydantic
warnings.filterwarnings("ignore", message="Field name .* shadows an attribute in parent .*")

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
    message_body: str = Field(sa_column=Column(JSON))
    signature: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    def __repr__(self):
        return f"<Message(id={self.id}, sending_sn_pastelid='{self.sending_sn_pastelid}', receiving_sn_pastelid='{self.receiving_sn_pastelid}', message_type='{self.message_type}', timestamp='{self.timestamp}')>"
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        json_schema_extra = {
            "example": {
                "sending_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "receiving_sn_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sending_sn_txid_vout": "0x1234...:0",
                "receiving_sn_txid_vout": "0x5678...:0",
                "message_type": "text",
                "message_body": "Hello, how are you?",
                "signature": "0xabcd...",
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }

class UserMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    from_pastelid: str = Field(index=True)
    to_pastelid: str = Field(index=True)
    message_body: str = Field(sa_column=Column(JSON))
    message_signature: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        json_schema_extra = {
            "example": {
                "from_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "to_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "message_body": "Hey, let's meet up!",
                "message_signature": "0xdef0...",
                "timestamp": "2023-06-01T12:30:00Z"
            }
        }
        
class SupernodeUserMessage(Message, table=True):
    user_message_id: Optional[int] = Field(default=None, foreign_key="usermessage.id", index=True)
    user_message: Optional[UserMessage] = Relationship(back_populates="supernode_user_messages")
    class Config:
        arbitrary_types_allowed = True  # Allow arbitrary types
        json_schema_extra = {
            "example": {
                "sending_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "receiving_sn_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sending_sn_txid_vout": "0x1234...:0",
                "receiving_sn_txid_vout": "0x5678...:0",
                "message_type": "user_message",
                "message_body": '{"from_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk", "to_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP", "message_body": "Hey, let\'s meet up!", "message_signature": "0xdef0..."}',
                "signature": "0xabcd...",
                "timestamp": "2023-06-01T12:30:00Z",
                "user_message_id": 1
            }
        }

class MessageMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    total_messages: int
    total_senders: int
    total_receivers: int
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        json_schema_extra = {
            "example": {
                "total_messages": 1000,
                "total_senders": 100,
                "total_receivers": 200,
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }

class MessageSenderMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    sending_sn_txid_vout: str = Field(index=True)
    sending_sn_pubkey: str = Field(index=True)
    total_messages_sent: int
    total_data_sent_bytes: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        json_schema_extra = {
            "example": {
                "sending_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sending_sn_txid_vout": "0x1234...:0",
                "sending_sn_pubkey": "0xabcd...",
                "total_messages_sent": 500,
                "total_data_sent_bytes": 1000000,
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }
        
class MessageReceiverMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    receiving_sn_pastelid: str = Field(index=True)
    receiving_sn_txid_vout: str = Field(index=True)
    total_messages_received: int
    total_data_received_bytes: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        json_schema_extra = {
            "example": {
                "receiving_sn_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "receiving_sn_txid_vout": "0x5678...:0",
                "total_messages_received": 300,
                "total_data_received_bytes": 500000,
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }

class MessageSenderReceiverMetadata(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sending_sn_pastelid: str = Field(index=True)
    receiving_sn_pastelid: str = Field(index=True)
    total_messages: int
    total_data_bytes: float
    timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
    class Config:
        json_schema_extra = {
            "example": {
                "sending_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "receiving_sn_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "total_messages": 200,
                "total_data_bytes": 400000,
                "timestamp": "2023-06-01T12:00:00Z"
            }
        }

class SendMessageResponse(SQLModel):
    status: str
    message: str    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "message": "Message sent successfully"
            }
        }
        
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
    class Config:
        json_schema_extra = {
            "example": {
                "supernode_status": "active",
                "protocol_version": "1.0",
                "supernode_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "lastseentime": "2023-06-01T12:00:00Z",
                "activeseconds": 3600,
                "lastpaidtime": "2023-06-01T00:00:00Z",
                "lastpaidblock": 123456,
                "ipaddress_port": "127.0.0.1:9932",
                "rank": 1,
                "pubkey": "0xabcd...",
                "extAddress": "external_address",
                "extP2P": "external_p2p",
                "extKey": "external_key",
                "activedays": 30.5
            }
        }
        
class LocalMachineSupernodeInfo(SQLModel):
    local_machine_supernode_data: SupernodeData
    local_sn_rank: int
    local_sn_pastelid: str
    local_machine_ip_with_proper_port: str
    class Config:
        json_schema_extra = {
            "example": {
                "local_machine_supernode_data": {
                    "supernode_status": "active",
                    "protocol_version": "1.0",
                    "supernode_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                    "lastseentime": "2023-06-01T12:00:00Z",
                    "activeseconds": 3600,
                    "lastpaidtime": "2023-06-01T00:00:00Z",
                    "lastpaidblock": 123456,
                    "ipaddress_port": "127.0.0.1:9999",
                    "rank": 1,
                    "pubkey": "0xabcd...",
                    "extAddress": "external_address",
                    "extP2P": "external_p2p",
                    "extKey": "external_key",
                    "activedays": 30.5
                },
                "local_sn_rank": 1,
                "local_sn_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "local_machine_ip_with_proper_port": "127.0.0.1:9999"
            }
        }
        
class ReputationScoreUpdate(SQLModel):
    supernode_pastelid: str
    reputation_score: float
    class Config:
        json_schema_extra = {
            "example": {
                "supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "reputation_score": 4.5
            }
        }
            
#_________________________________________________________________________________________
# Credit pack related models:

class CreditPackPurchaseRequest(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    requesting_end_user_pastelid: str = Field(index=True)
    requested_initial_credits_in_credit_pack: int
    list_of_authorized_pastelids_allowed_to_use_credit_pack: str = Field(sa_column=Column(JSON))
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_timestamp_utc_iso_string: str
    request_pastel_block_height: int
    credit_purchase_request_message_version_string: str
    requesting_end_user_pastelid_signature_on_request_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "requested_initial_credits_in_credit_pack": 1000,
                "list_of_authorized_pastelids_allowed_to_use_credit_pack": ["jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk"],
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_timestamp_utc_iso_string": "2023-06-01T12:00:00Z",
                "request_pastel_block_height": 123456,
                "credit_purchase_request_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x5678...",
                "requesting_end_user_pastelid_signature_on_request_hash": "0xabcd..."
            }
        }

class CreditPackPurchaseRequestRejection(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    rejection_reason_string: str
    rejection_timestamp_utc_iso_string: str
    rejection_pastel_block_height: int
    credit_purchase_request_rejection_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_rejection_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "rejection_reason_string": "Invalid credit usage tracking PSL address",
                "rejection_timestamp_utc_iso_string": "2023-06-01T12:10:00Z",
                "rejection_pastel_block_height": 123457,
                "credit_purchase_request_rejection_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_rejection_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash": "0xdef0..."
            }
        }
        
class CreditPackPurchaseRequestPreliminaryPriceQuote(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_usage_tracking_psl_address: str = Field(index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    preliminary_quoted_price_per_credit_in_psl: float
    preliminary_total_cost_of_credit_pack_in_psl: float
    preliminary_price_quote_timestamp_utc_iso_string: str
    preliminary_price_quote_pastel_block_height: int
    preliminary_price_quote_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "preliminary_quoted_price_per_credit_in_psl": 0.1,
                "preliminary_total_cost_of_credit_pack_in_psl": 100,
                "preliminary_price_quote_timestamp_utc_iso_string": "2023-06-01T12:05:00Z",
                "preliminary_price_quote_pastel_block_height": 123456,
                "preliminary_price_quote_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash": "0xdef0..."
            }
        }

class CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    agree_with_preliminary_price_quote: bool
    credit_usage_tracking_psl_address: str = Field(index=True)
    preliminary_quoted_price_per_credit_in_psl: float
    preliminary_price_quote_response_timestamp_utc_iso_string: str
    preliminary_price_quote_response_pastel_block_height: int
    preliminary_price_quote_response_message_version_string: str
    requesting_end_user_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields": "0x5678...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "agree_with_preliminary_price_quote": True,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "preliminary_quoted_price_per_credit_in_psl": 0.1,
                "preliminary_price_quote_response_timestamp_utc_iso_string": "2023-06-01T12:10:00Z",
                "preliminary_price_quote_response_pastel_block_height": 123457,
                "preliminary_price_quote_response_message_version_string": "1.0",
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields": "0xdef0...",
                "requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash": "0x1234..."
            }
        }
        
class CreditPackPurchasePriceAgreementRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(index=True)
    supernode_requesting_price_agreement_pastelid: str = Field(index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    credit_usage_tracking_psl_address: str = Field(index=True)
    proposed_psl_price_per_credit: float
    price_agreement_request_timestamp_utc_iso_string: str
    price_agreement_request_pastel_block_height: int
    price_agreement_request_message_version_string: str
    sha3_256_hash_of_price_agreement_request_fields: str = Field(index=True)
    supernode_requesting_price_agreement_pastelid_signature_on_request_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x1234...",
                "supernode_requesting_price_agreement_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "proposed_psl_price_per_credit": 0.1,
                "price_agreement_request_timestamp_utc_iso_string": "2023-06-01T12:20:00Z",
                "price_agreement_request_pastel_block_height": 123457,
                "price_agreement_request_message_version_string": "1.0",
                "sha3_256_hash_of_price_agreement_request_fields": "0xabcd...",
                "supernode_requesting_price_agreement_pastelid_signature_on_request_hash": "0xdef0..."
            }
        }
        
class CreditPackPurchasePriceAgreementRequestResponse(SQLModel, table=True):
    sha3_256_hash_of_price_agreement_request_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    agree_with_proposed_price: bool
    credit_usage_tracking_psl_address: str = Field(unique=True,index=True)
    proposed_psl_price_per_credit: float
    proposed_price_agreement_response_timestamp_utc_iso_string: str
    proposed_price_agreement_response_pastel_block_height: int
    proposed_price_agreement_response_message_version_string: str
    responding_supernode_signature_on_credit_pack_purchase_request_fields_json: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_price_agreement_request_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_price_agreement_request_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_price_agreement_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "agree_with_proposed_price": True,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "proposed_psl_price_per_credit": 0.1,
                "proposed_price_agreement_response_timestamp_utc_iso_string": "2023-06-01T12:25:00Z",
                "proposed_price_agreement_response_pastel_block_height": 123458,
                "proposed_price_agreement_response_message_version_string": "1.0",
                "responding_supernode_signature_on_credit_pack_purchase_request_fields_json": "0x1234...",
                "responding_supernode_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sha3_256_hash_of_price_agreement_request_response_fields": "0xabcd...",
                "responding_supernode_signature_on_price_agreement_request_response_hash": "0xdef0...",
            }
        }
        
class CreditPackPurchaseRequestResponseTermination(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    termination_reason_string: str
    termination_timestamp_utc_iso_string: str
    termination_pastel_block_height: int
    credit_purchase_request_termination_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_termination_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_termination_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "termination_reason_string": "Insufficient agreeing supernodes",
                "termination_timestamp_utc_iso_string": "2023-06-01T12:30:00Z",
                "termination_pastel_block_height": 123459,
                "credit_purchase_request_termination_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_termination_fields": "0xabcd...",
                "responding_supernode_signature_on_credit_pack_purchase_request_termination_hash": "0xdef0..."
            }
        }        
        
class CreditPackPurchaseRequestResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    psl_cost_per_credit: float
    proposed_total_cost_of_credit_pack_in_psl: float
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_response_timestamp_utc_iso_string: str
    request_response_pastel_block_height: int
    credit_purchase_request_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms: str = Field(sa_column=Column(JSON))
    list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash: str = Field(sa_column=Column(JSON))
    list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk", ...}',
                "psl_cost_per_credit": 0.1,
                "proposed_total_cost_of_credit_pack_in_psl": 100,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_response_timestamp_utc_iso_string": "2023-06-01T12:15:00Z",
                "request_response_pastel_block_height": 123457,
                "credit_purchase_request_response_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms": ["jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk", "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP"],
                "list_of_agreeing_supernode_pastelids_signatures_on_price_agreement_request_response_hash": ["0x1234...", "0x5678..."],
                "list_of_agreeing_supernode_pastelids_signatures_on_credit_pack_purchase_request_fields_json": ["0xabcd...", "0xef01..."],
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x9abc...",
                "responding_supernode_signature_on_credit_pack_purchase_request_response_hash": "0xdef0..."
            }
        }
        
class CreditPackPurchaseRequestResponseTxidMapping(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_fields", primary_key=True, index=True)
    pastel_api_credit_pack_ticket_registration_txid: str = Field(unique=True, index=True)        
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "pastel_api_credit_pack_ticket_registration_txid": "0123456789abcdef..."
            }
        }
        
class CreditPackPurchaseRequestConfirmation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    requesting_end_user_pastelid: str = Field(index=True)
    txid_of_credit_purchase_burn_transaction: str = Field(index=True)
    credit_purchase_request_confirmation_utc_iso_string: str
    credit_purchase_request_confirmation_pastel_block_height: int
    credit_purchase_request_confirmation_message_version_string: str
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(unique=True, index=True)
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x5678...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "txid_of_credit_purchase_burn_transaction": "0xabcd...",
                "credit_purchase_request_confirmation_utc_iso_string": "2023-06-01T12:30:00Z",
                "credit_purchase_request_confirmation_pastel_block_height": 123458,
                "credit_purchase_request_confirmation_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0xdef0...",
                "requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x1234..."
            }
        }

class CreditPackPurchaseRequestConfirmationResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(foreign_key="creditpackpurchaserequestconfirmation.sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields", index=True)
    credit_pack_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str = Field(index=True)
    credit_pack_confirmation_failure_reason_if_applicable: str
    credit_purchase_request_confirmation_response_utc_iso_string: str
    credit_purchase_request_confirmation_response_pastel_block_height: int
    credit_purchase_request_confirmation_response_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x5678...",
                "credit_pack_confirmation_outcome_string": "success",
                "pastel_api_credit_pack_ticket_registration_txid": "0xabcd...",
                "credit_pack_confirmation_failure_reason_if_applicable": "",
                "credit_purchase_request_confirmation_response_utc_iso_string": "2023-06-01T12:45:00Z",
                "credit_purchase_request_confirmation_response_pastel_block_height": 123459,
                "credit_purchase_request_confirmation_response_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_response_fields": "0xdef0...",
                "responding_supernode_signature_on_credit_pack_purchase_request_confirmation_response_hash": "0x1234..."
            }
        }

class CreditPackRequestStatusCheck(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    requesting_end_user_pastelid: str = Field(index=True)
    requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields": "0x5678..."
            }
        }

class CreditPackPurchaseRequestStatus(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
    status: str = Field(index=True)
    status_details: str
    status_update_timestamp_utc_iso_string: str
    status_update_pastel_block_height: int
    credit_purchase_request_status_message_version_string: str
    responding_supernode_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_purchase_request_status_fields: str = Field(unique=True, index=True)
    responding_supernode_signature_on_credit_pack_purchase_request_status_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "status": "in_progress",
                "status_details": "Waiting for price agreement responses from supernodes",
                "status_update_timestamp_utc_iso_string": "2023-06-01T12:30:00Z",
                "status_update_pastel_block_height": 123456,
                "credit_purchase_request_status_message_version_string": "1.0",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "sha3_256_hash_of_credit_pack_purchase_request_status_fields": "0x5678...",
                "responding_supernode_signature_on_credit_pack_purchase_request_status_hash": "0xef01..."
            }
        }

class CreditPackStorageRetryRequest(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(primary_key=True, index=True)
    credit_pack_purchase_request_fields_json: str = Field(sa_column=Column(JSON))
    requesting_end_user_pastelid: str = Field(index=True)
    closest_agreeing_supernode_to_retry_storage_pastelid: str = Field(index=True)
    credit_pack_storage_retry_request_timestamp_utc_iso_string: str
    credit_pack_storage_retry_request_pastel_block_height: int
    credit_pack_storage_retry_request_message_version_string: str
    sha3_256_hash_of_credit_pack_storage_retry_request_fields: str
    requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_response_fields": "0x1234...",
                "credit_pack_purchase_request_fields_json": '{"sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...", ...}',
                "requesting_end_user_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "closest_agreeing_supernode_to_retry_storage_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "credit_pack_storage_retry_request_timestamp_utc_iso_string": "2023-06-01T12:50:00Z",
                "credit_pack_storage_retry_request_pastel_block_height": 123460,
                "credit_pack_storage_retry_request_message_version_string": "1.0",
                "sha3_256_hash_of_credit_pack_storage_retry_request_fields": "0xabcd...",
                "requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash": "0xdef0..."
            }
        }

class CreditPackStorageRetryRequestResponse(SQLModel, table=True):
    sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
    sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
    credit_pack_storage_retry_confirmation_outcome_string: str
    pastel_api_credit_pack_ticket_registration_txid: str
    credit_pack_storage_retry_confirmation_failure_reason_if_applicable: str
    credit_pack_storage_retry_confirmation_response_utc_iso_string: str
    credit_pack_storage_retry_confirmation_response_pastel_block_height: int
    credit_pack_storage_retry_confirmation_response_message_version_string: str
    closest_agreeing_supernode_to_retry_storage_pastelid: str = Field(index=True)
    sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields: str
    closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash: str    
    class Config:
        json_schema_extra = {
            "example": {
                "sha3_256_hash_of_credit_pack_purchase_request_fields": "0x1234...",
                "sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields": "0x5678...",
                "credit_pack_storage_retry_confirmation_outcome_string": "success",
                "pastel_api_credit_pack_ticket_registration_txid": "0xabcd...",
                "credit_pack_storage_retry_confirmation_failure_reason_if_applicable": "",
                "credit_pack_storage_retry_confirmation_response_utc_iso_string": "2023-06-01T12:55:00Z",
                "credit_pack_storage_retry_confirmation_response_pastel_block_height": 123461,
                "credit_pack_storage_retry_confirmation_response_message_version_string": "1.0",
                "closest_agreeing_supernode_to_retry_storage_pastelid": "jXa1s9mKDr4m6P8s7bKK1rYFgL7hkfGMLX1NozVSX4yTnfh9EjuP",
                "sha3_256_hash_of_credit_pack_storage_retry_confirmation_response_fields": "0xdef0...",
                "closest_agreeing_supernode_to_retry_storage_pastelid_signature_on_credit_pack_storage_retry_confirmation_response_hash": "0x1234..."
            }
        }

##______________________________________________________________________________________________________________________
# Inference request related models (i.e., using the credit packs to do inferences):
    
class InferenceAPIUsageRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_request_id: str = Field(unique=True, index=True)
    requesting_pastelid: str = Field(index=True)
    credit_pack_ticket_pastel_txid: str = Field(index=True)
    requested_model_canonical_string: str
    model_inference_type_string: str
    model_parameters_json: str  # Store the dict serialized as a JSON string
    model_input_data_json_b64: str
    inference_request_utc_iso_string: str
    inference_request_pastel_block_height: int
    inference_request_message_version_string: str
    sha3_256_hash_of_inference_request_fields: str
    requesting_pastelid_signature_on_request_hash: str
    @field_validator("model_parameters_json", mode="before")
    def serialize_dict_to_json(cls, v):
        if isinstance(v, dict):
            return json.dumps(v)
        return v
    @field_validator("model_parameters_json", mode="after")
    def deserialize_json_to_dict(cls, v):
        try:
            return json.loads(v)
        except ValueError:
            return {}
    class Config:
        protected_namespaces = ()
        json_schema_extra = {
            "example": {
                "inference_request_id": "0x1234...",
                "requesting_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "credit_pack_ticket_pastel_txid": "0x5678...",
                "requested_model_canonical_string": "gpt-3.5-turbo",
                "model_inference_type_string": "text-completion",
                "model_parameters_json": '{"max_tokens": 100, "temperature": 0.7}',
                "model_input_data_json_b64": "eyJwcm9tcHQiOiAiSGVsbG8sIGhvdyBhcmUgeW91PyJ9",
                "inference_request_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_request_pastel_block_height": 123456,
                "inference_request_message_version_string": "1.0",
                "sha3_256_hash_of_inference_request_fields": "0x5678...",
                "requesting_pastelid_signature_on_request_hash": "0xabcd..."
            }
        }

class InferenceAPIUsageResponse(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_response_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inferenceapiusagerequest.inference_request_id", index=True)
    proposed_cost_of_request_in_inference_credits: float
    remaining_credits_in_pack_after_request_processed: float
    credit_usage_tracking_psl_address: str = Field(index=True)
    request_confirmation_message_amount_in_patoshis: int
    max_block_height_to_include_confirmation_transaction: int
    inference_request_response_utc_iso_string: str
    inference_request_response_pastel_block_height: int
    inference_request_response_message_version_string: str    
    sha3_256_hash_of_inference_request_response_fields: str
    supernode_pastelid_and_signature_on_inference_request_response_hash: str
    class Config:
        json_schema_extra = {
            "example": {
                "inference_response_id": "0x1234...",
                "inference_request_id": "0x5678...",
                "proposed_cost_of_request_in_inference_credits": 10,
                "remaining_credits_in_pack_after_request_processed": 990,
                "credit_usage_tracking_psl_address": "tPj2wX5mjQErTju6nueVRkxGMCPuMkLn8CWdViJ38m9Wf6PBK5jV",
                "request_confirmation_message_amount_in_patoshis": 1000,
                "max_block_height_to_include_confirmation_transaction": 123456,
                "inference_request_response_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_request_response_pastel_block_height": 123456,
                "inference_request_response_message_version_string": "1.0",
                "sha3_256_hash_of_inference_request_response_fields": "0x5678...",
                "supernode_pastelid_and_signature_on_inference_request_response_hash": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk:0xabcd..."
            }
        }
        
class InferenceAPIOutputResult(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    inference_result_id: str = Field(unique=True, index=True)
    inference_request_id: str = Field(foreign_key="inferenceapiusagerequest.inference_request_id", index=True)
    inference_response_id: str = Field(foreign_key="inferenceapiusageresponse.inference_response_id", index=True)
    responding_supernode_pastelid: str = Field(index=True)
    inference_result_json_base64: str
    inference_result_file_type_strings: str
    inference_result_utc_iso_string: str
    inference_result_pastel_block_height: int
    inference_result_message_version_string: str    
    sha3_256_hash_of_inference_result_fields: str    
    responding_supernode_signature_on_inference_result_id: str
    class Config:
        json_schema_extra = {
            "example": {
                "inference_result_id": "0x1234...",
                "inference_request_id": "0x5678...",
                "inference_response_id": "0x9abc...",
                "responding_supernode_pastelid": "jXYJud3rmrR1Sk2scvR47N4E4J5Vv48uCC6se2nUHyfSJ17wacN7rVZLe6Sk",
                "inference_result_json_base64": "eyJvdXRwdXQiOiAiSGVsbG8sIEknbSBkb2luZyBncmVhdCEgSG93IGFib3V0IHlvdT8ifQ==",
                "inference_result_file_type_strings": "json",
                "inference_result_utc_iso_string": "2023-06-01T12:00:00Z",
                "inference_result_pastel_block_height": 123456,
                "inference_result_message_version_string": "1.0",
                "sha3_256_hash_of_inference_result_fields": "0x5678...",
                "responding_supernode_signature_on_inference_result_id": "0xdef0..."
            }
        }
        
class InferenceConfirmation(SQLModel):
    inference_request_id: str
    confirmation_transaction: dict
    class Config:
        json_schema_extra = {
            "example": {
                "inference_request_id": "0x1234...",
                "confirmation_transaction": {
                    "txid": "0x5678...",
                    "amount": 1000,
                    "block_height": 123456
                }
            }
        }


#_____________________________________________________________________________

engine = create_async_engine(DATABASE_URL, echo=False, future=True, connect_args={"check_same_thread": False}, execution_options={"isolation_level": "SERIALIZABLE"})    

@asynccontextmanager
async def Session() -> SQLModelSession:
    async_session = sessionmaker(engine, class_=SQLModelSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
        
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