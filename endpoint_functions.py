import service_functions
import database_code as db
from logger_config import logger
from fastapi import APIRouter, Depends, Query, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.exceptions import HTTPException
from starlette.background import BackgroundTask
from json import JSONEncoder
import json
import os
import tempfile
import uuid
import traceback
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Union, Dict, Any
from pydantic import SecretStr, BaseModel
from decouple import Config as DecoupleConfig, RepositoryEnv

config = DecoupleConfig(RepositoryEnv('.env'))
TEMP_OVERRIDE_LOCALHOST_ONLY = config.get("TEMP_OVERRIDE_LOCALHOST_ONLY", default=0)

# RPC Client Dependency
async def get_rpc_connection():
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = service_functions.get_local_rpc_settings_func() 
    return service_functions.AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")

router = APIRouter()

def localhost_only(request: Request):
    client_host = request.client.host
    if not TEMP_OVERRIDE_LOCALHOST_ONLY:
        if client_host != "127.0.0.1":
            raise HTTPException(status_code=401, detail="Unauthorized")
    
class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        
@router.get("/liveness_ping", response_model=dict)
async def liveness_ping_function():
    current_utc_timestamp = datetime.now(timezone.utc)
    response_dict = {'status': 'alive', 'timestamp': current_utc_timestamp}
    return response_dict     

@router.get("/supernode_list_json", response_model=dict)
async def get_supernode_list_json(
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Retrieves the list of Supernodes as JSON data.

    Returns a JSON object containing the Supernode list data.

    Raises:
    - HTTPException (status_code=500): If an error occurs while retrieving the Supernode list.

    Example response:
    {
        "1234567890abcdef": {
            "supernode_status": "ENABLED",
            "protocol_version": "1.0",
            "supernode_psl_address": "tPmkbohSbiocyAhXJVBZkBsKJiuVyNRp2GJ",
            "lastseentime": "2024-03-22T12:34:56.789000",
            "activeseconds": 3600,
            "lastpaidtime": "2024-03-22T11:34:56.789000",
            "lastpaidblock": 12345,
            "ipaddress:port": "127.0.0.1:9999",
            "rank": 1,
            "pubkey": "0x1234567890abcdef",
            "extAddress": "127.0.0.1:9999",
            "extP2P": "127.0.0.1:9998",
            "extKey": "1234567890abcdef",
            "activedays": 1.0
        },
        ...
    }
    """
    try:
        _, supernode_list_json = await service_functions.check_supernode_list_func()
        return JSONResponse(content=json.loads(supernode_list_json))
    except Exception as e:
        logger.error(f"Error getting supernode list JSON: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/supernode_list_csv")
async def get_supernode_list_csv(
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Retrieves the list of Supernodes as a normalized CSV file.

    Returns a StreamingResponse containing the CSV file data.

    Raises:
    - HTTPException (status_code=500): If an error occurs while retrieving or processing the Supernode list.

    Example response:
    A CSV file named "supernode_list.csv" will be downloaded containing the normalized Supernode list data.

    CSV file structure:
    supernode_status,protocol_version,supernode_psl_address,lastseentime,activeseconds,lastpaidtime,lastpaidblock,ipaddress:port,rank,pubkey,extAddress,extP2P,extKey,activedays
    ENABLED,1.0,tPmkbohSbiocyAhXJVBZkBsKJiuVyNRp2GJ,2024-03-22T12:34:56.789000,3600,2024-03-22T11:34:56.789000,12345,127.0.0.1:9999,1,0x1234567890abcdef,127.0.0.1:9999,127.0.0.1:9998,1234567890abcdef,1.0
    ...
    """
    try:
        supernode_list_df, _ = await service_functions.check_supernode_list_func()
        normalized_df = pd.json_normalize(supernode_list_df.to_dict(orient='records'))
        # Convert the normalized DataFrame to CSV
        csv_data = normalized_df.to_csv(index=False)
        # Create a StreamingResponse with the CSV data
        response = StreamingResponse(iter([csv_data]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=supernode_list.csv"
        return response
    except Exception as e:
        logger.error(f"Error getting supernode list CSV: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/get_local_machine_sn_info", response_model=db.LocalMachineSupernodeInfo, dependencies=[Depends(localhost_only)])
async def get_local_machine_sn_info(
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
        
    Retrieves information about the local machine's Supernode status.

    Returns a LocalMachineSupernodeInfo object containing the following fields:
    - `local_machine_supernode_data`: DataFrame containing the local machine's Supernode data (if it is a Supernode).
    - `local_sn_rank`: The rank of the local machine's Supernode (if it is a Supernode).
    - `local_sn_pastelid`: The PastelID of the local machine's Supernode (if it is a Supernode).
    - `local_machine_ip_with_proper_port`: The IP address and port of the local machine's Supernode (if it is a Supernode).

    Raises:
    - HTTPException (status_code=404): If the local machine is not a Supernode.
    - HTTPException (status_code=500): If an error occurs while retrieving the local machine's Supernode information.

    Example response:
    {
        "local_machine_supernode_data": {
            "supernode_status": "ENABLED",
            "protocol_version": "1.0",
            ...
        },
        "local_sn_rank": 1,
        "local_sn_pastelid": "1234567890abcdef",
        "local_machine_ip_with_proper_port": "127.0.0.1:9999"
    }
    """
    try:
        local_machine_supernode_data, local_sn_rank, local_sn_pastelid, local_machine_ip_with_proper_port = await service_functions.get_local_machine_supernode_data_func()
        if len(local_machine_supernode_data) > 0:
            return db.LocalMachineSupernodeInfo(
                local_machine_supernode_data=local_machine_supernode_data.to_dict(orient='records')[0],
                local_sn_rank=local_sn_rank,
                local_sn_pastelid=local_sn_pastelid,
                local_machine_ip_with_proper_port=local_machine_ip_with_proper_port
            )
        else:
            raise HTTPException(status_code=404, detail="Local machine is not a Supernode")
    except Exception as e:
        logger.error(f"Error getting local machine Supernode info: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    
@router.get("/get_sn_data_from_pastelid", response_model=db.SupernodeData, dependencies=[Depends(localhost_only)])
async def get_sn_data_from_pastelid(
    pastelid: str = Query(..., description="The PastelID of the Supernode"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
        
    Retrieves Supernode data based on the specified PastelID.

    - `pastelid`: The PastelID of the Supernode.

    Returns a SupernodeData object containing the Supernode data.

    Raises:
    - HTTPException (status_code=404): If the specified machine is not a Supernode.
    - HTTPException (status_code=500): If an error occurs while retrieving the Supernode data.

    Example response:
    {
        "supernode_status": "ENABLED",
        "protocol_version": "1.0",
        "supernode_psl_address": "tPmkbohSbiocyAhXJVBZkBsKJiuVyNRp2GJ",
        ...
    }
    """
    try:
        supernode_data = await service_functions.get_sn_data_from_pastelid_func(pastelid)
        if not supernode_data.empty:
            return db.SupernodeData(**supernode_data.to_dict(orient='records')[0])
        else:
            raise HTTPException(status_code=404, detail="Specified machine is not a Supernode")
    except Exception as e:
        logger.error(f"Error getting Supernode data from PastelID: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")    
    
    
@router.get("/get_sn_data_from_sn_pubkey", response_model=db.SupernodeData, dependencies=[Depends(localhost_only)])
async def get_sn_data_from_sn_pubkey(
    pubkey: str = Query(..., description="The public key of the Supernode"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
        
    Retrieves Supernode data based on the specified Supernode public key.

    - `pubkey`: The public key of the Supernode.

    Returns a SupernodeData object containing the Supernode data.

    Raises:
    - HTTPException (status_code=404): If the specified machine is not a Supernode.
    - HTTPException (status_code=500): If an error occurs while retrieving the Supernode data.

    Example response:
    {
        "supernode_status": "ENABLED",
        "protocol_version": "1.0",
        "supernode_psl_address": "tPmkbohSbiocyAhXJVBZkBsKJiuVyNRp2GJ",
        ...
    }
    """
    try:
        supernode_data = await service_functions.get_sn_data_from_sn_pubkey_func(pubkey)
        if not supernode_data.empty:
            return db.SupernodeData(**supernode_data.to_dict(orient='records')[0])
        else:
            raise HTTPException(status_code=404, detail="Specified machine is not a Supernode")
    except Exception as e:
        logger.error(f"Error getting Supernode data from public key: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
        

@router.get("/get_messages", response_model=List[db.Message], dependencies=[Depends(localhost_only)])
async def get_messages(
    last_k_minutes: Optional[int] = Query(100, description="Number of minutes to retrieve messages from"),
    message_type: Optional[str] = Query("all", description="Type of messages to retrieve ('all' or specific type)"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
        
    Retrieves Supernode messages from the last specified minutes.

    - `last_k_minutes`: Number of minutes to retrieve messages from (default: 100).
    - `message_type`: Type of messages to retrieve ('all' or specific type) (default: 'all').

    Returns a list of Message objects containing the message, message_type, sending_sn_pastelid, and timestamp.
    """
    try:
        messages = await service_functions.parse_sn_messages_from_last_k_minutes_func(last_k_minutes, message_type)
        return [
            db.Message(
                message=msg["message"],
                message_type=msg["message_type"],
                sending_sn_pastelid=msg["sending_sn_pastelid"],
                timestamp=msg["timestamp"]
            )
            for msg in messages
        ]
    except Exception as e:
        logger.error(f"Error retrieving messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")
    

@router.post("/broadcast_message_to_all_sns", response_model=db.SendMessageResponse, dependencies=[Depends(localhost_only)])
async def broadcast_message_to_all_sns(
    message: str = Query(..., description="Message to broadcast"),
    message_type: str = Query(..., description="Type of the message"),
    pastelid_passphrase: SecretStr = Query(..., description="Passphrase for the sending PastelID"),
    verbose: Optional[int] = Query(0, description="Verbose mode (0 or 1)"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
    
    Broadcasts a message to a list of Supernodes.

    - `message`: Message to broadcast.
    - `message_type`: Type of the message.
    - `pastelid_passphrase`: Passphrase for the sending PastelID.
    - `verbose`: Verbose mode (0 or 1) (default: 0).

    Returns a SendMessageResponse object containing the status and message.
    """
    try:
        signed_message = await service_functions.broadcast_message_to_all_sns_using_pastelid_func(
            message, message_type, pastelid_passphrase.get_secret_value(), verbose
        )
        return db.SendMessageResponse(status="success", message=f"Message broadcasted: {signed_message}")
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        return db.SendMessageResponse(status="error", message=f"Error broadcasting message: {str(e)}")


@router.get("/request_challenge/{pastelid}")
async def request_challenge(
    pastelid: str,
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Request a challenge string for authentication.

    - `pastelid`: The PastelID requesting the challenge.

    Returns a dictionary containing the challenge string and the challenge ID.
    """
    try:
        challenge, challenge_id = await service_functions.generate_challenge(pastelid)
        return {"challenge": challenge, "challenge_id": challenge_id}
    except Exception as e:
        logger.error(f"Error generating challenge: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating challenge: {str(e)}")


@router.post("/send_user_message", response_model=db.SupernodeUserMessage)
async def send_user_message(
    user_message: db.UserMessage = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Send a user message via Supernodes.

    This endpoint allows a user to send a message to another user via the Supernode network.
    The sender must provide a valid challenge signature to authenticate their identity.

    Parameters:
    - `user_message` (UserMessage): The user message to be sent, including:
        - `from_pastelid` (str): The PastelID of the sender.
        - `to_pastelid` (str): The PastelID of the recipient.
        - `message_body` (str): The content of the message.
        - `message_signature` (str): The signature of the message by the sender's PastelID.
    - `challenge` (str): The challenge string obtained from the `/request_challenge` endpoint.
    - `challenge_id` (str): The ID of the challenge string.
    - `challenge_signature` (str): The signature of the PastelID on the challenge string.

    Returns:
    - `SupernodeUserMessage`: The sent message details, including:
        - `message` (str): The content of the sent message.
        - `message_type` (str): The type of the message (always "user_message").
        - `sending_sn_pastelid` (str): The PastelID of the Supernode that sent the message.
        - `timestamp` (datetime): The timestamp of when the message was sent.
        - `id` (int): The unique identifier of the Supernode user message.
        - `user_message` (UserMessage): The details of the user message, including:
            - `from_pastelid` (str): The PastelID of the sender.
            - `to_pastelid` (str): The PastelID of the recipient.
            - `message_body` (str): The content of the message.
            - `message_signature` (str): The signature of the message by the sender's PastelID.
            - `id` (int): The unique identifier of the user message.
            - `timestamp` (datetime): The timestamp of when the user message was created.

    Raises:
    - HTTPException (status_code=401): If the provided challenge signature is invalid.
    - HTTPException (status_code=500): If an error occurs while sending the user message.
    """
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            user_message.from_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        supernode_user_message = await service_functions.send_user_message_via_supernodes(
            user_message.from_pastelid, user_message.to_pastelid, user_message.message_body, user_message.message_signature
        )
        return supernode_user_message 
    except Exception as e:
        logger.error(f"Error sending user message: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending user message: {str(e)}")


@router.get("/get_user_messages", response_model=List[db.UserMessage])
async def get_user_messages(
    pastelid: str = Query(..., description="The PastelID to retrieve messages for"),
    challenge: str = Query(..., description="The challenge string"),
    challenge_id: str = Query(..., description="The ID of the challenge string"),
    challenge_signature: str = Query(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Retrieve all user messages (sent and received) for a given PastelID.

    This endpoint allows a user to retrieve all messages associated with their PastelID.
    The user must provide a valid challenge signature to authenticate their identity.

    Parameters:
    - `pastelid` (str): The PastelID of the user to retrieve messages for.
    - `challenge` (str): The challenge string obtained from the `/request_challenge` endpoint.
    - `challenge_id` (str): The ID of the challenge string.
    - `challenge_signature` (str): The signature of the PastelID on the challenge string.

    Returns:
    - List[UserMessage]: A list of user messages associated with the provided PastelID, each including:
        - `from_pastelid` (str): The PastelID of the sender.
        - `to_pastelid` (str): The PastelID of the recipient.
        - `message_body` (str): The content of the message.
        - `message_signature` (str): The signature of the message by the sender's PastelID.
        - `id` (int): The unique identifier of the user message.
        - `timestamp` (datetime): The timestamp of when the user message was created.

    Raises:
    - HTTPException (status_code=401): If the provided challenge signature is invalid.
    - HTTPException (status_code=500): If an error occurs while retrieving the user messages.
    """
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        user_messages = await service_functions.get_user_messages_for_pastelid(pastelid)
        return [
            db.UserMessage(
                from_pastelid=message.from_pastelid,
                to_pastelid=message.to_pastelid,
                message_body=message.message_body,
                message_signature=message.message_signature,
                id=message.id,
                timestamp=message.timestamp
            )
            for message in user_messages
        ]
    except Exception as e:
        logger.error(f"Error retrieving user messages: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving user messages: {str(e)}")


#__________________________________________________________________________________________________________
# Endpoints related to Credit pack purchasing and provisioning:

class CreditPackTicketResponse(BaseModel):
    credit_pack_purchase_request_response: db.CreditPackPurchaseRequestResponse
    credit_pack_purchase_request_confirmation: db.CreditPackPurchaseRequestConfirmation
    
    
@router.get("/get_credit_pack_ticket_from_txid", response_model=CreditPackTicketResponse)
async def get_credit_pack_ticket_from_txid_endpoint(
    txid: str = Query(..., description="The transaction ID of the credit pack ticket"),
    pastelid: str = Query(..., description="The PastelID of the requesting party"),
    challenge: str = Query(..., description="The challenge string"),
    challenge_id: str = Query(..., description="The ID of the challenge string"),
    challenge_signature: str = Query(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        _, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation = await service_functions.retrieve_credit_pack_ticket_using_txid(txid)
        if not all((credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation)):
            raise HTTPException(status_code=404, detail=f"Credit pack ticket with TXID {txid} not found or was invalid!")
        credit_pack_ticket = CreditPackTicketResponse(
            credit_pack_purchase_request_response=credit_pack_purchase_request_response,
            credit_pack_purchase_request_confirmation=credit_pack_purchase_request_confirmation
        )        
        return credit_pack_ticket
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket from txid: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving credit pack ticket from txid: {str(e)}")
    
    
@router.post("/credit_purchase_initial_request", response_model=Union[db.CreditPackPurchaseRequestPreliminaryPriceQuote, db.CreditPackPurchaseRequestRejection])
async def credit_purchase_initial_request_endpoint(
    credit_pack_request: db.CreditPackPurchaseRequest = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            credit_pack_request.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        preliminary_price_quote = await service_functions.process_credit_purchase_initial_request(credit_pack_request)
        return preliminary_price_quote
    except Exception as e:
        logger.error(f"Error encountered with credit purchase initial request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error encountered with credit purchase initial request: {str(e)}")


@router.post("/credit_purchase_preliminary_price_quote_response", response_model=Union[db.CreditPackPurchaseRequestResponse, db.CreditPackPurchaseRequestResponseTermination])
async def credit_purchase_preliminary_price_quote_response_endpoint(
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    preliminary_price_quote_response: db.CreditPackPurchaseRequestPreliminaryPriceQuoteResponse = Body(...),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            preliminary_price_quote_response.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        result = await service_functions.process_credit_purchase_preliminary_price_quote_response(preliminary_price_quote_response)
        if isinstance(result, db.CreditPackPurchaseRequestResponse):
            result_dict = result.model_dump()
            result_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in result_dict.items()}
            service_functions.log_action_with_payload("processed", "credit purchase preliminary price quote response", result_dict)
            return result
        elif isinstance(result, db.CreditPackPurchaseRequestResponseTermination):
            result_dict = result.model_dump()
            result_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in result_dict.items()}
            logger.warning(f"Credit purchase preliminary price quote response terminated: {result_dict}")
            return result
        else:
            raise HTTPException(status_code=500, detail="Unexpected response type")
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing credit purchase preliminary price quote response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit purchase preliminary price quote response: {str(e)}")


@router.post("/credit_pack_price_agreement_request", response_model=db.CreditPackPurchasePriceAgreementRequestResponse)
async def credit_pack_price_agreement_request_endpoint(
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    credit_pack_price_agreement_request: db.CreditPackPurchasePriceAgreementRequest = Body(...),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            credit_pack_price_agreement_request.supernode_requesting_price_agreement_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        response = await service_functions.process_credit_pack_price_agreement_request(credit_pack_price_agreement_request)
        if isinstance(response, db.CreditPackPurchasePriceAgreementRequestResponse):
            response_dict = response.model_dump()
            response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in response_dict.items()}
            service_functions.log_action_with_payload("processed", "credit pack price agreement request", response_dict)
            return response
        else:
            raise HTTPException(status_code=400, detail=response)
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing credit pack price agreement request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit pack price agreement request: {str(e)}")
    

@router.post("/check_status_of_credit_purchase_request", response_model=db.CreditPackPurchaseRequestStatus)
async def check_status_of_credit_purchase_request_endpoint(
    credit_pack_request_status_check: db.CreditPackRequestStatusCheck = Body(...), 
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            credit_pack_request_status_check.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        status = await service_functions.get_credit_purchase_request_status(credit_pack_request_status_check)
        status_dict = status.model_dump()
        status_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in status_dict.items()}
        service_functions.log_action_with_payload("checking status of", "credit purchase request", status_dict)
        return status
    except Exception as e:
        logger.error(f"Error checking status of credit purchase request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking status of credit purchase request: {str(e)}")


@router.post("/confirm_credit_purchase_request", response_model=db.CreditPackPurchaseRequestConfirmationResponse)
async def confirm_credit_purchase_request_endpoint(
    confirmation: db.CreditPackPurchaseRequestConfirmation = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            confirmation.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        response = await service_functions.process_credit_purchase_request_confirmation(confirmation)
        response_dict = response.model_dump()
        response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in response_dict.items()}
        service_functions.log_action_with_payload("processed", "credit purchase request confirmation", response_dict)
        return response
    except Exception as e:
        logger.error(f"Error processing credit purchase request confirmation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit purchase request confirmation: {str(e)}")


@router.post("/credit_pack_purchase_request_final_response_announcement")
async def credit_pack_purchase_request_final_response_announcement_endpoint(
    response: db.CreditPackPurchaseRequestResponse = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            response.responding_supernode_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        await service_functions.process_credit_pack_purchase_request_final_response_announcement(response)
        service_functions.log_action_with_payload("processed", "credit pack purchase request final response announcement", response)
        return {"message": "Announcement processed successfully"}
    except Exception as e:
        logger.error(f"Error processing credit pack purchase request final response announcement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit pack purchase request final response announcement: {str(e)}")


@router.post("/credit_pack_purchase_completion_announcement")
async def credit_pack_purchase_completion_announcement_endpoint(
    confirmation: db.CreditPackPurchaseRequestConfirmation = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            confirmation.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        await service_functions.process_credit_pack_purchase_completion_announcement(confirmation)
        service_functions.log_action_with_payload("processed", "credit pack purchase completion announcement", confirmation)
        return {"message": "Announcement processed successfully"}
    except Exception as e:
        logger.error(f"Error processing credit pack purchase completion announcement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit pack purchase completion announcement: {str(e)}")


@router.post("/credit_pack_storage_completion_announcement")
async def credit_pack_storage_completion_announcement_endpoint(
    storage_completion_announcement: db.CreditPackPurchaseRequestConfirmationResponse = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            storage_completion_announcement.responding_supernode_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        await service_functions.process_credit_pack_storage_completion_announcement(storage_completion_announcement)
        service_functions.log_action_with_payload("processed", "credit pack storage completion announcement", storage_completion_announcement)
        return {"message": "Announcement processed successfully"}
    except Exception as e:
        logger.error(f"Error processing credit pack storage completion announcement: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit pack storage completion announcement: {str(e)}")


@router.post("/credit_pack_storage_retry_request", response_model=db.CreditPackStorageRetryRequestResponse)
async def credit_pack_storage_retry_request_endpoint(
    request: db.CreditPackStorageRetryRequest = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            request.requesting_end_user_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        response = await service_functions.process_credit_pack_storage_retry_request(request)
        response_dict = response.model_dump()
        response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in response_dict.items()}
        service_functions.log_action_with_payload("processed", "credit pack storage retry request", response_dict)
        return response
    except Exception as e:
        logger.error(f"Error processing credit pack storage retry request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing credit pack storage retry request: {str(e)}")
    
    
@router.post("/credit_pack_storage_retry_completion_announcement")
async def credit_pack_storage_retry_completion_announcement_endpoint(
    response: db.CreditPackStorageRetryRequestResponse = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            response.responding_supernode_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        await service_functions.process_credit_pack_storage_retry_completion_announcement(response)
        service_functions.log_action_with_payload("processed", "credit pack storage retry completion announcement", json.dumps(response))
        return {"message": "Announcement processed successfully"}
    except Exception as e:
        logger.error(f"Error processing credit pack storage retry completion announcement: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing credit pack storage retry completion announcement: {str(e)}")


@router.post("/get_valid_credit_pack_tickets_for_pastelid", response_model=List[dict])
async def get_valid_credit_pack_tickets_for_pastelid_endpoint(
    pastelid: str = Body(..., description="The PastelID to retrieve credit pack tickets for"),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(pastelid, challenge_signature, challenge_id)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        valid_tickets = await service_functions.get_valid_credit_pack_tickets_for_pastelid(pastelid)
        return valid_tickets
    except Exception as e:
        logger.error(f"Error retrieving valid credit pack tickets for PastelID {pastelid}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving valid credit pack tickets: {str(e)}")
    
    
@router.post("/check_credit_pack_balance", response_model=Dict[str, Any])
async def check_credit_pack_balance_endpoint(
    credit_pack_ticket_txid: str = Body(..., description="The transaction ID of the credit pack ticket"),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(credit_pack_ticket_txid, challenge_signature, challenge_id)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        current_credit_balance, number_of_transactions = await service_functions.determine_current_credit_pack_balance_based_on_tracking_transactions_new(credit_pack_ticket_txid)
        balance_info = {
            "current_credit_balance": current_credit_balance,
            "number_of_confirmation_transactions": number_of_transactions
        }
        logger.info(f"Checked credit pack balance for txid {credit_pack_ticket_txid}: {balance_info}")
        return balance_info
    except Exception as e:
        logger.error(f"Error checking credit pack balance for txid {credit_pack_ticket_txid}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error checking credit pack balance: {str(e)}")
    
    
@router.post("/retrieve_credit_pack_ticket_from_purchase_burn_txid", response_model=Dict[str, Any])
async def retrieve_credit_pack_ticket_endpoint(
    purchase_burn_txid: str = Body(..., description="The transaction ID of the credit pack purchase burn"),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(purchase_burn_txid, challenge_signature, challenge_id)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        credit_pack_purchase_request, credit_pack_purchase_request_response, credit_pack_purchase_request_confirmation = await service_functions.retrieve_credit_pack_ticket_from_purchase_burn_txid(purchase_burn_txid)
        if credit_pack_purchase_request is None or credit_pack_purchase_request_response is None or credit_pack_purchase_request_confirmation is None:
            raise HTTPException(status_code=404, detail="Credit pack ticket not found")
        ticket_info = {
            "credit_pack_purchase_request": credit_pack_purchase_request.model_dump(),
            "credit_pack_purchase_request_response": credit_pack_purchase_request_response.model_dump(),
            "credit_pack_purchase_request_confirmation": credit_pack_purchase_request_confirmation.model_dump()
        }
        logger.info(f"Retrieved credit pack ticket for purchase burn txid {purchase_burn_txid}: {ticket_info}")
        return ticket_info
    except Exception as e:
        logger.error(f"Error retrieving credit pack ticket for purchase burn txid {purchase_burn_txid}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error retrieving credit pack ticket: {str(e)}")
    
    
@router.post("/get_final_credit_pack_registration_txid_from_credit_purchase_burn_txid", response_model=Dict[str, Any])
async def get_final_credit_pack_registration_txid_endpoint(
    purchase_burn_txid: str = Body(..., description="The transaction ID of the credit pack purchase burn"),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(purchase_burn_txid, challenge_signature, challenge_id)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        sha3_256_hash_of_credit_pack_purchase_request_fields = await service_functions.get_final_credit_pack_registration_txid_from_credit_purchase_burn_txid(purchase_burn_txid)
        if sha3_256_hash_of_credit_pack_purchase_request_fields is None:
            raise HTTPException(status_code=404, detail="Credit pack ticket not found")
        logger.info(f"Retrieved final credit pack registration txid for purchase burn txid {purchase_burn_txid}: {sha3_256_hash_of_credit_pack_purchase_request_fields}")
        return {"final_credit_pack_registration_txid": sha3_256_hash_of_credit_pack_purchase_request_fields}
    except Exception as e:
        logger.error(f"Error retrieving final credit pack registration txid for purchase burn txid {purchase_burn_txid}: {str(e)}")
        traceback.print_exc()        
        raise HTTPException(status_code=500, detail=f"Error retrieving final credit pack registration txid: {str(e)}")

    
#__________________________________________________________________________________________________________


@router.post("/make_inference_api_usage_request", response_model=db.InferenceAPIUsageResponse)
async def make_inference_api_usage_request_endpoint(
    inference_api_usage_request: db.InferenceAPIUsageRequest = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(
            inference_api_usage_request.requesting_pastelid, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        # Validate and process the inference API usage request
        inference_response = await service_functions.process_inference_api_usage_request(inference_api_usage_request)
        inference_request_dict = inference_api_usage_request.model_dump()
        inference_request_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_request_dict.items()}
        # Abbreviate the 'model_input_data_json_b64' field to the first 32 characters
        inference_request_dict['model_input_data_json_b64'] = inference_request_dict['model_input_data_json_b64'][:32]        
        inference_response_dict = inference_response.model_dump()
        inference_response_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_response_dict.items()}
        combined_message_dict = {**inference_request_dict, **inference_response_dict}
        # Broadcast message to nearest SNs to requester's pastelid containing inference request/response message 
        response_message_body = json.dumps(combined_message_dict)
        response_message_type = "inference_request_response_announcement_message"
        _ = await service_functions.broadcast_message_to_n_closest_supernodes_to_given_pastelid(inference_api_usage_request.requesting_pastelid, response_message_body, response_message_type) 
        # Return the InferenceAPIUsageResponse as the API response
        return inference_response
    except Exception as e:
        logger.error(f"Error encountered with inference API usage request: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error encountered with inference API usage request: {str(e)}")
    

@router.post("/confirm_inference_request", response_model=db.InferenceConfirmation)
async def confirm_inference_request_endpoint(
    inference_confirmation: db.InferenceConfirmation = Body(...),
    challenge: str = Body(..., description="The challenge string"),
    challenge_id: str = Body(..., description="The ID of the challenge string"),
    challenge_signature: str = Body(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature_from_inference_request_id(
            inference_confirmation.inference_request_id, challenge_signature, challenge_id
        )
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        # Process the inference confirmation
        is_processed = await service_functions.process_inference_confirmation(
            inference_confirmation.inference_request_id, inference_confirmation
        )
        if is_processed:
            logger.info(f"Inference request {inference_confirmation.inference_request_id} confirmed successfully")
        else:
            logger.error(f"Error confirming inference request {inference_confirmation.inference_request_id}")
        return inference_confirmation
    except Exception as e:
        logger.error(f"Error sending inference confirmation: {str(e)}")
        traceback.print_exc()       
        raise HTTPException(status_code=500, detail=f"Error sending inference confirmation: {str(e)}")


@router.get("/check_status_of_inference_request_results/{inference_response_id}")
async def check_status_of_inference_request_results_endpoint(inference_response_id: str):
    try:
        request_result_is_available = await service_functions.check_status_of_inference_request_results(inference_response_id)
        if request_result_is_available is None:
            raise HTTPException(status_code=404, detail="Inference request result not found")
        return request_result_is_available
    except ValueError as ve:
        logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(ve)}")
    except Exception as e:
        logger.error(f"Error checking status of inference request results: {str(e)}")
        traceback.print_exc()        
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/retrieve_inference_output_results", response_model=db.InferenceAPIOutputResult)
async def retrieve_inference_output_results_endpoint(
    inference_response_id: str = Query(..., description="The ResponseID Associated with the Inference Request"),
    pastelid: str = Query(..., description="The PastelID of the requesting party"),
    challenge: str = Query(..., description="The challenge string"),
    challenge_id: str = Query(..., description="The ID of the challenge string"),
    challenge_signature: str = Query(..., description="The signature of the PastelID on the challenge string"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        is_valid_signature = await service_functions.verify_challenge_signature(pastelid, challenge_signature, challenge_id)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        inference_output_results = await service_functions.get_inference_output_results_and_verify_authorization(inference_response_id, pastelid)
        # Broadcast message to nearest SNs to requester's pastelid containing inference results
        inference_output_results_dict = inference_output_results.model_dump()
        inference_output_results_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_output_results_dict.items()}
        # Retrieve the inference API usage request from the database
        inference_usage_request_object = await service_functions.get_inference_api_usage_request_for_audit(inference_output_results_dict['inference_request_id'])
        inference_usage_request_dict = inference_usage_request_object.model_dump()
        inference_usage_request_dict = {k: (str(v) if isinstance(v, uuid.UUID) else v) for k, v in inference_usage_request_dict.items()}
        # Add model_parameters_json_b64 and other fields to the inference output results dict:
        inference_output_results_dict['model_parameters_json_b64'] = inference_usage_request_dict['model_parameters_json_b64']
        inference_output_results_dict['requested_model_canonical_string'] = inference_usage_request_dict['requested_model_canonical_string']
        inference_output_results_dict['model_inference_type_string'] = inference_usage_request_dict['model_inference_type_string']
        # Abbreviate the 'inference_result_json_base64' field to the first 32 characters
        inference_output_results_dict['inference_result_json_base64'] = inference_output_results_dict['inference_result_json_base64'][:32]
        result_message_body = json.dumps(inference_output_results_dict)
        result_message_type = "inference_request_result_announcement_message"
        _ = await service_functions.broadcast_message_to_n_closest_supernodes_to_given_pastelid(pastelid, result_message_body, result_message_type) 
        return inference_output_results
    except Exception as e:
        logger.error(f"Error retrieving inference output results: {str(e)}")
        traceback.print_exc()        
        raise HTTPException(status_code=500, detail=f"Error retrieving inference output results: {str(e)}")


@router.post("/audit_inference_request_response", response_model=db.InferenceAPIUsageResponse)
async def audit_inference_request_response_endpoint(
    inference_response_id: str = Body(..., description="The inference response ID"),
    pastel_id: str = Body(..., description="The PastelID of the requester"),
    signature: str = Body(..., description="The signature of the PastelID on the inference_response_id"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        # Retrieve the InferenceAPIOutputResult from the local database
        api_usage_response = await service_functions.get_inference_api_usage_response_for_audit(inference_response_id)
        if api_usage_response is None:
            raise HTTPException(status_code=404, detail="Inference result not found")
        # Verify the signature
        is_valid_signature = await service_functions.verify_message_with_pastelid_func(pastel_id, inference_response_id, signature)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        api_usage_request = await service_functions.get_inference_api_usage_request_for_audit(api_usage_response.inference_request_id)
        # Verify that the PastelID matches the one in the response
        if api_usage_request.requesting_pastelid != pastel_id:
            raise HTTPException(status_code=403, detail="PastelID does not match the one in the inference request")
        # Return the InferenceAPIOutputResult as the API response
        return api_usage_response
    except Exception as e:
        logger.error(f"Error auditing inference request result: {str(e)}")
        traceback.print_exc()        
        raise HTTPException(status_code=500, detail=f"Error auditing inference request result: {str(e)}")
    
    
@router.post("/audit_inference_request_result", response_model=db.InferenceAPIOutputResult)
async def audit_inference_request_result_endpoint(
    inference_response_id: str = Body(..., description="The inference response ID"),
    pastel_id: str = Body(..., description="The PastelID of the requester"),
    signature: str = Body(..., description="The signature of the PastelID on the inference_response_id"),
    rpc_connection=Depends(get_rpc_connection),
):
    try:
        # Retrieve the InferenceAPIOutputResult from the local database
        api_usage_result = await service_functions.get_inference_api_usage_result_for_audit(inference_response_id)
        if api_usage_result is None:
            raise HTTPException(status_code=404, detail="Inference result not found")
        # Verify the signature
        is_valid_signature = await service_functions.verify_message_with_pastelid_func(pastel_id, inference_response_id, signature)
        if not is_valid_signature:
            raise HTTPException(status_code=401, detail="Invalid PastelID signature")
        api_usage_request = await service_functions.get_inference_api_usage_request_for_audit(api_usage_result.inference_request_id)
        # Verify that the PastelID matches the one in the response
        if api_usage_request.requesting_pastelid != pastel_id:
            raise HTTPException(status_code=403, detail="PastelID does not match the one in the inference request")
        # Return the InferenceAPIOutputResult as the API response
        return api_usage_result
    except Exception as e:
        logger.error(f"Error auditing inference request result: {str(e)}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error auditing inference request result: {str(e)}")
    
    
@router.get("/get_inference_model_menu")
async def get_inference_model_menu_endpoint(
    rpc_connection=Depends(get_rpc_connection),
):
    model_menu = await service_functions.get_inference_model_menu()
    return model_menu


@router.get("/download/{file_name}")
async def download_file(file_name: str):
    file_location = os.path.join(tempfile.gettempdir(), file_name)
    if file_location in service_functions.file_store and service_functions.file_store[file_location] > datetime.utcnow():
        return FileResponse(file_location, background=BackgroundTask(service_functions.remove_file, file_location))
    else:
        service_functions.remove_file(file_location)
        raise HTTPException(status_code=404, detail="File not found or expired")


@router.post("/update_inference_sn_reputation_score")
async def update_inference_sn_reputation_score_endpoint(
    reputation_score_data: db.ReputationScoreUpdate,
    rpc_connection=Depends(get_rpc_connection),
):
    is_updated = await service_functions.update_inference_sn_reputation_score(reputation_score_data.supernode_pastelid, reputation_score_data.reputation_score)
    return {"is_updated": is_updated}

@router.get("/show_logs/{minutes}", response_class=HTMLResponse)
async def show_logs(minutes: int = 5):
    # read the entire log file and generate HTML with logs up to `minutes` minutes from now
    with open("pastel_supernode_inference_layer.log", "r") as f:
        lines = f.readlines()
    logs = []
    now = datetime.now(timezone('UTC'))  # get current time, make it timezone-aware
    for line in lines:
        if line.strip() == "":
            continue
        if line[0].isdigit():
            try:  # Try to parse the datetime
                log_datetime_str = line.split(" - ")[0]  # assuming the datetime is at the start of each line
                log_datetime = datetime.strptime(log_datetime_str, "%Y-%m-%d %H:%M:%S,%f")  # parse the datetime string to a datetime object
                log_datetime = log_datetime.replace(tzinfo=timezone('UTC'))  # set the datetime object timezone to UTC to match `now`
                if now - log_datetime <= timedelta(minutes=minutes):  # if the log is within `minutes` minutes from now
                    continue  # ignore the log and continue with the next line
            except ValueError:
                pass  # If the line does not start with a datetime, ignore the ValueError and process the line anyway                        
            logs.append(service_functions.highlight_rules_func(line.rstrip('\n')))  # add the highlighted log to the list and strip any newline at the end
    logs_as_string = "<br>".join(logs)  # joining with <br> directly
    logs_as_string_newlines_rendered = logs_as_string.replace("\n", "<br>")
    logs_as_string_newlines_rendered_font_specified = """
    <html>
    <head>
    <link href="https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" rel="stylesheet">
    <script>
    var logContainer;
    var lastLogs = `{0}`;
    var shouldScroll = true;
    var userScroll = false;
    var lastPosition = 0;
    var minutes = {1};
    function fetchLogs() {{
        if (typeof minutes !== 'undefined' && typeof lastPosition !== 'undefined') {{
            fetch('/show_logs_incremental/' + minutes + '/' + lastPosition)
            .then(response => response.json())
            .then(data => {{
                if (logContainer) {{
                    var div = document.createElement('div');
                    div.innerHTML = data.logs;
                    if (div.innerHTML) {{
                        lastLogs += div.innerHTML;
                        lastPosition = data.last_position;
                    }}
                    logContainer.innerHTML = lastLogs;
                    if (shouldScroll) {{
                        logContainer.scrollTop = logContainer.scrollHeight;
                    }}
                }}
            }});
        }}
    }}
    function checkScroll() {{
        if(logContainer.scrollTop + logContainer.clientHeight < logContainer.scrollHeight) {{
            userScroll = true;
            shouldScroll = false;
        }} else {{
            userScroll = false;
        }}
        if (!userScroll) {{
            setTimeout(function(){{ shouldScroll = true; }}, 10000);
        }}
    }}
    window.onload = function() {{
        let p = document.getElementsByTagName('p');
        for(let i = 0; i < p.length; i++) {{
            let color = window.getComputedStyle(p[i]).getPropertyValue('color');
            p[i].style.textShadow = `0 0 5px ${{color}}, 0 0 10px ${{color}}, 0 0 15px ${{color}}, 0 0 20px ${{color}}`;
        }}
        document.querySelector('#copy-button').addEventListener('click', function() {{
            var text = document.querySelector('#log-container').innerText;
            navigator.clipboard.writeText(text).then(function() {{
                console.log('Copying to clipboard was successful!');
            }}, function(err) {{
                console.error('Could not copy text: ', err);
            }});
        }});
        document.querySelector('#download-button').addEventListener('click', function() {{
            var text = document.querySelector('#log-container').innerText;
            var element = document.createElement('a');
            element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
            element.setAttribute('download', 'pastel_supernode_inference_layer_log__' + new Date().toISOString() + '.txt');
            element.style.display = 'none';
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }});
    }}
    document.addEventListener('DOMContentLoaded', (event) => {{
        logContainer = document.getElementById('log-container');
        logContainer.innerHTML = lastLogs;
        logContainer.addEventListener('scroll', checkScroll);
        fetchLogs();
        setInterval(fetchLogs, 1000);  // Fetch the logs every 1 second
    }});
    </script>
    </head>        
    <style>
    .log-container {{
        scroll-behavior: smooth;
        background-color: #2b2b2b; /* dark gray */
        color: #d3d3d3; /* light gray */
        background-image: linear-gradient(rgba(0,0,0,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.1) 1px, transparent 1px);
        background-size: 100% 10px, 10px 100%;
        background-position: 0 0, 0 0;
        animation: scan 1s linear infinite;
        @keyframes scan {{
            0% {{
                background-position: 0 0, 0 0;
            }}
            100% {{
                background-position: -10px -10px, -10px -10px;
            }}
        }}
        font-size: 14px;
        font-family: monospace;
        padding: 10px;
        height: 100vh;
        margin: 0;
        box-sizing: border-box;
        overflow: auto;
    }}
    .icon-button {{
        position: fixed;
        right: 10px;
        margin: 10px;
        background-color: #555;
        color: white;
        border: none;
        cursor: pointer;
        border-radius: 50%;
        width: 60px;
        height: 60px;
        font-size: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none;
    }}
    #copy-button {{
        bottom: 80px;  // Adjust this value as needed
    }}
    #download-button {{
        bottom: 10px;
    }}
    </style>
    <body>
    <pre id="log-container" class="log-container"></pre>
    <button id="copy-button" class="icon-button"><i class="fas fa-copy"></i></button>
    <button id="download-button" class="icon-button"><i class="fas fa-download"></i></button>
    </body>
    </html>""".format(logs_as_string_newlines_rendered, minutes)
    return logs_as_string_newlines_rendered_font_specified


@router.get("/show_logs", response_class=HTMLResponse)
async def show_logs_default():
    return show_logs(5)
        