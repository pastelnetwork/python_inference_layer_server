import service_functions
import database_code as db
from logger_config import setup_logger
from fastapi import APIRouter, Depends, Query, Request, Body
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.exceptions import HTTPException
from json import JSONEncoder
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, List
from pydantic import SecretStr
from decouple import Config as DecoupleConfig, RepositoryEnv

config = DecoupleConfig(RepositoryEnv('.env'))
TEMP_OVERRIDE_LOCALHOST_ONLY = config.get("TEMP_OVERRIDE_LOCALHOST_ONLY", default=0)
logger = setup_logger()

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
        

@router.get("/get_messages", response_model=List[db.MessageModel], dependencies=[Depends(localhost_only)])
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

    Returns a list of MessageModel objects containing the message, message_type, sending_sn_pastelid, and timestamp.
    """
    try:
        messages = await service_functions.parse_sn_messages_from_last_k_minutes_func(last_k_minutes, message_type)
        return [
            db.MessageModel(
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
async def send_message_to_list_of_sns(
    message: str = Query(..., description="Message to broadcast"),
    message_type: str = Query(..., description="Type of the message"),
    list_of_receiving_sn_pastelids: List[str] = Query(..., description="List of PastelIDs of the receiving Supernodes"),
    pastelid_passphrase: SecretStr = Query(..., description="Passphrase for the sending PastelID"),
    verbose: Optional[int] = Query(0, description="Verbose mode (0 or 1)"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Note: Endpoint only available on localhost.
        
    Broadcasts a message to a list of Supernodes.

    - `message`: Message to broadcast.
    - `message_type`: Type of the message.
    - `list_of_receiving_sn_pastelids`: List of PastelIDs of the receiving Supernodes.
    - `pastelid_passphrase`: Passphrase for the sending PastelID.
    - `verbose`: Verbose mode (0 or 1) (default: 0).

    Returns a SendMessageResponse object containing the status and message.
    """
    try:
        signed_message = await service_functions.broadcast_message_to_list_of_sns_using_pastelid_func(
            message, message_type, list_of_receiving_sn_pastelids, pastelid_passphrase.get_secret_value(), verbose
        )
        return db.SendMessageResponse(status="success", message=f"Message broadcasted: {signed_message}")
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        return db.SendMessageResponse(status="error", message=f"Error broadcasting message: {str(e)}")


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


@router.post("/send_user_message", response_model=db.SupernodeUserMessageModel)
async def send_user_message(
    user_message: db.UserMessageCreate = Body(...),
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
    - `user_message` (UserMessageCreate): The user message to be sent, including:
        - `from_pastelid` (str): The PastelID of the sender.
        - `to_pastelid` (str): The PastelID of the recipient.
        - `message_body` (str): The content of the message.
        - `message_signature` (str): The signature of the message by the sender's PastelID.
    - `challenge` (str): The challenge string obtained from the `/request_challenge` endpoint.
    - `challenge_id` (str): The ID of the challenge string.
    - `challenge_signature` (str): The signature of the PastelID on the challenge string.

    Returns:
    - `SupernodeUserMessageModel`: The sent message details, including:
        - `message` (str): The content of the sent message.
        - `message_type` (str): The type of the message (always "user_message").
        - `sending_sn_pastelid` (str): The PastelID of the Supernode that sent the message.
        - `timestamp` (datetime): The timestamp of when the message was sent.
        - `id` (int): The unique identifier of the Supernode user message.
        - `user_message` (UserMessageModel): The details of the user message, including:
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


@router.get("/get_user_messages", response_model=List[db.UserMessageModel])
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
    - List[UserMessageModel]: A list of user messages associated with the provided PastelID, each including:
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
            db.UserMessageModel(
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


@router.post("/make_inference_api_usage_request", response_model=db.InferenceAPIUsageResponseModel)
async def make_inference_api_usage_request_endpoint(
    inference_api_usage_request: db.InferenceAPIUsageRequestModel = Body(...),
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
        # Return the InferenceAPIUsageResponse as the API response
        return db.InferenceAPIUsageResponseModel(
            inference_response_id=inference_response.inference_response_id,
            inference_request_id=inference_response.inference_request_id,
            proposed_cost_of_request_in_inference_credits=inference_response.proposed_cost_of_request_in_inference_credits,
            remaining_credits_in_pack_after_request_processed=inference_response.remaining_credits_in_pack_after_request_processed,
            credit_usage_tracking_psl_address=inference_response.credit_usage_tracking_psl_address,
            request_confirmation_message_amount_in_patoshis=inference_response.request_confirmation_message_amount_in_patoshis,
            max_block_height_to_include_confirmation_transaction=inference_response.max_block_height_to_include_confirmation_transaction,
            supernode_pastelid_and_signature_on_inference_response_id=inference_response.supernode_pastelid_and_signature_on_inference_response_id
        )
    except Exception as e:
        logger.error(f"Error sending inference API usage request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending inference API usage request: {str(e)}")
    

@router.post("/confirm_inference_request", response_model=db.InferenceConfirmationModel)
async def confirm_inference_request_endpoint(
    inference_confirmation: db.InferenceConfirmationModel = Body(...),
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
            inference_confirmation.inference_request_id, inference_confirmation.confirmation_transaction
        )
        if is_processed:
            logger.info(f"Inference request {inference_confirmation.inference_request_id} confirmed successfully")
        else:
            logger.error(f"Error confirming inference request {inference_confirmation.inference_request_id}")
        return inference_confirmation
    except Exception as e:
        logger.error(f"Error sending inference confirmation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending inference confirmation: {str(e)}")


@router.get("/get_inference_output_results/{inference_response_id}", response_model=db.InferenceOutputResultsModel)
async def get_inference_output_results_endpoint(
    inference_response_id: str,
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
        inference_output_results = await service_functions.get_inference_output_results_and_verify_authorization(
            inference_response_id, pastelid)
        return inference_output_results
    except Exception as e:
        logger.error(f"Error retrieving inference output results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving inference output results: {str(e)}")
    
    
@router.get("/get_inference_model_menu")
async def get_inference_model_menu_endpoint(
    rpc_connection=Depends(get_rpc_connection),
):
    model_menu = await service_functions.get_inference_model_menu()
    return model_menu


@router.post("/update_inference_sn_reputation_score")
async def update_inference_sn_reputation_score_endpoint(
    reputation_score_data: db.ReputationScoreUpdateModel,
    rpc_connection=Depends(get_rpc_connection),
):
    is_updated = await service_functions.update_inference_sn_reputation_score(reputation_score_data.supernode_pastelid, reputation_score_data.reputation_score)
    return {"is_updated": is_updated}


@router.get("/show_logs/{minutes}", response_class=HTMLResponse)
async def show_logs(minutes: int = 5):
    # read the entire log file and generate HTML with logs up to `minutes` minutes from now
    with open("opennode_fastapi_log.txt", "r") as f:
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
            element.setAttribute('download', 'pastel_supernode_messaging_and_control_layer_log__' + new Date().toISOString() + '.txt');
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
        