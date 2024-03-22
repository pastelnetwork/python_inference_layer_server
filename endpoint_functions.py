import service_functions
from logger_config import setup_logger
from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, SecretStr
from json import JSONEncoder
from datetime import datetime, timedelta, timezone
from typing import Optional, List
logger = setup_logger()

# RPC Client Dependency
async def get_rpc_connection():
    rpc_host, rpc_port, rpc_user, rpc_password, other_flags = service_functions.get_local_rpc_settings_func() 
    return service_functions.AsyncAuthServiceProxy(f"http://{rpc_user}:{rpc_password}@{rpc_host}:{rpc_port}")

router = APIRouter()

class DateTimeEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()

class MessageModel(BaseModel):
    message: str
    message_type: str

class SendMessageResponse(BaseModel):
    status: str
    message: str

@router.get("/get_messages", response_model=List[MessageModel])
async def get_messages(
    last_k_minutes: Optional[int] = Query(10, description="Number of minutes to retrieve messages from"),
    message_type: Optional[str] = Query("all", description="Type of messages to retrieve ('all' or specific type)"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Retrieves Supernode messages from the last specified minutes.

    - `last_k_minutes`: Number of minutes to retrieve messages from (default: 10).
    - `message_type`: Type of messages to retrieve ('all' or specific type) (default: 'all').

    Returns a list of MessageModel objects containing the message and message_type.
    """
    messages = await service_functions.parse_sn_messages_from_last_k_minutes_func(last_k_minutes, message_type)
    return [MessageModel(message=msg["message"], message_type=msg["message_type"]) for msg in messages]


@router.post("/send_message", response_model=SendMessageResponse)
async def send_message(
    message: str = Query(..., description="Message to send"),
    message_type: str = Query(..., description="Type of the message"),
    receiving_sn_pastelid: str = Query(..., description="PastelID of the receiving Supernode"),
    pastelid_passphrase: SecretStr = Query(..., description="Passphrase for the sending PastelID"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
    Sends a message to a specific Supernode.

    - `message`: Message to send.
    - `message_type`: Type of the message.
    - `receiving_sn_pastelid`: PastelID of the receiving Supernode.
    - `pastelid_passphrase`: Passphrase for the sending PastelID.

    Returns a SendMessageResponse object containing the status and message.
    """
    try:
        signed_message = await service_functions.send_message_to_sn_using_pastelid_func(
            message, message_type, receiving_sn_pastelid, pastelid_passphrase.get_secret_value()
        )
        return SendMessageResponse(status="success", message=f"Message sent: {signed_message}")
    except Exception as e:
        logger.error(f"Error sending message: {str(e)}")
        return SendMessageResponse(status="error", message=f"Error sending message: {str(e)}")


@router.post("/broadcast_message", response_model=SendMessageResponse)
async def broadcast_message(
    message: str = Query(..., description="Message to broadcast"),
    message_type: str = Query(..., description="Type of the message"),
    list_of_receiving_sn_pastelids: List[str] = Query(..., description="List of PastelIDs of the receiving Supernodes"),
    pastelid_passphrase: SecretStr = Query(..., description="Passphrase for the sending PastelID"),
    verbose: Optional[int] = Query(0, description="Verbose mode (0 or 1)"),
    rpc_connection=Depends(get_rpc_connection),
):
    """
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
        return SendMessageResponse(status="success", message=f"Message broadcasted: {signed_message}")
    except Exception as e:
        logger.error(f"Error broadcasting message: {str(e)}")
        return SendMessageResponse(status="error", message=f"Error broadcasting message: {str(e)}")

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
            element.setAttribute('download', 'pastel_gateway_verification_monitor_log__' + new Date().toISOString() + '.txt');
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
        