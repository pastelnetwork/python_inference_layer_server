# Pastel Supernode Inference Layer

![Illustration](https://raw.githubusercontent.com/pastelnetwork/python_inference_layer_server/master/illustration.webp)

The Supernode Inference Layer for Pastgel Network is implemented using Python and FastAPI. It provides a set of API endpoints that allow communication and coordination among the Supernodes in the Pastel Network for the purpose of provisioning API credit packs in the Pastel Blockchain and then using these credit packs to pay for the cost of inference requests across a range of AI/LLM models from various services (including OpenAI, Anthropic, Groq, Mistral, OpenRouter, Stability, and others), and also locally hosted models on the Supernodes themselves using the related Swiss Army Llama package.

The key components of this layer include:

- Async RPC client: Enables communication with the Pastel Network's RPC interface.
- Messaging functionality: Allows sending messages between Supernodes using PastelIDs.
- Message broadcasting: Enables broadcasting messages to multiple Supernodes simultaneously.
- Message storage and retrieval: Stores messages in a local database for efficient retrieval and analysis.
- Message metadata tracking: Tracks metadata such as message counts, data sizes, and sender/receiver information.

The API endpoints provided by this layer include:

- `/get_messages`: Retrieves Supernode messages from the last specified minutes.
- `/send_message`: Sends a message to a specific Supernode.
- `/broadcast_message`: Broadcasts a message to a list of Supernodes.

## Features

- Asynchronous RPC communication with the Pastel Network using `AsyncAuthServiceProxy`.
- Efficient message monitoring and storage using SQLAlchemy and SQLite.
- Granular message metadata tracking, including sender/receiver information and message sizes.
- FastAPI-based API endpoints for retrieving messages, sending messages, and broadcasting messages.
- Background task for continuously monitoring new messages and updating metadata.
- Custom exception handling middleware for better error handling and logging.
- CORS middleware for allowing cross-origin requests.
- Utilizes uvloop for improved performance.

## Prerequisites

- Python 3.7 or higher
- Access to a running Pastel node with RPC enabled

## Installation

Instruction assume Ubuntu v22+.

Clone the repository and navigate to the project directory:

```
git clone https://github.com/pastelnetwork/python_inference_layer_server.git
cd python_inference_layer_server
```

Set up a virtual environment and activate it:

```
python -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install wheel
pip install -r requirements.txt
```

Alternatively, you can run:

```
cd ~
git clone https://github.com/pastelnetwork/python_inference_layer_server.git
cd python_inference_layer_server
chmod +x ./set_up_project.sh
./set_up_project.sh
```

This will install pyenv and use it to get Python 3.12 and create a virtual environment.

You can change the listening port from the default `7123` by modifying the `UVICORN_PORT` environment variable in the `.env` file.

Start the FastAPI server by running:
```
uvicorn main:app --reload
```

The server will start running at `http://localhost:7123`.

## Configuration

The application uses environment variables for configuration. You can set these variables in the `.env` file located in the project root directory. The following variables are required:

- `DATABASE_URL`: The URL for the SQLite database (e.g., `sqlite:///./database.db`).
- `RPC_HOST`: The hostname or IP address of the Pastel node's RPC server.
- `RPC_PORT`: The port number of the Pastel node's RPC server.
- `RPC_USER`: The username for authenticating with the RPC server.
- `RPC_PASSWORD`: The password for authenticating with the RPC server.
- `UVICORN_PORT`: The port number on which the FastAPI server should run (default: 7123).

## Endpoints

The following endpoints are available in the Pastel Supernode Inference Layer:

- `/supernode_list_json`: Retrieves the list of Supernodes as JSON data.
- `/supernode_list_csv`: Retrieves the list of Supernodes as a normalized CSV file.
- `/get_local_machine_sn_info`: Retrieves information about the local machine's Supernode status.
- `/get_sn_data_from_pastelid`: Retrieves Supernode data based on the specified PastelID.
- `/get_sn_data_from_sn_pubkey`: Retrieves Supernode data based on the specified Supernode public key.
- `/get_messages`: Retrieves Supernode messages from the last specified minutes.
- `/broadcast_message_to_all_sns`: Broadcasts a message to a list of Supernodes.
- `/request_challenge`: Request a challenge string for authentication.
- `/send_user_message`: Send a user message via Supernodes.
- `/get_user_messages/{pastelid}`: Retrieve all user messages (sent and received) for a given PastelID.
- `/get_inference_model_menu`: Retrieve the inference model menu.
- `/validate_inference_api_usage_request`: Validate an inference API usage request.
- `/process_inference_confirmation`: Process an inference confirmation.
- `/execute_inference_request`: Execute an inference request.
- `/send_inference_output_results`: Send inference output results.
- `/update_inference_sn_reputation_score`: Update the inference Supernode reputation score.
- `/show_logs/{minutes}`: Show application logs for the specified number of minutes.

Refer to the API documentation available at `http://localhost:7123` when the server is running. It is powered by Swagger and provides detailed information about the available endpoints, request/response schemas, and example usage.

## Database

The application uses SQLite as the database backend. The database file is located at `./super_node_inference_layer.sqlite` by default. The database schema is defined in the `database_code.py` file.

The following tables are used:

- `messages`: Stores the Supernode messages.
- `message_metadata`: Stores metadata about the messages.
- `message_sender_metadata`: Stores metadata about message senders.
- `message_receiver_metadata`: Stores metadata about message receivers.
- `message_sender_receiver_metadata`: Stores metadata about sender-receiver pairs.

## Logging

The application uses the `logging` module for logging. The logger is configured in the `logger_config.py` file. Log messages are written to the console and the `opennode_fastapi_log.txt` file.

The log format includes the timestamp, log level, and message. The log levels used are INFO, WARNING, and ERROR.

You can view the application logs by accessing the `/show_logs/{minutes}` endpoint, which displays the logs for the specified number of minutes.

## Usage

- To retrieve Supernode messages from the last specified minutes, make a GET request to `/get_messages` with optional query parameters `last_k_minutes` and `message_type`.
- To send a message to a specific Supernode, make a POST request to `/send_message` with the required payload.
- To broadcast a message to a list of Supernodes, make a POST request to `/broadcast_message` with the required payload.

Refer to the API documentation for more details on the request and response formats.

## Monitoring and Logging

The application logs various events and errors using the configured logger. The log messages are written to the console and can be further configured to write to a file or external logging service.

The `monitor_new_messages` function runs as a background task and continuously monitors for new messages, updates the message metadata, and logs relevant information.

## License

This project is licensed under the [MIT License](LICENSE).
