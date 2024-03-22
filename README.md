# Pastel Supernode Messaging and Control Layer

![Illustration](https://raw.githubusercontent.com/pastelnetwork/python_supernode_messaging_and_control_layer/master/illustration.webp)

The Supernode messaging and control layer is implemented using Python and FastAPI. It provides a set of API endpoints that allow communication and coordination among the Supernodes in the Pastel Network.

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

Simply run `./set_up_project.sh` to install the required dependencies and start the FastAPI server.

You can change the listening port from the default `7123` by modifying the `UVICORN_PORT` environment variable in the `.env` file.

Start the FastAPI server by running:
   ```
   uvicorn main:app --reload
   ```

The server will start running at `http://localhost:7123`.

## API Documentation

The API documentation is available at `http://localhost:7123` when the server is running. It is powered by Swagger and provides detailed information about the available endpoints, request/response schemas, and example usage.

## Usage

- To retrieve Supernode messages from the last specified minutes, make a GET request to `/get_messages` with optional query parameters `last_k_minutes` and `message_type`.
- To send a message to a specific Supernode, make a POST request to `/send_message` with the required payload.
- To broadcast a message to a list of Supernodes, make a POST request to `/broadcast_message` with the required payload.

Refer to the API documentation for more details on the request and response formats.

## Monitoring and Logging

The application logs various events and errors using the configured logger. The log messages are written to the console and can be further configured to write to a file or external logging service.

The `monitor_new_messages` function runs as a background task and continuously monitors for new messages, updates the message metadata, and logs relevant information.

## Contributing

Contributions to the Pastel Supernode Messaging and Control Layer are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).
