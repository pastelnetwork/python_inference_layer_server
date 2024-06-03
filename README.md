# Introduction and Overview of Pastel Inference Layer

![Illustration](https://raw.githubusercontent.com/pastelnetwork/inference_layer_documentation/master/inference_illustration.webp)


The new Pastel Inference Layer is [implemented](https://github.com/pastelnetwork/python_inference_layer_server) in Python on the server side. It works entirely with regular pasteld (`pasteld` is the Pastel Daemon binary, which is a compiled C++ binary that is the heart of the Pastel system) features accessed via the RPC (though some of this functionality has since been directly integrated as new or expanded pasteld RPC methods instead of being done externally all in Python).

There are 2 main components of the Inference Layer:


- **Credit Pack Tickets:** these are new Pastel blockchain tickets that can be created by users with the cooperation of a single responding Supernode (with verification/validation from other Supernodes before the ticket is finalized).
    - The purpose of the credit pack tickets is for the user to pre-pay for a certain amount of future inference requests ahead of time by paying a certain amount of PSL (which is actually burned) to create a credit pack ticket with a certain number of credits, where this number is determined by market prices and must be agreed upon by the majority of Supernodes. 
    - Once created, the credit pack ticket makes it simple to account for inference requests, the cost of which are denominated in terms of credits as opposed to in PSL.
        - They do this by specifying a particular PSL address as the “tracking address” for the credit pack, as well as a list of one or more PastelIDs that are authorized to spend credits from the credit pack.
        - Essentially, to pay for the cost of the credit pack, the end user has to send the full PSL cost of the credit pack to the central Pastel burn address, and this burn transaction must originate from the tracking address.
        - Once the credit pack has been created in the blockchain specifying an initial credit pack balance, credits can be consumed from the credit pack by the end user by creating inference requests. Each particular inference request has a certain cost measured in credits (e.g., generating a ChatGPT-type “text completion” of max length of 1,000 tokens might cost 41.2 inference credits), and this specific cost is told to the end user by the Responding Supernode before the request is actually done. 
        - If the user wants to move forward with the inference request at the quoted price in credits, the user does this by sending a particular small amount of PSL to the central burn address from the designated tracking address that is specified in the credit pack ticket they are using. The amount sent is a tiny bit of PSL that corresponds exactly to the number of credits that the user is authorizing to be spent from their credit pack ticket. For example, to authorize their text completion request for a cost of 41.2 credits, the end user would burn exactly 0.00412 PSL (or 412 “Patoshis”, the smallest individual unit of PSL) worth a small fraction of a penny.
        - The beauty of this system is that, beyond the initial credit pack ticket (the data for which is immutably stored forever in the Pastel blockchain as coin transactions), we don’t need to spend a lot of space in the blockchain and involve addition ticket overhead and complexity for tracking the current remaining balance of credits in a particular credit pack ticket; we can instead “encode” this data as tiny coin transactions which take up practically no space and which we already have various RPC methods in pasteld to keep track of (and we added additional methods for this purpose, such as the new `scanburntransactions` RPC method).
    - Unlike previous blockchain tickets in Pastel for Sense, Cascade, and NFTs, these new credit pack tickets work in a fundamentally different way: 
        - The cost of the credit pack ticket is paid by the end user, but this payment is burned rather than going to one or more specific Supernodes. This simplifies things a lot because we don’t need to have separate registration and activation stages and wait for block confirmations. 
        - The creation of a credit pack creates an obligation of the entire Pastel Network to honor future inference requests that are paid for using the credits in the credit pack; this is different from Cascade for example, where the registering Supernode (which receives most of the storage fee from the user rather than it being burned) is responsible for generating the storage symbols and spreading them through the network.
        - The new credit pack tickets are implemented in pasteld in a new “generic” way using a new ticket type called `contract`; the actual contents of these new `contract` tickets can be changed at will without requiring any changes in pasteld (or Gonode for that matter); pasteld is simply indexing them and making it quick and easy to retrieve them. This also means that we can use the same generic `contract` ticket for all sorts of new applications besides credit pack tickets without having to change anything in pasteld going forward.


- **Inference Requests:** these are simply REST requests made by the inference client (there is a reference client in Python, but the [primary client](https://github.com/pastelnetwork/pastel_inference_js_client) is written in JS and has a visual UI) to the Inference Layer server, which is implemented as a FastAPI application. 
    - Inference Requests can take many different forms, including:
        - Text completion (aka, LLMs, similar to ChatGPT) from a large variety of models/services, such as:
            - Various API-based services, including OpenAI, Claude3, Groq, Mistral, OpenRouter, etc. These services are *optional* according to the wishes of the Supernode operator. If the operator wants to support those models, they can include encrypted API keys for them in the [.env file](https://github.com/pastelnetwork/python_inference_layer_server/blob/master/.env) for the inference server, and they will automatically be used (and these API keys are automatically tested periodically to determine if they still work; if they don’t work, then that Supernode stops advertising support for that service). 
                - Note that there is a master list of supported models/services stored in github as the [model_menu.json](https://github.com/pastelnetwork/python_inference_layer_server/blob/master/model_menu.json) file; the local Supernode than filters this master list down to the list of models that it specifically supports, and these are advertised from a particular endpoint in the inference server code so that end users and other Supernodes can determine which models/services are supported by which specific Supernodes. In addition to specifying which models/services a Supernode supports, the model_menu.json has much more: it completely specifies in a uniform way which model parameters are available and which inference types they apply to, and the form of input and output that a model or service expects. The model_menu.json is also used in the JS Inference Client to automatically populate the UI elements instead of specifying these in a fixed way, so that updating the model_menu.json file means that everything updates automatically across the entire inference layer. It also makes adding support for new models and services easy, since you can just add new Swiss Army Llama models to the file and they will automatically work. For adding new services and model types, you also need to implement a function to submit the request to the particular service/model, but this is easy to do now with over 6 fully worked examples that all work in a similar way.
            - “Locally Hosted” LLMs using [Swiss Army Llama](https://github.com/Dicklesworthstone/swiss_army_llama); these do not require any API key and are completely decentralized/distributed, since they can be run on the Supernode machine itself doing inference on the CPU (which is slow, but still works). Such models allow for absolutely no censorship on the input prompts you can supply, and can also support models for which all the “safety” features have been removed so that they will answer questions which would be rejected by services such as OpenAI or Claude3 (e.g., “How do you hotwire a car?”). These models also support additional useful options that give you more control over the inference, such as being able to specify a grammar file to constrain the nature of the output of the model.
                - The reason why “Locally Hosted” is in quotes is because we also support running Swiss Army Llama on a remote machine. This is very useful because GPU enabled cloud instances are extremely expensive, often costing nearly $2/hour (over $1,000/month), which is much much more than the cost of running a regular CPU-only cloud instance for a Supernode, which can be as low as $30/month. But of course inference goes many times faster on a GPU enabled machine. Thus, Supernode operators can instead optionally use a service such as [vast.ai](https://vast.ai/) to rent a cheap GPU enabled instance with an Nvidia 4090 GPU for as little as $0.34/hour (around $250/month), which can be spun up and set up in just a couple minutes using a pre-made [Docker image template](https://cloud.vast.ai/?ref_id=78066&template_id=3c711e5ddd050882f0eb6f4ce1b8cc28) (which itself is mostly just using a regular [Docker image](https://hub.docker.com/repository/docker/jemanuel82/vastai_swiss_army_llama_template/general)) of Swiss Army Llama with full GPU mode enabled for everything. Then the details of this machine can be specified easily in the .env file for the inference server, and requests for Swiss Army Llama models will be seamlessly forwarded to the remote machine for execution (with a fallback to the local machine in case of an error). This means that a single GPU enabled instance can be shared between multiple Supernodes. This gives the best of both worlds in terms of decentralization/censorship resistance, inference performance, and economical low-cost operation.
        - Ask a question about an image that supports both Swiss Army Llama (via the Llava-Llama3 multimodal model) and via the GPT4o vision model.
        - Generate an image using Stable diffusion (with support easily added for others, such as Dall-e)
        - Using Swiss Army Llama to get embedding vectors for a whole document (e.g., text, html, pdf, doc, etc.) and do a semantic search against it.
        - Same for an audio file with speech, which will be automatically transcribed using Whisper and embeddings calculated optionally (along with an optional semantic search across those embeddings).


We explain above the main parts of the system, but not much about how they are actually implemented, which is quite involved and complex. The reason for the complexity is that we want the entire process to be very robust and secure. In particular, we want to ensure that:


- Users never spend money (in the form of either PSL or in Inference Credits) irreversibly but then don’t properly get the thing they paid for— either a new credit pack ticket that works, or the results of their inference request.

- Because obligations arising from the Inference Layer are “network wide” and thus shared by all Supernodes, this makes it a lot easier in theory to deal with problems. Even if a particular Supernode doesn’t follow through with what it is supposed to do for any reason (e.g., it’s being malicious, or it experienced technical issues like running out of disk space or poor network connectivity), then we want the other Supernodes to automatically realize that and have a robust process for automatically having the next Supernode step up to complete what the user has already paid for.

- We want to have confidence that each part of the complex processes involved in creating a new credit pack ticket or in creating and fulfilling a new inference request is 100% valid and correct and hasn’t been changed by any malicious parties, including the end user, the responding Supernode, or another Supernode, or even a “man in the middle”, like one of the ISPs involved in the process for the end user or the Supernodes. 
    - This means that each step in these processes is mediated by messages, and the messages are verified and validated in very particular ways. For example, the relevant fields of each message are combined in a repeatable way and hashed together, and this hash included in the message “envelope”; these message hashes are then signed by all parties involved using their PastelID private keys. This allows all parties to know for sure that the contents of the message hasn’t been altered in any way, because otherwise the hash wouldn’t match the hash that each party independently computes based on all the relevant message fields, and thus the signatures wouldn’t validate either. 
    - In addition to this, each message has the current UTC timestamp and current Pastel block height included. If a message arrives more than a minute after this timestamp or more than 2 blocks away from the specified block height, then the message is deemed invalid and is ignored.
    - For example, this means that if a user makes a request for a credit pack contain 1,000 credits at a cost of 30,000 PSL, say, then there is no way for a malicious responding Supernode to alter that request so that the end user pays the full 30,000 PSL but only gets a credit pack containing 100 credits. And the same goes for inference requests— the user is guaranteed to get what they requested and nothing more.
    
- We also want to avoid any pricing surprises for users. Before a user spends any PSL or Inference Credits, they need to fully understand what they are getting for their money. 
    - This means that requests for new credit packs are first given a price quote from the responding Supernode, and the end user has to agree to this price before moving forward, and the end user can specify a maximum PSL cost to the credit pack. Also, since end users might not know what a “reasonable” price level is for a credit pack in PSL terms, we automatically validate the pricing quote on the client side by checking the price quote using the current market price of PSL and the same rules that the Supernodes are supposed to use in arriving at the pricing of credits; if the offered price quote is too far from what it’s supposed to be, the client will automatically refuse to move forward even if the total credit pack price is below the maximum price specified by the end user.  
    - In terms of inference requests, this means that the user first fully specifies what they want to do in their inference request, and the responding Supernode does some calculations to determine how many credits the inference request will cost and tells the user; the user thus has a chance to see how much it will cost them before moving forward and spending any of their inference credits from their credit pack ticket until they are sure they want to do it at that price. Once they are ready, they indicate this by sending a confirmation transaction (we will describe this fully below), and only then are the credits permanently deducted from their credit pack.
    
- To make the system simpler and more reliable, the fundamental interaction is one between the end user and a single Supernode, called the “Responding Supernode”; in the case where we are creating a new credit pack, this is simply the Supernode that is “closest” to the end user’s PastelID using XOR distance. In the case of a new inference request, this would be the “closest” Supernode using XOR distance to the end user that also supports the exact model/service specified in the user’s inference request. But just because the end user primarily communicates with this single Responding Supernode, it doesn’t mean that the other Supernodes are kept in the dark, or that the single Responding Supernode is able to do whatever it wants without consulting the other Supernodes.
    - In the case of creating a new credit pack ticket, the Responding Supernode can propose a price quote to the end user, but then when the user confirms the price quote, that doesn’t mean that the process is over; the problem of course is that the creation of a new credit pack creates an obligation on the part of the entire network of Supernodes to honor those inference credits to complete future inference requests. 
    - So if the Responding Supernode decided to price the new credits at way below cost or the normal market price, while that might be great for that specific end user that got a special deal, it would be bad for the broader network because all the other Supernodes would then be stuck fulfilling inference requests that are unprofitable on a network level; i.e., where the users pro-rata amount of PSL burned in USD terms is much less than the USD cost of providing a particular inference request for those credits. This is an important concept here, since both the entire credit pack cost is burned, so nothing is *directly* profitable to the Supernode, but the Supernode benefits *indirectly* to the extent that PSL is burned and the supply reduced. 
    - In order to deal with this issue, there is a whole procedure by which the Responding Supernode, after giving the intial credit pack price quote in PSL to the end user and getting the end user’s agreement to the initial quote, must then communicate all the details of the new credit pack request to all the other Supernodes. Then a certain portion of all the Supernodes need to respond (the quorum percentage), and of the Supernodes that respond, a majority of those (the agreement percentage) must agree to the proposed pricing terms (i.e., agree that it’s OK to sell X inference credits for Y PSL). They indicate this agreement by signing their pastelid to the details of the entire credit pack proposal, and these signatures are included in the final credit pack data that is stored in the blockchain. 
    - Later, when the end user attempts to actually use the credit pack for an inference request, all of this information is re-validated to determine that it’s all correct and proper— that is, that the original Responding Supernode didn’t “go rogue” and offer terms that the other Supernodes wouldn’t agree to.

With those preliminaries out of the way, we can now get into the specifics of the implementations of both parts. We begin with the creation of credit pack tickets.

----------

## Credit Pack Ticket Implementation Details

Note: The following python message models are written using the [SQLModel library](https://sqlmodel.tiangolo.com/). This allows us to create a single class that can be used both as a database ORM (using sqlite via [SQLAlchemy](https://www.sqlalchemy.org/)) AND simultaneously as a response/validation model (using [Pydantic](https://docs.pydantic.dev/latest/)). Normally, we would need to specify both an ORM data model and a separate Pydantic model, since they are used for different things (ORM models are for adding, selecting, and deleting data from a database without having to use raw SQL queries, and Pydantic models are for automatically validating the form of inputs and responses to and from FastAPI endpoint functions and handling all the serialization to/from JSON).

As mentioned above, the entire flow can be viewed as an intricate exchange of messages between the end user and the Responding Supernode, and between the Responding Supernode and the rest of the Supernodes. These message exchanges are facilitated using standard REST API endpoints on the Supernodes.

For security purposes, all these endpoints require a challenge/response system to access them at all; in this process, before an end user or other Supernode (or anyone else— just call them the “client”) can access any of these endpoints, the client must first request a challenge string from the specific Supernode they want to connect to, and in this challenge request they must specify their PastelID. The Supernode then responds with a challenge, which has a random challenge ID and challenge string; the client then has to quickly sign this challenge string using their PastelID (which must match what they said their PastelID was in the original challenge request) and return both the challenge ID, challenge string, and signature, all before the challenge expires automatically. The Supernode then verifies that the signature is valid and matches the right PastelID, and only THEN will it allow the client to access the endpoint. 

Beyond that, there is security at the individual message level; you will see in the classes below that they all contain fields with names like `sha3_256_hash_of_credit_pack_purchase_request_fields` and `requesting_end_user_pastelid_signature_on_request_hash`. These are used to verify and validate ALL details of ALL messages at EVERY point in the whole process. 

Now, let’s introduce the actual flow for a user to request the creation of a new credit pack ticket:

1) First the end user picks a certain Supernode from the list ofSupernodes (the one whose pastelid has XOR distance closest to the best block hash on pastel now; call this Supernode the `responding_supernode`) and requests to purchase an credit pack ticket for a certain price using this message that is sent via a POST to the `/credit_purchase_initial_request` endpoint:

```python
    class CreditPackPurchaseRequest(SQLModel, table=True):
        id: uuid.UUID = Field(default_factory=uuid.uuid4, index=True, nullable=True)
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        requesting_end_user_pastelid: str = Field(index=True)
        requested_initial_credits_in_credit_pack: int
        list_of_authorized_pastelids_allowed_to_use_credit_pack: str = Field(sa_column=Column(JSON))
        credit_usage_tracking_psl_address: str = Field(index=True)
        request_timestamp_utc_iso_string: str
        request_pastel_block_height: int
        credit_purchase_request_message_version_string: str
        requesting_end_user_pastelid_signature_on_request_hash: str
```

2) The `responding_supernode` evaluates the message from the end user to check that all fields are valid. If anything in the request is invalid, then it responds to the POST request to the `/credit_purchase_initial_request` endpoint with a rejection message:

```python
    class CreditPackPurchaseRequestRejection(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        credit_pack_purchase_request_fields_json_b64: str
        rejection_reason_string: str
        rejection_timestamp_utc_iso_string: str
        rejection_pastel_block_height: int
        credit_purchase_request_rejection_message_version_string: str
        responding_supernode_pastelid: str = Field(index=True)
        sha3_256_hash_of_credit_pack_purchase_request_rejection_fields: str = Field(unique=True, index=True)
        responding_supernode_signature_on_credit_pack_purchase_request_rejection_hash: str
```

3) If the purchase request message fields are all valid, then the `responding_supernode` first determines the best price (in PSL per credit) that they would be willing to accept for the credit pack. To avoid wasting time and communication overhead, this preliminary price quote (preliminary because it is not considered valid by the network until it has been agreed to by enough other Supernodes) is first sent to the end user to determine if the end user is willing to agree to the quoted price (since if the end user doesn't like the price or doesn't have enough PSL to buy the credit pack for the total price, then there is no point in continuing any further). So in that case the POST request from the end user is responded to with this message:

```python
    class CreditPackPurchaseRequestPreliminaryPriceQuote(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        credit_usage_tracking_psl_address: str = Field(index=True)
        credit_pack_purchase_request_fields_json_b64: str
        preliminary_quoted_price_per_credit_in_psl: float
        preliminary_total_cost_of_credit_pack_in_psl: float
        preliminary_price_quote_timestamp_utc_iso_string: str
        preliminary_price_quote_pastel_block_height: int
        preliminary_price_quote_message_version_string: str
        responding_supernode_pastelid: str = Field(index=True)
        sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(unique=True, index=True)
        responding_supernode_signature_on_credit_pack_purchase_request_preliminary_price_quote_hash: str
```

4) The user then calls another endpoint on the`responding_supernode` (the `/credit_purchase_preliminary_price_quote_response` endpoint) to POST their response to the preliminary price quote with this message:

```python
    class CreditPackPurchaseRequestPreliminaryPriceQuoteResponse(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_fields: str = Field(index=True)
        credit_pack_purchase_request_fields_json_b64: str
        agree_with_preliminary_price_quote: bool
        credit_usage_tracking_psl_address: str = Field(index=True)
        preliminary_quoted_price_per_credit_in_psl: float
        preliminary_price_quote_response_timestamp_utc_iso_string: str
        preliminary_price_quote_response_pastel_block_height: int
        preliminary_price_quote_response_message_version_string: str
        requesting_end_user_pastelid: str = Field(index=True)
        sha3_256_hash_of_credit_pack_purchase_request_preliminary_price_quote_response_fields: str = Field(unique=True, index=True)
        requesting_end_user_pastelid_signature_on_preliminary_price_quote_response_hash: str
```

5) If the end user rejects the price quote (i.e., `agree_with_preliminary_price_quote` is false), then the process terminates now. The end user can repeat the process on the next block to try with a new `responding_supernode` that might be willing to accept a lower price. If the end user accepts the preliminary price quote (i.e., `agree_with_preliminary_price_quote` is true), then the `/credit_purchase_preliminary_price_quote_response` endpoint on the `responding_supernode` must now determine if enough other supernodes in the pastel network agree with the pricing it proposed to the end user. The `responding_supernode` does this by selecting the 12 supernodes whose hash(pastelid) has XOR distance closest to the best pastel block's merkle root; collectively these are known as the `potentially_agreeing_supernodes`. The `responding_supernode` calls a REST endpoint on each of the potentially_agreeing_supernodes called the `/credit_pack_price_agreement_request` endpoint and POSTs a message of the form:

```python
    class CreditPackPurchasePriceAgreementRequest(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(index=True)
        supernode_requesting_price_agreement_pastelid: str = Field(index=True)
        credit_pack_purchase_request_fields_json_b64: str
        credit_usage_tracking_psl_address: str = Field(index=True)
        proposed_psl_price_per_credit: float
        price_agreement_request_timestamp_utc_iso_string: str
        price_agreement_request_pastel_block_height: int
        price_agreement_request_message_version_string: str
        sha3_256_hash_of_price_agreement_request_fields: str = Field(index=True)
        supernode_requesting_price_agreement_pastelid_signature_on_request_hash: str
```

6) Each of the `potentially_agreeing_supernodes` checks all the fields of the message, including the fields of the related messages (i.e., `credit_pack_purchase_request_response_fields_json`) to determine if all fields and signatures are valid so far. Then it determines if it is willing to go along with the quoted credit pricing proposed by the `responding_supernode`. It does this by responding to the POST request to `/credit_purchase_preliminary_price_quote_response` endpoint with the following message structure:

```python
    class CreditPackPurchasePriceAgreementRequestResponse(SQLModel, table=True):
        sha3_256_hash_of_price_agreement_request_fields: str = Field(primary_key=True, index=True)
        credit_pack_purchase_request_fields_json_b64: str
        agree_with_proposed_price: bool
        credit_usage_tracking_psl_address: str = Field(unique=True,index=True)
        proposed_psl_price_per_credit: float
        proposed_price_agreement_response_timestamp_utc_iso_string: str
        proposed_price_agreement_response_pastel_block_height: int
        proposed_price_agreement_response_message_version_string: str
        responding_supernode_signature_on_credit_pack_purchase_request_fields_json_b64: str
        responding_supernode_pastelid: str = Field(index=True)
        sha3_256_hash_of_price_agreement_request_response_fields: str = Field(unique=True, index=True)
        responding_supernode_signature_on_price_agreement_request_response_hash: str
```

7) The `responding_supernode` waits for as many of the `potentially_agreeing_supernodes` to respond as possible in a set period of time (say, 30 seconds) and then aggregates all of their responses for processing. It counts up the number of valid responses received (the `valid_price_agreement_request_responses` from all of the `potentially_agreeing_supernodes` and computes
`(len(valid_price_agreement_request_responses)/len(potentially_agreeing_supernodes))` and checks that this result exceeds `0.51`. If it does not, then the entire process terminates now and the `responding_supernode` prepares a message for the end user for the next time the end user calls the REST endpoint `/check_status_of_credit_purchase_request` on the `responding_supernode`; first the end user POSTs the message:

```python
    class CreditPackRequestStatusCheck(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        requesting_end_user_pastelid: str = Field(index=True)
        requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_fields: str
```

and in the case of a termination, the `responding_supernode` responds with this message detailing why the request failed (in this case because not enough of the `potentially_agreeing_supernodes` responded in time with a valid response to the `responding_supernode`):

```python
    class CreditPackPurchaseRequestResponseTermination(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(primary_key=True, index=True)
        credit_pack_purchase_request_fields_json_b64: str
        termination_reason_string: str
        termination_timestamp_utc_iso_string: str
        termination_pastel_block_height: int
        credit_purchase_request_termination_message_version_string: str
        responding_supernode_pastelid: str = Field(index=True)
        sha3_256_hash_of_credit_pack_purchase_request_termination_fields: str = Field(unique=True, index=True)
        responding_supernode_signature_on_credit_pack_purchase_request_termination_hash: str
```

8) If enough of the `potentially_agreeing_supernodes` respond in time with valid responses, the next step is for the `responding_supernode` to tally up the responses to determine the number of `potentially_agreeing_supernodes agree` to the quoted credit price; call them the `agreeing_supernodes`. If `len(agreeing_supernodes)/len(valid_price_agreement_request_responses)` exceeds `0.85` then the price quote is deemed to be valid by the network as a whole. If `len(agreeing_supernodes)/len(valid_price_agreement_request_responses)` is LESS than `0.85`, then the entire process terminates, and when the end user next calls the `responding_supernode`'s `/check_status_of_credit_purchase_request` endpoint, the `responding_supernode` will again respond with a `CreditPackPurchaseRequestResponseTermination` (but now the `termination_reason_string` field will explain that the request terminated for a different reason— because not enough of the other Supernodes agreed to the proposed pricing). But if enough of them agreed to the pricing so that the propose credit pack is deemed to be valid by the network, then the `responding_supernode responds` with this message the next time the end user calls its `/check_status_of_credit_purchase_request` endpoint:

```python
    class CreditPackPurchaseRequestResponse(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
        credit_pack_purchase_request_fields_json_b64: str
        psl_cost_per_credit: float
        proposed_total_cost_of_credit_pack_in_psl: float
        credit_usage_tracking_psl_address: str = Field(index=True)
        request_response_timestamp_utc_iso_string: str
        request_response_pastel_block_height: int
        credit_purchase_request_response_message_version_string: str
        responding_supernode_pastelid: str = Field(index=True)
        list_of_potentially_agreeing_supernodes: str = Field(sa_column=Column(JSON))
        list_of_supernode_pastelids_agreeing_to_credit_pack_purchase_terms: str = Field(sa_column=Column(JSON))
        agreeing_supernodes_signatures_dict: str = Field(sa_column=Column(JSON))
        sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(unique=True, index=True)
        responding_supernode_signature_on_credit_pack_purchase_request_response_hash: str
```
, and the `responding_supernode` will also call the `/credit_pack_purchase_request_final_response_accouncement` endpoint on each of the `agreeing_supernodes` and will also POST the same `CreditPackPurchaseRequestResponse` message so that all of the `agreeing_supernodes` know all the details of the ticket.

9) At this point, the deal is agreed to in all particulars, and all that is left is for the end user to actually burn `proposed_total_cost_of_credit_pack_in_psl` PSL coins by sending exactly this many coins to the burn address from the end user's `credit_usage_tracking_psl_address` within 50 blocks of `request_response_pastel_block_height`. Once the end user does this, the transaction's UTXO is then communicated to the responding_supernode by the end user calling the responding_supernode's `/confirm_credit_purchase_request` endpoint by POSTing a message of the form:

```python
    class CreditPackPurchaseRequestConfirmation(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sha3_256_hash_of_credit_pack_purchase_request_fields: str = Field(foreign_key="creditpackpurchaserequest.sha3_256_hash_of_credit_pack_purchase_request_fields", index=True)
        sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(foreign_key="creditpackpurchaserequestresponse.sha3_256_hash_of_credit_pack_purchase_request_response_fields", index=True)
        credit_pack_purchase_request_fields_json_b64: str
        requesting_end_user_pastelid: str = Field(index=True)
        txid_of_credit_purchase_burn_transaction: str = Field(index=True)
        credit_purchase_request_confirmation_utc_iso_string: str
        credit_purchase_request_confirmation_pastel_block_height: int
        credit_purchase_request_confirmation_message_version_string: str
        sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str = Field(unique=True, index=True)
        requesting_end_user_pastelid_signature_on_sha3_256_hash_of_credit_pack_purchase_request_confirmation_fields: str
```

The end user also calls the `/credit_pack_purchase_completion_announcement` endpoint on each of the `agreeing_supernodes` and POSTs the same `CreditPackPurchaseRequestConfirmation` message to them to let them know that the payment was sent.

10) The `responding_supernode` then checks the details of this message and also checks the pastel blockchain directly to confirm that the transaction with txid `txid_of_credit_purchase_burn_transaction` really exists in the blockchain, was mined and confirmed by at least 3 new blocks (this check can be disabled to speed things up during debugging), and matches the expected amount exactly, and that it was indeed sent by the `credit_usage_tracking_psl_address` within 50 blocks of `request_response_pastel_block_height`. If all those things are true, then the `responding_supernode` writes the “combined ticket data” to the Pastel blockchain.

Essentially, this combined data is a nested JSON string that contains within it the JSON serialized messages for `CreditPackPurchaseRequest`, `CreditPackPurchaseRequestResponse`, and `CreditPackPurchaseRequestConfirmation`, which between them contain all the many salient pieces of information that are required to fully validate the legitimacy of the credit pack ticket— even by a newly joined Supernode that wasn’t around when the original credit pack was purchase and so never directly saw any of the many announcement messages that would have been sent to it if it were part of the network when the credit pack ticket was originally created. The process of writing this data to the blockchain, which is done by storing the Z-standard compressed JSON data in the form of Pay2FakeMultiSig coin transactions, is now completely handled within pasteld itself using a new RPC method, which is access in the Python Inference server code like this:

```python
    ticket_register_command_response = await rpc_connection.tickets('register', 'contract', ticket_json_b64, ticket_type_identifier, ticket_input_data_fully_parsed_sha3_256_hash)
```

This command returns a single Pastel TXID which uniquely picks out the credit pack ticket and is the primary way we refer to the completed credit pack ticket for all future operations (note that, of course, this TXID is itself not in the credit pack data, since that would create a “chicken and egg” problem.) For example, if we want to retrieve the credit pack from the blockchain we can use the RPC method in pasteld from Python like this and quickly parse out all the relevant data to get back a Python dict variable that has everything we’d need to fully validate the ticket contents:

```python
    ticket_get_command_response = await rpc_connection.tickets('get', ticket_txid , 1)
```

Because it would be wasteful to constantly get data from the blockchain and parse it, we only need to do this the first time (or if there is a chain re-org or rollback, which is detected and handled automatically in the Python Inference Server code) and the parsed data is serialized and turned into records in the Supernode’s local SQLite database. This is also true for all messages that the Supernode receives from end users or other Supernodes, as well as announcement messages that are sent using the built-in Pastel Masternode messaging system; all of these are automatically added to the database for ease of querying and efficiency (this is the beauty of using SQLmodel— all messages that are sent/received are automatically validated against the model and can be losslessly turned into the underlying ORM data models from the raw JSON, and can then be inserted easily into the database without a lot of annoying and verbose conversion/ingestion code).

After the ticket has been written to the blockchain successfully by the `responding_supernode`, it finally responds to the end user with this message:

```python
    class CreditPackPurchaseRequestConfirmationResponse(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
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
```

In addition, the `responding_supernode` calls the `/credit_pack_storage_completion_announcement` endpoint on all of the `agreeing_supernodes` and POSTs the same `CreditPackPurchaseRequestConfirmationResponse` message to them to let them know that the process has been successfully completed and everything is done and the credit pack ticket has been written to the blockchain correctly and is now valid and ready to be used, as well as the final TXID for the credit pack ticket.

11) If, for whatever reason, the `responding_supernode` is unable or unwilling to actually store the ticket data in the blockchain, but the end user has already burned the required PSL as needed, then we still need to deal with this situation, because otherwise it's very unfair to the end user. Luckily, any of the `agreeing_supernodes` can be called upon if needed by the end user to do this. But the end user is only permitted to ask one of the `agreeing_supernodes` to do this for them if more than 10 Pastel blocks have elapsed since the `CreditPackPurchaseRequestConfirmation` was sent by the end user without the original `responding_supernode` sending the `CreditPackPurchaseRequestConfirmationResponse` to the `agreeing_supernodes`. In that case, the end user chooses the "closest" of the `agreeing_supernodes` to the end user's pastelid (i.e., the XOR distance of `hash(agreeing_supernode_pastelid)` to `hash(end_user_pastelid)` is smallest of all the `agreeing_supernodes`; call this the `closest_agreeing_supernode`) and then the end user calls the `/credit_pack_storage_retry_request` endpoint on the `closest_agreeing_supernode` and POSTs the message:

```python
    class CreditPackStorageRetryRequest(SQLModel, table=True):
        sha3_256_hash_of_credit_pack_purchase_request_response_fields: str = Field(primary_key=True, index=True)
        credit_pack_purchase_request_fields_json_b64: str
        requesting_end_user_pastelid: str = Field(index=True)
        closest_agreeing_supernode_to_retry_storage_pastelid: str = Field(index=True)
        credit_pack_storage_retry_request_timestamp_utc_iso_string: str
        credit_pack_storage_retry_request_pastel_block_height: int
        credit_pack_storage_retry_request_message_version_string: str
        sha3_256_hash_of_credit_pack_storage_retry_request_fields: str
        requesting_end_user_pastelid_signature_on_credit_pack_storage_retry_request_hash: str
```

12) At this point, the `closest_agreeing_supernode` will check all the details to ensure that the retry request is valid (particularly, that the original `responding_supernode` has still not confirmed that the ticket storage took place successfully) in all respects, and if so, it will store the ticket in the Pastel blockchain itself. It can do this, including having the original `responding_supernode`'s pastelid signature on the ticket, because earlier in the process, the `responding_supernode` called the `closest_agreeing_supernode`'s `credit_pack_purchase_completion_announcement` endpoint, thus supplying all these details automatically. So now, the `closest_agreeing_supernode` simply stores that exact ticket data itself in the blockchain. When done, it responds to the end user's request to its `/credit_pack_storage_retry_request` endpoint with the following message:

```python
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
```

That completes the entire flow from start to finish of creating a new credit pack ticket. 


----------

In this next section, we will go into more detail about how the various service functions that enable this functionality are implemented, with a particular focus on the provisioning and validation of new and existing credit pack tickets. 

First off, we should discuss how credit pack tickets are priced in PSL terms in the first place; that is, “for X number of inference credits, the total cost should be Y PSL.”  The basic idea here is to keep the price of inference credits relatively stable in USD terms even if the market price of PSL (i.e,. how much USD 1.0 PSL is worth) is moving around quite a bit. Another goal in the pricing is to keep the cost of inference credits in line with the underlying cost to serve requests, plus a 10% "theoretical profit margin" for the Supernodes that are serving these inference requests. Note that this isn't a *real* profit margin, since the Supernodes don't actually receive *any* of the cost of the credit pack tickets— this is all burned by the end user. However, this coin burning does reduce the total supply of PSL outstanding which benefits all holders, including and especially Supernode operators. The profit margin concept is more to ensure that the intrinsic economics of the broader pastel inference system are economically viable.

Various service functions are used to do this. Let's go through each function and explain their purpose and how they contribute to the estimation process:


- **`fetch_current_psl_market_price`**:
    - This function retrieves the current market price of PSL (Pastel) from two sources: CoinMarketCap and CoinGecko.
    - It sends HTTP requests to the respective APIs and extracts the PSL price in USD from the responses.
    - If the price cannot be retrieved from either source, it retries after a short delay.
    - It calculates the average price based on the available prices from the sources.
    - The function validates the average price to ensure it falls within a reasonable range.
    - It returns the average PSL price in USD.
    
- **`estimated_market_price_of_inference_credits_in_psl_terms`**:
    - This function estimates the market price of inference credits in PSL terms.
    - It first retrieves the current PSL market price in USD using the `fetch_current_psl_market_price` function.
    - It then calculates the cost per credit in USD, considering a target value per credit and a target profit margin.
    - The target value per credit represents the underlying cost to serve inference requests, while the profit margin ensures the economic viability of the Pastel inference system.
    - The cost per credit in USD is converted to PSL terms by dividing it by the current PSL market price.
    - The function returns the estimated market price of 1.0 inference credit in PSL.
    
- **`calculate_price_difference_percentage`** (in the inference client):
    - This function calculates the percentage difference between a quoted price and an estimated price.
    - It takes the quoted price and estimated price as input and computes the absolute difference between them.
    - The difference is then divided by the estimated price to obtain the percentage difference.
    - It raises an error if the estimated price is zero to avoid division by zero.
    - The function returns the price difference percentage.
    
- **`confirm_preliminary_price_quote`** (in the inference client):
    - This function confirms the preliminary price quote for a credit pack purchase.
    - It takes the preliminary price quote, maximum total credit pack price, and maximum per credit price as input.
    - If the maximum prices are not provided, it uses default values.
    - It extracts the quoted price per credit, quoted total price, and requested credits from the preliminary price quote.
    - It estimates the fair market price for the credits using the `estimated_market_price_of_inference_credits_in_psl_terms` function.
    - It calculates the price difference percentage between the quoted price and the estimated fair price using the `calculate_price_difference_percentage` function.
    - The function compares the quoted prices with the maximum prices and the estimated fair price.
    - If the quoted prices are within the acceptable range and the price difference percentage is below a certain threshold, it confirms the preliminary price quote.
    - Otherwise, it logs a warning message and rejects the price quote.
    internal_estimate_of_credit_pack_ticket_cost_in_psl (in the inference client):
    - This function provides an internal estimate of the cost of a credit pack ticket in PSL terms.
    - It takes the desired number of credits and a price cushion percentage as input.
    - It retrieves the estimated market price per credit in PSL using the `estimated_market_price_of_inference_credits_in_psl_terms` function.
    - It calculates the estimated total cost of the ticket by multiplying the desired number of credits, estimated price per credit, and the price cushion percentage.
    - The price cushion percentage allows for some flexibility in the pricing to account for market fluctuations or other factors.
    - The function returns the estimated total cost of the credit pack ticket in PSL.

These functions work together to estimate the cost of a credit pack ticket in PSL terms, considering the desired number of credits and the underlying cost to serve inference requests. The `fetch_current_psl_market_price` function retrieves the current PSL market price, which is used by the `estimated_market_price_of_inference_credits_in_psl_terms` function to estimate the fair market price of inference credits in PSL.

The `confirm_preliminary_price_quote` function in the inference client uses these estimates to validate the quoted prices from the responding supernode. It ensures that the quoted prices are within acceptable ranges and not significantly different from the estimated fair market price. This helps protect the user from potential price manipulation or uninformed pricing.

The `internal_estimate_of_credit_pack_ticket_cost_in_psl` function provides an internal estimate of the total cost of a credit pack ticket based on the desired number of credits and a price cushion percentage. This estimate can be used by the client to set reasonable maximum prices when requesting a credit pack. Overall, these functions contribute to the estimation and validation of credit pack ticket costs in PSL terms, ensuring that the prices are economically viable for the Pastel inference system and fair for the users.


----------

Next, we go into the various service functions for validating existing credit pack tickets. Before any credit pack ticket can be used, it’s not enough for the end user to have a TXID that points to a credit pack ticket in the Pastel Blockchain; theoretically, something that looks like a credit pack ticket could be created by ANY user without it being valid. So the mere fact that a ticket exists in the blockchain doesn’t mean that it’s valid. We also can’t rely on pasteld to automatically validate the ticket in the same way that it does for Sense, Cascade, and NFT tickets, because in this new `contract` ticket type, pasteld just stores the raw ticket data without ever having to decode and parse it, let alone validate it.

This is what gives us the flexibility and power to dynamically revise the credit pack ticket fields/structure (which is also facilitated by the inclusion of a “message_version_string” in nearly every message, so we can deal with changes in a robust way to the inference protocol) and also to create completely new kinds of `contract` tickets for new applications (e.g., some kind of generic “smart contract” application that can run little WASM containers, similar to AWS Lambda functions). But it also means that all of the detailed level validation of the credit pack ticket data itself needs to be done by us in the Inference Layer server code in Python.

That’s no problem though— it’s easy enough to validate a credit pack ticket in Python. We just need to get the ticket data from the blockchain, parse out the component messages, and run them through the same validation process that we apply when the Supernode receives a message via a REST endpoint. In fact, we should review that function, because it’s quite long and intricate, because we use the same validation function to check ALL the various kinds of messages that are sent around during the process of creating a new credit pack (in fact, we also reuse this same function for inference request message validation). You can read the entire code for the function [here](https://github.com/pastelnetwork/python_inference_layer_server/blob/4176bb6e3e52cdeb6e3c61f291b2944e800b764f/service_functions.py#L5885).




Below is a detailed explanation in words of how the function works:

The `validate_credit_pack_ticket_message_data_func` is a comprehensive validation function that performs several checks on a received SQLModel instance to ensure the integrity and validity of the data. Here's a detailed explanation of how the function verifies the correctness of the message and ensures it hasn't been tampered with:


1. Timestamp Validation:
    - The function iterates through all the fields in the model instance and checks for fields ending with "_timestamp_utc_iso_string".
    - For each timestamp field, it attempts to convert the field value to a datetime object using `pd.to_datetime()`. If the conversion fails, it indicates an invalid timestamp format, and an error is appended to the `validation_errors` list.
    - Additionally, the function compares the timestamp in the field with the current timestamp using the `compare_datetimes()` function. If the difference between the timestamps exceeds a certain threshold (`MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS`), it suggests that the timestamp is too far from the current time, and an error is appended to the `validation_errors` list.
    
2. Pastel Block Height Validation:
    - The function retrieves the best block hash, Merkle root, and block height using the `get_best_block_hash_and_merkle_root_func()` function.
    - It then iterates through the fields in the model instance and checks for fields ending with "_pastel_block_height".
    - For each block height field, it compares the field value with the current block height obtained from the Pastel network. If the absolute difference between the field value and the current block height exceeds `MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS`, it indicates a discrepancy, and an error is appended to the `validation_errors` list.
    
3. Hash Validation:
    - The function computes the expected SHA3-256 hash of the response fields using the `compute_sha3_256_hash_of_sqlmodel_response_fields()` function.
    - It then searches for a field name starting with "sha3_256_hash_of" and ending with "_fields" to locate the hash field in the model instance.
    - If a hash field is found, the function compares the actual hash value stored in the field with the computed expected hash. If the hashes don't match, it indicates that the response fields have been modified, and an error is appended to the `validation_errors` list.
    
4. Pastel ID Signature Validation:
    - The function handles signature validation differently depending on the type of the model instance.
    - For `CreditPackPurchasePriceAgreementRequestResponse` instances:
        - It identifies the first Pastel ID field and signature fields based on naming conventions.
        - If the corresponding Pastel ID field is found, it verifies each signature field using the `verify_message_with_pastelid_func()` function, passing the Pastel ID, the message to verify (either the hash or the JSON-encoded request fields), and the signature.
        - If any signature verification fails, an error is appended to the `validation_errors` list.
    - For other model instances:
        - It identifies the last signature field and the last hash field based on naming conventions.
        - If both fields are found, it checks for the presence of the corresponding Pastel ID field.
        - If the Pastel ID field is found or not applicable (in case of combined Pastel ID and signature fields), it extracts the Pastel ID and signature from the fields.
        - It then verifies the signature using the `verify_message_with_pastelid_func()` function, passing the Pastel ID, the message to verify (the hash field value), and the signature.
        - If the signature verification fails, an error is appended to the `validation_errors` list.
        

Finally, the function returns the list of validation errors (`validation_errors`). If the list is empty, it indicates that the message passed all the validation checks successfully. If there are any errors, they are included in the returned list for further handling or reporting. By performing these comprehensive validations, the function ensures that the received message adheres to the expected format, timestamps are within acceptable ranges, block heights match the current network state, hashes are consistent, and signatures are valid. This helps in detecting any modifications or tampering of the message during transmission or by malicious actors.

In addition to the main validation function, several helper functions support the detailed validation process:


1. `compare_datetimes(datetime_input1, datetime_input2)`:
    - This function ensures that the datetime inputs are converted to timezone-aware datetime objects. It calculates the difference in seconds between the two datetime inputs and checks if they are within an acceptable range (`MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS`). If the difference is too large, a warning is logged.
    
2. `get_sha256_hash_of_input_data_func(input_data_or_string)`:
    - This function computes the SHA3-256 hash of the input data. If the input is a string, it is encoded to bytes before hashing. The function returns the hexadecimal representation of the hash.
    
3. `sort_dict_by_keys(input_dict)`:
    - This function sorts a dictionary and its nested dictionaries by keys. It converts the sorted dictionary into a JSON string, which can be useful for consistent and reproducible hash calculations.
    
4. `extract_response_fields_from_credit_pack_ticket_message_data_as_json_func(model_instance: SQLModel)`:
    - This asynchronous function extracts the response fields from a SQLModel instance and converts them into a JSON string. It handles various data types (e.g., datetime, list, dict, decimal) and ensures the fields are sorted for consistent hash computation.
    
5. `async def compute_sha3_256_hash_of_sqlmodel_response_fields(model_instance: SQLModel)`:
    - This asynchronous function uses `extract_response_fields_from_credit_pack_ticket_message_data_as_json_func` to get the JSON representation of the response fields and then computes the SHA3-256 hash of this JSON string.
    
6. `validate_credit_pack_blockchain_ticket_data_field_hashes(model_instance: SQLModel)`:
    - This asynchronous function validates the hashes of the response fields in a SQLModel instance. It compares the expected hash (computed from the response fields) with the actual hash stored in the model instance. If the hashes don't match, it appends an error to the `validation_errors` list.
    

Now we can also look at the full function for validating an existing credit pack ticket (`validate_existing_credit_pack_ticket`); the code for this function can be found [here](https://github.com/pastelnetwork/python_inference_layer_server/blob/4176bb6e3e52cdeb6e3c61f291b2944e800b764f/service_functions.py#L2969).


And a detailed explanation of how it works in words is below:

The `validate_existing_credit_pack_ticket` function is an asynchronous function that takes a `credit_pack_ticket_txid` as input and performs a comprehensive validation of an existing Pastel credit pack ticket. The purpose of this function is to ensure that the credit pack ticket is valid and has not been tampered with, even if the Supernode performing the validation was not present when the ticket was initially created. Here's a detailed breakdown of how the function works:


1. The function starts by retrieving the credit pack ticket data from the blockchain using the `retrieve_credit_pack_ticket_from_blockchain_using_txid` function. This function returns three objects: `credit_pack_purchase_request`, `credit_pack_purchase_request_response`, and `credit_pack_purchase_request_confirmation`.
2. It initializes a `validation_results` dictionary to store the validation results, including the overall validity of the ticket, individual validation checks, and any validation failure reasons.
3. Payment Validation:
    - The function calls the `check_burn_transaction` function to validate the payment associated with the credit pack ticket.
    - It checks if a matching or exceeding burn transaction exists for the specified `txid_of_credit_purchase_burn_transaction`, `credit_usage_tracking_psl_address`, `proposed_total_cost_of_credit_pack_in_psl`, and `request_response_pastel_block_height`.
    - If a matching or exceeding transaction is found, the payment is considered valid. Otherwise, the ticket is marked as invalid, and the reason is added to the `validation_failure_reasons_list`.
4. Supernode Validation:
    - The function retrieves the count and details of active supernodes at the time of the credit pack purchase using the `fetch_active_supernodes_count_and_details` function, based on the `request_response_pastel_block_height`.
    - It checks if all the potentially agreeing supernodes and agreeing supernodes listed in the ticket were valid supernodes at the time of the purchase by comparing their pastelids with the list of active supernodes.
    - If any supernode is not found in the list of active supernodes, the ticket is marked as invalid, and the reason is added to the `validation_failure_reasons_list`.
5. Ticket Response Hash Validation:
    - The function validates the hashes of the credit pack purchase request response and confirmation objects using the `validate_credit_pack_blockchain_ticket_data_field_hashes` function.
    - It checks if the computed hashes match the hashes included in the ticket objects.
    - If any hash mismatch is detected, the ticket is marked as invalid, and the reason is added to the `validation_failure_reasons_list`.
6. Signature Validation:
    - The function verifies the signatures of the agreeing supernodes using the `verify_message_with_pastelid_func` function.
    - It iterates over each agreeing supernode pastelid and verifies their signature on the `credit_pack_purchase_request_fields_json_b64` field.
    - If any signature fails validation, the ticket is marked as invalid, and the reason is added to the `validation_failure_reasons_list`.
7. Agreeing Supernodes Validation:
    - The function calculates the quorum percentage and agreeing percentage based on the number of potentially agreeing supernodes, agreeing supernodes, and total active supernodes at the time of the purchase.
    - It checks if the quorum percentage and agreeing percentage meet the required thresholds (`SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE` and `SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE`).
    - If either the quorum or agreeing percentage is not met, the ticket is marked as invalid, and the reason is added to the `validation_failure_reasons_list`.
8. Handling Validation Failures:
    - If any validation failures are detected, the function logs the failure reasons and adds the invalid credit pack ticket TXID to a "known bad" table in the database using the `insert_credit_pack_ticket_txid_into_known_bad_table_in_db` function.
9. Finally, the function returns the `validation_results` dictionary, which includes the overall validity of the ticket, individual validation checks, and any validation failure reasons.

This function ensures the integrity and validity of an existing Pastel credit pack ticket by performing comprehensive validations, including payment validation, supernode validation, hash validation, signature validation, and agreeing supernodes validation. By checking against historical data and verifying the consistency of the ticket data, the function can determine the validity of a credit pack ticket even if the validating supernode was not present during the ticket's creation.

We also offer an convenience endpoint for use by an end user where the user can specify their PastelID and get back a list of all their valid credit pack tickets (this function can also determine the current credit balance of those tickets; how this works will be explained in more detail below when we explain the flow for Inference Requests). The code for that function can be found [here](https://github.com/pastelnetwork/python_inference_layer_server/blob/4176bb6e3e52cdeb6e3c61f291b2944e800b764f/service_functions.py#L3091).

And here is the breakdown of how that function works:

The `get_valid_credit_pack_tickets_for_pastelid` function is an asynchronous function that retrieves a list of valid credit pack tickets for a given PastelID. It provides a convenient way for an end user to retrieve all their credit pack tickets and their current credit balances. Here's a detailed breakdown of how the function works:


1. The function starts by querying the database using SQLAlchemy's `select` statement to retrieve all the `CreditPackPurchaseRequestConfirmation` records associated with the given `pastelid`. This is done within an asynchronous database session using `db_code.Session()`.
2. It initializes an empty list called `complete_tickets` to store the complete credit pack tickets.
3. For each `request_confirmation` retrieved from the database, the function performs the following steps:
    - Extracts the `sha3_256_hash_of_credit_pack_purchase_request_fields` from the `request_confirmation`.
    - Queries the database to find the corresponding `CreditPackPurchaseRequestResponseTxidMapping` record using the extracted hash. This mapping contains the `pastel_api_credit_pack_ticket_registration_txid` (TXID) associated with the credit pack ticket.
    - If a `txid_mapping` is found, the function proceeds to retrieve the complete credit pack ticket data.
4. Retrieving Complete Credit Pack Ticket Data:
    - If a `txid_mapping` is found, the function queries the database to check if there is an existing `CreditPackCompleteTicketWithBalance` record associated with the TXID.
    - If an existing record is found (`existing_data`), it means the complete credit pack ticket data is already stored in the database. In this case:
        - The function loads the `complete_credit_pack_data_json` from the `existing_data` record and parses it into a `complete_ticket` dictionary.
        - It calls the `determine_current_credit_pack_balance_based_on_tracking_transactions` function to determine the current credit balance and the number of confirmation transactions for the credit pack ticket based on the TXID.
        - The `credit_pack_current_credit_balance` and `balance_as_of_datetime` fields are added to the `complete_ticket` dictionary.
        - The updated `complete_ticket` is then converted back to JSON format and stored in the `existing_data` record, along with the updated `datetime_last_updated`.
        - The changes are committed to the database within an asynchronous database session.
    - If an existing record is not found, it means the complete credit pack ticket data needs to be retrieved and stored in the database. In this case:
        - The function calls the `determine_current_credit_pack_balance_based_on_tracking_transactions` function to determine the current credit balance and the number of confirmation transactions for the credit pack ticket based on the TXID.
        - It retrieves the `credit_pack_purchase_request_response` and `credit_pack_purchase_request_confirmation` using the `retrieve_credit_pack_ticket_using_txid` function.
        - If both the response and confirmation are successfully retrieved, the function proceeds to create a `complete_ticket` dictionary containing the `credit_pack_purchase_request`, `credit_pack_purchase_request_response`, `credit_pack_purchase_request_confirmation`, `credit_pack_registration_txid`, `credit_pack_current_credit_balance`, and `balance_as_of_datetime`.
        - The `complete_ticket` dictionary is then processed by converting UUIDs to strings and normalizing the data using the `convert_uuids_to_strings` and `normalize_data` functions.
        - The processed `complete_ticket` is converted to JSON format.
        - If an `existing_data` record exists, the `complete_credit_pack_data_json` and `datetime_last_updated` fields are updated with the new data.
        - If an `existing_data` record doesn't exist, a new `CreditPackCompleteTicketWithBalance` record is created with the TXID, `complete_credit_pack_data_json`, and `datetime_last_updated`.
        - The changes are committed to the database within an asynchronous database session.
5. Finally, the `complete_ticket` is appended to the `complete_tickets` list.
6. After processing all the `request_confirmations`, the function returns the `complete_tickets` list containing all the valid credit pack tickets for the given `pastelid`.

This function efficiently retrieves and processes credit pack tickets for a specific PastelID. It leverages the database to store and retrieve complete ticket data, including the current credit balance. By caching the ticket data in the database, subsequent requests for the same ticket can be served faster without the need to retrieve the data from the blockchain every time. The function also ensures that the ticket data is updated with the latest credit balance and timestamp whenever it is retrieved.

The process of retrieving valid credit pack tickets for a specific PastelID using the `get_valid_credit_pack_tickets_for_pastelid` function relies on the local Supernode's SQLite database. However, to ensure that the local database contains up-to-date information about all the credit pack tickets in the blockchain, the Supernode periodically runs background tasks to gather ticket data from the blockchain, parse it, and ingest it into the local database, including the following:


1. `retrieve_generic_ticket_data_from_blockchain`:
    - This function retrieves the ticket data for a specific ticket TXID from the blockchain using the `rpc_connection.tickets('get', ...)` command.
    - It retrieves the ticket's input data, including the fully parsed SHA3-256 hash and the ticket input data dictionary.
    - It computes the SHA3-256 hash of the retrieved ticket input data and compares it with the retrieved fully parsed hash to ensure data integrity.
    - If the hashes match, it returns the credit pack combined blockchain ticket data as a JSON string.
2. `get_list_of_credit_pack_ticket_txids_already_in_db`:
    - This function retrieves the list of credit pack ticket TXIDs that are already stored in the local SQLite database.
    - It queries the `CreditPackPurchaseRequestResponseTxidMapping` table to get the list of TXIDs.
    - It returns a list of unique TXIDs that are already stored in the database.
3. `list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them`:
    - This function retrieves all the blockchain tickets of a specific type (e.g., "INFERENCE_API_CREDIT_PACK_TICKET") starting from a given block height.
    - It uses the `rpc_connection.tickets('list', ...)` command to retrieve the ticket data from the blockchain.
    - It checks the internal consistency of the retrieved tickets by comparing the computed SHA3-256 hashes of the ticket input data with the retrieved fully parsed hashes.
    - It retrieves the list of already stored credit pack TXIDs and the list of known bad credit pack TXIDs from the local database.
    - If `force_revalidate_all_tickets` is set to `True`, it attempts to validate all retrieved tickets, even those already stored in the database or in the known bad list.
    - If `force_revalidate_all_tickets` is set to `False`, it only attempts to validate tickets that are not already stored in the database or in the known bad list.
    - For each ticket TXID that needs validation, it calls the `validate_existing_credit_pack_ticket` function to perform in-depth validation of all aspects of the ticket.
    - If a ticket passes validation, it is saved to the local database using the `retrieve_credit_pack_ticket_using_txid` function.
    - It returns the list of retrieved ticket input data as JSON strings and the list of fully validated ticket TXIDs.
4. `periodic_ticket_listing_and_validation`:
    - This function is an infinite loop that periodically calls the `list_generic_tickets_in_blockchain_and_parse_and_validate_and_store_them` function.
    - It runs every 30 minutes (adjustable) to continuously update the local database with the latest credit pack ticket data from the blockchain.
    - If an error occurs during the periodic execution, it logs the error and continues the loop.

The `startup` function is called when the Supernode starts up. It initializes the database, generates or loads the encryption key, decrypts sensitive fields, and creates background tasks for various operations, including the `periodic_ticket_listing_and_validation` task.
By periodically running the `periodic_ticket_listing_and_validation` task, the Supernode ensures that its local SQLite database is continuously updated with the latest credit pack ticket data from the blockchain. This allows the `get_valid_credit_pack_tickets_for_pastelid` function to retrieve valid credit pack tickets for a specific PastelID efficiently from the local database, without having to query the blockchain every time.

This approach optimizes performance and reduces the load on the blockchain by caching the ticket data in the local database and periodically synchronizing it with the blockchain data. It enables fast retrieval of valid credit pack tickets for a given PastelID while ensuring data consistency and integrity through periodic updates and validations.

One obvious issue with this approach is that we need it to be robust to possible Pastel blockchain level re-orgs and rollbacks; essentially, the local database can go “out of sync” with the underlying blockchain if the blockchain itself changes (this obviously doesn’t happen in normal operation). We do this by using various additional functions and background services which are periodically running:

The code provided includes several functions that work together to handle chain reorgs and rollbacks in the process of tracking and storing burn transactions and block hashes. Let's break down how these functions contribute to handling chain reorgs and rollbacks:


1. `detect_chain_reorg_and_rescan`:
    - This function runs continuously in the background, periodically checking for chain reorgs.
    - It retrieves the latest stored block hash from the database and compares it with the block hash at the same height in the blockchain.
    - If the stored block hash doesn't match the block hash in the blockchain, it indicates that a chain reorg has occurred.
    - When a chain reorg is detected, the function triggers a full rescan of burn transactions by deleting all existing records in the `BurnAddressTransaction` and `BlockHash` tables and calling the `full_rescan_burn_transactions` function.
    - The function sleeps for a specified interval (e.g., 6000 seconds) before checking for chain reorgs again.
2. `full_rescan_burn_transactions`:
    - This function is called when a chain reorg is detected or when no burn transaction records are found in the database.
    - It performs a full rescan of burn transactions starting from the genesis block.
    - It uses the `rpc_connection.scanburntransactions("*")` method to retrieve all burn transactions from any address.
    - The retrieved burn transactions are then processed in chunks using the `process_transactions_in_chunks` function, which inserts the transactions into the database.
    - After processing the burn transactions, the function checks if block hash records exist in the database. If not, it calls the `fetch_and_insert_block_hashes` function to retrieve and store block hashes for the entire blockchain.
3. `fetch_and_insert_block_hashes`:
    - This function fetches and inserts block hashes into the database for a specified range of block heights.
    - It uses the `rpc_connection.getblockhash` method to retrieve block hashes in batches.
    - The retrieved block hashes are processed to construct potential block insert tuples, including the previous and next block hashes if available.
    - The function filters out duplicate heights within the batch and checks the database for existing block heights to avoid inserting duplicates.
    - The final block hashes to be inserted are passed to the `bulk_insert_block_hashes` function for bulk insertion into the database.
    - The function logs progress after inserting a certain number of block hashes (e.g., every 500 inserts) and continues processing the next range of block heights.
4. `bulk_insert_block_hashes`:
    - This function performs the bulk insertion of block hashes into the database.
    - It rechecks the existing block heights just before inserting to handle potential race conditions.
    - The function creates new `BlockHash` instances for the block hashes that don't already exist in the database.
    - It adds the new block hashes to the database session and attempts to commit the changes.
    - If an exception occurs during the bulk insert, the changes are rolled back, and an error is logged.
    

By using these functions together, the code can handle chain reorgs and rollbacks effectively:


- The `detect_chain_reorg_and_rescan` function continuously monitors for chain reorgs by comparing the stored block hashes with the actual block hashes in the blockchain.
- When a chain reorg is detected, the function triggers a full rescan of burn transactions and block hashes by calling the `full_rescan_burn_transactions` function.
- The `full_rescan_burn_transactions` function retrieves all burn transactions from the genesis block and processes them in chunks, inserting them into the database.
- If block hash records are missing, the `fetch_and_insert_block_hashes` function is called to retrieve and store block hashes for the entire blockchain.
- The `fetch_and_insert_block_hashes` function fetches block hashes in batches, filters out duplicates, and passes them to the `bulk_insert_block_hashes` function for bulk insertion into the database.

By performing a full rescan of burn transactions and block hashes when a chain reorg is detected, the code ensures that the stored data is consistent with the current state of the blockchain, effectively handling chain reorgs and rollbacks. Also, as we will see in the next section, these background tasks cache certain information about burn transactions in the local database that are critical for efficiently computing the remaining available balance on a credit pack ticket, so that Supernodes don’t accept inference requests from credit pack tickets that don’t have enough credit value left on them to cover the full cost of the inference request. 

OK, we are now ready to discuss how Inference Requests work and how they interact with credit pack tickets in Pastel.


----------
## Inference Requests

We already gave a pretty detailed background on how inference requests work in the abstract, and in particular, how the system tracks the usage of credits from credit packs in order to determine the remaining available balance of each credit pack (again, only the credit pack’s initial balance amount is stored in the original blockchain ticket— all the information for tracking the current remaining available credit balance must be dynamically computed by the Supernodes by keeping track of all the various burn transactions sent from the designated PSL tracking address that is specified in the original credit pack ticket to determine how many credits have already been used from that specific credit pack ticket). 

We now introduce the detailed flow involved in an end user creating a new inference request and how that request is fulfilled by the responding_supernode:

**Inference Request Flow:**

1. **Initiating the Inference Request:**
    - The user creates a new inference request by sending a POST request to the `/make_inference_api_usage_request` endpoint with an `InferenceAPIUsageRequest` message, which looks like this:

```python
    class InferenceAPIUsageRequest(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        inference_request_id: str = Field(unique=True, index=True)
        requesting_pastelid: str = Field(index=True)
        credit_pack_ticket_pastel_txid: str = Field(index=True)
        requested_model_canonical_string: str
        model_inference_type_string: str
        model_parameters_json_b64: str
        model_input_data_json_b64: str
        inference_request_utc_iso_string: str
        inference_request_pastel_block_height: int
        status: str = Field(index=True)
        inference_request_message_version_string: str
        sha3_256_hash_of_inference_request_fields: str
        requesting_pastelid_signature_on_request_hash: str
```

    - This message contains essential information such as the user's PastelID, the credit pack ticket TXID, the requested model, inference type, model parameters, input data, and various other details.
    - To avoid any issues with quoted JSON or escaping text, all model inputs and all model parameters are supplied as base64 encoded JSON. This also provides flexibility to work with binary inputs, such as input images for the “Ask a Question About an Image” inference requests.
    - The user signs the hash of the inference request fields using their PastelID to ensure the integrity and authenticity of the request.
    - The inclusion of the credit pack ticket TXID allows the responding supernode to verify that the user has sufficient credits to cover the cost of the inference request.
    
2. **Processing the Inference Request:**
    - The responding supernode, determined by the XOR distance between the user's PastelID and the supernode's PastelID, receives the inference request.
    - The supernode first validates the request by checking the signature, ensuring that the request comes from the claimed PastelID and that the data hasn't been tampered with.
    - Next, the supernode checks if it supports the requested model and inference type. This is crucial because different supernodes may have different capabilities and support different models/services.
    - If the supernode supports the requested model, it then ensures that the provided input data matches the expected format for that specific model/service. For example, if the inference request is for text completion, the supernode verifies that the input is indeed a text prompt and not an image file.
    - The supernode then calculates the cost of the inference request based on the specific details provided. This cost calculation varies depending on the type of inference:
        - For text completion requests, the supernode uses the relevant tokenizer for the requested model/service to count the number of tokens in the input prompt. The cost is determined based on the token count and the specific pricing implemented by the API provider in the case of service based offerings; a target “profit margin” is added on top of this to determine the cost of the inference request in credit pack credits. 
        - In the case of models that are hosted locally or remotely using Swiss Army Llama, where there is no API based pricing, we estimate the cost using parameters stored in the model_menu.json file particular to each model that takes into account the memory and processing load of running the model. Eventually, when we have gathered more operating data, we will convert these into estimates of total time used on a GPU enabled instance, and use the average live pricing level for a service like vast.ai to come up with more accurate pricing that covers the true underlying cost of using each model locally or remotely.
        - For audio transcription and embedding requests, the cost is calculated based on the duration of the audio file supplied by the user, measured in seconds.
        - For image generation requests, the relevant API based pricing is again used to arrive at an estimated cost in credits that should cover the Supernode owner’s operating expenses from using the relevant API for that request.
        - For “Ask a Question About an Image” requests, the relevant pricing for the API is again used, taking into account the actual resolution of the input image. 
    - Once the cost is determined, the supernode generates an `InferenceAPIUsageResponse` message, which includes the proposed cost in inference credits, the remaining credits in the user's credit pack after processing the request, and other relevant details, which looks like this:

```python
    class InferenceAPIUsageResponse(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
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
    ```

    - The supernode sends this response back to the user, providing them with the cost estimate for their inference request.
    
2. **Broadcasting the Inference Request and Response:**
    - After sending the `InferenceAPIUsageResponse` back to the user, the responding supernode broadcasts a combined message containing both the `InferenceAPIUsageRequest` and `InferenceAPIUsageResponse` to the nearest supernodes based on the XOR distance to the user's PastelID.
    - This broadcast ensures that multiple supernodes are aware of the inference request and can assist in processing it if needed, providing redundancy and fault tolerance.
    - The broadcast message includes the original inference request details, the proposed cost, and the remaining credits in the user's credit pack.
    
3. **Confirming the Inference Request:**
    - Upon receiving the `InferenceAPIUsageResponse` from the responding supernode, the user reviews the proposed cost and decides whether to proceed with the inference request.
    - If the user agrees to the cost, they confirm the inference request by first sending the corresponding number Patoshis from the designated PSL tracking address to the Pastel burn address; once they do that and get a TXID for that burn transactions, they send a POST request to the `/confirm_inference_request` endpoint with an `InferenceConfirmation` message:

```python
    class InferenceConfirmation(SQLModel):
        inference_request_id: str
        requesting_pastelid: str
        confirmation_transaction: dict
```

    - The `InferenceConfirmation` message includes the inference request ID, the user's PastelID, and a confirmation transaction.
    - The confirmation transaction serves as proof that the user has agreed to the proposed cost and has authorized the deduction of the required inference credits from their credit pack. Since only the authorized creator of the credit pack would have control over the tracking address, everyone knows for sure that the request originated with the user for all intents and purposes (ignoring cases where the user got hacked, for example). 
    - Note that multiple different PastelIDs (the ones specifically listed in the original credit pack ticket data) can create and authorize inference requests from the credit pack ticket; this means that multiple users (each with their own PastelID) could in theory share a single credit pack and easily keep track of who has used what in terms of credits from the pack, but each of these would need to have the private key of the tracking address imported into their local wallet. 
    - However, since only a very tiny amount of PSL is required for the tracking transactions, this address could contain, say, 10 PSL or less, so there would be very little risk in sharing the private key, since there isn’t much PSL value to steal. Critically, even if a third party were able to get the private key to the tracking address, they still wouldn’t be able to “steal” inference requests using it unless they had ALSO stolen the PastelID private keys for one of the PastelIDs included in the list of authorized PastelIDs for that specific credit pack.
    
4. **Executing the Inference Request:**
    - Once the `responding_supernode` receives the `InferenceConfirmation` from the user, it proceeds to process and execute the inference request.
    - The Supernode verifies the confirmation transaction to ensure that the user has authorized the deduction of the inference credits.
    - If the confirmation is valid, the Supernode begins executing the inference request using the specified model and parameters.
    - The execution process varies depending on the type of inference; for example:
        - For text completion requests, the Supernode feeds the input prompt to the specified language model or service using the (optional) user-supplied parameters (e.g., number of completions; number of tokens in each completion; sampling temperature) and generates the completion text based on the model's output.
        - For image generation requests, the Supernode feeds the image prompt and specific user parameters to the image generation model (e.g., Stable Diffusion) and generates an output image to be sent back to the user.
        - For “Ask a Question About an Image” requests, the user supplies an input image and question string, and the Supernode sends that to the multi-modal model (e.g., GPT4o, Llava, etc.) and returns the output response text to the user.
        - For document embedding requests, the supernode uses the specified embedding model to generate vector representations of the extracted sentences in the input document, which can be extracted from various common document formats such as PDF, DOC, HTML, TXT, and even scanned images and PDF using OCR. The user can also optionally supply a semantic query string which will be used to search across the embeddings, returning the most relevant parts of the document text.
        - For audio transcription requests, the Supernode uses the specified audio transcription model to convert the provided audio file into written text and optionally computes embedding vectors for the transcribed text using the specified LLM for calculating the embeddings. The user can also optionally supply a semantic query string which will be used to search across the embeddings, returning the most relevant parts of the transcribed text.
    - During the execution process, the Supernode may need to communicate with external APIs or services, depending on the requested model/service.
    - The Supernode monitors the progress of the inference execution and handles any errors or exceptions that may occur.
    
5. **Generating the Inference Output Result:**
    - Once the inference execution is completed, the `responding_supernode` generates an `InferenceAPIOutputResult` message, which looks like this:

```python
    class InferenceAPIOutputResult(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
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
```

    - This message contains the inference result ID, the original inference request ID, the inference response ID, the `responding_supernode`'s PastelID, and the actual inference output data.
    - The inference output data is serialized and stored in the `inference_result_json_base64` field as a base64-encoded JSON string.
    - The Supernode also includes the file type of the inference output (e.g., JSON, text, image) in the `inference_result_file_type_strings` field.
    - The Supernode signs the hash of the inference result fields using its PastelID to ensure the integrity and authenticity of the output.
    
6. **Checking the Status of Inference Request Results:**
    - The user can check the status of their inference request by sending a GET request to the `/check_status_of_inference_request_results/{inference_response_id}` endpoint.
    - The endpoint expects the `inference_response_id` as a path parameter, which uniquely identifies the inference response associated with the user's request.
    - The `responding_supernode` receives the status check request and verifies that the `inference_response_id` exists and is associated with the requesting user.
    - If the inference request is still being processed, the Supernode responds with a status indicating that the results are not yet available.
    - If the inference request has been completed, the Supernode responds with a status indicating that the results are ready for retrieval.
    
7. **Retrieving the Inference Output Results:**
    - When the inference results are available, the user can retrieve them by sending a POST request to the `/retrieve_inference_output_results` endpoint.
    - The user provides the `inference_response_id`, their PastelID, and a challenge-response signature for authentication purposes.
    - The `responding_supernode` verifies the user's authentication by checking the provided PastelID and challenge-response signature.
    - If the authentication is successful, the Supernode retrieves the `InferenceAPIOutputResult` associated with the provided `inference_response_id`.
    - The Supernode sends the `InferenceAPIOutputResult` back to the user, allowing them to access the inference output data.
    - Additionally, the Supernode broadcasts the `InferenceAPIOutputResult` to the nearest Supernodes based on the XOR distance to the user's PastelID.
    - This broadcast ensures that multiple Supernodes have a record of the inference output and can provide the results to the user if the original `responding_supernode` becomes unavailable.
    
8. **Auditing the Inference Request Response:**
    - The user has the option to audit the inference request response by sending a POST request to the `/audit_inference_request_response` endpoint.
    - The user provides the `inference_response_id`, their PastelID, and a signature to prove their identity and authorization.
    - The `responding_supernode` verifies the user's signature and checks if the provided PastelID matches the one associated with the inference request.
    - If the authentication and authorization are successful, the Supernode retrieves the `InferenceAPIUsageResponse` associated with the provided `inference_response_id`.
    - The Supernode returns the `InferenceAPIUsageResponse` to the user, allowing them to review the details of the inference request response, such as the proposed cost and remaining credits.
    - This audit process provides transparency and allows the user to verify that the inference request was processed correctly and that the appropriate amount of credits were deducted from their credit pack.
    
9. **Auditing the Inference Request Result:**
    - Similar to auditing the inference request response, the user can also audit the inference request result by sending a POST request to the `/audit_inference_request_result` endpoint.
    - The user provides the `inference_response_id`, their PastelID, and a signature for authentication and authorization.
    - The `responding_supernode` verifies the user's signature and checks if the provided PastelID matches the one associated with the inference request.
    - If the authentication and authorization are successful, the supernode retrieves the `InferenceAPIOutputResult` associated with the provided `inference_response_id`.
    - The Supernode returns the `InferenceAPIOutputResult` to the user, allowing them to review the details of the inference output, such as the actual inference result data and file type.
    - This audit process ensures that the user can verify the correctness and integrity of the inference output received from the Supernode.
    

**Additional Endpoints:**

- `/get_inference_model_menu`:
    - This endpoint allows users to retrieve information about the available inference models and their parameters.
    - The `responding_supernode` maintains a menu of supported models and their corresponding parameters, such as input formats, output formats, and any specific options or configurations.
    - When a user sends a GET request to this endpoint, the Supernode responds with the inference model menu, providing the user with the necessary information to construct a valid inference request.
    - The model menu helps users understand the capabilities and requirements of each supported model, enabling them to select the appropriate model for their specific inference needs.
    
- `/download/{file_name}`:
    - This endpoint facilitates the download of inference result files, particularly when the inference output is not easily representable as a JSON string (e.g., generated images or audio files).
    - When an inference request results in a file output, the `responding_supernode` stores the file temporarily and provides a unique file name to the user as part of the `InferenceAPIOutputResult`.
    - The user can then send a GET request to the `/download/{file_name}` endpoint, specifying the file name they received.
    - The Supernode verifies that the requested file exists and is associated with the user's inference request.
    - If the file is found and the user is authorized to access it, the Supernode initiates a file download response, allowing the user to retrieve the inference output file.
    - The Supernode may implement additional security measures, such as file expiration or authentication, to ensure that only authorized users can download the inference output files.


----------

Now let’s get into more details about how this functionality is actually implemented in the Inference Layer server code.  We will introduce a category of inference related service functions, and then give detailed breakdowns of the names of these functions and how they work (inputs, outputs, and purpose/rationale). The code for all these functions can be found [here](https://github.com/pastelnetwork/python_inference_layer_server/blob/master/service_functions.py).


**Messaging Related Service Functions:**

These functions use the built-in Masternode messaging system that is already part of pasteld, and build additional abstractions on top of these so we can use PastelIDs as the basis of identity and security/signing. These messages are used internally by the Supernodes for helping with the flows for both credit pack creation and inference request processing, but we also expose “user oriented” messaging functionality that permits any Pastel user (i.e., not just a Supernode) to use their PastelID as a sort of “email address” for sending and receiving validated and signed messages (that is, users can know for sure that a given message really came from the specified PastelID sender and that the message wasn’t modified in transmit by one or more Supernodes; it does so by checking the PastelID signature on the signed message hash). In this scenario, the Supernodes (which are the only nodes that are able to actually send and receive the lower level messages using the built-in masternode messaging system) act as a kind of “[SMTP](https://en.wikipedia.org/wiki/Simple_Mail_Transfer_Protocol)” mail server on behalf of regular Pastel users.

Note that messages are not currently encrypted, so any Supernode can see the contents of a message sent between two ordinary users. However, it should be relatively simple to add some form of encryption using the pastelIDs of the users to automatically create shared secrets between pairs of users (this is on the list of future features). 

There are several functions crucial for handling inference requests and processing broadcast messages in the Inference Layer server. Let's go through each function and explain how they tie into the various steps of the inference request flow:


- **`broadcast_message_to_n_closest_supernodes_to_given_pastelid`**
    - This function is responsible for broadcasting a message to the closest supernodes based on a given PastelID. It takes the input PastelID, message body, and message type as parameters. First, it retrieves the list of supernodes and filters them based on their support for the desired model using the `is_model_supported` function. It excludes the local supernode from the list of supported supernodes. If no other supported supernodes are found, it falls back to using all supernodes except the local one. The function then selects the N closest supernodes to the input PastelID from the supported supernodes list. It signs the message using the local supernode's PastelID and broadcasts it to the selected supernodes. This function is used in the "Broadcasting the Inference Request and Response" step to inform nearby supernodes about the inference request and response.
    
- **`process_broadcast_messages`**
    This function processes broadcast messages received by the supernode. It takes the message and a database session as input. The function first checks the message type to determine how to handle it.
    For `inference_request_response_announcement_message` messages:
    - It checks if the corresponding `InferenceAPIUsageRequest` and `InferenceAPIUsageResponse` records already exist in the database.
    - If neither record exists, it creates new entries for both the request and response in the database.
    - If either record already exists, it skips the insertion to avoid duplicates.
    For `inference_request_result_announcement_message` messages:
    - It checks if the corresponding `InferenceAPIOutputResult` record already exists in the database.
    - If the record doesn't exist, it creates a new entry for the output result in the database.
    - If the record already exists, it skips the insertion to avoid duplicates.
    This function is used to process and store the information received from broadcast messages related to inference requests and results.
    
- **`monitor_new_messages`**
    This function continuously monitors for new messages received by the Supernode. It runs in an infinite loop, periodically checking for new messages. It retrieves the last processed timestamp from the database to determine the starting point for processing new messages. The function fetches new messages from the masternode messaging system using the `list_sn_messages_func`. For each new message, it checks if the message already exists in the database to avoid duplicates. If the message is new, it updates various metadata tables in the database, including:
    - `MessageSenderMetadata`: Tracks the total messages sent and data sent by each sending Supernode.
    - `MessageReceiverMetadata`: Tracks the total messages received and data received by each receiving Supernode.
    - `MessageSenderReceiverMetadata`: Tracks the total messages and data exchanged between each pair of sending and receiving Supernodes.
    - `MessageMetadata`: Tracks the overall total messages, senders, and receivers in the system.
    After updating the metadata, the function processes the new messages concurrently using the `process_broadcast_messages` function. This function ensures that the Supernode stays up to date with new messages and processes them accordingly, tying into the overall flow of handling inference requests and responses.

**Corresponding SQLModel Data Models**
The data models used to support these functions are defined using SQLModel, which combines SQLAlchemy ORM models with Pydantic response models. Some of the key models include:


- `Message` - Represents a message exchanged between Supernodes:

```python
    class Message(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sending_sn_pastelid: str = Field(index=True)
        receiving_sn_pastelid: str = Field(index=True)
        sending_sn_txid_vout: str = Field(index=True)
        receiving_sn_txid_vout: str = Field(index=True)
        message_type: str = Field(index=True)
        message_body: str = Field(sa_column=Column(JSON))
        signature: str
        timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
```

- `MessageMetadata` - Tracks the overall total messages, senders, and receivers in the system:

```python
    class MessageMetadata(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        total_messages: int
        total_senders: int
        total_receivers: int
        timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
```

- `MessageSenderMetadata` - Tracks the total messages sent and data sent by each sending supernode:

```python
    class MessageSenderMetadata(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sending_sn_pastelid: str = Field(index=True)
        sending_sn_txid_vout: str = Field(index=True)
        sending_sn_pubkey: str = Field(index=True)
        total_messages_sent: int
        total_data_sent_bytes: float
        timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
```

- `MessageReceiverMetadata` - Tracks the total messages received and data received by each receiving supernode:

```python
    class MessageReceiverMetadata(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        receiving_sn_pastelid: str = Field(index=True)
        receiving_sn_txid_vout: str = Field(index=True)
        total_messages_received: int
        total_data_received_bytes: float
        timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
```

- `MessageSenderReceiverMetadata` - Tracks the total messages and data exchanged between each pair of sending and receiving Supernodes:

```python
    class MessageSenderReceiverMetadata(SQLModel, table=True):
        id: Optional[uuid.UUID] = Field(default_factory=uuid.uuid4, primary_key=True, index=True)
        sending_sn_pastelid: str = Field(index=True)
        receiving_sn_pastelid: str = Field(index=True)
        total_messages: int
        total_data_bytes: float
        timestamp: datetime = Field(default_factory=lambda: datetime.now(dt.UTC), index=True)
```

These models work together with the functions to facilitate the processing of inference requests, broadcasting of messages, and handling of received broadcast messages in the Inference Layer server. They enable the Supernodes to communicate and coordinate effectively, ensuring that inference requests are properly processed and the necessary information is stored and tracked in the database.


----------

**Model Menu and API Based Service Related Functions:** 

The next group of service functions we will review are related to how Supernodes determine which models they themselves support, and which ones are supported by other Supernodes. It also includes functions which are used by the Supernodes to automatically test and verify API keys for use with API based services such as OpenAI, Stability, Groq, Mistral, etc. These functions include the following:

- **`is_model_supported`**:
    - This function checks if a desired model is supported by a Supernode. It takes the model menu, desired model canonical string, desired model inference type string, and desired model parameters JSON as input. 
    - The function compares the desired model information with the models available in the model menu, ensuring that the desired model canonical string matches any of the models in the menu with a similarity threshold of 95% using fuzzy matching (via the [FuzzyWuzzy](https://pypi.org/project/fuzzywuzzy/) library). 
    - If a match is found, it further checks if the desired inference type string is supported by the matched model and verifies that the desired model parameters are valid according to the parameter specifications in the model menu. 
    - This function is used in the "Processing the Inference Request" step to ensure that the `responding_supernode` supports the requested model and inference type.


- **`get_inference_model_menu`**:
    - This function is responsible for retrieving the inference model menu, which contains information about the available models and their supported parameters.
    - It first loads the API key test results from a file using the `load_api_key_tests` function.
    - It then fetches the latest model menu from a GitHub URL using an asynchronous HTTP client.
    - The function filters the model menu based on the availability of valid API keys for each model provider (e.g., Stability, OpenAI, Mistral, Groq, Anthropic, OpenRouter).
    - It checks the validity of each API key using the `is_api_key_valid` function, which either retrieves the test result from the loaded API key tests or runs a new test using the `run_api_key_test` function.
    - Models that don't require API keys are automatically included in the filtered model menu.
    - The filtered model menu is saved locally as a JSON file.
    - Finally, the function returns the filtered model menu.
    - This function is used in the "Processing the Inference Request" step to provide the available models and their parameters to the `is_model_supported` function.
    
- **`load_api_key_tests`**:
    - This function loads the API key test results from a JSON file.
    - It reads the contents of the file and returns the loaded JSON data.
    - If the file is not found, it returns an empty dictionary.
    - This function is used by `get_inference_model_menu` to load the existing API key test results.
    
- **`save_api_key_tests`**:
    - This function saves the API key test results to a JSON file.
    - It takes the `api_key_tests` dictionary as input and writes it to the specified file path.
    - This function is used by `get_inference_model_menu` to save the updated API key test results after running new tests.


- **`is_api_key_valid`**:
    - This function checks the validity of an API key for a specific API provider.
    - It takes the `api_name` and `api_key_tests` dictionary as input.
    - If the API name is not found in the `api_key_tests` dictionary or the existing test result is outdated (based on the `is_test_result_valid` function), it runs a new API key test using the `run_api_key_test` function.
    - If the test passes, it updates the `api_key_tests` dictionary with the new test result and timestamp.
    - It returns the validity status of the API key (True or False).
    - This function is used by `get_inference_model_menu` to determine which models should be included in the filtered model menu based on the availability of valid API keys.
    
- **`is_test_result_valid`**:
    - This function checks if an API key test result is still valid based on a specified validity duration.
    - It takes the `test_timestamp` as input and compares it with the current timestamp.
    - If the difference between the current timestamp and the test timestamp is less than the specified validity duration (e.g., 24 hours), it considers the test result as valid.
    - This function is used by `is_api_key_valid` to determine if a new API key test needs to be run.
    
- **`run_api_key_test`**:
    - This function runs an API key test for a specific API provider.
    - It takes the `api_name` as input and calls the corresponding test function based on the API name.
    - It supports testing API keys for Stability, OpenAI, Mistral, Groq, Anthropic (Claude), and OpenRouter.
    - Each test function sends a test request to the respective API endpoint using the provided API key and checks the response status or content to determine if the API key is valid.
    - If the test passes, it returns True; otherwise, it returns False.
    - This function is used by `is_api_key_valid` to run a new API key test when needed.
    

These functions work together to manage the inference model menu and ensure that only models with valid API keys are included. The `get_inference_model_menu` function is a key part of the "Processing the Inference Request" step, where it provides the available models and their parameters to the `is_model_supported` function. By regularly testing the API keys and filtering the model menu accordingly, the system ensures that inference requests are only processed for models with valid API keys, preventing potential errors or failures during the execution of the inference request.

It also allows Supernodes to accurately advertise the specific models and services they support. This allows the Inference Layer to provide a lot more options to users without introducing necessary centralization, since Supernode Operators are free to decide which API based services, if any, they want to support with their own Supernodes; if they do want to support a particular API service, such as *OpenAI* or *Groq*, then it’s totally up to the Supernode operators themselves to procure a valid API key and include it in their .env file in encrypted form. The result of all this are exposed by each Supernode at the following endpoint:

```python
    @router.get("/get_inference_model_menu")
    async def get_inference_model_menu_endpoint(
        rpc_connection=Depends(get_rpc_connection),
    ):
        model_menu = await service_functions.get_inference_model_menu()
        return model_menu 
```

which sends back the filtered model menu after removing any entries in the master model_menu.json that the particular Supernode doesn’t support. For instance, if that particular Supernode is controlled by a Supernode operator who never bothered to get a Stability API key, then the returned model menu from that Supernode will not include the “text_to_image” inference type and associated Stability models; if an end user wants to use that inference type, they will need to select a different Supernode as their closest model-supporting `responding_supernode`.

To really understand how the `model_menu.json`  file works and its importance in the wider system, we need to go into a lot more detail. For starters, here is what the beginning of this file looks like:

```JSON
    {
      "models": [
        {
          "model_name": "swiss_army_llama-Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M",
          "model_url": "https://huggingface.co/NousResearch/Hermes-2-Theta-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-Instruct-Merged-DPO-Q4_K_M.gguf",
          "description": "Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleaned version of the OpenHermes 2.5 Dataset, as well as a newly introduced Function Calling and JSON Mode dataset developed in-house. This version excels at Function Calling, JSON Structured Outputs, and has improved on several other metrics, scoring 90% on function calling evaluation and 84% on structured JSON Output evaluation.",
          "supported_inference_type_strings": [
            "text_completion",
            "embedding",
            "embedding_document",
            "embedding_audio"
          ],
          "input_fields": [
            {
              "name": "input_prompt",
              "file_type": "text",
              "optional": false
            }
          ],
          "output_fields": [
            {
              "name": "generated_text",
              "file_type": "text",
              "optional": false
            }
          ],
          "model_parameters": [
            {
              "name": "number_of_tokens_to_generate",
              "type": "int",
              "default": 1000,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The maximum number of tokens to generate [Optional]"
            },
            {
              "name": "temperature",
              "type": "float",
              "default": 0.7,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The temperature for sampling [Optional]"
            },
            {
              "name": "number_of_completions_to_generate [Optional]",
              "type": "int",
              "default": 1,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The number of completions to generate"
            },
            {
              "name": "grammar_file_string",
              "type": "string",
              "default": "",
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "Grammar File String: The grammar file used to restrict text generation [Optional] (Default is to not use any grammar file. Examples: `json`, `list`)"
            },
            {
              "name": "corpus_identifier_string",
              "type": "string",
              "default": "",
              "inference_types_parameter_applies_to": "['embedding_document', 'embedding_audio']",
              "description": "Corpus Identifier String: Identifies the subset of the stored embeddings to use in the semantic search [Optional] (Default is to use just the embeddings from the uploaded document or audio file transcript)."
            }
          ],
          "credit_costs": {
            "text_completion": {
              "input_tokens": 1.8,
              "output_tokens": 1.4,
              "compute_cost": 1.0,
              "memory_cost": 0.9
            },
            "embedding": {
              "input_tokens": 0.7,
              "compute_cost": 0.8,
              "memory_cost": 0.5
            },
            "embedding_document": {
              "average_tokens_per_sentence": 0.2,
              "total_sentences": 0.1,
              "query_string_included": 0.4,
              "compute_cost": 1.2,
              "memory_cost": 0.8
            },
            "embedding_audio": {
              "audio_file_length_in_seconds": 0.5,
              "query_string_included": 1.5,
              "compute_cost": 1.2,
              "memory_cost": 1.0
            }
          }
        },
```
Compare the above entry for a “locally hosted” Swiss Army Llama based LLM to the entry below for an API service based model, which has fewer options to specify and which uses a very different approach to calculating the cost of a specific inference request (the API service based models try to estimate the actual cost to the Supernode operator of making that particular inference request, whereas the Swiss Army Llama based models instead try to quantify the compute/memory usage of a particular inference request):

```JSON
        {
          "model_name": "openai-gpt-4o",
          "model_url": "",
          "description": "GPT-4o is OpenAI's most advanced multimodal model that\u2019s faster and cheaper than GPT-4 Turbo with stronger vision capabilities. The model has 128K context and an October 2023 knowledge cutoff.",
          "supported_inference_type_strings": [
            "text_completion"
          ],
          "input_fields": [
            {
              "name": "input_prompt",
              "file_type": "text",
              "optional": true
            }
          ],
          "output_fields": [
            {
              "name": "generated_text",
              "file_type": "text",
              "optional": false
            }
          ],
          "model_parameters": [
            {
              "name": "number_of_tokens_to_generate",
              "type": "int",
              "default": 1000,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The maximum number of tokens to generate [Optional]"
            },
            {
              "name": "temperature",
              "type": "float",
              "default": 0.7,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The temperature for sampling [Optional]"
            },
            {
              "name": "number_of_completions_to_generate [Optional]",
              "type": "int",
              "default": 1,
              "inference_types_parameter_applies_to": "['text_completion']",
              "description": "The number of completions to generate"
            }
          ],
          "credit_costs": {
            "api_based_pricing": 1
          }
        }, 
```

Or for an image generation (“text_to_image” inference type) model’s entry, which specifies wildly different input parameters and output type (a binary image encoded as base64 text rather than a plain text output like you would get from an LLM model doing a text_completion inference request):

```JSON
     {
          "model_name": "stability-core",
          "model_url": "",
          "description": "Stable Image Core is the best quality achievable at high speed for text-to-image generation.",
          "supported_inference_type_strings": [
            "text_to_image",
            "creative_upscale"
          ],
          "input_fields": [
            {
              "name": "prompt",
              "file_type": "text",
              "optional": false
            }
          ],
          "output_fields": [
            {
              "name": "generated_image",
              "file_type": "base64_image",
              "optional": false
            }
          ],
          "model_parameters": [
            {
              "name": "aspect_ratio",
              "type": "string",
              "default": "1:1",
              "inference_types_parameter_applies_to": "['text_to_image']",
              "description": "The aspect ratio of the generated image [Optional]"
            },
            {
              "name": "seed",
              "type": "int",
              "default": 0,
              "inference_types_parameter_applies_to": "['text_to_image']",
              "description": "Random seed to use for generation [Optional]"
            },
            {
              "name": "style_preset",
              "type": "string",
              "default": null,
              "inference_types_parameter_applies_to": "['text_to_image']",
              "description": "Style Preset: Guides the image model towards a particular style (e.g., '3d-model', 'analog-film', 'anime', etc.) [Optional]"
            },
            {
              "name": "output_format",
              "type": "string",
              "default": "jpeg",
              "inference_types_parameter_applies_to": "['text_to_image']",
              "description": "The format of the generated image (e.g., 'png', 'jpeg', 'webp', etc.) [Optional]"
            },
            {
              "name": "negative_prompt",
              "type": "string",
              "default": null,
              "inference_types_parameter_applies_to": "['text_to_image']",
              "description": "Negative Prompt: A blurb of text describing what you do not wish to see in the output image [Optional]"
            }
          ],
          "credit_costs": {
            "api_based_pricing": 1
          }
        },
```

As you can probably see from the examples, the structure is quite generic and covers all the functionality we might need from any kind of model, with clear methods indicated for how to expand the coverage to new and different inference types, services, and models. The `model_menu.json` file plays a crucial role in the Inference Layer server code. It serves as a central configuration file that defines the available models, their capabilities, and the parameters they support for different inference types. The file is designed to be flexible and extensible, allowing easy addition of new models and services without requiring significant changes to the codebase.

Let's take a detailed look at the structure of the `model_menu.json` file and explain the rationale behind its design:


1. The root of the JSON file contains a single key, "models", which is an array of model objects.


2. Each model object represents a specific model or service that the Inference Layer supports. It contains various properties that describe the model's characteristics and capabilities. The key properties of a model object are:
    - `model_name`: A unique identifier for the model, used throughout the system to refer to the specific model.
    - `model_url`: The URL or path to the model file, if applicable. This is used for locally hosted models using Swiss Army Llama.
    - `description`: A brief description of the model, providing information about its capabilities, training data, or any other relevant details.
    - `supported_inference_type_strings`: An array of strings indicating the inference types supported by the model. This allows the system to determine which models can be used for a particular inference request.
    - `input_fields`: An array of objects specifying the input fields required by the model for each inference type. Each input field object contains the following properties:
        - `name`: The name of the input field.
        - `file_type`: The type of data expected for the input field (e.g., text, base64, etc.).
        - `optional`: A boolean indicating whether the input field is optional or required.
    - `output_fields`: An array of objects specifying the output fields returned by the model for each inference type. Each output field object contains the following properties:
        - `name`: The name of the output field.
        - `file_type`: The type of data returned in the output field (e.g., text, base64_image, etc.).
        - `optional`: A boolean indicating whether the output field is optional or guaranteed to be present.
    - `model_parameters`: An array of objects defining the parameters supported by the model for each inference type. Each parameter object contains the following properties:
        - `name`: The name of the parameter.
        - `type`: The data type of the parameter (e.g., int, float, string, etc.).
        - `default`: The default value for the parameter, if applicable.
        - `inference_types_parameter_applies_to`: An array of inference types to which the parameter applies. This allows for parameters to be specific to certain inference types.
        - `description`: A description of the parameter, providing information about its purpose and usage.
    - `credit_costs`: An object that defines the credit costs associated with using the model for different inference types. It can contain either specific cost breakdowns for each inference type or a flag indicating that the model uses API-based pricing.
    

The rationale behind this design is to provide a declarative and data-driven approach to configuring the available models and their capabilities. By storing all the necessary information about models in the `model_menu.json` file, the Inference Layer server code can dynamically adapt to changes in the model lineup without requiring code modifications.

One of the key benefits of this design is that it enables the automatic generation of UI elements in the visual frontend for each model and inference type. The frontend code can iterate over the models defined in the `model_menu.json` file and dynamically create the appropriate input fields, parameter controls, and output displays based on the specifications provided in the JSON file.
For example, when a user selects a specific model and inference type in the frontend, the UI can be dynamically populated with the relevant input fields and parameter controls as defined in the `model_menu.json` file. This eliminates the need for hard-coded UI elements and allows for a highly flexible and adaptable user interface.

Furthermore, the `model_menu.json` file serves as a single source of truth for model configuration. It provides a clear and centralized location for defining the available models, their capabilities, and the parameters they support. This makes it easier to manage and update the model lineup, as well as maintain consistency across different parts of the system.



----------

**Inference Request Cost Estimation Service Functions**

The next batch of functions pertain to estimating the cost of performing specific inference requests depending on the nature of the request and specified models and model parameters. Let's go through each function and explain their purpose and how they contribute to the cost estimation process.


1. **`get_tokenizer`** **and** **`count_tokens`**:
- The `get_tokenizer` function maps model names to their corresponding tokenizer names using a predefined mapping. It uses fuzzy string matching to find the best match for the given model name and returns the appropriate tokenizer name.
- The `count_tokens` function takes a model name and input data as parameters and counts the number of tokens in the input data using the appropriate tokenizer for the specified model.
- It handles different types of tokenizers, such as GPT-2, GPT-Neo, BERT, RoBERTa, and specific tokenizers for models like Claude, Whisper, and CLIP.
- The token count is important for estimating the cost of inference requests, as many pricing models are based on the number of input and output tokens.


2. **`calculate_api_cost`**:
- This function calculates the estimated cost of an inference request based on the pricing data for each API service and model.
- It takes the model name, input data, and model parameters as input and uses fuzzy string matching to find the best match for the model name in the predefined pricing data.
- For Stability models, the cost is calculated based on the credits per call and the number of completions to generate.
- For OpenAI's GPT-4o-vision model, the cost is calculated based on the input image resolution, number of tiles, and tokens, as well as the question tokens.
- For other models, the cost is calculated based on the number of input tokens, output tokens to generate, number of completions, and per-call cost.
- The function returns the estimated cost in dollars.


3. **`convert_document_to_sentences`**:
- This function converts a document file to sentences using the Swiss Army Llama service.
- It checks if either the local or remote Swiss Army Llama service is responding and selects the appropriate port.
- It uploads the document file and retrieves the file metadata (URL, hash, size) using the `upload_and_get_file_metadata` function.
- It sends a POST request to the Swiss Army Llama service with the file metadata and security token to convert the document to sentences.
- If the remote Swiss Army Llama service fails, it falls back to the local service if available.
- The function returns a dictionary containing the individual sentences and the total number of sentences in the document.


4. **`calculate_proposed_inference_cost_in_credits`**:
- This function calculates the proposed cost of an inference request in credits based on the requested model, model parameters, inference type, and input data.
- It distinguishes between API-based models and local LLM models (Swiss Army Llama).
- For API-based models, it calls the `calculate_api_cost` function to estimate the cost in dollars and then converts it to credits based on a target value per credit and profit margin.
- For local LLM models, it retrieves the credit costs from the model data and calculates the cost based on the inference type:
- For text completion and question-answering on images, it considers the input tokens, output tokens, number of completions, compute cost, and memory cost.
- For embedding documents, it converts the document to sentences using the `convert_document_to_sentences` function and calculates the cost based on the total tokens, total sentences, query string inclusion, compute cost, and memory cost.
- For embedding audio, it calculates the cost based on the audio length in seconds, query string inclusion, compute cost, and memory cost.
- The function applies a credit cost multiplier factor and ensures a minimum cost in credits.
- It returns the final proposed cost in credits for the inference request.

These functions work together to estimate the cost of performing specific inference requests based on various factors such as the model, input data, model parameters, and inference type. They take into account the pricing models of different API services and local LLM models, converting the estimated costs to credits for a consistent pricing scheme. The `calculate_proposed_inference_cost_in_credits` function serves as the main entry point for estimating the cost of an inference request, utilizing the other functions as needed.


----------

**Misc Utility Functions for Inference Requests:**

Next up are a couple utility functions that are used with Swiss Army Llama:


- **`is_swiss_army_llama_responding`**:
    - This function checks if the Swiss Army Llama service is responding, either locally or remotely.
    - It takes a boolean parameter `local` to determine whether to check the local or remote service.
    - Based on the `local` parameter, it sets the appropriate port number for the service.
    - It sends a GET request to the `/get_list_of_available_model_names/` endpoint of the Swiss Army Llama service, including a security token as a parameter.
    - If the response status code is 200 (OK), it means the service is responding, and the function returns `True`. Otherwise, it returns `False`.
    - Checking the responsiveness of the Swiss Army Llama service is important to ensure that it is available for performing inference requests and other tasks.
    
- **`check_if_input_text_would_get_rejected_from_api_services`**:
    - This is an asynchronous function that checks if the input text for an inference request is likely to be rejected or cause problems with API services like OpenAI.
    - It first checks if the Swiss Army Llama service is responding, either locally or remotely, based on the configuration settings.
    - If neither the local nor the remote Swiss Army Llama service is responding, it logs an error message and returns `None`, indicating that the check cannot be performed.
    - If the Swiss Army Llama service is responding, it constructs an input prompt that asks whether the given inference request text is problematic or offensive and likely to be rejected by OpenAI.
    - It sends a POST request to the `/get_text_completions_from_input_prompt/` endpoint of the Swiss Army Llama service, passing the input prompt, model name, and other parameters.
    - The function expects the Swiss Army Llama service to respond with either "ACCEPT" or "REJECT" based on its assessment of the input text (it constrains the results of the model by using a grammar file which only allows responses that conform to either “ACCEPT” or “REJECT”
    - If the response is "ACCEPT", it means the inference request is not problematic, and the function returns `True`, indicating that the request can be passed on to the API service.
    - If the response is "REJECT", it means the inference request is determined to be problematic and likely to result in a rejection or ban from the API service. In this case, the function logs an error message and returns `False`.
    - If the response is neither "ACCEPT" nor "REJECT", it logs a warning message and returns the current value of `inference_request_allowed`.
    - In case of any exceptions during the Swiss Army Llama request, it logs an error message and attempts to fall back to the local Swiss Army Llama service if the remote service was being used.
    

The `check_if_input_text_would_get_rejected_from_api_services` function plays a crucial role in the inference request flow. Before sending an inference request to an external API service, it is important to ensure that the input text is not problematic or offensive, as sending such requests could result in the API key being banned or the request being rejected.

By leveraging the Swiss Army Llama service, this function performs a pre-check on the input text to determine its suitability for the API service. It uses a specific model and parameters to assess the input text and expects a clear "ACCEPT" or "REJECT" response.

This pre-check step helps to mitigate the risk of sending inappropriate or offensive requests to the API service, protecting the Pastel inference system (more specifically, protecting individual Supernode operators themselves) from potential bans or rejections. It allows the system to proactively filter out problematic requests and ensure a smoother and more reliable inference request flow.

The function also handles scenarios where the Swiss Army Llama service may not be responding and provides fallback mechanisms to ensure the pre-check can still be performed using the local service if available.

----------

**Validating and Processing New Inference Requests:**

The next batch of service functions play crucial roles in the inference request flow, specifically in validating and processing incoming inference requests. They also help us to determine how much a given credit pack ticket has been used so we can determine if it contains enough remaining credits to cover the cost of the currently contemplated inference request, as well as detecting the corresponding confirmation tracking transactions that are used by end users to authorize a new inference request. Let's go through each function and explain their purpose and how they tie into the inference flow:

- **`validate_inference_api_usage_request`**:
    - This function takes an `InferenceAPIUsageRequest` object as input and performs a series of validations to ensure the integrity and validity of the inference request.
    - It first calls the `validate_inference_request_message_data_func` to validate the request message data against predefined validation rules. If any validation errors are found, an exception is raised.
    - It extracts relevant information from the request, such as the requesting PastelID, credit pack ticket TXID, requested model, inference type, model parameters, and input data.
    - The function validates the credit pack ticket TXID using the `validate_pastel_txid_string` function to ensure it follows the expected format.
    - It retrieves the credit pack purchase request response using the `retrieve_credit_pack_ticket_using_txid` function and checks if the requesting PastelID is authorized to use the credit pack.
    - The function then verifies if the requested model is available in the model menu obtained using the `get_inference_model_menu` function. If the model is not found or the requested inference type is not supported by the model, the function returns `False`.
    - If the requested model is a Swiss Army Llama model (not API-based), the function checks if the Swiss Army Llama service is available, either locally or remotely. If the service is not responding, the function returns `False`.
    - The function decodes the model parameters and input data from base64 format and performs additional checks based on the detected data type.
    - If the inference request is for an API-based model and local checking is enabled, the function calls the `check_if_input_text_would_get_rejected_from_api_services` function to determine if the input text is likely to be rejected by the API service. If the request is deemed risky, the function returns `False`.
    - The function calculates the proposed cost of the inference request in credits using the `calculate_proposed_inference_cost_in_credits` function based on the requested model, model parameters, inference type, and input data.
    - It validates the credit pack ticket using the `validate_existing_credit_pack_ticket` function to ensure its validity.
    - The function determines the current credit balance of the credit pack ticket using the `determine_current_credit_pack_balance_based_on_tracking_transactions` function and checks if there are sufficient credits available for the requested inference. If the balance is insufficient, the function returns `False`.
    - Finally, the function returns a tuple indicating the validity of the request, the proposed cost in credits, and the remaining credits after the request.

- **`check_burn_address_for_tracking_transaction`**:
    - This function checks the burn address for a specific tracking transaction related to an inference request confirmation.
    - It takes the tracking address, expected amount, transaction ID (TXID), maximum block height, and optional retry parameters as input.
    - The function uses the RPC connection to retrieve transactions from the blockchain and searches for a transaction that matches the specified criteria.
    - If a transaction with the exact expected amount is found and has sufficient confirmations (or if the confirmation check is skipped), the function returns `True` along with the transaction details.
    - If a transaction with an amount greater than or equal to the expected amount is found, the function returns `False` for an exact match but `True` for an exceeding transaction, along with the transaction details.
    - If no matching transaction is found after the specified number of retries, the function returns `False`.
    - This function is used to verify that the user has indeed sent the required tracking transaction to the burn address, confirming their agreement to pay for the inference request using their credits.

-  **`determine_current_credit_pack_balance_based_on_tracking_transactions`**:
    - This function is a crucial part of the inference flow, as it calculates the current balance of credits available in a specific credit pack ticket. This function is essential for ensuring that users have sufficient credits to cover the cost of their inference requests and for keeping track of the credits consumed over time.
    - The function takes the `credit_pack_ticket_txid` as input, which uniquely identifies the credit pack ticket.
    - It retrieves the credit pack ticket data using the `retrieve_credit_pack_ticket_using_txid` function, which fetches the ticket information from the blockchain using the provided TXID.
    - From the retrieved ticket data, the function extracts the initial credit balance (`requested_initial_credits_in_credit_pack`) and the credit usage tracking PSL address (`credit_usage_tracking_psl_address`). These pieces of information are necessary for calculating the current credit balance.
    - The function then queries the database to retrieve all the burn address transactions associated with the specific credit usage tracking address. These transactions represent the inference requests that have consumed credits from the credit pack.
    - It calculates the total credits consumed by summing up the amounts of all the retrieved burn address transactions. The transaction amounts are adjusted using the `CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER` to convert them from the tracking amount to the actual credit amount.
    - The function then checks if there are any new blocks added to the blockchain since the last update of the database. It does this by comparing the current block height (`current_block_height`) with the latest block height stored in the database (`latest_db_block_height`).
    - If there are new blocks, the function retrieves the new burn transactions from those blocks using the `listsinceblock` RPC method. It filters out the transactions that are not relevant to the specific credit pack ticket based on the burn address and tracking address.
    - The function decodes the new burn transactions using the `process_transactions_in_chunks` function, which retrieves the detailed transaction data in chunks to handle a large number of transactions efficiently.
    - It then queries the database again to retrieve any new tracking transactions associated with the credit pack ticket that occurred in the new blocks or are still pending.
    - The total credits consumed is updated by adding the amounts of the new tracking transactions to the previously calculated total.
    - Optionally, the function can update the block hashes in the database for the new blocks using the `fetch_and_insert_block_hashes` function. This step is useful for maintaining a local copy of the block hashes for faster lookups.
    - Finally, the function calculates the current credit balance by subtracting the total credits consumed from the initial credit balance. It also counts the total number of confirmation transactions from the tracking address to the burn address.
    - The function returns the current credit balance and the number of confirmation transactions.
    - The `determine_current_credit_pack_balance_based_on_tracking_transactions` function is closely tied to the `check_burn_address_for_tracking_transaction` function, which is used to verify the tracking transactions sent by users to confirm their inference requests.
        - When a user sends a confirmation transaction to the burn address, the `check_burn_address_for_tracking_transaction` function checks if the transaction meets the expected criteria, such as the correct amount and sufficient confirmations. If the transaction is valid, it indicates that the user has authorized the consumption of credits from their credit pack for the inference request.
        - The `determine_current_credit_pack_balance_based_on_tracking_transactions` function, in turn, relies on these tracking transactions to calculate the current balance of the credit pack. By querying the database and retrieving the relevant tracking transactions, the function can accurately determine how many credits have been consumed and update the credit pack balance accordingly.
    - This function is crucial for maintaining the integrity of the credit system and ensuring that users can only consume the credits they have available in their credit packs. It allows the inference flow to proceed smoothly, with the `responding_supernode` being able to verify that the user has sufficient credits before executing the inference request.
    - Moreover, the function's ability to handle new blocks and transactions ensures that the credit balance remains up to date, even as new inference requests are processed and new tracking transactions are added to the blockchain.

- **`process_inference_api_usage_request`**:
    - This function takes an `InferenceAPIUsageRequest` object as input and processes the inference request.
    - It first calls the `validate_inference_api_usage_request` function to validate the request. If the request is invalid, an exception is raised.
    - If the request is valid, the function saves the inference API usage request using the `save_inference_api_usage_request` function.
    - It retrieves the credit pack purchase request response using the `retrieve_credit_pack_ticket_using_txid` function to obtain the credit usage tracking PSL address.
    - The function then calls the `create_and_save_inference_api_usage_response` function to create and save an `InferenceAPIUsageResponse` object based on the saved request, proposed cost in credits, remaining credits after the request, and credit usage tracking PSL address.
    - Finally, the function returns the created `InferenceAPIUsageResponse` object.

- **`create_and_save_inference_api_usage_response`**:
    - This is a helper function used by `process_inference_api_usage_request` to create and save an `InferenceAPIUsageResponse` object.
    - It takes the saved inference API usage request, proposed cost in credits, remaining credits after the request, and credit usage tracking PSL address as input.
    - The function generates a unique identifier for the inference response and creates an `InferenceAPIUsageResponse` instance with the relevant information.
    - It computes the hash of the response fields using the `compute_sha3_256_hash_of_sqlmodel_response_fields` function and signs the hash with the local Supernode's PastelID using the `sign_message_with_pastelid_func` function.
    - The function then saves the `InferenceAPIUsageResponse` object to the database using an asynchronous database session.
    - Finally, it returns the saved `InferenceAPIUsageResponse` object.

These functions play a critical role in the inference request flow by validating incoming requests, checking the availability of requested models and services, calculating the cost of the inference in credits, and ensuring sufficient credits are available in the associated credit pack ticket. They also handle the creation and saving of the `InferenceAPIUsageResponse` object, which represents the Supernode's response to the inference request, including the proposed cost and remaining credits.

The `validate_inference_api_usage_request` function acts as a gatekeeper, performing various checks and validations to ensure the integrity and validity of the inference request before proceeding with the actual processing. It helps prevent invalid or unauthorized requests from being processed and ensures that the requested model and inference type are supported.

The `process_inference_api_usage_request` function orchestrates the overall processing of the inference request. It relies on the `validate_inference_api_usage_request` function to validate the request and then saves the request, retrieves necessary information, and creates and saves the corresponding `InferenceAPIUsageResponse` object.

----------

**Processing Inference Confirmations and Saving Output Results:**

The next batch of service functions pertain to several important functions related to processing inference confirmations, saving inference output results. Let's go through each function and explain their purpose and how they tie into the inference flow:

- **`process_inference_confirmation`**:
    - This function processes the confirmation of an inference request sent by the user.
    - It takes the inference request ID and an `InferenceConfirmation` object as input.
    - The function retrieves the corresponding `InferenceAPIUsageRequest` and `InferenceAPIUsageResponse` from the database using the provided inference request ID.
    - It ensures that the burn address is tracked by the local wallet by importing it if necessary.
    - The function then calls the `check_burn_address_for_tracking_transaction` function to check if the burn address has received the expected tracking transaction from the user.
    - If a matching transaction is found, the function computes the current credit pack balance based on the tracking transactions using the `determine_current_credit_pack_balance_based_on_tracking_transactions` function.
    - It updates the status of the inference request to "confirmed" in the database.
    - Finally, the function triggers the execution of the inference request by creating a new task using `asyncio.create_task(execute_inference_request(inference_request_id))`.
    - If any error occurs during the process, the function logs the error and re-raises the exception.
    
- **`save_inference_output_results`**:
    - This function saves the output results of an inference request to the database.
    - It takes the inference request ID, inference response ID, output results dictionary, and output results file type strings as input.
    - The function generates a unique identifier for the inference result and creates an `InferenceAPIOutputResult` record without the hash and signature fields.
    - It populates the record with the relevant information, including the`responding_supernode`'s PastelID, the base64-encoded output results JSON, file type strings, timestamp, block height, and message version.
    - The function computes the hash of the inference result fields using the `compute_sha3_256_hash_of_sqlmodel_response_fields` function and signs the hash with the `responding_supernode`'s PastelID using the `sign_message_with_pastelid_func` function.
    - Finally, it saves the `InferenceAPIOutputResult` record to the database using an asynchronous database session.
    - If any error occurs during the process, the function logs the error and re-raises the exception.
    
These functions play important roles in the inference flow, specifically in the confirmation and result saving stages. The `process_inference_confirmation` function is responsible for handling the user's confirmation of an inference request. When the user sends a tracking transaction to the burn address, indicating their agreement to pay for the inference using their credits, this function verifies the transaction and updates the status of the inference request to "confirmed". It also triggers the execution of the inference request by creating a new task.

The `save_inference_output_results` function is called after the inference request has been executed and the output results are available. It takes the output results, along with other relevant information, and creates an `InferenceAPIOutputResult` record in the database. This record contains the inference results, file type information, `responding_supernode`'s PastelID, and other metadata. The function also computes a hash of the result fields and signs it with the Supernode's PastelID to ensure the integrity and authenticity of the results.


----------


**Inference Request Execution Functions for API-Based Services and Models**

The next batch of service functions we will review are related to how inference requests are executed for API-based services and models, such as those from Stability, OpenAI, Anthropic, Mistral, Groq, and OpenRouter. These functions are responsible for submitting the inference requests to the respective APIs, handling the responses, and processing the output results. Let's go through each function and explain their purpose and how they contribute to the execution of inference requests:

1. **`get_claude3_model_name`**:
    - This is a helper function specific to Anthropic's Claude API. It maps the model names used in the Pastel Inference Layer to the corresponding model names recognized by the Claude API.
    - It takes the `model_name` as input and returns the corresponding Claude API model name using a predefined mapping.
    - If the provided `model_name` is not found in the mapping, it returns an empty string.
    - This function ensures that the correct model name is used when submitting inference requests to the Claude API.
    
2. **`submit_inference_request_to_stability_api`**:
    - This function is responsible for submitting inference requests to the Stability API for image generation tasks.
    - It supports two types of inference requests: "text_to_image" and "creative_upscale".
    - For "text_to_image" requests:
        - It extracts the model parameters and input prompt from the `inference_request` object.
        - It constructs the necessary data payload based on the model parameters, such as aspect ratio, output format, negative prompt, seed, and style preset.
        - It sends a POST request to the Stability API endpoint with the appropriate headers and data payload.
        - If the response is successful (status code 200), it processes the response JSON and extracts the generated image as base64-encoded data.
        - It returns the output results and file type strings.
    - For "creative_upscale" requests:
        - It extracts the model parameters and input image from the `inference_request` object.
        - It constructs the necessary data payload based on the model parameters, such as prompt, output format, creativity, seed, and negative prompt.
        - It sends a POST request to the Stability API endpoint with the appropriate headers, files, and data payload.
        - If the response is successful (status code 200), it retrieves the `generation_id` from the response.
        - It then enters a loop to poll for the upscaled image result using the `generation_id`.
        - Once the upscaled image is available, it encodes the image as base64 and returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
3. **`submit_inference_request_to_openai_api`**:
    - This function is responsible for submitting inference requests to the OpenAI API for various tasks, including text completion, embedding, and question-answering on images.
    - For "text_completion" requests:
        - It extracts the model parameters and input prompt from the `inference_request` object.
        - It determines the number of completions to generate based on the model parameters.
        - It sends a POST request to the OpenAI API endpoint with the appropriate headers and JSON payload, including the model, input prompt, max tokens, temperature, and number of completions.
        - If the response is successful (status code 200), it processes the response JSON and extracts the generated text completions.
        - It keeps track of the total input and output tokens used for the request.
        - It returns the output results and file type strings.
    - For "embedding" requests:
        - It extracts the input text from the `inference_request` object.
        - It sends a POST request to the OpenAI API endpoint with the appropriate headers and JSON payload, including the model and input text.
        - If the response is successful (status code 200), it processes the response JSON and extracts the embedding vector.
        - It returns the output results and file type strings.
    - For "ask_question_about_an_image" requests:
        - It extracts the model parameters, number of completions, input image, and question from the `inference_request` object.
        - It encodes the input image as base64.
        - It sends a POST request to the OpenAI API endpoint with the appropriate headers and JSON payload, including the model, question, image URL, and max tokens.
        - If the response is successful (status code 200), it processes the response JSON and extracts the generated text responses.
        - It keeps track of the total input and output tokens used for the request.
        - It returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
4. **`submit_inference_request_to_openrouter`**:
    - This function is responsible for submitting inference requests to the OpenRouter API for text completion tasks.
    - It extracts the model parameters and input prompt from the `inference_request` object.
    - It constructs the necessary JSON payload based on the model parameters, such as model, input prompt, max tokens, and temperature.
    - It sends a POST request to the OpenRouter API endpoint with the appropriate headers and JSON payload.
    - If the response is successful (status code 200), it processes the response JSON and extracts the generated text completion.
    - It returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
5. **`submit_inference_request_to_mistral_api`**:
    - This function is responsible for submitting inference requests to the Mistral API for text completion and embedding tasks.
    - For "text_completion" requests:
        - It extracts the model parameters, input prompt, and number of completions from the `inference_request` object.
        - It uses the Mistral API client to send a streaming request for text completion, providing the model, input prompt, max tokens, and temperature.
        - It processes the streaming response chunks and accumulates the generated text completion.
        - It keeps track of the total input and output tokens used for the request.
        - It returns the output results and file type strings.
    - For "embedding" requests:
        - It extracts the input text from the `inference_request` object.
        - It uses the Mistral API client to send a request for embedding, providing the model and input text.
        - It processes the response and extracts the embedding vector.
        - It returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
6. **`submit_inference_request_to_groq_api`**:
    - This function is responsible for submitting inference requests to the Groq API for text completion and embedding tasks.
    - For "text_completion" requests:
        - It extracts the model parameters, input prompt, and number of completions from the `inference_request` object.
        - It uses the Groq API client to send a request for text completion, providing the model, input prompt, max tokens, and temperature.
        - It processes the response and extracts the generated text completions.
        - It keeps track of the total input and output tokens used for the request.
        - It returns the output results and file type strings.
    - For "embedding" requests:
        - It extracts the input text from the `inference_request` object.
        - It uses the Groq API client to send a request for embedding, providing the model and input text.
        - It processes the response and extracts the embedding vector.
        - It returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
7. **`submit_inference_request_to_claude_api`**:
    - This function is responsible for submitting inference requests to the Claude API (by Anthropic) for text completion and embedding tasks.
    - It uses the `get_claude3_model_name` helper function to map the model name from the `inference_request` to the corresponding Claude API model name.
    - For "text_completion" requests:
        - It extracts the model parameters, input prompt, and number of completions from the `inference_request` object.
        - It uses the Claude API client to send a streaming request for text completion, providing the model, input prompt, max tokens, and temperature.
        - It processes the streaming response and accumulates the generated text completions.
        - It keeps track of the total input and output tokens used for the request.
        - It returns the output results and file type strings.
    - For "embedding" requests:
        - It extracts the input text from the `inference_request` object.
        - It uses the Claude API client to send a request for embedding, providing the model and input text.
        - It processes the response and extracts the embedding vector.
        - It returns the output results and file type strings.
    - If any errors occur during the process or if the inference type is not supported, it logs an error message and returns `None`.
    
These functions play an important role in executing inference requests for API-based services and models. They handle the communication with the respective APIs, construct the necessary payloads based on the inference request details, and process the API responses to extract the relevant output results.

The rationale behind having separate functions for each API is to provide a modular and extensible approach to integrating different APIs into the Pastel Inference Layer. Each API has its own specific requirements, endpoints, and response formats, so having dedicated functions allows for customized handling of each API's peculiarities.

These functions also ensure that the output results are properly formatted and returned along with the appropriate file type strings. This is important for the Pastel Inference Layer to correctly interpret and handle the output results based on their data types. The functions enable the system to seamlessly integrate with various APIs, execute inference requests based on user specifications, and retrieve the generated output results. By providing a consistent interface for different APIs, these functions contribute to the flexibility and extensibility of the Pastel Inference Layer, allowing it to support a wide range of models and services.

----------

**Inference Request Execution Functions for Swiss Army Llama**

The next batch of service functions we will review are related to how inference requests are executed for models hosted using [Swiss Army Llama](https://github.com/Dicklesworthstone/swiss_army_llama), which is an open-source framework for running large language models (LLMs) and computing and storing embedding vectors of documents (and also of audio files, which are transcribed to text using [Whisper](https://github.com/SYSTRAN/faster-whisper)) in one of two ways:

- **Locally**— i.e., running on the CPU on the actual Supernode server itself;

- **On a remote server**— i.e., a GPU-enabled instance set up by the Supernode using a cost effective service such as [vast.ai](https://vast.ai/) and a [template image](https://cloud.vast.ai/?ref_id=78066&template_id=3c711e5ddd050882f0eb6f4ce1b8cc28) already set up for Swiss Army Llama; this remote GPU instance can then be shared across all of the Supernode operators various Supernodes; 

These functions are responsible for submitting the inference requests to the Swiss Army Llama service, handling the responses, and processing the output results. Let's go through each function and explain their purpose and how they contribute to the execution of inference requests:

1. **`determine_swiss_army_llama_port`**:
    - This function determines the appropriate port to use for communicating with the Swiss Army Llama service.
    - It checks if the local Swiss Army Llama service is responding using the `is_swiss_army_llama_responding` function with the `local` parameter set to `True`.
    - If the `USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE` configuration is set to `True`, it also checks if the remote Swiss Army Llama service is responding by calling `is_swiss_army_llama_responding` with `local` set to `False`.
    - If the remote Swiss Army Llama service is responding and the configuration allows its usage, the function returns the `REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT`.
    - If the local Swiss Army Llama service is responding, it returns the `SWISS_ARMY_LLAMA_PORT`.
    - If neither service is responding, it returns `None`.
    - This function helps in selecting the appropriate port based on the availability and configuration of the Swiss Army Llama services.
    
2. **`handle_swiss_army_llama_exception`**:
    - This is an exception handling function specific to Swiss Army Llama requests.
    - It takes the exception object `e`, the `client` instance, the `inference_request`, `model_parameters`, `port`, `is_fallback` flag, and the `handler_function` as parameters.
    - If the exception occurs while using the remote Swiss Army Llama service (indicated by the `port` being `REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT`) and it's not a fallback attempt, the function logs a message indicating a fallback to the local Swiss Army Llama service.
    - It then recursively calls the `handler_function` with the `port` set to `SWISS_ARMY_LLAMA_PORT` and `is_fallback` set to `True`.
    - If the exception occurs in any other scenario, it returns `None` for both the output results and file type strings.
    - This function provides a mechanism to handle exceptions during Swiss Army Llama requests and fallback to the local service if necessary.
    
3. **`handle_swiss_army_llama_text_completion`**:
    - This function handles text completion requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It constructs the payload for the text completion request, including the input prompt, LLM model name, temperature, number of tokens to generate, number of completions to generate, and grammar file string.
    - It sends a POST request to the Swiss Army Llama service endpoint `/get_text_completions_from_input_prompt/` with the payload and security token.
    - If the request is successful (status code 200), it processes the response JSON and extracts the generated text completion.
    - It identifies the data type of the output text using the `magika` library and sets the appropriate output file type strings.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output text and file type strings.
    
4. **`handle_swiss_army_llama_image_question`**:
    - This function handles image question requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It extracts the image data and question from the input data, which is assumed to be base64-encoded JSON.
    - It constructs the payload for the image question request, including the question, LLM model name, temperature, number of tokens to generate, and number of completions to generate.
    - It sends a POST request to the Swiss Army Llama service endpoint `/ask_question_about_image/` with the payload, image file, and security token.
    - If the request is successful (status code 200), it processes the response JSON and extracts the generated text answer.
    - It identifies the data type of the output text using the `magika` library and sets the appropriate output file type strings.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output text and file type strings.
    
5. **`handle_swiss_army_llama_embedding_document`**:
    - This function handles document embedding requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It extracts the document file data from the input data, which is assumed to be base64-encoded JSON.
    - It uploads the document file using the `upload_and_get_file_metadata` function and retrieves the file metadata (URL, hash, size).
    - It constructs the parameters for the document embedding request, including the LLM model name, embedding pooling method, corpus identifier string, JSON format, and whether to send back a JSON or ZIP file.
    - It sends a POST request to the Swiss Army Llama service endpoint `/get_all_embedding_vectors_for_document/` with the parameters, file metadata, and security token.
    - If the request is successful (status code 200), it processes the response based on the specified format (JSON or ZIP).
    - For JSON format, it directly returns the response JSON as the output results.
    - For ZIP format, it encodes the ZIP file content as base64 and sets the appropriate output file type strings.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output results and file type strings.
    
6. **`handle_swiss_army_llama_embedding_audio`**:
    - This function handles audio embedding requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It extracts the audio file data from the input data, which is assumed to be base64-encoded JSON.
    - It uploads the audio file using the `upload_and_get_file_metadata` function and retrieves the file metadata (URL, hash, size).
    - It constructs the parameters for the audio embedding request, including the option to compute embeddings for the resulting transcript document, LLM model name, embedding pooling method, and corpus identifier string.
    - It sends a POST request to the Swiss Army Llama service endpoint `/compute_transcript_with_whisper_from_audio/` with the parameters, file metadata, and security token.
    - If the request is successful (status code 200), it processes the response JSON as the output results.
    - If a query text is provided in the model parameters, it sends an additional request to the `/search_stored_embeddings_with_query_string_for_semantic_similarity/` endpoint to perform a semantic search on the embeddings.
    - It appends the search results to the output results.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output results and file type strings.
    
7. **`handle_swiss_army_llama_semantic_search`**:
    - This function handles semantic search requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It constructs the payload for the semantic search request, including the query text, number of most similar strings to return, LLM model name, embedding pooling method, and corpus identifier string.
    - It sends a POST request to the Swiss Army Llama service endpoint `/search_stored_embeddings_with_query_string_for_semantic_similarity/` with the payload and security token.
    - If the request is successful (status code 200), it processes the response JSON as the output results.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output results and file type strings.
    
8. **`handle_swiss_army_llama_advanced_semantic_search`**:
    - This function handles advanced semantic search requests using the Swiss Army Llama service.
    - It takes the `client` instance, `inference_request`, `model_parameters`, `port`, and `is_fallback` flag as parameters.
    - It constructs the payload for the advanced semantic search request, including the query text, LLM model name, embedding pooling method, corpus identifier string, similarity filter percentage, number of most similar strings to return, and result sorting metric.
    - It sends a POST request to the Swiss Army Llama service endpoint `/advanced_search_stored_embeddings_with_query_string_for_semantic_similarity/` with the payload and security token.
    - If the request is successful (status code 200), it processes the response JSON as the output results.
    - If an exception occurs during the request, it calls the `handle_swiss_army_llama_exception` function to handle the exception and potentially fallback to the local service.
    - It returns the output results and file type strings.
    
9. **`submit_inference_request_to_swiss_army_llama`**:
    - This is the main function for submitting inference requests to the Swiss Army Llama service.
    - It takes the `inference_request` and an optional `is_fallback` flag as parameters.
    - It determines the appropriate port for the Swiss Army Llama service using the `determine_swiss_army_llama_port` function.
    - If no valid port is found (indicating that neither the local nor remote service is responding), it logs an error message and returns `None` for both the output results and file type strings.
    - It decodes the model parameters from base64-encoded JSON.
    - Based on the `model_inference_type_string` of the inference request, it calls the corresponding handler function:
        - For "text_completion", it calls `handle_swiss_army_llama_text_completion`.
        - For "embedding_document", it calls `handle_swiss_army_llama_embedding_document`.
        - For "embedding_audio", it calls `handle_swiss_army_llama_embedding_audio`.
        - For "ask_question_about_an_image", it calls `handle_swiss_army_llama_image_question`.
        - For "semantic_search", it calls `handle_swiss_army_llama_semantic_search`.
        - For "advanced_semantic_search", it calls `handle_swiss_army_llama_advanced_semantic_search`.
    - If the inference type is not supported, it logs a warning message and returns `None` for both the output results and file type strings.
    - It returns the output results and file type strings obtained from the respective handler function.
    
These functions play a crucial role in executing inference requests using the Swiss Army Llama service: they handle the communication with the Swiss Army Llama service, construct the necessary payloads based on the inference request details, and process the service responses to extract the relevant output results.

The rationale behind having separate handler functions for different inference types is to provide a modular and extensible approach to supporting various capabilities of the Swiss Army Llama service. Each inference type has its own specific requirements, parameters, and response formats, so having dedicated handler functions allows for customized handling of each type.

The `submit_inference_request_to_swiss_army_llama` function serves as the main entry point for submitting inference requests to the Swiss Army Llama service. It determines the appropriate port to use based on the availability and configuration of the local and remote services. It then delegates the request to the corresponding handler function based on the inference type.

The exception handling mechanism implemented in the `handle_swiss_army_llama_exception` function provides a way to gracefully handle errors that may occur during the communication with the Swiss Army Llama service. It allows for fallback to the local service if the remote service encounters an exception, ensuring that the inference request can still be processed.

Overall, these functions enable the Pastel Inference Layer to seamlessly integrate with the Swiss Army Llama service, providing users with the ability to run large language models locally or on a remote server. They handle the execution of various inference types, such as text completion, document and audio embedding, image question answering, and semantic search. By offering flexibility in the deployment options and supporting different inference capabilities, these functions contribute to the versatility and robustness of the Pastel Inference Layer.

----------

**Inference Request Execution and Result Retrieval Functions**

The final batch of service functions we will review are related to the overall execution of inference requests, checking the status of inference results, and retrieving the output results while verifying authorization. These functions tie together the various components of the inference request flow and provide the necessary endpoints for users to interact with the system. Let's go through each function and explain their purpose and how they contribute to the inference request execution and result retrieval process:

1. **`execute_inference_request`**:
    - This is the main function responsible for executing an inference request.
    - It takes the `inference_request_id` as a parameter, which uniquely identifies the inference request.
    - It retrieves the corresponding `InferenceAPIUsageRequest` from the database using the provided `inference_request_id`.
    - If the `InferenceAPIUsageRequest` is not found, it logs a warning message and returns, indicating an invalid inference request ID.
    - It also retrieves the associated `InferenceAPIUsageResponse` from the database using the same `inference_request_id`.
    - Based on the `requested_model_canonical_string` of the `InferenceAPIUsageRequest`, it determines the appropriate API or service to submit the inference request to:
        - If the model starts with "stability-", it calls the `submit_inference_request_to_stability_api` function.
        - If the model starts with "openai-", it calls the `submit_inference_request_to_openai_api` function.
        - If the model starts with "mistralapi-", it calls the `submit_inference_request_to_mistral_api` function.
        - If the model starts with "groq-", it calls the `submit_inference_request_to_groq_api` function.
        - If the model contains "claude" (case-insensitive), it calls the `submit_inference_request_to_claude_api` function.
        - If the model starts with "openrouter/", it calls the `submit_inference_request_to_openrouter` function.
        - If the model starts with "swiss_army_llama-", it calls the `submit_inference_request_to_swiss_army_llama` function.
        - If none of the above conditions match, it raises a `ValueError` indicating an unsupported provider or model.
    - The selected submission function is called with the `InferenceAPIUsageRequest` object, and it returns the `output_results` and `output_results_file_type_strings`.
    - If the `output_results` and `output_results_file_type_strings` are not `None`, it calls the `save_inference_output_results` function to save the inference output results to the database.
    - If any exception occurs during the execution process, it logs the error, prints the traceback, and re-raises the exception.
    
2. **`check_status_of_inference_request_results`**:
    - This function checks the status of an inference request's results.
    - It takes the `inference_response_id` as a parameter, which uniquely identifies the inference response.
    - It retrieves the corresponding `InferenceAPIOutputResult` from the database using the provided `inference_response_id`.
    - If the `InferenceAPIOutputResult` is found, it means the inference request has been processed and the results are available, so the function returns `True`.
    - If the `InferenceAPIOutputResult` is not found, it means the inference request is still being processed or has not yet started, so the function returns `False`.
    - If any exception occurs during the status check, it logs the error and re-raises the exception.
    
3. **`get_inference_output_results_and_verify_authorization`**:
    - This function retrieves the inference output results and verifies the authorization of the requesting user.
    - It takes the `inference_response_id` and `requesting_pastelid` as parameters.
    - It retrieves the corresponding `InferenceAPIOutputResult` from the database using the provided `inference_response_id`.
    - If the `InferenceAPIOutputResult` is not found, it raises a `ValueError` indicating that the inference output results are not found.
    - It then retrieves the associated `InferenceAPIUsageRequest` from the database using the `inference_request_id` from the `InferenceAPIOutputResult`.
    - If the `InferenceAPIUsageRequest` is not found or the `requesting_pastelid` does not match the `requesting_pastelid` of the `InferenceAPIUsageRequest`, it raises a `ValueError` indicating unauthorized access to the inference output results.
    - If the authorization is successful, it returns the `InferenceAPIOutputResult` object.
    
These functions play a critical role in the overall execution and management of inference requests in the Pastel Inference Layer. The `execute_inference_request` function serves as the central point of control for executing an inference request. It determines the appropriate API or service to submit the request to based on the model specified in the `InferenceAPIUsageRequest`. By delegating the actual submission to the respective functions for each API or service, it maintains a modular and extensible architecture.

The `check_status_of_inference_request_results` function provides a way for users to check the status of their inference requests. It allows them to determine whether the results are available or if the request is still being processed. This function is typically called by the user periodically to poll for the availability of the results.

The `get_inference_output_results_and_verify_authorization` function is responsible for retrieving the inference output results and verifying the authorization of the requesting user. It ensures that only the user who initiated the inference request can access the corresponding output results. This function is called when the user wants to retrieve the results after confirming their availability through the status check.

The use of database queries and transactions in these functions ensures data consistency and integrity. The functions retrieve the necessary objects from the database, such as `InferenceAPIUsageRequest`, `InferenceAPIUsageResponse`, and `InferenceAPIOutputResult`, to perform the required operations. The database transactions are managed using the `db_code.Session` context manager, which handles the session lifecycle and ensures proper resource management.

Error handling is also incorporated into these functions. If any exceptions occur during the execution or retrieval process, the functions log the errors, print the tracebacks, and re-raise the exceptions. This allows for proper error propagation and handling at higher levels of the system.

Overall, these functions provide the necessary functionality for executing inference requests, checking the status of results, and retrieving the output results while verifying authorization. They encapsulate the complexity of interacting with different APIs and services, and provide a consistent interface for users to interact with the Pastel Inference Layer. By leveraging the modular design and database transactions, these functions ensure a reliable and secure execution and retrieval process for inference requests.

----------

## Misc Other Infrastructure Code

Here, we explain other parts of the Inference Server system code that we haven't touched on yet above. First of these is the code used by the Inference Server to initially install, set up, configure, and update Swiss Army Llama for local operation on the Supernode server itself (setting Swiss Army Llama up on a remote machine for use with the Inference Server is a separate process that we will detail in a later section). This is handled from the code file `setup_swiss_army_llama.py`, which is called from the `main.py` entrypoint code file in the Inference Server as a startup task of the FastAPI server. The complete code for `setup_swiss_army_llama.py` can be found [here](https://github.com/pastelnetwork/python_inference_layer_server/blob/master/setup_swiss_army_llama.py).

The `setup_swiss_army_llama.py` file contains various utility functions and a main function `check_and_setup_swiss_army_llama` that orchestrates the entire setup process. Let's go through each function and explain their purpose and how they contribute to setting up and managing the Swiss Army Llama service:

1. `get_external_ip_func`:
    - This function attempts to retrieve the external IP address of the server by querying several public IP address providers.
    - It iterates through a list of providers and sends HTTP GET requests to each provider until a successful response is received.
    - If a provider returns a valid IP address, the function returns it.
    - If all providers fail to provide an IP address, the function returns "Unknown".
    
2. `run_command`:
    - This function is a wrapper around the `subprocess.run` function to execute shell commands.
    - It takes the command as a string or a list of command arguments, along with optional parameters for environment variables, output capture, error checking, and timeout.
    - It uses the default shell (`/bin/zsh` if available, otherwise `/bin/bash`) to execute the command.
    - If `capture_output` is set to `True`, the function captures and logs the command's stdout and stderr.
    - If the command times out or fails, the function logs the appropriate error message.
    - It returns the `subprocess.CompletedProcess` object representing the executed command.
    
3. `is_port_available`:
    - This function checks if a specified port is available (not in use) on the server.
    - It uses the `lsof` command to list open files and sockets associated with the port.
    - If the command's return code is non-zero, it means the port is available.
    
4. `is_swiss_army_llama_responding`:
    - This function checks if the Swiss Army Llama service is responding on a specified IP address and port.
    - It constructs a URL using the provided IP address, port, and the `/get_list_of_available_model_names/` endpoint.
    - It sends an HTTP GET request to the URL with the security token as a parameter.
    - If the response status code is 200, it means the service is responding.
    
5. `update_security_token`:
    - This function updates the security token in the Swiss Army Llama configuration file.
    - It reads the content of the specified file and uses regular expressions to replace the existing security token with the provided new token.
    - It writes the updated content back to the file.
    
6. `is_pyenv_installed`:
    - This function checks if the `pyenv` tool is installed on the server.
    - It runs the `pyenv --version` command and checks the return code.
    - If the return code is 0, it means `pyenv` is installed.
    
7. `is_python_3_12_installed`:
    - This function checks if Python 3.12 is installed using `pyenv`.
    - It runs the `pyenv versions` command and checks if "3.12" is present in the output.
    
8. `is_rust_installed`:
    - This function checks if the Rust programming language is installed on the server.
    - It runs the `rustc --version` command and checks the return code.
    - If the return code is 0, it means Rust is installed.
    - If the command throws a `FileNotFoundError`, it means Rust is not installed.
    
9. `setup_virtual_environment`:
    - This function sets up a Python virtual environment for Swiss Army Llama.
    - It creates a virtual environment directory if it doesn't exist.
    - It upgrades `pip`, installs the `wheel` package, and installs the dependencies from the `requirements.txt` file.
    - It returns the path to the Python executable within the virtual environment.
    
10. `set_timezone_utc`:
    - This function sets the timezone to UTC.
    - It sets the `TZ` environment variable to 'UTC'.
    - It adds the `export TZ=UTC` line to the user's shell profile file (`~/.zshrc` or `~/.bashrc`).
    
11. `check_systemd_service_exists`:
    - This function checks if a systemd service with the specified name exists and is enabled.
    - It runs the `systemctl is-enabled <service_name>` command and checks the return code and output.
    - If the return code is 0 and the output contains 'enabled', it means the service exists.
    
12. `create_systemd_service`:
    - This function creates a systemd service file for Swiss Army Llama.
    - It constructs the service file content with the provided service name, user, working directory, and command to execute.
    - It writes the service file content to a temporary file and moves it to the systemd directory.
    - It reloads the systemd daemon, enables the service, and starts the service.
    - It logs the status of the service after starting it.
    
13. `ensure_pyenv_setup`:
    - This function ensures that `pyenv` is installed and Python 3.12 is set up.
    - If `pyenv` is not installed, it installs the necessary dependencies and installs `pyenv`.
    - If Python 3.12 is not installed, it installs Python 3.12 using `pyenv` and sets it as the global Python version.
    
14. `configure_shell_for_pyenv`:
    - This function configures the shell environment for `pyenv`.
    - It adds the necessary `pyenv` initialization commands to the user's shell profile file (`~/.zshrc` or `~/.bashrc`).
    - It updates the `PYENV_ROOT` and `PATH` environment variables to include `pyenv` paths.
    
15. `has_repo_been_updated`:
    - This function checks if the Swiss Army Llama repository has been updated.
    - It fetches the latest changes from the remote repository.
    - It compares the local commit hash with the remote commit hash of the `main` branch.
    - If the commit hashes are different, it means the repository has been updated.
    
16. `setup_swiss_army_llama`:
    - This function performs the actual setup of Swiss Army Llama.
    - It sets the timezone to UTC.
    - It clones the Swiss Army Llama repository if it doesn't exist.
    - It checks for updates in the repository and pulls the latest changes if updates are available.
    - It configures the shell environment for `pyenv` and ensures `pyenv` and Python 3.12 are installed.
    - It sets up a Python virtual environment for Swiss Army Llama and installs the dependencies.
    - If Rust is not installed, it installs Rust and sets up the necessary environment variables.
    - If the Swiss Army Llama systemd service doesn't exist, it creates the service file and starts the service.
    - If the service already exists, it reloads the systemd daemon, enables the service, and starts it.
    
17. `kill_running_instances_of_swiss_army_llama`:
    - This function kills any running instances of Swiss Army Llama.
    - It stops the Swiss Army Llama systemd service.
    - It finds and kills any remaining Swiss Army Llama processes using the `ps` and `kill` commands.
    
18. `check_and_setup_swiss_army_llama`:
    - This is the main function that orchestrates the setup process.
    - It retrieves the external IP address of the server.
    - It checks if the Swiss Army Llama repository has been updated.
    - It checks if the Swiss Army Llama service is responding on the specified port.
    - It checks if the Swiss Army Llama port is available.
    - If the service is responding and the repository hasn't been updated, it considers Swiss Army Llama to be already set up and skips the setup process.
    - If the service is not responding or the repository has been updated, it kills any running instances of Swiss Army Llama and runs the setup process.

These functions work together to automate the setup and management of the Swiss Army Llama service on the Supernode server. The `check_and_setup_swiss_army_llama` function serves as the entry point and is called from the `main.py` file as a startup task of the FastAPI server.

By checking for updates in the Swiss Army Llama repository, ensuring the necessary dependencies are installed, configuring the environment, and managing the systemd service, this code ensures that the Swiss Army Llama service is properly set up and running on the Supernode server. It also handles scenarios where the service may need to be restarted or updated based on changes in the repository or the service's responsiveness.

----------

**Environment Variable Setup and Processing, Including Decryption of API Keys and Passwords**

The Pastel Inference Layer server relies on environment variables to store and manage various configuration settings, including sensitive information such as API keys and passwords. These environment variables are typically stored in a `.env` file, which is loaded and processed during the server startup.

The `.env` file contains a wide range of configuration settings, covering aspects such as:

- Server port numbers
- Timeout values
- Transaction settings
- Credit pack settings
- API keys for external services (e.g., OpenAI, Anthropic, Stability)
- Passphrases for local and remote Pastel IDs

You can see the sample .env file below. Not that the very long values are actually encrypted secrets that rely upon the existence of a saved decryption key stored somewhere on the machine:

```Dockerfile
    UVICORN_PORT=7123
    TEMP_OVERRIDE_LOCALHOST_ONLY=1
    CHALLENGE_EXPIRATION_TIME_IN_SECONDS=300
    NUMBER_OF_DAYS_BEFORE_MESSAGES_ARE_CONSIDERED_OBSOLETE=3
    GITHUB_MODEL_MENU_URL=https://raw.githubusercontent.com/pastelnetwork/python_inference_layer_server/master/model_menu.json
    MESSAGING_TIMEOUT_IN_SECONDS=60
    BASE_TRANSACTION_AMOUNT=0.000001
    FEE_PER_KB=0.0001
    MAXIMUM_NUMBER_OF_PASTEL_BLOCKS_FOR_USER_TO_SEND_BURN_AMOUNT_FOR_CREDIT_TICKET=50
    MAXIMUM_LOCAL_CREDIT_PRICE_DIFFERENCE_TO_ACCEPT_CREDIT_PRICING=0.25
    MAXIMUM_LOCAL_PASTEL_BLOCK_HEIGHT_DIFFERENCE_IN_BLOCKS=1
    MINIMUM_NUMBER_OF_PASTEL_BLOCKS_BEFORE_TICKET_STORAGE_RETRY_ALLOWED=10
    MINIMUM_CONFIRMATION_BLOCKS_FOR_CREDIT_PACK_BURN_TRANSACTION=1
    MAXIMUM_LOCAL_UTC_TIMESTAMP_DIFFERENCE_IN_SECONDS=55.0
    BURN_TRANSACTION_MAXIMUM_AGE_IN_DAYS=3
    SUPERNODE_CREDIT_PRICE_AGREEMENT_QUORUM_PERCENTAGE=0.51
    SUPERNODE_CREDIT_PRICE_AGREEMENT_MAJORITY_PERCENTAGE=0.85
    CREDIT_PACK_FILE=credit_pack.json
    CREDIT_USAGE_TO_TRACKING_AMOUNT_MULTIPLIER=10
    CREDIT_COST_MULTIPLIER_FACTOR=0.02
    API_KEY_TEST_VALIDITY_HOURS=72
    TARGET_VALUE_PER_CREDIT_IN_USD=0.01
    TARGET_PROFIT_MARGIN=0.15
    MAXIMUM_PER_CREDIT_PRICE_IN_PSL_FOR_CLIENT=100.0
    MINIMUM_COST_IN_CREDITS=0.5
    SWISS_ARMY_LLAMA_PORT=8089
    USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE=1
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_SSH_KEY_PATH=/home/ubuntu/vastai_privkey
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_IP_ADDRESS=106.185.159.136
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_PORT=20765
    REMOTE_SWISS_ARMY_LLAMA_EXPOSED_PORT=8089
    REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT=8087
    SWISS_ARMY_LLAMA_SECURITY_TOKEN=gAAAAABmEFF7K1lIlVGm3P8HxZoRJMdr0EIioE3Ug1x9PKaqLi620LOvSALMEcG7drPr61deEkcP2SRplE9qv-sjgwjE964axw%3D%3D
    OPENAI_API_KEY=gAAAAABmEXkqBi83ORWgoTMTvvg0d4ehGjTuCVFguKPVETPAsg7VhRE_Ecs4tb0WFylElFkhQTtHthcDQtzHQ_Do2M0SO2hYqG53vTANcBhPUbSAxwM0sRXcmqeEX1OWhaHbRze67DzkvzqBS1KnQtaUzBtUuusQ6w%3D%3D
    CLAUDE3_API_KEY=gAAAAABmEFF73nD1mrch6DKztNtk9-Kv6PvPpnHSYknatGR4ejc2tDyYc7aqqmxPR03AO23eR_VXIChhNt9q0ZSHB4A2tDVtTCRpsB3xywCe91_TTViyg0VDXj1hlKM-urXGHoKLnyn_xWqsrhTj6WP-ebjqWhdyK0cM-izqWsHFuXWvac1RGwmgZpadweFWYJOAvviTLClcBJaHANviG2_HGkvxfWr8GQ%3D%3D
    GROQ_API_KEY=gAAAAABmEJe9xW7Fejwg2buHRymIH5v7TcJGk7P_lLPEG91RJ28iLwHIJ7PCt77drz_Hz0z-aOAp-GgC6h6_DhZ0yw9H5vFsCtMAMmEOKRcJOMnbrGs-hcJRm6r3Cg3_tqwcSUG1XjVk4MeEYim9ZkbTBcG_AeHi0g%3D%3D
    MISTRAL_API_KEY=gAAAAABmEJe98LKZmYTux4IMIiSLHV3jftgxCr7zR_RKZx3xuMwiw5xv2rCy4hOY9Inyg2kkoFke8_o55RBLxNUe8UHOZIQmoa38o6XhHCxpydLlJORytFkb7fI6pKWv0hPNgnFiMjw6
    STABILITY_API_KEY=gAAAAABmEM0J9Y4HtTRPcqbhy5R3UZpDQvHZrUY6UKbcMRh-RsV7JROD1nCxGcDElDkdt61qgVZWOUiCpNDdE5w_hxV7wmAuQosh-Vsfapp25cb2qMA3xJzK8Aoo1X1gdyakDq2xHJWhhtFWpCDTBcrt6p799tMfCQ%3D%3D
    OPENROUTER_API_KEY=gAAAAABmFzAGXWhWQZox78SWuj5SUvesTkCU-ry5uf2YZyqfUaF1vZjkR6zwbghs9vdbqLEv4rSqu0FSGewDiIG-jFw8JpY2p9AIt9h2M0Mhc8JkRp9fXGVnCR8YFqaw3aQZmmbpwSobshjqw9oWdgnl3rmpd1SSRXtX5RLpXX7vWPiFbrBZj_8%3D
    LOCAL_PASTEL_ID_PASSPHRASE=gAAAAABmEFF72YuPdwL_cNZSu54NLVxiT9s6QwGpUn0dnHoBbHRq7T-vK_vAuBV0HFrpnlquzxrmknsYKuaiWktTjOSH4knDnA%3D%3D
    MY_LOCAL_PASTELID=jXYdog1FfN1YBphHrrRuMVsXT76gdfMTvDBo2aJyjQnLdz2HWtHUdE376imdgeVjQNK93drAmwWoc7A3G4t2Pj
    MY_PASTELID_PASSPHRASE=gAAAAABmEFF7yMxMmyz4xaSa7ykie-487B42GOgzlR0HGa0K08M7Ow3DZHNt6W46M_SQQQaKtFMM-OTVJDIE51AQh4CBchmU1g%3D%3D
```

The use of environment variables allows for flexible configuration of the server without the need to modify the codebase directly. It also enables easy management of different configurations for various environments (e.g., development, staging, production).

To process the environment variables, the Pastel Inference Layer server utilizes the `python-decouple` library. This library provides a convenient way to load environment variables from a file and access them throughout the application. The `Config` class from `python-decouple` is used to create a `config` object, which acts as a centralized point of access for the environment variables.
One critical aspect of the environment variable setup is the handling of sensitive information, such as API keys and passphrases. To ensure the security of these sensitive values, the Pastel Inference Layer server employs encryption techniques. The sensitive fields in the `.env` file are encrypted using the `Fernet` symmetric encryption algorithm from the `cryptography` library.

The encryption and decryption process is handled by a set of dedicated functions in the `service_functions.py` module. These functions include:


- `generate_or_load_encryption_key_sync`: Generates or loads the encryption key synchronously.
- `encrypt_sensitive_fields`: Encrypts the sensitive fields in the `.env` file.
- `decrypt_sensitive_data`: Decrypts the encrypted data using the provided encryption key.
- `encrypt_sensitive_data`: Encrypts the sensitive data using the provided encryption key.
- `decrypt_sensitive_fields`: Decrypts the sensitive fields and assigns them to global variables.

The encryption key is generated or loaded during the server startup process. If an encryption key file doesn't exist, a new key is generated using the `Fernet.generate_key()` method and saved to a file for future use. The encryption key is then used to encrypt the sensitive fields in the `.env` file, replacing the plain-text values with encrypted strings.

During runtime, the `decrypt_sensitive_fields` function is called to decrypt the sensitive fields using the encryption key. The decrypted values are assigned to global variables, making them accessible throughout the application. This approach ensures that the sensitive information remains protected at rest (in the `.env` file) and is only decrypted when needed during execution.
The encryption and decryption process strikes a balance between security and usability. By encrypting the sensitive fields, the Pastel Inference Layer server minimizes the risk of unauthorized access to critical information. At the same time, the decryption of these fields during runtime allows the server to utilize the necessary API keys and passphrases for its operations.

In summary, the environment variable setup and processing in the Pastel Inference Layer server play a vital role in the configuration and security of the application. The use of a `.env` file and the `python-decouple` library enables flexible and centralized management of configuration settings. The encryption and decryption of sensitive fields using the `Fernet` algorithm ensures the protection of critical information while still allowing seamless integration with the server's functionality. This setup demonstrates a thoughtful approach to balancing configuration flexibility, security, and ease of use in the Pastel Inference Layer server.

----------

**Ansible Playbook for Deploying Inference Server**

The Inference Server repo includes an ansible playbook called [pastel_inference_layer_deployment_playbook.yml](https://github.com/pastelnetwork/python_inference_layer_server/blob/master/setup_scripts_and_playbooks/pastel_inference_layer_deployment_playbook.yml) that largely automates the process of deploying the system on a fresh machine, or updating a configured inference server Supernode when the inference code is updated. 

The complete playbook is shown below, with explanation to follow:

```YAML
---
- name: Pastel Inference Layer Deployment
  hosts: all
  become: yes
  vars:
    ubuntu_user: ubuntu
    oh_my_zsh_install_script: "https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh"
    rust_install_script: "https://sh.rustup.rs"
    atuin_install_script: "https://setup.atuin.sh"
    zshrc_path: "/home/{{ ubuntu_user }}/.zshrc"
    bashrc_path: "/home/{{ ubuntu_user }}/.bashrc"
    oh_my_zsh_install_flag: "/home/{{ ubuntu_user }}/.oh-my-zsh"
    home_dir: "/home/{{ ubuntu_user }}"

  tasks:
    - name: Update and upgrade apt packages
      ansible.builtin.apt:
        update_cache: yes
        upgrade: dist
        autoremove: yes

    - name: Check if zsh is installed
      command: which zsh
      register: zsh_installed
      ignore_errors: yes

    - name: Ensure zsh_installed is defined
      set_fact:
        zsh_installed: { "rc": 1 }
      when: zsh_installed is not defined

    - name: Check if Oh My Zsh is installed
      stat:
        path: "{{ oh_my_zsh_install_flag }}"
      register: oh_my_zsh_installed

    - name: Install Oh My Zsh
      become_user: "{{ ubuntu_user }}"
      shell: >
        sh -c "$(curl -fsSL {{ oh_my_zsh_install_script }})" && touch {{ oh_my_zsh_install_flag }}
      when: not oh_my_zsh_installed.stat.exists

    - name: Install Rust
      become_user: "{{ ubuntu_user }}"
      shell: >
        curl -fsSL {{ rust_install_script }} | sh -s -- -y

    - name: Ensure Rust environment is loaded
      lineinfile:
        path: "{{ zshrc_path }}"
        regexp: 'source $HOME/.cargo/env'
        line: 'source $HOME/.cargo/env'
        state: present
      become_user: "{{ ubuntu_user }}"

    - name: Install Atuin
      become_user: "{{ ubuntu_user }}"
      shell: >
        /bin/bash -c "$(curl --proto '=https' --tlsv1.2 -sSf {{ atuin_install_script }})"

    - name: Ensure Atuin environment is loaded
      lineinfile:
        path: "{{ zshrc_path }}"
        regexp: 'eval "$(atuin init zsh)"'
        line: 'eval "$(atuin init zsh)"'
        state: present
      become_user: "{{ ubuntu_user }}"

    - name: Install dependencies for pyenv
      apt:
        name:
          - build-essential
          - libssl-dev
          - zlib1g-dev
          - libbz2-dev
          - libreadline-dev
          - libsqlite3-dev
          - wget
          - curl
          - llvm
          - libncurses5-dev
          - libncursesw5-dev
          - xz-utils
          - tk-dev
          - libffi-dev
          - liblzma-dev
          - python3-openssl
          - git
        state: present
        update_cache: yes

    - name: Clone pyenv repository
      git:
        repo: 'https://github.com/pyenv/pyenv.git'
        dest: '{{ home_dir }}/.pyenv'
        update: yes
        force: yes

    - name: Set pyenv environment variables in .zshrc if zsh is found
      blockinfile:
        path: '{{ zshrc_path }}'
        block: |
          export PYENV_ROOT="$HOME/.pyenv"
          export PATH="$PYENV_ROOT/bin:$PATH"
          eval "$(pyenv init --path)"
      when: zsh_installed.rc == 0

    - name: Set pyenv environment variables in .bashrc if zsh is not found
      blockinfile:
        path: '{{ bashrc_path }}'
        block: |
          export PYENV_ROOT="$HOME/.pyenv"
          export PATH="$PYENV_ROOT/bin:$PATH"
          eval "$(pyenv init --path)"
      when: zsh_installed.rc != 0

    - name: Ensure pyenv directories have correct permissions
      file:
        path: /home/ubuntu/.pyenv
        state: directory
        owner: ubuntu
        group: ubuntu
        mode: '0755'
        recurse: yes

    - name: Ensure pyenv is available and install Python 3.12
      become: no
      shell: |
        export PATH="/home/ubuntu/.pyenv/bin:$PATH"
        eval "$(pyenv init --path)"
        eval "$(pyenv init -)"
        pyenv install 3.12
      args:
        executable: /bin/bash
      environment:
        PYENV_ROOT: "/home/ubuntu/.pyenv"
      become_user: ubuntu

    - name: Check again if zsh is installed
      command: which zsh
      register: zsh_installed
      ignore_errors: yes
      failed_when: false

    - name: Replace /root/ with /home/ubuntu/ in .zshrc
      replace:
        path: "/home/{{ ubuntu_user }}/.zshrc"
        regexp: "/root/"
        replace: "/home/{{ ubuntu_user }}/"

    - name: Determine the user shell
      shell: echo $SHELL
      register: user_shell

    - name: Set shell path and profile file
      set_fact:
        shell_path: "{{ user_shell.stdout }}"
        profile_file: "{{ 'zshrc' if '/zsh' in user_shell.stdout else 'bashrc' }}"

    - name: Check if the application directory exists
      stat:
        path: /home/{{ ubuntu_user }}/python_inference_layer_server
      register: app_dir

    - name: Clone the repository if the directory doesn't exist
      git:
        repo: https://github.com/pastelnetwork/python_inference_layer_server
        dest: /home/{{ ubuntu_user }}/python_inference_layer_server
      when: not app_dir.stat.exists

    - name: Run initial setup script if the directory was just created
      shell: |
        chmod +x setup_scripts_and_playbooks/initial_inference_server_setup_script.sh
        ./setup_scripts_and_playbooks/initial_inference_server_setup_script.sh
      args:
        chdir: /home/{{ ubuntu_user }}/python_inference_layer_server
      when: not app_dir.stat.exists

    - name: Update code
      shell: |
        source /home/{{ ubuntu_user }}/.{{ profile_file }}
        git stash
        git pull
      args:
        chdir: /home/{{ ubuntu_user }}/python_inference_layer_server
        executable: "{{ shell_path }}"

    - name: Get the name of the existing tmux session
      shell: tmux list-sessions -F '#{session_name}' | head -1
      register: tmux_session_name
      ignore_errors: true

    - name: Create tmux session if it doesn't exist
      shell: tmux new-session -d -s default_session
      when: tmux_session_name.stdout == ""
      args:
        executable: "{{ shell_path }}"

    - name: Set the tmux session name
      set_fact:
        session_name: "{{ tmux_session_name.stdout if tmux_session_name.stdout else 'default_session' }}"

    - name: Check if supernode_script window exists
      shell: tmux list-windows -t {{ session_name }} -F '#{window_name}' | grep -q '^supernode_script$'
      register: window_exists
      ignore_errors: true

    - name: Kill supernode_script window if it exists
      shell: tmux kill-window -t {{ session_name }}:supernode_script
      when: window_exists.rc == 0

    - name: Create temporary script
      copy:
        content: |
          #!/bin/{{ 'zsh' if '/zsh' in shell_path else 'bash' }}
          source /home/{{ ubuntu_user }}/.{{ profile_file }}
          cd /home/{{ ubuntu_user }}/python_inference_layer_server
          pyenv local 3.12
          source venv/bin/activate
          python -m pip install --upgrade pip
          python -m pip install --upgrade setuptools
          python -m pip install wheel
          pip install -r requirements.txt
          python main.py
        dest: /home/{{ ubuntu_user }}/run_script.sh
        mode: '0755'

    - name: Launch script in new tmux window
      shell: |
        tmux new-window -t {{ session_name }}: -n supernode_script -d "{{ shell_path }} -c '/home/{{ ubuntu_user }}/run_script.sh'"
      args:
        executable: "{{ shell_path }}"

    - name: Remove temporary script
      file:
        path: /home/{{ ubuntu_user }}/run_script.sh
        state: absent

```

**Detailed Breakdown of the Ansible Playbook for the Pastel Inference Layer Deployment:**

The Ansible playbook above is designed to automate the deployment of the Pastel Inference Layer on a server. It performs various tasks to prepare the environment, install necessary software, and set up the application for running within a `tmux` session. Below is a comprehensive breakdown of each step:

#### Initial Setup and Package Management
The playbook begins by updating the package lists and upgrading the installed packages. This ensures that the system is up to date with the latest security patches and software versions. The `autoremove` option cleans up unnecessary packages, keeping the system lean.

#### Checking and Installing Oh My Zsh
The playbook checks if Oh My Zsh is already installed by looking for a specific file. If it is not installed, it uses a shell script to install Oh My Zsh, which provides an enhanced shell experience.

#### Installing Rust
Rust is installed using a provided script. Rust is essential for compiling certain dependencies and tools that may be required by the Pastel Inference Layer.

#### Configuring Shell Environments
The playbook ensures that the Rust environment variables are loaded by adding them to the `.zshrc` file. This allows the user to use Rust tools seamlessly.

#### Installing Atuin
Atuin, a tool for enhancing shell history, is installed. The playbook adds the necessary initialization commands to `.zshrc` to ensure it is available in the shell.

#### Setting Up pyenv and Python
The playbook installs dependencies required for building Python from source and then clones the `pyenv` repository. `pyenv` allows for easy management of multiple Python versions.

Depending on whether `zsh` or `bash` is the default shell, it adds `pyenv` initialization commands to either `.zshrc` or `.bashrc`. This ensures `pyenv` is set up correctly regardless of the shell in use.

#### Ensuring pyenv Initialization
The playbook runs commands to initialize `pyenv` in the respective shell and checks if `pyenv` is correctly installed by verifying its version. This step ensures that the environment is correctly set up for Python version management.

#### Installing Python 3.12 and Setting Up Virtual Environment
The playbook installs Python 3.12 using `pyenv`. It then creates and activates a virtual environment within the project directory. This isolated environment ensures that dependencies for the Pastel Inference Layer do not interfere with other Python projects on the system.

#### Cloning the Application Repository
The playbook checks if the application directory exists. If it does not, it clones the repository from GitHub and runs an initial setup script. This script likely sets up necessary configurations and dependencies specific to the Pastel Inference Layer.

#### Updating the Application Code
If the repository already exists, the playbook stashes any local changes and pulls the latest code from the repository. This ensures the application is up to date with the latest changes.

#### Setting Up and Managing tmux Session
The playbook uses `tmux` to create a new session or attach to an existing one. This allows the application to run in the background, even if the SSH session is disconnected.

It checks if a specific `tmux` window for running the application exists. If it does, it kills the window to prevent multiple instances from running. It then creates a new window within the `tmux` session to run the application.

A temporary script is created and executed within the `tmux` window to start the application. This script sets up the environment, activates the virtual environment, installs any missing dependencies, and starts the application.

#### Cleaning Up
After launching the application, the playbook removes the temporary script to keep the system clean.

----------

**Using a Remote GPU-Enabled Instance of Swiss Army Llama on Vast.ai to Speed Up Inference**
 
The Pastel Inference Layer server supports the use of a remote GPU-enabled instance of Swiss Army Llama hosted on Vast.ai to significantly speed up inference tasks. This approach leverages the power of GPU acceleration while maintaining a cost-effective and decentralized infrastructure.
The process involves using a custom Docker image that encapsulates the necessary dependencies and configurations for running Swiss Army Llama on a GPU-enabled instance. The Docker image is based on the PyTorch image with CUDA and cuDNN pre-installed, ensuring compatibility with GPU acceleration.

The Dockerfile performs the following steps:

1. Sets environment variables for FAISS GPU build and CUDA support.
2. Installs necessary packages and dependencies, including build tools, libraries, and utilities.
3. Installs Rust and Atuin for additional functionality.
4. Clones the Swiss Army Llama repository and sets the working directory.
5. Installs Python dependencies using conda and pip, including the GPU-enabled version of FAISS.
6. Exposes the necessary ports for the Swiss Army Llama server and Redis.
7. Creates a benchmark script to assess the performance of the host machine.
8. Defines the command to start the benchmark, Redis server, and Swiss Army Llama server when the container starts.
9. 

The resulting Docker image is published on Docker Hub [here](https://hub.docker.com/repository/docker/jemanuel82/vastai_swiss_army_llama_template/general), making it easily accessible for deployment. The exact Dockerfile contents is shown below:

```bash
    # Use the PyTorch image with CUDA and cuDNN pre-installed
    FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel
    
    # Set environment variables for FAISS GPU build and CUDA support
    ENV DEBIAN_FRONTEND=noninteractive
    ENV PYTHONUNBUFFERED=1
    ENV DATA_DIRECTORY=/swiss_army_llama
    ENV FAISS_ENABLE_GPU=ON
    ENV LLAMA_CUDA=on
    ENV PATH="/root/.cargo/bin:${PATH}"
    
    # Pre-accept the Microsoft EULA for ttf-mscorefonts-installer and update/install packages
    RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections && \
        apt-get update && \
        apt-get install -y \
        build-essential curl pkg-config ffmpeg gcc g++ gdb gdisk git cmake htop psmisc sysbench \
        libboost-all-dev libssl-dev make nano nginx nodejs npm openssh-client iperf3 \
        openssh-server openssl rsync libpulse-dev software-properties-common inetutils-ping \
        ubuntu-release-upgrader-core ubuntu-restricted-extras ufw unzip vcsh vim vpnc \
        zip zlib1g-dev zstd libpq-dev libmagic1 libxml2-dev libxslt1-dev antiword \
        unrtf poppler-utils tesseract-ocr flac lame libmad0 libsox-fmt-mp3 sox \
        libjpeg-dev swig redis-server wget libffi-dev libbz2-dev libreadline-dev \
        libsqlite3-dev llvm libncurses5-dev xz-utils tk-dev libxmlsec1-dev liblzma-dev \
        libncursesw5-dev python3-openssl libpoppler-cpp-dev pstotext net-tools iproute2 multitail && \
        apt-get upgrade -y && \
        apt-get autoremove -y && \
        rm -rf /var/lib/apt/lists/*
    
    # Install Rust and Atuin
    RUN curl https://sh.rustup.rs -sSf | sh -s -- -y && \
        /bin/bash -c "$(curl --proto '=https' --tlsv1.2 -sSf https://setup.atuin.sh)"
    
    # Clone the repository
    RUN git clone https://github.com/Dicklesworthstone/swiss_army_llama.git /swiss_army_llama
    WORKDIR /swiss_army_llama
    
    # Install Python dependencies using conda and pip
    RUN conda update -n base -c defaults conda && \
        conda install -c conda-forge python=3.12 pip setuptools wheel && \
        conda install -c conda-forge faiss-gpu && \
        CMAKE_ARGS="-DLLAMA_CUDA=on" /opt/conda/bin/pip install --no-cache-dir -r requirements.txt
    
    # Expose the port the app runs on and Redis default port
    EXPOSE 8089 6379
    
    # Create benchmark script
    RUN echo '#!/bin/bash' > /swiss_army_llama/benchmark.sh && \
        echo 'echo "CPU Benchmark:" > /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'sysbench --test=cpu --cpu-max-prime=5000 run >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'echo "\nMemory Benchmark:" >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'sysbench --test=memory --memory-block-size=1M --memory-total-size=2G run >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'echo "\nDisk Benchmark:" >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'dd if=/dev/zero of=/tmp/test1.img bs=1G count=1 oflag=dsync 2>> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'rm -f /tmp/test1.img' >> /swiss_army_llama/benchmark.sh && \
        echo 'echo "\nNetwork Benchmark (localhost):" >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'iperf3 -s & sleep 1' >> /swiss_army_llama/benchmark.sh && \
        echo 'iperf3 -c localhost -t 10 >> /swiss_army_llama/host_machine_benchmark_results.txt' >> /swiss_army_llama/benchmark.sh && \
        echo 'pkill iperf3' >> /swiss_army_llama/benchmark.sh && \
        chmod +x /swiss_army_llama/benchmark.sh
    
    # Start benchmark, Redis, and swiss_army_llama server when the container starts
    CMD /swiss_army_llama/benchmark.sh && service redis-server start && python /swiss_army_llama/swiss_army_llama.py
```


To utilize this GPU-enabled Swiss Army Llama instance, users can leverage the Vast.ai platform. Vast.ai is a decentralized marketplace that allows individuals to rent out their spare GPU capacity at affordable rates. Users can spin up a powerful machine with a 4090 GPU and ample RAM for as low as 34 cents per hour, significantly lower than traditional cloud providers like AWS, Azure, or Lambda. It’s sort of like the “UberX” of GPU cloud instance providers, with a disruptive and compelling business model. 

To get started with Vast.ai, users need to follow these steps:


1. Create an account on Vast.ai and purchase credits using a credit card (via Stripe) or by paying with cryptocurrency.
2. Generate an SSH key and add it to their Vast.ai account for secure access to provisioned instances.
3. Select the [Swiss Army Llama template](https://cloud.vast.ai/?ref_id=78066&template_id=3c711e5ddd050882f0eb6f4ce1b8cc28](https://cloud.vast.ai/?ref_id=78066&template_id=3c711e5ddd050882f0eb6f4ce1b8cc28) to speed up the deployment process. This template specifies the custom Docker image and the required storage of 60GB.
4. Provision the instance on Vast.ai, which will provide the IP address and port for SSH access.
5. Access the provisioned machine using SSH with the command provided by Vast.ai, similar to:

```bash
    ssh -i ~/vastai_privkey -p 20765 root@106.185.159.136
```

6. Once connected to the instance, navigate to the Swiss Army Llama directory and start the server:

```bash
    cd /swiss_army_llama/ 
    git update python swiss_army_llama.py
    cd /swiss_army_llama/ git update python swiss_army_llama.py
```

7. This will initiate the process of downloading the default Swiss Army Llama model files from Hugging Face.

To integrate the remote Swiss Army Llama instance with the local Supernode inference servers, users need to add the following optional fields to their `.env` file:

```ini
    USE_REMOTE_SWISS_ARMY_LLAMA_IF_AVAILABLE=1
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_SSH_KEY_PATH=/home/ubuntu/vastai_privkey
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_IP_ADDRESS=106.185.159.136
    REMOTE_SWISS_ARMY_LLAMA_INSTANCE_PORT=20765
    REMOTE_SWISS_ARMY_LLAMA_EXPOSED_PORT=8089
    REMOTE_SWISS_ARMY_LLAMA_MAPPED_PORT=8087
```

With these configurations in place, the Pastel Inference Layer server will automatically detect the availability of the remote Swiss Army Llama instance and establish an SSH tunnel to the Vast.ai machine. If the connection is successful, all Swiss Army Llama requests from the Supernode inference servers will be routed to the associated remote GPU-enabled instance.
This setup offers several benefits:

1. Improved Inference Performance: By utilizing a GPU-enabled instance, inference tasks can be accelerated significantly, reducing latency and improving overall performance.
2. Cost-Effectiveness: Vast.ai's decentralized marketplace allows users to rent GPU resources at affordable rates, making it more cost-effective compared to traditional cloud providers.
3. Decentralization and Censorship Resistance: By leveraging a decentralized platform like Vast.ai, the Pastel Inference Layer server maintains its decentralized nature and resists censorship attempts.
4. Resource Sharing: Multiple Supernode inference servers can share a single remote GPU instance, optimizing resource utilization and reducing costs.

By integrating a remote GPU-enabled instance of Swiss Army Llama hosted on Vast.ai, the Pastel Inference Layer server achieves a balance between inference performance, cost-effectiveness, and decentralization. This approach empowers users to harness the power of GPU acceleration while maintaining the core principles of the Pastel network.


