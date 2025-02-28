# Grok3 API

Grok3 is cool, smart, and useful, but there is no official API available. This is an **unofficial Python client** for interacting with the Grok 3 API. It leverages cookies from browser requests to authenticate and access the API endpoints. The API also provides OpenAI-compatible endpoints for easy integration with existing applications.

---

## Setup

Follow these steps to get started with the Grok3 API client.

### 1. Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/mem0ai/grok3-api.git
```

### 2. Install the Package
Navigate to the project directory, create a virtual environment, and install the package:

```
cd grok3-api
virtualenv pyenv
source pyenv/bin/activate
pip install .
```

### 3. Obtain Cookie Values

To use this client, you need to extract authentication cookies from a browser session:

* Open grok.com in your browser.
* Log in if you aren't already logged in.
* Open the browser's developer tools (e.g., F12 or right-click > Inspect).
* Go to the "Network" tab and filter for requests containing the new-chat endpoint (e.g., https://grok.com/rest/app-chat/conversations/new).
* Right-click the request, select "Copy as cURL," and paste it somewhere.
From the curl command, extract the following cookie values from the -H 'cookie: ...' header:
    * x-anonuserid
    * x-challenge
    * x-signature
    * sso
    * sso-rw

Example cookie string from a curl command:
```
-H 'cookie: x-anonuserid=ffdd32e1; x-challenge=TkC4D...; x-signature=fJ0U00...; sso=eyJhbGci...; sso-rw=eyJhbGci...'
```

### 4. Use the Client

#### 4.1 Direct Client Usage

Pass the extracted cookie values to the GrokClient and send a message:

```python
from grok_client import GrokClient

# Your cookie values
cookies = {
    "x-anonuserid": "ffdd32e1",
    "x-challenge": "TkC4D..",
    "x-signature": "fJ0...",
    "sso": "ey...",
    "sso-rw": "ey..."
}

# Initialize the client
client = GrokClient(cookies)

# Send a message and get response
response = client.send_message("write a poem")
print(response)
```

#### 4.2 OpenAI-Compatible API Server

The package includes an OpenAI-compatible API server that allows you to use Grok with any OpenAI-compatible client library or application.

##### Start the Server

1. Create a `.env` file in the project directory using the provided `.env.example` template:
```bash
cp grok_client/.env.example .env
```

2. Update the `.env` file with your Grok cookie values:
```env
GROK_SSO=your_sso_cookie
GROK_SSO_RW=your_sso_rw_cookie
# Optional configurations
API_HOST=127.0.0.1
API_PORT=8000
MODEL_NAME=grok-3
```

3. Start the API server:
```bash
uvicorn grok_client.server:app --reload --host 0.0.0.0 --port 8000
```

##### Use with OpenAI Python Client

```python
from openai import OpenAI

# Initialize client pointing to local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy-key"  # Any non-empty string will work
)

# Create a chat completion
response = client.chat.completions.create(
    model="grok-3",  # Model name can be configured in .env
    messages=[
        {"role": "user", "content": "Hello, how can you help me?"}
    ]
)

print(response.choices[0].message.content)
```

##### Interactive Chat Script

The package includes an interactive chat script that uses the OpenAI-compatible endpoint:

```bash
python grok_client/interactive.py
```

This provides a command-line interface for chatting with Grok using the OpenAI-compatible API.

### 5. Optional: Add Memory with Mem0

If you want Grok to remember conversations, you can integrate it with Mem0. Mem0 provides a memory layer for AI applications.

#### 5.1 Install Mem0

```bash
pip install mem0ai
```

#### 5.2 Add & Retrieve Memory

```
from mem0 import Memory

memory = Memory()

# for user alice
result = memory.add("I like to take long walks on weekends.", user_id="alice")

# Retrieve memories
related_memories = memory.search(, user_id="alice")
print(related_memories)
```


# Disclaimer
This is an unofficial API client for Grok3 and is not affiliated with or endorsed by xAI, the creators of Grok. It relies on reverse-engineering browser requests and may break if the underlying API changes. Use at your own risk. The authors are not responsible for any consequences arising from its use, including but not limited to account suspension, data loss, or legal issues. Ensure you comply with Grok's terms of service and applicable laws when using this client