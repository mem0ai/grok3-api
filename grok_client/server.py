from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from .client import GrokClient
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    role: str
    content: str
    function_call: Optional[Dict[str, Any]] = None

class FunctionCall(BaseModel):
    name: str
    arguments: str

class Function(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    functions: Optional[List[Function]] = None
    function_call: Optional[Union[str, Dict[str, str]]] = None
    response_format: Optional[Dict[str, str]] = None

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]

class GrokAPI:
    def __init__(self, cookies: Dict[str, str]):
        self.client = GrokClient(cookies)

    def _prepare_system_message(self, request: ChatCompletionRequest) -> str:
        # Default to simple responses unless specifically asked for structured output
        system_content = "You are a helpful assistant. Provide direct, simple answers to questions."
        
        # Add function calling instructions if needed
        if request.functions:
            system_content = "You are a helpful assistant that provides structured data."
            system_content += f" Available functions: {[f.name for f in request.functions]}"
            system_content += f" Function schemas: {json.dumps([f.dict() for f in request.functions])}"
        
        # Add JSON format instructions if needed
        elif request.response_format and request.response_format.get("type") == "json_object":
            system_content = "You are a helpful assistant that always responds in valid JSON format."
        
        return system_content

    def stream_chat(self, request: ChatCompletionRequest):
        try:
            # Prepare the conversation context
            system_msg = self._prepare_system_message(request)
            conversation = f"system: {system_msg}\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
            
            logger.debug(f"Sending conversation to Grok: {conversation}")
            
            # Get streaming response from Grok
            response_stream = self.client.send_message(conversation)
            logger.debug(f"Got response stream from Grok: {response_stream}")
            
            # Stream the response in OpenAI format
            for token in response_stream.split():
                chunk = ChatCompletionChunk(
                    id="chatcmpl-" + str(int(time.time())),
                    created=int(time.time()),
                    model="grok-3",
                    choices=[{
                        "index": 0,
                        "delta": {"content": token + " "},
                        "finish_reason": None
                    }]
                )
                yield f"data: {json.dumps(chunk.dict())}\n\n"
            
            # Send the final chunk
            final_chunk = ChatCompletionChunk(
                id="chatcmpl-final",
                created=int(time.time()),
                model="grok-3",
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in stream_chat: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

@app.get("/v1/models")
async def list_models():
    return {
        "data": [
            {
                "id": "grok-3",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "xai",
                "permission": [],
                "root": "grok-3",
                "parent": None
            }
        ]
    }

@app.post("/v1/chat/completions")
async def create_chat_completion(raw_request: Request):
    try:
        # Get request body
        body = await raw_request.json()
        logger.debug(f"Received request body: {body}")
        
        # Parse request into ChatCompletionRequest
        request = ChatCompletionRequest(**body)
        
        # Get cookies from request headers
        headers = dict(raw_request.headers)
        logger.debug(f"Received headers: {headers}")
        
        cookies = {'Cookie': headers.get('cookie', '')} if headers.get('cookie') else {}
        logger.debug(f"Extracted cookies: {cookies}")
        
        if not cookies:
            raise HTTPException(status_code=401, detail="No authentication cookies provided")
        
        # Initialize Grok API with cookies
        grok = GrokAPI(cookies)
        
        if request.stream:
            return StreamingResponse(
                grok.stream_chat(request),
                media_type="text/event-stream"
            )
        
        # For non-streaming response
        system_msg = grok._prepare_system_message(request)
        conversation = f"system: {system_msg}\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in request.messages])
        logger.debug(f"Sending conversation to Grok: {conversation}")
        
        response = grok.client.send_message(conversation)
        logger.debug(f"Received response from Grok: {response}")
        
        if not response:
            logger.error("Empty response from Grok API")
            raise HTTPException(status_code=500, detail="Empty response from Grok API")
        
        # Handle function calling
        if request.functions and request.function_call:
            try:
                # Try to parse the response as JSON
                parsed_response = json.loads(response)
                
                # Get the function name from the request
                function_name = request.function_call.get("name", request.functions[0].name) if isinstance(request.function_call, dict) else request.functions[0].name
                
                message = ChatMessage(
                    role="assistant",
                    content="",
                    function_call={
                        "name": function_name,
                        "arguments": json.dumps(parsed_response)
                    }
                )
            except json.JSONDecodeError:
                # If response is not valid JSON, wrap it in a basic structure
                function_name = request.function_call.get("name", request.functions[0].name) if isinstance(request.function_call, dict) else request.functions[0].name
                message = ChatMessage(
                    role="assistant",
                    content="",
                    function_call={
                        "name": function_name,
                        "arguments": json.dumps({"result": response})
                    }
                )
        else:
            # Regular response or JSON format
            if request.response_format and request.response_format.get("type") == "json_object":
                try:
                    # Ensure the response is valid JSON
                    json.loads(response)
                    message = ChatMessage(
                        role="assistant",
                        content=response
                    )
                except json.JSONDecodeError:
                    # If not valid JSON, wrap it in a JSON structure
                    message = ChatMessage(
                        role="assistant",
                        content=json.dumps({"response": response})
                    )
            else:
                message = ChatMessage(
                    role="assistant",
                    content=response
                )
        
        # Create response object
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{str(int(time.time()))}",
            created=int(time.time()),
            model=request.model,
            choices=[ChatCompletionChoice(
                message=message,
                finish_reason="stop"
            )]
        )
        
        logger.debug(f"Sending response: {chat_response.dict()}")
        return chat_response
    
    except Exception as e:
        logger.error(f"Error in create_chat_completion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": "Failed to process request"}
        )