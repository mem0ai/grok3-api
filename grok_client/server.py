from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from .client import GrokClient
from .errors import GrokAPIError, AuthenticationError, NetworkError, ConfigurationError, GrokClientError
import json
import time
import logging
import os
import uuid

# Set up logging
_DEFAULT_LOG_LEVEL = "INFO"
_ALLOWED_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

def get_log_level_from_env():
    env_log_level = os.environ.get("GROK_LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()
    if env_log_level not in _ALLOWED_LOG_LEVELS:
        logging.basicConfig(level=logging.WARNING) # Temp for this message
        logging.warning(f"Invalid GROK_LOG_LEVEL '{env_log_level}'. Defaulting to '{_DEFAULT_LOG_LEVEL}'.")
        logging.basicConfig(level=_DEFAULT_LOG_LEVEL) # Reset
        return _DEFAULT_LOG_LEVEL
    return env_log_level

LOG_LEVEL = get_log_level_from_env()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(name)s - [%(request_id)s] - %(message)s', defaults={'request_id': 'N/A'})
logger = logging.getLogger(__name__)

app = FastAPI()

async def add_request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    request.state.request_id = request_id
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

app.middleware("http")(add_request_id_middleware)

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
        elif request_data.response_format and request_data.response_format.get("type") == "json_object":
            system_content = "You are a helpful assistant that always responds in valid JSON format."
        
        return system_content

    def stream_chat(self, request_data: ChatCompletionRequest, request_id: str):
        try:
            # Prepare the conversation context
            system_msg = self._prepare_system_message(request_data)
            conversation = f"system: {system_msg}\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in request_data.messages])
            
            logger.debug(f"Sending conversation to Grok: {conversation}", extra={'request_id': request_id})
            
            # Get streaming response from Grok
            response_stream = self.client.send_message(conversation) # send_message itself logs with its own context
            logger.debug(f"Got response stream from Grok: {response_stream}", extra={'request_id': request_id})
            
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
                model="grok-3", # Or use request_data.model
                choices=[{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            )
            yield f"data: {json.dumps(final_chunk.dict())}\n\n"
            # yield "data: [DONE]\n\n" # This is handled by finally
        except AuthenticationError as e:
            logger.error(f"AuthenticationError during streaming: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'AuthenticationError', 'code': 401}})}\n\n"
        except ConfigurationError as e:
            logger.error(f"ConfigurationError during streaming: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'ConfigurationError', 'code': 400}})}\n\n"
        except NetworkError as e:
            logger.error(f"NetworkError during streaming: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'NetworkError', 'code': 502}})}\n\n"
        except GrokAPIError as e:
            logger.error(f"GrokAPIError during streaming: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'GrokAPIError', 'code': 502}})}\n\n"
        except GrokClientError as e:
            logger.error(f"GrokClientError during streaming: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': str(e), 'type': 'GrokClientError', 'code': 500}})}\n\n"
        except Exception as e:
            logger.error(f"Unexpected error in stream_chat: {str(e)}", exc_info=True, extra={'request_id': request_id})
            yield f"data: {json.dumps({'error': {'message': f'An unexpected error occurred during streaming: {str(e)}', 'type': 'ServerError', 'code': 500}})}\n\n"
        finally:
            logger.info("Finished streaming chat attempt.", extra={'request_id': request_id})
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
        request_id = raw_request.state.request_id # Get request_id from middleware
        # Get request body
        body = await raw_request.json()
        logger.debug(f"Received request body: {body}", extra={'request_id': request_id})
        
        # Parse request into ChatCompletionRequest
        request_data = ChatCompletionRequest(**body) # Renamed to request_data
        
        # Get cookies from request headers
        headers = dict(raw_request.headers)
        logger.debug(f"Received headers: {headers}", extra={'request_id': request_id})
        
        cookies = {'Cookie': headers.get('cookie', '')} if headers.get('cookie') else {}
        logger.debug(f"Extracted cookies: {cookies}", extra={'request_id': request_id})
        
        if not cookies:
            # Log before raising HTTPException, as HTTPException might not be logged with request_id by default handler
            logger.warning("No authentication cookies provided.", extra={'request_id': request_id})
            raise HTTPException(status_code=401, detail="No authentication cookies provided")
        
        # Initialize Grok API with cookies
        grok = GrokAPI(cookies) # GrokClient init logs internally, request_id not directly available there
        
        if request_data.stream:
            return StreamingResponse(
                grok.stream_chat(request_data, request_id), # Pass request_id
                media_type="text/event-stream"
            )
        
        # For non-streaming response
        system_msg = grok._prepare_system_message(request_data)
        conversation = f"system: {system_msg}\n" + "\n".join([f"{msg.role}: {msg.content}" for msg in request_data.messages])
        logger.debug(f"Sending conversation to Grok: {conversation}", extra={'request_id': request_id})
        
        response = grok.client.send_message(conversation) # send_message logs internally
        logger.debug(f"Received response from Grok: {response}", extra={'request_id': request_id})
        
        if not response:
            logger.error("Empty response from Grok API", extra={'request_id': request_id})
            raise HTTPException(status_code=500, detail="Empty response from Grok API")
        
        # Handle function calling
        if request_data.functions and request_data.function_call:
            try:
                # Try to parse the response as JSON
                parsed_response = json.loads(response)
                
                # Get the function name from the request
                function_name = request_data.function_call.get("name", request_data.functions[0].name) if isinstance(request_data.function_call, dict) else request_data.functions[0].name
                
                message = ChatMessage(
                    role="assistant",
                    content="",
                    function_call={
                        "name": function_name,
                        "arguments": json.dumps(parsed_response)
                    }
                )
            except json.JSONDecodeError:
                logger.warning(f"Function call response is not valid JSON. Original response: {response}", extra={'request_id': request_id})
                # If response is not valid JSON, wrap it in a basic structure
                function_name = request_data.function_call.get("name", request_data.functions[0].name) if isinstance(request_data.function_call, dict) else request_data.functions[0].name
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
            if request_data.response_format and request_data.response_format.get("type") == "json_object":
                try:
                    # Ensure the response is valid JSON
                    json.loads(response) # Validate
                    message = ChatMessage(
                        role="assistant",
                        content=response
                    )
                except json.JSONDecodeError:
                    logger.warning(f"JSON format requested, but response is not valid JSON. Original response: {response}", extra={'request_id': request_id})
                    # If not valid JSON, wrap it in a JSON structure
                    message = ChatMessage(
                        role="assistant",
                        content=json.dumps({"response": response}) # Wrap to make it JSON
                    )
            else:
                message = ChatMessage(
                    role="assistant",
                    content=response
                )
        
        # Create response object
        chat_response = ChatCompletionResponse(
            id=f"chatcmpl-{str(int(time.time()))}", # Consider using request_id or part of it for traceability
            created=int(time.time()),
            model=request_data.model,
            choices=[ChatCompletionChoice(
                message=message,
                finish_reason="stop"
            )]
        )
        
        logger.debug(f"Sending response: {chat_response.dict()}", extra={'request_id': request_id})
        return chat_response
    
    except AuthenticationError as e:
        logger.error(f"AuthenticationError in chat completion: {str(e)}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        return JSONResponse(status_code=401, content={"error": {"message": str(e), "type": "AuthenticationError"}})
    except ConfigurationError as e: 
        logger.error(f"ConfigurationError in chat completion: {str(e)}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        return JSONResponse(status_code=400, content={"error": {"message": str(e), "type": "ConfigurationError"}})
    except NetworkError as e: 
        logger.error(f"NetworkError in chat completion: {str(e)}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "NetworkError"}})
    except GrokAPIError as e: 
        logger.error(f"GrokAPIError in chat completion: {str(e)}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        return JSONResponse(status_code=502, content={"error": {"message": str(e), "type": "GrokAPIError"}})
    except GrokClientError as e: 
        logger.error(f"GrokClientError in chat completion: {str(e)}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        return JSONResponse(status_code=500, content={"error": {"message": str(e), "type": "GrokClientError"}})
    except HTTPException as http_exc:
        # If we want to log HTTPExceptions with request_id, we need to catch, log, and re-raise or return response
        logger.error(f"HTTPException in chat completion: Status {http_exc.status_code}, Detail {http_exc.detail}", exc_info=True, extra={'request_id': raw_request.state.request_id if hasattr(raw_request.state, 'request_id') else 'N/A'})
        raise http_exc # Re-raise to let FastAPI handle the response
    except Exception as e:
        # Ensure request_id is available for logging if possible
        req_id = 'N/A'
        if hasattr(raw_request, 'state') and hasattr(raw_request.state, 'request_id'):
            req_id = raw_request.state.request_id
        logger.error(f"Unexpected error in create_chat_completion: {str(e)}", exc_info=True, extra={'request_id': req_id}) 
        return JSONResponse(
            status_code=500,
            content={"error": {"message": f"An unexpected server error occurred: {str(e)}", "type": "ServerError"}}
        )