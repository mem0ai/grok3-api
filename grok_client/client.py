import requests
import json
import time
"""
Grok Direct Client Module
=========================

This module provides the `GrokClient` class, a low-level client for direct interaction
with the Grok API. It handles request preparation, sending messages, and processing
responses, including error handling specific to Grok API interactions.

It is intended for use cases where direct control over the Grok API is needed,
as opposed to using an OpenAI-compatible interface.
"""
import logging
import re
import os
import requests # Import requests for type hinting session and response
from typing import Dict, List, Any, Union, Optional # Import necessary types

from .errors import GrokAPIError, AuthenticationError, NetworkError, ConfigurationError, GrokClientError

# Set up logging
_DEFAULT_LOG_LEVEL: str = "INFO"
_ALLOWED_LOG_LEVELS: List[str] = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

def get_log_level_from_env() -> str:
    """
    Retrieves and validates the log level from the GROK_LOG_LEVEL environment variable.

    If the environment variable is not set or contains an invalid value,
    a warning is logged, and the default log level ("INFO") is returned.

    Returns:
        str: The validated log level string (e.g., "DEBUG", "INFO").
    """
    env_log_level: str = os.environ.get("GROK_LOG_LEVEL", _DEFAULT_LOG_LEVEL).upper()
    if env_log_level not in _ALLOWED_LOG_LEVELS:
        # Log a warning using a temporary basic config if the level is invalid, then default
        # This initial basicConfig is for this specific warning message only.
        # The main basicConfig later will use the determined (or default) LOG_LEVEL.
        logging.basicConfig(level=logging.WARNING) 
        logging.warning(f"Invalid GROK_LOG_LEVEL '{env_log_level}'. Defaulting to '{_DEFAULT_LOG_LEVEL}'.")
        return _DEFAULT_LOG_LEVEL
    return env_log_level

LOG_LEVEL: str = get_log_level_from_env()
# The main basicConfig for the logger, using the determined LOG_LEVEL.
# Note: If the logger was already configured by the above warning, this might reconfigure it
# or be ignored depending on Python's logging internals. Ideally, only configure once.
# To ensure single configuration, we can check if root logger has handlers.
if not logging.root.handlers:
    logging.basicConfig(level=LOG_LEVEL)
else: # If already configured (e.g. by the warning message), just set level
    logging.getLogger().setLevel(LOG_LEVEL)

logger = logging.getLogger(__name__)

class GrokClient:
    """
    A client for interacting directly with the Grok API.

    This client handles the necessary authentication (via cookies) and request
    formatting to send messages to Grok and retrieve responses. It is designed
    for lower-level access to the Grok service.

    Attributes:
        base_url (str): The base URL for the Grok API chat endpoint.
        cookies (Dict[str, str]): Cookies used for authentication, must include 'sso' and 'sso-rw'.
        headers (Dict[str, str]): Standard headers sent with each request.
    """
    base_url: str
    cookies: Dict[str, str]
    headers: Dict[str, str]

    def __init__(self, cookies: Dict[str, str]) -> None:
        """
        Initializes the GrokClient.

        Args:
            cookies (Dict[str, str]): A dictionary containing authentication cookies.
                Must include 'sso' and 'sso-rw' keys with their respective token values.

        Raises:
            AuthenticationError: If 'sso' or 'sso-rw' cookies are missing or empty.
        """
        self.base_url = "https://grok.com/rest/app-chat/conversations/new"
        
        # Convert cookie string to dict if needed
        if isinstance(cookies.get('Cookie'), str):
            cookie_dict = {}
            for cookie in cookies.get('Cookie', '').split(';'):
                if cookie.strip():
                    name, value = cookie.strip().split('=', 1)
                    cookie_dict[name.strip()] = value.strip()
            self.cookies = cookie_dict
        else:
            self.cookies = cookies # type: ignore # Assuming cookies can be other types initially
        
        # Ensure self.cookies is Dict[str, str] after processing
        if not isinstance(self.cookies, dict) or \
           not all(isinstance(k, str) and isinstance(v, str) for k, v in self.cookies.items()):
             # This case should ideally be handled by stricter input validation or type checking earlier
             # For now, if it's not a Dict[str, str] after processing, it's an issue.
             # We'll assume the conversion logic above makes it Dict[str, str] or raises.
             # If not, the following checks might fail or behave unexpectedly.
             # To be robust, one might add: self.cookies = {} if not isinstance(self.cookies, dict) else self.cookies
             pass


        if not self.cookies.get('sso') or not self.cookies.get('sso-rw'):
            raise AuthenticationError("Missing required SSO cookies (sso, sso-rw) for GrokClient initialization.")
            
        logger.debug(f"Using cookies: {self.cookies}")
        
        self.headers = { # type: Dict[str, str]
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        }
        logger.debug(f"Initialized GrokClient with headers: {self.headers}")

    def _prepare_payload(self, message: str) -> Dict[str, Any]:
        """
        Prepares the JSON payload for sending a message to the Grok API.

        Args:
            message (str): The user's input message.

        Returns:
            Dict[str, Any]: A dictionary representing the JSON payload.
        """
        payload: Dict[str, Any] = {
            "temporary": False,
            "modelName": "grok-3", # Or make this configurable via __init__
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": False,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": False,
            "imageGenerationCount": 0,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False
        }
        logger.debug(f"Prepared payload: {payload}")
        return payload

    def _clean_json_response(self, response: str) -> str:
        """
        Cleans up a JSON-like string response from Grok.
        
        This method attempts to remove common markdown code block delimiters (```json ... ```)
        and then tries to parse and re-serialize the JSON to ensure it's well-formed.
        If the input string contains nested JSON strings (e.g., in 'response' or 
        'function_call.arguments'), it attempts to extract and format those.

        Args:
            response (str): The raw string response which may contain JSON.

        Returns:
            str: A cleaned-up JSON string, or the original string if it's not valid JSON
                 or if extraction logic doesn't apply.
        """
        # Remove markdown code blocks
        cleaned_response: str = re.sub(r'```json\s*', '', response)
        cleaned_response = re.sub(r'```\s*$', '', cleaned_response)
        
        try:
            # Try to parse as JSON
            json_data: Any = json.loads(cleaned_response)
            
            # If the response has a nested response or function_call, extract it
            if isinstance(json_data, dict):
                if "response" in json_data and isinstance(json_data["response"], (dict, list, str)): # Check type of inner response
                    json_data = json_data["response"]
                elif "function_call" in json_data and \
                     isinstance(json_data.get("function_call"), dict) and \
                     "arguments" in json_data["function_call"] and \
                     isinstance(json_data["function_call"]["arguments"], str):
                    # Attempt to parse arguments if they are a string-encoded JSON
                    try:
                        json_data = json.loads(json_data["function_call"]["arguments"])
                    except json.JSONDecodeError:
                        # If arguments are not valid JSON, keep them as string or handle as error
                        # For now, we assume it should be parsable if it's a function call argument.
                        # If not, it might remain a string within the function_call structure.
                        # This part might need more specific error handling or schema validation.
                        pass # Keep json_data as the function_call dict if arguments are not JSON string.
            
            return json.dumps(json_data, indent=2)
        except json.JSONDecodeError:
            # If it's not valid JSON after cleaning, return the cleaned string as is.
            return cleaned_response

    def send_message(self, message: str, stream_callback: Optional[Any] = None) -> str:
        """
        Sends a message to the Grok API and processes the response.

        This method can handle both streaming and non-streaming responses.
        If `stream_callback` is provided, it's expected that this method (or the underlying
        request logic) will invoke the callback with chunks of data from the stream.
        The current implementation primarily collects a full response, but the
        `stream_callback` argument is kept for compatibility with potential future
        true streaming implementations or for how the server part uses it.

        Args:
            message (str): The user's input message.
            stream_callback (Optional[Any]): An optional callback function to handle
                streaming data. (Note: Current client implementation collects full response;
                true client-side streaming via this callback is not fully implemented here).

        Returns:
            str: The complete, cleaned response text from Grok. If the response is streamed
                 and `stream_callback` is used, the return value might be the final
                 accumulated response or an empty string if all data is handled by callback.
                 Currently, it returns the `xai_generated_text` from the last valid packet.
        
        Raises:
            AuthenticationError: If the API returns a 401 or 403 status code.
            GrokAPIError: For other 4xx/5xx API errors or if the API response
                          indicates an error (e.g., within the JSON payload).
            NetworkError: For network-level issues like connection errors or timeouts.
        """
        try:
            logger.debug(f"Sending message to Grok: {message}")
            payload: Dict[str, Any] = self._prepare_payload(message)
            
            logger.debug(f"Making POST request to {self.base_url}")
            logger.debug(f"Using cookies: {self.cookies}")
            
            session: requests.Session = requests.Session()
            for cookie_name, cookie_value in self.cookies.items():
                session.cookies.set(cookie_name, cookie_value)
            
            response: requests.Response = session.post(
                self.base_url, # type: ignore # self.base_url is str
                headers=self.headers, # type: ignore # self.headers is Dict[str, str]
                json=payload,
                stream=True # Always stream to inspect line by line
            )
            
            logger.debug(f"Response status code: {response.status_code}")

            if response.status_code == 401 or response.status_code == 403:
                raise AuthenticationError(f"Authentication failed with Grok API: {response.status_code} - {response.text}")
            if response.status_code >= 400:
                raise GrokAPIError(f"Grok API request failed: {response.status_code} - {response.text}")
            
            full_response_accumulator: str = ""
            last_processed_response_text: Optional[str] = None # To store the text from the last meaningful packet

            logger.debug("Processing response stream...")
            for line in response.iter_lines(): # type: bytes
                if line:
                    decoded_line: str = ""
                    try:
                        decoded_line = line.decode('utf-8')
                        logger.debug(f"Received line: {decoded_line}")
                        
                        # Assuming each line is a separate JSON object, as per typical SSE-like streams
                        json_data: Any = json.loads(decoded_line)
                        logger.debug(f"Parsed JSON from line: {json_data}")
                        
                        # Error checking within the JSON payload itself
                        if isinstance(json_data, dict) and "error" in json_data:
                            error_msg: str = str(json_data.get("error", "Unknown API error in JSON payload"))
                            logger.error(f"API Error in response payload: {error_msg}")
                            raise GrokAPIError(f"Grok API returned an error in payload: {error_msg}")
                        
                        # Data extraction logic (this part is highly dependent on Grok's actual streaming format)
                        # The existing code seems to expect a structure like:
                        # {"result": {"response": {"modelResponse": {"message": "..."}}}} or
                        # {"result": {"response": {"token": "..."}}}
                        # And also a top-level "xai_generated_text" in some cases (often in final packets).
                        
                        current_text_piece: Optional[str] = None
                        if isinstance(json_data, dict):
                            if "xai_generated_text" in json_data: # Often in final non-streaming style packet
                                current_text_piece = json_data["xai_generated_text"]
                            
                            result_data = json_data.get("result")
                            if isinstance(result_data, dict):
                                response_data = result_data.get("response")
                                if isinstance(response_data, dict):
                                    if "modelResponse" in response_data and \
                                       isinstance(response_data["modelResponse"], dict) and \
                                       "message" in response_data["modelResponse"]:
                                        # This seems like a full message override
                                        current_text_piece = response_data["modelResponse"]["message"]
                                    elif "token" in response_data: # Streaming token
                                        token_text = response_data.get("token")
                                        if isinstance(token_text, str):
                                            full_response_accumulator += token_text
                                            # For streaming, current_text_piece might be just the token
                                            # or we rely on full_response_accumulator.
                                            # The old logic used `last_response` for the whole json_data.
                                            # Let's assume for now the goal is to get any text.
                                            if not current_text_piece: # Prioritize xai_generated_text or full message
                                                current_text_piece = token_text


                        if stream_callback and current_text_piece is not None:
                            # If there's a callback, send the current piece of data.
                            # The callback might receive individual tokens or larger chunks.
                            # The structure of `json_data` or `current_text_piece` should align with callback needs.
                            # The original server.py implies callback gets the whole json_data dict.
                            stream_callback(json_data) # Pass the whole JSON object to callback
                            
                        if current_text_piece is not None:
                             last_processed_response_text = current_text_piece # Keep track of text from last meaningful packet

                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode JSON from line: {decoded_line if decoded_line else line.decode('utf-8', errors='ignore')}")
                        # Decide if this is fatal or skippable. For now, skip.
                        continue 
                    except GrokAPIError: # Re-raise API errors found in payload
                        raise
                    except Exception as e: # Catch other errors during line processing
                        logger.error(f"Error processing response line: {decoded_line} - Error: {e}", exc_info=True)
                        raise GrokAPIError(f"Error processing Grok API response line: {str(e)}")

            # After iterating through all lines:
            # Determine what to return. The old logic returned based on 'last_response'.
            # If streaming tokens were accumulated:
            if full_response_accumulator:
                logger.debug(f"Returning accumulated streaming response: {full_response_accumulator.strip()}")
                return self._clean_json_response(full_response_accumulator.strip())
            
            # If individual text pieces were processed (e.g., from xai_generated_text or modelResponse.message):
            if last_processed_response_text is not None:
                 logger.debug(f"Returning last processed text: {last_processed_response_text.strip()}")
                 return self._clean_json_response(last_processed_response_text.strip())

            # If we reach here, no meaningful data was extracted or accumulated.
            logger.error("No valid/extractable content received from Grok API stream.")
            raise GrokAPIError("No valid/extractable content message received from Grok API stream.")
            
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error while contacting Grok API: {str(e)}", exc_info=True)
            raise NetworkError(f"Connection error while contacting Grok API: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout while contacting Grok API: {str(e)}", exc_info=True)
            raise NetworkError(f"Timeout while contacting Grok API: {str(e)}")
        except requests.exceptions.RequestException as e: # Other request-related errors
            logger.error(f"Network request to Grok API failed: {str(e)}", exc_info=True)
            raise NetworkError(f"Network request to Grok API failed: {str(e)}")
        except GrokClientError: # Re-raise already handled custom Grok client errors
            raise
        except Exception as e: # Catch-all for unexpected errors
            logger.error(f"An unexpected error occurred while processing Grok response: {str(e)}", exc_info=True)
            raise GrokAPIError(f"An unexpected error occurred while processing Grok response: {str(e)}")