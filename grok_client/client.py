import requests
import json
import time
import logging
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class GrokClient:
    def __init__(self, cookies):
        """
        Initialize the Grok client with cookie values

        Args:
            cookies (dict): Dictionary containing cookie values
                - sso
                - sso-rw
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
            self.cookies = cookies
            
        logger.debug(f"Using cookies: {self.cookies}")
        
        self.headers = {
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

    def _prepare_payload(self, message):
        """Prepare the default payload with the user's message"""
        payload = {
            "temporary": False,
            "modelName": "grok-3",
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

    def _clean_json_response(self, response):
        """Clean up JSON response by removing markdown and code blocks"""
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        try:
            # Try to parse as JSON
            json_data = json.loads(response)
            
            # If the response has a nested response or function_call, extract it
            if isinstance(json_data, dict):
                if "response" in json_data:
                    json_data = json_data["response"]
                elif "function_call" in json_data:
                    json_data = json_data["function_call"]["arguments"]
                    if isinstance(json_data, str):
                        json_data = json.loads(json_data)
            
            return json.dumps(json_data, indent=2)
        except json.JSONDecodeError:
            return response

    def send_message(self, message):
        """
        Send a message to Grok and collect the streaming response

        Args:
            message (str): The user's input message

        Returns:
            str: The complete response from Grok
        """
        try:
            logger.debug(f"Sending message to Grok: {message}")
            payload = self._prepare_payload(message)
            
            logger.debug(f"Making POST request to {self.base_url}")
            logger.debug(f"Using cookies: {self.cookies}")
            
            session = requests.Session()
            for cookie_name, cookie_value in self.cookies.items():
                session.cookies.set(cookie_name, cookie_value)
            
            response = session.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                stream=True
            )
            
            logger.debug(f"Response status code: {response.status_code}")
            response.raise_for_status()  # Raise an exception for bad status codes
            
            full_response = ""
            last_response = None

            logger.debug("Processing response stream...")
            for line in response.iter_lines():
                if line:
                    try:
                        decoded_line = line.decode('utf-8')
                        logger.debug(f"Received line: {decoded_line}")
                        
                        json_data = json.loads(decoded_line)
                        logger.debug(f"Parsed JSON: {json_data}")
                        
                        # Check for error in response
                        if "error" in json_data:
                            error_msg = json_data["error"]
                            logger.error(f"Error in response: {error_msg}")
                            raise Exception(error_msg)
                        
                        result = json_data.get("result", {})
                        response_data = result.get("response", {})
                        logger.debug(f"Response data: {response_data}")

                        # Check for complete response
                        if "modelResponse" in response_data:
                            complete_response = response_data["modelResponse"].get("message", "")
                            if complete_response:
                                logger.debug(f"Got complete response: {complete_response}")
                                return self._clean_json_response(complete_response)
                            
                        # Collect streaming tokens
                        token = response_data.get("token", "")
                        if token:
                            full_response += token
                            last_response = full_response  # Keep track of last valid response
                            logger.debug(f"Current response: {full_response}")
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode JSON: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing response line: {e}")
                        if str(e):  # If we have an error message
                            raise Exception(f"Error in response: {str(e)}")
                        continue

            # Return the last valid response if we have one
            if last_response:
                logger.debug(f"Returning last valid response: {last_response.strip()}")
                return self._clean_json_response(last_response.strip())
            
            # If we got here without a response, raise an exception
            logger.error("No valid response received from Grok API")
            raise Exception("No valid response received from Grok API")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise Exception(f"Request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to process response: {e}")
            raise Exception(f"Failed to process response: {str(e)}")

        logger.warning("Returning empty response as fallback")
        return ""  # Fallback empty response