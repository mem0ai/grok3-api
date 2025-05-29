import os
import openai
import logging

logger = logging.getLogger(__name__)

class GrokOpenAIClient:
    """
    A client for interacting with a Grok API that mimics the OpenAI API structure.
    """
    def __init__(self, api_host=None, api_port=None, model_name=None, sso_token=None, sso_rw_token=None, load_from_env=True):
        """
        Initializes the GrokOpenAIClient.

        Args:
            api_host (str, optional): The API host. Defaults to '127.0.0.1'.
            api_port (str, optional): The API port. Defaults to '8000'.
            model_name (str, optional): The model name. Defaults to 'grok-3'.
            sso_token (str, optional): The SSO token.
            sso_rw_token (str, optional): The SSO RW token.
            load_from_env (bool, optional): Whether to load parameters from environment variables. Defaults to True.

        Raises:
            ValueError: If essential cookie information (sso_token, sso_rw_token) is missing.
        """
        if load_from_env:
            api_host = api_host or os.getenv('GROK_API_HOST', '127.0.0.1')
            api_port = api_port or os.getenv('GROK_API_PORT', '8000')
            model_name = model_name or os.getenv('GROK_MODEL_NAME', 'grok-3')
            sso_token = sso_token or os.getenv('GROK_SSO_TOKEN')
            sso_rw_token = sso_rw_token or os.getenv('GROK_SSO_RW_TOKEN')

        if not sso_token or not sso_rw_token:
            raise ValueError("SSO token and SSO RW token are required.")

        self.model_name = model_name
        base_url = f"http://{api_host}:{api_port}/v1"
        
        self.client = openai.OpenAI(
            base_url=base_url,
            api_key="dummy_key",  # OpenAI client requires an API key, but Grok uses SSO tokens
            default_headers={
                "Cookie": f"sso={sso_token}; sso-rw={sso_rw_token}"
            }
        )
        logger.info(f"GrokOpenAIClient initialized with model: {self.model_name}, API: {base_url}")

    def chat_completion(self, messages, stream=False, temperature=0.7, response_format=None, max_tokens=None, **kwargs):
        """
        Creates a chat completion using the configured Grok model.

        Args:
            messages (list): A list of message objects, similar to OpenAI's API.
            stream (bool, optional): Whether to stream the response. Defaults to False.
            temperature (float, optional): Sampling temperature. Defaults to 0.7.
            response_format (dict, optional): The response format. Defaults to None.
            max_tokens (int, optional): The maximum number of tokens to generate. Defaults to None.
            **kwargs: Additional keyword arguments to pass to the OpenAI client.

        Returns:
            The response from the OpenAI client, which could be a streaming object or a completion object.
        """
        params = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
        }
        if response_format is not None:
            params["response_format"] = response_format
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        
        params.update(kwargs)  # Add any other common parameters

        logger.debug(f"Sending chat completion request with params: {params}")
        return self.client.chat.completions.create(**params)

    def process_streaming_response(self, stream_response):
        """
        Processes a streaming response, printing each chunk's content and returning the full response.

        Args:
            stream_response: The streaming response object from chat_completion with stream=True.

        Returns:
            str: The accumulated full response string.
        """
        full_response = []
        for chunk in stream_response:
            if hasattr(chunk, 'choices') and chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content_piece = delta.content
                    print(content_piece, end='', flush=True)
                    full_response.append(content_piece)
        print()  # Newline after the stream is complete
        return "".join(full_response)
