import requests
import json

class GrokClient:
    def __init__(self, cookies):
        """
        Initialize the Grok client with cookie values

        Args:
            cookies (dict): Dictionary containing cookie values
                - x-anonuserid
                - x-challenge
                - x-signature
                - sso
                - sso-rw
        """
        self.base_url = "https://grok.com/rest/app-chat/conversations/new"
        self.cookies = cookies
        self.headers = {
            "accept": "*/*",
            "accept-language": "en-GB,en;q=0.9",
            "content-type": "application/json",
            "origin": "https://grok.com",
            "priority": "u=1, i",
            "referer": "https://grok.com/",
            "sec-ch-ua": '"Not/A)Brand";v="8", "Chromium";v="126", "Brave";v="126"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"macOS"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "sec-gpc": "1",
            "user-agent": ""PostmanRuntime/7.32.3""
        }

    def _prepare_payload(self, message):
        """Prepare the default payload with the user's message"""
        return {
            "temporary": False,
            "modelName": "grok-3",
            "message": message,
            "fileAttachments": [],
            "imageAttachments": [],
            "disableSearch": False,
            "enableImageGeneration": True,
            "returnImageBytes": False,
            "returnRawGrokInXaiRequest": False,
            "enableImageStreaming": True,
            "imageGenerationCount": 2,
            "forceConcise": False,
            "toolOverrides": {},
            "enableSideBySide": True,
            "isPreset": False,
            "sendFinalMetadata": True,
            "customInstructions": "",
            "deepsearchPreset": "",
            "isReasoning": False
        }

    def send_message(self, message):
        """
        Send a message to Grok and collect the streaming response

        Args:
            message (str): The user's input message

        Returns:
            str: The complete response from Grok
        """
        payload = self._prepare_payload(message)
        response = requests.post(
            self.base_url,
            headers=self.headers,
            cookies=self.cookies,
            json=payload,
            stream=True
        )

        full_response = ""

        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                try:
                    json_data = json.loads(decoded_line)
                    result = json_data.get("result", {})
                    response_data = result.get("response", {})

                    if "modelResponse" in response_data:
                        return response_data["modelResponse"]["message"]

                    token = response_data.get("token", "")
                    if token:
                        full_response += token

                except json.JSONDecodeError:
                    continue

        return full_response.strip()
