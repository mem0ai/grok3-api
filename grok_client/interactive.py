import os
import sys
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_client():
    """Set up and return the OpenAI client with Grok API configuration"""
    # Load environment variables
    load_dotenv()
    
    # Get configuration from environment
    api_host = os.getenv('API_HOST', '127.0.0.1')
    api_port = os.getenv('API_PORT', '8000')
    model_name = os.getenv('MODEL_NAME', 'grok-3')
    grok_sso = os.getenv('GROK_SSO')
    grok_sso_rw = os.getenv('GROK_SSO_RW')
    
    if not all([grok_sso, grok_sso_rw]):
        logger.error("Missing required environment variables. Please check your .env file.")
        logger.info("Required variables: GROK_SSO, GROK_SSO_RW")
        logger.info("Optional variables: API_HOST, API_PORT, MODEL_NAME")
        sys.exit(1)
    
    # Initialize OpenAI client with local endpoint
    client = OpenAI(
        base_url=f"http://{api_host}:{api_port}/v1",
        api_key="dummy-key",  # Not used but required
        default_headers={
            "Cookie": f"sso={grok_sso}; sso-rw={grok_sso_rw}"
        }
    )
    
    return client, model_name

def interactive_chat():
    """Run an interactive chat session with Grok"""
    client, model_name = setup_client()
    
    print("\n===== Grok Interactive Chat =====")
    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("==============================\n")
    
    # Initialize conversation history
    conversation = []
    
    try:
        while True:
            # Get user input
            user_input = input("\nYou: ")
            
            # Check for exit commands
            if user_input.lower() in ['exit', 'quit']:
                print("\nExiting chat. Goodbye!")
                break
            
            # Check for clear command
            if user_input.lower() == 'clear':
                conversation = []
                print("\nConversation history cleared.")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            try:
                # Send request to Grok API
                print("\nGrok: ", end="", flush=True)
                
                # Use streaming for a more interactive experience
                stream = client.chat.completions.create(
                    model=model_name,
                    messages=conversation,
                    stream=True
                )
                
                # Collect the full response while streaming
                full_response = ""
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        print(content, end="", flush=True)
                        full_response += content
                
                print()  # Add a newline after the response
                
                # Add assistant response to conversation history
                conversation.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n\nExiting chat. Goodbye!")

if __name__ == "__main__":
    interactive_chat()