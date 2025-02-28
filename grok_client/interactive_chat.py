import os
import sys
import logging
import argparse
from dotenv import load_dotenv
from .grok_openai_client import GrokOpenAIClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_arguments():
    """
    Parse command line arguments for the interactive chat application.
    
    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Interactive chat with Grok API using OpenAI-compatible interface')
    parser.add_argument('--host', help='API host (default: from .env or 127.0.0.1)')
    parser.add_argument('--port', help='API port (default: from .env or 8000)')
    parser.add_argument('--model', help='Model name (default: from .env or grok-3)')
    parser.add_argument('--sso', help='SSO token (default: from .env)')
    parser.add_argument('--sso-rw', help='SSO-RW token (default: from .env)')
    parser.add_argument('--json', action='store_true', help='Request responses in JSON format')
    parser.add_argument('--system', help='Custom system message')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for response generation (default: 1.0)')
    
    return parser.parse_args()

def setup_client(args):
    """
    Set up the Grok OpenAI client using command line arguments or environment variables.
    
    Args:
        args (argparse.Namespace): The parsed command line arguments.
        
    Returns:
        GrokOpenAIClient: The initialized client.
    """
    try:
        # Initialize client with args or environment variables
        client = GrokOpenAIClient(
            api_host=args.host,
            api_port=args.port,
            model_name=args.model,
            sso_token=args.sso,
            sso_rw_token=args.sso_rw,
            load_from_env=True  # Always try to load from env first
        )
        
        return client
    except ValueError as e:
        logger.error(f"Error initializing client: {e}")
        logger.info("Make sure you have the required environment variables set in .env file or provided as arguments.")
        logger.info("Required variables: GROK_SSO, GROK_SSO_RW")
        logger.info("Optional variables: API_HOST, API_PORT, MODEL_NAME")
        sys.exit(1)

def interactive_chat():
    """
    Run an interactive chat session with Grok using the OpenAI-compatible interface.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up client
    client = setup_client(args)
    model_name = client.model_name
    
    # Set up system message
    system_message = args.system
    if args.json and not system_message:
        system_message = "You are a helpful assistant that always responds in valid JSON format."
    elif not system_message:
        system_message = "You are a helpful assistant."
    
    print(f"\n===== Grok Interactive Chat ({model_name}) =====")
    print("Type 'exit', 'quit', or Ctrl+C to end the conversation.")
    print("Type 'clear' to start a new conversation.")
    print("Type '/help' to see available commands.")
    print("==============================\n")
    
    # Initialize conversation history
    conversation = []
    if system_message:
        conversation.append({"role": "system", "content": system_message})
    
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
                if system_message:
                    conversation.append({"role": "system", "content": system_message})
                print("\nConversation history cleared.")
                continue
            
            # Check for help command
            if user_input.lower() == '/help':
                print("\nAvailable commands:")
                print("  exit, quit - Exit the chat")
                print("  clear - Clear conversation history")
                print("  /help - Show this help message")
                print("  /json - Toggle JSON response format")
                print("  /temp <value> - Set temperature (0.0-2.0)")
                print("  /system <message> - Set system message")
                continue
            
            # Check for JSON toggle command
            if user_input.lower() == '/json':
                args.json = not args.json
                print(f"\nJSON response format: {'enabled' if args.json else 'disabled'}")
                continue
            
            # Check for temperature command
            if user_input.lower().startswith('/temp '):
                try:
                    new_temp = float(user_input.split(' ', 1)[1])
                    if 0.0 <= new_temp <= 2.0:
                        args.temperature = new_temp
                        print(f"\nTemperature set to: {args.temperature}")
                    else:
                        print("\nTemperature must be between 0.0 and 2.0")
                except (ValueError, IndexError):
                    print("\nInvalid temperature value. Format: /temp 0.7")
                continue
            
            # Check for system message command
            if user_input.lower().startswith('/system '):
                system_message = user_input.split(' ', 1)[1]
                # Update the system message in the conversation
                conversation = [msg for msg in conversation if msg["role"] != "system"]
                conversation.insert(0, {"role": "system", "content": system_message})
                print(f"\nSystem message updated.")
                continue
            
            # Add user message to conversation
            conversation.append({"role": "user", "content": user_input})
            
            try:
                # Send request to Grok API
                print("\nGrok: ", end="", flush=True)
                
                # Prepare request parameters
                params = {
                    "messages": conversation,
                    "stream": True,
                    "temperature": args.temperature
                }
                
                # Add JSON format if requested
                if args.json:
                    params["response_format"] = {"type": "json_object"}
                
                # Use streaming for a more interactive experience
                stream = client.chat_completion(**params)
                
                # Process the streaming response
                full_response = client.process_streaming_response(stream)
                
                # Add assistant response to conversation history
                conversation.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print(f"\nAn error occurred: {str(e)}")
    
    except KeyboardInterrupt:
        print("\n\nExiting chat. Goodbye!")

def main():
    # Load environment variables
    load_dotenv()
    
    # Run interactive chat
    interactive_chat()

if __name__ == "__main__":
    main()