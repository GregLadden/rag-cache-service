import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY not found in .env")
    sys.exit(1)

print("Connecting to Gemini API...")
try:
    client = genai.Client(api_key=api_key)
    print("Successfully initialized client. Querying available models...")
    
    # List available models for this API key
    response = client.models.list()
    
    print("\nAvailable models for your API key:")
    print("--------------------------------------------------")
    for m in response:
        # Check if model supports generateContent
        methods = m.supported_methods if hasattr(m, 'supported_methods') else []
        if 'generateContent' in methods:
            print(f"- {m.name} (Supports content generation)")
        else:
            print(f"- {m.name}")
    print("--------------------------------------------------")
except Exception as e:
    print(f"Error querying Gemini API: {e}")
