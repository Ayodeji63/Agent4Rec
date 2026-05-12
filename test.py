
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
print('Sending test request...')
response = client.models.generate_content(
    model='gemini-3.1-flash-lite',
    contents='Say hello in one word.',
    config=types.GenerateContentConfig(max_output_tokens=10)
)
print('Response:', response.text)
