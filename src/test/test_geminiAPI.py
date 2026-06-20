import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

print(api_key[:5])      # Should print something like AQabc
print(api_key is None)  # Should print False

client = genai.Client(api_key=api_key)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say Hello"
)

print(response.text)