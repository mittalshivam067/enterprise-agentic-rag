
import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

url = "https://generativelanguage.googleapis.com/v1beta/models"

response = requests.get(
    url,
    params={"key": api_key},
    timeout=30
)

print("Status:", response.status_code)

if response.status_code == 200:

    models = response.json().get("models", [])

    print("\nModels supporting text generation:\n")

    for model in models:

        supported_methods = model.get(
            "supportedGenerationMethods", []
        )

        if "generateContent" in supported_methods:

            print(model["name"])

else:
    print(response.text)
