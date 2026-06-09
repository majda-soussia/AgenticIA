from huggingface_hub import InferenceClient
import os

token = os.getenv("api_key")
client = InferenceClient(
    token = os.getenv("api_key")
)

messages = [
    {
        "role": "system",
        "content": "Tu es un assistant utile."
    },
    {
        "role": "user",
        "content": "Bonjour"
    }
]

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    messages=messages,
    max_tokens=50,
    temperature=0.1
)

print(response.choices[0].message.content)