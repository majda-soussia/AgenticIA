from ollama import chat

response = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": "Bonjour"
        }
    ]
)

print(response.message.content)
