from ollama import chat

messages = []

while True:
    user_input = input("Vous : ")

    if user_input.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = chat(
        model="llama3",
        messages=messages
    )

    assistant_reply = response.message.content

    print("\nLlama :", assistant_reply)

    messages.append({
        "role": "assistant",
        "content": assistant_reply
    })