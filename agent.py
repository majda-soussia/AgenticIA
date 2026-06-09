from ollama import chat

messages = []

while True:
    user_input = input("Vous : ")

    if user_input.lower() == "exit":
        break

    # Outil calculatrice
    if user_input.startswith("calc "):
        expression = user_input[5:]

        try:
            result = eval(expression)
            print(f"\nCalculatrice : {result}")
        except:
            print("\nExpression invalide")

        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = chat(
        model="llama3",
        messages=messages
    )

    reply = response.message.content

    print("\nLlama :", reply)

    messages.append({
        "role": "assistant",
        "content": reply
    })