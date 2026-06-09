from ollama import chat
from ddgs import DDGS

messages = []

while True:
    user_input = input("> ")

    if user_input.lower() == "exit":
        break

    if user_input.startswith("web "):
        query = user_input[4:].strip()

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))

            context = "\n".join(
                f"{r['title']}\n{r['body']}"
                for r in results
            )

            messages.append({
                "role": "user",
                "content": f"Question: {query}\n\nInformations trouvées:\n{context}"
            })

            response = chat(
                model="llama3",
                messages=messages
            )

            answer = response.message.content

            print("\nLlama :", answer)

            messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            print("Erreur :", e)

        continue