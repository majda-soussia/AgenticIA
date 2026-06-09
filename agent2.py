# AgenticAI/agent2.py

import re
from huggingface_hub import InferenceClient
from tools import TOOLS

import os
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.3",
    token=os.getenv("HF_TOKEN"),
)

def appeler_llm(messages):
    prompt = ""
    for m in messages:
        if m["role"] == "system":
            prompt += f"<s>[INST] {m['content']}\n"
        elif m["role"] == "user":
            prompt += f"{m['content']} [/INST]"
        elif m["role"] == "assistant":
            prompt += f"{m['content']} </s><s>[INST] "

    return client.text_generation(
        prompt,
        max_new_tokens=512,
        temperature=0.1,
        stop_sequences=["Observation:"],
    )


def build_system_prompt(tools):
    tools_text = ""
    for nom, info in tools.items():
        tools_text += f"\n- {nom}(argument) : {info['description']}\n"
        tools_text += f"  Exemple : {info['exemple']}\n"

    return f"""Tu es un agent IA intelligent.

Tu as accès aux outils suivants :
{tools_text}
Règles STRICTES :
1. Réponds TOUJOURS dans ce format :
   Thought: [ta réflexion]
   Action: nom_outil("argument")

2. Si tu n'as pas besoin d'outil :
   Thought: Je peux répondre directement.
   Final Answer: [ta réponse]

3. N'invente jamais de résultat. Attends l'Observation.
4. Un seul outil par tour.
"""


SYSTEM_PROMPT = build_system_prompt(TOOLS)


def parse_action(llm_output):
    match = re.search(
        r'Action:\s*(\w+)\s*\(\s*["\']?(.*?)["\']?\s*\)',
        llm_output,
        re.DOTALL
    )
    if match:
        return match.group(1), match.group(2)
    return None, None


def executer_outil(tool_name, argument):
    if tool_name not in TOOLS:
        return f"Outil inconnu : '{tool_name}'. Disponibles : {list(TOOLS.keys())}"
    try:
        return str(TOOLS[tool_name]["fonction"](argument))
    except Exception as e:
        return f"Erreur dans {tool_name} : {e}"


def agent(question, max_tours=6):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]

    for tour in range(max_tours):
        print(f"\n{'─'*40}\nTour {tour + 1}")

        sortie = appeler_llm(messages)   # ← une seule ligne, plus de doublon
        print(sortie)

        if "Final Answer:" in sortie:
            return sortie.split("Final Answer:")[-1].strip()

        tool_name, argument = parse_action(sortie)
        if not tool_name:
            print("[Aucune action trouvée]")
            break

        observation = executer_outil(tool_name, argument)
        print(f"\nObservation : {observation}")

        messages.append({"role": "assistant", "content": sortie})
        messages.append({"role": "user",      "content": f"Observation: {observation}"})

    return "Limite de tours atteinte."


if __name__ == "__main__":
    print("Agent IA prêt. Tape 'exit' pour quitter.\n")
    while True:
        question = input("Vous : ")
        if question.lower() == "exit":
            break
        print(f"\nAgent : {agent(question)}")