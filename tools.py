# AgenticAI/tools.py

import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient

client = InferenceClient(token="hf_TON_TOKEN_ICI")

modele_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Le chiffre d'affaires de janvier est 45000 DT.",
    "La réunion est prévue le 15 mars à 10h.",
    "Le client Ahmed a commandé 3 unités le 5 janvier.",
]
vecteurs_docs = modele_embed.encode(documents)


def calculatrice(expression):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Erreur : {e}"


def lire_csv(path):
    try:
        df = pd.read_csv(path.strip())
        return f"Colonnes : {list(df.columns)}\n{df.head(5).to_string()}"
    except Exception as e:
        return f"Erreur CSV : {e}"


def recherche_web(query):
    try:
        url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        resultats = [s.get_text() for s in soup.select(".result__snippet")[:3]]
        return " | ".join(resultats) if resultats else "Aucun résultat"
    except Exception as e:
        return f"Erreur web : {e}"


def chercher_doc(question):
    vecteur_q = modele_embed.encode([question])
    scores = np.dot(vecteurs_docs, vecteur_q.T).flatten()
    meilleur = int(np.argmax(scores))
    return documents[meilleur]


def analyser_sentiment(texte):
    try:
        result = client.text_classification(
            texte,
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        label = result[0]["label"]
        score = round(result[0]["score"] * 100, 1)
        return f"{label} ({score}% de confiance)"
    except Exception as e:
        return f"Erreur sentiment : {e}"


def traduire(texte):
    try:
        result = client.translation(texte, model="Helsinki-NLP/opus-mt-fr-en")
        return result.translation_text
    except Exception as e:
        return f"Erreur traduction : {e}"


def resumer(texte):
    try:
        result = client.summarization(texte, model="facebook/bart-large-cnn")
        return result.summary_text
    except Exception as e:
        return f"Erreur résumé : {e}"


TOOLS = {
    "calculatrice": {
        "description": (
            "Calcule une expression mathématique. "
            "Utilise pour toute question avec des chiffres, "
            "additions, soustractions, multiplications, pourcentages."
        ),
        "exemple": 'calculatrice("15 * 8")',
        "fonction": calculatrice,
    },
    "lire_csv": {
        "description": (
            "Lit un fichier CSV et retourne son contenu. "
            "Utilise si l'utilisateur mentionne un fichier .csv."
        ),
        "exemple": 'lire_csv("donnees.csv")',
        "fonction": lire_csv,
    },
    "recherche_web": {
        "description": (
            "Cherche des informations sur internet. "
            "Utilise pour l'actualité, des faits récents."
        ),
        "exemple": 'recherche_web("prix bitcoin aujourd\'hui")',
        "fonction": recherche_web,
    },
    "chercher_doc": {
        "description": (
            "Cherche dans les documents personnels. "
            "Utilise quand l'utilisateur pose une question sur ses fichiers."
        ),
        "exemple": 'chercher_doc("chiffre affaires janvier")',
        "fonction": chercher_doc,
    },
    "analyser_sentiment": {
        "description": "Analyse si un texte est positif, négatif ou neutre.",
        "exemple": 'analyser_sentiment("Ce produit est excellent !")',
        "fonction": analyser_sentiment,
    },
    "traduire": {
        "description": "Traduit un texte du français vers l'anglais.",
        "exemple": 'traduire("Bonjour le monde")',
        "fonction": traduire,
    },
    "resumer": {
        "description": "Résume un long texte en quelques phrases.",
        "exemple": 'resumer("Long texte ici...")',
        "fonction": resumer,
    },
}