# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np

# Modèle léger, tourne en local, pas besoin d'API
modele_embed = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Tes documents (texte brut ou extrait de PDF/CSV)
documents = [
    "Le chiffre d'affaires de janvier est 45000 DT.",
    "La réunion est prévue le 15 mars à 10h.",
    "Le client Ahmed a commandé 3 unités le 5 janvier.",
]

# Précalcule les vecteurs une seule fois
vecteurs_docs = modele_embed.encode(documents)

def chercher_doc(question):
    vecteur_question = modele_embed.encode([question])
    # Similarité cosinus entre la question et chaque doc
    scores = np.dot(vecteurs_docs, vecteur_question.T).flatten()
    meilleur_idx = int(np.argmax(scores))
    return documents[meilleur_idx]


