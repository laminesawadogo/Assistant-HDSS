"""
Coeur du systeme RAG : recherche des chunks pertinents dans l'index,
construit le prompt, appelle le LLM, renvoie la reponse + les sources
(pour pouvoir verifier que l'agent ne repond pas hors-sujet / n'hallucine
pas - cf. etape d'evaluation du RAG).
"""

import os
import pickle
import re
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path(__file__).parent / "data" / "index" / "index.pkl"
INSTRUCTIONS_PATH = Path(__file__).parent / "instructions_systeme.md"

_cache = {}


class IndexNotBuiltError(RuntimeError):
    """Leve quand l'index de recherche n'a pas encore ete construit."""


def index_exists() -> bool:
    return INDEX_PATH.exists()


def reset_cache() -> None:
    """Vide le cache memoire de l'index. A appeler juste apres avoir
    reconstruit l'index (ingest.build_index()) dans une application qui
    tourne deja (Streamlit) : sans ca, l'ancien index resterait utilise en
    memoire jusqu'au redemarrage complet du serveur, meme si le fichier sur
    disque a bien ete mis a jour."""
    _cache.pop("index", None)


def _load_index():
    if "index" not in _cache:
        if not INDEX_PATH.exists():
            raise IndexNotBuiltError(
                "L'index de recherche n'existe pas encore. "
                "Il faut d'abord construire l'index (bouton 'Construire l'index' "
                "dans la barre laterale, ou commande `python ingest.py`)."
            )
        with open(INDEX_PATH, "rb") as f:
            _cache["index"] = pickle.load(f)
    return _cache["index"]


def _load_system_instructions() -> str:
    if INSTRUCTIONS_PATH.exists():
        return INSTRUCTIONS_PATH.read_text(encoding="utf-8")
    return "Tu es l'assistant de l'Observatoire de Population de Ouagadougou (OPO)."


# Documents de reference transversaux (schema, correspondance des tables) :
# leurs chunks sont plus longs et plus varies lexicalement qu'une simple
# definition de variable ("Variable X : Y"), donc systematiquement desavantages
# par la similarite cosinus (qui dilue le score sur un vecteur plus riche).
# On leur applique un leger boost pour qu'ils restent competitifs sur les
# questions transversales ("comment X est reliee a Y"), sans les faire
# dominer les questions tres specifiques a une seule variable.
SOURCES_PRIORITAIRES = {"00_schema_relations", "00_correspondance_tables"}
# Le corpus s'est beaucoup elargi (fiches terrain, manuels, presentations
# ajoutes ensuite) : plus de chunks concurrents signifie qu'un boost fixe
# perd en efficacite relative. 1.5 (au lieu de 1.35) redonne de la marge pour
# rester dans le top des resultats sur les questions transversales, verifie
# empiriquement sur le corpus actuel (cf. tests/test_rag.py).
BOOST_SOURCES_PRIORITAIRES = 1.5


def retrieve(query: str, k: int = 5) -> list[dict]:
    """Renvoie les k chunks les plus proches de la question (TF-IDF + cosinus)."""
    index = _load_index()
    q_vec = index["vectorizer"].transform([query])
    scores = cosine_similarity(q_vec, index["matrix"])[0].copy()

    for i, chunk in enumerate(index["chunks"]):
        if chunk["source"] in SOURCES_PRIORITAIRES:
            scores[i] *= BOOST_SOURCES_PRIORITAIRES

    top_idx = scores.argsort()[::-1][:k]
    results = []
    for i in top_idx:
        if scores[i] <= 0:
            continue
        chunk = index["chunks"][i]
        results.append({**chunk, "score": float(scores[i])})
    return results


def build_prompt(query: str, chunks: list[dict], historique: list[dict] | None = None) -> str:
    context = "\n".join(f"- {c['text']}" for c in chunks) if chunks else "(aucun document pertinent trouve)"

    bloc_historique = ""
    if historique:
        tours = "\n".join(f"{h['role'].capitalize()} : {h['contenu']}" for h in historique)
        bloc_historique = (
            "Echanges precedents de cette meme conversation (pour comprendre une question de "
            "suivi comme \"et pour l'autre table ?\" ou \"peux-tu detailler ?\") :\n"
            f"{tours}\n\n"
        )

    return (
        f"{_load_system_instructions()}\n\n"
        "Reponds uniquement a partir des extraits de documents ci-dessous (dictionnaire de "
        "donnees, fiches, manuels, notes de reference). Si l'information n'y figure pas, dis "
        "clairement que tu ne sais pas plutot que d'inventer. Redige une reponse claire, complete "
        "et bien ecrite, comme dans une vraie conversation : reformule et explique avec tes propres "
        "mots, ne te contente jamais de recopier les extraits tels quels. Si la question demande un "
        "exercice ou un QCM, construis-le a partir de ces memes extraits, avec le corrige.\n\n"
        f"{bloc_historique}"
        f"Extraits des documents de reference (contexte recupere) :\n{context}\n\n"
        f"Question de l'utilisateur : {query}\n\n"
        "Reponse :"
    )


def has_llm_configured(groq_key: str | None = None, anthropic_key: str | None = None) -> bool:
    return bool(groq_key or os.getenv("GROQ_API_KEY") or anthropic_key or os.getenv("ANTHROPIC_API_KEY"))


def call_llm(prompt: str, groq_key: str | None = None, anthropic_key: str | None = None) -> str:
    """Appel au LLM. Choisit le fournisseur selon la cle d'API disponible.

    Les cles peuvent etre passees en argument (ex: saisies dans l'interface)
    ou definies en variable d'environnement GROQ_API_KEY / ANTHROPIC_API_KEY.

    - GROQ_API_KEY  : rapide, gratuit (modeles Llama 3 / Mixtral)
    - ANTHROPIC_API_KEY : Claude
    Si aucune cle n'est configuree, renvoie un message explicite (pour pouvoir
    tester l'interface et la recuperation de contexte sans LLM branche).
    """
    groq_key = groq_key or os.getenv("GROQ_API_KEY")
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")

    # Priorite a Anthropic (Claude) quand les deux cles sont disponibles : la
    # qualite de redaction et de comprehension est nettement meilleure que le
    # petit modele Groq gratuit - Groq ne sert de repli que si aucune cle
    # Anthropic n'est renseignee.
    #
    # Modele Claude Sonnet 5 (et non Haiku) : demande explicite de
    # l'observatoire suite a des reponses jugees pas assez precises,
    # notamment sur le contenu des bases de donnees - Haiku privilegie la
    # vitesse/le cout, Sonnet est nettement meilleur en comprehension/
    # raisonnement pour un cout par requete qui reste raisonnable au volume
    # d'un observatoire (pas besoin d'Opus, plus cher, pour cet usage).
    if anthropic_key:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text

    if groq_key:
        from groq import Groq

        client = Groq(api_key=groq_key)
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    return (
        "⚠️ **Aucune clé d'API n'est configurée** (ni GROQ_API_KEY, ni ANTHROPIC_API_KEY).\n\n"
        "Ajoute une clé dans la barre latérale (\"Configuration du modèle\") pour obtenir "
        "une vraie réponse rédigée. En attendant, voici le contexte du dictionnaire "
        "qui a été retrouvé pour ta question :\n\n---\n\n" + prompt
    )


def analyser_image(
    image_bytes: bytes, mime_type: str, question: str | None = None, anthropic_key: str | None = None
) -> str:
    """Envoie une image (ex: photo ou scan d'une fiche terrain remplie) a
    Claude avec une question, pour qu'il en decrive ou transcrive le contenu.

    Necessite une cle Anthropic : la lecture d'image (vision) n'est pas geree
    par le modele Groq gratuit utilise par ailleurs dans cette application."""
    anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
    if not anthropic_key:
        return (
            "⚠️ **La lecture d'image nécessite une clé Anthropic** (modèle Claude). "
            "Configure-la dans `.env` ou la barre latérale, puis réessaie."
        )

    import base64

    import anthropic

    client = anthropic.Anthropic(api_key=anthropic_key)
    image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    resp = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1200,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_b64}},
                    {
                        "type": "text",
                        "text": question or "Décris le contenu de cette image et transcris le texte lisible.",
                    },
                ],
            }
        ],
    )
    return resp.content[0].text


ACTIONS_CONNUES = {"REPARTITION", "ECHANTILLON", "DOUBLONS", "COHERENCE", "LISTE_TABLES", "AUCUNE"}


def classifier_intention(
    question: str, colonnes: list[str], groq_key: str | None = None, anthropic_key: str | None = None
) -> tuple[str, str | int | None]:
    """Demande au LLM de classer une question en une action exploitable sur la
    table chargee, plutot que de se fier uniquement a des mots-cles figes.

    Renvoie un tuple (action, parametre) parmi :
      ("REPARTITION", nom_de_colonne)
      ("ECHANTILLON", n)
      ("DOUBLONS", None)
      ("COHERENCE", None)
      ("LISTE_TABLES", None)  -> question sur le nombre/la liste des tables chargees
      ("AUCUNE", None)   -> la question ne concerne pas une action sur la table

    Si aucune cle LLM n'est configuree, ou si la reponse du modele est
    inexploitable, renvoie ("AUCUNE", None) : l'appelant doit alors prevoir un
    repli (message d'aide, ou recherche documentaire).
    """
    if not has_llm_configured(groq_key, anthropic_key):
        return "AUCUNE", None

    prompt = (
        "Tu classes une question posee sur une table de donnees en UNE SEULE action, "
        "parmi exactement ces formats de reponse possibles :\n"
        "REPARTITION:<nom_de_colonne>\n"
        "ECHANTILLON:<nombre_de_lignes>\n"
        "DOUBLONS\n"
        "COHERENCE\n"
        "LISTE_TABLES\n"
        "AUCUNE\n\n"
        f"Colonnes disponibles dans la table : {', '.join(colonnes)}\n"
        f"Question : {question}\n\n"
        "Reponds uniquement avec l'une de ces lignes, sans aucune explication. "
        "Utilise LISTE_TABLES des que la question porte sur les tables/fichiers/feuilles "
        "actuellement charges eux-memes (combien il y en a, lesquels, si tu les as bien recus, "
        "confirmation de ce qui a ete envoye...) plutot que sur le contenu d'une table precise. "
        "Si la question ne correspond a aucune de ces actions (par exemple une question "
        "sur la signification d'une variable), reponds AUCUNE."
    )

    try:
        reponse = call_llm(prompt, groq_key=groq_key, anthropic_key=anthropic_key).strip()
    except Exception:
        return "AUCUNE", None

    m = re.search(
        r"\b(REPARTITION|ECHANTILLON|DOUBLONS|COHERENCE|LISTE_TABLES|AUCUNE)\b\s*:?\s*([\w À-ÿ]*)",
        reponse, re.IGNORECASE,
    )
    if not m:
        return "AUCUNE", None

    action = m.group(1).upper()
    parametre = m.group(2).strip() or None

    if action == "REPARTITION":
        if parametre is None:
            return "AUCUNE", None
        # On ne garde le nom de colonne que s'il correspond reellement a une
        # colonne existante (evite qu'un LLM invente un nom de colonne).
        correspondance = next((c for c in colonnes if c.lower() == parametre.lower()), None)
        return ("REPARTITION", correspondance) if correspondance else ("AUCUNE", None)

    if action == "ECHANTILLON":
        try:
            return "ECHANTILLON", int(parametre)
        except (TypeError, ValueError):
            return "ECHANTILLON", 100

    if action in ("DOUBLONS", "COHERENCE", "LISTE_TABLES", "AUCUNE"):
        return action, None

    return "AUCUNE", None


# En dessous de ce score pour le meilleur resultat, on considere que la
# recherche TF-IDF n'a probablement pas bien "compris" la question (question
# trop reformulee, vocabulaire different du corpus) et on tente une seule
# reformulation via le LLM avant d'abandonner - sans alourdir le cas normal
# (question qui recoupe deja bien le vocabulaire du corpus).
SEUIL_SCORE_FAIBLE = 0.12


def _reformuler_requete(query: str, groq_key: str | None = None, anthropic_key: str | None = None) -> str | None:
    """Demande au LLM de reformuler la question en une requete de recherche
    plus proche du vocabulaire technique du corpus (noms de tables, de
    variables, termes de l'observatoire), pour rattraper une recherche TF-IDF
    qui n'a rien trouve de pertinent sur la formulation initiale."""
    prompt = (
        "Tu reformules une question en une courte requete de recherche documentaire, "
        "en te rapprochant du vocabulaire technique probable du corpus (noms de tables, "
        "noms de variables, termes de demographie/observatoire de population). "
        "Reponds uniquement avec la requete reformulee, sans guillemets ni explication.\n\n"
        f"Question : {query}\n\nRequete reformulee :"
    )
    try:
        reponse = call_llm(prompt, groq_key=groq_key, anthropic_key=anthropic_key).strip()
    except Exception:
        return None
    if not reponse or len(reponse) > 300 or "clé d'api" in reponse.lower() or "cle d'api" in reponse.lower():
        return None
    return reponse


def answer(
    query: str,
    k: int = 5,
    groq_key: str | None = None,
    anthropic_key: str | None = None,
    historique: list[dict] | None = None,
) -> dict:
    chunks = retrieve(query, k=k)
    meilleur_score = chunks[0]["score"] if chunks else 0.0

    if meilleur_score < SEUIL_SCORE_FAIBLE and has_llm_configured(groq_key, anthropic_key):
        reformulation = _reformuler_requete(query, groq_key=groq_key, anthropic_key=anthropic_key)
        if reformulation and reformulation.strip().lower() != query.strip().lower():
            nouveaux_chunks = retrieve(reformulation, k=k)
            if nouveaux_chunks and nouveaux_chunks[0]["score"] > meilleur_score:
                chunks = nouveaux_chunks

    prompt = build_prompt(query, chunks, historique=historique)
    response = call_llm(prompt, groq_key=groq_key, anthropic_key=anthropic_key)
    return {"answer": response, "sources": chunks}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "Que signifie la variable fatherid ?"
    result = answer(q)
    print("QUESTION:", q)
    print("\nSOURCES RECUPEREES:")
    for s in result["sources"]:
        print(f"  ({s['score']:.2f}) {s['text']}")
    print("\nREPONSE:\n", result["answer"])
