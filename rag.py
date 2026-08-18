"""
Coeur du systeme RAG : recherche des chunks pertinents dans l'index,
construit le prompt, appelle le LLM, renvoie la reponse + les sources
(pour pouvoir verifier que l'agent ne repond pas hors-sujet / n'hallucine
pas - cf. etape d'evaluation du RAG).
"""

import json
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
        "exercice ou un QCM, construis-le a partir de ces memes extraits, avec le corrige. "
        "Termine, quand c'est pertinent, par une courte suggestion d'une question de suivi liee au "
        "sujet (une seule phrase, comme le ferait un assistant qui propose la prochaine etape plutot "
        "que de laisser la conversation s'arreter net) - jamais si la reponse est deja un refus "
        "('je ne sais pas').\n\n"
        f"{bloc_historique}"
        f"Extraits des documents de reference (contexte recupere) :\n{context}\n\n"
        f"Question de l'utilisateur : {query}\n\n"
        "Reponse :"
    )


def has_llm_configured(groq_key: str | None = None, anthropic_key: str | None = None) -> bool:
    return bool(groq_key or os.getenv("GROQ_API_KEY") or anthropic_key or os.getenv("ANTHROPIC_API_KEY"))


def _texte_reponse_anthropic(resp) -> str:
    """Extrait le texte d'une reponse `client.messages.create(...)` Anthropic.

    Bug reel corrige ici : `resp.content[0].text` supposait que le PREMIER
    bloc de `resp.content` est toujours le texte - vrai avec les anciens
    modeles, mais Claude Opus 5 (modele de raisonnement hybride) peut
    renvoyer un ou plusieurs blocs `ThinkingBlock` (son raisonnement interne,
    attribut `.thinking`, pas `.text`) AVANT le bloc de texte reel. Prendre
    `content[0]` sans distinction faisait planter l'appli en production
    (`AttributeError: 'ThinkingBlock' object has no attribute 'text'`) a
    chaque reponse ou le modele activait la reflexion etendue - tres
    probablement aussi la cause des erreurs frontend "removeChild" observees
    juste avant (le script Python plantait en cours de rendu, laissant le
    DOM du navigateur dans un etat incoherent).

    Ne fait plus confiance a la position du bloc : ne garde que les blocs
    dont `type == \"text\"` (il peut y en avoir plusieurs, concatenes)."""
    return "".join(
        bloc.text for bloc in (resp.content or []) if getattr(bloc, "type", None) == "text"
    )


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
    # Modele Claude Opus 5 : demande explicite de l'observatoire pour la
    # meilleure precision possible, notamment sur les questions qui croisent
    # plusieurs tables - historique : Haiku (jugee pas assez precise) puis
    # Sonnet 5, avant ce dernier passage sur Opus 5 (le plus capable de la
    # gamme, mais aussi le plus cher).
    if anthropic_key:
        import anthropic

        client = anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-opus-5",
            max_tokens=1200,
            messages=[{"role": "user", "content": prompt}],
        )
        return _texte_reponse_anthropic(resp)

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
        model="claude-opus-5",
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
    return _texte_reponse_anthropic(resp)


ACTIONS_CONNUES = {"REPARTITION", "ECHANTILLON", "DOUBLONS", "COHERENCE", "LISTE_TABLES", "REQUETE", "AUCUNE"}

# Operateurs de filtre reconnus par `data_tools.executer_requete_donnees` -
# dupliques ici uniquement pour construire le prompt (garder les deux listes
# synchronisees si un operateur est ajoute).
OPERATEURS_REQUETE_CONNUS = ["==", "!=", ">", "<", ">=", "<=", "contient"]


def classifier_intention(
    question: str, colonnes: list[str], groq_key: str | None = None, anthropic_key: str | None = None
) -> tuple[str, str | int | dict | None]:
    """Demande au LLM de classer une question en une action exploitable sur la
    table chargee, plutot que de se fier uniquement a des mots-cles figes.

    Renvoie un tuple (action, parametre) parmi :
      ("REPARTITION", nom_de_colonne)
      ("ECHANTILLON", n)
      ("DOUBLONS", None)
      ("COHERENCE", None)
      ("LISTE_TABLES", None)  -> question sur le nombre/la liste des tables chargees
      ("REQUETE", specification)  -> calcul precis (compter/lister/moyenne/
        somme/min/max), eventuellement filtre - specification est un dict
        {"operation": ..., "colonne_cible": str|None, "filtres": [...]}
        a executer via `data_tools.executer_requete_donnees` - c'est ce qui
        permet de repondre a une question comme "combien de naissances a
        Ouahigouya en 2026 ?" directement a partir des donnees reellement
        chargees, au lieu de se limiter aux 4 analyses fixes ci-dessus ou de
        retomber sur le dictionnaire documentaire.
      ("AUCUNE", None)   -> la question ne concerne pas une action sur la table

    Si aucune cle LLM n'est configuree, ou si la reponse du modele est
    inexploitable, renvoie ("AUCUNE", None) : l'appelant doit alors prevoir un
    repli (message d'aide, ou recherche documentaire).
    """
    if not has_llm_configured(groq_key, anthropic_key):
        return "AUCUNE", None

    prompt = (
        "Tu classes une question posee sur des donnees reellement chargees (une ou plusieurs tables) "
        "en UNE SEULE action, parmi exactement ces formats de reponse possibles :\n"
        "REPARTITION:<nom_de_colonne>\n"
        "ECHANTILLON:<nombre_de_lignes>\n"
        "DOUBLONS\n"
        "COHERENCE\n"
        "LISTE_TABLES\n"
        "REQUETE:<JSON>\n"
        "AUCUNE\n\n"
        f"Colonnes disponibles : {', '.join(colonnes)}\n"
        f"Question : {question}\n\n"
        "Reponds uniquement avec l'une de ces lignes, sans aucune explication, sans backticks. "
        "Utilise LISTE_TABLES des que la question porte sur les tables/fichiers/feuilles "
        "actuellement charges eux-memes (combien il y en a, lesquels, si tu les as bien recus, "
        "confirmation de ce qui a ete envoye...) plutot que sur le contenu d'une table precise.\n\n"
        "Utilise REQUETE des que la question demande un CALCUL PRECIS a partir des donnees reelles : "
        "compter des lignes, lister des lignes, ou calculer une moyenne/somme/min/max d'une colonne "
        "numerique - eventuellement avec une ou plusieurs conditions (ex: \"combien de naissances a "
        "Ouahigouya en 2026 ?\", \"liste des individus dont l'age > 60\", \"age moyen des mères\"). "
        "Le JSON qui suit REQUETE: doit avoir EXACTEMENT cette forme, sur une seule ligne, "
        "avec uniquement des noms de colonnes qui existent reellement dans la liste ci-dessus "
        "(jamais un nom invente) :\n"
        '{"operation": "compter", "colonne_cible": null, "filtres": '
        '[{"colonne": "<nom_colonne_existante>", "operateur": "==", "valeur": "<valeur>"}]}\n'
        "operation est l'une de : compter, lister, moyenne, somme, min, max. "
        "colonne_cible est requis (nom de colonne existante) pour moyenne/somme/min/max, sinon null. "
        f"operateur est l'un de : {', '.join(OPERATEURS_REQUETE_CONNUS)}. filtres peut etre une liste vide.\n\n"
        "IMPORTANT : les colonnes listees ci-dessus appartiennent A UNE SEULE table. Si la question "
        "necessite de comparer avec le contenu d'une AUTRE table non representee ici (ex : \"qui sont "
        "aussi dans...\", \"qui ne sont pas dans...\", \"present dans X mais pas dans Y\", \"en commun "
        "avec...\"), ne propose JAMAIS de filtre ou de listing approximatif pour t'en approcher (ex : un "
        "filtre sans valeur reelle, ou un listing complet non filtre) : reponds AUCUNE, une autre methode "
        "plus capable (qui voit toutes les tables a la fois) prendra le relais.\n\n"
        "Si la question ne correspond a aucune de ces actions (par exemple une question "
        "sur la signification d'une variable), reponds AUCUNE."
    )

    try:
        reponse = call_llm(prompt, groq_key=groq_key, anthropic_key=anthropic_key).strip()
    except Exception:
        return "AUCUNE", None

    if re.match(r"^REQUETE\s*:", reponse, re.IGNORECASE):
        return _parser_reponse_requete(reponse, colonnes)

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


def _parser_reponse_requete(reponse: str, colonnes: list[str]) -> tuple[str, dict | None]:
    """Extrait et valide le JSON d'une reponse `REQUETE:<JSON>` du
    classifieur - jamais de confiance aveugle dans un JSON genere par un LLM :
    toute colonne (cible ou de filtre) qui ne correspond a aucune colonne
    reellement chargee est silencieusement retiree plutot que transmise telle
    quelle a `data_tools.executer_requete_donnees` (qui revalide de toute
    facon, mais autant ne pas propager un nom invente jusque-la). Renvoie
    ("AUCUNE", None) si le JSON est absent/invalide/vide apres nettoyage."""
    correspondance_colonne = {c.lower(): c for c in colonnes}

    bloc = re.sub(r"^REQUETE\s*:", "", reponse, flags=re.IGNORECASE).strip()
    m_json = re.search(r"\{.*\}", bloc, re.DOTALL)
    if not m_json:
        return "AUCUNE", None
    try:
        specification = json.loads(m_json.group(0))
    except (json.JSONDecodeError, ValueError):
        return "AUCUNE", None
    if not isinstance(specification, dict):
        return "AUCUNE", None

    operation = str(specification.get("operation", "")).strip().lower()
    if operation not in ("compter", "lister", "moyenne", "somme", "min", "max"):
        return "AUCUNE", None

    colonne_cible = specification.get("colonne_cible")
    if colonne_cible:
        colonne_cible = correspondance_colonne.get(str(colonne_cible).lower())
    if operation in ("moyenne", "somme", "min", "max") and not colonne_cible:
        return "AUCUNE", None

    filtres_valides = []
    filtre_sans_valeur_exploitable = False
    for f in specification.get("filtres") or []:
        if not isinstance(f, dict):
            continue
        col = correspondance_colonne.get(str(f.get("colonne", "")).lower())
        op = f.get("operateur")
        if not col or op not in OPERATEURS_REQUETE_CONNUS:
            # Colonne halluc inee ou operateur inconnu : filtre ignore en
            # silence, comme avant (l'operation reste exploitable sans lui).
            continue
        valeur = f.get("valeur")
        if "valeur" not in f or valeur is None or (isinstance(valeur, str) and not valeur.strip()):
            # Colonne et operateur valides, mais aucune valeur exploitable :
            # signe que le LLM essaie d'exprimer quelque chose que cette
            # action MONO-TABLE ne sait pas representer (ex: "existe aussi
            # dans une autre table" -> comparaison sans valeur reelle, comme
            # observe sur "individid != "). Plutot que de renvoyer un compte
            # non filtre en le faisant passer pour une reponse precise,
            # on abandonne cette action pour laisser la main au repli plus
            # capable (voir app.py:tenter_requete_sql, qui sait joindre
            # plusieurs tables).
            filtre_sans_valeur_exploitable = True
            continue
        filtres_valides.append({"colonne": col, "operateur": op, "valeur": valeur})

    if filtre_sans_valeur_exploitable:
        return "AUCUNE", None

    return "REQUETE", {"operation": operation, "colonne_cible": colonne_cible, "filtres": filtres_valides}


def generer_requete_sql(
    question: str,
    schema: str,
    groq_key: str | None = None,
    anthropic_key: str | None = None,
    tentative_precedente: str | None = None,
    erreur_precedente: str | None = None,
    contexte_dictionnaire: str | None = None,
    historique: list[dict] | None = None,
) -> str | None:
    """Demande au LLM d'ecrire une requete SQL en LECTURE SEULE repondant a
    la question, a partir du SCHEMA REEL de TOUTES les tables chargees
    (fourni par l'appelant : nom de chaque table et liste exacte de ses
    colonnes, avec si possible des indices de jointure - voir
    `app.py:_description_schema`) - capacite GENERALE permettant de croiser
    2, 3, 4 tables ou plus dans une seule requete (jointures, filtres,
    agregations, groupby), la ou l'action REQUETE de `classifier_intention`
    ne sait interroger qu'UNE seule table a la fois, sans jointure.

    `tentative_precedente`/`erreur_precedente` : si une premiere requete a
    echoue a l'execution (voir `data_tools.executer_sql`), l'appelant peut
    redemander une correction en transmettant la requete fautive et le
    message d'erreur DuckDB - permet une auto-correction en un aller-retour
    plutot que d'abandonner directement sur une simple faute de syntaxe ou un
    nom de colonne legerement incorrect.

    `contexte_dictionnaire` : extraits du dictionnaire de donnees/manuels/
    fiches deja indexes (voir `retrieve`), pertinents pour la question -
    l'observatoire documente deja precisement le sens et le role de chaque
    identifiant (ex: respondid = repondant, locationid = UCH, socialgpid =
    menage, individid = individu, observationid = observation d'un individu
    dans le menage). Ce contexte donne au modele une source autorisee pour
    choisir les bonnes colonnes de jointure, en plus des indices purement
    structurels (colonnes de meme nom) deja presents dans `schema`.

    `historique` : tours precedents de cette meme conversation (meme format
    que `build_prompt`), pour comprendre une question de suivi qui ne
    reprend pas tout le contexte (ex: "et pour les femmes ?" apres une
    premiere question sur un sous-groupe precis). Bug reel corrige ici :
    cette fonction ne recevait JAMAIS l'historique (contrairement a
    `answer()`), donc toute question de suivi passant par la generation SQL
    etait traitee dans le vide, sans aucune memoire de ce qui venait d'etre
    demande - cause probable du symptome rapporte "le modele n'arrive pas a
    lier les conversations".

    Renvoie la requete SQL brute (str) si le LLM en propose une, ou None si
    aucune cle LLM n'est configuree, si le modele indique ne pas pouvoir
    repondre avec ce schema, ou si l'appel echoue. La requete renvoyee n'est
    PAS revalidee ici : c'est `data_tools.executer_sql` qui verifie qu'il
    s'agit bien d'une simple lecture (SELECT) avant toute execution - cette
    fonction ne fait que demander au modele et nettoyer le texte renvoye."""
    if not has_llm_configured(groq_key, anthropic_key):
        return None

    bloc_historique = ""
    if historique:
        tours = "\n".join(f"{h['role'].capitalize()} : {h['contenu']}" for h in historique)
        bloc_historique = (
            "Echanges precedents de cette meme conversation (pour comprendre une question de suivi "
            "qui ne repete pas tout le contexte, ex: \"et pour les femmes seulement ?\", \"et dans "
            "l'autre table ?\", \"meme chose mais pour 2025\") - reformule la question actuelle a la "
            "lumiere de ces echanges avant d'ecrire la requete, sans jamais perdre de vue qu'elle "
            "doit rester STRICTEMENT limitee aux tables/colonnes reellement listees ci-dessous :\n"
            f"{tours}\n\n"
        )

    prompt = (
        "Tu ecris UNE SEULE requete SQL (dialecte DuckDB, proche de PostgreSQL standard) EN LECTURE "
        "SEULE (SELECT, ou WITH ... SELECT) pour repondre a la question - jamais INSERT/UPDATE/"
        "DELETE/DROP/CREATE/ALTER/PRAGMA/ATTACH. Utilise UNIQUEMENT les tables et colonnes listees "
        "ci-dessous, avec leur nom exact (jamais un nom invente) :\n\n"
        f"{schema}\n\n"
        f"{bloc_historique}"
        f"Question : {question}\n\n"
        "Si la question necessite de croiser plusieurs tables, utilise UNIQUEMENT les colonnes "
        "listees comme jointures CONFIRMEES par le dictionnaire de donnees de l'observatoire pour "
        "faire un JOIN. Les colonnes listees a part comme NON confirmees (un nom qui ressemble a un "
        "identifiant, ex: \"id\", \"menage_id\", \"round_id\", \"enquete_id\", sans etre documente "
        "comme une vraie cle) sont INTERDITES en jointure, MEME en dernier recours, MEME si aucune "
        "colonne confirmee ne permet de repondre a la question - un nom de colonne evocateur n'est "
        "JAMAIS une preuve suffisante dans ce schema, et fabriquer un JOIN dessus produirait un "
        "resultat qui a l'air precis mais qui ne repond a rien de reel (deux compteurs locaux sans "
        "lien, pas une vraie relation entre les tables). Si croiser les tables necessaires a la "
        "question demanderait un JOIN sur une colonne non confirmee, ne fabrique PAS cette requete : "
        "reponds exactement AUCUNE plutot que de deviner. "
        "Limite le resultat a 200 lignes (LIMIT 200) sauf pour un simple COUNT/AVG/SUM/MIN/MAX qui "
        "renvoie une seule ligne. Reponds UNIQUEMENT avec la requete SQL brute, sans backticks, sans "
        "explication, sans point-virgule final. Si la question ne peut pas etre traduite en requete "
        "SQL exploitable avec ce schema (aucune table/colonne pertinente pour y repondre, ou seul un "
        "JOIN non confirme le permettrait), reponds exactement : AUCUNE"
    )
    if contexte_dictionnaire:
        prompt += (
            "\n\nExtraits du dictionnaire de donnees/manuels/fiches de l'observatoire, pertinents "
            "pour cette question (utilise-les pour bien choisir les colonnes de jointure et le sens "
            "des identifiants, ex: quel identifiant relie quelles tables) :\n"
            f"{contexte_dictionnaire}"
        )
    if tentative_precedente and erreur_precedente:
        prompt += (
            "\n\nUne premiere tentative a echoue a l'execution :\n"
            f"Requete : {tentative_precedente}\n"
            f"Erreur : {erreur_precedente}\n"
            "Corrige la requete en tenant compte de cette erreur (ex: nom de colonne/table exact, "
            "syntaxe de jointure), toujours en te limitant strictement aux tables/colonnes listees."
        )

    try:
        reponse = call_llm(prompt, groq_key=groq_key, anthropic_key=anthropic_key).strip()
    except Exception:
        return None

    reponse = re.sub(r"^```(?:sql)?\s*|\s*```\s*$", "", reponse, flags=re.IGNORECASE).strip()
    if not reponse or reponse.upper() == "AUCUNE":
        return None
    return reponse


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
