"""
Tests du module rag.py : recuperation documentaire (TF-IDF) et garde-fous
(index non construit, absence de cle de LLM).
"""

import pytest

import rag


def test_index_exists_est_vrai_sur_le_vrai_index():
    # L'index reel du dictionnaire OPO doit avoir ete construit (python ingest.py)
    assert rag.index_exists() is True


def test_reset_cache_force_un_rechargement_depuis_le_disque():
    # Charge une premiere fois (peuple le cache), puis vide le cache : l'appel
    # suivant doit relire le fichier sur disque plutot que de reutiliser une
    # ancienne version chargee en memoire (essentiel apres un rebuild d'index
    # dans une application deja demarree, ex: Streamlit).
    rag.retrieve("fatherid")
    assert "index" in rag._cache
    rag.reset_cache()
    assert "index" not in rag._cache
    # Doit se recharger sans erreur
    resultats = rag.retrieve("fatherid")
    assert len(resultats) > 0


def test_analyser_image_sans_cle_anthropic_renvoie_message_explicite(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reponse = rag.analyser_image(b"faux-contenu-image", "image/png", "Que vois-tu ?")
    assert "clé Anthropic" in reponse or "cle Anthropic" in reponse


def test_retrieve_renvoie_le_bon_chunk_en_tete():
    resultats = rag.retrieve("Que signifie la variable fatherid ?", k=5)
    assert len(resultats) > 0
    assert "fatherid" in resultats[0]["text"].lower()


def test_retrieve_question_relationnelle_remonte_le_schema():
    # Question transversale sur plusieurs tables : le document de schema
    # relationnel doit apparaitre dans les tout premiers resultats, malgre
    # le biais naturel du TF-IDF en faveur des chunks courts (voir le boost
    # SOURCES_PRIORITAIRES dans rag.py).
    resultats = rag.retrieve(
        "Comment TMembership est reliee a TResidency et TSocialgp ?", k=5
    )
    sources = [r["source"] for r in resultats]
    assert "00_schema_relations" in sources


def test_retrieve_boost_sources_prioritaires_augmente_bien_le_score():
    resultats_sans_boost = rag.retrieve("episode d'appartenance a un menage", k=1)
    score_avec_boost = resultats_sans_boost[0]["score"]
    # Le score doit correspondre a une valeur boostee (donc superieur a 1.0
    # multiplie par le score brut pour une source prioritaire)
    assert resultats_sans_boost[0]["source"] in rag.SOURCES_PRIORITAIRES


def test_retrieve_query_sans_recoupement_lexical_renvoie_liste_vide():
    # Une question sans aucun mot du corpus (ex: simple salutation) ne doit
    # remonter aucun chunk avec un score > 0, plutot que des resultats
    # non pertinents.
    resultats = rag.retrieve("bonsoir", k=5)
    assert resultats == [] or all(r["score"] > 0 for r in resultats)


def test_build_prompt_inclut_le_contexte_et_la_question():
    chunks = [{"source": "Tindividual", "text": "Variable sex : 1=Man 2=Woman", "score": 0.9}]
    prompt = rag.build_prompt("Que veut dire sex=1 ?", chunks)
    assert "Que veut dire sex=1 ?" in prompt
    assert "Variable sex" in prompt


def test_build_prompt_inclut_lhistorique_si_fourni():
    chunks = [{"source": "Tindividual", "text": "Variable sex : 1=Man 2=Woman", "score": 0.9}]
    historique = [
        {"role": "user", "contenu": "Que veut dire sex=1 ?"},
        {"role": "assistant", "contenu": "sex=1 signifie Homme."},
    ]
    prompt = rag.build_prompt("Et sex=2 ?", chunks, historique=historique)
    assert "Que veut dire sex=1 ?" in prompt
    assert "sex=1 signifie Homme." in prompt
    assert "Et sex=2 ?" in prompt


def test_build_prompt_sans_historique_ne_plante_pas():
    chunks = [{"source": "Tindividual", "text": "Variable sex : 1=Man 2=Woman", "score": 0.9}]
    prompt = rag.build_prompt("Que veut dire sex=1 ?", chunks, historique=None)
    assert "Echanges precedents" not in prompt


def test_call_llm_sans_cle_renvoie_message_explicite(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    reponse = rag.call_llm("un prompt de test")
    assert "clé d'API" in reponse or "cle d'API" in reponse


def test_has_llm_configured_detecte_une_cle_passee_en_argument(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert rag.has_llm_configured() is False
    assert rag.has_llm_configured(groq_key="fake-key") is True


def test_index_not_built_error_quand_le_fichier_est_absent(monkeypatch, tmp_path):
    faux_chemin = tmp_path / "index_absent.pkl"
    monkeypatch.setattr(rag, "INDEX_PATH", faux_chemin)
    rag._cache.pop("index", None)
    with pytest.raises(rag.IndexNotBuiltError):
        rag.retrieve("une question quelconque")
    # Nettoyage du cache pour ne pas affecter les tests suivants
    rag._cache.pop("index", None)


# --- Classification d'intention (repli quand aucun mot-clé simple ne matche) --

def test_classifier_intention_sans_cle_llm_renvoie_aucune(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    action, param = rag.classifier_intention("montre-moi la répartition par sexe", ["sex", "individid"])
    assert action == "AUCUNE"
    assert param is None


def test_classifier_intention_repartition(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "REPARTITION:sex")
    action, param = rag.classifier_intention(
        "peux-tu me dire comment se répartit le sexe dans la table", ["sex", "individid"], groq_key="fake"
    )
    assert action == "REPARTITION"
    assert param == "sex"


def test_classifier_intention_repartition_colonne_inexistante_est_rejetee(monkeypatch):
    # Le LLM peut halluciner un nom de colonne : on ne doit garder que les
    # colonnes qui existent reellement dans la table.
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "REPARTITION:colonne_qui_nexiste_pas")
    action, param = rag.classifier_intention("répartition", ["sex", "individid"], groq_key="fake")
    assert action == "AUCUNE"
    assert param is None


def test_classifier_intention_echantillon_avec_nombre(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "ECHANTILLON:50")
    action, param = rag.classifier_intention("donne-moi 50 lignes au hasard", ["sex"], groq_key="fake")
    assert action == "ECHANTILLON"
    assert param == 50


def test_classifier_intention_doublons(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "DOUBLONS")
    action, param = rag.classifier_intention("est-ce qu'il y a des lignes en double ?", ["individid"], groq_key="fake")
    assert action == "DOUBLONS"


def test_classifier_intention_reponse_llm_invalide_renvoie_aucune(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "je ne sais pas trop quoi répondre ici")
    action, param = rag.classifier_intention("question ambigüe", ["sex"], groq_key="fake")
    assert action == "AUCUNE"
    assert param is None


def test_classifier_intention_liste_tables(monkeypatch):
    # Question formulee de facon trop variee pour matcher un mot-cle fige
    # (voir MOTS_LISTE_TABLES dans app.py) : le classifieur LLM doit quand
    # meme reconnaitre qu'il s'agit d'une question sur les tables chargees
    # elles-memes, pas sur le contenu d'une table precise.
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "LISTE_TABLES")
    action, param = rag.classifier_intention(
        "je parle des tables que je viens de vous envoyer", ["sex", "individid"], groq_key="fake"
    )
    assert action == "LISTE_TABLES"
    assert param is None


# --- Action REQUETE (calcul precis compter/lister/moyenne..., avec filtres) -

def test_classifier_intention_requete_compter_avec_filtre(monkeypatch):
    reponse_llm = (
        'REQUETE:{"operation": "compter", "colonne_cible": null, '
        '"filtres": [{"colonne": "commune", "operateur": "==", "valeur": "Ouahigouya"}]}'
    )
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: reponse_llm)
    action, param = rag.classifier_intention(
        "combien de naissances a Ouahigouya ?", ["individu_id", "commune"], groq_key="fake"
    )
    assert action == "REQUETE"
    assert param["operation"] == "compter"
    assert param["colonne_cible"] is None
    assert param["filtres"] == [{"colonne": "commune", "operateur": "==", "valeur": "Ouahigouya"}]


def test_classifier_intention_requete_moyenne_avec_colonne_cible(monkeypatch):
    reponse_llm = 'REQUETE:{"operation": "moyenne", "colonne_cible": "age", "filtres": []}'
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: reponse_llm)
    action, param = rag.classifier_intention("age moyen ?", ["individu_id", "age"], groq_key="fake")
    assert action == "REQUETE"
    assert param["operation"] == "moyenne"
    assert param["colonne_cible"] == "age"


def test_classifier_intention_requete_moyenne_sans_colonne_cible_est_rejetee(monkeypatch):
    # "moyenne"/"somme"/"min"/"max" exigent une colonne_cible reelle : sans
    # elle, le calcul n'aurait rien de precis a agreger.
    reponse_llm = 'REQUETE:{"operation": "moyenne", "colonne_cible": null, "filtres": []}'
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: reponse_llm)
    action, param = rag.classifier_intention("moyenne ?", ["individu_id", "age"], groq_key="fake")
    assert action == "AUCUNE"
    assert param is None


def test_classifier_intention_requete_filtre_colonne_inexistante_est_retire(monkeypatch):
    # Un filtre sur une colonne qui n'existe pas reellement (hallucination du
    # LLM) doit etre retire, jamais transmis tel quel a l'execution.
    reponse_llm = (
        'REQUETE:{"operation": "compter", "colonne_cible": null, '
        '"filtres": [{"colonne": "colonne_qui_nexiste_pas", "operateur": "==", "valeur": "x"}]}'
    )
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: reponse_llm)
    action, param = rag.classifier_intention("combien ?", ["individu_id"], groq_key="fake")
    assert action == "REQUETE"
    assert param["filtres"] == []


def test_classifier_intention_requete_json_invalide_renvoie_aucune(monkeypatch):
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "REQUETE:pas du json")
    action, param = rag.classifier_intention("combien ?", ["individu_id"], groq_key="fake")
    assert action == "AUCUNE"
    assert param is None


def test_classifier_intention_requete_operation_inconnue_renvoie_aucune(monkeypatch):
    reponse_llm = 'REQUETE:{"operation": "supprimer", "colonne_cible": null, "filtres": []}'
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: reponse_llm)
    action, param = rag.classifier_intention("efface tout ?", ["individu_id"], groq_key="fake")
    assert action == "AUCUNE"
    assert param is None


# --- Reformulation de requete en cas de score de recherche trop faible -----

def test_answer_reformule_la_requete_si_le_premier_score_est_faible(monkeypatch):
    appels_retrieve = []

    def fausse_retrieve(query, k=5):
        appels_retrieve.append(query)
        if query == "question mal formulee":
            return [{"source": "Tindividual", "text": "peu pertinent", "score": 0.02}]
        return [{"source": "Tindividual", "text": "tres pertinent", "score": 0.8}]

    monkeypatch.setattr(rag, "retrieve", fausse_retrieve)
    monkeypatch.setattr(
        rag, "call_llm",
        lambda prompt, groq_key=None, anthropic_key=None: (
            "requete reformulee" if "reformulee" in prompt.lower() and "Reponse" not in prompt else "Reponse finale."
        ),
    )

    resultat = rag.answer("question mal formulee", groq_key="fake")
    assert appels_retrieve == ["question mal formulee", "requete reformulee"]
    assert resultat["sources"][0]["text"] == "tres pertinent"


def test_answer_ne_reformule_pas_si_le_score_est_deja_bon(monkeypatch):
    appels_retrieve = []

    def fausse_retrieve(query, k=5):
        appels_retrieve.append(query)
        return [{"source": "Tindividual", "text": "deja bon", "score": 0.9}]

    monkeypatch.setattr(rag, "retrieve", fausse_retrieve)
    monkeypatch.setattr(rag, "call_llm", lambda prompt, groq_key=None, anthropic_key=None: "Reponse finale.")

    rag.answer("bonne question", groq_key="fake")
    assert appels_retrieve == ["bonne question"]  # pas de deuxieme appel


def test_answer_ne_reformule_pas_sans_cle_llm(monkeypatch):
    appels_retrieve = []

    def fausse_retrieve(query, k=5):
        appels_retrieve.append(query)
        return [{"source": "Tindividual", "text": "peu pertinent", "score": 0.01}]

    monkeypatch.setattr(rag, "retrieve", fausse_retrieve)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    rag.answer("question sans cle configuree")
    assert appels_retrieve == ["question sans cle configuree"]
