"""
Tests bout-en-bout de app.py via AppTest (streamlit.testing.v1) : simule de
vraies conversations multi-tours dans l'interface, sans navigateur. Complete
les tests unitaires de data_tools.py/rag.py en verifiant que le routage des
questions (route_question) se comporte correctement une fois branche a une
vraie session Streamlit (tables chargees, historique de messages...).
"""

import re

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def _app_avec_tables(tables: dict) -> AppTest:
    at = AppTest.from_file("app.py", default_timeout=60)
    # Simule une session deja authentifiee (voir auth.verifier_acces : le
    # formulaire de connexion n'est pas rejoue si authentication_status vaut
    # deja True), pour tester le routage des questions sans avoir a remplir
    # le formulaire de connexion a chaque test.
    at.session_state["authentication_status"] = True
    at.session_state["username"] = "test_utilisateur"
    at.session_state["name"] = "Testeur"
    at.run()
    at.session_state["tables"] = tables
    at.session_state["messages"] = []
    return at


@pytest.fixture
def tables_education_presence():
    return {
        "FNewEducation": pd.DataFrame({"individid": [1, 2, 3, 4, 5, 6]}),
        "FNewPresences": pd.DataFrame({"individid": [2, 3, 6]}),
    }


def test_difference_reconnait_les_noms_informels_de_table(tables_education_presence):
    # L'equipe dit "education"/"presences", pas les noms techniques complets
    # "FNewEducation"/"FNewPresences" - doit quand meme calculer la vraie
    # difference plutot que de demander de preciser les tables.
    at = _app_avec_tables(tables_education_presence)
    at.chat_input[0].set_value(
        "combien d'individus sont dans education et pas dans presences"
    ).run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "3" in reponse
    assert "FNewEducation" in reponse
    assert "FNewPresences" in reponse
    assert "Précise les deux tables" not in reponse


def test_difference_resout_seule_avec_exactement_deux_tables_chargees(tables_education_presence):
    # Une seule table est nommee, mais il n'y en a que deux au total : l'autre
    # est evidente, pas besoin de redemander en repetant la phrase standard.
    at = _app_avec_tables(tables_education_presence)
    at.chat_input[0].set_value(
        "combien sont dans FNewEducation mais pas dans l'autre table"
    ).run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewPresences" in reponse
    assert "Tu veux comparer" not in reponse


def test_difference_relance_dynamique_avec_une_seule_table_nommee():
    # Avec 3+ tables chargees et une seule nommee, la relance doit lister les
    # VRAIES autres tables chargees (pas un exemple fige type Tindividual).
    tables = {
        "FNewBase_HistMig": pd.DataFrame({"individid": [1, 2]}),
        "FNewBase_HistMat": pd.DataFrame({"individid": [1]}),
        "FNewEducation": pd.DataFrame({"individid": [1]}),
    }
    at = _app_avec_tables(tables)
    at.chat_input[0].set_value("combien sont dans FNewBase_HistMig mais pas dans l'autre").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Tu veux comparer" in reponse
    assert "FNewBase_HistMig" in reponse
    assert "FNewBase_HistMat" in reponse
    assert "FNewEducation" in reponse


def test_difference_relance_dynamique_sans_aucune_table_nommee():
    # Sans aucune table nommee, la liste complete des vraies tables chargees
    # doit apparaitre dans la relance, jamais un exemple fige.
    tables = {
        "FNewBase_HistMig": pd.DataFrame({"individid": [1, 2]}),
        "FNewBase_HistMat": pd.DataFrame({"individid": [1]}),
        "FNewEducation": pd.DataFrame({"individid": [1]}),
    }
    at = _app_avec_tables(tables)
    at.chat_input[0].set_value("combien sont dans une table mais pas dans l'autre").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Précise les deux tables" in reponse
    assert "FNewBase_HistMig" in reponse
    assert "FNewBase_HistMat" in reponse
    assert "FNewEducation" in reponse
    assert "Presence" not in reponse and "Tindividual" not in reponse


def test_relance_courte_redeclenche_le_calcul_de_difference(tables_education_presence):
    # Reproduit le bug signale : apres une reponse (hesitante ou non), une
    # relance courte qui ne repete pas le mot-cle de difference ("il faut
    # analyser directement") doit quand meme redeclencher le vrai calcul
    # deterministe, pas tomber sur une reponse generique du LLM qui n'a pas
    # d'environnement d'execution.
    at = _app_avec_tables(tables_education_presence)
    at.chat_input[0].set_value(
        "combien d'individus sont dans education et pas dans presences"
    ).run()
    at.chat_input[0].set_value("il faut analyser directement").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "3" in reponse
    assert "n'ont pas de correspondance" in reponse
    assert "environnement d'exécution" not in reponse.lower()


def test_relance_sans_historique_pertinent_ne_declenche_rien_a_tort(tables_education_presence):
    # Garde-fou : si aucune question de difference/fusion/relation n'a ete
    # posee avant, un simple "merci, c'est parfait" ne doit pas etre pris a
    # tort pour une relance de calcul.
    at = _app_avec_tables(tables_education_presence)
    at.chat_input[0].set_value("bonjour").run()
    at.chat_input[0].set_value("merci, c'est parfait, direct et clair").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "n'ont pas de correspondance" not in reponse


@pytest.fixture
def tables_individual():
    return {
        "FNewIndividual": pd.DataFrame({
            "individid": [1, 2, 3, 4, 4],
            "sex": [1, 2, 1, 2, 2],
            "birth_date": ["1990-01-01", "2005-05-05", "1899-01-01", "2027-01-01", "2010-01-01"],
        }),
    }


def test_repartition_inclut_la_syntaxe_r_et_stata(tables_individual):
    at = _app_avec_tables(tables_individual)
    at.chat_input[0].set_value("répartition de sex").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "```r" in reponse and "```stata" in reponse
    assert "tab sex" in reponse


def test_doublons_inclut_la_syntaxe_r_et_stata(tables_individual):
    at = _app_avec_tables(tables_individual)
    at.chat_input[0].set_value("doublons dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "```r" in reponse and "```stata" in reponse
    assert "duplicates tag individid" in reponse


def test_echantillon_inclut_la_syntaxe_r_et_stata(tables_individual):
    at = _app_avec_tables(tables_individual)
    at.chat_input[0].set_value("échantillon de 2 dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "```r" in reponse and "```stata" in reponse
    assert "sample 2, count" in reponse


def test_coherence_inclut_la_syntaxe_r_et_stata(tables_individual):
    at = _app_avec_tables(tables_individual)
    at.chat_input[0].set_value("cohérence de FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "```r" in reponse and "```stata" in reponse
    assert "duplicates report individid" in reponse


@pytest.fixture
def tables_analyse_complete():
    return {
        "FNewIndividual": pd.DataFrame({
            "individid": [1, 2, 3, 4, 4, 5, 6],
            "sex": [1, 2, 1, 2, 2, 1, 2],
            "education_level": [
                "primaire", "secondaire", "primaire", "superieur", "superieur", "secondaire", "primaire",
            ],
            "birth_date": [
                "1990-01-01", "2005-05-05", "1899-01-01", "2027-01-01", "2010-01-01", "2000-01-01", "1995-01-01",
            ],
            "field_wrkr": ["A", "B", "A", "A", "B", "B", "A"],
        }),
    }


def test_tableau_croise_bivarie(tables_analyse_complete):
    at = _app_avec_tables(tables_analyse_complete)
    at.chat_input[0].set_value("tableau croisé entre sex et education_level dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Tableau croisé" in reponse
    assert "```r" in reponse and "```stata" in reponse
    assert "tab sex education_level" in reponse


def test_correlation_ne_declenche_pas_la_relation_entre_tables(tables_analyse_complete):
    # "corrélation" contient "relation" comme sous-chaine : ne doit jamais
    # etre confondu avec une question de relation ENTRE TABLES.
    at = _app_avec_tables(tables_analyse_complete)
    at.chat_input[0].set_value("corrélation entre individid et sex dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Matrice de corrélation" in reponse
    assert "Il faut au moins deux tables" not in reponse


def test_analyse_multivariee_sur_trois_colonnes(tables_analyse_complete):
    at = _app_avec_tables(tables_analyse_complete)
    at.chat_input[0].set_value(
        "analyse multivariée de sex, education_level et field_wrkr dans FNewIndividual"
    ).run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Analyse multivariée" in reponse
    assert "```r" in reponse and "```stata" in reponse


def test_relation_entre_tables_toujours_reconnue_avec_correspondance_entiere(tables_education_presence):
    at = _app_avec_tables(tables_education_presence)
    at.chat_input[0].set_value("quelle est la relation entre FNewEducation et FNewPresences").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "colonne(s)" in reponse or "Aucune colonne commune" in reponse


def test_performance_agents_enqueteurs(tables_analyse_complete):
    at = _app_avec_tables(tables_analyse_complete)
    at.chat_input[0].set_value("quelle est la performance des agents enquêteurs dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Rapport de performance par agent" in reponse
    assert "n_fiches" in reponse
    assert "```r" in reponse and "```stata" in reponse


def test_mention_documentaire_dun_agent_ne_declenche_pas_a_tort(tables_analyse_complete):
    # Garde-fou : le mot "agent" seul (sans mot de performance/qualite) ne
    # doit pas declencher a tort le rapport par agent enqueteur.
    at = _app_avec_tables(tables_analyse_complete)
    at.chat_input[0].set_value("répartition de sex dans FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Rapport de performance par agent" not in reponse


def test_colonne_ambigue_est_calculee_pour_toutes_les_tables_concernees():
    # "il faut tout lire, pas une seule base par defaut" : si `sex` existe
    # dans deux tables sans qu'aucune ne soit nommee, les deux doivent
    # apparaitre dans la reponse plutot qu'une seule choisie silencieusement.
    tables = {
        "FNewEducation": pd.DataFrame({"individid": [1, 2, 3], "sex": [1, 2, 1]}),
        "FNewEmploi": pd.DataFrame({"individid": [1, 2], "sex": [2, 2]}),
        "FNewSante": pd.DataFrame({"individid": [1], "autre_col": ["x"]}),
    }
    at = _app_avec_tables(tables)
    at.chat_input[0].set_value("répartition de sex").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "FNewEmploi" in reponse
    assert "FNewSante" not in reponse


@pytest.fixture
def tables_audit_complet():
    presence = pd.DataFrame({
        "individid": [1, 2, 3, 4, 5],
        "sleep_lastnight": [1, 1, 0, 1, 1],
        "arrive_date": ["2024-01-01"] * 5,
        "depart_date": [None, None, None, "2024-01-01", None],
    })
    education = pd.DataFrame({"individid": [1, 4, 6]})
    individual = pd.DataFrame({
        "individid": ["1", "2", "33", "4", "5"],
        "fatherid": [100, 101, 102, 103, None],
        "motherid": [100, 111, None, None, None],
        "birth_date": ["2023-01-01", "1990-01-01", "1985-01-01", "2010-01-01", "1995-01-01"],
    })
    deces = pd.DataFrame({"individid": [3, 999]})
    return {
        "FNewPresences": presence, "FNewEducation": education,
        "FNewIndividual": individual, "FNewDeath": deces,
    }


def test_audit_complet_couvre_toutes_les_tables_et_les_controles_croises(tables_audit_complet):
    at = _app_avec_tables(tables_audit_complet)
    at.chat_input[0].set_value("audit complet de cohérence").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Audit de cohérence avancé" in reponse
    assert "FNewPresences" in reponse and "FNewIndividual" in reponse
    assert "Éligibilité présence" in reponse
    assert "Décédé mais présent" in reponse
    assert "Rappel" in reponse


def test_audit_complet_peut_etre_cible_sur_une_seule_table(tables_audit_complet):
    at = _app_avec_tables(tables_audit_complet)
    at.chat_input[0].set_value("audit complet de FNewIndividual").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewIndividual" in reponse
    assert "FNewDeath" not in reponse
    assert "Contrôles croisés" not in reponse


@pytest.fixture
def tables_performance():
    presence = pd.DataFrame({
        "individid": [1, 2, 3, 4, 5, 6, 7, 8],
        "menageid": [10, 10, 11, 12, 12, 13, 14, 14],
        "field_wrkr": ["A1", "A1", "A1", "A2", "A2", "A2", "A3", "A3"],
        "visit_date": [
            "01/01/2026", "01/01/2026", "02/01/2026",
            "01/01/2026", "02/01/2026", "02/01/2026",
            "03/01/2026", "03/01/2026",
        ],
    })
    naissance = pd.DataFrame({"individid": [1, 2, 3], "field_wrkr": ["A1", "A2", "A1"]})
    equipe = pd.DataFrame({"field_wrkr": ["A1", "A2", "A3"], "controleur": ["C1", "C1", "C2"]})
    return {"FNewPresences": presence, "FNewBirth": naissance, "Equipe": equipe}


def test_performance_terrain_donne_le_rapport_par_agent_et_la_projection(tables_performance):
    at = _app_avec_tables(tables_performance)
    at.chat_input[0].set_value("bilan de terrain : combien de ménages visités par agent ? objectif 20").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Rapport de performance de terrain" in reponse
    assert "A1" in reponse and "A2" in reponse and "A3" in reponse
    assert "objectif atteint vers le" in reponse or "Avancement vers l'objectif" in reponse


def test_performance_terrain_ne_collide_pas_avec_le_controle_qualite_des_agents(tables_performance):
    # "performance des agents enqueteurs" (controle qualite existant, Task
    # #41) doit continuer a fonctionner sans etre intercepte par le nouveau
    # module de volume de terrain, meme si les deux partagent le mot
    # "performance".
    at = _app_avec_tables(tables_performance)
    at.chat_input[0].set_value("quelle est la performance des agents enquêteurs ?").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Rapport de performance par agent" in reponse
    assert "Rapport de performance de terrain" not in reponse


def test_performance_terrain_exclut_les_agents_demandes(tables_performance):
    at = _app_avec_tables(tables_performance)
    at.chat_input[0].set_value("bilan de terrain en excluant les agents A3").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "exclusion de 1 agent" in reponse
    assert "A1" in reponse and "A2" in reponse
    assert not re.search(r"\|\s*A3\s*\|", reponse)


def test_historique_des_actualisations_liste_les_tables_chargees(tables_performance):
    at = _app_avec_tables(tables_performance)
    # _app_avec_tables charge les tables directement en session_state (pas via
    # l'uploader), donc l'historique de chargement est vide : on le peuple
    # nous-memes pour verifier le formatage de la reponse.
    from datetime import datetime
    at.session_state["historique_chargements"] = [
        {"horodatage": datetime(2026, 1, 1, 8, 30), "fichier": "presence.csv", "table": "FNewPresences", "n_lignes": 8},
    ]
    at.chat_input[0].set_value("historique des actualisations").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewPresences" in reponse
    assert "01/01/2026" in reponse


def test_recherche_instantanee_par_identifiant(tables_performance):
    at = _app_avec_tables(tables_performance)
    at.chat_input[0].set_value("recherche l'individu 3").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewPresences" in reponse
    assert "FNewBirth" in reponse


def test_recherche_instantanee_identifiant_introuvable(tables_performance):
    at = _app_avec_tables(tables_performance)
    at.chat_input[0].set_value("recherche l'individu 999999").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "Aucune fiche trouvée" in reponse


@pytest.fixture
def tables_colonnes_partagees():
    return {
        "FNewEducation": pd.DataFrame({
            "individid": [1, 2, 3, 4], "sex": [1, 2, 1, 2], "education_level": ["primaire", "secondaire", "primaire", "aucun"],
        }),
        "FNewEmploi": pd.DataFrame({
            "individid": [1, 2, 3], "sex": [2, 2, 1], "education_level": ["primaire", "primaire", "secondaire"],
        }),
        "FNewSante": pd.DataFrame({"individid": [1], "autre_col": ["x"]}),
    }


def test_tableau_croise_sans_nommer_de_table_couvre_toutes_les_tables_concernees(tables_colonnes_partagees):
    # "il ne faut pas lire une seule base par defaut" : sex et
    # education_level existent dans DEUX tables sans qu'aucune ne soit
    # nommee -> les deux doivent apparaitre, pas seulement la table par
    # defaut de la barre laterale (la premiere chargee).
    at = _app_avec_tables(tables_colonnes_partagees)
    at.chat_input[0].set_value("tableau croisé entre sex et education_level").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "FNewEmploi" in reponse
    assert "FNewSante" not in reponse


def test_correlation_sans_nommer_de_table_couvre_toutes_les_tables_concernees():
    tables = {
        "FNewIndividual": pd.DataFrame({"individid": [1, 2, 3], "age": [20, 30, 40], "poids": [60, 70, 80]}),
        "FNewSante": pd.DataFrame({"individid": [1, 2], "age": [20, 30], "poids": [60, 70]}),
        "FNewEducation": pd.DataFrame({"individid": [1], "autre_col": ["x"]}),
    }
    at = _app_avec_tables(tables)
    at.chat_input[0].set_value("corrélation entre age et poids").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewIndividual" in reponse
    assert "FNewSante" in reponse
    assert "FNewEducation" not in reponse


# --- Plus de "table par defaut" : toutes les tables travaillent au depart ---

@pytest.fixture
def tables_ambigues():
    return {
        "FNewEducation": pd.DataFrame({"individid": [1, 1, 2], "niveau": ["primaire", "primaire", "secondaire"]}),
        "FNewEmploi": pd.DataFrame({"individid": [10, 10, 11], "secteur": ["informel", "informel", "formel"]}),
        "FNewSante": pd.DataFrame({"individid": [20, 21], "etat": ["bon", "mauvais"]}),
    }


def test_doublons_sans_rien_nommer_couvre_toutes_les_tables(tables_ambigues):
    # Aucune table nommee, aucune colonne mentionnee, aucun historique : avant,
    # ceci retombait sur la table "par defaut" de la barre laterale (retiree
    # sur demande) - maintenant, ca doit calculer sur TOUTES les tables.
    at = _app_avec_tables(tables_ambigues)
    at.chat_input[0].set_value("il y a des doublons ?").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "FNewEmploi" in reponse
    assert "FNewSante" in reponse


def test_coherence_sans_rien_nommer_couvre_toutes_les_tables(tables_ambigues):
    at = _app_avec_tables(tables_ambigues)
    at.chat_input[0].set_value("vérifie la cohérence").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "FNewEmploi" in reponse
    assert "FNewSante" in reponse


def test_echantillon_sans_rien_nommer_couvre_toutes_les_tables(tables_ambigues):
    at = _app_avec_tables(tables_ambigues)
    at.chat_input[0].set_value("donne-moi un échantillon").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "FNewEmploi" in reponse
    assert "FNewSante" in reponse


def test_repartition_sans_rien_nommer_demande_de_preciser_avec_les_vraies_tables(tables_ambigues):
    # Aucune colonne mentionnee nulle part : impossible de calculer quoi que
    # ce soit sans deviner - doit demander de preciser, avec les VRAIES
    # tables/colonnes chargees (jamais un exemple generique).
    at = _app_avec_tables(tables_ambigues)
    at.chat_input[0].set_value("donne-moi la répartition").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewEducation" in reponse
    assert "niveau" in reponse
    assert "FNewEmploi" in reponse
    assert "secteur" in reponse


def test_table_specifique_encore_ciblable_directement(tables_ambigues):
    # Nommer explicitement une table doit toujours fonctionner comme avant -
    # la suppression de la table par defaut ne doit pas empecher de cibler
    # UNE table precise quand on le demande.
    at = _app_avec_tables(tables_ambigues)
    at.chat_input[0].set_value("doublons dans FNewSante").run()
    reponse = at.session_state["messages"][-1]["content"]
    assert "FNewSante" in reponse
    assert "FNewEducation" not in reponse
    assert "FNewEmploi" not in reponse
