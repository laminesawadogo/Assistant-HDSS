"""
Tests du module data_tools.py : analyse generique de tables (quel que soit
le schema reel), suppression des colonnes nominatives, detection de
doublons et de dates invraisemblables.
"""

import pandas as pd
import pytest

import data_tools as dt


@pytest.fixture
def table_exemple(tmp_path):
    """Table synthetique avec une colonne 'nom' (a retirer), un doublon
    d'identifiant et deux dates invraisemblables (future + avant 1900)."""
    df = pd.DataFrame({
        "individid": [1, 2, 3, 4, 4],
        "nom": ["Diallo", "Ouedraogo", "Kabore", "Sawadogo", "Sawadogo"],
        "sex": [1, 2, 1, 2, 2],
        "birth_date": ["1990-01-01", "2005-05-05", "1899-01-01", "2027-01-01", "2010-01-01"],
        "entry_date": ["2020-01-01"] * 5,
    })
    path = tmp_path / "table_exemple.csv"
    df.to_csv(path, index=False)
    return str(path)


# --- Import Stata (.dta) -----------------------------------------------------

def test_load_table_lit_un_fichier_stata(tmp_path):
    path = tmp_path / "table_exemple.dta"
    df_origine = pd.DataFrame({"individid": [1, 2, 3], "sex": [1, 2, 1], "nom": ["A", "B", "C"]})
    df_origine.to_stata(path, write_index=False)

    df = dt.load_table(str(path))
    assert list(df["individid"]) == [1, 2, 3]
    assert "nom" not in df.columns  # colonne nominative retiree comme pour csv/xlsx


# --- Recherche large sur toutes les tables (colonne ambigue) ----------------

def test_tables_avec_colonne_renvoie_toutes_les_correspondances():
    tables = {
        "FNewEducation": pd.DataFrame({"individid": [1], "sex": [1]}),
        "FNewEmploi": pd.DataFrame({"individid": [1], "sex": [2]}),
        "FNewSante": pd.DataFrame({"individid": [1], "autre_col": ["x"]}),
    }
    trouvees = dt.tables_avec_colonne("répartition de sex", tables)
    assert set(trouvees) == {"FNewEducation", "FNewEmploi"}


def test_tables_avec_colonne_vide_si_aucune_correspondance():
    tables = {"FNewSante": pd.DataFrame({"autre_col": ["x"]})}
    assert dt.tables_avec_colonne("répartition de sex", tables) == []


# --- Relations et fusion entre tables chargees -------------------------------

def test_detecter_tables_mentionnees():
    tables = {"Tindividual": pd.DataFrame({"individid": [1]}), "Tsocialgp": pd.DataFrame({"socialgpid": [1]})}
    assert dt.detecter_tables_mentionnees("relation entre Tindividual et Tsocialgp", tables) == ["Tindividual", "Tsocialgp"]
    assert dt.detecter_tables_mentionnees("question sans nom de table", tables) == []


def test_detecter_tables_mentionnees_reconnait_les_noms_informels():
    # L'equipe omet naturellement le prefixe technique "FNew" et parfois le
    # pluriel : "education"/"presence" doivent quand meme etre reconnus comme
    # FNewEducation/FNewPresences, sinon les questions de difference/fusion
    # echouent des que les vrais noms de tables de l'observatoire sont utilises.
    tables = {
        "FNewEducation": pd.DataFrame({"individid": [1, 2]}),
        "FNewPresences": pd.DataFrame({"individid": [1]}),
    }
    trouvees = dt.detecter_tables_mentionnees(
        "combien d'individus sont dans la table education et pas dans la table presence", tables
    )
    assert set(trouvees) == {"FNewEducation", "FNewPresences"}


def test_alias_table_ignore_prefixe_technique_et_pluriel():
    assert "education" in dt.alias_table("FNewEducation")
    assert "presence" in dt.alias_table("FNewPresences")
    assert "presences" in dt.alias_table("FNewPresences")


def test_alias_table_exclut_les_fragments_trop_courts():
    # Un alias de moins de 4 caracteres serait trop generique (faux positifs
    # sur un mot sans rapport) : il est exclu plutot que renvoye tel quel.
    for alias in dt.alias_table("FNewIN"):
        assert len(alias) >= 4


def test_detecter_cles_communes_trouve_la_colonne_partagee():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1, 2], "sex": [1, 2]}),
        "TMembership": pd.DataFrame({"individid": [1, 2], "socialgpid": [10, 20]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [10, 20], "chef_menage": [1, 2]}),
    }
    communes = dt.detecter_cles_communes(tables)
    assert communes[("Tindividual", "TMembership")] == ["individid"]
    assert communes[("TMembership", "Tsocialgp")] == ["socialgpid"]
    assert ("Tindividual", "Tsocialgp") not in communes  # aucune colonne en commun


def test_relation_entre_tables_decrit_la_colonne_commune():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "TMembership": pd.DataFrame({"individid": [1], "socialgpid": [10]}),
    }
    texte = dt.relation_entre_tables("Tindividual", "TMembership", tables)
    assert "individid" in texte
    assert "Tindividual" in texte and "TMembership" in texte


def test_relation_entre_tables_sans_colonne_commune():
    tables = {
        "A": pd.DataFrame({"x": [1]}),
        "B": pd.DataFrame({"y": [1]}),
    }
    texte = dt.relation_entre_tables("A", "B", tables)
    assert "Aucune colonne commune" in texte


def test_rapport_relations_resume_toutes_les_paires():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "TMembership": pd.DataFrame({"individid": [1], "socialgpid": [10]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [10]}),
    }
    texte = dt.rapport_relations(tables)
    assert "Tindividual" in texte and "TMembership" in texte and "Tsocialgp" in texte


def test_rapport_relations_avec_une_seule_table():
    texte = dt.rapport_relations({"Tindividual": pd.DataFrame({"individid": [1]})})
    assert "au moins deux tables" in texte


# --- Resume "meta" des tables chargees (combien, lesquelles) ----------------

def test_resume_tables_chargees_liste_toutes_les_tables():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1, 2], "sex": [1, 2]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [1]}),
    }
    texte = dt.resume_tables_chargees(tables)
    assert "2 table(s) chargée(s)" in texte
    assert "Tindividual" in texte and "2 lignes" in texte
    assert "Tsocialgp" in texte and "1 ligne," in texte


def test_resume_tables_chargees_aucune_table():
    texte = dt.resume_tables_chargees({})
    assert "Aucune table" in texte


def test_fusionner_tables_sur_cle_detectee_automatiquement():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1, 2], "sex": [1, 2]}),
        "TMembership": pd.DataFrame({"individid": [1, 2], "socialgpid": [10, 20]}),
    }
    fusion = dt.fusionner_tables("Tindividual", "TMembership", tables)
    assert len(fusion) == 2
    assert "sex" in fusion.columns and "socialgpid" in fusion.columns


def test_fusionner_tables_leve_erreur_sans_colonne_commune():
    tables = {"A": pd.DataFrame({"x": [1]}), "B": pd.DataFrame({"y": [1]})}
    with pytest.raises(ValueError):
        dt.fusionner_tables("A", "B", tables)


def test_fusionner_tables_leve_erreur_si_table_introuvable():
    tables = {"A": pd.DataFrame({"x": [1]})}
    with pytest.raises(ValueError):
        dt.fusionner_tables("A", "Inconnue", tables)


# --- Difference d'ensembles (anti-jointure) ---------------------------------

def test_difference_tables_trouve_les_lignes_sans_correspondance():
    tables = {
        "Presence": pd.DataFrame({"individid": [1, 2, 3, 4]}),
        "Education": pd.DataFrame({"individid": [2, 3]}),
    }
    diff = dt.difference_tables("Presence", "Education", tables)
    assert sorted(diff["individid"]) == [1, 4]
    assert len(diff) == 2


def test_difference_tables_est_non_symetrique():
    tables = {
        "Presence": pd.DataFrame({"individid": [1, 2, 3]}),
        "Education": pd.DataFrame({"individid": [2, 3, 4, 5]}),
    }
    presence_sans_education = dt.difference_tables("Presence", "Education", tables)
    education_sans_presence = dt.difference_tables("Education", "Presence", tables)
    assert sorted(presence_sans_education["individid"]) == [1]
    assert sorted(education_sans_presence["individid"]) == [4, 5]


def test_difference_tables_leve_erreur_sans_colonne_commune():
    tables = {"A": pd.DataFrame({"x": [1]}), "B": pd.DataFrame({"y": [1]})}
    with pytest.raises(ValueError):
        dt.difference_tables("A", "B", tables)


def test_difference_tables_leve_erreur_si_table_introuvable():
    tables = {"A": pd.DataFrame({"x": [1]})}
    with pytest.raises(ValueError):
        dt.difference_tables("A", "Inconnue", tables)


# --- Detection de la cle de jointure et syntaxe R/Stata ----------------------

def test_detecter_cle_jointure_renvoie_la_colonne_commune():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "TMembership": pd.DataFrame({"individid": [1], "socialgpid": [1]}),
    }
    assert dt.detecter_cle_jointure("Tindividual", "TMembership", tables) == "individid"


def test_detecter_cle_jointure_none_si_aucune_commune():
    tables = {"A": pd.DataFrame({"x": [1]}), "B": pd.DataFrame({"y": [1]})}
    assert dt.detecter_cle_jointure("A", "B", tables) is None


# --- Cle de jointure fiable : ignorer "id" au profit d'un identifiant reel --
# Bug reel corrige ici : sur les vraies tables opo_hypervel_*, "id" (cle
# primaire LOCALE a chaque table, jamais une reference vers une autre table
# dans ce schema) est quasi toujours la premiere colonne commune detectee -
# une question "individus dont on a fait l'education mais pas dans la fiche
# presence" avait ainsi ete jointe sur `id` au lieu de `individid`, donnant
# un resultat qui a l'air precis (ex: "15630 lignes sans correspondance")
# mais qui ne repond a rien de reel.

def test_detecter_cle_jointure_ignore_id_au_profit_dun_identifiant_reel():
    tables = {
        "opo_hypervel_education": pd.DataFrame({"id": [1, 2], "individid": [10, 20]}),
        "opo_hypervel_presences": pd.DataFrame({"id": [1, 2, 3], "individid": [10, 30, 40]}),
    }
    assert dt.detecter_cle_jointure("opo_hypervel_education", "opo_hypervel_presences", tables) == "individid"


def test_detecter_cle_jointure_none_si_seule_id_est_commune():
    tables = {
        "A": pd.DataFrame({"id": [1, 2], "x": [1, 2]}),
        "B": pd.DataFrame({"id": [1, 2], "y": [1, 2]}),
    }
    assert dt.detecter_cle_jointure("A", "B", tables) is None


def test_difference_tables_utilise_individid_pas_id_sur_les_vraies_tables():
    tables = {
        "opo_hypervel_education": pd.DataFrame({"id": [1, 2, 3], "individid": [10, 20, 30]}),
        "opo_hypervel_presences": pd.DataFrame({"id": [9, 8], "individid": [10, 20]}),
    }
    diff = dt.difference_tables("opo_hypervel_education", "opo_hypervel_presences", tables)
    # Sur individid, seul individid=30 n'a pas de correspondance. Une
    # jointure erronee sur "id" aurait laisse passer/exclu des lignes au
    # hasard puisque les "id" des deux tables sont des compteurs locaux sans
    # rapport entre eux.
    assert sorted(diff["individid"]) == [30]


def test_fusionner_tables_sans_cle_choisit_individid_pas_id():
    tables = {
        "opo_hypervel_education": pd.DataFrame({"id": [1, 2], "individid": [10, 20], "note": ["a", "b"]}),
        "opo_hypervel_presences": pd.DataFrame({"id": [5, 6], "individid": [10, 20], "present": [1, 0]}),
    }
    fusion = dt.fusionner_tables("opo_hypervel_education", "opo_hypervel_presences", tables)
    assert "present" in fusion.columns and "note" in fusion.columns
    assert len(fusion) == 2


def test_fusionner_tables_leve_erreur_si_seule_id_est_commune():
    tables = {
        "A": pd.DataFrame({"id": [1, 2], "x": [1, 2]}),
        "B": pd.DataFrame({"id": [1, 2], "y": [1, 2]}),
    }
    with pytest.raises(ValueError):
        dt.fusionner_tables("A", "B", tables)


def test_relation_entre_tables_recommande_individid_pas_id():
    tables = {
        "opo_hypervel_education": pd.DataFrame({"id": [1, 2], "individid": [10, 20]}),
        "opo_hypervel_presences": pd.DataFrame({"id": [5, 6], "individid": [10, 20]}),
    }
    texte = dt.relation_entre_tables("opo_hypervel_education", "opo_hypervel_presences", tables)
    assert "`individid`" in texte
    assert "candidate la plus probable comme clé de jointure" in texte
    assert "identifiant confirmé" in texte


def test_relation_entre_tables_avertit_si_seule_id_est_commune():
    tables = {
        "A": pd.DataFrame({"id": [1, 2], "x": [1, 2]}),
        "B": pd.DataFrame({"id": [1, 2], "y": [1, 2]}),
    }
    texte = dt.relation_entre_tables("A", "B", tables)
    assert "Aucune de ces colonnes communes n'est fiable" in texte


# --- Ne pas confondre "individus" mot generique et table opo_hypervel_individus
# Bug reel corrige ici : la question "les individus dont on a fait l'education
# ne sont pas dans la fiche presence" mentionne "individus" au sens general
# (les gens), pas la table opo_hypervel_individus - mais une correspondance
# nue sur ce mot faisait quand meme compter cette table comme "mentionnee",
# polluant la resolution automatique de la paire de tables visee par la
# question (qui retenait alors education+individus au lieu d'education+presences).

def test_detecter_tables_mentionnees_ignore_individus_utilise_comme_mot_courant():
    tables = {
        "opo_hypervel_individus": pd.DataFrame({"individid": [1]}),
        "opo_hypervel_education": pd.DataFrame({"individid": [1]}),
        "opo_hypervel_presences": pd.DataFrame({"individid": [1]}),
    }
    question = (
        "quelles sont les individus dont on a fait leur education mais qui ne se "
        "trouvent pas dans la fiche presence"
    )
    trouvees = dt.detecter_tables_mentionnees(question, tables)
    assert "opo_hypervel_individus" not in trouvees
    assert "opo_hypervel_education" in trouvees
    assert "opo_hypervel_presences" in trouvees


def test_detecter_tables_mentionnees_reconnait_individus_quand_ancre():
    tables = {
        "opo_hypervel_individus": pd.DataFrame({"individid": [1]}),
        "opo_hypervel_education": pd.DataFrame({"individid": [1]}),
    }
    question = "fusionne la table individus avec la table education"
    trouvees = dt.detecter_tables_mentionnees(question, tables)
    assert "opo_hypervel_individus" in trouvees
    assert "opo_hypervel_education" in trouvees


def test_syntaxe_fusion_contient_r_et_stata():
    texte = dt.syntaxe_fusion("Tindividual", "TMembership", "individid")
    assert "merge(Tindividual, TMembership" in texte
    assert "merge 1:1 individid using TMembership" in texte


def test_syntaxe_difference_contient_r_et_stata():
    texte = dt.syntaxe_difference("Presence", "Education", "individid")
    assert "anti_join(Presence, Education" in texte
    assert "keep if _merge == 1" in texte


# --- Syntaxe R/Stata pour repartition/echantillon/doublons/coherence --------

def test_syntaxe_repartition_contient_r_et_stata():
    texte = dt.syntaxe_repartition("Tindividual", "sex")
    assert "table(Tindividual$sex)" in texte
    assert "tab sex" in texte


def test_syntaxe_echantillon_contient_r_et_stata():
    texte = dt.syntaxe_echantillon("Tindividual", 100, seed=20260729)
    assert "set.seed(20260729)" in texte
    assert "sample 100, count" in texte


def test_syntaxe_doublons_contient_r_et_stata():
    texte = dt.syntaxe_doublons("Tindividual", "individid")
    assert "duplicated(Tindividual$individid)" in texte
    assert "duplicates tag individid" in texte


def test_syntaxe_coherence_contient_r_et_stata():
    texte = dt.syntaxe_coherence("Tindividual", ["individid"], ["birth_date"])
    assert "duplicated(use_table$individid)" in texte
    assert "duplicates report individid" in texte
    assert "birth_date" in texte


def test_syntaxe_coherence_sans_colonnes_detectees():
    texte = dt.syntaxe_coherence("Tindividual", [], [])
    assert "Aucune colonne d'identifiant ou de date detectee" in texte


# --- Analyse bivariee / multivariee / correlation ---------------------------

def test_tableau_croise_calcule_les_effectifs_avec_marges():
    df = pd.DataFrame({"sex": [1, 2, 1, 2], "niveau": ["a", "b", "a", "a"]})
    tab = dt.tableau_croise(df, "sex", "niveau")
    assert tab.loc[1, "a"] == 2
    assert tab.loc["Total", "Total"] == 4


def test_tableau_croise_leve_erreur_si_colonne_absente():
    df = pd.DataFrame({"sex": [1, 2]})
    with pytest.raises(ValueError):
        dt.tableau_croise(df, "sex", "colonne_inexistante")


def test_colonnes_numeriques_detecte_les_bonnes_colonnes():
    df = pd.DataFrame({"individid": [1, 2], "sex": [1, 2], "niveau": ["a", "b"]})
    assert set(dt.colonnes_numeriques(df)) == {"individid", "sex"}


def test_matrice_correlation_leve_erreur_si_moins_de_deux_colonnes_numeriques():
    df = pd.DataFrame({"individid": [1, 2, 3], "niveau": ["a", "b", "c"]})
    with pytest.raises(ValueError):
        dt.matrice_correlation(df)


def test_matrice_correlation_calcule_correctement():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 2, 3, 4]})
    mat = dt.matrice_correlation(df, ["x", "y"])
    assert mat.loc["x", "y"] == 1.0


def test_tableau_multivarie_groupe_sur_plusieurs_colonnes():
    df = pd.DataFrame({"sex": [1, 1, 2], "niveau": ["a", "a", "b"], "agent": ["X", "X", "Y"]})
    tab = dt.tableau_multivarie(df, ["sex", "niveau", "agent"])
    assert tab["effectif"].sum() == 3
    assert len(tab) == 2  # deux combinaisons distinctes observees


# --- Controle qualite des agents enqueteurs ---------------------------------

def test_detect_agent_columns_reconnait_field_wrkr():
    df = pd.DataFrame({"individid": [1], "field_wrkr": ["A"]})
    assert "field_wrkr" in dt.detect_agent_columns(df)


def test_rapport_agents_compte_fiches_et_doublons_par_agent():
    df = pd.DataFrame({
        "individid": [1, 2, 3, 3],  # 3 est en doublon
        "field_wrkr": ["A", "A", "B", "B"],
    })
    rapport = dt.rapport_agents(df)
    ligne_b = rapport[rapport["agent"] == "B"].iloc[0]
    assert ligne_b["n_fiches"] == 2
    assert ligne_b["doublons_id"] == 2  # les deux lignes de individid=3


def test_rapport_agents_leve_erreur_sans_colonne_agent():
    df = pd.DataFrame({"individid": [1, 2]})
    with pytest.raises(ValueError):
        dt.rapport_agents(df)


def test_rapport_agents_utilise_directement_agent_name(tmp_path):
    # Verification bout-en-bout du vrai schema opo_hypervel_* : `agent_name`
    # est detectee comme colonne d'agent ET contient deja le vrai nom
    # complet - le rapport de performance doit donc afficher des noms
    # directement, sans avoir besoin d'une jointure vers une autre table.
    df = pd.DataFrame({
        "individid": [1, 2, 3, 3],
        "agent_name": ["BADINI RACHIDE", "BADINI RACHIDE", "PASGO RENE", "PASGO RENE"],
    })
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    charge = dt.load_table(str(path))
    rapport = dt.rapport_agents(charge)
    assert set(rapport["agent"]) == {"BADINI RACHIDE", "PASGO RENE"}
    ligne = rapport[rapport["agent"] == "PASGO RENE"].iloc[0]
    assert ligne["n_fiches"] == 2
    assert ligne["doublons_id"] == 2


def test_syntaxe_rapport_agents_contient_r_et_stata():
    texte = dt.syntaxe_rapport_agents("FNewIndividual", "field_wrkr")
    assert "dplyr::count(FNewIndividual, field_wrkr" in texte
    assert "bysort field_wrkr" in texte


# --- Classeurs Excel multi-feuilles -----------------------------------------

def test_est_classeur_excel():
    assert dt.est_classeur_excel("table.xlsx") is True
    assert dt.est_classeur_excel("table.xls") is True
    assert dt.est_classeur_excel("table.csv") is False


def test_charger_classeur_reconnait_chaque_feuille_comme_une_table(tmp_path):
    path = tmp_path / "classeur.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"individid": [1, 2], "sex": [1, 2]}).to_excel(writer, sheet_name="Tindividual", index=False)
        pd.DataFrame({"socialgpid": [1, 2], "nom": ["A", "B"]}).to_excel(writer, sheet_name="Tsocialgp", index=False)

    feuilles = dt.charger_classeur(str(path))

    assert set(feuilles.keys()) == {"Tindividual", "Tsocialgp"}
    assert list(feuilles["Tindividual"].columns) == ["individid", "sex"]
    # La colonne nominative doit etre retiree, meme feuille par feuille
    assert "nom" not in feuilles["Tsocialgp"].columns


def test_charger_classeur_ignore_les_feuilles_vides(tmp_path):
    path = tmp_path / "classeur_avec_vide.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"individid": [1]}).to_excel(writer, sheet_name="Tindividual", index=False)
        pd.DataFrame().to_excel(writer, sheet_name="FeuilleVide", index=False)

    feuilles = dt.charger_classeur(str(path))
    assert "FeuilleVide" not in feuilles
    assert "Tindividual" in feuilles


def test_strip_names_retire_les_colonnes_nominatives(table_exemple):
    df = dt.load_table(table_exemple)
    assert "nom" not in df.columns
    assert "individid" in df.columns


def test_strip_names_garde_le_nom_de_lagent(tmp_path):
    # Bug reel corrige : `agent_name` contient "name" et etait donc retiree
    # par erreur comme n'importe quel nom de personne enquetee - alors que
    # la quasi-totalite des vraies tables de l'observatoire (opo_hypervel_*)
    # portent cette colonne avec le vrai nom de l'agent enqueteur, et que
    # c'est precisement ce que l'observatoire demande de pouvoir afficher.
    # Un vrai nom de repondant/individu (colonne "nom") doit lui rester
    # retire (vie privee des personnes enquetees).
    df = pd.DataFrame({
        "individid": [1, 2],
        "nom": ["Diallo", "Ouedraogo"],
        "agent_name": ["BADINI RACHIDE", "PASGO RENE"],
    })
    path = tmp_path / "table.csv"
    df.to_csv(path, index=False)
    charge = dt.load_table(str(path))
    assert "agent_name" in charge.columns
    assert list(charge["agent_name"]) == ["BADINI RACHIDE", "PASGO RENE"]
    assert "nom" not in charge.columns


def test_load_table_detecte_le_separateur_point_virgule(tmp_path):
    # Bug reel corrige, decouvert sur les vrais exports de l'observatoire :
    # environ la moitie des fichiers CSV reels (opo_hypervel_*) utilisent un
    # point-virgule comme separateur (export en locale francaise), pas une
    # virgule. Sans detection automatique, le fichier se chargeait comme une
    # SEULE colonne (tout le contenu concatene), rendant la table entiere
    # inutilisable - sans aucune erreur visible pour signaler le probleme.
    path = tmp_path / "table_point_virgule.csv"
    path.write_text(
        "individid;respondid;agent_name\n1;10;BADINI RACHIDE\n2;20;PASGO RENE\n", encoding="utf-8"
    )
    df = dt.load_table(str(path))
    assert list(df.columns) == ["individid", "respondid", "agent_name"]
    assert len(df) == 2
    assert list(df["agent_name"]) == ["BADINI RACHIDE", "PASGO RENE"]


def test_detect_id_columns(table_exemple):
    df = dt.load_table(table_exemple)
    assert dt.detect_id_columns(df) == ["individid"]


def test_detect_date_columns(table_exemple):
    df = dt.load_table(table_exemple)
    date_cols = dt.detect_date_columns(df)
    assert "birth_date" in date_cols
    assert "entry_date" in date_cols


def test_repartition_pourcentages_somment_a_100(table_exemple):
    df = dt.load_table(table_exemple)
    rep = dt.repartition(df, "sex")
    assert rep["effectif"].sum() == len(df)
    assert abs(rep["pourcentage"].sum() - 100.0) < 0.5


def test_repartition_colonne_absente_leve_erreur(table_exemple):
    df = dt.load_table(table_exemple)
    with pytest.raises(ValueError):
        dt.repartition(df, "colonne_inexistante")


def test_echantillon_est_reproductible(table_exemple):
    df = dt.load_table(table_exemple)
    a = dt.echantillon(df, n=3, seed=42)
    b = dt.echantillon(df, n=3, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_echantillon_taille_plafonnee_a_la_table(table_exemple):
    df = dt.load_table(table_exemple)
    ech = dt.echantillon(df, n=1000)
    assert len(ech) == len(df)


def test_doublons_detecte_le_bon_identifiant(table_exemple):
    df = dt.load_table(table_exemple)
    dups = dt.doublons(df)
    assert set(dups["individid"]) == {4}
    assert len(dups) == 2


# --- Requete precise (compter/lister/moyenne/somme/min/max, avec filtres) --
# (action REQUETE du classifieur, voir rag.classifier_intention)

@pytest.fixture
def table_requete():
    return pd.DataFrame({
        "individu_id": [1, 2, 3, 4, 5],
        "commune": ["Ouahigouya", "Ouahigouya", "Kaya", "Kaya", "Ouahigouya"],
        "age": [10, 25, 40, None, 60],
    })


def test_executer_requete_compter_sans_filtre(table_requete):
    resultat = dt.executer_requete_donnees(table_requete, "compter", None, None)
    assert resultat == {"operation": "compter", "resultat": 5, "filtres_appliques": []}


def test_executer_requete_compter_avec_filtre_egalite(table_requete):
    resultat = dt.executer_requete_donnees(
        table_requete, "compter", None, [{"colonne": "commune", "operateur": "==", "valeur": "Ouahigouya"}]
    )
    assert resultat["resultat"] == 3
    assert resultat["filtres_appliques"] == ["commune == Ouahigouya"]


def test_executer_requete_compter_filtre_insensible_a_la_casse(table_requete):
    resultat = dt.executer_requete_donnees(
        table_requete, "compter", None, [{"colonne": "commune", "operateur": "==", "valeur": "ouahigouya"}]
    )
    assert resultat["resultat"] == 3


def test_executer_requete_lister_renvoie_le_sous_ensemble(table_requete):
    resultat = dt.executer_requete_donnees(
        table_requete, "lister", None, [{"colonne": "commune", "operateur": "==", "valeur": "Kaya"}]
    )
    assert resultat["n_total"] == 2
    assert set(resultat["resultat"]["individu_id"]) == {3, 4}


def test_executer_requete_moyenne_ignore_les_valeurs_manquantes(table_requete):
    resultat = dt.executer_requete_donnees(table_requete, "moyenne", "age", None)
    # (10 + 25 + 40 + 60) / 4, la valeur manquante (individu 4) est exclue
    assert resultat["resultat"] == pytest.approx(33.75)
    assert resultat["n_valeurs"] == 4


def test_executer_requete_somme_avec_filtre_numerique(table_requete):
    resultat = dt.executer_requete_donnees(
        table_requete, "somme", "age", [{"colonne": "age", "operateur": ">", "valeur": "20"}]
    )
    assert resultat["resultat"] == pytest.approx(25 + 40 + 60)


def test_executer_requete_min_max(table_requete):
    assert dt.executer_requete_donnees(table_requete, "min", "age", None)["resultat"] == 10
    assert dt.executer_requete_donnees(table_requete, "max", "age", None)["resultat"] == 60


def test_executer_requete_moyenne_sans_valeur_numerique_renvoie_none(table_requete):
    vide = table_requete[table_requete["commune"] == "commune_inexistante"]
    resultat = dt.executer_requete_donnees(vide, "moyenne", "age", None)
    assert resultat["resultat"] is None


def test_executer_requete_colonne_cible_absente_leve_erreur(table_requete):
    with pytest.raises(dt.RequeteInvalide):
        dt.executer_requete_donnees(table_requete, "moyenne", "colonne_qui_nexiste_pas", None)


def test_executer_requete_operation_inconnue_leve_erreur(table_requete):
    with pytest.raises(dt.RequeteInvalide):
        dt.executer_requete_donnees(table_requete, "supprimer", None, None)


def test_executer_requete_filtre_colonne_absente_est_ignore(table_requete):
    # Un filtre sur une colonne inexistante est simplement ignore (deja
    # filtre en amont par rag.classifier_intention, mais re-verifie ici) :
    # ne doit jamais faire planter le calcul.
    resultat = dt.executer_requete_donnees(
        table_requete, "compter", None, [{"colonne": "colonne_absente", "operateur": "==", "valeur": "x"}]
    )
    assert resultat["resultat"] == 5
    assert resultat["filtres_appliques"] == []


# --- Requete SQL en lecture seule sur plusieurs tables (jointures) ---------
# (repli general pour une question qui doit croiser 2, 3, 4 tables ou plus -
# voir app.py:tenter_requete_sql / rag.generer_requete_sql)

@pytest.fixture
def tables_trois_bases():
    return {
        "opo_hypervel_presences": pd.DataFrame({"individu_id": [1, 2, 3, 4], "enquete_id": [10, 10, 20, 20]}),
        "opo_hypervel_d_e_c_e_s": pd.DataFrame({"individu_id": [2, 5]}),
        "opo_hypervel_enquete_or_visites": pd.DataFrame({"id": [10, 20], "agent_id": ["A", "B"]}),
    }


def test_executer_sql_jointure_sur_trois_tables(tables_trois_bases):
    sql = (
        "SELECT e.agent_id, COUNT(*) AS n_deces "
        "FROM opo_hypervel_presences p "
        "JOIN opo_hypervel_d_e_c_e_s d ON p.individu_id = d.individu_id "
        "JOIN opo_hypervel_enquete_or_visites e ON p.enquete_id = e.id "
        "GROUP BY e.agent_id"
    )
    resultat = dt.executer_sql(tables_trois_bases, sql)
    assert list(resultat["agent_id"]) == ["A"]
    assert list(resultat["n_deces"]) == [1]


def test_executer_sql_simple_select(tables_trois_bases):
    resultat = dt.executer_sql(tables_trois_bases, "SELECT COUNT(*) AS n FROM opo_hypervel_presences")
    assert resultat["n"].iloc[0] == 4


def test_executer_sql_refuse_les_instructions_non_select(tables_trois_bases):
    for requete in [
        "DROP TABLE opo_hypervel_presences",
        "DELETE FROM opo_hypervel_presences",
        "UPDATE opo_hypervel_presences SET individu_id = 0",
        "ATTACH 'x.db'",
        "SELECT 1; DROP TABLE opo_hypervel_presences",
    ]:
        with pytest.raises(dt.RequeteSQLInvalide):
            dt.executer_sql(tables_trois_bases, requete)


def test_executer_sql_requete_vide_leve_erreur(tables_trois_bases):
    with pytest.raises(dt.RequeteSQLInvalide):
        dt.executer_sql(tables_trois_bases, "")


def test_executer_sql_erreur_de_syntaxe_leve_requete_invalide(tables_trois_bases):
    with pytest.raises(dt.RequeteSQLInvalide):
        dt.executer_sql(tables_trois_bases, "SELECT colonne_qui_nexiste_pas FROM opo_hypervel_presences")


def test_executer_sql_limite_a_200_lignes():
    grande_table = {"t": pd.DataFrame({"x": range(500)})}
    resultat = dt.executer_sql(grande_table, "SELECT * FROM t")
    assert len(resultat) == 200


def test_dates_incoherentes_detecte_futur_et_avant_1900(table_exemple):
    df = dt.load_table(table_exemple)
    bad = dt.dates_incoherentes(df, "birth_date")
    # 1899-01-01 (avant la borne) et 2027-01-01 (dans le futur)
    assert len(bad) == 2


def test_rapport_coherence_signale_les_bonnes_anomalies(table_exemple):
    df = dt.load_table(table_exemple)
    rapport = dt.rapport_coherence(df)
    assert rapport["n_lignes"] == 5
    assert rapport["anomalies"]["doublons::individid"] == 2
    assert rapport["anomalies"]["dates_invraisemblables::birth_date"] == 2


def test_rapport_coherence_indique_les_colonnes_verifiees(table_exemple):
    # Precision : le rapport doit dire explicitement quelles colonnes ont ete
    # examinees, pas seulement les anomalies trouvees (evite un "aucune
    # anomalie" ambigu si aucune colonne pertinente n'avait ete detectee).
    df = dt.load_table(table_exemple)
    rapport = dt.rapport_coherence(df)
    assert "individid" in rapport["colonnes_id_verifiees"]
    assert "birth_date" in rapport["colonnes_date_verifiees"]


def test_rapport_coherence_colonnes_verifiees_vides_si_aucune_detectee():
    df = pd.DataFrame({"valeur": [1, 2, 3]})
    rapport = dt.rapport_coherence(df)
    assert rapport["colonnes_id_verifiees"] == []
    assert rapport["colonnes_date_verifiees"] == []


def test_rapport_coherence_sans_anomalie():
    df = pd.DataFrame({
        "individid": [1, 2, 3],
        "sex": [1, 2, 1],
        "birth_date": pd.to_datetime(["1990-01-01", "1995-01-01", "2000-01-01"]),
    })
    rapport = dt.rapport_coherence(df)
    assert rapport["anomalies"] == {}


# --- Export multi-format (CSV / Excel / Stata) ------------------------------

@pytest.fixture
def petite_table():
    return pd.DataFrame({
        "individid": [1, 2, 3],
        "sex": [1, 2, 1],
        "birth_date": pd.to_datetime(["1990-01-01", "1995-06-15", "2000-12-31"]),
        "une colonne avec espaces et accents éà": ["a", "b", "c"],
    })


def test_exporter_csv_contient_les_donnees(petite_table):
    data = dt.exporter(petite_table, "csv")
    texte = data.decode("utf-8-sig")
    assert "individid" in texte
    assert "1990-01-01" in texte


def test_exporter_xlsx_est_relisible(petite_table, tmp_path):
    data = dt.exporter(petite_table, "xlsx")
    path = tmp_path / "export.xlsx"
    path.write_bytes(data)
    relu = pd.read_excel(path)
    assert list(relu["individid"]) == [1, 2, 3]


def test_exporter_dta_sanitize_les_noms_de_colonnes(petite_table):
    import io
    data = dt.exporter(petite_table, "dta")
    relu = pd.read_stata(io.BytesIO(data))
    # Le nom de colonne avec espaces/accents doit avoir ete assaini
    assert "une colonne avec espaces et accents éà" not in relu.columns
    assert len(relu) == 3


def test_exporter_format_inconnu_leve_erreur(petite_table):
    with pytest.raises(ValueError):
        dt.exporter(petite_table, "pdf")


# --- Resolution de la table ciblee par une question -------------------------

def test_resoudre_table_ciblee_par_nom_mentionne():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [1]}),
    }
    nom, df = dt.resoudre_table_ciblee("échantillon de Tsocialgp", tables, nom_par_defaut="Tindividual")
    assert nom == "Tsocialgp"


def test_resoudre_table_ciblee_par_nom_informel_sans_prefixe():
    tables = {
        "FNewEducation": pd.DataFrame({"individid": [1, 2]}),
        "FNewPresences": pd.DataFrame({"individid": [1]}),
    }
    nom, df = dt.resoudre_table_ciblee("échantillon de la table education", tables)
    assert nom == "FNewEducation"


def test_resoudre_table_ciblee_retombe_sur_le_defaut():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [1]}),
    }
    nom, df = dt.resoudre_table_ciblee("répartition par sexe", tables, nom_par_defaut="Tindividual")
    assert nom == "Tindividual"


def test_resoudre_table_ciblee_sans_table_disponible():
    nom, df = dt.resoudre_table_ciblee("répartition par sexe", {}, nom_par_defaut=None)
    assert nom is None and df is None


def test_resoudre_table_ciblee_par_colonne_unique_sans_nommer_la_table():
    # Aucune table n'est nommee dans la question, et aucune table n'est
    # selectionnee comme "active" (nom_par_defaut=None) : la colonne "sex"
    # n'appartenant qu'a Tindividual doit suffire a la retrouver, pour que
    # toutes les tables chargees soient aussi faciles a interroger.
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "Tsocialgp": pd.DataFrame({"socialgpid": [1], "chef_menage": [1]}),
    }
    nom, df = dt.resoudre_table_ciblee("répartition de sex", tables, nom_par_defaut=None)
    assert nom == "Tindividual"


def test_resoudre_table_ciblee_colonne_partagee_retombe_sur_le_defaut():
    # "individid" est present dans les deux tables : impossible de trancher
    # par la seule colonne, on retombe donc sur la table active par defaut.
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "TMembership": pd.DataFrame({"individid": [1], "socialgpid": [1]}),
    }
    nom, df = dt.resoudre_table_ciblee("doublons sur individid", tables, nom_par_defaut="TMembership")
    assert nom == "TMembership"


# --- Memoire conversationnelle (historique) ---------------------------------

def test_tables_mentionnees_dans_historique_du_plus_recent_au_plus_ancien():
    tables = {"Tindividual": pd.DataFrame({"individid": [1]}), "Tsocialgp": pd.DataFrame({"socialgpid": [1]})}
    historique = [
        {"role": "user", "contenu": "parle-moi de Tindividual"},
        {"role": "assistant", "contenu": "Tindividual contient les individus."},
        {"role": "user", "contenu": "et pour Tsocialgp ?"},
    ]
    assert dt.tables_mentionnees_dans_historique(historique, tables) == ["Tsocialgp", "Tindividual"]


def test_tables_mentionnees_dans_historique_vide_sans_historique():
    tables = {"Tindividual": pd.DataFrame({"individid": [1]})}
    assert dt.tables_mentionnees_dans_historique(None, tables) == []
    assert dt.tables_mentionnees_dans_historique([], tables) == []


def test_resoudre_table_ciblee_utilise_lhistorique_avant_le_defaut():
    # Question de suivi ("les doublons ?") qui ne nomme aucune table et dont
    # aucune colonne n'est mentionnee : la table discutee juste avant dans la
    # conversation doit l'emporter sur la table par defaut de l'interface.
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1], "sex": [1]}),
        "TMembership": pd.DataFrame({"individid": [1], "socialgpid": [1]}),
    }
    historique = [{"role": "user", "contenu": "montre-moi TMembership"}]
    nom, df = dt.resoudre_table_ciblee(
        "les doublons ?", tables, nom_par_defaut="Tindividual", historique=historique
    )
    assert nom == "TMembership"


def test_resoudre_table_ciblee_priorite_au_nom_explicite_meme_avec_historique():
    tables = {
        "Tindividual": pd.DataFrame({"individid": [1]}),
        "TMembership": pd.DataFrame({"individid": [1]}),
    }
    historique = [{"role": "user", "contenu": "montre-moi TMembership"}]
    nom, df = dt.resoudre_table_ciblee(
        "doublons de Tindividual", tables, nom_par_defaut=None, historique=historique
    )
    assert nom == "Tindividual"


# --- Catalogue de controles de coherence avances ----------------------------

def test_controle_id_longueur_signale_les_ids_de_taille_differente():
    df = pd.DataFrame({"individid": ["1", "22", "3", "4"]})
    resultat = dt.controle_id_longueur(df)
    assert resultat["n_anomalies"] == 1


def test_controle_id_longueur_none_sans_colonne_id():
    df = pd.DataFrame({"sex": [1, 2]})
    assert dt.controle_id_longueur(df) is None


def test_controle_auto_reference_detecte_individid_egal_individid2():
    df = pd.DataFrame({"individid": [1, 2, 3], "individid2": [1, 3, 3]})
    resultat = dt.controle_auto_reference(df)
    assert resultat["n_anomalies"] == 2


def test_controle_parents_identiques():
    df = pd.DataFrame({"fatherid": [10, 20], "motherid": [10, 21]})
    resultat = dt.controle_parents_identiques(df)
    assert resultat["n_anomalies"] == 1


def test_controle_parent_manquant_jeune_enfant():
    df = pd.DataFrame({
        "birth_date": ["2024-01-01", "1990-01-01", "2023-06-01"],
        "motherid": [10, 21, None],
        "fatherid": [10, 20, None],
    })
    resultat = dt.controle_parent_manquant_jeune_enfant(df, seuil_age=5.0)
    assert resultat["n_anomalies"] == 1  # seule la ligne 2 (< 5 ans, aucun parent)


def test_controle_sentinelle_poids_et_taille():
    df = pd.DataFrame({"poids": [3200, 9999], "taille": [50, 99]})
    assert dt.controle_sentinelle(df, dt.WEIGHT_LIKE, 9999)["n_anomalies"] == 1
    assert dt.controle_sentinelle(df, dt.HEIGHT_LIKE, 99)["n_anomalies"] == 1


def test_controle_gps_hors_zone_detecte_coordonnees_invalides():
    df = pd.DataFrame({"lat": [12.3, 40.0, None], "lon": [-1.5, 2.0, 0.5]})
    resultat = dt.controle_gps_hors_zone(df)
    assert resultat["n_anomalies"] == 2  # ligne hors BF + ligne avec coordonnee manquante


def test_controle_telephone_format_detecte_numero_invalide():
    df = pd.DataFrame({"NumTelephone": ["70123456", "701234", "70123456/70654321"]})
    resultat = dt.controle_telephone_format(df)
    assert resultat["n_anomalies"] == 1


def test_controle_dates_arrivee_depart_detecte_incoherence():
    df = pd.DataFrame({
        "arrive_date": ["2024-01-01", "2024-01-01", "2024-05-01"],
        "depart_date": ["2024-01-01", "2023-12-01", None],
    })
    resultat = dt.controle_dates_arrivee_depart(df)
    assert resultat["n_anomalies"] == 2


def test_controle_residence_multiple_detecte_individu_dans_plusieurs_menages():
    df = pd.DataFrame({"individid": [1, 1, 2, 3], "locationid": [10, 20, 10, 10]})
    resultat = dt.controle_residence_multiple(df)
    assert resultat["n_anomalies"] == 1


def test_controle_tranche_age_detecte_hors_plage():
    df = pd.DataFrame({"birth_date": ["2020-01-01", "2000-01-01", "1990-01-01"]})
    resultat = dt.controle_tranche_age(df, 5, 34)
    assert resultat["n_anomalies"] == 1


def test_controle_eligibilite_croisee_presence_education():
    presence = pd.DataFrame({
        "individid": [1, 2, 3, 4, 5],
        "sleep_lastnight": [1, 1, 0, 1, 1],
        "depart_date": [None, None, None, "2024-01-01", None],
    })
    education = pd.DataFrame({"individid": [1, 4, 6]})
    tables = {"FNewPresences": presence, "FNewEducation": education}
    resultat = dt.controle_eligibilite_croisee(tables, "FNewPresences", "FNewEducation")
    assert resultat["n_eligibles_sans_fiche"] == 2
    assert resultat["n_fiche_sans_eligibilite"] == 2
    assert set(resultat["eligibles_sans_fiche"]) == {2, 5}
    assert set(resultat["fiche_sans_eligibilite"]) == {4, 6}


def test_controle_deces_present_detecte_le_chevauchement():
    deces = pd.DataFrame({"individid": [3, 999]})
    presence = pd.DataFrame({"individid": [1, 2, 3, 4, 5]})
    tables = {"FNewDeath": deces, "FNewPresences": presence}
    resultat = dt.controle_deces_present(tables, "FNewDeath", "FNewPresences")
    assert resultat["n_anomalies"] == 1


def test_rapport_coherence_avancee_couvre_toutes_les_tables_et_croises():
    presence = pd.DataFrame({
        "individid": [1, 2, 3, 4, 5],
        "sleep_lastnight": [1, 1, 0, 1, 1],
        "depart_date": [None, None, None, "2024-01-01", None],
    })
    education = pd.DataFrame({"individid": [1, 4, 6]})
    deces = pd.DataFrame({"individid": [3, 999]})
    tables = {"FNewPresences": presence, "FNewEducation": education, "FNewDeath": deces}

    rapport = dt.rapport_coherence_avancee(tables)
    assert set(rapport["par_table"].keys()) == {"FNewPresences", "FNewEducation", "FNewDeath"}
    libelles_croises = [c[0] for c in rapport["croises"]]
    assert any("ligibilit" in l for l in libelles_croises)
    assert any("écédé" in l for l in libelles_croises)


def test_rapport_coherence_avancee_sur_une_seule_table():
    tables = {
        "FNewIndividual": pd.DataFrame({"individid": ["1", "2"], "fatherid": [1, 2], "motherid": [1, 3]}),
        "FNewAutre": pd.DataFrame({"x": [1]}),
    }
    rapport = dt.rapport_coherence_avancee(tables, nom_table="FNewIndividual")
    assert list(rapport["par_table"].keys()) == ["FNewIndividual"]


# --- Module "Performances" : volume d'activite de terrain par agent ---------

@pytest.fixture
def tables_performance_terrain():
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
    deces = pd.DataFrame({"individid": [9], "field_wrkr": ["A2"]})
    equipe = pd.DataFrame({"field_wrkr": ["A1", "A2", "A3"], "controleur": ["C1", "C1", "C2"]})
    return {
        "FNewPresences": presence, "FNewBirth": naissance,
        "FNewDeath": deces, "Equipe": equipe,
    }


def test_rapport_performance_agents_agrege_toutes_les_tables(tables_performance_terrain):
    rapport = dt.rapport_performance_agents(tables_performance_terrain)
    assert set(rapport["agent"]) == {"A1", "A2", "A3"}
    ligne_a1 = rapport[rapport["agent"] == "A1"].iloc[0]
    assert ligne_a1["Ménages/UCH visités"] == 3
    assert ligne_a1["Ménages/UCH distincts"] == 2
    assert ligne_a1["Naissances enregistrées"] == 2
    assert ligne_a1["total_fiches"] == 5  # menages/UCH visites (3) + naissances (2), pas les distincts


def test_rapport_performance_agents_exclut_les_agents_demandes(tables_performance_terrain):
    rapport = dt.rapport_performance_agents(tables_performance_terrain, exclure=["a3"])
    assert set(rapport["agent"]) == {"A1", "A2"}


def test_rapport_performance_agents_ignore_la_table_equipe(tables_performance_terrain):
    # La table "Equipe" (avec une colonne controleur) ne doit jamais etre
    # comptee comme une fiche d'activite de terrain.
    rapport = dt.rapport_performance_agents(tables_performance_terrain)
    assert not any(c.startswith("Autres fiches") for c in rapport.columns)


def test_rapport_performance_par_jour(tables_performance_terrain):
    par_jour = dt.rapport_performance_par_jour(tables_performance_terrain)
    assert list(par_jour.columns) == ["date", "agent", "n_fiches"]
    total = par_jour["n_fiches"].sum()
    assert total == 8  # les 8 lignes de la fiche presence


def test_fusion_agent_controleur_ajoute_la_colonne(tables_performance_terrain):
    rapport = dt.rapport_performance_agents(tables_performance_terrain)
    fusionne, nom_equipe = dt.fusion_agent_controleur(rapport, tables_performance_terrain)
    assert nom_equipe == "Equipe"
    assert "controleur" in fusionne.columns
    assert fusionne.set_index("agent")["controleur"]["A3"] == "C2"


def test_fusion_agent_controleur_sans_table_equipe():
    rapport = pd.DataFrame({"agent": ["A1"], "total_fiches": [5]})
    fusionne, nom_equipe = dt.fusion_agent_controleur(rapport, {"Autre": pd.DataFrame({"x": [1]})})
    assert nom_equipe is None
    assert "controleur" not in fusionne.columns


def test_fusion_agent_controleur_reconnait_la_colonne_contro_tronquee_stata():
    # Bug reel rencontre sur un vrai export Stata (.dta) d'equipe fourni par
    # l'observatoire : la colonne "Controleur" y est nommee "Contro" (nom de
    # variable Stata tronque), donc CONTROLEUR_LIKE (qui exigeait le mot
    # complet "controleur"/"superviseur"...) ne la reconnaissait pas - la
    # jointure agent<->controleur echouait silencieusement (colonne absente,
    # jamais d'erreur visible) meme avec une vraie table d'equipe chargee.
    equipe = pd.DataFrame({
        "field_wrkr": ["BADINI RACHIDE", "PASGO RENE", "MIHIN JUDITH STACY"],
        "gender": [1, 1, 2],
        "Contro": ["OUEDRAOGO MOUSSA AHMED", "OUEDRAOGO MOUSSA AHMED", "ZONGO SOMPOGOBNOMA HYACINTHE"],
    })
    rapport = pd.DataFrame({"agent": ["BADINI RACHIDE", "MIHIN JUDITH STACY"], "n_fiches": [12, 9]})
    fusionne, nom_equipe = dt.fusion_agent_controleur(rapport, {"equipe": equipe})
    assert nom_equipe == "equipe"
    assert fusionne.set_index("agent")["controleur"]["BADINI RACHIDE"] == "OUEDRAOGO MOUSSA AHMED"
    assert fusionne.set_index("agent")["controleur"]["MIHIN JUDITH STACY"] == "ZONGO SOMPOGOBNOMA HYACINTHE"


def test_controleur_like_ne_matche_pas_un_controle_qualite():
    # Garde-fou : l'ajout du cas tronque "Contro" (ancre ^...$) ne doit
    # jamais faire matcher par erreur une colonne de controle QUALITE (ex:
    # "controle_qualite", "date_controle"), qui n'a rien a voir avec
    # l'identite d'un superviseur d'equipe.
    assert dt.CONTROLEUR_LIKE.search("controle_qualite") is None
    assert dt.CONTROLEUR_LIKE.search("date_controle") is None


def test_prevision_objectif_calcule_la_date_de_fin(tables_performance_terrain):
    par_jour = dt.rapport_performance_par_jour(tables_performance_terrain)
    prevision = dt.prevision_objectif(par_jour, objectif=20)
    assert prevision["cumul_actuel"] == 8
    assert prevision["reste_a_faire"] == 12
    assert prevision["date_fin_projetee"] is not None


def test_prevision_objectif_none_sans_donnees():
    assert dt.prevision_objectif(pd.DataFrame(), objectif=100) is None


def test_simulation_rythme():
    resultat = dt.simulation_rythme(reste_a_faire=100, jours_disponibles=10)
    assert resultat["rythme_journalier_necessaire"] == 10.0


def test_simulation_rythme_leve_erreur_si_zero_jour():
    with pytest.raises(ValueError):
        dt.simulation_rythme(reste_a_faire=100, jours_disponibles=0)


def test_rechercher_identifiant_trouve_dans_plusieurs_tables(tables_performance_terrain):
    resultats = dt.rechercher_identifiant("3", tables_performance_terrain)
    assert "FNewPresences" in resultats
    assert "FNewBirth" in resultats
    assert "FNewDeath" not in resultats


def test_rechercher_identifiant_introuvable(tables_performance_terrain):
    assert dt.rechercher_identifiant("999999", tables_performance_terrain) == {}


def test_generer_rapport_performance_docx_produit_des_bytes(tables_performance_terrain):
    rapport = dt.rapport_performance_agents(tables_performance_terrain)
    par_jour = dt.rapport_performance_par_jour(tables_performance_terrain)
    prevision = dt.prevision_objectif(par_jour, objectif=20)
    contenu = dt.generer_rapport_performance_docx(rapport, prevision, objectif=20)
    assert isinstance(contenu, bytes)
    assert len(contenu) > 0


# --- Recherche multi-table pour les analyses a plusieurs colonnes -----------
# ("il ne faut pas lire une seule base par defaut ; il faut tout lire")

@pytest.fixture
def tables_colonnes_partagees():
    return {
        "FNewEducation": pd.DataFrame({
            "individid": [1, 2, 3], "sex": [1, 2, 1], "education_level": ["primaire", "secondaire", "primaire"],
        }),
        "FNewEmploi": pd.DataFrame({
            "individid": [1, 2], "sex": [2, 2], "revenu": [50000, 60000],
        }),
        "FNewSante": pd.DataFrame({"individid": [1], "autre_col": ["x"]}),
    }


def test_colonnes_mentionnees_cherche_dans_toutes_les_tables(tables_colonnes_partagees):
    trouvees = dt.colonnes_mentionnees("croise sex et education_level", tables_colonnes_partagees)
    assert "sex" in trouvees
    assert "education_level" in trouvees
    assert "autre_col" not in trouvees


def test_tables_avec_toutes_colonnes_ne_garde_que_les_tables_completes(tables_colonnes_partagees):
    resultat = dt.tables_avec_toutes_colonnes(["individid", "sex"], tables_colonnes_partagees)
    assert set(resultat) == {"FNewEducation", "FNewEmploi"}


def test_tables_avec_toutes_colonnes_vide_sans_colonnes():
    assert dt.tables_avec_toutes_colonnes([], {"X": pd.DataFrame({"a": [1]})}) == []


# --- Alignement sur le vrai schema de l'observatoire (base Hypervel) --------
# Les tables reelles s'appellent "opo_hypervel_<nom>", certaines epellent un
# sigle avec un underscore entre chaque lettre (ex: "opo_hypervel_d_e_c_e_s"),
# et l'identite de l'agent n'est saisie qu'une fois par enquete/visite (table
# "opo_hypervel_enquete_or_visites"), pas directement sur chaque fiche.

def test_alias_table_reconnait_prefixe_opo_hypervel():
    assert "naissances" in dt.alias_table("opo_hypervel_naissances")
    assert "presences" in dt.alias_table("opo_hypervel_presences")


def test_alias_table_reconnait_sigle_epelle_avec_underscores():
    # "opo_hypervel_d_e_c_e_s" est le nom reel genere automatiquement pour la
    # table des deces (Laravel/Hypervel epelle "DECES" lettre par lettre).
    assert "deces" in dt.alias_table("opo_hypervel_d_e_c_e_s")
    assert "cpns" in dt.alias_table("opo_hypervel_c_p_n_s")


def test_table_correspond_reconnait_sigle_epelle_avec_underscores():
    assert dt._table_correspond("opo_hypervel_d_e_c_e_s", ["death", "deces"])


def test_detecter_tables_mentionnees_insensible_aux_accents():
    tables = {"opo_hypervel_d_e_c_e_s": pd.DataFrame({"individu_id": [1]})}
    # La question tape avec accent ("décès") doit reconnaitre la table meme
    # si son nom reel/alias normalise ne porte pas d'accent.
    assert dt.detecter_tables_mentionnees("combien de décès enregistrés", tables) == ["opo_hypervel_d_e_c_e_s"]


def test_resoudre_table_ciblee_insensible_aux_accents():
    tables = {
        "opo_hypervel_d_e_c_e_s": pd.DataFrame({"individu_id": [1]}),
        "opo_hypervel_naissances": pd.DataFrame({"individu_id": [1]}),
    }
    nom, _ = dt.resoudre_table_ciblee("nombre de décès", tables)
    assert nom == "opo_hypervel_d_e_c_e_s"


def test_agent_like_ne_capture_pas_enquete_id_ni_date_enquete():
    df = pd.DataFrame({"enquete_id": [1], "date_enquete": ["2026-01-01"], "individu_id": [1]})
    assert dt.detect_agent_columns(df) == []


def test_agent_like_capture_agent_id():
    df = pd.DataFrame({"agent_id": [1], "individu_id": [1]})
    assert dt.detect_agent_columns(df) == ["agent_id"]


def test_colonne_agent_effective_colonne_directe_prioritaire():
    tables = {"opo_hypervel_presences": pd.DataFrame({"agent_id": ["A", "B"], "individu_id": [1, 2]})}
    df, col = dt.colonne_agent_effective(tables["opo_hypervel_presences"], tables)
    assert col == "agent_id"


def test_colonne_agent_effective_jointure_via_enquete_id():
    # Fiche naissance : pas de colonne agent directe, seulement enquete_id
    # (schema reel) -> l'agent doit etre retrouve par jointure vers la table
    # enquetes/visites qui, elle, porte agent_id.
    tables = {
        "opo_hypervel_naissances": pd.DataFrame({"individu_id": [1, 2, 3], "enquete_id": [10, 10, 20]}),
        "opo_hypervel_enquete_or_visites": pd.DataFrame({"id": [10, 20], "agent_id": ["A", "B"]}),
    }
    df, col = dt.colonne_agent_effective(tables["opo_hypervel_naissances"], tables)
    assert col == "__agent_via_enquete__"
    assert list(df[col]) == ["A", "A", "B"]


def test_colonne_agent_effective_sans_table_enquetes_renvoie_none():
    tables = {"opo_hypervel_naissances": pd.DataFrame({"individu_id": [1], "enquete_id": [10]})}
    df, col = dt.colonne_agent_effective(tables["opo_hypervel_naissances"], tables)
    assert col is None


def test_rapport_performance_agents_via_jointure_enquete_id():
    # Bout en bout : une fiche sans colonne agent directe doit quand meme
    # apparaitre dans le rapport de performance grace a la jointure automatique.
    # (La table enquetes/visites elle-meme porte une colonne agent directe et
    # est donc aussi comptee pour son propre role - chaque ligne y represente
    # une visite de terrain reellement effectuee par l'agent.)
    tables = {
        "opo_hypervel_naissances": pd.DataFrame({"individu_id": [1, 2, 3], "enquete_id": [10, 10, 20]}),
        "opo_hypervel_enquete_or_visites": pd.DataFrame({"id": [10, 20], "agent_id": ["A", "B"]}),
    }
    rapport = dt.rapport_performance_agents(tables)
    assert set(rapport["agent"]) == {"A", "B"}
    ligne_a = rapport[rapport["agent"] == "A"].iloc[0]
    assert ligne_a["Naissances enregistrées"] == 2


def test_fusion_identite_agent_ajoute_email_depuis_la_table_users():
    rapport_agents = pd.DataFrame({"agent": ["1", "2"], "total_fiches": [5, 3]})
    tables = {
        "opo_hypervel_users": pd.DataFrame({"id": [1, 2], "email": ["a@obs.bf", "b@obs.bf"]}),
    }
    fusionne, nom_table = dt.fusion_identite_agent(rapport_agents, tables)
    assert nom_table == "opo_hypervel_users"
    assert list(fusionne.sort_values("agent")["email_agent"]) == ["a@obs.bf", "b@obs.bf"]


def test_fusion_identite_agent_sans_table_users_renvoie_inchange():
    rapport_agents = pd.DataFrame({"agent": ["1"], "total_fiches": [5]})
    fusionne, nom_table = dt.fusion_identite_agent(rapport_agents, {})
    assert nom_table is None
    assert "email_agent" not in fusionne.columns


def test_tranches_age_reconnait_histoire_marietales_reelle():
    mots_cles = next(m for m, b in dt.TRANCHES_AGE_PAR_TYPE_FICHE if "marietal" in m)
    assert dt._table_correspond("opo_hypervel_histoire_marietales", mots_cles)


def test_rapport_coherence_avancee_grossesse_reelle_exclut_issue_grossesses():
    tables = {
        "opo_hypervel_grossesses": pd.DataFrame({"individu_id": [1, 2]}),
        "opo_hypervel_issue_grossesses": pd.DataFrame({"individu_id": [1]}),
    }
    resultat = dt.rapport_coherence_avancee(tables)
    croises = resultat["croises"]
    # Un controle croise "grossesse sans issue" doit avoir ete detecte, avec
    # les deux tables reelles correctement distinguees (jamais la meme table
    # des deux cotes de la comparaison).
    correspondances = [c for c in croises if "issue" in c[0].lower()]
    assert len(correspondances) == 1
    _, source, cible, _ = correspondances[0]
    assert source == "opo_hypervel_grossesses"
    assert cible == "opo_hypervel_issue_grossesses"


def test_rapport_coherence_avancee_depart_reconnu_comme_migration_out():
    tables = {
        "opo_hypervel_presences": pd.DataFrame({"individu_id": [1, 2]}),
        "opo_hypervel_departs": pd.DataFrame({"individu_id": [1]}),
    }
    resultat = dt.rapport_coherence_avancee(tables)
    correspondances = [c for c in resultat["croises"] if "départ" in c[0].lower()]
    assert len(correspondances) == 1
