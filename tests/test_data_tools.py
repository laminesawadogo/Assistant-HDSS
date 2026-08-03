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
