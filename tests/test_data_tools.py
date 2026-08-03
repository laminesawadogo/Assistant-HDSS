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


# --- Relations et fusion entre tables chargees -------------------------------

def test_detecter_tables_mentionnees():
    tables = {"Tindividual": pd.DataFrame({"individid": [1]}), "Tsocialgp": pd.DataFrame({"socialgpid": [1]})}
    assert dt.detecter_tables_mentionnees("relation entre Tindividual et Tsocialgp", tables) == ["Tindividual", "Tsocialgp"]
    assert dt.detecter_tables_mentionnees("question sans nom de table", tables) == []


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
