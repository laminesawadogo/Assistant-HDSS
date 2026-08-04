"""
Tests du module drive_sync.py : synchronisation automatique des tables
depuis le dossier Google Drive de l'observatoire.

Les fonctions pures (extraction de date, regroupement par table, selection
du dernier export) sont testees directement. Le flux complet (`synchroniser`)
est teste avec un client Drive ENTIEREMENT SIMULE (aucun appel reseau reel,
aucun identifiant necessaire) pour rester rapide et reproductible en CI.
"""

from datetime import datetime

import pytest

import drive_sync as ds


# --- extraire_date_export ---------------------------------------------------

def test_extraire_date_export_format_iso_tirets():
    assert ds.extraire_date_export("FNewIndividual_2026-08-04.csv") == datetime(2026, 8, 4)


def test_extraire_date_export_format_compact():
    assert ds.extraire_date_export("FNewIndividual_20260804.xlsx") == datetime(2026, 8, 4)


def test_extraire_date_export_format_jour_mois_annee():
    assert ds.extraire_date_export("export_04-08-2026.csv") == datetime(2026, 8, 4)


def test_extraire_date_export_absente():
    assert ds.extraire_date_export("FNewIndividual_final.csv") is None


def test_extraire_date_export_ne_confond_pas_un_identifiant_avec_une_date():
    # Une longue suite de chiffres qui ne correspond a aucun motif de date
    # plausible (ex: un identifiant de version) ne doit pas etre interpretee
    # a tort comme une date.
    assert ds.extraire_date_export("export_v99999999.csv") is None


# --- nom_base_table ----------------------------------------------------------

def test_nom_base_table_retire_extension_et_date():
    assert ds.nom_base_table("FNewIndividual_2026-08-04.csv") == "FNewIndividual"
    assert ds.nom_base_table("FNewIndividual_2026-08-03.csv") == "FNewIndividual"


def test_nom_base_table_deux_exports_meme_table_donnent_le_meme_nom():
    a = ds.nom_base_table("FNewEducation_20260803.xlsx")
    b = ds.nom_base_table("FNewEducation_20260804.xlsx")
    assert a == b == "FNewEducation"


def test_nom_base_table_sans_date_garde_le_nom_stem():
    assert ds.nom_base_table("FNewIndividual.csv") == "FNewIndividual"


# --- derniers_fichiers_par_table --------------------------------------------

def test_derniers_fichiers_par_table_garde_le_plus_recent_par_date_dans_le_nom():
    fichiers = [
        {"id": "1", "name": "FNewIndividual_2026-08-02.csv", "modifiedTime": "2026-08-02T10:00:00Z"},
        {"id": "2", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "3", "name": "FNewIndividual_2026-08-03.csv", "modifiedTime": "2026-08-03T10:00:00Z"},
    ]
    retenus = ds.derniers_fichiers_par_table(fichiers)
    assert set(retenus.keys()) == {"FNewIndividual"}
    assert retenus["FNewIndividual"]["id"] == "2"


def test_derniers_fichiers_par_table_plusieurs_tables_distinctes():
    fichiers = [
        {"id": "1", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "2", "name": "FNewEducation_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "3", "name": "FNewEducation_2026-08-03.csv", "modifiedTime": "2026-08-03T10:00:00Z"},
    ]
    retenus = ds.derniers_fichiers_par_table(fichiers)
    assert set(retenus.keys()) == {"FNewIndividual", "FNewEducation"}
    assert retenus["FNewEducation"]["id"] == "2"


def test_derniers_fichiers_par_table_se_rabat_sur_modifiedtime_sans_date_dans_le_nom():
    fichiers = [
        {"id": "1", "name": "FNewIndividual_export.csv", "modifiedTime": "2026-08-02T10:00:00Z"},
        {"id": "2", "name": "FNewIndividual_export_final.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
    ]
    # Les deux noms sont legerement differents donc pas regroupes ensemble
    # automatiquement : ce test verifie surtout qu'aucune exception n'est
    # levee et que la date de derniere modification sert bien de repli
    # lorsqu'aucune date n'est detectable dans le nom.
    retenus = ds.derniers_fichiers_par_table(fichiers)
    assert len(retenus) == 2
    for info in retenus.values():
        assert info["date_export"] is None
        assert info["modifiedTime"] is not None


def test_derniers_fichiers_par_table_ignore_les_extensions_non_reconnues():
    fichiers = [
        {"id": "1", "name": "manuel_utilisateur.pdf", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "2", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
    ]
    retenus = ds.derniers_fichiers_par_table(fichiers)
    assert set(retenus.keys()) == {"FNewIndividual"}


def test_derniers_fichiers_par_table_liste_vide():
    assert ds.derniers_fichiers_par_table([]) == {}


# --- synchroniser (client Drive entierement simule) -------------------------

class _FausseRequete:
    def __init__(self, resultat=None, contenu: bytes = b""):
        self._resultat = resultat
        self._contenu = contenu

    def execute(self):
        return self._resultat


class _FauxTelechargeur:
    """Simule googleapiclient.http.MediaIoBaseDownload : un seul next_chunk()
    suffit a ecrire tout le contenu simule dans le tampon fourni."""

    def __init__(self, tampon, requete, contenu: bytes):
        self._tampon = tampon
        self._contenu = contenu

    def next_chunk(self):
        self._tampon.write(self._contenu)
        return None, True


class _FauxFichiers:
    def __init__(self, fichiers_disponibles, contenus_par_id):
        self._fichiers_disponibles = fichiers_disponibles
        self._contenus_par_id = contenus_par_id

    def list(self, q=None, fields=None, pageToken=None, pageSize=None):
        return _FausseRequete({"files": self._fichiers_disponibles, "nextPageToken": None})

    def get_media(self, fileId):
        return ("requete_media", fileId)


class _FauxServiceDrive:
    """Simule l'objet renvoye par googleapiclient.discovery.build('drive', 'v3', ...)."""

    def __init__(self, fichiers_disponibles, contenus_par_id):
        self._fichiers = _FauxFichiers(fichiers_disponibles, contenus_par_id)
        self._contenus_par_id = contenus_par_id

    def files(self):
        return self._fichiers


@pytest.fixture(autouse=True)
def _telechargement_simule(monkeypatch):
    """Remplace drive_sync.telecharger_contenu (qui utilise en interne
    MediaIoBaseDownload) par une version qui lit directement dans le
    dictionnaire de contenus simules attache au faux service, pour eviter
    toute dependance a googleapiclient.http dans les tests."""

    def _faux_telecharger(service, file_id):
        return service._contenus_par_id[file_id]

    monkeypatch.setattr(ds, "telecharger_contenu", _faux_telecharger)


def test_synchroniser_renvoie_le_dernier_export_par_table():
    fichiers = [
        {"id": "1", "name": "FNewIndividual_2026-08-03.csv", "modifiedTime": "2026-08-03T10:00:00Z"},
        {"id": "2", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
    ]
    contenus_par_id = {"1": b"vieux_contenu", "2": b"nouveau_contenu"}
    service = _FauxServiceDrive(fichiers, contenus_par_id)

    contenus, meta, avertissements = ds.synchroniser(folder_id="dossier_test", service=service)

    assert avertissements == []
    assert contenus == {"FNewIndividual_2026-08-04.csv": b"nouveau_contenu"}
    assert meta["FNewIndividual_2026-08-04.csv"]["table"] == "FNewIndividual"
    assert meta["FNewIndividual_2026-08-04.csv"]["date_export"] == datetime(2026, 8, 4)


def test_synchroniser_plusieurs_tables():
    fichiers = [
        {"id": "1", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "2", "name": "FNewEducation_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
    ]
    contenus_par_id = {"1": b"contenu_individual", "2": b"contenu_education"}
    service = _FauxServiceDrive(fichiers, contenus_par_id)

    contenus, meta, avertissements = ds.synchroniser(folder_id="dossier_test", service=service)

    assert avertissements == []
    assert set(contenus.keys()) == {"FNewIndividual_2026-08-04.csv", "FNewEducation_2026-08-04.csv"}


def test_synchroniser_dossier_vide():
    service = _FauxServiceDrive([], {})
    contenus, meta, avertissements = ds.synchroniser(folder_id="dossier_test", service=service)
    assert contenus == {}
    assert meta == {}
    assert avertissements == []


def test_synchroniser_signale_un_echec_de_telechargement_sans_bloquer_les_autres(monkeypatch):
    fichiers = [
        {"id": "1", "name": "FNewIndividual_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
        {"id": "2", "name": "FNewEducation_2026-08-04.csv", "modifiedTime": "2026-08-04T10:00:00Z"},
    ]
    service = _FauxServiceDrive(fichiers, {"2": b"contenu_education"})

    def _telecharger_qui_echoue_parfois(service, file_id):
        if file_id == "1":
            raise RuntimeError("panne reseau simulee")
        return service._contenus_par_id[file_id]

    monkeypatch.setattr(ds, "telecharger_contenu", _telecharger_qui_echoue_parfois)

    contenus, meta, avertissements = ds.synchroniser(folder_id="dossier_test", service=service)

    assert "FNewEducation_2026-08-04.csv" in contenus
    assert "FNewIndividual_2026-08-04.csv" not in contenus
    assert len(avertissements) == 1
    assert "FNewIndividual" in avertissements[0]


# --- configuration (secrets / fichier local) --------------------------------

def test_en_dict_modifiable_convertit_recursivement():
    class FauxSecretsReadOnly(dict):
        def __setitem__(self, key, value):
            raise TypeError("Secrets does not support item assignment.")

    imbrique = FauxSecretsReadOnly({"a": FauxSecretsReadOnly({"b": "c"})})
    resultat = ds._en_dict_modifiable(imbrique)
    resultat["a"]["b"] = "modifie"  # ne doit pas lever d'exception
    assert resultat["a"]["b"] == "modifie"


def test_obtenir_config_service_account_leve_une_erreur_claire_sans_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(ds, "_obtenir_secrets_gdrive", lambda: None)
    monkeypatch.setattr(ds, "CONFIG_PATH", tmp_path / "service_account_inexistant.json")
    with pytest.raises(RuntimeError, match="compte de service"):
        ds.obtenir_config_service_account()


def test_obtenir_config_service_account_lit_le_fichier_local(monkeypatch, tmp_path):
    faux_fichier = tmp_path / "service_account.json"
    faux_fichier.write_text('{"type": "service_account", "client_email": "test@example.com"}', encoding="utf-8")
    monkeypatch.setattr(ds, "_obtenir_secrets_gdrive", lambda: None)
    monkeypatch.setattr(ds, "CONFIG_PATH", faux_fichier)
    config = ds.obtenir_config_service_account()
    assert config["client_email"] == "test@example.com"


def test_obtenir_folder_id_par_defaut(monkeypatch):
    monkeypatch.setattr(ds, "_obtenir_secrets_gdrive", lambda: None)
    monkeypatch.delenv("GDRIVE_FOLDER_ID", raising=False)
    assert ds.obtenir_folder_id() == ds.FOLDER_ID_PAR_DEFAUT


def test_obtenir_folder_id_variable_environnement(monkeypatch):
    monkeypatch.setattr(ds, "_obtenir_secrets_gdrive", lambda: None)
    monkeypatch.setenv("GDRIVE_FOLDER_ID", "un_autre_dossier")
    assert ds.obtenir_folder_id() == "un_autre_dossier"
