"""
Synchronisation automatique des tables depuis un dossier Google Drive.

Remplace le depot manuel de fichier dans l'application : les tables
analysees par l'assistant (indicateurs, echantillons, controles de
coherence, performance de terrain...) sont desormais chargees
automatiquement depuis un dossier Google Drive dans lequel une exportation
(CSV, Excel ou Stata) de chaque table est deposee chaque jour par un
processus externe a l'application. Comme plusieurs exports successifs d'une
meme table peuvent coexister dans le dossier (un nouveau fichier par jour,
avec sa date dans le nom), ce module ne retient que le plus recent par
table.

Acces securise : authentification par un compte de service Google (pas de
connexion interactive de l'utilisateur), dont la cle est chargee soit depuis
les secrets Streamlit (st.secrets["gdrive"]["service_account"], recommande
en production), soit depuis un fichier local `service_account.json` non
suivi par Git (usage local uniquement). Le dossier Drive doit etre partage
explicitement, en lecture seule, avec l'adresse e-mail de ce compte de
service - voir le README, section "Connexion automatique a Google Drive",
pour la procedure pas a pas.

Aucune donnee individuelle n'est modifiee ni supprimee sur le Drive : ce
module ne fait que LIRE les fichiers (scope drive.readonly).
"""
from __future__ import annotations

import io
import json
import os
import re
from datetime import datetime
from pathlib import Path

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Dossier "opo_db_exports" partage par l'equipe, extrait de l'URL de partage
# (https://drive.google.com/drive/folders/<ID>). Reste modifiable sans toucher
# au code via st.secrets["gdrive"]["folder_id"] ou la variable d'environnement
# GDRIVE_FOLDER_ID (utile si un autre dossier doit etre utilise plus tard).
FOLDER_ID_PAR_DEFAUT = "1qjV_hHhGIE5klnQYUzxLT-OJQp827v0l"

CONFIG_PATH = Path(__file__).parent / "service_account.json"

EXTENSIONS_RECONNUES = (".csv", ".xlsx", ".xls", ".dta")

# --- Detection de la date d'export a partir du nom de fichier -------------

# Chaque motif est essaye dans l'ordre ; le premier qui correspond gagne.
# Formats courants d'export automatique : 2026-08-04, 2026_08_04, 20260804,
# 04-08-2026... (jour/mois devine par la position du groupe a 4 chiffres).
_MOTIFS_DATE = [
    (re.compile(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), lambda m: (int(m[1]), int(m[2]), int(m[3]))),
    (re.compile(r"(?<!\d)(\d{2})[-_.](\d{2})[-_.](20\d{2})(?!\d)"), lambda m: (int(m[3]), int(m[2]), int(m[1]))),
]


def extraire_date_export(nom_fichier: str):
    """Cherche une date d'export dans le nom du fichier (plusieurs formats
    courants reconnus : AAAA-MM-JJ, AAAAMMJJ, JJ-MM-AAAA, avec separateurs
    -, _ ou .). Renvoie un datetime, ou None si aucune date n'est
    reconnaissable dans le nom - le tri se rabat alors sur la date de
    derniere modification fournie par Drive."""
    for motif, extracteur in _MOTIFS_DATE:
        m = motif.search(nom_fichier)
        if m:
            try:
                annee, mois, jour = extracteur(m)
                return datetime(annee, mois, jour)
            except ValueError:
                continue
    return None


def nom_base_table(nom_fichier: str) -> str:
    """Deduit le nom de table a partir du nom de fichier, en retirant
    l'extension et le fragment de date d'export, pour regrouper les exports
    successifs d'une meme table : FNewIndividual_2026-08-03.csv et
    FNewIndividual_2026-08-04.csv doivent etre reconnus comme deux exports de
    la MEME table, dont on ne garde ensuite que le plus recent."""
    base = re.sub(r"\.(csv|xlsx|xls|dta)$", "", nom_fichier, flags=re.IGNORECASE)
    for motif, _ in _MOTIFS_DATE:
        base = motif.sub("", base)
    base = re.sub(r"[_\-.\s]{2,}", "_", base)
    base = base.strip("_-. ")
    return base or nom_fichier


def derniers_fichiers_par_table(fichiers: list[dict]) -> dict[str, dict]:
    """`fichiers` : liste de dicts {"id", "name", "modifiedTime"} tels que
    renvoyes par l'API Drive (ou un mock de test) - "modifiedTime" peut etre
    une chaine ISO 8601 ou un objet datetime deja converti.

    Ne garde, pour chaque table deduite du nom de fichier (voir
    `nom_base_table`), que l'export le plus recent : d'abord par date
    detectee dans le nom du fichier, puis par date de derniere modification
    Drive en cas d'egalite ou d'absence de date exploitable dans le nom.
    Les fichiers dont l'extension n'est pas reconnue (ni CSV, Excel ou
    Stata) sont ignores silencieusement (ce ne sont pas des exports de
    table, ex : un mode d'emploi PDF depose dans le meme dossier par erreur).

    Renvoie un dict {nom_de_table: metadonnees_du_fichier_retenu}."""
    retenus: dict[str, dict] = {}
    for f in fichiers:
        nom = f.get("name", "")
        if not nom.lower().endswith(EXTENSIONS_RECONNUES):
            continue
        table = nom_base_table(nom)
        date_nom = extraire_date_export(nom)
        modifie = f.get("modifiedTime")
        if isinstance(modifie, str):
            try:
                modifie = datetime.fromisoformat(modifie.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                modifie = None
        cle_tri = (date_nom or modifie or datetime.min, modifie or datetime.min)
        candidat = dict(f)
        candidat["table"] = table
        candidat["date_export"] = date_nom
        candidat["modifiedTime"] = modifie
        candidat["_cle_tri"] = cle_tri
        actuel = retenus.get(table)
        if actuel is None or candidat["_cle_tri"] > actuel["_cle_tri"]:
            retenus[table] = candidat
    return retenus


# --- Configuration (secrets Streamlit ou fichier local) --------------------

def _en_dict_modifiable(objet):
    """st.secrets est en lecture seule a TOUS les niveaux d'imbrication (pas
    seulement au premier) : une conversion recursive est necessaire avant de
    pouvoir passer ces valeurs a une bibliotheque qui les manipule comme un
    dict Python ordinaire (meme precaution que auth._en_dict_modifiable)."""
    if hasattr(objet, "items"):
        return {cle: _en_dict_modifiable(valeur) for cle, valeur in objet.items()}
    if isinstance(objet, (list, tuple)):
        return [_en_dict_modifiable(v) for v in objet]
    return objet


def _obtenir_secrets_gdrive():
    try:
        import streamlit as st
        return st.secrets.get("gdrive")
    except Exception:
        return None


def obtenir_config_service_account() -> dict:
    """Charge les identifiants du compte de service Google, en priorite
    depuis les secrets Streamlit (deploiement), sinon depuis un fichier local
    `service_account.json` non suivi par Git (usage local). Leve une
    RuntimeError explicite si aucune des deux sources n'est configuree,
    plutot que de laisser l'appel Google echouer avec une erreur technique
    peu comprehensible pour une equipe non specialiste."""
    secrets_gdrive = _obtenir_secrets_gdrive()
    if secrets_gdrive and secrets_gdrive.get("service_account"):
        return _en_dict_modifiable(secrets_gdrive["service_account"])
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    raise RuntimeError(
        "Aucun identifiant de compte de service Google configure (ni "
        "st.secrets['gdrive']['service_account'], ni service_account.json local). "
        "Voir le README, section « Connexion automatique a Google Drive »."
    )


def obtenir_folder_id() -> str:
    secrets_gdrive = _obtenir_secrets_gdrive()
    if secrets_gdrive and secrets_gdrive.get("folder_id"):
        return secrets_gdrive["folder_id"]
    return os.environ.get("GDRIVE_FOLDER_ID", FOLDER_ID_PAR_DEFAUT)


# --- Appels a l'API Google Drive -------------------------------------------

def construire_service():
    """Construit le client de l'API Drive a partir des identifiants du
    compte de service. Import differe des bibliotheques Google (au lieu d'un
    import en tete de module) pour que l'ensemble du reste du module -
    notamment les fonctions pures testees unitairement ci-dessus - reste
    utilisable meme dans un environnement de test ou ces bibliotheques ne
    seraient pas installees."""
    # Les identifiants sont verifies AVANT d'importer les bibliotheques
    # Google (assez lourdes a l'import - httplib2/pyparsing) : en l'absence
    # de configuration (environnement de test, ou instance locale pas encore
    # configuree), l'erreur explicite est levee immediatement, sans payer ce
    # cout d'import pour rien.
    info = obtenir_config_service_account()

    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    creds = Credentials.from_service_account_info(info, scopes=SCOPES)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def lister_fichiers_dossier(service, folder_id: str) -> list[dict]:
    """Liste tous les fichiers directement presents dans le dossier Drive
    (avec pagination, au cas ou le dossier depasse 200 fichiers a terme)."""
    fichiers: list[dict] = []
    page_token = None
    requete = f"'{folder_id}' in parents and trashed = false"
    while True:
        reponse = (
            service.files()
            .list(
                q=requete,
                fields="nextPageToken, files(id, name, modifiedTime, mimeType)",
                pageToken=page_token,
                pageSize=200,
            )
            .execute()
        )
        fichiers.extend(reponse.get("files", []))
        page_token = reponse.get("nextPageToken")
        if not page_token:
            break
    return fichiers


def telecharger_contenu(service, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    requete = service.files().get_media(fileId=file_id)
    tampon = io.BytesIO()
    telechargeur = MediaIoBaseDownload(tampon, requete)
    termine = False
    while not termine:
        _, termine = telechargeur.next_chunk()
    return tampon.getvalue()


# --- Point d'entree principal ----------------------------------------------

def synchroniser(folder_id: str | None = None, service=None):
    """Recupere, pour chaque table detectee dans le dossier Drive, le
    contenu binaire de son export le plus recent.

    `service` peut etre fourni directement (utilise par les tests, pour
    injecter un client Drive simule sans appel reseau reel) ; sinon un
    client reel est construit a partir des identifiants configures.

    Renvoie un tuple (contenus, metadonnees, avertissements) :
      - contenus : dict {nom_de_fichier: contenu_bytes} des derniers exports ;
      - metadonnees : dict {nom_de_fichier: infos} (table, date_export, modifiedTime...) ;
      - avertissements : liste de messages texte (fichier illisible, etc.) -
        l'echec du telechargement d'UN export ne doit jamais empecher le
        chargement des autres tables.

    Peut lever RuntimeError si les identifiants du compte de service ne sont
    pas configures, ou l'exception native de l'API Google en cas de probleme
    de connexion ou de droits d'acces (dossier non partage avec le compte de
    service, par exemple) - a l'appelant (interface) de les intercepter pour
    afficher un message clair plutot que de laisser l'application planter.
    """
    service = service or construire_service()
    fid = folder_id or obtenir_folder_id()
    fichiers = lister_fichiers_dossier(service, fid)
    retenus = derniers_fichiers_par_table(fichiers)

    contenus: dict[str, bytes] = {}
    meta: dict[str, dict] = {}
    avertissements: list[str] = []
    for table, info in sorted(retenus.items()):
        try:
            contenus[info["name"]] = telecharger_contenu(service, info["id"])
            meta[info["name"]] = info
        except Exception as e:
            avertissements.append(f"Impossible de telecharger {info['name']} ({table}) : {e}")
    return contenus, meta, avertissements
