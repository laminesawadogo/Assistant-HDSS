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
import zipfile
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

# Le processus d'export externe peut organiser les exports de trois facons
# differentes, toutes reconnues automatiquement sans configuration :
#   1. des fichiers CSV/Excel/Stata directement dans le dossier, un par
#      table, avec la date d'export dans le nom ;
#   2. un SOUS-DOSSIER par export (ex. "export_2026-07-30_12-50-37"),
#      contenant les fichiers de chaque table pour ce jour-la (organisation
#      constatee en pratique sur le dossier "opo_db_exports" - Drive
#      propose de telecharger un dossier sous forme de .zip, ce qui peut
#      donner l'impression que l'export "est" un fichier zip) ;
#   3. une ARCHIVE .zip par export (a la racine du dossier ou dans un
#      sous-dossier), contenant les CSV de chaque table.
DOSSIER_MIME = "application/vnd.google-apps.folder"

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


def _parser_date_drive(valeur):
    """Convertit une date fournie par l'API Drive (chaine ISO 8601, ex.
    "2026-08-04T10:00:00.000Z") en datetime naif ; renvoie None si absente
    ou illisible plutot que de lever une exception sur une valeur inattendue."""
    if valeur is None or isinstance(valeur, datetime):
        return valeur
    try:
        return datetime.fromisoformat(str(valeur).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def _est_dossier(item: dict) -> bool:
    return item.get("mimeType") == DOSSIER_MIME


MIME_GOOGLE_SHEETS_NATIF = "application/vnd.google-apps.spreadsheet"


def _est_fichier_table(item: dict) -> bool:
    """Un fichier peut etre reconnu comme un export de table soit par son
    extension (CSV/Excel/Stata), soit - bug reel rencontre - parce que c'est
    un Google Sheets NATIF (`MIME_GOOGLE_SHEETS_NATIF`), qui peut ne porter
    AUCUNE extension reconnue dans son nom affiche (ex: un Sheets cree
    directement dans Drive, sans jamais avoir ete un fichier .xlsx, s'appelle
    souvent juste "Export OPO" sans ".xlsx"). Sans ce deuxieme critere, un tel
    fichier serait ignore des la resolution des candidats, avant meme
    d'atteindre le code qui sait pourtant le convertir via `exporter_contenu`."""
    nom = str(item.get("name", "")).lower()
    return nom.endswith(EXTENSIONS_RECONNUES) or item.get("mimeType") == MIME_GOOGLE_SHEETS_NATIF


def _plus_recent(items: list[dict]):
    """Choisit l'element le plus recent d'une liste (fichiers ou dossiers) :
    d'abord par date detectee dans son nom, sinon par date de derniere
    modification, sinon par date de creation Drive."""
    if not items:
        return None

    def cle(item):
        date_nom = extraire_date_export(item.get("name", ""))
        modifie = _parser_date_drive(item.get("modifiedTime") or item.get("createdTime"))
        return (date_nom or modifie or datetime.min, modifie or datetime.min)

    return max(items, key=cle)


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
        if not _est_fichier_table(f):
            continue
        table = nom_base_table(nom)
        date_nom = extraire_date_export(nom)
        modifie = _parser_date_drive(f.get("modifiedTime"))
        cle_tri = (date_nom or modifie or datetime.min, modifie or datetime.min)
        candidat = dict(f)
        candidat["table"] = table
        candidat["date_export"] = date_nom or f.get("date_lot")
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
    """Liste tous les elements (fichiers ET sous-dossiers) directement
    presents dans un dossier Drive donne (avec pagination, au cas ou le
    dossier depasse 200 elements a terme)."""
    elements: list[dict] = []
    page_token = None
    requete = f"'{folder_id}' in parents and trashed = false"
    while True:
        reponse = (
            service.files()
            .list(
                q=requete,
                fields="nextPageToken, files(id, name, modifiedTime, createdTime, mimeType)",
                pageToken=page_token,
                pageSize=200,
            )
            .execute()
        )
        elements.extend(reponse.get("files", []))
        page_token = reponse.get("nextPageToken")
        if not page_token:
            break
    return elements


def telecharger_contenu(service, file_id: str) -> bytes:
    from googleapiclient.http import MediaIoBaseDownload

    requete = service.files().get_media(fileId=file_id)
    tampon = io.BytesIO()
    telechargeur = MediaIoBaseDownload(tampon, requete)
    termine = False
    while not termine:
        _, termine = telechargeur.next_chunk()
    return tampon.getvalue()


# Mime-type Office standard vers lequel exporter un Google Sheets natif -
# c'est le meme format que `data_tools.load_table`/`charger_classeur` savent
# deja lire (pd.read_excel), aucun changement necessaire cote lecture.
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def exporter_contenu(service, file_id: str, mime_cible: str = MIME_XLSX) -> bytes:
    """Comme `telecharger_contenu`, mais pour un fichier Google natif (Sheets,
    Docs...) qui n'a pas de contenu binaire propre a telecharger (`get_media`
    echoue dessus) - utilise `export_media`, qui demande a Google de generer
    a la volee une conversion dans le format cible (xlsx par defaut)."""
    from googleapiclient.http import MediaIoBaseDownload

    requete = service.files().export_media(fileId=file_id, mimeType=mime_cible)
    tampon = io.BytesIO()
    telechargeur = MediaIoBaseDownload(tampon, requete)
    termine = False
    while not termine:
        _, termine = telechargeur.next_chunk()
    return tampon.getvalue()


def _extraire_tables_dune_archive(octets_zip: bytes) -> dict[str, bytes]:
    """Extrait, en memoire (sans jamais rien ecrire sur disque), les fichiers
    de table (CSV/Excel/Stata) contenus dans une archive .zip d'export.
    Ignore les dossiers internes, les fichiers caches et les artefacts
    macOS (__MACOSX) parfois presents dans un zip cree depuis un Mac."""
    resultat: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(octets_zip)) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            nom = Path(info.filename).name
            if not nom or nom.startswith(".") or "__MACOSX" in info.filename:
                continue
            if not nom.lower().endswith(EXTENSIONS_RECONNUES):
                continue
            resultat[nom] = archive.read(info)
    return resultat


def resoudre_elements_du_dernier_export(service, folder_id: str) -> tuple[list[dict], dict | None]:
    """Determine ou se trouvent les fichiers du DERNIER export, quelle que
    soit l'organisation retenue par le processus d'export externe (voir les
    trois organisations possibles documentees en tete de module) :

    1. Si le dossier contient des SOUS-DOSSIERS (un par export, ex.
       "export_2026-07-30_12-50-37"), le plus recent est choisi (date dans
       son nom, sinon date de modification/creation), et on regarde a
       l'interieur de CE sous-dossier plutot qu'a la racine.
    2. Parmi les elements ainsi retenus (racine ou sous-dossier choisi), si
       on trouve des fichiers de table (CSV/Excel/Stata) directement
       visibles, ils sont renvoyes tels quels (organisation "a plat").
    3. Sinon, si on trouve une archive .zip (et aucun fichier de table
       visible), c'est elle qui est renvoyee seule, a charge pour
       `synchroniser` de la telecharger et de l'extraire.

    Renvoie (elements_candidats, date_du_lot) : `date_du_lot` est la date
    detectee sur le sous-dossier d'export choisi (ou None si les fichiers
    sont directement a la racine sans regroupement par sous-dossier), pour
    permettre d'afficher une date d'export meme quand les fichiers
    individuels a l'interieur n'en portent pas dans leur propre nom."""
    elements = lister_fichiers_dossier(service, folder_id)
    sous_dossiers = [e for e in elements if _est_dossier(e)]
    fichiers_racine = [e for e in elements if not _est_dossier(e)]

    date_du_lot = None
    if sous_dossiers:
        dernier_dossier = _plus_recent(sous_dossiers)
        candidats = [e for e in lister_fichiers_dossier(service, dernier_dossier["id"]) if not _est_dossier(e)]
        date_du_lot = extraire_date_export(dernier_dossier.get("name", "")) or _parser_date_drive(
            dernier_dossier.get("modifiedTime") or dernier_dossier.get("createdTime")
        )
    else:
        candidats = fichiers_racine

    tables_visibles = [c for c in candidats if _est_fichier_table(c)]
    if tables_visibles:
        return tables_visibles, date_du_lot

    zips = [c for c in candidats if c.get("name", "").lower().endswith(".zip")]
    if zips:
        return [_plus_recent(zips)], date_du_lot

    return [], date_du_lot


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

    candidats, date_du_lot = resoudre_elements_du_dernier_export(service, fid)

    contenus: dict[str, bytes] = {}
    meta: dict[str, dict] = {}
    avertissements: list[str] = []

    if not candidats:
        return contenus, meta, avertissements

    est_archive = candidats[0].get("name", "").lower().endswith(".zip")
    if est_archive:
        archive = candidats[0]
        try:
            octets_zip = telecharger_contenu(service, archive["id"])
        except Exception as e:
            return {}, {}, [f"Impossible de télécharger l'archive {archive['name']} : {e}"]
        try:
            extraits = _extraire_tables_dune_archive(octets_zip)
        except zipfile.BadZipFile as e:
            return {}, {}, [f"Archive {archive['name']} illisible (fichier zip corrompu ?) : {e}"]
        date_archive = date_du_lot or extraire_date_export(archive.get("name", "")) or _parser_date_drive(
            archive.get("modifiedTime")
        )
        if not extraits:
            avertissements.append(
                f"L'archive {archive['name']} ne contient aucun fichier CSV/Excel/Stata reconnu."
            )
        for nom_fichier, contenu in extraits.items():
            contenus[nom_fichier] = contenu
            meta[nom_fichier] = {
                "id": archive["id"],
                "name": nom_fichier,
                "table": nom_base_table(nom_fichier),
                "date_export": date_archive,
                "modifiedTime": _parser_date_drive(archive.get("modifiedTime")),
            }
        return contenus, meta, avertissements

    for c in candidats:
        c.setdefault("date_lot", date_du_lot)
    retenus = derniers_fichiers_par_table(candidats)
    for table, info in sorted(retenus.items()):
        # Bug reel rencontre : un fichier depose dans Drive (upload d'un vrai
        # .xlsx existant, confirme par l'equipe) peut malgre tout finir en
        # Google Sheets NATIF cote Drive (mimeType application/vnd.google-
        # apps.spreadsheet, URL en docs.google.com/spreadsheets/... plutot que
        # drive.google.com/file/...) - meme si son nom affiche garde
        # l'extension .xlsx. Un tel fichier n'a pas de contenu binaire
        # telechargeable via `get_media` (utilise par `telecharger_contenu`),
        # qui echoue avec une erreur HTTP Google brute ("Only files with
        # binary content can be downloaded"). Pour un Sheets natif
        # specifiquement, on peut recuperer un .xlsx equivalent via
        # `export_media` (voir `exporter_contenu`) - aucune manipulation
        # supplementaire demandee a l'equipe, la conversion est transparente.
        # Les AUTRES formats Google natifs (Docs, Slides, Forms...) ne sont
        # jamais des donnees tabulaires exploitables : ceux-la restent
        # simplement signales par un avertissement, sans tentative de
        # conversion.
        mime = str(info.get("mimeType", ""))
        nom_fichier = info["name"]
        try:
            if mime == MIME_GOOGLE_SHEETS_NATIF:
                if not nom_fichier.lower().endswith((".xlsx", ".xls")):
                    nom_fichier = f"{nom_fichier}.xlsx"
                contenus[nom_fichier] = exporter_contenu(service, info["id"])
                meta[nom_fichier] = info
            elif mime.startswith("application/vnd.google-apps."):
                avertissements.append(
                    f"{info['name']} ({table}) est un document Google natif non tabulaire "
                    "(Docs/Slides/Forms...) et ne peut pas être lu comme une table de données."
                )
            else:
                contenus[nom_fichier] = telecharger_contenu(service, info["id"])
                meta[nom_fichier] = info
        except Exception as e:
            avertissements.append(f"Impossible de telecharger {info['name']} ({table}) : {e}")
    return contenus, meta, avertissements
