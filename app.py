"""
Interface web de l'Assistant OPO — chat RAG sur le dictionnaire de donnees +
analyse d'une table deposee (indicateurs, echantillon, controles de
coherence). Lancer avec : streamlit run app.py
"""

import os
import re
import tempfile
import traceback
import unicodedata
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import auth
import data_tools as dt
import drive_sync
import ingest
import rag


@st.cache_data(ttl=900, show_spinner="Synchronisation avec Google Drive...")
def _telecharger_depuis_drive():
    """Recupere depuis le dossier Google Drive de l'observatoire le dernier
    export de chaque table (voir drive_sync.py). Mis en cache 15 minutes :
    Streamlit relance tout le script a chaque question posee dans le chat,
    et les exports Drive n'ont lieu qu'une fois par jour - interroger
    l'API Drive a chaque interaction serait a la fois inutile et lent. Le
    bouton "Recharger depuis Google Drive maintenant" (barre laterale) vide
    ce cache pour forcer une verification immediate si besoin."""
    return drive_sync.synchroniser()


def sans_accents(texte: str) -> str:
    """Retire les accents d'un texte, pour que la reconnaissance de mots-cles
    ne depende jamais de la presence ou non d'un accent (ex: "cohérence" tape
    par l'equipe ne matchait jamais le mot-cle "coheren" en ASCII pur - meme
    principe pour "repartition"/"répartition", "echantillon"/"échantillon"...).
    Applique a la question ET aux mots-cles au moment de la comparaison, pour
    que n'importe quelle orthographe (avec ou sans accent) soit reconnue."""
    return "".join(c for c in unicodedata.normalize("NFD", texte) if unicodedata.category(c) != "Mn")


def contient_mot_cle(q_normalise: str, mots: list[str], entiers: bool = False) -> bool:
    """Vérifie si un des mots-cles (accentues ou non) apparait dans une
    question deja normalisee (minuscules + sans accents via `sans_accents`).

    `entiers=True` exige une correspondance de MOT ENTIER (frontieres de mot)
    plutot qu'une simple sous-chaine - necessaire pour des mots courts qui
    risqueraient sinon de matcher a tort a l'interieur d'un autre mot plus
    long (ex: "relation" ne doit pas matcher dans "corrélation", "lien" ne
    doit pas matcher dans "italien"). Par defaut (False), une simple
    sous-chaine suffit - utile pour reconnaitre plusieurs formes d'un meme
    radical en une seule entree (ex: "envoy" pour envoyer/envoyé/envoie)."""
    for m in mots:
        motif = sans_accents(m.lower())
        if entiers:
            if re.search(r"\b" + re.escape(motif) + r"\b", q_normalise):
                return True
        elif motif in q_normalise:
            return True
    return False

# Charge la cle du modele (GROQ_API_KEY / ANTHROPIC_API_KEY) depuis un fichier
# .env place a cote de ce fichier, pour que la cle soit configuree une seule
# fois "en arriere-plan" (deploiement sur un serveur/VPS). Sans fichier .env,
# rien ne se passe : l'appli retombe sur les champs de saisie manuelle.
load_dotenv(Path(__file__).parent / ".env")

# Sur Streamlit Community Cloud, la cle est plutot configuree via le
# gestionnaire de secrets de la plateforme (Settings -> Secrets), qui la rend
# disponible via st.secrets plutot que comme variable d'environnement
# classique. On la reexporte ici pour que tout le reste du code (rag.py)
# continue de fonctionner de la meme facon, quel que soit l'hebergement.
for _cle in ("ANTHROPIC_API_KEY", "GROQ_API_KEY"):
    if not os.getenv(_cle):
        try:
            _valeur = st.secrets.get(_cle)
        except Exception:
            _valeur = None
        if _valeur:
            os.environ[_cle] = _valeur

# Icone d'onglet du navigateur : utilise le logo ISSP s'il a ete depose
# (voir auth.LOGO_PATH), sinon retombe sur l'emoji par defaut - ne bloque
# jamais le demarrage si le logo n'est pas encore fourni.
_icone_page = str(auth.LOGO_PATH) if auth.LOGO_PATH.exists() else "📊"
st.set_page_config(page_title="Assistant OPO", page_icon=_icone_page, layout="wide")

# Ecran de connexion : bloque tout le reste de la page (st.stop() dans
# auth.verifier_acces) tant que l'utilisateur n'est pas authentifie. Doit
# rester le tout premier element d'interface apres set_page_config, pour
# qu'aucune donnee (tables, chat, documents) ne soit accessible sans compte
# valide - voir auth_config.yaml pour la gestion des comptes.
identite_utilisateur = auth.verifier_acces()

# --- Préparation silencieuse de l'index ---------------------------------------
# L'index de recherche doit exister ET être à jour avant de pouvoir répondre
# a une question documentaire. Plutot que d'exposer un bouton technique
# "construire l'index" a l'equipe, on le (re)construit automatiquement des
# que necessaire (utile notamment sur un hebergement neuf, ex: Streamlit
# Community Cloud, ET quand un document est depose directement dans
# data/source_documents/ sans passer par le bouton d'upload de l'interface -
# voir ingest.index_obsolete, qui corrige un vrai bug ou l'index ne se
# reconstruisait plus jamais une fois cree la premiere fois, meme perime).
if ingest.index_obsolete():
    with st.spinner("Préparation de l'assistant (indexation des documents)..."):
        try:
            ingest.build_index()
            rag.reset_cache()
        except Exception:
            st.error(
                "L'assistant n'a pas pu s'initialiser correctement. "
                "Contacte la personne responsable du déploiement."
            )
            st.stop()

# --- Identité visuelle -------------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --opo-navy: #0B2545;
        --opo-jaune: #E8A33D;
        --opo-jaune-clair: #F4C874;
        --opo-sand: #F4EDE4;
    }
    .opo-header {
        background: linear-gradient(135deg, var(--opo-jaune) 0%, var(--opo-jaune-clair) 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(11, 37, 69, 0.25);
    }
    .opo-header h1 {
        color: var(--opo-navy);
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .opo-header p {
        color: rgba(11, 37, 69, 0.8);
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .opo-badge {
        display: inline-block;
        background: rgba(11, 37, 69, 0.12);
        color: var(--opo-navy);
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        margin-top: 0.6rem;
    }
    section[data-testid="stSidebar"] {
        background-color: var(--opo-sand);
    }
    .opo-footer {
        margin-top: 2.5rem;
        padding-top: 0.8rem;
        border-top: 1px solid rgba(11, 37, 69, 0.15);
        color: #6b7280;
        font-size: 0.8rem;
        text-align: center;
    }
    </style>
    <div class="opo-header">
        <h1>📊 Assistant IA — Observatoire de Population de Ouagadougou</h1>
        <p>Dictionnaire de données, indicateurs, échantillonnage et contrôle de cohérence.</p>
        <span class="opo-badge">Version bêta · Équipe OPO</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Barre laterale ---------------------------------------------------------

with st.sidebar:
    st.header("🔑 Configuration du modèle")

    cle_anthropic_env = os.getenv("ANTHROPIC_API_KEY")
    cle_groq_env = os.getenv("GROQ_API_KEY")

    if cle_anthropic_env or cle_groq_env:
        # Une clé est déjà chargée depuis le fichier .env (voir .env.exemple) :
        # on ne redemande rien à l'équipe, personne n'a besoin de saisir ni de
        # voir la clé pour utiliser l'assistant.
        fournisseur = "Anthropic (Claude)" if cle_anthropic_env else "Groq"
        st.success(f"Modèle configuré automatiquement — {fournisseur}")
        groq_key_input = ""
        anthropic_key_input = ""
    else:
        st.caption(
            "⚠️ Aucune clé n'est configurée en arrière-plan (fichier `.env` absent ou vide — "
            "voir `.env.exemple`). En attendant, tu peux en saisir une ici pour tester ; elle "
            "n'est utilisée que pour cette session, jamais enregistrée."
        )
        anthropic_key_input = st.text_input(
            "Clé Anthropic — recommandé (rendu le plus proche de Claude)", type="password", key="anthropic_key_input"
        )
        groq_key_input = st.text_input("Clé Groq — alternative gratuite, plus rapide", type="password", key="groq_key_input")

        if rag.has_llm_configured(groq_key_input, anthropic_key_input):
            st.success("Modèle configuré pour cette session")
        else:
            st.warning("Aucune clé : l'assistant ne fait qu'afficher du contexte brut pour l'instant")

    st.divider()
    st.header("📁 Tables / bases de données")
    st.caption(
        "Les tables analysables (indicateurs, échantillons, doublons, performance de terrain...) "
        "sont chargées **automatiquement** depuis le dossier Google Drive de l'observatoire, où "
        "un export (CSV, Excel ou Stata) de chaque table est déposé chaque jour. Aucun dépôt "
        "manuel de fichier n'est nécessaire : seul le dernier export de chaque table est retenu."
    )

    if "tables" not in st.session_state:
        st.session_state["tables"] = {}
    if "fichiers_traites" not in st.session_state:
        st.session_state["fichiers_traites"] = set()  # (nom, taille) déjà chargés
    if "historique_chargements" not in st.session_state:
        st.session_state["historique_chargements"] = []  # trace chaque (re)chargement de table

    if st.button("🔄 Recharger depuis Google Drive maintenant"):
        _telecharger_depuis_drive.clear()

    try:
        contenus_drive, meta_drive, avertissements_drive = _telecharger_depuis_drive()
    except Exception as e:
        contenus_drive, meta_drive, avertissements_drive = {}, {}, []
        st.error(
            "Connexion à Google Drive impossible pour l'instant : "
            f"{e} — le chat reste utilisable pour les questions sur le dictionnaire, mais "
            "aucune table n'est disponible tant que la connexion n'est pas rétablie."
        )

    for avert in avertissements_drive:
        st.warning(avert)

    # Panneau de depannage affiche DIRECTEMENT dans l'application - plus
    # accessible pour l'equipe que d'aller chercher les logs du serveur
    # d'hebergement (souvent difficiles a localiser sans habitude technique).
    # Montre, sans aucun filtrage, tout ce que le compte de service voit
    # reellement dans le dossier configure : permet de distinguer d'un coup
    # d'oeil "le dossier est bien vu mais vide/sans fichier reconnu" de "le
    # dossier lui-meme n'est pas accessible" - les deux se manifestent sinon
    # de la meme facon cote equipe ("rien ne se charge, aucune erreur").
    with st.expander("🔧 Diagnostic Drive (dépannage)"):
        if st.button("Lancer le diagnostic", key="diagnostic_drive"):
            diagnostic = drive_sync.diagnostiquer()
            st.write(f"**Dossier interrogé (folder_id)** : `{diagnostic['folder_id']}`")
            if not diagnostic["credentials_ok"]:
                st.error(f"Connexion impossible : {diagnostic['erreur']}")
            elif diagnostic["erreur"]:
                st.error(f"Dossier interrogé mais erreur lors de la lecture : {diagnostic['erreur']}")
            elif not diagnostic["elements_bruts"]:
                st.warning(
                    "Connexion réussie, mais **aucun élément trouvé** dans ce dossier — vérifie que "
                    "le fichier est bien déplacé (pas seulement en raccourci) directement dans CE "
                    "dossier précis, et que le compte de service y a bien accès."
                )
            else:
                st.success(f"{len(diagnostic['elements_bruts'])} élément(s) trouvé(s) dans ce dossier :")
                for e in diagnostic["elements_bruts"]:
                    st.write(f"- **{e['name']}** — type : `{e['mimeType']}`")

    # Meme controle anti-relecture que pour un depot manuel : Streamlit
    # relance tout le script a chaque interaction (chaque question posee
    # dans le chat), et un export dont le nom (donc la date) n'a pas change
    # depuis la derniere synchronisation ne doit pas etre reparse a chaque
    # fois - seul un nouvel export (nom de fichier different, date plus
    # recente) declenche un nouveau chargement.
    for nom_fichier, contenu in contenus_drive.items():
        signature = (nom_fichier, len(contenu))
        if signature in st.session_state["fichiers_traites"]:
            continue
        info = meta_drive.get(nom_fichier, {})
        try:
            nom_bas = nom_fichier.lower()
            if nom_bas.endswith((".xlsx", ".xls")):
                suffix = ".xlsx"
            elif nom_bas.endswith(".dta"):
                suffix = ".dta"
            else:
                suffix = ".csv"
            # delete=False est necessaire pour pouvoir rouvrir le fichier par
            # son chemin (pd.read_excel/read_stata...) une fois ferme ; le
            # nettoyage est fait explicitement dans le bloc `finally`
            # juste en dessous, pour ne jamais laisser une donnee
            # synchronisee depuis le Drive trainer sur le disque du serveur
            # au-dela du temps de son chargement (securite des donnees).
            tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
            try:
                with open(tmp_path, "wb") as f:
                    f.write(contenu)

                horodatage = datetime.now()
                date_export = info.get("date_export")
                if dt.est_classeur_excel(nom_fichier):
                    # Un classeur Excel peut contenir plusieurs feuilles, chacune
                    # une table distincte (ex: une feuille par table de
                    # l'observatoire) : on les reconnait toutes, pas seulement
                    # la premiere.
                    feuilles = dt.charger_classeur(tmp_path)
                    for nom_feuille, df_feuille in feuilles.items():
                        nom_table = nom_feuille if nom_feuille not in st.session_state["tables"] else (
                            f"{Path(nom_fichier).stem}_{nom_feuille}"
                        )
                        st.session_state["tables"][nom_table] = df_feuille
                        st.session_state["historique_chargements"].append({
                            "horodatage": horodatage, "fichier": nom_fichier,
                            "table": nom_table, "n_lignes": len(df_feuille),
                            "date_export": date_export, "source": "Google Drive",
                        })
                else:
                    nom_table = info.get("table") or re.sub(
                        r"\.(csv|xlsx|xls|dta)$", "", nom_fichier, flags=re.IGNORECASE
                    )
                    st.session_state["tables"][nom_table] = dt.load_table(tmp_path)
                    st.session_state["historique_chargements"].append({
                        "horodatage": horodatage, "fichier": nom_fichier,
                        "table": nom_table, "n_lignes": len(st.session_state["tables"][nom_table]),
                        "date_export": date_export, "source": "Google Drive",
                    })

                st.session_state["fichiers_traites"].add(signature)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            st.error(f"Impossible de lire {nom_fichier} (Google Drive) : {e}")

    tables = st.session_state["tables"]
    if tables:
        st.success(f"{len(tables)} table(s) chargée(s) depuis Google Drive : {', '.join(tables.keys())}")
        st.caption(
            "Les colonnes de type nom/prénom sont automatiquement retirées. "
            "**Toutes les tables chargées sont ouvertes par défaut, aucune n'est \"active\" par "
            "défaut** : mentionne une colonne (ex. « répartition de sex ») ou le nom d'une table dans "
            "ta question pour cibler une table précise, sinon l'assistant cherche automatiquement dans "
            "toutes les tables concernées plutôt que d'en choisir une au hasard."
        )
        with st.expander("Voir les colonnes et la date d'export de chaque table chargée"):
            for nom, df_apercu in tables.items():
                derniere_entree = next(
                    (e for e in reversed(st.session_state["historique_chargements"]) if e["table"] == nom),
                    None,
                )
                suffixe_date = ""
                if derniere_entree and derniere_entree.get("date_export"):
                    suffixe_date = f" — export du {derniere_entree['date_export'].strftime('%d/%m/%Y')}"
                st.markdown(f"**{nom}**{suffixe_date} : {', '.join(f'`{c}`' for c in df_apercu.columns)}")
    else:
        st.info(
            "Aucune table disponible depuis Google Drive pour l'instant — le chat répond "
            "depuis le dictionnaire."
        )

    st.divider()
    st.header("📚 Ajouter un document de référence")
    st.caption(
        "Dépose ici une fiche, un manuel, une présentation... (PDF, Word, PowerPoint, texte). "
        "Il est ajouté **en permanence** à la base de connaissances : toute l'équipe pourra "
        "ensuite poser des questions dessus, y compris dans une autre conversation."
    )

    if "documents_traites" not in st.session_state:
        st.session_state["documents_traites"] = set()  # (nom, taille) déjà ajoutés

    nouveaux_documents = st.file_uploader(
        "Document(s) à ajouter",
        type=["pdf", "doc", "docx", "ppt", "pptx", "txt", "md"],
        accept_multiple_files=True,
        key="uploader_documents",
    )

    if nouveaux_documents:
        a_traiter = [
            doc for doc in nouveaux_documents
            if (doc.name, doc.size) not in st.session_state["documents_traites"]
        ]
        if a_traiter:
            with st.spinner(f"Ajout de {len(a_traiter)} document(s) à la base de connaissances..."):
                dossier_source = Path(__file__).parent / "data" / "source_documents"
                dossier_source.mkdir(parents=True, exist_ok=True)
                for doc in a_traiter:
                    (dossier_source / doc.name).write_bytes(doc.getbuffer())
                try:
                    ingest.build_index()
                    rag.reset_cache()
                    for doc in a_traiter:
                        st.session_state["documents_traites"].add((doc.name, doc.size))
                    st.success(
                        f"{len(a_traiter)} document(s) ajouté(s) et indexé(s) : "
                        f"{', '.join(d.name for d in a_traiter)}"
                    )
                except Exception as e:
                    st.error(f"Échec de l'indexation : {e}")

    st.divider()
    st.header("🖼️ Lire une image")
    st.caption(
        "Dépose une photo ou un scan (ex : fiche terrain remplie) et pose une question dessus. "
        "Nécessite une clé Anthropic active (lecture d'image non gérée par Groq)."
    )
    image_deposee = st.file_uploader("Image (PNG, JPG)", type=["png", "jpg", "jpeg"], key="uploader_image")
    question_image = st.text_input(
        "Question sur l'image", key="question_image", placeholder="Ex : que contient cette fiche ?"
    )
    if st.button("🔍 Analyser l'image") and image_deposee is not None:
        with st.spinner("Analyse de l'image en cours..."):
            reponse_image = rag.analyser_image(
                image_deposee.getvalue(),
                image_deposee.type or "image/png",
                question_image,
                anthropic_key=anthropic_key_input,
            )
        st.session_state["messages"].append(
            {"role": "user", "content": f"[Image déposée : {image_deposee.name}] {question_image}".strip()}
        )
        st.session_state["messages"].append({"role": "assistant", "content": reponse_image})
        # Pas de st.rerun() ici : ce bloc est deja dans la meme execution de
        # script que la boucle d'affichage des messages plus bas (ligne
        # ~452) - les deux nouveaux messages s'affichent donc naturellement
        # a la suite, sans avoir besoin de relancer le script. Un st.rerun()
        # juste apres une modification de session_state, dans une appli qui
        # utilise aussi st.chat_input, est un des declencheurs connus du bug
        # frontend Streamlit "Failed to execute 'removeChild' on 'Node'"
        # (deux rendus qui se chevauchent avant que le navigateur ait fini
        # de reconcilier le premier) - a eviter sauf necessite reelle.

# --- Historique de chat ------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state["messages"] = []


FORMATS_EXPORT = (
    ("csv", "text/csv", "csv"),
    ("xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"),
    ("dta", "application/octet-stream", "dta"),
)


def calculer_exports(table) -> dict:
    """Calcule une seule fois les bytes CSV/Excel/Stata d'une table, pour ne
    pas refaire ce travail a chaque rerun Streamlit (l'appli relance tout le
    script a chaque question posee dans le chat)."""
    exports = {}
    for fmt, _, _ in FORMATS_EXPORT:
        try:
            exports[fmt] = dt.exporter(table, fmt)
        except Exception as e:
            exports[fmt] = None
            exports[f"{fmt}_erreur"] = str(e)
    return exports


def afficher_boutons_export(exports: dict, label: str, cle: str):
    """Affiche trois boutons de telechargement a partir de bytes deja calcules."""
    colonnes = st.columns(3)
    for col, (fmt, mime, ext) in zip(colonnes, FORMATS_EXPORT):
        with col:
            data = exports.get(fmt)
            if data is not None:
                st.download_button(
                    f"⬇️ {fmt.upper()}", data=data, file_name=f"{label}.{ext}",
                    mime=mime, key=f"dl_{fmt}_{cle}",
                )
            else:
                st.caption(f"Export {fmt.upper()} indisponible : {exports.get(f'{fmt}_erreur', '')}")


def afficher_bouton_docx(docx_bytes: bytes, label: str, cle: str):
    """Bouton de telechargement dedie au rapport Word (performance de
    terrain) - separe des exports csv/xlsx/dta puisqu'il ne s'agit pas d'un
    export brut d'une table mais d'un rapport mis en forme."""
    st.download_button(
        "⬇️ Rapport Word (.docx)", data=docx_bytes, file_name=f"{label}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"dl_docx_{cle}",
    )


for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources utilisées"):
                for s in msg["sources"]:
                    st.markdown(f"- ({s['score']:.2f}) {s['text']}")
        if msg.get("exports") is not None:
            afficher_boutons_export(msg["exports"], msg.get("table_label", "export"), cle=f"hist_{i}")
        if msg.get("docx_bytes") is not None:
            afficher_bouton_docx(msg["docx_bytes"], msg.get("docx_label", "rapport"), cle=f"hist_{i}")
        if msg.get("chart_data") is not None:
            st.caption("Courbe de progression (cumul de fiches par jour)")
            st.line_chart(msg["chart_data"])


def historique_recent(max_tours: int = 6) -> list[dict]:
    """Renvoie les derniers echanges de la conversation (hors la question en
    cours de traitement, deja ajoutee a l'historique juste avant l'appel a
    route_question), pour permettre les questions de suivi qui font reference
    a ce qui vient d'etre dit ("et pour l'autre table ?", "peux-tu detailler ?")."""
    messages = st.session_state.get("messages", [])
    precedents = messages[:-1] if messages else []
    recents = precedents[-max_tours:]
    return [{"role": m["role"], "contenu": m["content"]} for m in recents]


# Avec entiers=True (mot entier, voir contient_mot_cle), chaque accord
# grammatical (singulier/pluriel) doit etre liste explicitement - c'est ce qui
# evite qu'un mot comme "relation" ne matche a tort a l'interieur de
# "corrélation" (voir MOTS_CORRELATION), mais ca veut dire que "relations"
# (pluriel) ne matche plus via "relation" seul : il faut lister les deux.
MOTS_RELATION = [
    "relation", "relations", "reliee", "reliees", "relie", "relies",
    "lien", "liens", "liees", "en commun", "cle commune", "cles communes",
    "clé commune", "clés communes",
]
MOTS_FUSION = ["fusion", "fusionner", "fusionne", "jointure", "joindre", "joins", "merge", "merger"]
MOTS_DIFFERENCE = [
    "mais pas dans", "et pas dans", "pas dans", "absent de", "absents de", "absente de", "absentes de",
    "sauf", "n'est pas dans", "ne sont pas dans", "n'apparaissent pas dans", "n'apparait pas dans",
    "n'apparaît pas dans", "non dans", "ni dans", "non present dans", "non presents dans",
    "non presente dans", "non presentes dans", "non inclus dans", "non incluses dans",
]
MOTS_VICE_VERSA = ["vice versa", "vice-versa", "et inversement", "et l'inverse", "et réciproquement", "et reciproquement"]

# Une reponse generique/hesitante du LLM ("je n'ai pas d'environnement
# d'execution...") peut pousser l'equipe a relancer avec une phrase courte qui
# ne repete PAS le mot-cle d'action initial (ex: "il faut analyser
# directement" apres une question de difference) : sans memoire de l'action
# demandee, cette relance retombe elle aussi sur le LLM au lieu de declencher
# le vrai calcul deterministe. On detecte ce type de relance et on va
# rechercher, dans le dernier message utilisateur de l'historique, le mot-cle
# d'action qui avait ete utilise pour la question initiale.
MOTS_RELANCE_CALCUL = [
    "directement", "vraiment", "réellement", "reellement", "un vrai", "une vraie",
    "résultat exact", "resultat exact", "le vrai nombre", "le vrai chiffre",
    "pas une estimation", "fais le calcul", "fais-le", "fais le", "exécute", "execute",
    "sois précis", "sois precis", "pour de vrai", "concrètement", "concretement",
    "analyse les données", "analyse les donnees", "calcule-le", "calcule le",
]


def action_deja_demandee_dans_historique(
    historique: list[dict] | None, mots_action: list[str], entiers: bool = False
) -> bool:
    """Vérifie si un message RECENT de l'UTILISATEUR (pas de l'assistant, dont
    les propres réponses répètent souvent des mots comme "directement") dans
    l'historique contenait déjà un des mots-clés d'une action (difference,
    fusion, relation) - pour qu'une relance courte qui ne repete pas ce
    mot-cle ("il faut analyser directement") redeclenche bien le meme calcul
    deterministe plutôt que de retomber sur une réponse générique du LLM."""
    if not historique:
        return False
    for message in reversed(historique):
        if message.get("role") != "user":
            continue
        contenu = sans_accents(str(message.get("contenu", "")).lower())
        if contient_mot_cle(contenu, mots_action, entiers=entiers):
            return True
    return False


def resoudre_paire_tables(tables_mentionnees: list[str], tables: dict) -> tuple[str, str] | None:
    """Tente de resoudre automatiquement la paire de tables visee par une
    question de difference/fusion, meme si une seule (ou aucune) n'est
    explicitement nommee dans la question - pour ne pas obliger l'equipe a
    toujours ecrire une phrase du type "combien sont dans X mais pas dans Y"
    avec les deux noms explicites : si une seule autre table est chargee en
    tout, la reponse est evidente et ne merite pas une question de relance."""
    if len(tables_mentionnees) >= 2:
        return tables_mentionnees[0], tables_mentionnees[1]
    if len(tables_mentionnees) == 1 and len(tables) == 2:
        autres = [n for n in tables if n != tables_mentionnees[0]]
        if len(autres) == 1:
            return tables_mentionnees[0], autres[0]
    return None


def message_precision_tables(verbe: str, tables: dict, tables_mentionnees: list[str]) -> str:
    """Message de relance dynamique - utilise les VRAIES tables actuellement
    chargees (jamais un exemple fige type "Presence"/"Education" qui pourrait
    ne pas exister) - quand une question de difference/fusion ne permet pas
    de determiner sans ambiguite les deux tables visees. Si une seule table
    est deja identifiee, ne redemande que l'autre, parmi les tables reellement
    chargees, plutot que de tout redemander depuis le debut."""
    if not tables:
        return "Aucune table n'est chargée pour l'instant : dépose d'abord les fichiers à comparer."
    if len(tables_mentionnees) == 1:
        seule = tables_mentionnees[0]
        autres = [n for n in tables if n != seule]
        return (
            f"Tu veux {verbe} **{seule}** avec laquelle des autres tables chargées ? "
            f"{', '.join(f'**{n}**' for n in autres)}."
        )
    return (
        f"Précise les deux tables à {verbe}, parmi celles chargées : "
        f"{', '.join(f'**{n}**' for n in tables)}."
    )


def message_precision_colonne(tables: dict) -> str:
    """Message de relance dynamique quand une analyse necessitant une ou
    plusieurs colonnes (repartition, tableau croise, correlation,
    multivariee) ne peut identifier AUCUNE colonne mentionnee dans AUCUNE
    des tables chargees, et qu'aucune table n'est nommee non plus : liste
    les vraies tables et leurs vraies colonnes, plutot que de deviner une
    table au hasard ("il ne faut pas lire une seule base par defaut")."""
    lignes = ["Précise sur quelle colonne (et éventuellement quelle table) travailler. Tables chargées :"]
    for nom, df in tables.items():
        apercu = ", ".join(f"`{c}`" for c in df.columns[:12])
        if len(df.columns) > 12:
            apercu += ", ..."
        lignes.append(f"- **{nom}** : {apercu}")
    return "\n".join(lignes)


def formater_rapport_coherence(rapport: dict, nom_table: str) -> str:
    """Met en forme le rapport de cohérence en indiquant explicitement quelles
    colonnes ont été vérifiées (précision : évite un "aucune anomalie" ambigu
    qui pourrait laisser croire à un contrôle exhaustif si en réalité aucune
    colonne d'ID ou de date n'a été détectée dans la table)."""
    lignes = [f"**Rapport de cohérence** sur **{nom_table}** ({rapport['n_lignes']} lignes) :"]

    id_verifiees = rapport.get("colonnes_id_verifiees", [])
    date_verifiees = rapport.get("colonnes_date_verifiees", [])
    lignes.append(
        "_Colonnes vérifiées — identifiants : "
        + (", ".join(f"`{c}`" for c in id_verifiees) if id_verifiees else "aucune détectée")
        + " ; dates : "
        + (", ".join(f"`{c}`" for c in date_verifiees) if date_verifiees else "aucune détectée")
        + "._"
    )

    if rapport["anomalies"]:
        for k, v in rapport["anomalies"].items():
            lignes.append(f"- {k} : {v} cas")
    else:
        lignes.append("- Aucune anomalie détectée sur les colonnes vérifiées ci-dessus.")

    lignes.append(
        "\n_Rappel : ceci est un signalement, pas une correction automatique. "
        "La validation reste réservée aux personnes habilitées._"
    )
    return "\n".join(lignes)


def formater_rapport_coherence_avancee(rapport: dict) -> str:
    """Met en forme le catalogue de controles de coherence avances
    (dt.rapport_coherence_avancee) : ce qui a ete verifie table par table, ce
    qui a ete ignore faute de colonnes reconnues (transparence), et les
    controles croises entre tables (eligibilite presence <-> education/
    emploi/..., deces <-> presence, etc.)."""
    if not rapport["par_table"] and not rapport["croises"]:
        return (
            "Aucun contrôle avancé n'a pu s'appliquer aux tables actuellement chargées "
            "(colonnes attendues non reconnues automatiquement)."
        )

    lignes = ["**Audit de cohérence avancé** (catalogue de contrôles spécifiques à l'observatoire) :"]

    for nom_table, details in rapport["par_table"].items():
        lignes.append(f"\n**{nom_table}**")
        for libelle, resultat in details["controles_ok"]:
            n = resultat.get("n_anomalies", 0)
            detail_txt = " ; ".join(resultat.get("detail", [])) if resultat.get("detail") else ""
            if n > 0:
                lignes.append(f"- ⚠️ {libelle} : **{n} cas**" + (f" ({detail_txt})" if detail_txt else ""))
            else:
                lignes.append(f"- ✅ {libelle} : aucune anomalie")
        if details["controles_ignores"]:
            lignes.append(
                "- _Non applicable ici (colonnes non détectées) : "
                + ", ".join(details["controles_ignores"]) + "._"
            )

    if rapport["croises"]:
        lignes.append("\n**Contrôles croisés entre tables**")
        for libelle, nom_a, nom_b, resultat in rapport["croises"]:
            if "n_eligibles_sans_fiche" in resultat:
                lignes.append(
                    f"- {libelle} (**{nom_a}** ↔ **{nom_b}**) : "
                    f"**{resultat['n_eligibles_sans_fiche']}** éligible(s) sans fiche, "
                    f"**{resultat['n_fiche_sans_eligibilite']}** avec fiche sans être éligible."
                )
            else:
                n = resultat.get("n_anomalies", 0)
                symbole = "⚠️" if n > 0 else "✅"
                lignes.append(f"- {symbole} {libelle} (**{nom_a}** ↔ **{nom_b}**) : **{n} cas**")

    lignes.append(
        "\n_Rappel : signalement basé sur une détection automatique des colonnes pertinentes par leur nom — "
        "pas une correction automatique, la validation reste réservée aux personnes habilitées. Certains "
        "contrôles très spécifiques au questionnaire (codes de réponse détaillés) ne sont pas encore couverts "
        "s'ils n'ont pas pu être reconnus automatiquement._"
    )
    return "\n".join(lignes)


def formater_reponse_requete(resultat: dict, nom_table: str) -> dict:
    """Met en forme le resultat de `dt.executer_requete_donnees` (action
    REQUETE du classifieur) - calcul precis sur les donnees reelles
    (compter/lister/moyenne/somme/min/max, avec filtres), par opposition aux
    4 analyses fixes (repartition/echantillon/doublons/coherence)."""
    filtres_txt = (
        " (" + ", ".join(resultat["filtres_appliques"]) + ")" if resultat.get("filtres_appliques") else ""
    )
    operation = resultat["operation"]

    explication_filtres = (
        " en appliquant le(s) filtre(s) demandé(s) (" + ", ".join(resultat["filtres_appliques"]) + ")"
        if resultat.get("filtres_appliques") else " sans filtre"
    )

    if operation == "compter":
        contenu = _habiller_reponse(
            f"**{resultat['resultat']}** ligne(s) dans **{nom_table}**{filtres_txt}.",
            intro=(
                f"Réponse directe : **{resultat['resultat']}** ligne(s)."
                if not resultat.get("filtres_appliques")
                else f"En appliquant ta condition, je trouve **{resultat['resultat']}** ligne(s) correspondantes."
            ),
            explication=f"j'ai compté les lignes de **{nom_table}**{explication_filtres}.",
            suggestions=[f"la liste détaillée de ces lignes plutôt que le seul décompte"],
        )
        return {"content": contenu}

    if operation == "lister":
        table_resultat = resultat["resultat"]
        apercu = "aperçu des 50 premières" if resultat["n_total"] > 50 else "liste complète"
        contenu = (
            f"**{resultat['n_total']}** ligne(s) trouvée(s) dans **{nom_table}**{filtres_txt} ({apercu}) :\n\n"
            + table_resultat.to_markdown(index=False)
        )
        contenu = _habiller_reponse(
            contenu,
            intro=(
                f"J'ai trouvé **{resultat['n_total']}** ligne(s) correspondant à ta demande dans **{nom_table}**"
                + (f", je t'affiche les 50 premières" if resultat["n_total"] > 50 else "") + "."
            ),
            explication=f"j'ai sélectionné les lignes de **{nom_table}**{explication_filtres}.",
            suggestions=[f"le décompte seul plutôt que le détail", f"une moyenne/somme sur une colonne numérique de **{nom_table}** pour ce même sous-groupe"],
        )
        return {"content": contenu, "table": table_resultat, "table_label": f"requete_{nom_table}"}

    if resultat["resultat"] is None:
        return {
            "content": (
                f"Aucune valeur numérique exploitable pour `{resultat.get('colonne_cible')}` "
                f"dans **{nom_table}**{filtres_txt}."
            )
        }
    libelles = {"moyenne": "Moyenne", "somme": "Somme", "min": "Minimum", "max": "Maximum"}
    contenu = (
        f"**{libelles[operation]}** de `{resultat['colonne_cible']}` dans **{nom_table}**{filtres_txt} : "
        f"**{resultat['resultat']:.2f}** (calculé sur {resultat['n_valeurs']} valeur(s))."
    )
    contenu = _habiller_reponse(
        contenu,
        intro=f"La {libelles[operation].lower()} de `{resultat['colonne_cible']}` demandée est de **{resultat['resultat']:.2f}**.",
        explication=(
            f"j'ai calculé {libelles[operation].lower()} de `{resultat['colonne_cible']}` sur les "
            f"{resultat['n_valeurs']} valeur(s) numériques exploitables de **{nom_table}**{explication_filtres} "
            "(les valeurs manquantes ou non numériques ont été ignorées, pas comptées comme des zéros)."
        ),
        suggestions=[f"la répartition de `{resultat['colonne_cible']}` pour voir sa distribution complète"],
    )
    return {"content": contenu}


def tenter_requete_donnees_multi_table(
    question: str, tables: dict, groq_key: str | None, anthropic_key: str | None
) -> dict | None:
    """Tente de repondre a une question de calcul precis (compter/lister/
    moyenne/somme/min/max, avec filtres) meme quand aucune table n'a pu etre
    resolue explicitement (voir `dt.resoudre_table_ciblee`) - en interrogeant
    le classifieur LLM avec l'union de TOUTES les colonnes chargees, puis en
    executant la requete sur CHAQUE table qui possede reellement les colonnes
    necessaires (jamais une seule table par defaut, meme principe que le
    reste de l'assistant). Renvoie None si le classifieur ne renvoie pas
    REQUETE ou si aucune table ne convient, pour laisser l'appelant retomber
    sur son propre repli (message de precision ou recherche documentaire)."""
    toutes_colonnes, vues = [], set()
    for df in tables.values():
        for c in df.columns:
            if str(c).lower() not in vues:
                vues.add(str(c).lower())
                toutes_colonnes.append(str(c))
    if not toutes_colonnes:
        return None

    action, parametre = rag.classifier_intention(
        question, toutes_colonnes, groq_key=groq_key, anthropic_key=anthropic_key
    )
    if action != "REQUETE" or not parametre:
        return None

    colonnes_necessaires = [f["colonne"] for f in (parametre.get("filtres") or []) if f.get("colonne")]
    if parametre.get("colonne_cible"):
        colonnes_necessaires.append(parametre["colonne_cible"])
    if not colonnes_necessaires:
        return None

    morceaux, table_resultat, label_resultat = [], None, None
    for nom, df in tables.items():
        if not all(c in df.columns for c in colonnes_necessaires):
            continue
        try:
            resultat = dt.executer_requete_donnees(
                df, parametre.get("operation"), parametre.get("colonne_cible"), parametre.get("filtres")
            )
        except dt.RequeteInvalide:
            continue
        reponse = formater_reponse_requete(resultat, nom)
        morceaux.append(reponse["content"])
        if "table" in reponse:
            table_resultat, label_resultat = reponse["table"], reponse["table_label"]

    if not morceaux:
        return None
    resultat_final = {"content": "\n\n---\n\n".join(morceaux)}
    if table_resultat is not None:
        resultat_final["table"], resultat_final["table_label"] = table_resultat, label_resultat
    return resultat_final


# Identifiants REELS documentes dans le schema relationnel de l'observatoire
# - source canonique unique : `dt.IDENTIFIANTS_REELS_DOCUMENTES` (deplace
# dans data_tools.py pour que les fonctions de jointure deterministe
# `detecter_cle_jointure`/`fusionner_tables`/`relation_entre_tables`, qui n'y
# avaient plus acces depuis app.py, partagent la MEME liste que celle
# utilisee ici pour la generation de requetes SQL - evite toute divergence
# entre les deux). Garde un alias local pour ne rien casser des usages
# existants dans ce fichier.
IDENTIFIANTS_REELS_DOCUMENTES = dt.IDENTIFIANTS_REELS_DOCUMENTES


def _description_schema(tables: dict) -> str:
    """Decrit le schema (nom de table + colonnes, PUIS indices de jointure)
    de TOUTES les tables chargees, pour le fournir au LLM dans
    `tenter_requete_sql` - c'est ce schema REEL, et lui seul, qui doit guider
    la requete SQL generee (jamais un nom de table/colonne invente).

    Les indices de jointure (colonnes reellement communes a deux tables,
    voir `dt.detecter_cles_communes`) reduisent le risque d'une jointure
    inventee ou faite a tort - notamment le piege classique d'une colonne
    "id" partagee par hasard entre deux tables SANS lien reel entre elles
    (chaque table du schema reel a sa propre cle primaire locale "id" - ce
    n'est jamais, dans ce schema, une reference vers une autre table),
    volontairement exclue de ces indices.

    Distingue en plus, parmi les colonnes communes restantes, celles qui
    correspondent a un identifiant CONFIRME par le dictionnaire de donnees
    (`IDENTIFIANTS_REELS_DOCUMENTES`) de celles qui n'ont qu'un nom evocateur
    (ex: "menage_id") sans etre documentees comme une vraie cle - consigne
    explicite de l'observatoire : ne jamais se fier a un nom de colonne qui
    "ressemble" a un identifiant, seule la documentation fait foi."""
    lignes = [f"- {nom}({', '.join(str(c) for c in df.columns)})" for nom, df in tables.items()]

    indices_confirmes, indices_a_verifier = [], []
    for (a, b), communes in dt.detecter_cles_communes(tables).items():
        candidates = [c for c in communes if str(c).strip().lower() != "id"]
        confirmes = [c for c in candidates if str(c).strip().lower() in IDENTIFIANTS_REELS_DOCUMENTES]
        a_verifier = [c for c in candidates if c not in confirmes]
        if confirmes:
            indices_confirmes.append(f"- {a} <-> {b} sur : {', '.join(confirmes)}")
        if a_verifier:
            indices_a_verifier.append(f"- {a} <-> {b} sur : {', '.join(a_verifier)}")

    if indices_confirmes:
        lignes.append(
            "\nColonnes de jointure CONFIRMÉES par le dictionnaire de données de l'observatoire "
            "(à utiliser en priorité) :"
        )
        lignes.extend(indices_confirmes)
    if indices_a_verifier:
        lignes.append(
            "\nAutres colonnes communes détectées automatiquement, mais NON confirmées comme de "
            "vraies clés par le dictionnaire (un nom qui ressemble à un identifiant, ex: \"id\", "
            "\"menage_id\", \"round_id\", \"enquete_id\", n'en est PAS la preuve) - INTERDIT de les "
            "utiliser comme clé de JOIN, MÊME en dernier recours, MÊME si aucune colonne confirmée "
            "ci-dessus ne permet de répondre à la question. Si seules ces colonnes non confirmées "
            "relient les tables nécessaires, ne fabrique PAS de JOIN dessus : réponds plutôt que tu "
            "ne peux pas relier ces tables de façon fiable avec le schéma actuellement documenté. "
            "Listées ici seulement à titre informatif :"
        )
        lignes.extend(indices_a_verifier)

    return "\n".join(lignes)


def _contexte_dictionnaire_pour_sql(question: str, k: int = 4) -> str | None:
    """Va chercher, dans le dictionnaire de donnees/manuels/fiches deja
    indexes (`rag.retrieve`), les extraits les plus pertinents pour la
    question - l'observatoire y documente deja precisement le sens de
    chaque identifiant et les tables qu'il relie (ex: respondid = repondant,
    locationid = UCH, socialgpid = menage, individid = individu,
    observationid = observation d'un individu dans le menage). Ce contexte
    est transmis a `rag.generer_requete_sql` en plus des indices purement
    structurels (`_description_schema`) : il donne au LLM une source
    autorisee sur le sens des colonnes, pas seulement une coincidence de nom.

    Renvoie None si l'index n'existe pas encore ou si la recherche echoue -
    ne doit jamais faire echouer `tenter_requete_sql` pour autant, la requete
    SQL reste generable avec le seul schema structurel."""
    try:
        chunks = rag.retrieve(question, k=k)
    except Exception:
        return None
    if not chunks:
        return None
    return "\n".join(f"- ({c.get('source', '?')}) {c['text']}" for c in chunks)


def tenter_requete_sql(
    question: str, tables: dict, groq_key: str | None, anthropic_key: str | None,
    historique: list[dict] | None = None,
) -> dict | None:
    """Repli GENERAL pour une question qui necessite de croiser PLUSIEURS
    tables a la fois (2, 3, 4 ou plus) - au-dela de ce que couvrent l'action
    REQUETE mono-table (`tenter_requete_donnees_multi_table`) ou le controle
    croise fixe deces/depart (`reponse_statut_croise_dans_table`).

    Le LLM ecrit une requete SQL en lecture seule a partir du schema REEL de
    toutes les tables chargees (`_description_schema`, avec indices de
    jointure), executee via DuckDB directement sur les DataFrame en memoire
    (`dt.executer_sql`, qui revalide que la requete est bien un SELECT avant
    toute execution - jamais d'execution de code LLM arbitraire).

    Trois garde-fous supplementaires pour rester fiable :
    - Contexte dictionnaire : les extraits pertinents du dictionnaire de
      donnees/manuels/fiches deja indexes (`_contexte_dictionnaire_pour_sql`)
      sont transmis au LLM en plus du schema structurel - l'observatoire y
      documente deja le sens exact des identifiants (respondid, locationid,
      socialgpid, individid, observationid, etc.) et les tables qu'ils
      relient, une source plus fiable qu'une simple coincidence de nom de
      colonne.
    - Auto-correction en un aller-retour : si la premiere requete echoue a
      l'execution (colonne/syntaxe), l'erreur DuckDB est renvoyee au LLM pour
      une deuxieme tentative avant d'abandonner.
    - Alerte de sur-jointure : si le resultat contient plus de lignes que la
      plus grande table utilisee, un avertissement est ajoute (indice d'une
      jointure qui multiplie les lignes au lieu de les relier correctement -
      jamais une simple table par defaut a la place d'un vrai controle).

    Renvoie None si aucune cle LLM n'est configuree, si le modele ne propose
    rien d'exploitable, ou si les deux tentatives echouent - l'appelant
    retombe alors sur son propre repli."""
    if not tables or not rag.has_llm_configured(groq_key, anthropic_key):
        return None

    schema = _description_schema(tables)
    contexte_dictionnaire = _contexte_dictionnaire_pour_sql(question)
    requete_sql, erreur = None, None
    for tentative in range(2):
        requete_precedente = requete_sql
        requete_sql = rag.generer_requete_sql(
            question, schema, groq_key=groq_key, anthropic_key=anthropic_key,
            tentative_precedente=requete_precedente, erreur_precedente=erreur,
            contexte_dictionnaire=contexte_dictionnaire, historique=historique,
        )
        if not requete_sql:
            return None
        try:
            resultat = dt.executer_sql(tables, requete_sql)
            erreur = None
            break
        except dt.RequeteSQLInvalide as e:
            erreur = str(e)
            if tentative == 1:
                return None

    # Tables reellement mentionnees dans la requete generee (heuristique par
    # nom, juste pour rendre l'explication concrete - la validite de la
    # requete elle-meme est deja garantie par dt.executer_sql, pas par ceci).
    tables_utilisees = [nom for nom in tables if re.search(rf"\b{re.escape(nom)}\b", requete_sql, re.IGNORECASE)]
    tables_txt = ", ".join(f"**{n}**" for n in tables_utilisees) if tables_utilisees else "les tables concernées"

    explication_sql = (
        f"ta question demandait de croiser plusieurs tables, donc j'ai écrit et exécuté (en lecture "
        f"seule) une requête SQL directement sur {tables_txt}, à partir de leur schéma réel et des "
        "clés de jointure documentées par l'observatoire — elle est affichée ci-dessus pour que tu "
        "puisses la vérifier ou la réutiliser telle quelle."
    )

    if resultat.empty:
        contenu = f"Aucun résultat trouvé.\n\n_Requête utilisée (sur les données réellement chargées) :_ `{requete_sql}`"
        contenu = _habiller_reponse(
            contenu,
            intro=f"J'ai interrogé {tables_txt}, mais aucune ligne ne correspond à ta question.",
            explication=explication_sql,
            suggestions=["la même question avec un critère moins strict (ex: sans le filtre de date)"],
        )
        return {"content": contenu}

    contenu = (
        f"{resultat.to_markdown(index=False)}\n\n"
        f"_Requête utilisée (sur les données réellement chargées) :_ `{requete_sql}`"
    )
    plus_grande_table = max((len(df) for df in tables.values()), default=0)
    surjointure = bool(plus_grande_table and len(resultat) > plus_grande_table)
    if surjointure:
        contenu += (
            "\n\n⚠️ _Ce résultat contient plus de lignes que la plus grande table utilisée : "
            "vérifie la jointure avant de t'y fier (indice possible d'une jointure qui multiplie "
            "les lignes au lieu de les relier correctement)._"
        )
    contenu = _habiller_reponse(
        contenu,
        intro=(
            f"En croisant {tables_txt}, j'obtiens **{len(resultat)}** ligne(s)."
            + (" ⚠️ Ce chiffre me semble suspect, voir l'avertissement ci-dessous." if surjointure else "")
        ),
        explication=explication_sql,
        suggestions=["une autre question sur ces mêmes tables croisées", "un contrôle de cohérence sur l'une des tables utilisées"],
    )
    return {"content": contenu, "table": resultat, "table_label": "requete_sql"}


# Les quatre fonctions ci-dessous centralisent le calcul ET la syntaxe R/Stata
# equivalente pour chaque operation sur une table (repartition, echantillon,
# doublons, coherence), pour que les DEUX chemins qui y menent (mots-cles
# directs, et repli via classifier_intention) donnent exactement la meme
# reponse complete - plutot que de dupliquer la logique a deux endroits et
# risquer qu'elle diverge (l'un avec la syntaxe, l'autre sans).


def _habiller_reponse(
    contenu: str, explication: str | None = None,
    suggestions: list[str] | None = None, intro: str | None = None,
) -> str:
    """Enrichit le contenu brut d'une reponse (label + tableau/chiffre, deja
    construit par l'appelant) avec une phrase d'introduction AVANT le
    resultat, une explication en langage clair de la methode utilisee, et des
    suggestions de questions de suivi - demande explicite de l'equipe OPO :
    ne jamais renvoyer un tableau ou un chiffre brut sans phrase qui
    l'introduit ni sans dire comment il a ete obtenu (utile aux personnes qui
    ne maitrisent pas les cles techniques), et proposer, comme le fait
    Claude a la fin de ses reponses, une ou deux pistes de question
    suivante plutot que de laisser la conversation s'arreter net.

    `contenu` porte deja, le cas echeant, le tableau ET la syntaxe R/Stata
    reproductible (dt.syntaxe_*) : cette fonction ne fait qu'ENCADRER ce
    contenu, jamais le reconstruire ni le dupliquer."""
    morceaux = []
    if intro:
        morceaux.append(intro)
    morceaux.append(contenu)
    if explication:
        morceaux.append(f"**Comment ce résultat a été obtenu :** {explication}")
    if suggestions:
        puces = "\n".join(f"- {s}" for s in suggestions)
        morceaux.append(f"**Tu peux aussi demander :**\n{puces}")
    return "\n\n".join(morceaux)


def reponse_repartition(df, nom_table: str, colonne: str) -> dict:
    rep = dt.repartition(df, colonne)
    contenu = (
        f"Répartition de `{colonne}` dans **{nom_table}** :\n\n{rep.to_markdown()}"
        f"\n\n{dt.syntaxe_repartition(nom_table, colonne)}"
    )
    n_valeurs = len(rep)
    valeur_frequente, effectif_frequent, part_frequente = rep.index[0], int(rep["effectif"].iloc[0]), rep["pourcentage"].iloc[0]
    autre_colonne = next((c for c in df.columns if c != colonne), None)
    contenu = _habiller_reponse(
        contenu,
        intro=(
            f"`{colonne}` prend **{n_valeurs}** valeur(s) distincte(s) dans **{nom_table}** "
            f"({len(df)} ligne(s) au total) — la plus fréquente est **{valeur_frequente}**, "
            f"portée par {effectif_frequent} ligne(s) (soit {part_frequente:.1f} % de la table)."
        ),
        explication=(
            f"pour chaque valeur distincte de `{colonne}`, j'ai compté combien de lignes de "
            f"**{nom_table}** la portent — c'est un simple dénombrement sur la colonne entière, sans "
            f"filtre (équivalent à `df['{colonne}'].value_counts()` en Python, ou `tab {colonne}` en "
            f"Stata/R)."
        ),
        suggestions=[
            (
                f"le croisement entre `{colonne}` et `{autre_colonne}` (ex: \"tableau croisé "
                f"{colonne} et {autre_colonne}\")" if autre_colonne
                else f"un contrôle de cohérence sur **{nom_table}**"
            ),
            f"la même répartition sur un sous-groupe précis (ex: \"répartition de {colonne} chez les femmes\")",
        ],
    )
    return {"content": contenu, "table": rep.reset_index(), "table_label": f"repartition_{colonne}_{nom_table}"}


def reponse_echantillon(df, nom_table: str, n: int) -> dict:
    ech = dt.echantillon(df, n=n)
    contenu = (
        f"Échantillon reproductible de {len(ech)} lignes (graine fixée) issu de **{nom_table}** :\n\n"
        f"{ech.to_markdown(index=False)}\n\n{dt.syntaxe_echantillon(nom_table, len(ech), seed=20260729)}"
    )
    contenu = _habiller_reponse(
        contenu,
        intro=(
            f"Voici {len(ech)} ligne(s) tirée(s) au hasard parmi les {len(df)} de **{nom_table}** "
            f"({df.shape[1]} colonnes chacune)."
        ),
        explication=(
            f"j'ai utilisé un tirage aléatoire (`sample`) avec une graine fixe (20260729) plutôt qu'un "
            "simple `head()` : le tirage n'est donc pas biaisé vers le début du fichier, et relancer "
            "exactement la même demande te redonnera le même échantillon (reproductibilité)."
        ),
        suggestions=[
            f"un échantillon plus grand sur **{nom_table}**",
            f"un contrôle de cohérence sur **{nom_table}** pour vérifier que rien d'anormal ne s'y cache",
        ],
    )
    return {"content": contenu, "table": ech, "table_label": f"echantillon_{nom_table}"}


def reponse_doublons(df, nom_table: str) -> dict:
    colonnes_id = dt.detect_id_columns(df)
    if not colonnes_id:
        return {"content": f"Aucune colonne d'identifiant détectée automatiquement dans **{nom_table}**."}
    colonne = colonnes_id[0]
    dups = dt.doublons(df, colonne)
    if len(dups) == 0:
        return {"content": f"Aucun doublon d'identifiant détecté dans **{nom_table}** (colonne `{colonne}`)."}
    n_identifiants_dupliques = dups[colonne].nunique()
    contenu = (
        f"**{len(dups)} lignes en doublon** trouvées dans **{nom_table}** (colonne `{colonne}`) :\n\n"
        f"{dups.to_markdown(index=False)}\n\n{dt.syntaxe_doublons(nom_table, colonne)}"
    )
    contenu = _habiller_reponse(
        contenu,
        intro=(
            f"J'ai trouvé **{n_identifiants_dupliques}** valeur(s) de `{colonne}` répétée(s) plus "
            f"d'une fois dans **{nom_table}**, représentant {len(dups)} ligne(s) au total sur "
            f"{len(df)} — à vérifier : ressaisie accidentelle de la même fiche, ou plusieurs "
            f"événements réels qui partagent le même identifiant ?"
        ),
        explication=(
            f"j'ai détecté automatiquement `{colonne}` comme colonne d'identifiant de **{nom_table}**, "
            f"compté combien de fois chaque valeur y apparaît, et gardé toutes les lignes dont "
            f"l'identifiant apparaît plus d'une fois (donc chaque doublon est bien listé avec ses "
            f"copies, pas seulement signalé)."
        ),
        suggestions=[
            f"un contrôle de cohérence complet sur **{nom_table}** (dates, autres colonnes)",
            f"la fiche complète de l'un de ces identifiants (ex: \"recherche l'identifiant {dups[colonne].iloc[0]}\")",
        ],
    )
    return {"content": contenu, "table": dups, "table_label": f"doublons_{nom_table}"}


def reponse_coherence(df, nom_table: str) -> dict:
    rapport = dt.rapport_coherence(df)
    contenu = (
        formater_rapport_coherence(rapport, nom_table)
        + "\n\n"
        + dt.syntaxe_coherence(
            nom_table, rapport.get("colonnes_id_verifiees", []), rapport.get("colonnes_date_verifiees", [])
        )
    )
    anomalies = rapport.get("anomalies") or {}
    total_anomalies = sum(anomalies.values())
    if total_anomalies:
        pire = max(anomalies, key=anomalies.get)
        intro = (
            f"J'ai trouvé **{total_anomalies} anomalie(s)** au total dans **{nom_table}** — la plus "
            f"fréquente est « {pire} » ({anomalies[pire]} cas). Rien de bloquant automatiquement, mais "
            "à faire vérifier par une personne habilitée."
        )
    else:
        intro = (
            f"Bonne nouvelle : **{nom_table}** ne présente aucune anomalie sur les colonnes vérifiées "
            "ci-dessous."
        )
    contenu = _habiller_reponse(
        contenu,
        intro=intro,
        explication=(
            "j'ai cherché les doublons sur les colonnes d'identifiant détectées automatiquement, et "
            "vérifié les colonnes de date reconnues (dates hors d'une plage plausible, ou dans un "
            "ordre incohérent entre deux dates liées) — uniquement sur les colonnes listées ci-dessus, "
            "jamais sur une colonne non reconnue avec certitude."
        ),
        suggestions=[
            "l'audit de cohérence avancé (catalogue de contrôles spécifiques à l'observatoire, au-delà des doublons/dates génériques)",
        ],
    )
    return {"content": contenu}


def reponse_tableau_croise(df, nom_table: str, colonne1: str, colonne2: str) -> dict:
    tab = dt.tableau_croise(df, colonne1, colonne2)
    contenu = (
        f"Tableau croisé (analyse bivariée) de `{colonne1}` et `{colonne2}` dans **{nom_table}** :\n\n"
        f"{tab.to_markdown()}\n\n{dt.syntaxe_tableau_croise(nom_table, colonne1, colonne2)}"
    )
    # Cellule la plus peuplee HORS marges "Total" - la combinaison la plus
    # frequente entre les deux colonnes, plus parlant qu'un simple "voici le
    # tableau" pour quelqu'un qui doit d'abord reperer ou regarder.
    sous_tableau = tab.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
    effectifs_empiles = sous_tableau.stack() if not sous_tableau.empty else None
    cellule_max = effectifs_empiles.idxmax() if effectifs_empiles is not None and not effectifs_empiles.empty else None
    contenu = _habiller_reponse(
        contenu,
        intro=(
            (
                f"La combinaison la plus fréquente est `{colonne1}`={cellule_max[0]} avec "
                f"`{colonne2}`={cellule_max[1]} ({int(effectifs_empiles.max())} ligne(s)) dans "
                f"**{nom_table}** — voir le détail complet ci-dessous."
            ) if cellule_max is not None
            else f"Voici le croisement entre `{colonne1}` et `{colonne2}` dans **{nom_table}**."
        ),
        explication=(
            f"j'ai construit un tableau de contingence : pour chaque combinaison de valeurs de "
            f"`{colonne1}` (en ligne) et `{colonne2}` (en colonne), j'ai compté le nombre de lignes de "
            f"**{nom_table}** concernées, avec les totaux en marge."
        ),
        suggestions=[
            f"une matrice de corrélation si `{colonne1}` et `{colonne2}` sont numériques",
            f"la répartition de `{colonne1}` seule, sans croisement",
        ],
    )
    return {
        "content": contenu, "table": tab.reset_index(),
        "table_label": f"croise_{colonne1}_{colonne2}_{nom_table}",
    }


def reponse_correlation(df, nom_table: str, colonnes: list[str] | None) -> dict:
    mat = dt.matrice_correlation(df, colonnes)
    cols_utilisees = list(mat.columns)
    contenu = (
        f"Matrice de corrélation (analyse multivariée) dans **{nom_table}** "
        f"sur : {', '.join(f'`{c}`' for c in cols_utilisees)} :\n\n"
        f"{mat.to_markdown()}\n\n{dt.syntaxe_correlation(nom_table, cols_utilisees)}"
    )
    # Paire la plus fortement correlee (en valeur absolue), hors diagonale
    # (toujours 1.0 avec elle-meme, pas d'interet) - ce qui interesse
    # concretement, plutot que la matrice brute seule. `stack()` (sans
    # `dropna=False`) ignore nativement les NaN qu'on vient de poser sur la
    # diagonale, pas besoin de numpy pour la masquer.
    sans_diagonale = mat.copy()
    for c in cols_utilisees:
        sans_diagonale.loc[c, c] = None
    effectifs_abs = sans_diagonale.abs().stack()
    paire_max = effectifs_abs.idxmax() if not effectifs_abs.empty else None
    if paire_max is not None:
        c1, c2 = paire_max
        valeur = mat.loc[c1, c2]
        sens = "dans le même sens" if valeur > 0 else "en sens opposé"
        intro = (
            f"Sur les {len(cols_utilisees)} colonnes numériques testées de **{nom_table}**, la paire "
            f"la plus liée est `{c1}` / `{c2}` (corrélation de {valeur:.2f}, {sens})."
        )
    else:
        intro = f"Voici la matrice de corrélation calculée sur **{nom_table}**."
    contenu = _habiller_reponse(
        contenu,
        intro=intro,
        explication=(
            "j'ai calculé le coefficient de corrélation de Pearson entre chaque paire de colonnes "
            "numériques listées : proche de +1, les deux colonnes augmentent ensemble ; proche de -1, "
            "l'une augmente quand l'autre diminue ; proche de 0, pas de lien linéaire détecté (une "
            "corrélation ne prouve jamais un lien de cause à effet)."
        ),
        suggestions=[f"un tableau croisé entre `{cols_utilisees[0]}` et une autre colonne de **{nom_table}**"] if cols_utilisees else None,
    )
    return {"content": contenu, "table": mat.reset_index(), "table_label": f"correlation_{nom_table}"}


def reponse_tableau_multivarie(df, nom_table: str, colonnes: list[str]) -> dict:
    tab = dt.tableau_multivarie(df, colonnes)
    contenu = (
        f"Analyse multivariée (effectifs croisés) de {', '.join(f'`{c}`' for c in colonnes)} "
        f"dans **{nom_table}** ({len(tab)} combinaisons observées) :\n\n"
        f"{tab.head(30).to_markdown(index=False)}\n\n{dt.syntaxe_tableau_multivarie(nom_table, colonnes)}"
    )
    # dt.tableau_multivarie trie deja par "effectif" decroissant : la
    # premiere ligne EST la combinaison la plus frequente.
    ligne_max = tab.iloc[0] if len(tab) else None
    tronque = len(tab) > 30
    if ligne_max is not None:
        combinaison = ", ".join(f"`{c}`={ligne_max[c]}" for c in colonnes)
        intro = (
            f"J'ai trouvé **{len(tab)}** combinaison(s) observée(s) de {', '.join(f'`{c}`' for c in colonnes)} "
            f"dans **{nom_table}** ; la plus fréquente est {combinaison} ({int(ligne_max['effectif'])} ligne(s))."
        )
    else:
        intro = f"Voici les effectifs croisés sur les {len(colonnes)} colonnes demandées de **{nom_table}**."
    contenu = _habiller_reponse(
        contenu,
        intro=intro,
        explication=(
            "j'ai groupé les lignes de **" + nom_table + "** par combinaison de valeurs sur les "
            "colonnes demandées et compté le nombre de lignes observées pour chacune"
            + (
                f" (le tableau affiché est limité aux 30 premières combinaisons sur {len(tab)}, "
                "mais le tableau complet reste disponible via le bouton de téléchargement)"
                if tronque else ""
            ) + "."
        ),
        suggestions=[f"une matrice de corrélation si ces colonnes sont numériques"] if len(colonnes) >= 2 else None,
    )
    return {"content": contenu, "table": tab, "table_label": f"multivarie_{nom_table}"}


def reponse_agents(df, nom_table: str) -> dict:
    try:
        rapport = dt.rapport_agents(df)
    except ValueError as e:
        return {"content": f"⚠️ {e}"}
    colonne_agent = dt.detect_agent_columns(df)[0]
    contenu = (
        f"**Rapport de performance par agent enquêteur** dans **{nom_table}** "
        f"(colonne `{colonne_agent}`, {len(rapport)} agent(s)) :\n\n"
        f"{rapport.to_markdown(index=False)}\n\n"
        "_Colonnes : `n_fiches` = nombre de fiches saisies, `doublons_id` = fiches de cet agent "
        "impliquées dans un doublon d'identifiant, `dates_invraisemblables` = dates hors plage "
        "détectées, `taux_valeurs_manquantes_moyen` = proportion moyenne de cellules vides sur les "
        "fiches de cet agent._\n\n"
        "_Rappel : ceci est un signalement, pas une évaluation individuelle définitive — la "
        "validation reste réservée aux personnes habilitées._\n\n"
        f"{dt.syntaxe_rapport_agents(nom_table, colonne_agent)}"
    )
    total_doublons = int(rapport["doublons_id"].sum())
    total_dates = int(rapport["dates_invraisemblables"].sum())
    if total_doublons or total_dates:
        pire_agent = rapport.loc[(rapport["doublons_id"] + rapport["dates_invraisemblables"]).idxmax()]
        intro = (
            f"Sur les **{len(rapport)}** agent(s) de **{nom_table}**, {total_doublons} doublon(s) "
            f"d'identifiant et {total_dates} date(s) invraisemblable(s) ont été détectés au total — "
            f"l'agent `{pire_agent['agent']}` en concentre le plus (à vérifier en priorité)."
        )
    else:
        intro = (
            f"Sur les **{len(rapport)}** agent(s) de **{nom_table}**, aucun doublon d'identifiant ni "
            "date invraisemblable détecté."
        )
    contenu = _habiller_reponse(
        contenu,
        intro=intro,
        explication=(
            f"j'ai regroupé les lignes de **{nom_table}** par `{colonne_agent}` et calculé, pour "
            "chaque agent, le nombre de fiches, les doublons d'identifiant, les dates invraisemblables "
            "et le taux moyen de valeurs manquantes (détail des colonnes ci-dessus)."
        ),
        suggestions=[
            "le rapport de performance de terrain (volume d'activité par agent, distinct de ce contrôle qualité)",
        ],
    )
    return {"content": contenu, "table": rapport, "table_label": f"agents_{nom_table}"}


def extraire_objectif(q: str, defaut: int = 17000) -> int:
    """Extrait un objectif numérique (ex: "objectif 20000 ménages") de la
    question, sinon retombe sur la valeur par défaut communiquée par
    l'observatoire (17000 ménages)."""
    m = re.search(r"objectif\D{0,15}?([\d][\d\s.,]*)", q)
    if not m:
        return defaut
    chiffres = re.sub(r"[^\d]", "", m.group(1))
    return int(chiffres) if chiffres else defaut


def extraire_agents_exclus(question: str) -> list[str] | None:
    """Extrait une liste d'identifiants d'agent à exclure du rapport de
    performance (ex: "en excluant les agents 12, 45 et 67"), sinon None
    (aucune exclusion demandée dans cette question). Suppose que la liste
    d'identifiants suit le mot "agent(s)" jusqu'à la fin de la question -
    formulation la plus naturelle pour ce type de demande."""
    q = question.lower()
    if not re.search(r"exclu\w*|sans les agents?|hors agents?", q):
        return None
    m = re.search(r"agents?\s*[:\-]?\s*([\w][\w,\s]*)$", q)
    if not m:
        return None
    agents = [a.strip() for a in re.split(r"[,;]| et ", m.group(1)) if a.strip()]
    return agents or None


def extraire_identifiant_recherche(question: str) -> str | None:
    """Extrait l'identifiant à rechercher d'une question de recherche
    instantanée ("recherche l'individu 1024") : priorité à une séquence de
    chiffres, sinon le dernier mot alphanumérique de la question."""
    m = re.search(r"\b(\d+)\b", question)
    if m:
        return m.group(1)
    mots = re.findall(r"[A-Za-z0-9_\-]{3,}", question)
    return mots[-1] if mots else None


def reponse_performance_terrain(tables: dict, exclure: list[str] | None, objectif: int, q: str = "") -> dict:
    rapport = dt.rapport_performance_agents(tables, exclure=exclure)
    if rapport.empty:
        return {
            "content": (
                "Aucune colonne d'agent enquêteur détectée automatiquement dans les tables "
                "chargées : impossible de calculer un rapport de performance de terrain."
            )
        }
    rapport, nom_equipe = dt.fusion_agent_controleur(rapport, tables)
    rapport, nom_table_users = dt.fusion_identite_agent(rapport, tables)

    # Ne calcule/affiche l'avancement vers un objectif que si la question le
    # demande explicitement (objectif/avancement/projection/prevision) - une
    # question ciblee ("combien de menages collectes par agent ?") ne doit
    # pas se voir imposer un avertissement hors-sujet sur une projection
    # qu'elle n'a jamais demandee.
    demande_projection = contient_mot_cle(q, MOTS_OBJECTIF_PROJECTION)
    par_jour = dt.rapport_performance_par_jour(tables)
    prevision = dt.prevision_objectif(par_jour, objectif=objectif) if not par_jour.empty else None

    # Si la question cible une ou plusieurs categories precises (menage/UCH,
    # naissances, deces, grossesses), le TEXTE de la reponse se limite a ces
    # colonnes plutot que d'afficher systematiquement les 4 categories + le
    # total - le tableau complet reste neanmoins disponible via les boutons
    # d'export (voir la cle "table", jamais filtree).
    categories_demandees = colonnes_categories_demandees(q)
    if categories_demandees:
        colonnes_texte = ["agent"]
        if "controleur" in rapport.columns:
            colonnes_texte.append("controleur")
        if "email_agent" in rapport.columns:
            colonnes_texte.append("email_agent")
        colonnes_texte += [c for c in categories_demandees if c in rapport.columns]
        if "Ménages/UCH visités" in categories_demandees and "Ménages/UCH distincts" in rapport.columns:
            colonnes_texte.append("Ménages/UCH distincts")
        rapport_texte = rapport[colonnes_texte]
    else:
        rapport_texte = rapport

    morceaux = [
        f"**Rapport de performance de terrain** ({len(rapport)} agent(s)"
        + (f", exclusion de {len(exclure)} agent(s) non-terrain" if exclure else "")
        + ") :\n\n" + rapport_texte.to_markdown(index=False)
    ]
    if nom_equipe:
        morceaux.append(f"_Contrôleur ajouté à partir de la table équipe **{nom_equipe}**._")
    elif contient_mot_cle(q, MOTS_CONTROLEUR_EQUIPE):
        morceaux.append(
            "_Aucune table équipe (agent ↔ contrôleur) détectée parmi les tables chargées : "
            "dépose-la pour faire apparaître la colonne `controleur`._"
        )
    if nom_table_users:
        morceaux.append(
            f"_Identité ajoutée (email) à partir de la table **{nom_table_users}** — "
            "seule identité disponible pour un agent dans les données actuelles, "
            "qui ne comportent pas de nom/prénom._"
        )

    if demande_projection:
        if prevision:
            morceaux.append(
                f"**Avancement vers l'objectif ({prevision['objectif']})** : "
                f"{prevision['cumul_actuel']} réalisé(s), {prevision['reste_a_faire']} restant(s), "
                f"rythme moyen {prevision['rythme_journalier_moyen']}/jour "
                f"({prevision['n_jours_observes']} jour(s) observé(s), "
                f"{prevision['date_debut']} → {prevision['date_derniere_donnee']})."
                + (
                    f" Au rythme actuel, objectif atteint vers le **{prevision['date_fin_projetee']}** "
                    f"(≈{prevision['jours_restants_estimes']} jour(s))."
                    if prevision.get("date_fin_projetee") else ""
                )
            )
        else:
            morceaux.append(
                "_Aucune colonne de date détectée en plus de la colonne d'agent : la projection vers "
                "l'objectif n'a pas pu être calculée._"
            )

    docx_bytes = dt.generer_rapport_performance_docx(rapport, prevision, objectif=objectif)

    # Courbe de progression (cumul journalier) demandee par l'observatoire :
    # affichee comme un vrai graphique (st.line_chart, natif Streamlit, pas
    # besoin de matplotlib) en plus du texte, pas seulement decrite en mots.
    chart_data = None
    if not par_jour.empty:
        cumul = par_jour.groupby("date")["n_fiches"].sum().sort_index().cumsum()
        chart_data = cumul.rename("cumul_fiches").to_frame()

    contenu_terrain = _habiller_reponse(
        "\n\n".join(morceaux),
        intro="Voici le rapport de performance de terrain (volume d'activité par agent).",
        explication=(
            "j'ai regroupé les fiches de toutes les tables chargées par agent enquêteur et compté, "
            "pour chacun, les ménages/UCH visités, naissances, décès et grossesses enregistrés"
            + (", puis comparé ce cumul à l'objectif fixé" if demande_projection else "") + "."
        ),
        suggestions=["l'historique des actualisations de cette session", "un rapport de qualité par agent (doublons, dates invraisemblables)"],
    )
    return {
        "content": contenu_terrain, "table": rapport, "table_label": "performance_terrain",
        "docx_bytes": docx_bytes, "docx_label": "rapport_performance_terrain",
        "chart_data": chart_data,
    }


def reponse_historique_actualisations() -> dict:
    historique = st.session_state.get("historique_chargements", [])
    if not historique:
        return {"content": "Aucune table n'a encore été chargée durant cette session."}
    lignes = ["**Historique des actualisations de cette session** :"]
    for entree in reversed(historique):
        date_export = entree.get("date_export")
        suffixe_export = f", export du {date_export.strftime('%d/%m/%Y')}" if date_export else ""
        source = entree.get("source", "dépôt manuel")
        lignes.append(
            f"- {entree['horodatage'].strftime('%d/%m/%Y %H:%M:%S')} — **{entree['table']}** "
            f"({entree['n_lignes']} ligne(s)), depuis `{entree['fichier']}` ({source}{suffixe_export})"
        )
    derniere = historique[-1]
    contenu = _habiller_reponse(
        "\n".join(lignes),
        intro=(
            f"**{len(historique)}** chargement(s) enregistré(s) cette session, le plus récent étant "
            f"**{derniere['table']}** à {derniere['horodatage'].strftime('%H:%M:%S')}."
        ),
        explication="j'ai listé, du plus récent au plus ancien, chaque table chargée avec l'heure, la source et le nombre de lignes.",
        suggestions=[f"un contrôle de cohérence sur **{derniere['table']}**, la dernière table chargée"],
    )
    return {"content": contenu}


def reponse_recherche_identifiant(identifiant: str, tables: dict) -> dict:
    resultats = dt.rechercher_identifiant(identifiant, tables)
    if not resultats:
        return {"content": f"Aucune fiche trouvée pour l'identifiant **{identifiant}** dans les tables chargées."}
    morceaux = [f"**Recherche de l'identifiant `{identifiant}`** — trouvé dans {len(resultats)} table(s) :"]
    for nom, df in resultats.items():
        morceaux.append(f"\n**{nom}** ({len(df)} ligne(s)) :\n\n{df.head(10).to_markdown(index=False)}")
    noms_tables = ", ".join(f"**{n}**" for n in resultats)
    contenu = _habiller_reponse(
        "\n".join(morceaux),
        intro=f"L'identifiant `{identifiant}` apparaît dans {len(resultats)} table(s) : {noms_tables}.",
        explication=(
            f"j'ai recherché la valeur `{identifiant}` dans les colonnes d'identifiant de chaque table "
            "chargée, et affiché (jusqu'à 10 lignes par table) toutes celles où elle apparaît — utile "
            "pour reconstituer le parcours complet d'un individu ou d'un ménage à travers les fiches."
        ),
    )
    return {"content": contenu}


# Analyse bivariee (tableau croise entre deux colonnes d'UNE MEME table) -
# mots-cles volontairement distincts de MOTS_RELATION (qui sert aux relations
# ENTRE TABLES) pour ne jamais confondre les deux echelles d'analyse.
# Avec entiers=True (mot entier), chaque accord grammatical du francais
# (masculin/feminin/singulier/pluriel : croisé/croisée/croisés/croisées) doit
# etre liste explicitement - une simple sous-chaine ne suffit plus a couvrir
# toutes les terminaisons une fois qu'on exige une frontiere de mot exacte.
MOTS_BIVARIE = [
    "tableau croise", "croisement", "table croisee",
    "bivarie", "bivariee", "bivaries", "bivariees",
    "croise", "croisee", "croises", "croisees",
]

# Analyse multivariee : deux formes distinctes selon le type de variables -
# correlation pour des variables numeriques/quantitatives, groupement pour
# des variables categorielles (3 colonnes ou plus).
MOTS_CORRELATION = ["correlation", "correle", "correlee", "correles", "correlees"]
MOTS_MULTIVARIE = [
    "multivarie", "multivariee", "multivaries", "multivariees",
    "multi-varie", "multi varie",
]

# Controle qualite des agents enqueteurs/du personnel de terrain : necessite
# la combinaison d'un mot designant le personnel ET d'un mot lie a la
# performance/qualite (meme principe que est_question_meta_tables), pour
# eviter qu'une simple mention du mot "agent" ou "enqueteur" dans une
# question documentaire ne declenche a tort un calcul sur une table.
MOTS_AGENT_PERSONNE = ["agent", "agents", "enqueteur", "enqueteurs", "enqueteuse", "enqueteuses", "interviewer", "interviewers"]
MOTS_AGENT_QUALITE = [
    "performance", "erreur", "erreurs", "qualite", "anomalie", "anomalies",
    "travail de terrain", "charge de travail", "controle",
]


def est_question_agents(q: str) -> bool:
    """Detecte une question sur la performance/qualite du travail des agents
    enqueteurs (et non une simple mention documentaire du mot "agent")."""
    return contient_mot_cle(q, MOTS_AGENT_PERSONNE) and contient_mot_cle(q, MOTS_AGENT_QUALITE)


MOTS_LISTE_TABLES = [
    "combien de table", "combien de feuille", "quelles tables", "quelles sont les tables",
    "liste des tables", "tables chargees", "tables chargées", "tables disponibles",
    "nombre de tables", "nombre de feuilles", "quelles feuilles", "liste des feuilles",
]

# Audit complet (catalogue de controles specifiques a l'observatoire, au-dela
# des doublons/dates generiques) - opere sur TOUTES les tables chargees a la
# fois, donc verifie avant toute resolution a une seule table.
MOTS_COHERENCE_AVANCEE = [
    "controle avance", "controles avances", "audit complet", "audit de coherence",
    "toutes les incoherences", "catalogue de controles", "toutes les regles de coherence",
    "controle qualite complet", "verification complete", "controles de coherence avances",
]

# Question de coherence CROISEE entre deux tables ("il y a des décédés dans
# presence ?", "des individus partis qu'on retrouve encore en présence ?")
# formulee en langage naturel, SANS la phrase figee "audit complet" -
# reutilise le meme controle deja calcule par rapport_coherence_avancee
# (dt.controle_deces_present) au lieu d'exiger cette phrase precise pour y
# acceder. Necessaire car la table ciblee (ex: "presence") ne porte ELLE-MEME
# aucune colonne de statut deces/depart : un simple calcul sur ses propres
# colonnes (action REQUETE) ne peut jamais repondre a ce type de question,
# qui necessite de croiser avec une AUTRE table (ici la table deces/departs).
# Chaque entree : (mots-cles de la question, mots-cles du role de la table de
# statut a chercher - meme principe que CATEGORIES_PERFORMANCE_TERRAIN,
# libelle affiche).
STATUTS_CROISES_TABLE = [
    (["decede", "décédé", "deces", "décès", "mort"], ["death", "deces"], "décès"),
    (["depart", "parti", "partie", "migration"], ["migration_out", "migrationout", "depart"], "départ"),
]


def reponse_statut_croise_dans_table(q: str, nom_table_cible: str, tables: dict) -> dict | None:
    """Repond a une question de coherence croisee ("des décédés dans
    <table> ?") en reutilisant `dt.controle_deces_present`. Renvoie None si
    la question ne correspond a aucun statut connu, si aucune table de statut
    n'est chargee, si c'est la table de statut elle-meme qui est ciblee
    (question directe sur les décès, pas une question croisee), ou si aucune
    cle de jointure n'est detectee - pour laisser l'appelant retomber sur son
    propre repli plutot que d'echouer."""
    for mots_question, mots_table_statut, libelle in STATUTS_CROISES_TABLE:
        if not contient_mot_cle(q, mots_question):
            continue
        nom_statut = dt.trouver_table_par_role(tables, mots_table_statut)
        if nom_statut is None or nom_statut == nom_table_cible:
            continue
        resultat = dt.controle_deces_present(tables, nom_statut, nom_table_cible)
        if resultat is None:
            continue
        cle = resultat["colonnes_verifiees"][0]
        n = resultat["n_anomalies"]
        contenu = (
            f"**{n}** individu(s) apparaissent à la fois dans **{nom_statut}** ({libelle}) et "
            f"**{nom_table_cible}** (sur la clé `{cle}`)."
            + (
                " Cela peut indiquer une incohérence à vérifier (ex : décédé/parti mais toujours "
                "marqué présent)."
                if n > 0 else " Aucune incohérence détectée sur ce point."
            )
        )
        contenu = _habiller_reponse(
            contenu,
            intro=f"J'ai croisé **{nom_statut}** et **{nom_table_cible}** pour vérifier ce point.",
            explication=(
                f"j'ai joint **{nom_statut}** et **{nom_table_cible}** sur la clé `{cle}` et compté "
                f"les individus présents dans les deux à la fois."
            ),
        )
        return {"content": contenu}
    return None

# Module "Performances" : volume d'activite de terrain par agent (menages/UCH
# visites, naissances/deces/grossesses enregistres), distinct du controle
# qualite (est_question_agents/MOTS_AGENT_QUALITE, qui mesure les
# erreurs/doublons, pas le volume). Meme principe de DEUX familles de mots
# combinees que est_question_agents/est_question_meta_tables, pour ne jamais
# se confondre avec "performance des agents enquêteurs" (qualité, sans mot de
# volume terrain) - opere sur TOUTES les tables chargees a la fois, comme
# l'audit de coherence avance.
MOTS_TERRAIN_VOLUME = ["menage", "menages", "ménage", "ménages", "uch", "terrain", "collecte", "collectes"]
MOTS_TERRAIN_RAPPORT = [
    "performance", "avancement", "suivi", "bilan", "tableau de bord",
    "rapport", "objectif", "projection", "prevision", "prévision",
]

# Sous-familles de mots-cles pour NE PAS noyer une question ciblee ("combien
# de menages collectes par agent ?") dans le rapport complet à 4+ colonnes
# ni dans des avertissements hors-sujet ("aucune table equipe...", "aucune
# colonne de date...") qui ne concernent que l'avancement vers un objectif
# ou le controleur - jamais demandes dans ce type de question precise.
MOTS_OBJECTIF_PROJECTION = ["objectif", "avancement", "projection", "prevision", "prévision"]
MOTS_CONTROLEUR_EQUIPE = ["controleur", "contrôleur", "superviseur", "encadrement", "equipe", "équipe"]

# Chaque entree associe les mots-cles qu'une question peut employer au
# libelle EXACT de colonne produit par `dt.rapport_performance_agents`
# (voir `dt.CATEGORIES_PERFORMANCE_TERRAIN`) - permet de filtrer le rapport
# a la seule categorie demandee plutot que d'afficher systematiquement les
# 4 categories + le total, meme quand une seule est demandee.
CATEGORIES_MOTS_CLES_QUESTION = [
    ("Ménages/UCH visités", ["menage", "menages", "ménage", "ménages", "uch"]),
    ("Naissances enregistrées", ["naissance", "naissances"]),
    ("Décès enregistrés", ["deces", "décès", "dèces"]),
    ("Grossesses enregistrées", ["grossesse", "grossesses"]),
]


def colonnes_categories_demandees(q: str) -> list[str]:
    """Renvoie les libelles de colonnes du rapport de performance de terrain
    explicitement demandes dans la question (ex: "menage" -> "Ménages/UCH
    visités"), dans l'ordre de `CATEGORIES_MOTS_CLES_QUESTION` - liste vide
    si la question ne cible aucune categorie precise (rapport complet)."""
    return [libelle for libelle, mots in CATEGORIES_MOTS_CLES_QUESTION if contient_mot_cle(q, mots)]


def est_question_performance_terrain(q: str) -> bool:
    """Detecte une question sur le volume d'activite de terrain (menages/UCH
    visites, avancement vers un objectif) - distincte du controle qualite des
    agents (est_question_agents)."""
    return contient_mot_cle(q, MOTS_TERRAIN_VOLUME) and contient_mot_cle(q, MOTS_TERRAIN_RAPPORT)

MOTS_HISTORIQUE_ACTUALISATION = [
    "historique des actualisations", "historique des mises a jour", "historique des mises à jour",
    "quand les tables ont ete mises a jour", "quand les tables ont été mises à jour",
    "derniere actualisation", "dernière actualisation", "historique des chargements",
    "historique des imports",
]

MOTS_RECHERCHE_ID = [
    "recherche l'identifiant", "recherche l'individu", "recherche identifiant",
    "fiche complete de", "fiche complète de", "dossier complet de",
    "historique complet de l'individu", "toutes les fiches de l'individu", "toutes les fiches de",
    "recherche instantanee", "recherche instantanée",
]

# Complement de MOTS_LISTE_TABLES : plutot que d'enumerer indefiniment de
# nouvelles formulations exactes (approche fragile - chaque nouvelle facon de
# demander "combien de tables sont chargees" en langage libre passait a cote,
# ex: "je parle de table au niveau de l'importation des tables"), on detecte
# une question "meta" sur les tables elles-memes via DEUX familles de mots
# combinees : un mot generique designant les tables ET un verbe/mot lie a
# l'import/au chargement/au denombrement. Beaucoup plus robuste, et evite de
# dependre uniquement du classifieur LLM (qui peut se tromper sur une
# formulation ambigue - voir classifier_intention/LISTE_TABLES pour le filet
# de securite quand meme cette detection ne suffit pas).
MOTS_TABLE_GENERIQUE = ["table", "tables", "feuille", "feuilles", "classeur", "classeurs"]
# "combien" et "liste" sont volontairement EXCLUS d'ici : ce sont des mots bien
# trop courants dans une vraie question de contenu ("combien d'individus dans
# la table X ?", "liste des colonnes de la table X") pour servir de signal
# fiable une fois combines au simple mot "table" - ils restent geres seulement
# via les formulations exactes et sans ambiguite de MOTS_LISTE_TABLES
# ("combien de table(s)", "liste des tables/feuilles").
MOTS_ACTION_META = [
    "dispon", "importation", "importer", "importe", "importée", "importées", "importé", "importés",
    "envoy", "reçu", "recu", "reçois", "recois", "reçoit", "recoit", "fourni",
    "depos", "déposé", "deposee", "déposée", "deposees", "déposées",
]


def est_question_meta_tables(q: str) -> bool:
    """Detecte une question portant sur les tables actuellement chargees
    elles-memes (nombre, noms, confirmation qu'elles ont bien ete recues) -
    par opposition a une question sur le contenu d'une table precise."""
    return contient_mot_cle(q, MOTS_TABLE_GENERIQUE) and contient_mot_cle(q, MOTS_ACTION_META)


def route_question(question: str) -> dict:
    """Determine si la question porte sur une table deposee (indicateur,
    echantillon, coherence), sur une relation/fusion entre plusieurs tables
    chargees, sur la liste des tables/feuilles elles-memes, ou sur le
    dictionnaire (RAG)."""
    q = sans_accents(question.lower())
    tables = st.session_state.get("tables", {})

    # Question "meta" sur la session en cours (combien de tables/feuilles
    # sont chargees, lesquelles) : ne concerne pas le contenu d'une table ni
    # le dictionnaire, donc a verifier en tout premier, avant toute
    # resolution de table.
    if contient_mot_cle(q, MOTS_LISTE_TABLES) or est_question_meta_tables(q):
        return {"content": dt.resume_tables_chargees(tables)}

    # Questions de relation ou de fusion entre plusieurs tables chargees
    # (fichiers separes ou feuilles d'un meme classeur Excel : traitees de
    # facon identique une fois chargees) - a verifier AVANT la resolution
    # a une seule table, puisque ces questions portent par nature sur au
    # moins deux tables a la fois.
    tables_mentionnees = dt.detecter_tables_mentionnees(question, tables)
    if len(tables_mentionnees) < 2:
        # Complete avec les tables mentionnees dans les echanges precedents
        # (memoire conversationnelle), sans dupliquer celles deja trouvees
        # dans la question en cours.
        for nom in dt.tables_mentionnees_dans_historique(historique_recent(), tables):
            if nom not in tables_mentionnees:
                tables_mentionnees.append(nom)
            if len(tables_mentionnees) >= 2:
                break

    # Audit complet (catalogue de controles specifiques a l'observatoire) -
    # opere sur TOUTES les tables chargees a la fois (y compris les controles
    # croises entre tables), donc verifie avant toute resolution a une seule
    # table. Une table precise peut etre ciblee en la nommant dans la
    # question (ex: "audit complet de FNewPresences").
    if contient_mot_cle(q, MOTS_COHERENCE_AVANCEE):
        nom_cible = tables_mentionnees[0] if tables_mentionnees else None
        rapport_avance = dt.rapport_coherence_avancee(tables, nom_table=nom_cible)
        contenu = _habiller_reponse(
            formater_rapport_coherence_avancee(rapport_avance),
            intro="J'ai passé les données au catalogue de contrôles de cohérence spécifiques à l'observatoire.",
            explication=(
                "pour chaque table, j'ai appliqué les contrôles internes reconnus (colonnes détectées "
                "automatiquement), puis les contrôles croisés entre tables sur les clés de jointure "
                "documentées par le dictionnaire de données (ex : individid, socialgpid, peventid) — "
                "jamais sur une colonne au nom seulement évocateur."
            ),
        )
        return {"content": contenu}

    # Question de coherence croisee formulee naturellement ("il y a des
    # décédés dans presence ?"), sans la phrase figee "audit complet" -
    # verifie tot pour la meme raison que l'audit complet (la table ciblee ne
    # porte elle-meme aucune colonne de statut deces/depart, un simple calcul
    # sur ses propres colonnes ne peut jamais y repondre). Necessite qu'une
    # table soit identifiable (nommee, ou resolue via nom/colonne/historique).
    if contient_mot_cle(q, [m for mots, _, _ in STATUTS_CROISES_TABLE for m in mots]):
        nom_cible_croise = tables_mentionnees[0] if tables_mentionnees else dt.resoudre_table_ciblee(
            question, tables, historique=historique_recent()
        )[0]
        if nom_cible_croise:
            reponse_croisee = reponse_statut_croise_dans_table(q, nom_cible_croise, tables)
            if reponse_croisee is not None:
                return reponse_croisee

    # Historique des actualisations de cette session : ne depend d'aucune
    # table en particulier, verifie tot pour ne pas etre masque par la
    # resolution a une seule table.
    if contient_mot_cle(q, MOTS_HISTORIQUE_ACTUALISATION):
        return reponse_historique_actualisations()

    # Recherche instantanee d'un identifiant a travers TOUTES les tables
    # chargees ("recherche l'individu 1024") - meme principe que l'audit
    # complet : opere sur l'ensemble des tables, pas sur une seule resolue.
    if contient_mot_cle(q, MOTS_RECHERCHE_ID):
        identifiant = extraire_identifiant_recherche(question)
        if identifiant is None:
            return {"content": "Précise l'identifiant à rechercher (ex : « recherche l'individu 1024 »)."}
        return reponse_recherche_identifiant(identifiant, tables)

    # Performance de terrain par agent (menages/UCH visites, naissances/
    # deces/grossesses enregistres) - agrege sur TOUTES les tables chargees,
    # avec objectif et exclusion d'agents extraits directement de la question
    # si presents (sinon objectif par defaut 17000 menages, aucune exclusion).
    if est_question_performance_terrain(q):
        objectif = extraire_objectif(q)
        exclure = extraire_agents_exclus(q)
        return reponse_performance_terrain(tables, exclure, objectif, q)

    # Une relance courte ("il faut analyser directement") qui ne repete pas le
    # mot-cle d'action initial : on regarde si le tour precedent de l'equipe
    # demandait deja une difference/fusion/relation, pour redeclencher le bon
    # calcul plutot que de laisser la question filer vers le LLM generique.
    relance_calcul = contient_mot_cle(q, MOTS_RELANCE_CALCUL)

    # Difference d'ensembles ("qui est dans X mais pas dans Y", et
    # eventuellement "vice versa" pour les deux sens a la fois) - a verifier
    # AVANT la fusion generale, puisque ce sont deux operations distinctes.
    difference_demandee = contient_mot_cle(q, MOTS_DIFFERENCE) or (
        relance_calcul and action_deja_demandee_dans_historique(historique_recent(), MOTS_DIFFERENCE)
    )
    if difference_demandee:
        paire = resoudre_paire_tables(tables_mentionnees, tables)
        if paire is None:
            return {"content": message_precision_tables("comparer", tables, tables_mentionnees)}
        a, b = paire
        try:
            diff_ab = dt.difference_tables(a, b, tables)
        except ValueError as e:
            return {"content": f"⚠️ {e}"}

        cle = dt.detecter_cle_jointure(a, b, tables)
        nom_resultat_ab = f"difference_{a}_sans_{b}"
        st.session_state["tables"][nom_resultat_ab] = diff_ab

        morceaux = [
            f"**{len(diff_ab)}** ligne(s) de **{a}** n'ont pas de correspondance dans **{b}** "
            f"(sur la clé `{cle}`). Résultat enregistré sous **{nom_resultat_ab}** — tu peux directement "
            f"demander un indicateur ou un échantillon dessus ensuite.\n\n{diff_ab.head(20).to_markdown(index=False)}"
        ]
        table_resultat, label_resultat = diff_ab, nom_resultat_ab

        vice_versa_demande = contient_mot_cle(q, MOTS_VICE_VERSA) or (
            relance_calcul and action_deja_demandee_dans_historique(historique_recent(), MOTS_VICE_VERSA)
        )
        if vice_versa_demande:
            diff_ba = dt.difference_tables(b, a, tables)
            nom_resultat_ba = f"difference_{b}_sans_{a}"
            st.session_state["tables"][nom_resultat_ba] = diff_ba
            morceaux.append(
                f"Et inversement : **{len(diff_ba)}** ligne(s) de **{b}** n'ont pas de correspondance dans "
                f"**{a}**. Résultat enregistré sous **{nom_resultat_ba}**.\n\n{diff_ba.head(20).to_markdown(index=False)}"
            )
            # En cas de "vice versa", le deuxieme resultat (b sans a) est celui
            # propose en telechargement immediat (le premier reste consultable
            # via son propre nom pour une question de suivi).
            table_resultat, label_resultat = diff_ba, nom_resultat_ba

        if cle:
            morceaux.append(dt.syntaxe_difference(a, b, cle))

        return {"content": "\n\n".join(morceaux), "table": table_resultat, "table_label": label_resultat}

    fusion_demandee = contient_mot_cle(q, MOTS_FUSION) or (
        relance_calcul and action_deja_demandee_dans_historique(historique_recent(), MOTS_FUSION)
    )
    if fusion_demandee:
        paire = resoudre_paire_tables(tables_mentionnees, tables)
        if paire is None:
            return {"content": message_precision_tables("fusionner", tables, tables_mentionnees)}
        a, b = paire
        try:
            fusion = dt.fusionner_tables(a, b, tables)
        except ValueError as e:
            return {"content": f"⚠️ {e}"}

        cle = dt.detecter_cle_jointure(a, b, tables)
        nom_resultat = f"fusion_{a}_{b}"
        st.session_state["tables"][nom_resultat] = fusion

        morceaux = [
            f"Fusion de **{a}** et **{b}** ({len(fusion)} lignes obtenues, sur la clé `{cle}`). "
            f"Résultat enregistré sous **{nom_resultat}** — interrogeable directement ensuite "
            f"(indicateur, échantillon...).\n\n{fusion.head(20).to_markdown(index=False)}"
        ]
        if cle:
            morceaux.append(dt.syntaxe_fusion(a, b, cle))

        return {"content": "\n\n".join(morceaux), "table": fusion, "table_label": nom_resultat}

    relation_demandee = contient_mot_cle(q, MOTS_RELATION, entiers=True) or (
        relance_calcul and action_deja_demandee_dans_historique(historique_recent(), MOTS_RELATION, entiers=True)
    )
    if relation_demandee:
        if len(tables_mentionnees) >= 2:
            reponse_relation = dt.relation_entre_tables(tables_mentionnees[0], tables_mentionnees[1], tables)
        else:
            reponse_relation = dt.rapport_relations(tables)
        # Le controle structurel ci-dessus ne detecte qu'un lien "de meme nom
        # de colonne" entre les tables. Beaucoup de vrais liens du schema OPO
        # passent par des colonnes de noms differents (ex: headid -> individid,
        # documentes dans le dictionnaire, voir 00_schema_relations) : avant
        # d'abandonner sur ce controle structurel, on tente le repli SQL
        # general (qui, lui, s'appuie aussi sur le dictionnaire indexe) -
        # jamais de reponse "aucun lien trouve" alors qu'une reponse precise
        # est possible avec les vraies donnees chargees.
        if "Aucune colonne commune détectée" in reponse_relation:
            reponse_sql = tenter_requete_sql(
                question, tables, groq_key_input, anthropic_key_input, historique=historique_recent(),
            )
            if reponse_sql:
                return reponse_sql
        return {"content": reponse_relation}

    # Si aucune table n'est nommee explicitement et qu'une colonne mentionnee
    # existe dans PLUSIEURS tables a la fois (ex: `individid` present dans 20
    # tables), on lit TOUTES les tables concernees plutot que d'en ignorer
    # silencieusement certaines et de retomber sur un choix par defaut - "il
    # faut tout lire, pas une seule base par defaut".
    if not tables_mentionnees:
        candidats_colonne = dt.tables_avec_colonne(question, tables)
        if len(candidats_colonne) >= 2:
            if contient_mot_cle(q, ["doublon"]):
                morceaux = [reponse_doublons(tables[nom], nom)["content"] for nom in candidats_colonne]
                return {"content": "\n\n---\n\n".join(morceaux)}
            if contient_mot_cle(q, ["incoherence", "coherence", "anomalie"]):
                morceaux = [reponse_coherence(tables[nom], nom)["content"] for nom in candidats_colonne]
                return {"content": "\n\n---\n\n".join(morceaux)}
            if contient_mot_cle(q, ["echantillon"]):
                m_n = re.search(r"\d+", q)
                n = int(m_n.group()) if m_n else 100
                morceaux = [reponse_echantillon(tables[nom], nom, n)["content"] for nom in candidats_colonne]
                return {"content": "\n\n---\n\n".join(morceaux)}
            if contient_mot_cle(q, ["repartition", "indicateur"]):
                col_trouvee = next(
                    (c for nom in candidats_colonne for c in tables[nom].columns if sans_accents(str(c).lower()) in q),
                    None,
                )
                if col_trouvee:
                    morceaux = [
                        reponse_repartition(tables[nom], nom, col_trouvee)["content"]
                        for nom in candidats_colonne if col_trouvee in tables[nom].columns
                    ]
                    if morceaux:
                        return {"content": "\n\n---\n\n".join(morceaux)}

        # Meme principe pour les analyses a plusieurs colonnes (bivariee,
        # correlation, multivariee) : sans nom de table, on cherche TOUTES
        # les colonnes mentionnees (union de toutes les tables chargees) et
        # on calcule sur CHAQUE table qui les contient toutes ensemble,
        # plutot que de retomber sur la table par defaut de la barre
        # laterale des qu'aucune table n'est explicitement nommee.
        colonnes_visees_globales = dt.colonnes_mentionnees(question, tables)
        if len(colonnes_visees_globales) >= 2:
            candidats_multi = dt.tables_avec_toutes_colonnes(colonnes_visees_globales, tables)
            if candidats_multi:
                if contient_mot_cle(q, MOTS_BIVARIE, entiers=True):
                    morceaux = []
                    for nom in candidats_multi:
                        try:
                            morceaux.append(reponse_tableau_croise(
                                tables[nom], nom, colonnes_visees_globales[0], colonnes_visees_globales[1]
                            )["content"])
                        except ValueError:
                            continue
                    if morceaux:
                        return {"content": "\n\n---\n\n".join(morceaux)}
                if contient_mot_cle(q, MOTS_CORRELATION, entiers=True):
                    morceaux = []
                    for nom in candidats_multi:
                        try:
                            morceaux.append(reponse_correlation(tables[nom], nom, colonnes_visees_globales)["content"])
                        except ValueError:
                            continue
                    if morceaux:
                        return {"content": "\n\n---\n\n".join(morceaux)}
                if len(colonnes_visees_globales) >= 3 and contient_mot_cle(q, MOTS_MULTIVARIE, entiers=True):
                    morceaux = []
                    for nom in candidats_multi:
                        try:
                            morceaux.append(reponse_tableau_multivarie(tables[nom], nom, colonnes_visees_globales)["content"])
                        except ValueError:
                            continue
                    if morceaux:
                        return {"content": "\n\n---\n\n".join(morceaux)}

    # Plus de "table par defaut" choisie a l'avance dans la barre laterale :
    # sans nom de table, sans colonne reconnaissable qui n'appartienne qu'a
    # une seule table, et sans historique exploitable, `resoudre_table_ciblee`
    # renvoie maintenant (None, None) plutot que de deviner une table -
    # "toutes les tables travaillent au debut, on selectionne seulement si on
    # veut UNE table precise". Voir le bloc juste en dessous pour ce qui se
    # passe alors : l'operation demandee est appliquee a TOUTES les tables
    # chargees plutot que d'etre perdue ou de retomber sur le dictionnaire.
    nom_table, df = dt.resoudre_table_ciblee(question, tables, historique=historique_recent())

    if df is None and tables:
        if est_question_agents(q):
            candidats = [n for n, d in tables.items() if dt.detect_agent_columns(d)]
            if candidats:
                morceaux = [reponse_agents(tables[n], n)["content"] for n in candidats]
                return {"content": "\n\n---\n\n".join(morceaux)}
        if contient_mot_cle(q, ["doublon"]):
            morceaux = [reponse_doublons(tables[n], n)["content"] for n in tables]
            return {"content": "\n\n---\n\n".join(morceaux)}
        if contient_mot_cle(q, ["incoherence", "coherence", "anomalie"]):
            morceaux = [reponse_coherence(tables[n], n)["content"] for n in tables]
            return {"content": "\n\n---\n\n".join(morceaux)}
        if contient_mot_cle(q, ["echantillon"]):
            m_ech = re.search(r"\d+", q)
            n_ech = int(m_ech.group()) if m_ech else 100
            morceaux = [reponse_echantillon(tables[n], n, n_ech)["content"] for n in tables]
            return {"content": "\n\n---\n\n".join(morceaux)}

        # Aucun mot-cle simple n'a matche : avant de demander de preciser ou
        # de retomber sur le dictionnaire, tente une requete precise
        # (compter/lister/moyenne/somme/min/max, avec filtres) via le
        # classifieur LLM sur TOUTES les tables chargees - c'est ce qui
        # permet de repondre a une question sur les donnees reelles ("combien
        # de X ont Y ?") meme quand aucune table n'a pu etre resolue
        # explicitement (nom non cite, colonne ambigue entre plusieurs
        # tables...), sans jamais se limiter a une seule table par defaut.
        reponse_requete = tenter_requete_donnees_multi_table(question, tables, groq_key_input, anthropic_key_input)
        if reponse_requete is not None:
            return reponse_requete

        # Toujours rien : tente le repli GENERAL (requete SQL sur le schema
        # complet, capable de croiser 2, 3, 4 tables ou plus) avant de
        # demander de preciser ou de retomber sur le dictionnaire.
        reponse_sql = tenter_requete_sql(
                question, tables, groq_key_input, anthropic_key_input, historique=historique_recent(),
            )
        if reponse_sql is not None:
            return reponse_sql

        # Repartition/bivarie/correlation/multivarie ont besoin d'au moins
        # une colonne reconnue quelque part pour ne rien calculer au hasard :
        # si la question semble viser une de ces analyses mais qu'aucune
        # colonne ni table n'a pu etre identifiee, on demande de preciser
        # avec les vraies tables/colonnes chargees plutot que de deviner.
        if contient_mot_cle(q, ["repartition", "indicateur"]) or contient_mot_cle(
            q, MOTS_BIVARIE + MOTS_CORRELATION + MOTS_MULTIVARIE, entiers=True
        ):
            return {"content": message_precision_colonne(tables)}

    if df is not None and est_question_agents(q):
        return reponse_agents(df, nom_table)

    if df is not None and contient_mot_cle(q, MOTS_BIVARIE, entiers=True):
        colonnes_visees = [c for c in df.columns if sans_accents(str(c).lower()) in q]
        if len(colonnes_visees) >= 2:
            try:
                return reponse_tableau_croise(df, nom_table, colonnes_visees[0], colonnes_visees[1])
            except ValueError as e:
                return {"content": f"⚠️ {e}"}
        return {
            "content": (
                f"Précise les deux colonnes à croiser (table **{nom_table}**). "
                f"Colonnes disponibles : {', '.join(df.columns)}"
            )
        }

    if df is not None and contient_mot_cle(q, MOTS_CORRELATION, entiers=True):
        colonnes_visees = [c for c in df.columns if sans_accents(str(c).lower()) in q]
        try:
            return reponse_correlation(df, nom_table, colonnes_visees if len(colonnes_visees) >= 2 else None)
        except ValueError as e:
            return {"content": f"⚠️ {e}"}

    if df is not None and contient_mot_cle(q, MOTS_MULTIVARIE, entiers=True):
        colonnes_visees = [c for c in df.columns if sans_accents(str(c).lower()) in q]
        if len(colonnes_visees) >= 3:
            try:
                return reponse_tableau_multivarie(df, nom_table, colonnes_visees)
            except ValueError as e:
                return {"content": f"⚠️ {e}"}
        return {
            "content": (
                f"Précise au moins trois colonnes à croiser pour une analyse multivariée "
                f"(table **{nom_table}**). Colonnes disponibles : {', '.join(df.columns)}"
            )
        }

    if df is not None and contient_mot_cle(q, ["doublon"]):
        return reponse_doublons(df, nom_table)

    if df is not None and contient_mot_cle(q, ["incoherence", "coherence", "anomalie"]):
        return reponse_coherence(df, nom_table)

    if df is not None and contient_mot_cle(q, ["echantillon"]):
        m = re.search(r"\d+", q)
        n = int(m.group()) if m else 100
        return reponse_echantillon(df, nom_table, n)

    if df is not None and contient_mot_cle(q, ["repartition", "indicateur"]):
        col_trouvee = next((c for c in df.columns if sans_accents(str(c).lower()) in q), None)
        if col_trouvee:
            return reponse_repartition(df, nom_table, col_trouvee)
        return {
            "content": (
                f"Précise sur quelle colonne calculer la répartition (table **{nom_table}**). "
                f"Colonnes disponibles : {', '.join(df.columns)}"
            )
        }

    # Aucun mot-clé simple n'a matché. Si une table est chargée, on essaie une
    # classification plus souple via le LLM avant d'abandonner sur le dictionnaire
    # (sinon l'agent semble "ignorer" les données déposées).
    if df is not None:
        action, parametre = rag.classifier_intention(
            question, list(df.columns), groq_key=groq_key_input, anthropic_key=anthropic_key_input
        )

        if action == "REPARTITION" and parametre:
            return reponse_repartition(df, nom_table, parametre)
        if action == "ECHANTILLON":
            return reponse_echantillon(df, nom_table, parametre or 100)
        if action == "DOUBLONS":
            return reponse_doublons(df, nom_table)
        if action == "COHERENCE":
            return reponse_coherence(df, nom_table)
        if action == "LISTE_TABLES":
            # Question formulee de facon trop variee pour MOTS_LISTE_TABLES
            # (ex: "je parle des tables que je viens de vous envoyer") mais
            # qui porte bien sur l'ensemble des tables chargees, pas sur la
            # seule table resolue par defaut : on repond avec la liste reelle
            # et complete plutot que de laisser le LLM deviner a partir d'un
            # historique qui ne mentionne jamais qu'une seule table a la fois.
            return {"content": dt.resume_tables_chargees(tables)}
        if action == "REQUETE" and parametre:
            # Calcul precis (compter/lister/moyenne/somme/min/max, avec
            # filtres) directement sur la table resolue - repond a une
            # question sur les donnees reelles ("combien de X ont Y ?") sans
            # se limiter aux 4 analyses fixes ci-dessus.
            try:
                resultat = dt.executer_requete_donnees(
                    df, parametre.get("operation"), parametre.get("colonne_cible"), parametre.get("filtres")
                )
                return formater_reponse_requete(resultat, nom_table)
            except dt.RequeteInvalide:
                pass  # colonne cible invalide malgre la validation du classifieur : repli documentaire ci-dessous

        # La table resolue seule ne suffit pas (ex: question qui necessite de
        # croiser avec une AUTRE table non consideree ci-dessus) : tente le
        # repli GENERAL (requete SQL sur le schema de TOUTES les tables
        # chargees, capable de croiser 2, 3, 4 tables ou plus) avant
        # d'abandonner sur le dictionnaire documentaire.
        reponse_sql = tenter_requete_sql(
                question, tables, groq_key_input, anthropic_key_input, historique=historique_recent(),
            )
        if reponse_sql is not None:
            return reponse_sql

        # Le classifieur a conclu que ce n'est pas une action sur la table (ou
        # aucune clé LLM n'est configurée pour trancher) : on tente la piste
        # documentaire, sans jamais laisser la table chargée passer inaperçue.
        result = rag.answer(
            question, groq_key=groq_key_input, anthropic_key=anthropic_key_input,
            historique=historique_recent(),
        )
        astuce = (
            f"\n\n---\n_Table **{nom_table}** chargée (colonnes : {', '.join(df.columns)}). "
            "Pour l'analyser directement, essaie par exemple : « répartition de <colonne> », "
            "« échantillon de 100 », « doublons », « cohérence »._"
        )
        return {"content": result["answer"] + astuce, "sources": result["sources"]}

    # Aucune table chargée : question documentaire -> RAG sur le dictionnaire
    result = rag.answer(
        question, groq_key=groq_key_input, anthropic_key=anthropic_key_input,
        historique=historique_recent(),
    )
    return {"content": result["answer"], "sources": result["sources"]}


question = st.chat_input("Pose ta question...")
if question:
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        exports = None
        try:
            with st.spinner("Recherche en cours..."):
                reponse = route_question(question)
            st.markdown(reponse["content"])
            if reponse.get("sources"):
                with st.expander("Sources utilisées"):
                    for s in reponse["sources"]:
                        st.markdown(f"- ({s['score']:.2f}) {s['text']}")
            if reponse.get("table") is not None:
                # Calcule les exports une seule fois maintenant, pour ne pas
                # les refaire a chaque rerun futur (voir calculer_exports).
                exports = calculer_exports(reponse["table"])
                afficher_boutons_export(
                    exports, reponse.get("table_label", "export"),
                    cle=f"new_{len(st.session_state['messages'])}",
                )
            if reponse.get("docx_bytes") is not None:
                afficher_bouton_docx(
                    reponse["docx_bytes"], reponse.get("docx_label", "rapport"),
                    cle=f"new_{len(st.session_state['messages'])}",
                )
            if reponse.get("chart_data") is not None:
                st.caption("Courbe de progression (cumul de fiches par jour)")
                st.line_chart(reponse["chart_data"])
        except rag.IndexNotBuiltError:
            # Ne devrait normalement pas arriver (l'index est prepare au
            # demarrage), mais on tente une reconstruction automatique une
            # fois avant d'abandonner, plutot que d'exposer un bouton
            # technique a l'equipe.
            try:
                with st.spinner("Préparation de l'assistant..."):
                    ingest.build_index()
                    rag.reset_cache()
                reponse = route_question(question)
                st.markdown(reponse["content"])
            except Exception:
                reponse = {"content": "⚠️ L'assistant n'a pas pu se préparer. Réessaie dans un instant."}
                st.warning(reponse["content"])
        except Exception:
            reponse = {"content": "❌ Une erreur inattendue s'est produite. Réessaie, ou reformule ta question."}
            st.error(reponse["content"])
            with st.expander("Détail technique (pour diagnostic)"):
                st.code(traceback.format_exc())

    st.session_state["messages"].append(
        {
            "role": "assistant",
            "content": reponse["content"],
            "sources": reponse.get("sources"),
            "exports": exports,
            "table_label": reponse.get("table_label"),
            "docx_bytes": reponse.get("docx_bytes"),
            "docx_label": reponse.get("docx_label"),
            "chart_data": reponse.get("chart_data"),
        }
    )

st.markdown(
    '<div class="opo-footer">Assistant interne — Observatoire de Population de Ouagadougou (OPO). '
    "Ne remplace pas la validation humaine des corrections de données.</div>",
    unsafe_allow_html=True,
)
