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
import ingest
import rag


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
# L'index de recherche doit exister avant de pouvoir répondre a une question
# documentaire. Plutot que d'exposer un bouton technique "construire l'index"
# a l'equipe, on le construit automatiquement au premier demarrage si besoin
# (utile notamment sur un hebergement neuf, ex: Streamlit Community Cloud).
if not rag.index_exists():
    with st.spinner("Préparation de l'assistant (première initialisation)..."):
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
        --opo-teal: #13795B;
        --opo-sand: #F4EDE4;
    }
    .opo-header {
        background: linear-gradient(135deg, var(--opo-navy) 0%, var(--opo-teal) 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.4rem;
        box-shadow: 0 4px 18px rgba(11, 37, 69, 0.25);
    }
    .opo-header h1 {
        color: #FFFFFF;
        font-size: 1.7rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 0.2px;
    }
    .opo-header p {
        color: var(--opo-sand);
        margin: 0.35rem 0 0 0;
        font-size: 0.95rem;
    }
    .opo-badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        color: #FFFFFF;
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
        "Fichiers CSV, Excel (.xlsx, .xls) ou Stata (.dta) — un fichier = une table analysable "
        "(indicateurs, échantillons, doublons...). Un classeur Excel avec plusieurs feuilles "
        "est reconnu automatiquement comme plusieurs tables, une par feuille."
    )

    if "tables" not in st.session_state:
        st.session_state["tables"] = {}
    if "fichiers_traites" not in st.session_state:
        st.session_state["fichiers_traites"] = set()  # (nom, taille) déjà chargés
    if "historique_chargements" not in st.session_state:
        st.session_state["historique_chargements"] = []  # trace chaque (re)chargement de table

    fichiers = st.file_uploader(
        "Déposer une ou plusieurs tables (CSV, Excel ou Stata)",
        type=["csv", "xlsx", "xls", "dta"],
        accept_multiple_files=True,
    )

    # Streamlit relance tout le script a chaque interaction (chaque question
    # posee dans le chat, par exemple) : sans ce controle, chaque fichier
    # deja depose serait relu et re-parse a chaque fois, ce qui rend l'appli
    # tres lente des que les tables sont un peu grosses ou nombreuses.
    if fichiers:
        for fichier in fichiers:
            signature = (fichier.name, fichier.size)
            if signature in st.session_state["fichiers_traites"]:
                continue
            try:
                nom_bas = fichier.name.lower()
                if nom_bas.endswith((".xlsx", ".xls")):
                    suffix = ".xlsx"
                elif nom_bas.endswith(".dta"):
                    suffix = ".dta"
                else:
                    suffix = ".csv"
                # tempfile.gettempdir() fonctionne sur Windows, macOS et Linux
                # (un chemin code en dur type "/tmp/..." plante sous Windows).
                # delete=False est necessaire pour pouvoir rouvrir le fichier
                # par son chemin (pd.read_excel/read_stata...) une fois
                # ferme ; le nettoyage est fait explicitement dans le bloc
                # `finally` juste en dessous, pour ne jamais laisser une
                # donnee deposee trainer sur le disque du serveur au-dela du
                # temps de son chargement (securite des donnees importees).
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
                try:
                    with open(tmp_path, "wb") as f:
                        f.write(fichier.getbuffer())

                    horodatage = datetime.now()
                    if dt.est_classeur_excel(fichier.name):
                        # Un classeur Excel peut contenir plusieurs feuilles, chacune
                        # une table distincte (ex: une feuille par table de
                        # l'observatoire) : on les reconnait toutes, pas seulement
                        # la premiere.
                        feuilles = dt.charger_classeur(tmp_path)
                        for nom_feuille, df_feuille in feuilles.items():
                            nom_table = nom_feuille if nom_feuille not in st.session_state["tables"] else (
                                f"{Path(fichier.name).stem}_{nom_feuille}"
                            )
                            st.session_state["tables"][nom_table] = df_feuille
                            st.session_state["historique_chargements"].append({
                                "horodatage": horodatage, "fichier": fichier.name,
                                "table": nom_table, "n_lignes": len(df_feuille),
                            })
                    else:
                        nom_table = re.sub(r"\.(csv|xlsx|xls|dta)$", "", fichier.name, flags=re.IGNORECASE)
                        st.session_state["tables"][nom_table] = dt.load_table(tmp_path)
                        st.session_state["historique_chargements"].append({
                            "horodatage": horodatage, "fichier": fichier.name,
                            "table": nom_table, "n_lignes": len(st.session_state["tables"][nom_table]),
                        })

                    st.session_state["fichiers_traites"].add(signature)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
            except Exception as e:
                st.error(f"Impossible de lire {fichier.name} : {e}")

    tables = st.session_state["tables"]
    if tables:
        st.success(f"{len(tables)} table(s) chargée(s) : {', '.join(tables.keys())}")
        table_active_nom = st.selectbox("Table par défaut (si la question est ambiguë)", list(tables.keys()))
        st.write("Colonnes :", list(tables[table_active_nom].columns))
        st.caption(
            "Les colonnes de type nom/prénom sont automatiquement retirées. "
            "Toutes les tables chargées sont interrogeables directement : mentionne une colonne "
            "(ex. « répartition de sex ») ou le nom d'une table dans ta question, l'assistant devine "
            "automatiquement laquelle cibler. La table ci-dessus ne sert que de repli si la question "
            "est vraiment ambiguë (aucun nom de table ni colonne reconnaissable)."
        )
    else:
        table_active_nom = None
        st.info("Aucune table déposée pour l'instant — le chat répond depuis le dictionnaire.")

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
        st.rerun()

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
    "n'apparaît pas dans",
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


# Les quatre fonctions ci-dessous centralisent le calcul ET la syntaxe R/Stata
# equivalente pour chaque operation sur une table (repartition, echantillon,
# doublons, coherence), pour que les DEUX chemins qui y menent (mots-cles
# directs, et repli via classifier_intention) donnent exactement la meme
# reponse complete - plutot que de dupliquer la logique a deux endroits et
# risquer qu'elle diverge (l'un avec la syntaxe, l'autre sans).

def reponse_repartition(df, nom_table: str, colonne: str) -> dict:
    rep = dt.repartition(df, colonne)
    contenu = (
        f"Répartition de `{colonne}` dans **{nom_table}** :\n\n{rep.to_markdown()}"
        f"\n\n{dt.syntaxe_repartition(nom_table, colonne)}"
    )
    return {"content": contenu, "table": rep.reset_index(), "table_label": f"repartition_{colonne}_{nom_table}"}


def reponse_echantillon(df, nom_table: str, n: int) -> dict:
    ech = dt.echantillon(df, n=n)
    contenu = (
        f"Échantillon reproductible de {len(ech)} lignes (graine fixée) issu de **{nom_table}** :\n\n"
        f"{ech.to_markdown(index=False)}\n\n{dt.syntaxe_echantillon(nom_table, len(ech), seed=20260729)}"
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
    contenu = (
        f"**{len(dups)} lignes en doublon** trouvées dans **{nom_table}** (colonne `{colonne}`) :\n\n"
        f"{dups.to_markdown(index=False)}\n\n{dt.syntaxe_doublons(nom_table, colonne)}"
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
    return {"content": contenu}


def reponse_tableau_croise(df, nom_table: str, colonne1: str, colonne2: str) -> dict:
    tab = dt.tableau_croise(df, colonne1, colonne2)
    contenu = (
        f"Tableau croisé (analyse bivariée) de `{colonne1}` et `{colonne2}` dans **{nom_table}** :\n\n"
        f"{tab.to_markdown()}\n\n{dt.syntaxe_tableau_croise(nom_table, colonne1, colonne2)}"
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
    return {"content": contenu, "table": mat.reset_index(), "table_label": f"correlation_{nom_table}"}


def reponse_tableau_multivarie(df, nom_table: str, colonnes: list[str]) -> dict:
    tab = dt.tableau_multivarie(df, colonnes)
    contenu = (
        f"Analyse multivariée (effectifs croisés) de {', '.join(f'`{c}`' for c in colonnes)} "
        f"dans **{nom_table}** ({len(tab)} combinaisons observées) :\n\n"
        f"{tab.head(30).to_markdown(index=False)}\n\n{dt.syntaxe_tableau_multivarie(nom_table, colonnes)}"
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


def reponse_performance_terrain(tables: dict, exclure: list[str] | None, objectif: int) -> dict:
    rapport = dt.rapport_performance_agents(tables, exclure=exclure)
    if rapport.empty:
        return {
            "content": (
                "Aucune colonne d'agent enquêteur détectée automatiquement dans les tables "
                "chargées : impossible de calculer un rapport de performance de terrain."
            )
        }
    rapport, nom_equipe = dt.fusion_agent_controleur(rapport, tables)
    par_jour = dt.rapport_performance_par_jour(tables)
    prevision = dt.prevision_objectif(par_jour, objectif=objectif) if not par_jour.empty else None

    morceaux = [
        f"**Rapport de performance de terrain** ({len(rapport)} agent(s)"
        + (f", exclusion de {len(exclure)} agent(s) non-terrain" if exclure else "")
        + ") :\n\n" + rapport.to_markdown(index=False)
    ]
    if nom_equipe:
        morceaux.append(f"_Contrôleur ajouté à partir de la table équipe **{nom_equipe}**._")
    else:
        morceaux.append(
            "_Aucune table équipe (agent ↔ contrôleur) détectée parmi les tables chargées : "
            "dépose-la pour faire apparaître la colonne `controleur`._"
        )

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

    return {
        "content": "\n\n".join(morceaux), "table": rapport, "table_label": "performance_terrain",
        "docx_bytes": docx_bytes, "docx_label": "rapport_performance_terrain",
        "chart_data": chart_data,
    }


def reponse_historique_actualisations() -> dict:
    historique = st.session_state.get("historique_chargements", [])
    if not historique:
        return {"content": "Aucune table n'a encore été chargée durant cette session."}
    lignes = ["**Historique des actualisations de cette session** :"]
    for entree in reversed(historique):
        lignes.append(
            f"- {entree['horodatage'].strftime('%d/%m/%Y %H:%M:%S')} — **{entree['table']}** "
            f"({entree['n_lignes']} ligne(s)), depuis `{entree['fichier']}`"
        )
    return {"content": "\n".join(lignes)}


def reponse_recherche_identifiant(identifiant: str, tables: dict) -> dict:
    resultats = dt.rechercher_identifiant(identifiant, tables)
    if not resultats:
        return {"content": f"Aucune fiche trouvée pour l'identifiant **{identifiant}** dans les tables chargées."}
    morceaux = [f"**Recherche de l'identifiant `{identifiant}`** — trouvé dans {len(resultats)} table(s) :"]
    for nom, df in resultats.items():
        morceaux.append(f"\n**{nom}** ({len(df)} ligne(s)) :\n\n{df.head(10).to_markdown(index=False)}")
    return {"content": "\n".join(morceaux)}


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
        return {"content": formater_rapport_coherence_avancee(rapport_avance)}

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
        return reponse_performance_terrain(tables, exclure, objectif)

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
            return {"content": dt.relation_entre_tables(tables_mentionnees[0], tables_mentionnees[1], tables)}
        return {"content": dt.rapport_relations(tables)}

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

    nom_table, df = dt.resoudre_table_ciblee(question, tables, table_active_nom, historique=historique_recent())

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
