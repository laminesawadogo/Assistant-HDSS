"""
Interface web de l'Assistant OPO — chat RAG sur le dictionnaire de donnees +
analyse d'une table deposee (indicateurs, echantillon, controles de
coherence). Lancer avec : streamlit run app.py
"""

import os
import re
import tempfile
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import data_tools as dt
import ingest
import rag

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

st.set_page_config(page_title="Assistant OPO", page_icon="📊", layout="wide")

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
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=suffix).name
                with open(tmp_path, "wb") as f:
                    f.write(fichier.getbuffer())

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
                else:
                    nom_table = re.sub(r"\.(csv|xlsx|xls|dta)$", "", fichier.name, flags=re.IGNORECASE)
                    st.session_state["tables"][nom_table] = dt.load_table(tmp_path)

                st.session_state["fichiers_traites"].add(signature)
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


for i, msg in enumerate(st.session_state["messages"]):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources utilisées"):
                for s in msg["sources"]:
                    st.markdown(f"- ({s['score']:.2f}) {s['text']}")
        if msg.get("exports") is not None:
            afficher_boutons_export(msg["exports"], msg.get("table_label", "export"), cle=f"hist_{i}")


def historique_recent(max_tours: int = 6) -> list[dict]:
    """Renvoie les derniers echanges de la conversation (hors la question en
    cours de traitement, deja ajoutee a l'historique juste avant l'appel a
    route_question), pour permettre les questions de suivi qui font reference
    a ce qui vient d'etre dit ("et pour l'autre table ?", "peux-tu detailler ?")."""
    messages = st.session_state.get("messages", [])
    precedents = messages[:-1] if messages else []
    recents = precedents[-max_tours:]
    return [{"role": m["role"], "contenu": m["content"]} for m in recents]


MOTS_RELATION = ["relation", "reliee", "reliees", "relie", "relies", "lien", "liees", "en commun", "cle commune", "cles communes", "clé commune", "clés communes"]
MOTS_FUSION = ["fusion", "fusionner", "fusionne", "jointure", "joindre", "joins", "merge", "merger"]
MOTS_DIFFERENCE = [
    "mais pas dans", "et pas dans", "pas dans", "absent de", "absents de", "absente de", "absentes de",
    "sauf", "n'est pas dans", "ne sont pas dans", "n'apparaissent pas dans", "n'apparait pas dans",
    "n'apparaît pas dans",
]
MOTS_VICE_VERSA = ["vice versa", "vice-versa", "et inversement", "et l'inverse", "et réciproquement", "et reciproquement"]


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


MOTS_LISTE_TABLES = [
    "combien de table", "combien de feuille", "quelles tables", "quelles sont les tables",
    "liste des tables", "tables chargees", "tables chargées", "tables disponibles",
    "nombre de tables", "nombre de feuilles", "quelles feuilles", "liste des feuilles",
    "tables que je viens de", "tables que je vous ai", "table que je viens de",
    "table que je vous ai", "tables que je t'ai", "table que je t'ai",
    "je viens de vous envoyer", "je viens de vous envoyé", "je viens de t'envoyer",
    "je viens de charger", "je viens de déposer", "je viens de deposer",
]


def route_question(question: str) -> dict:
    """Determine si la question porte sur une table deposee (indicateur,
    echantillon, coherence), sur une relation/fusion entre plusieurs tables
    chargees, sur la liste des tables/feuilles elles-memes, ou sur le
    dictionnaire (RAG)."""
    q = question.lower()
    tables = st.session_state.get("tables", {})

    # Question "meta" sur la session en cours (combien de tables/feuilles
    # sont chargees, lesquelles) : ne concerne pas le contenu d'une table ni
    # le dictionnaire, donc a verifier en tout premier, avant toute
    # resolution de table.
    if any(m in q for m in MOTS_LISTE_TABLES):
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

    # Difference d'ensembles ("qui est dans X mais pas dans Y", et
    # eventuellement "vice versa" pour les deux sens a la fois) - a verifier
    # AVANT la fusion generale, puisque ce sont deux operations distinctes.
    if any(m in q for m in MOTS_DIFFERENCE):
        if len(tables_mentionnees) < 2:
            return {
                "content": (
                    "Précise les deux tables à comparer en les nommant explicitement, ex. : "
                    f"« combien sont dans {list(tables.keys())[0] if tables else 'Presence'} mais pas dans "
                    f"{list(tables.keys())[1] if len(tables) > 1 else 'Education'} »."
                )
            }
        a, b = tables_mentionnees[0], tables_mentionnees[1]
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

        if any(m in q for m in MOTS_VICE_VERSA):
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

    if any(m in q for m in MOTS_FUSION):
        if len(tables_mentionnees) >= 2:
            a, b = tables_mentionnees[0], tables_mentionnees[1]
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
        return {
            "content": (
                "Précise les deux tables à fusionner en les nommant explicitement, ex. : "
                f"« fusionne {list(tables.keys())[0] if tables else 'Tindividual'} et "
                f"{list(tables.keys())[1] if len(tables) > 1 else 'Tsocialgp'} »."
            )
        }

    if any(m in q for m in MOTS_RELATION):
        if len(tables_mentionnees) >= 2:
            return {"content": dt.relation_entre_tables(tables_mentionnees[0], tables_mentionnees[1], tables)}
        return {"content": dt.rapport_relations(tables)}

    nom_table, df = dt.resoudre_table_ciblee(question, tables, table_active_nom, historique=historique_recent())

    if df is not None and "doublon" in q:
        dups = dt.doublons(df)
        if len(dups) == 0:
            return {"content": f"Aucun doublon d'identifiant détecté dans **{nom_table}**."}
        return {
            "content": f"**{len(dups)} lignes en doublon** trouvées dans **{nom_table}** :\n\n{dups.to_markdown(index=False)}",
            "table": dups,
            "table_label": f"doublons_{nom_table}",
        }

    if df is not None and any(m in q for m in ["incoheren", "coheren", "anomalie"]):
        rapport = dt.rapport_coherence(df)
        return {"content": formater_rapport_coherence(rapport, nom_table)}

    if df is not None and any(m in q for m in ["echantillon", "échantillon"]):
        m = re.search(r"\d+", q)
        n = int(m.group()) if m else 100
        ech = dt.echantillon(df, n=n)
        return {
            "content": f"Échantillon reproductible de {len(ech)} lignes (graine fixée) issu de **{nom_table}** :\n\n{ech.to_markdown(index=False)}",
            "table": ech,
            "table_label": f"echantillon_{nom_table}",
        }

    if df is not None and any(m in q for m in ["repartition", "répartition", "indicateur"]):
        col_trouvee = next((c for c in df.columns if c.lower() in q), None)
        if col_trouvee:
            rep = dt.repartition(df, col_trouvee)
            return {
                "content": f"Répartition de `{col_trouvee}` dans **{nom_table}** :\n\n{rep.to_markdown()}",
                "table": rep.reset_index(),
                "table_label": f"repartition_{col_trouvee}_{nom_table}",
            }
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
            rep = dt.repartition(df, parametre)
            return {
                "content": f"Répartition de `{parametre}` dans **{nom_table}** :\n\n{rep.to_markdown()}",
                "table": rep.reset_index(),
                "table_label": f"repartition_{parametre}_{nom_table}",
            }
        if action == "ECHANTILLON":
            ech = dt.echantillon(df, n=parametre or 100)
            return {
                "content": f"Échantillon reproductible de {len(ech)} lignes (graine fixée) issu de **{nom_table}** :\n\n{ech.to_markdown(index=False)}",
                "table": ech,
                "table_label": f"echantillon_{nom_table}",
            }
        if action == "DOUBLONS":
            dups = dt.doublons(df)
            if len(dups) == 0:
                return {"content": f"Aucun doublon d'identifiant détecté dans **{nom_table}**."}
            return {
                "content": f"**{len(dups)} lignes en doublon** trouvées dans **{nom_table}** :\n\n{dups.to_markdown(index=False)}",
                "table": dups,
                "table_label": f"doublons_{nom_table}",
            }
        if action == "COHERENCE":
            rapport = dt.rapport_coherence(df)
            return {"content": formater_rapport_coherence(rapport, nom_table)}
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
        }
    )

st.markdown(
    '<div class="opo-footer">Assistant interne — Observatoire de Population de Ouagadougou (OPO). '
    "Ne remplace pas la validation humaine des corrections de données.</div>",
    unsafe_allow_html=True,
)
