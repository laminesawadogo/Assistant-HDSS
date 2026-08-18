"""
Ingestion et vectorisation du corpus documentaire de l'OPO.

Lit tous les fichiers texte de data/docs/ (generes a partir du dictionnaire
de donnees), les decoupe en chunks (un chunk = une variable ou une ligne de
correspondance de table, avec le contexte de la table en prefixe), les
vectorise avec TF-IDF (scikit-learn - pas de dependance lourde, pas de
telechargement de modele, fonctionne hors ligne), et sauvegarde l'index.

Pour passer a des embeddings semantiques (sentence-transformers / Hugging
Face) plus tard, voir la note en bas de fichier.
"""

import os
import pickle
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

import prepare_corpus

DOCS_DIR = Path(__file__).parent / "data" / "docs"
INDEX_PATH = Path(__file__).parent / "data" / "index" / "index.pkl"


def index_obsolete() -> bool:
    """Detecte si l'index est perime : absent, ou plus ancien qu'un document
    depose/modifie depuis dans data/docs/ ou data/source_documents/.

    Bug reel corrige ici : l'appli ne reconstruisait l'index QU'AU TOUT
    PREMIER demarrage (`if not rag.index_exists()`), jamais ensuite - un
    fichier depose directement dans data/source_documents/ (en dehors du
    bouton d'upload de l'interface, qui reconstruit deja l'index lui-meme)
    restait donc silencieusement ignore indefiniment, meme apres un
    redemarrage de l'appli, puisque l'index existait deja (juste perime).
    Constate concretement : plusieurs fiches R14 (.doc) et deux documents
    (Dictionnaire_variables.txt, Manual of.txt) deposes le 11 aout etaient
    absents de l'index construit le 3 aout, sans aucune erreur visible.

    Ne verifie que les dates de modification (rapide, aucune conversion) -
    la conversion elle-meme reste geree par `prepare_corpus.preparer_corpus`,
    appele par `build_index` uniquement si necessaire."""
    if not INDEX_PATH.exists():
        return True
    index_mtime = INDEX_PATH.stat().st_mtime
    for dossier in (DOCS_DIR, prepare_corpus.SOURCE_DIR):
        if not dossier.exists():
            continue
        for f in dossier.iterdir():
            if f.is_file() and f.stat().st_mtime > index_mtime:
                return True
    return False


def chunk_file(path: Path) -> list[dict]:
    """Decoupe un fichier doc en chunks a partir des blocs separes par une
    ligne vide (un bloc = une variable pour les docs du dictionnaire, un
    paragraphe pour les docs de type schema/correspondance).

    Le contexte utilise comme prefixe est uniquement la premiere ligne du
    fichier (ex: "Table: Tindividual" ou "Document: Schema relationnel..."),
    pas tout le premier bloc - sinon un bloc d'intro un peu long pollue tous
    les chunks suivants et dilue la pertinence du TF-IDF.
    """
    import re

    contenu = path.read_text(encoding="utf-8").strip()
    if not contenu:
        return []

    blocs = re.split(r"\n\s*\n", contenu)
    premiere_ligne = blocs[0].splitlines()[0].strip()
    contexte = f"[{path.stem}] {premiere_ligne}"

    corps = blocs[1:] if len(blocs) > 1 else blocs
    chunks = []
    for bloc in corps:
        texte = " ".join(l.strip() for l in bloc.splitlines() if l.strip())
        if not texte:
            continue
        chunks.append({"source": path.stem, "text": f"{contexte} — {texte}"})

    return chunks


def build_index() -> None:
    # Convertit d'abord tout fichier brut depose dans data/source_documents/
    # (Word, PDF, texte...) en texte dans data/docs/, pour que l'index reflete
    # toujours l'ensemble des documents de reference disponibles, pas
    # seulement les fiches du dictionnaire generees a la main.
    nouveaux = prepare_corpus.preparer_corpus()
    if nouveaux:
        print(f"{len(nouveaux)} document(s) source converti(s) : {', '.join(nouveaux)}")

    all_chunks: list[dict] = []
    for f in sorted(DOCS_DIR.glob("*.txt")):
        all_chunks.extend(chunk_file(f))

    if not all_chunks:
        raise RuntimeError(f"Aucun document trouve dans {DOCS_DIR}")

    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_df=0.9,
    )
    matrix = vectorizer.fit_transform(texts)

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(
            {"vectorizer": vectorizer, "matrix": matrix, "chunks": all_chunks},
            f,
        )

    print(f"Index construit : {len(all_chunks)} chunks, {len(list(DOCS_DIR.glob('*.txt')))} documents source.")
    print(f"Sauvegarde dans {INDEX_PATH}")


if __name__ == "__main__":
    build_index()

# ---------------------------------------------------------------------------
# Pour passer a des embeddings semantiques (meilleure comprehension du sens,
# mais necessite de telecharger un modele ~90 Mo la premiere fois) :
#   pip install sentence-transformers
#   from sentence_transformers import SentenceTransformer
#   model = SentenceTransformer("all-MiniLM-L6-v2")
#   matrix = model.encode(texts)
# Puis adapter rag.py pour utiliser model.encode(query) au lieu du TF-IDF.
# ---------------------------------------------------------------------------
