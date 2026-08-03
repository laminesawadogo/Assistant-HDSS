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
