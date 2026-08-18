"""
Tests de ingest.py - en particulier `index_obsolete()`, qui detecte qu'un
document source a ete ajoute ou modifie apres la derniere construction de
l'index.

Bug reel corrige par cette fonction : l'application ne (re)construisait
l'index QU'AU TOUT PREMIER demarrage (`if not rag.index_exists()`), jamais
ensuite - un fichier depose directement dans data/source_documents/ (en
dehors du bouton d'upload de l'interface, qui reconstruit deja l'index
lui-meme immediatement) restait donc silencieusement ignore indefiniment,
meme apres un redemarrage de l'appli, puisque l'index existait deja (juste
perime). Constate concretement sur le vrai projet : plusieurs fiches R14
(.doc) et deux documents (Dictionnaire_variables.txt, Manual of.txt)
deposes le 11 aout etaient absents de l'index construit le 3 aout, sans
aucune erreur visible nulle part.
"""

import time

import ingest


def _isoler_dossiers(tmp_path, monkeypatch):
    """Redirige ingest.DOCS_DIR / ingest.INDEX_PATH / prepare_corpus.SOURCE_DIR
    vers des dossiers temporaires isoles, pour ne jamais toucher les vrais
    documents ni le vrai index du projet pendant les tests."""
    docs = tmp_path / "docs"
    docs.mkdir()
    source = tmp_path / "source_documents"
    source.mkdir()
    index_path = tmp_path / "index.pkl"
    monkeypatch.setattr(ingest, "DOCS_DIR", docs)
    monkeypatch.setattr(ingest, "INDEX_PATH", index_path)
    monkeypatch.setattr(ingest.prepare_corpus, "SOURCE_DIR", source)
    return docs, source, index_path


def test_index_obsolete_si_index_absent(tmp_path, monkeypatch):
    _isoler_dossiers(tmp_path, monkeypatch)
    assert ingest.index_obsolete() is True


def test_index_a_jour_si_rien_de_plus_recent_que_lindex(tmp_path, monkeypatch):
    docs, source, index_path = _isoler_dossiers(tmp_path, monkeypatch)
    (docs / "a.txt").write_text("contenu")
    (source / "b.docx").write_bytes(b"")
    time.sleep(0.05)
    index_path.write_bytes(b"")  # index construit APRES les documents existants
    assert ingest.index_obsolete() is False


def test_index_obsolete_si_nouveau_document_source_depose(tmp_path, monkeypatch):
    docs, source, index_path = _isoler_dossiers(tmp_path, monkeypatch)
    index_path.write_bytes(b"")
    time.sleep(0.05)
    (source / "nouveau_document.txt").write_text("deverse directement dans le dossier, sans passer par l'upload")
    assert ingest.index_obsolete() is True


def test_index_obsolete_si_nouveau_doc_converti_present(tmp_path, monkeypatch):
    # Cas d'un fichier deja converti dans docs/ (par prepare_corpus) mais pas
    # encore repris dans le pickle de l'index lui-meme.
    docs, source, index_path = _isoler_dossiers(tmp_path, monkeypatch)
    index_path.write_bytes(b"")
    time.sleep(0.05)
    (docs / "source_nouveau.txt").write_text("Document: nouveau\n\nContenu converti")
    assert ingest.index_obsolete() is True
