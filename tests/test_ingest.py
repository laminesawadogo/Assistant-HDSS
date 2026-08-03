"""
Tests du decoupage en chunks (ingest.chunk_file) : un bloc separe par une
ligne vide = un chunk, avec un contexte court (premiere ligne du fichier)
en prefixe - et non tout le premier bloc, qui polluerait tous les chunks
suivants pour les documents de type prose (voir schema_relations.txt).
"""

from pathlib import Path

from ingest import chunk_file


def test_chunk_file_un_bloc_par_variable(tmp_path):
    contenu = (
        "Table: Tindividual\n\n"
        "Titre: Table Tindividual\nDescription: Core individual identification\n\n"
        "Variable individid (table Tindividual) : Individal ID.\n\n"
        "Variable sex (table Tindividual) : Sex.\n"
    )
    f = tmp_path / "Tindividual.txt"
    f.write_text(contenu, encoding="utf-8")

    chunks = chunk_file(f)
    # 3 blocs apres le premier (le contexte) : Titre/Description, individid, sex
    assert len(chunks) == 3
    assert all(c["source"] == "Tindividual" for c in chunks)
    # Le contexte (premiere ligne seulement) doit prefixer chaque chunk
    assert all("Table: Tindividual" in c["text"] for c in chunks)
    # Le corps du chunk doit contenir la variable, pas tout le premier bloc
    assert any("individid" in c["text"] for c in chunks)
    assert any("sex" in c["text"] for c in chunks)


def test_chunk_file_prose_ne_repete_pas_le_premier_paragraphe(tmp_path):
    contenu = (
        "Document: Schema relationnel\n\n"
        "Ceci est un long paragraphe d'introduction qui ne doit pas polluer "
        "tous les chunks suivants.\n\n"
        "TMembership : cle primaire episodeid, cle etrangere socialgpid.\n\n"
        "TResidency : cle primaire episodeid, cle etrangere locationid.\n"
    )
    f = tmp_path / "00_schema_test.txt"
    f.write_text(contenu, encoding="utf-8")

    chunks = chunk_file(f)
    assert len(chunks) == 3
    # Le long paragraphe d'intro ne doit apparaitre que dans son propre chunk
    textes_avec_intro = [c for c in chunks if "long paragraphe" in c["text"]]
    assert len(textes_avec_intro) == 1
    # Les chunks TMembership et TResidency doivent rester courts et distincts
    chunk_membership = next(c for c in chunks if "TMembership" in c["text"])
    assert "TResidency" not in chunk_membership["text"].split("—", 1)[-1]


def test_chunk_file_fichier_vide(tmp_path):
    f = tmp_path / "vide.txt"
    f.write_text("", encoding="utf-8")
    assert chunk_file(f) == []
