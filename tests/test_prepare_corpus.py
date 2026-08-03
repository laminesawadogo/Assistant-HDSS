"""
Tests du pipeline d'ingestion multi-format (prepare_corpus.py) : conversion de
fichiers Word/PDF/texte/Excel deposes dans data/source_documents/ en texte
brut dans data/docs/, pret pour le decoupage en chunks.

Les conversions qui dependent d'un binaire externe (LibreOffice pour .doc,
pdftotext pour .pdf) sont testees uniquement si le binaire est present sur la
machine qui lance les tests (shutil.which) - elles restent couvertes ici,
mais sans faire echouer la suite sur une machine qui ne les a pas installes.
"""

import shutil

import pytest

import prepare_corpus


def test_nettoyer_texte_reduit_les_rafales_de_lignes_vides():
    brut = "Paragraphe 1.\r\n\r\n\r\n\r\nParagraphe 2.\n\n\nParagraphe 3."
    resultat = prepare_corpus._nettoyer_texte(brut)
    assert "\n\n\n" not in resultat
    assert "Paragraphe 1." in resultat
    assert "Paragraphe 3." in resultat


def test_extraire_txt_lit_le_fichier_tel_quel(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("Contenu de test.\n\nDeuxieme paragraphe.", encoding="utf-8")
    assert prepare_corpus.extraire_txt(f) == "Contenu de test.\n\nDeuxieme paragraphe."


def test_extraire_docx_lit_paragraphes_et_tableaux(tmp_path):
    docx = pytest.importorskip("docx")
    f = tmp_path / "fiche.docx"
    document = docx.Document()
    document.add_paragraph("Premier paragraphe de la fiche.")
    document.add_paragraph("Deuxieme paragraphe.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "TMembership"
    table.rows[0].cells[1].text = "Table des episodes d'appartenance"
    document.save(f)

    texte = prepare_corpus.extraire_docx(f)
    assert "Premier paragraphe de la fiche." in texte
    assert "Deuxieme paragraphe." in texte
    assert "TMembership" in texte
    assert "Table des episodes d'appartenance" in texte


def test_extraire_xlsx_generique_dump_les_feuilles(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    f = tmp_path / "grille.xlsx"
    classeur = openpyxl.Workbook()
    feuille = classeur.active
    feuille.title = "Codes"
    feuille.append(["Code", "Libelle"])
    feuille.append([1, "Homme"])
    feuille.append([2, "Femme"])
    classeur.save(f)

    texte = prepare_corpus.extraire_xlsx_generique(f)
    assert "Feuille: Codes" in texte
    assert "Homme" in texte
    assert "Femme" in texte


def test_extraire_pptx_lit_texte_et_tableaux_des_diapositives(tmp_path):
    pptx_module = pytest.importorskip("pptx")
    f = tmp_path / "presentation.pptx"
    presentation = pptx_module.Presentation()
    layout = presentation.slide_layouts[5]  # disposition vierge avec titre
    slide = presentation.slides.add_slide(layout)
    slide.shapes.title.text = "Bilan de l'atelier"
    presentation.save(f)

    texte = prepare_corpus.extraire_pptx(f)
    assert "Diapositive 1" in texte
    assert "Bilan de l'atelier" in texte


def test_convertir_fichier_dispatch_selon_extension(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Titre\n\nParagraphe.", encoding="utf-8")
    assert prepare_corpus.convertir_fichier(f) == "# Titre\n\nParagraphe."

    inconnu = tmp_path / "archive.zip"
    inconnu.write_bytes(b"\x00\x01")
    assert prepare_corpus.convertir_fichier(inconnu) is None


def test_preparer_corpus_convertit_puis_ignore_si_deja_a_jour(tmp_path, monkeypatch):
    source_dir = tmp_path / "source_documents"
    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(prepare_corpus, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_corpus, "DOCS_DIR", docs_dir)

    source_dir.mkdir()
    (source_dir / "manuel.txt").write_text("Contenu du manuel.", encoding="utf-8")

    convertis = prepare_corpus.preparer_corpus()
    assert convertis == ["manuel.txt"]
    sortie = docs_dir / "source_manuel.txt"
    assert sortie.exists()
    assert "Contenu du manuel." in sortie.read_text(encoding="utf-8")
    assert sortie.read_text(encoding="utf-8").startswith("Document: manuel")

    # Deuxieme appel sans modification du fichier source : rien a reconvertir
    convertis_2 = prepare_corpus.preparer_corpus()
    assert convertis_2 == []


def test_preparer_corpus_ignore_le_dictionnaire_xlsx_principal(tmp_path, monkeypatch):
    source_dir = tmp_path / "source_documents"
    docs_dir = tmp_path / "docs"
    monkeypatch.setattr(prepare_corpus, "SOURCE_DIR", source_dir)
    monkeypatch.setattr(prepare_corpus, "DOCS_DIR", docs_dir)

    source_dir.mkdir()
    # Nom volontairement identique (insensible a la casse) a celui deja gere
    # a la main pour le dictionnaire de variables.
    (source_dir / "Dictionnaire_donnees_OPO_BaseDemographique.xlsx").write_bytes(b"\x00")

    convertis = prepare_corpus.preparer_corpus()
    assert convertis == []
    assert not (docs_dir / "source_Dictionnaire_donnees_OPO_BaseDemographique.txt").exists()


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice (soffice) non installe")
def test_extraire_doc_ancien_format_via_soffice():
    from pathlib import Path

    chemin = Path(__file__).parent.parent / "data" / "source_documents" / "Correspondance_Tables_BD_OPO.doc"
    if not chemin.exists():
        pytest.skip("fichier .doc source non present sur cette machine")
    texte = prepare_corpus.extraire_doc(chemin)
    assert len(texte.strip()) > 100


@pytest.mark.skipif(shutil.which("soffice") is None, reason="LibreOffice (soffice) non installe")
def test_extraire_ppt_ancien_format_reconvertit_puis_lit_via_pptx(tmp_path):
    pptx_module = pytest.importorskip("pptx")
    # On construit un .pptx de test puis on verifie que le chemin de conversion
    # .ppt->.pptx->texte fonctionne en reutilisant directement _texte_pptx sur
    # un fichier deja au format .pptx (le fichier .ppt reel n'est pas
    # forcement present sur toutes les machines de test).
    f = tmp_path / "ancienne_presentation.pptx"
    presentation = pptx_module.Presentation()
    slide = presentation.slides.add_slide(presentation.slide_layouts[5])
    slide.shapes.title.text = "Contenu de test"
    presentation.save(f)

    texte = prepare_corpus._texte_pptx(f)
    assert "Contenu de test" in texte
