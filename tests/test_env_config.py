"""
Tests du chargement de la cle du modele depuis un fichier .env
(python-dotenv), pour que l'equipe n'ait jamais besoin de ressaisir sa cle
dans l'application (cf. app.py:load_dotenv et .env.exemple).
"""

from pathlib import Path

from dotenv import load_dotenv

import rag

# Le ".env.exemple" est le gabarit destine a etre partage/copie : il ne doit
# jamais contenir une vraie cle, seulement un placeholder. Le ".env" reel, lui,
# est le fichier de config *personnel* de l'equipe une fois rempli : il est
# normal et attendu qu'il contienne une vraie cle a ce moment-la, donc on ne
# le verifie pas ici (sous peine de faire echouer les tests des que quelqu'un
# configure correctement sa cle, ce qui serait un faux positif).
ENV_EXEMPLE_PATH = Path(__file__).parent.parent / ".env.exemple"


def test_dotenv_charge_bien_une_cle_dans_les_variables_denvironnement(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    faux_env = tmp_path / ".env"
    faux_env.write_text("ANTHROPIC_API_KEY=sk-ant-cle-de-test\n", encoding="utf-8")
    load_dotenv(faux_env)

    assert rag.has_llm_configured() is True


def test_dotenv_ignore_une_ligne_commentee(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    faux_env = tmp_path / ".env"
    faux_env.write_text("# ANTHROPIC_API_KEY=sk-ant-cle-de-test\n", encoding="utf-8")
    load_dotenv(faux_env)

    assert rag.has_llm_configured() is False


def test_le_gabarit_env_exemple_ne_contient_jamais_de_cle_active():
    """Le gabarit .env.exemple (destine a etre copie/partage) ne doit jamais
    contenir de vraie cle active, seulement un exemple clairement factice."""
    if not ENV_EXEMPLE_PATH.exists():
        return
    for ligne in ENV_EXEMPLE_PATH.read_text(encoding="utf-8").splitlines():
        ligne_nettoyee = ligne.strip()
        if ligne_nettoyee.startswith("ANTHROPIC_API_KEY=") and "colle-ta-cle" not in ligne_nettoyee:
            raise AssertionError(
                "Le gabarit .env.exemple semble contenir une vraie cle : il doit "
                "toujours rester un exemple factice, jamais un vrai secret."
            )
