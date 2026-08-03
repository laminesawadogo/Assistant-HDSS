"""
Tests de la configuration d'authentification (auth_config.yaml) :
structure attendue et verification que les mots de passe peuvent bien
etre hashes et valides (sans dependre du runtime Streamlit).
"""

from pathlib import Path

import streamlit_authenticator as stauth
import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "auth_config.yaml"


def charger_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_config_existe_et_est_valide():
    assert CONFIG_PATH.exists()
    config = charger_config()
    assert "credentials" in config
    assert "usernames" in config["credentials"]
    assert "cookie" in config
    assert config["cookie"]["name"]
    assert config["cookie"]["key"]


def test_chaque_compte_a_un_role_connu():
    config = charger_config()
    roles_valides = {"consultation", "correction"}
    for username, infos in config["credentials"]["usernames"].items():
        assert "password" in infos
        assert "name" in infos
        assert infos.get("role") in roles_valides, f"Rôle invalide ou manquant pour {username}"


def test_hash_et_verification_du_mot_de_passe():
    mot_de_passe = "UnMotDePasseDeTest123!"
    hashe = stauth.Hasher.hash(mot_de_passe)
    assert hashe != mot_de_passe
    assert stauth.Hasher.check_pw(mot_de_passe, hashe) is True
    assert stauth.Hasher.check_pw("mauvais_mot_de_passe", hashe) is False


def test_cle_cookie_par_defaut_doit_etre_changee_avant_prod():
    # Rappel de securite : le fichier livre par defaut contient un
    # placeholder, il doit etre remplace avant tout deploiement reel.
    config = charger_config()
    cle = config["cookie"]["key"]
    if cle == "REMPLACER_PAR_UNE_CLE_SECRETE_ALEATOIRE":
        import warnings
        warnings.warn(
            "auth_config.yaml utilise encore la cle cookie par defaut : "
            "a remplacer avant tout deploiement reel.",
            UserWarning,
        )
