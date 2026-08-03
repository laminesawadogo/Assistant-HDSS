"""
Authentification de l'assistant OPO (streamlit-authenticator) : un ecran de
connexion bloque l'acces a toute la session tant qu'un compte valide n'a pas
ete utilise - l'appli n'est plus consultable/modifiable par n'importe qui
disposant simplement du lien de deploiement.

Les identifiants sont lus en priorite depuis st.secrets["auth"] (gestionnaire
de secrets Streamlit Cloud - jamais commite dans le depot Git, meme public),
et a defaut depuis auth_config.yaml (pratique en local, mais qui NE DOIT
JAMAIS contenir de vrais mots de passe si le depot est public - voir les
instructions en tete de ce fichier de configuration).
"""

from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
import yaml

CONFIG_PATH = Path(__file__).parent / "auth_config.yaml"

MOTS_DE_PASSE_PLACEHOLDER = {"ChangezMoi123!", "ChangezMoiAussi123!"}
CLE_COOKIE_PLACEHOLDER = "REMPLACER_PAR_UNE_CLE_SECRETE_ALEATOIRE"


def _en_dict_modifiable(objet):
    """Convertit RECURSIVEMENT un objet st.secrets (en lecture seule a tous
    les niveaux - meme les sous-tables comme chaque compte utilisateur) en
    dict/list Python standards et modifiables. Necessaire car
    streamlit_authenticator ecrit dans les identifiants en place (ex: il
    remplace le mot de passe en clair par sa version hachee au premier
    lancement) : une conversion en surface seulement (dict(...) sur le
    premier niveau) laisse les sous-objets encore verrouilles, d'ou un
    `TypeError: Secrets does not support item assignment` observe en
    deploiement reel."""
    if hasattr(objet, "items"):
        return {cle: _en_dict_modifiable(valeur) for cle, valeur in objet.items()}
    if isinstance(objet, (list, tuple)):
        return [_en_dict_modifiable(v) for v in objet]
    return objet


def charger_config() -> dict:
    """Charge la configuration d'authentification depuis st.secrets["auth"]
    (prioritaire, utilise sur Streamlit Cloud) ou depuis auth_config.yaml
    (repli local pour le developpement)."""
    try:
        secrets_auth = st.secrets.get("auth")
    except Exception:
        secrets_auth = None
    if secrets_auth:
        secrets_auth = _en_dict_modifiable(secrets_auth)
        return {
            "credentials": {"usernames": secrets_auth["usernames"]},
            "cookie": secrets_auth.get("cookie", {}),
        }

    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _avertissement_config_par_defaut(config: dict) -> str | None:
    """Signale, directement dans l'interface (pas seulement dans les tests),
    si la configuration livree par defaut (mots de passe/cle cookie
    d'exemple) n'a pas encore ete remplacee - pour que l'equipe le voie
    tout de suite plutot que de le decouvrir apres coup."""
    usernames = config.get("credentials", {}).get("usernames", {})
    mots_de_passe_par_defaut = any(
        infos.get("password") in MOTS_DE_PASSE_PLACEHOLDER for infos in usernames.values()
    )
    cle_par_defaut = config.get("cookie", {}).get("key") == CLE_COOKIE_PLACEHOLDER
    if mots_de_passe_par_defaut or cle_par_defaut:
        return (
            "⚠️ **Configuration de connexion par défaut détectée** (auth_config.yaml) : "
            "mots de passe et/ou clé de cookie d'exemple encore utilisés. À remplacer avant "
            "toute utilisation avec de vraies données — idéalement via les *Secrets* de "
            "Streamlit Cloud plutôt que dans le fichier (voir le haut de auth_config.yaml)."
        )
    return None


def _construire_authenticator(
    identifiants: dict, nom_cookie: str, cle_cookie: str, expiry_days: float
) -> stauth.Authenticate:
    """Construit l'objet Authenticate. Reconstruit a chaque rerun Streamlit,
    comme le prevoit la bibliotheque elle-meme (tous ses exemples officiels
    suivent ce pattern) : NE PAS mettre en cache via st.cache_resource, le
    composant de gestion de cookies qu'il cree en interne (CookieManager) se
    comporte comme un widget Streamlit, et Streamlit interdit explicitement
    les widgets a l'interieur d'une fonction mise en cache (CachedWidgetWarning)
    - ca a ete teste en deploiement reel et ca casse la connexion (KeyError
    sur st.session_state['logout'], l'initialisation interne de la
    bibliotheque n'a pas pu se derouler normalement)."""
    return stauth.Authenticate(identifiants, nom_cookie, cle_cookie, expiry_days)


def verifier_acces() -> dict:
    """Affiche l'ecran de connexion et bloque l'execution du reste de l'appli
    (st.stop()) tant que l'utilisateur n'est pas authentifie avec succes.

    A appeler tout en haut de app.py, avant tout autre element d'interface
    (juste apres st.set_page_config). Si une session est deja authentifiee
    (rerun Streamlit normal, ou session de test qui a pre-rempli
    st.session_state["authentication_status"]), le formulaire n'est pas
    rejoue.

    Renvoie {"name", "username", "role"} de la personne connectee."""
    config = charger_config()
    identifiants = config["credentials"]
    cookie = config.get("cookie", {})
    authenticator = _construire_authenticator(
        identifiants,
        cookie.get("name", "opo_auth_cookie"),
        cookie.get("key", CLE_COOKIE_PLACEHOLDER),
        cookie.get("expiry_days", 7),
    )

    if st.session_state.get("authentication_status") is not True:
        authenticator.login(location="main")
        statut = st.session_state.get("authentication_status")

        if statut is False:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
            st.stop()
        if statut is None:
            st.info("Connecte-toi pour accéder à l'assistant OPO.")
            st.stop()

    username = st.session_state.get("username")
    infos_utilisateur = config["credentials"]["usernames"].get(username, {})
    role = infos_utilisateur.get("role", "consultation")

    with st.sidebar:
        st.caption(f"Connecté(e) : **{st.session_state.get('name', username)}** ({role})")
        authenticator.logout("Se déconnecter", location="sidebar")

    avertissement = _avertissement_config_par_defaut(config)
    if avertissement:
        st.warning(avertissement)

    return {"name": st.session_state.get("name", username), "username": username, "role": role}
