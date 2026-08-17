"""
Tests du module d'authentification (auth.py) - en particulier de l'ajout du
bouton "afficher/masquer" sur le champ mot de passe (voir
auth._activer_affichage_mot_de_passe).

streamlit_authenticator construit lui-meme son st.form et son
st.text_input(type="password") a l'interieur de sa propre methode login()
(authentication_view.py de la bibliotheque) : ce widget n'est pas expose,
impossible d'y ajouter une case a cocher Python cote serveur sans forker la
bibliotheque. La solution retenue est un petit script cote client (via
st.iframe, avec repli sur components.v1.html pour les anciennes versions de
Streamlit) qui repere le champ input[type=password] dans le document PARENT
et lui ajoute un bouton oeil. Ces tests ne peuvent pas verifier le
comportement JS lui-meme (AppTest ne rend pas de vrai DOM/navigateur) : ils
verifient seulement que l'injection ne fait pas planter l'application et
qu'un element est bien rendu.
"""

from streamlit.testing.v1 import AppTest

SCRIPT_TOGGLE_MDP = """
import auth
auth._activer_affichage_mot_de_passe()
"""


def _app_toggle_mdp() -> AppTest:
    at = AppTest.from_string(SCRIPT_TOGGLE_MDP, default_timeout=30)
    at.run()
    return at


def test_activer_affichage_mot_de_passe_ne_leve_pas_dexception():
    at = _app_toggle_mdp()
    assert not at.exception


def test_activer_affichage_mot_de_passe_rend_bien_un_composant():
    at = _app_toggle_mdp()
    assert not at.exception
    # Le composant html/iframe qui porte le script d'injection du bouton
    # oeil doit avoir ete rendu (peu importe l'API Streamlit utilisee en
    # interne - st.iframe ou son repli components.v1.html).
    assert len(at.main) == 1


def test_activer_affichage_mot_de_passe_est_reappele_sans_erreur():
    # verifie_acces() rappelle _activer_affichage_mot_de_passe() a chaque
    # rerun tant que l'utilisateur n'est pas connecte (Streamlit re-execute
    # tout le script a chaque interaction) : deux appels d'affilee, comme le
    # sondage JS interne (setInterval), ne doivent jamais lever d'exception.
    script = SCRIPT_TOGGLE_MDP + "\nauth._activer_affichage_mot_de_passe()\n"
    at = AppTest.from_string(script, default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.main) == 2
