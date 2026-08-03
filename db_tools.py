"""
Connexion directe a la vraie base de donnees de l'observatoire (ex: PostgreSQL
de l'application HRS-OPO), en plus des fichiers CSV/Excel deposes a la main.

Les identifiants de connexion sont lus depuis une variable d'environnement
(DATABASE_URL, chargee en arriere-plan comme la cle du LLM - voir .env.exemple
et rag.py) : personne n'a besoin de les ressaisir dans l'interface.

Important - portee reseau : cette connexion ne fonctionne que si la machine
qui execute l'application peut atteindre le serveur de base de donnees sur le
reseau. Si ce serveur est uniquement accessible sur le reseau interne de
l'OPO, l'application doit alors tourner sur ce meme reseau (ou via VPN) -
pas depuis un hebergement public comme Streamlit Community Cloud, qui n'a
aucun moyen d'atteindre un serveur qui n'est pas expose sur internet.

Une fois une table chargee depuis la base (`charger_table_sql`), elle est
traitee exactement comme une table deposee en CSV/Excel : memes fonctions
d'analyse (data_tools.py), meme retrait automatique des colonnes
nominatives (strip_names), memes exports CSV/Excel/Stata.
"""

from __future__ import annotations

import os

import pandas as pd

import data_tools as dt

# Format attendu (chaine de connexion SQLAlchemy), a definir dans .env :
#   DATABASE_URL=postgresql+psycopg2://utilisateur:motdepasse@hote:5432/nom_base
# Un compte de connexion en lecture seule (SELECT uniquement) est fortement
# recommande : l'assistant ne doit jamais pouvoir modifier la base reelle,
# conformement a la regle "signalement, jamais de correction automatique".
DATABASE_URL_ENV = "DATABASE_URL"

_cache_moteur = {}


class ConnexionBaseIndisponible(RuntimeError):
    """Levee quand DATABASE_URL n'est pas configuree ou que la connexion echoue."""


def base_configuree(database_url: str | None = None) -> bool:
    return bool(database_url or os.getenv(DATABASE_URL_ENV))


def _obtenir_moteur(database_url: str | None = None):
    """Cree (ou reutilise) le moteur SQLAlchemy de connexion a la base."""
    url = database_url or os.getenv(DATABASE_URL_ENV)
    if not url:
        raise ConnexionBaseIndisponible(
            "Aucune base connectee : configure DATABASE_URL (dans .env ou les secrets "
            "de l'hebergement) pour te connecter directement a la base de l'observatoire."
        )
    if url not in _cache_moteur:
        from sqlalchemy import create_engine

        try:
            _cache_moteur[url] = create_engine(url, pool_pre_ping=True)
        except Exception as e:
            raise ConnexionBaseIndisponible(f"Échec de connexion à la base : {e}") from e
    return _cache_moteur[url]


def reset_cache() -> None:
    """Vide le cache des moteurs de connexion (utile si DATABASE_URL change)."""
    _cache_moteur.clear()


def lister_tables(database_url: str | None = None) -> list[str]:
    """Renvoie la liste des tables visibles dans la base connectee."""
    from sqlalchemy import inspect

    moteur = _obtenir_moteur(database_url)
    try:
        return sorted(inspect(moteur).get_table_names())
    except Exception as e:
        raise ConnexionBaseIndisponible(f"Impossible de lister les tables : {e}") from e


def charger_table_sql(
    nom_table: str, database_url: str | None = None, limite: int | None = None
) -> pd.DataFrame:
    """Charge une table de la base connectee dans un DataFrame, avec les memes
    garde-fous que pour un fichier depose (retrait des colonnes nominatives).

    `limite` permet de ne charger que les N premieres lignes (utile pour une
    table volumineuse, ex: verifier la structure avant de tout charger)."""
    moteur = _obtenir_moteur(database_url)
    requete = f'SELECT * FROM "{nom_table}"'
    if limite is not None:
        requete += f" LIMIT {int(limite)}"
    try:
        df = pd.read_sql(requete, moteur)
    except Exception as e:
        raise ConnexionBaseIndisponible(f"Échec de lecture de la table '{nom_table}' : {e}") from e
    return dt.strip_names(df)
