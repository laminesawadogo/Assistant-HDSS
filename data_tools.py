"""
Analyse generique de tables (CSV/Excel) : fonctionne quelles que soient les
colonnes reellement presentes, sans dependre de la structure exacte du
dictionnaire. Detecte automatiquement les colonnes d'identifiant et de date
par le nom des colonnes / le contenu.

Regle de securite : toute colonne qui ressemble a un nom/prenom est
systematiquement exclue des resultats affiches (meme si elle est presente
dans le fichier source), conformement a la regle de non-exposition des
donnees nominatives de l'assistant OPO.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

import numpy as np
import pandas as pd

NAME_LIKE = re.compile(r"(nom|name|prenom|prénom|surname|firstname|lastname)", re.IGNORECASE)
ID_LIKE = re.compile(r"(id|identif)", re.IGNORECASE)


def load_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return strip_names(df)


def strip_names(df: pd.DataFrame) -> pd.DataFrame:
    """Retire toute colonne qui ressemble a un nom/prenom avant analyse."""
    to_drop = [c for c in df.columns if NAME_LIKE.search(str(c))]
    if to_drop:
        df = df.drop(columns=to_drop)
    return df


def detect_id_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if ID_LIKE.search(str(c))]


def detect_date_columns(df: pd.DataFrame) -> list[str]:
    date_cols = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            date_cols.append(c)
            continue
        if df[c].dtype == object:
            sample = df[c].dropna().astype(str).head(20)
            if len(sample) == 0:
                continue
            parsed = pd.to_datetime(sample, errors="coerce", dayfirst=True)
            if parsed.notna().mean() > 0.7:
                date_cols.append(c)
    return date_cols


def repartition(df: pd.DataFrame, colonne: str) -> pd.DataFrame:
    if colonne not in df.columns:
        raise ValueError(f"Colonne '{colonne}' absente. Colonnes disponibles : {list(df.columns)}")
    out = df[colonne].value_counts(dropna=False).rename("effectif").to_frame()
    out["pourcentage"] = (100 * out["effectif"] / out["effectif"].sum()).round(1)
    return out


def echantillon(df: pd.DataFrame, n: int = 100, seed: int = 20260729) -> pd.DataFrame:
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def doublons(df: pd.DataFrame, colonne: str | None = None) -> pd.DataFrame:
    """Doublons sur une colonne d'identifiant (auto-detectee si non precisee)."""
    if colonne is None:
        ids = detect_id_columns(df)
        if not ids:
            raise ValueError("Aucune colonne d'identifiant detectee automatiquement ; precisez `colonne`.")
        colonne = ids[0]
    counts = df[colonne].value_counts()
    dup_values = counts[counts > 1].index
    return df[df[colonne].isin(dup_values)].sort_values(colonne)


def dates_incoherentes(df: pd.DataFrame, colonne: str, borne_min: str = "1900-01-01") -> pd.DataFrame:
    parsed = pd.to_datetime(df[colonne], errors="coerce", dayfirst=True)
    aujourdhui = pd.Timestamp(datetime.now())
    mask = parsed.isna() | (parsed > aujourdhui) | (parsed < pd.Timestamp(borne_min))
    return df[mask]


def rapport_coherence(df: pd.DataFrame) -> dict:
    """Rapport generique : doublons sur les colonnes d'ID detectees, dates
    invraisemblables sur les colonnes de date detectees. Aucune correction
    n'est appliquee : uniquement un signalement, a valider par une personne
    habilitee."""
    rapport = {"n_lignes": len(df), "colonnes": list(df.columns), "anomalies": {}}

    for id_col in detect_id_columns(df):
        dups = doublons(df, id_col)
        if len(dups) > 0:
            rapport["anomalies"][f"doublons::{id_col}"] = len(dups)

    for date_col in detect_date_columns(df):
        bad = dates_incoherentes(df, date_col)
        if len(bad) > 0:
            rapport["anomalies"][f"dates_invraisemblables::{date_col}"] = len(bad)

    taux_manquants = df.isna().mean().round(3)
    rapport["taux_valeurs_manquantes"] = taux_manquants[taux_manquants > 0].to_dict()

    return rapport


def resoudre_table_ciblee(question: str, tables: dict, nom_par_defaut: str | None = None):
    """Determine quelle table (parmi plusieurs deposees) est visee par une
    question : si le nom d'une table est mentionne explicitement dans la
    question, elle est prioritaire ; sinon on retombe sur la table active
    par defaut (choisie par l'utilisateur dans l'interface).

    Renvoie un tuple (nom_de_la_table, dataframe) ou (None, None) si aucune
    table n'est disponible.
    """
    q = question.lower()
    for nom in tables:
        if nom.lower() in q:
            return nom, tables[nom]
    if nom_par_defaut is not None and nom_par_defaut in tables:
        return nom_par_defaut, tables[nom_par_defaut]
    return None, None


def _noms_compatibles_stata(df: pd.DataFrame) -> pd.DataFrame:
    """Adapte les noms de colonnes aux contraintes Stata (<= 32 caracteres,
    commence par une lettre/underscore, uniquement lettres/chiffres/underscore),
    sans modifier le DataFrame original."""
    df = df.copy()
    nouveaux_noms = {}
    vus = set()
    for c in df.columns:
        nom = re.sub(r"[^A-Za-z0-9_]", "_", str(c))
        if not nom or not (nom[0].isalpha() or nom[0] == "_"):
            nom = "v_" + nom
        nom = nom[:32]
        base, i = nom, 1
        while nom in vus:
            suffix = f"_{i}"
            nom = base[: 32 - len(suffix)] + suffix
            i += 1
        vus.add(nom)
        nouveaux_noms[c] = nom
    return df.rename(columns=nouveaux_noms)


def exporter(df: pd.DataFrame, format: str) -> bytes:
    """Convertit une table en bytes pretes a telecharger, dans le format demande.

    format : "csv", "xlsx" ou "dta" (Stata).
    """
    format = format.lower().lstrip(".")
    buffer = io.BytesIO()

    if format == "csv":
        buffer.write(df.to_csv(index=False).encode("utf-8-sig"))

    elif format in ("xlsx", "excel"):
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="donnees")

    elif format in ("dta", "stata"):
        df_stata = _noms_compatibles_stata(df)
        # Stata n'accepte pas les colonnes entierement vides de type object,
        # ni les booleens : on les convertit en texte pour eviter une erreur.
        for c in df_stata.columns:
            if df_stata[c].dtype == bool:
                df_stata[c] = df_stata[c].astype(str)
        df_stata.to_stata(buffer, write_index=False, version=118)

    else:
        raise ValueError(f"Format d'export non supporte : {format} (attendu : csv, xlsx, dta)")

    buffer.seek(0)
    return buffer.getvalue()
