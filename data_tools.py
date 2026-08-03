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
    chemin = str(path).lower()
    if chemin.endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    elif chemin.endswith(".dta"):
        df = pd.read_stata(path)
    else:
        df = pd.read_csv(path)
    return strip_names(df)


def est_classeur_excel(path: str) -> bool:
    return str(path).lower().endswith((".xlsx", ".xls"))


def charger_classeur(path: str) -> dict[str, pd.DataFrame]:
    """Charge TOUTES les feuilles d'un classeur Excel comme autant de tables
    distinctes (au lieu de ne lire que la premiere feuille), pour reconnaitre
    directement un classeur qui contient plusieurs tables (ex: une feuille
    par table de l'observatoire dans un seul fichier).

    Renvoie un dict {nom_de_feuille: dataframe}, en ignorant les feuilles
    vides et en retirant les colonnes nominatives de chaque feuille (memes
    garde-fous que load_table)."""
    classeur = pd.ExcelFile(path)
    tables = {}
    for nom_feuille in classeur.sheet_names:
        df = classeur.parse(nom_feuille)
        if df.empty or len(df.columns) == 0:
            continue
        tables[nom_feuille] = strip_names(df)
    return tables


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
    habilitee.

    Le rapport indique explicitement quelles colonnes ont ete examinees
    (`colonnes_id_verifiees`, `colonnes_date_verifiees`), pour que la reponse
    reste precise sur ce qui a ete effectivement controle - plutot qu'un
    simple "aucune anomalie" qui pourrait laisser croire a un controle
    exhaustif alors qu'aucune colonne pertinente n'aurait ete detectee."""
    colonnes_id = detect_id_columns(df)
    colonnes_date = detect_date_columns(df)

    rapport = {
        "n_lignes": len(df),
        "colonnes": list(df.columns),
        "colonnes_id_verifiees": colonnes_id,
        "colonnes_date_verifiees": colonnes_date,
        "anomalies": {},
    }

    for id_col in colonnes_id:
        dups = doublons(df, id_col)
        if len(dups) > 0:
            rapport["anomalies"][f"doublons::{id_col}"] = len(dups)

    for date_col in colonnes_date:
        bad = dates_incoherentes(df, date_col)
        if len(bad) > 0:
            rapport["anomalies"][f"dates_invraisemblables::{date_col}"] = len(bad)

    taux_manquants = df.isna().mean().round(3)
    rapport["taux_valeurs_manquantes"] = taux_manquants[taux_manquants > 0].to_dict()

    return rapport


def tables_mentionnees_dans_historique(historique: list[dict] | None, tables: dict) -> list[str]:
    """Recherche, dans les derniers echanges de la conversation (du plus
    recent au plus ancien), les noms de tables chargees qui ont deja ete
    mentionnes - pour qu'une question de suivi qui ne renomme pas
    explicitement la table ("et les doublons ?" apres avoir parle de
    Tindividual) reste rattachee au bon contexte plutot que de retomber sur
    un choix par defaut potentiellement different.

    `historique` est une liste de {"role": ..., "contenu": ...} (voir
    app.py:historique_recent), du plus ancien au plus recent."""
    if not historique:
        return []
    trouvees: list[str] = []
    for message in reversed(historique):
        contenu = str(message.get("contenu", "")).lower()
        for nom in tables:
            if nom.lower() in contenu and nom not in trouvees:
                trouvees.append(nom)
    return trouvees


def resoudre_table_ciblee(
    question: str, tables: dict, nom_par_defaut: str | None = None, historique: list[dict] | None = None
):
    """Determine quelle table (parmi plusieurs deposees) est visee par une
    question, pour que toutes les tables chargees soient aussi faciles a
    interroger les unes que les autres (pas seulement la table "active") :

    1. Si le nom d'une table est mentionne explicitement dans la question,
       elle est prioritaire.
    2. Sinon, si un nom de colonne mentionne dans la question n'appartient
       qu'a une seule des tables chargees, cette table est retenue
       automatiquement (ex: "repartition de sex" cible directement la table
       qui contient la colonne "sex", sans avoir a la nommer ni a la
       selectionner comme table active).
    3. Sinon, si une table a ete mentionnee dans les echanges precedents de
       la conversation, elle est retenue (memoire conversationnelle - une
       question de suivi comme "et les doublons ?" reste rattachee au bon
       contexte).
    4. En dernier recours (aucun historique exploitable), on retombe sur la
       table active par defaut choisie dans l'interface.

    Renvoie un tuple (nom_de_la_table, dataframe) ou (None, None) si aucune
    table n'est disponible.
    """
    q = question.lower()

    for nom in tables:
        if nom.lower() in q:
            return nom, tables[nom]

    tables_correspondantes = []
    for nom, df in tables.items():
        if any(str(colonne).lower() in q for colonne in df.columns):
            tables_correspondantes.append(nom)
    if len(tables_correspondantes) == 1:
        nom = tables_correspondantes[0]
        return nom, tables[nom]

    for nom in tables_mentionnees_dans_historique(historique, tables):
        return nom, tables[nom]

    if nom_par_defaut is not None and nom_par_defaut in tables:
        return nom_par_defaut, tables[nom_par_defaut]
    return None, None


def resume_tables_chargees(tables: dict) -> str:
    """Decrit les tables actuellement chargees (nom, nombre de lignes et de
    colonnes) - reponse a une question "meta" sur la session en cours (ex:
    "combien de tables/feuilles sont chargees ?", "quelles tables sont
    disponibles ?"), qui ne porte pas sur le contenu d'une table mais sur ce
    qui est effectivement charge a l'instant. Ce n'est pas une question
    documentaire (le dictionnaire ne sait rien de la session en cours), donc
    elle doit etre traitee ici plutot que par le RAG."""
    if not tables:
        return "Aucune table n'est chargée pour l'instant."

    lignes = [f"**{len(tables)} table(s) chargée(s)** :"]
    for nom, df in tables.items():
        apercu_colonnes = ", ".join(f"`{c}`" for c in df.columns[:8])
        if len(df.columns) > 8:
            apercu_colonnes += ", ..."
        n_lignes = len(df)
        n_colonnes = len(df.columns)
        lignes.append(
            f"- **{nom}** : {n_lignes} ligne{'s' if n_lignes != 1 else ''}, "
            f"{n_colonnes} colonne{'s' if n_colonnes != 1 else ''} ({apercu_colonnes})"
        )
    return "\n".join(lignes)


def detecter_tables_mentionnees(question: str, tables: dict) -> list[str]:
    """Renvoie les noms des tables chargees explicitement mentionnees dans la
    question (peu importe qu'elles viennent de fichiers separes ou de
    plusieurs feuilles d'un meme classeur : les deux sont traitees de facon
    identique une fois chargees dans `tables`)."""
    q = question.lower()
    return [nom for nom in tables if nom.lower() in q]


def _colonnes_communes(df1: pd.DataFrame, df2: pd.DataFrame) -> list[str]:
    """Colonnes presentes dans les deux tables (comparaison insensible a la
    casse), dans l'ordre des colonnes de df1."""
    colonnes_b = {str(c).lower() for c in df2.columns}
    return [c for c in df1.columns if str(c).lower() in colonnes_b]


def detecter_cles_communes(tables: dict) -> dict[tuple[str, str], list[str]]:
    """Detecte, pour chaque paire de tables chargees, les colonnes presentes
    dans les deux (candidates comme cle de jointure) - a partir des vraies
    donnees chargees, pas seulement de la documentation du dictionnaire."""
    noms = list(tables.keys())
    resultat = {}
    for i in range(len(noms)):
        for j in range(i + 1, len(noms)):
            a, b = noms[i], noms[j]
            communes = _colonnes_communes(tables[a], tables[b])
            if communes:
                resultat[(a, b)] = communes
    return resultat


def relation_entre_tables(nom1: str, nom2: str, tables: dict) -> str:
    """Decrit, a partir des vraies donnees chargees, comment deux tables sont
    reliees (colonnes en commun, candidates comme cle de jointure)."""
    manquantes = [n for n in (nom1, nom2) if n not in tables]
    if manquantes:
        return f"Table introuvable parmi celles chargées : {', '.join(manquantes)}."

    communes = _colonnes_communes(tables[nom1], tables[nom2])
    if not communes:
        return (
            f"Aucune colonne commune détectée entre **{nom1}** et **{nom2}** dans les données "
            "chargées : impossible de déterminer un lien direct entre ces deux tables telles quelles."
        )

    lignes = [
        f"**{nom1}** et **{nom2}** partagent {len(communes)} colonne(s) : "
        + ", ".join(f"`{c}`" for c in communes) + "."
    ]
    lignes.append(
        f"`{communes[0]}` est la candidate la plus probable comme clé de jointure "
        f"(présente dans les deux tables). Demande « fusionne {nom1} et {nom2} » pour obtenir "
        "une table combinée."
    )
    return "\n".join(lignes)


def rapport_relations(tables: dict) -> str:
    """Resume les relations detectees (colonnes communes) entre toutes les
    paires de tables actuellement chargees (fichiers separes ou feuilles d'un
    meme classeur Excel : traites de la meme facon)."""
    if len(tables) < 2:
        return "Il faut au moins deux tables chargées pour détecter une relation entre elles."

    communes = detecter_cles_communes(tables)
    if not communes:
        return "Aucune colonne commune détectée entre les tables actuellement chargées."

    lignes = ["Relations détectées entre les tables chargées (colonnes en commun) :"]
    for (a, b), cols in communes.items():
        lignes.append(f"- **{a}** ↔ **{b}** : " + ", ".join(f"`{c}`" for c in cols))
    return "\n".join(lignes)


def fusionner_tables(nom1: str, nom2: str, tables: dict, cle: str | None = None) -> pd.DataFrame:
    """Fusionne (jointure) deux tables chargees sur une colonne commune.

    Si `cle` n'est pas precisee, la premiere colonne commune detectee entre
    les deux tables est utilisee automatiquement. Leve une ValueError si les
    tables sont introuvables, si aucune colonne commune n'existe, ou si la
    colonne demandee n'existe pas dans les deux tables."""
    manquantes = [n for n in (nom1, nom2) if n not in tables]
    if manquantes:
        raise ValueError(f"Table introuvable parmi celles chargées : {', '.join(manquantes)}")

    df1, df2 = tables[nom1], tables[nom2]

    if cle is None:
        communes = _colonnes_communes(df1, df2)
        if not communes:
            raise ValueError(f"Aucune colonne commune trouvée entre '{nom1}' et '{nom2}' pour fusionner.")
        cle = communes[0]
    elif cle not in df1.columns or cle not in df2.columns:
        raise ValueError(f"La colonne '{cle}' n'existe pas dans les deux tables.")

    return df1.merge(df2, on=cle, how="inner", suffixes=(f"_{nom1}", f"_{nom2}"))


def detecter_cle_jointure(nom1: str, nom2: str, tables: dict) -> str | None:
    """Renvoie la colonne commune utilisee (ou qui serait utilisee) comme cle
    de jointure entre deux tables chargees, ou None si aucune ne convient -
    utilise a la fois par le calcul reel (fusion/difference) et par la
    generation de syntaxe R/Stata, pour rester coherent sur la cle choisie."""
    if nom1 not in tables or nom2 not in tables:
        return None
    communes = _colonnes_communes(tables[nom1], tables[nom2])
    return communes[0] if communes else None


def difference_tables(nom1: str, nom2: str, tables: dict, cle: str | None = None) -> pd.DataFrame:
    """Lignes de la table `nom1` dont la cle n'a AUCUNE correspondance dans la
    table `nom2` (anti-jointure / difference d'ensembles) : repond a une
    question du type "qui est present dans X mais pas dans Y" (ex: individus
    enregistres sur une fiche mais absents d'une autre).

    Si `cle` n'est pas precisee, la premiere colonne commune detectee entre
    les deux tables est utilisee automatiquement (voir `detecter_cle_jointure`).
    Leve une ValueError si les tables sont introuvables, si aucune colonne
    commune n'existe, ou si la colonne demandee n'existe pas dans les deux
    tables."""
    manquantes = [n for n in (nom1, nom2) if n not in tables]
    if manquantes:
        raise ValueError(f"Table introuvable parmi celles chargées : {', '.join(manquantes)}")

    df1, df2 = tables[nom1], tables[nom2]

    if cle is None:
        cle = detecter_cle_jointure(nom1, nom2, tables)
        if cle is None:
            raise ValueError(f"Aucune colonne commune trouvée entre '{nom1}' et '{nom2}' pour comparer.")
    elif cle not in df1.columns or cle not in df2.columns:
        raise ValueError(f"La colonne '{cle}' n'existe pas dans les deux tables.")

    fusion = df1.merge(df2[[cle]].drop_duplicates(), on=cle, how="left", indicator=True)
    return fusion[fusion["_merge"] == "left_only"].drop(columns="_merge")


def syntaxe_fusion(nom1: str, nom2: str, cle: str) -> str:
    """Syntaxe R et Stata equivalente a une fusion (jointure) entre deux
    tables sur une cle commune, a fournir en complement du resultat direct."""
    r = f'resultat <- merge({nom1}, {nom2}, by = "{cle}")'
    stata = f"use {nom1}, clear\nmerge 1:1 {cle} using {nom2}"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def syntaxe_difference(nom1: str, nom2: str, cle: str) -> str:
    """Syntaxe R (dplyr::anti_join) et Stata equivalente a une difference
    d'ensembles (lignes de nom1 sans correspondance dans nom2)."""
    r = f'resultat <- dplyr::anti_join({nom1}, {nom2}, by = "{cle}")'
    stata = f"use {nom1}, clear\nmerge 1:1 {cle} using {nom2}\nkeep if _merge == 1"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


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
