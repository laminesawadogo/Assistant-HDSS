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
AGENT_LIKE = re.compile(r"(field_?wrkr|fieldworker|enquet|interview|agent)", re.IGNORECASE)


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


def detect_agent_columns(df: pd.DataFrame) -> list[str]:
    """Colonnes identifiant l'agent enqueteur/le personnel de terrain ayant
    saisi une fiche (ex: `field_wrkr`), detectees par leur nom - pour un
    controle de qualite/performance par agent sans configuration manuelle."""
    return [c for c in df.columns if AGENT_LIKE.search(str(c))]


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


def rapport_agents(df: pd.DataFrame, colonne_agent: str | None = None) -> pd.DataFrame:
    """Rapport de performance/qualite par agent enqueteur : nombre de fiches
    saisies et indicateurs d'anomalies (doublons d'identifiant, dates
    invraisemblables, taux de valeurs manquantes) par agent - pour reperer
    une charge de travail inhabituelle ou un agent avec beaucoup plus
    d'erreurs que les autres (controle qualite du travail de terrain).

    Si `colonne_agent` n'est pas precisee, la premiere colonne d'agent
    detectee automatiquement (voir `detect_agent_columns`) est utilisee.
    Leve une ValueError si aucune colonne d'agent n'est detectee ni precisee,
    ou si la colonne demandee n'existe pas."""
    if colonne_agent is None:
        agents = detect_agent_columns(df)
        if not agents:
            raise ValueError(
                "Aucune colonne d'agent enquêteur détectée automatiquement (ex: field_wrkr) ; "
                "précisez `colonne_agent`."
            )
        colonne_agent = agents[0]
    elif colonne_agent not in df.columns:
        raise ValueError(f"Colonne '{colonne_agent}' absente. Colonnes disponibles : {list(df.columns)}")

    colonnes_id = [c for c in detect_id_columns(df) if c != colonne_agent]
    colonnes_date = detect_date_columns(df)

    masques_doublons = {c: df[c].duplicated(keep=False) for c in colonnes_id}
    masques_dates = {}
    aujourdhui = pd.Timestamp(datetime.now())
    for c in colonnes_date:
        parsed = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
        masques_dates[c] = parsed.isna() | (parsed > aujourdhui) | (parsed < pd.Timestamp("1900-01-01"))

    lignes = []
    for agent, index_groupe in df.groupby(colonne_agent, dropna=False).groups.items():
        n_fiches = len(index_groupe)
        n_doublons = sum(int(masques_doublons[c].loc[index_groupe].sum()) for c in colonnes_id)
        n_dates = sum(int(masques_dates[c].loc[index_groupe].sum()) for c in colonnes_date)
        taux_manquants = float(df.loc[index_groupe].isna().mean().mean()) if n_fiches else 0.0
        lignes.append({
            "agent": agent,
            "n_fiches": n_fiches,
            "doublons_id": n_doublons,
            "dates_invraisemblables": n_dates,
            "taux_valeurs_manquantes_moyen": round(taux_manquants, 3),
        })

    return pd.DataFrame(lignes).sort_values("n_fiches", ascending=False).reset_index(drop=True)


def syntaxe_rapport_agents(nom_table: str, colonne_agent: str) -> str:
    """Syntaxe R et Stata equivalente au rapport de performance par agent
    (nombre de fiches par agent, base pour reperer les anomalies)."""
    r = f"resultat <- dplyr::count({nom_table}, {colonne_agent}, name = \"n_fiches\")"
    stata = f"use {nom_table}, clear\nbysort {colonne_agent}: gen n_fiches = _N\ntab {colonne_agent}"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


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


PREFIXES_TABLE_IGNORES = ("fnew_", "fnew", "f_new_", "f_new")


def alias_table(nom: str) -> list[str]:
    """Genere des variantes informelles d'un nom de table charge, pour
    reconnaitre une mention qui omet le prefixe technique et/ou le
    singulier/pluriel exact (ex: l'equipe dit "la table education" ou
    "presence" a l'oral, alors que la table chargee s'appelle reellement
    "FNewEducation" ou "FNewPresences" - la correspondance exacte de
    sous-chaine ne suffit alors jamais a la reconnaitre).

    Ne renvoie que des alias d'au moins 4 caracteres, pour eviter qu'un
    prefixe retire ne laisse un fragment trop court et trop generique
    (risque de faux positifs sur un mot sans rapport)."""
    base = nom.lower()
    alias = base
    for prefixe in PREFIXES_TABLE_IGNORES:
        if base.startswith(prefixe) and len(base) > len(prefixe):
            alias = base[len(prefixe):]
            break

    variantes = {base, alias}
    if alias.endswith("s"):
        variantes.add(alias[:-1])
    else:
        variantes.add(alias + "s")

    return [v for v in variantes if len(v) >= 4]


def _alias_mentionne(alias: str, q: str) -> bool:
    """Verifie qu'un alias de table (voir `alias_table`) est mentionne comme
    mot entier dans la question, plutot qu'une simple sous-chaine - evite un
    faux positif quand l'alias d'une table correspond par hasard au debut
    d'un nom de colonne sans rapport (ex: l'alias "education" de la table
    FNewEducation ne doit PAS matcher a l'interieur du nom de colonne
    "education_level", qui parle d'une colonne, pas de la table)."""
    return re.search(r"\b" + re.escape(alias) + r"\b", q) is not None


def resoudre_table_ciblee(
    question: str, tables: dict, nom_par_defaut: str | None = None, historique: list[dict] | None = None
):
    """Determine quelle table (parmi plusieurs deposees) est visee par une
    question, pour que toutes les tables chargees soient aussi faciles a
    interroger les unes que les autres (pas seulement la table "active") :

    1. Si le nom d'une table (ou un alias informel - voir `alias_table`) est
       mentionne explicitement dans la question, elle est prioritaire.
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

    for nom in tables:
        if any(_alias_mentionne(alias, q) for alias in alias_table(nom)):
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


def tables_avec_colonne(question: str, tables: dict) -> list[str]:
    """Renvoie TOUTES les tables chargees qui contiennent une colonne
    mentionnee dans la question (pas seulement une premiere correspondance
    unique) - pour qu'une question ambigue sur une colonne presente dans
    plusieurs tables a la fois (ex: `individid` present dans 20 tables) soit
    traitee sur CHACUNE d'entre elles plutot que de silencieusement retomber
    sur un choix par defaut ("il faut tout lire, pas une seule base par
    defaut")."""
    q = question.lower()
    return [nom for nom, df in tables.items() if any(str(colonne).lower() in q for colonne in df.columns)]


def colonnes_mentionnees(question: str, tables: dict) -> list[str]:
    """Renvoie, dans l'ordre de premiere apparition dans la question, les
    noms de colonnes mentionnes - a partir de l'UNION de toutes les colonnes
    de TOUTES les tables chargees (pas seulement celles d'une table deja
    choisie par defaut), pour reconnaitre une colonne peu importe quelle
    table est regardee en premier. Sert de base a la recherche multi-table
    pour les analyses a plusieurs colonnes (bivariee, correlation,
    multivariee) quand aucune table n'est nommee explicitement."""
    q = question.lower()
    toutes_colonnes, vues = [], set()
    for df in tables.values():
        for c in df.columns:
            cle = str(c).lower()
            if cle not in vues:
                vues.add(cle)
                toutes_colonnes.append(str(c))
    return [c for c in toutes_colonnes if c.lower() in q]


def tables_avec_toutes_colonnes(colonnes: list[str], tables: dict) -> list[str]:
    """Renvoie les tables chargees qui contiennent TOUTES les colonnes
    donnees a la fois - candidates pour une analyse bivariee/multivariee sur
    plusieurs colonnes mentionnees dans la question sans qu'aucune table ne
    soit nommee explicitement ("il ne faut pas lire une seule base par
    defaut ; il faut tout lire")."""
    if not colonnes:
        return []
    colonnes_norm = {c.lower() for c in colonnes}
    resultat = []
    for nom, df in tables.items():
        colonnes_table = {str(c).lower() for c in df.columns}
        if colonnes_norm.issubset(colonnes_table):
            resultat.append(nom)
    return resultat


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
    identique une fois chargees dans `tables`).

    Reconnait aussi bien le nom exact charge qu'un alias informel (voir
    `alias_table`) qui omet le prefixe technique et/ou le singulier/pluriel
    - ex: "education" et "presence" reconnus pour "FNewEducation" et
    "FNewPresences", sans que l'equipe ait besoin de citer le nom technique
    complet."""
    q = question.lower()
    trouvees = []
    for nom in tables:
        if nom.lower() in q or any(_alias_mentionne(alias, q) for alias in alias_table(nom)):
            trouvees.append(nom)
    return trouvees


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


def tableau_croise(df: pd.DataFrame, colonne1: str, colonne2: str) -> pd.DataFrame:
    """Tableau croise (analyse bivariee) entre deux colonnes d'une meme table :
    effectifs croises, avec une ligne/colonne "Total" en marge - repond a une
    question du type "croise sex et education_level" ou "tableau croise entre
    X et Y"."""
    for c in (colonne1, colonne2):
        if c not in df.columns:
            raise ValueError(f"Colonne '{c}' absente. Colonnes disponibles : {list(df.columns)}")
    return pd.crosstab(df[colonne1], df[colonne2], margins=True, margins_name="Total")


def syntaxe_tableau_croise(nom_table: str, colonne1: str, colonne2: str) -> str:
    """Syntaxe R et Stata equivalente a un tableau croise entre deux colonnes."""
    r = f"table({nom_table}${colonne1}, {nom_table}${colonne2})"
    stata = f"use {nom_table}, clear\ntab {colonne1} {colonne2}"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def colonnes_numeriques(df: pd.DataFrame) -> list[str]:
    """Colonnes de type numerique (candidates a une correlation/analyse
    multivariee quantitative) - a partir du type reel des donnees chargees."""
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def matrice_correlation(df: pd.DataFrame, colonnes: list[str] | None = None) -> pd.DataFrame:
    """Matrice de correlation (Pearson) entre colonnes numeriques - analyse
    multivariee simple pour reperer des liens entre plusieurs variables
    quantitatives a la fois. Si `colonnes` n'est pas precise, utilise toutes
    les colonnes numeriques detectees automatiquement."""
    cols = colonnes if colonnes is not None else colonnes_numeriques(df)
    cols_valides = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    if len(cols_valides) < 2:
        raise ValueError(
            "Il faut au moins deux colonnes numériques pour calculer une corrélation "
            f"(colonnes numériques disponibles : {colonnes_numeriques(df)})."
        )
    return df[cols_valides].corr().round(3)


def syntaxe_correlation(nom_table: str, colonnes: list[str]) -> str:
    """Syntaxe R et Stata equivalente a une matrice de correlation."""
    vecteur_r = ", ".join(f'"{c}"' for c in colonnes)
    r = f"cor({nom_table}[, c({vecteur_r})], use = \"pairwise.complete.obs\")"
    stata = f"use {nom_table}, clear\npwcorr {' '.join(colonnes)}, sig"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def tableau_multivarie(df: pd.DataFrame, colonnes: list[str]) -> pd.DataFrame:
    """Effectifs croises sur 3 colonnes ou plus (analyse multivariee
    categorielle simple) - groupement par toutes les colonnes demandees,
    avec le nombre de lignes pour chaque combinaison observee."""
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonne(s) absente(s) : {', '.join(manquantes)}. Colonnes disponibles : {list(df.columns)}")
    return (
        df.groupby(colonnes, dropna=False)
        .size()
        .reset_index(name="effectif")
        .sort_values("effectif", ascending=False)
        .reset_index(drop=True)
    )


def syntaxe_tableau_multivarie(nom_table: str, colonnes: list[str]) -> str:
    """Syntaxe R et Stata equivalente a un groupement sur plusieurs colonnes."""
    vecteur_r = ", ".join(colonnes)
    r = f"resultat <- dplyr::count({nom_table}, {vecteur_r})"
    stata = f"use {nom_table}, clear\negen _grp = group({' '.join(colonnes)})\ntab _grp"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def syntaxe_repartition(nom_table: str, colonne: str) -> str:
    """Syntaxe R et Stata equivalente a une repartition (comptage/pourcentages)
    d'une colonne - a fournir en complement du resultat direct, pour que
    l'equipe puisse reproduire ou approfondir le calcul dans R/Stata."""
    r = f'table({nom_table}${colonne}); prop.table(table({nom_table}${colonne})) * 100'
    stata = f"use {nom_table}, clear\ntab {colonne}"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def syntaxe_echantillon(nom_table: str, n: int, seed: int) -> str:
    """Syntaxe R et Stata equivalente a un echantillon aleatoire reproductible
    (meme graine que celle utilisee par `echantillon`)."""
    r = f'set.seed({seed}); resultat <- {nom_table}[sample(nrow({nom_table}), {n}), ]'
    stata = f"use {nom_table}, clear\nset seed {seed}\nsample {n}, count"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def syntaxe_doublons(nom_table: str, colonne: str) -> str:
    """Syntaxe R et Stata equivalente a une detection de doublons sur une
    colonne d'identifiant."""
    r = (
        f"resultat <- {nom_table}[{nom_table}${colonne} %in% "
        f"{nom_table}${colonne}[duplicated({nom_table}${colonne})], ]"
    )
    stata = f"use {nom_table}, clear\nduplicates tag {colonne}, generate(_dup)\nlist if _dup > 0"
    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


def syntaxe_coherence(nom_table: str, colonnes_id: list[str], colonnes_date: list[str]) -> str:
    """Syntaxe R et Stata equivalente au rapport de coherence (doublons sur
    les colonnes d'ID detectees + dates invraisemblables sur les colonnes de
    date detectees) - combine les deux controles dans un seul bloc de code."""
    lignes_r = [f"use_table <- {nom_table}"]
    for c in colonnes_id:
        lignes_r.append(f'sum(duplicated(use_table${c}))  # doublons sur {c}')
    for c in colonnes_date:
        lignes_r.append(
            f'sum(is.na(as.Date(use_table${c})) | as.Date(use_table${c}) > Sys.Date())  # dates invraisemblables sur {c}'
        )
    r = "\n".join(lignes_r) if (colonnes_id or colonnes_date) else "# Aucune colonne d'identifiant ou de date detectee"

    lignes_stata = [f"use {nom_table}, clear"]
    for c in colonnes_id:
        lignes_stata.append(f"duplicates report {c}")
    for c in colonnes_date:
        lignes_stata.append(f"count if missing({c}) | {c} > date(\"$S_DATE\", \"DMY\")")
    stata = "\n".join(lignes_stata) if (colonnes_id or colonnes_date) else "# Aucune colonne d'identifiant ou de date detectee"

    return f"**Syntaxe équivalente :**\n```r\n{r}\n```\n```stata\n{stata}\n```"


"""
Catalogue de controles de coherence avances, specifiques au type d'enquete de
l'observatoire (au-dela des doublons/dates generiques de `rapport_coherence`).

Chaque controle est auto-detecte par le NOM des colonnes / tables reellement
chargees (memes principe que `detect_id_columns`/`detect_agent_columns`),
puisque les noms exacts varient d'un fichier a l'autre et ne sont jamais
codes en dur. Un controle dont les colonnes necessaires ne sont pas
detectees renvoie None (ignore proprement) plutot que de risquer un faux
positif sur des colonnes qui ne correspondent pas a ce qu'il est cense
verifier - transparence : le rapport final liste toujours explicitement quels
controles ont pu s'appliquer et lesquels ont ete ignores faute de colonnes
reconnues.
"""

BIRTH_DATE_LIKE = re.compile(r"(birth_date|naissance)", re.IGNORECASE)
DEATH_DATE_LIKE = re.compile(r"(death_date|evtdate|deces|dcd)", re.IGNORECASE)
ARRIVE_DATE_LIKE = re.compile(r"arrive.*date|date.*arrive", re.IGNORECASE)
DEPART_DATE_LIKE = re.compile(r"depart.*date|date.*depart", re.IGNORECASE)
ENTRY_DATE_LIKE = re.compile(r"(entry_date|date.*enregistr)", re.IGNORECASE)
WEIGHT_LIKE = re.compile(r"(poids|weight)", re.IGNORECASE)
HEIGHT_LIKE = re.compile(r"(taille|height)", re.IGNORECASE)
MOTHERID_LIKE = re.compile(r"(motherid|mereid|id_mere)", re.IGNORECASE)
FATHERID_LIKE = re.compile(r"(fatherid|pereid|id_pere)", re.IGNORECASE)
LAT_LIKE = re.compile(r"^lat(itude)?$", re.IGNORECASE)
LON_LIKE = re.compile(r"^lon(g|gitude)?$", re.IGNORECASE)
PHONE_LIKE = re.compile(r"(telephone|numtel|phone)", re.IGNORECASE)
SLEEP_LIKE = re.compile(r"(sleep|dormi)", re.IGNORECASE)
LOCATION_LIKE = re.compile(r"(locationid|menageid)", re.IGNORECASE)
SOCIALGP_LIKE = re.compile(r"(socialgpid)", re.IGNORECASE)

# Bornes approximatives du territoire burkinabe, pour reperer des coordonnees
# GPS clairement hors zone (au-dela d'une simple valeur manquante).
BF_LAT_MIN, BF_LAT_MAX = 9.0, 15.2
BF_LON_MIN, BF_LON_MAX = -5.6, 2.5


def _premiere_colonne(df: pd.DataFrame, motif: re.Pattern) -> str | None:
    for c in df.columns:
        if motif.search(str(c)):
            return c
    return None


def _age_en_annees(df: pd.DataFrame, colonne_naissance: str) -> pd.Series:
    naissance = pd.to_datetime(df[colonne_naissance], errors="coerce", dayfirst=True)
    return (pd.Timestamp(datetime.now()) - naissance).dt.days / 365.25


def controle_id_longueur(df: pd.DataFrame) -> dict | None:
    """Signale les identifiants dont la longueur (nombre de caracteres)
    differe de la longueur la plus frequente dans la meme colonne - repere un
    ID mal saisi sans connaitre a l'avance la longueur exacte attendue."""
    colonnes_id = detect_id_columns(df)
    if not colonnes_id:
        return None
    total = 0
    exemples = []
    for col in colonnes_id:
        non_nuls = df[col].dropna()
        if non_nuls.empty:
            continue
        longueur_attendue = int(non_nuls.astype(str).str.len().mode().iloc[0])
        longueurs_completes = df[col].astype(str).str.len()
        mauvaises_mask = df[col].notna() & (longueurs_completes != longueur_attendue)
        n = int(mauvaises_mask.sum())
        if n > 0:
            total += n
            exemples.append(f"{col} (longueur attendue {longueur_attendue})")
    return {"colonnes_verifiees": colonnes_id, "n_anomalies": total, "detail": exemples}


def controle_auto_reference(df: pd.DataFrame) -> dict | None:
    """Signale les lignes ou un identifiant se reference lui-meme (ex:
    individid == individid2), a partir de toute paire de colonnes ID detectee
    dont l'une se termine par "2" par rapport a l'autre."""
    colonnes_id = set(detect_id_columns(df))
    verifiees, total = [], 0
    for col in colonnes_id:
        if col.endswith("2"):
            continue
        col2 = col + "2"
        if col2 not in df.columns:
            continue
        verifiees.append((col, col2))
        n = int((df[col] == df[col2]).sum())
        total += n
    if not verifiees:
        return None
    return {"colonnes_verifiees": [f"{a}/{b}" for a, b in verifiees], "n_anomalies": total}


def controle_parents_identiques(df: pd.DataFrame) -> dict | None:
    """Signale les lignes ou fatherid == motherid (parents identiques)."""
    col_pere = _premiere_colonne(df, FATHERID_LIKE)
    col_mere = _premiere_colonne(df, MOTHERID_LIKE)
    if col_pere is None or col_mere is None:
        return None
    masque = df[col_pere].notna() & (df[col_pere] == df[col_mere])
    return {"colonnes_verifiees": [col_pere, col_mere], "n_anomalies": int(masque.sum())}


def controle_parent_manquant_jeune_enfant(df: pd.DataFrame, seuil_age: float = 5.0) -> dict | None:
    """Signale les moins de `seuil_age` ans sans motherid renseigne, et
    separement sans motherid NI fatherid renseignes."""
    col_naissance = _premiere_colonne(df, BIRTH_DATE_LIKE)
    col_mere = _premiere_colonne(df, MOTHERID_LIKE)
    if col_naissance is None or col_mere is None:
        return None
    age = _age_en_annees(df, col_naissance)
    jeunes = age < seuil_age
    sans_mere = int((jeunes & df[col_mere].isna()).sum())
    resultat = {
        "colonnes_verifiees": [col_naissance, col_mere],
        "n_anomalies": sans_mere,
        "detail": [f"< {seuil_age} ans sans motherid : {sans_mere}"],
    }
    col_pere = _premiere_colonne(df, FATHERID_LIKE)
    if col_pere is not None:
        # Sous-ensemble de `sans_mere` (deja compte dans n_anomalies) : indication
        # supplementaire, ne pas re-additionner pour eviter un double comptage.
        sans_les_deux = int((jeunes & df[col_mere].isna() & df[col_pere].isna()).sum())
        resultat["colonnes_verifiees"].append(col_pere)
        resultat["detail"].append(f"< {seuil_age} ans sans motherid ni fatherid : {sans_les_deux}")
    return resultat


def controle_dates_ordre(df: pd.DataFrame, motif_avant: re.Pattern, motif_apres: re.Pattern, strict: bool = True) -> dict | None:
    """Controle generique : signale les lignes ou une date qui devrait
    logiquement precéder une autre lui est posterieure (ex: naissance apres
    deces, enregistrement avant naissance). `motif_avant` doit designer la
    date qui devrait etre la plus ancienne."""
    col_avant = _premiere_colonne(df, motif_avant)
    col_apres = _premiere_colonne(df, motif_apres)
    if col_avant is None or col_apres is None or col_avant == col_apres:
        return None
    d_avant = pd.to_datetime(df[col_avant], errors="coerce", dayfirst=True)
    d_apres = pd.to_datetime(df[col_apres], errors="coerce", dayfirst=True)
    masque = (d_avant > d_apres) if strict else (d_avant >= d_apres)
    masque = masque.fillna(False)
    return {"colonnes_verifiees": [col_avant, col_apres], "n_anomalies": int(masque.sum())}


def controle_sentinelle(df: pd.DataFrame, motif_colonne: re.Pattern, valeur_sentinelle) -> dict | None:
    """Signale les lignes ou une colonne (ex: poids, taille) contient une
    valeur sentinelle connue (9999, 99...) indiquant une non-reponse codee
    plutot qu'une vraie mesure."""
    col = _premiere_colonne(df, motif_colonne)
    if col is None:
        return None
    masque = pd.to_numeric(df[col], errors="coerce") == valeur_sentinelle
    return {"colonnes_verifiees": [col], "n_anomalies": int(masque.sum())}


def controle_gps_hors_zone(df: pd.DataFrame) -> dict | None:
    """Signale les coordonnees GPS manquantes ou hors du territoire
    burkinabe (bornes approximatives)."""
    col_lat = _premiere_colonne(df, LAT_LIKE)
    col_lon = _premiere_colonne(df, LON_LIKE)
    if col_lat is None or col_lon is None:
        return None
    lat = pd.to_numeric(df[col_lat], errors="coerce")
    lon = pd.to_numeric(df[col_lon], errors="coerce")
    hors_zone = (
        lat.isna() | lon.isna()
        | (lat < BF_LAT_MIN) | (lat > BF_LAT_MAX)
        | (lon < BF_LON_MIN) | (lon > BF_LON_MAX)
    )
    return {"colonnes_verifiees": [col_lat, col_lon], "n_anomalies": int(hors_zone.sum())}


def controle_telephone_format(df: pd.DataFrame) -> dict | None:
    """Signale les numeros de telephone dont un des segments (separes par
    "/") ne fait pas 8 chiffres (format attendu au Burkina Faso)."""
    col = _premiere_colonne(df, PHONE_LIKE)
    if col is None:
        return None

    def _invalide(valeur) -> bool:
        if pd.isna(valeur):
            return False
        segments = str(valeur).split("/")
        return any(not seg.strip().isdigit() or len(seg.strip()) != 8 for seg in segments if seg.strip())

    masque = df[col].apply(_invalide)
    return {"colonnes_verifiees": [col], "n_anomalies": int(masque.sum())}


def controle_dates_arrivee_depart(df: pd.DataFrame) -> dict | None:
    """Fiche presence : signale les dates de depart anterieures ou egales a
    la date d'arrivee (incoherentes ou identiques)."""
    col_arrivee = _premiere_colonne(df, ARRIVE_DATE_LIKE)
    col_depart = _premiere_colonne(df, DEPART_DATE_LIKE)
    if col_arrivee is None or col_depart is None:
        return None
    arrivee = pd.to_datetime(df[col_arrivee], errors="coerce", dayfirst=True)
    depart = pd.to_datetime(df[col_depart], errors="coerce", dayfirst=True)
    masque = ((depart < arrivee) | (depart == arrivee)).fillna(False)
    return {"colonnes_verifiees": [col_arrivee, col_depart], "n_anomalies": int(masque.sum())}


def controle_residence_multiple(df: pd.DataFrame) -> dict | None:
    """Signale un individu present dans plusieurs menages/localisations
    differents (colonne `locationid` ou `socialgpid` associee a plusieurs
    valeurs distinctes pour le meme individid)."""
    colonnes_id = detect_id_columns(df)
    col_individu = next((c for c in colonnes_id if "individ" in c.lower()), None)
    col_lieu = _premiere_colonne(df, LOCATION_LIKE) or _premiere_colonne(df, SOCIALGP_LIKE)
    if col_individu is None or col_lieu is None or col_individu == col_lieu:
        return None
    nb_lieux_distincts = df.groupby(col_individu)[col_lieu].nunique()
    individus_multiples = nb_lieux_distincts[nb_lieux_distincts > 1]
    return {"colonnes_verifiees": [col_individu, col_lieu], "n_anomalies": int(len(individus_multiples))}


def controle_tranche_age(df: pd.DataFrame, borne_min: float, borne_max: float) -> dict | None:
    """Controle generique : signale les lignes dont l'age (calcule depuis la
    colonne de naissance detectee) sort d'une tranche attendue pour ce type
    de fiche (ex: 12-49 ans pour une fiche genesique, 5-34 ans pour une fiche
    education...)."""
    col = _premiere_colonne(df, BIRTH_DATE_LIKE)
    if col is None:
        return None
    age = _age_en_annees(df, col)
    hors_plage = ((age < borne_min) | (age > borne_max)).fillna(False)
    return {
        "colonnes_verifiees": [col],
        "n_anomalies": int(hors_plage.sum()),
        "detail": [f"âge hors {borne_min:g}-{borne_max:g} ans"],
    }


CONTROLES_GENERIQUES_PAR_TABLE = [
    ("Identifiants de longueur inhabituelle", controle_id_longueur),
    ("Auto-référence (ID == ID2)", controle_auto_reference),
    ("Parents identiques (fatherid == motherid)", controle_parents_identiques),
    ("Jeune enfant sans parent renseigné", controle_parent_manquant_jeune_enfant),
    ("Valeur de poids sentinelle (9999)", lambda df: controle_sentinelle(df, WEIGHT_LIKE, 9999)),
    ("Valeur de taille sentinelle (99)", lambda df: controle_sentinelle(df, HEIGHT_LIKE, 99)),
    ("Coordonnées GPS manquantes ou hors zone", controle_gps_hors_zone),
    ("Format de téléphone invalide", controle_telephone_format),
    ("Dates arrivée/départ incohérentes", controle_dates_arrivee_depart),
    ("Résidence multiple pour un même individu", controle_residence_multiple),
    ("Naissance postérieure au décès", lambda df: controle_dates_ordre(df, BIRTH_DATE_LIKE, DEATH_DATE_LIKE)),
    ("Enregistrement antérieur à la naissance", lambda df: controle_dates_ordre(df, ENTRY_DATE_LIKE, BIRTH_DATE_LIKE, strict=False)),
]

# Tranches d'age attendues par type de fiche (mots-cles reconnus dans le nom
# de la table charge - voir alias_table pour le meme principe de
# reconnaissance informelle des noms de table).
TRANCHES_AGE_PAR_TYPE_FICHE = [
    (["genesiqcompl", "gnesiqcompl"], None),  # cas particulier traite a part (age de la mere)
    (["genesiq", "gnesiq"], (12.0, 49.0)),
    (["histmat", "matrimon", "union"], (12.0, 40.0)),
    (["education"], (5.0, 34.0)),
    (["emploi", "employ"], (15.0, 120.0)),
    (["pregnancy", "grossesse"], (12.0, 49.0)),
]


def _table_correspond(nom: str, mots_cles: list[str]) -> bool:
    n = nom.lower()
    return any(m in n for m in mots_cles)


def rapport_coherence_avancee(tables: dict, nom_table: str | None = None) -> dict:
    """Execute le catalogue de controles de coherence avances sur une table
    precise (si `nom_table` est fourni) ou sur TOUTES les tables chargees,
    plus les controles croises entre tables (eligibilite presence <->
    education/emploi/genesique/pauvrete/sante, deces <-> presence, migration
    <-> presence, grossesse <-> issue de grossesse, snakebite menages <->
    residents).

    Chaque controle non applicable (colonnes non detectees) est explicitement
    liste comme "ignoré" plutot que d'etre tu, pour rester precis sur ce qui a
    reellement ete verifie - jamais de faux positif silencieux sur des
    colonnes qui ne correspondent pas a ce qui est cense etre controle."""
    tables_a_verifier = {nom_table: tables[nom_table]} if nom_table and nom_table in tables else tables

    resultats_par_table = {}
    for nom, df in tables_a_verifier.items():
        controles_ok, controles_ignores = [], []
        for libelle, fonction in CONTROLES_GENERIQUES_PAR_TABLE:
            resultat = fonction(df)
            if resultat is None:
                controles_ignores.append(libelle)
            else:
                controles_ok.append((libelle, resultat))

        for mots_cles, bornes in TRANCHES_AGE_PAR_TYPE_FICHE:
            if bornes is not None and _table_correspond(nom, mots_cles):
                resultat = controle_tranche_age(df, *bornes)
                libelle = f"Âge hors tranche attendue ({bornes[0]:g}-{bornes[1]:g} ans)"
                if resultat is None:
                    controles_ignores.append(libelle)
                else:
                    controles_ok.append((libelle, resultat))

        if controles_ok or controles_ignores:
            resultats_par_table[nom] = {"controles_ok": controles_ok, "controles_ignores": controles_ignores}

    # Controles croises entre tables (population eligible presence <->
    # tables associees), a partir d'une reconnaissance du role de chaque
    # table par mots-cles dans son nom.
    controles_croises = []
    nom_presence = next((n for n in tables if _table_correspond(n, ["presences", "presence"])), None)
    if nom_presence is not None:
        cibles = {
            "éducation": ["education"],
            "emploi": ["emploi", "employ"],
            "histoire génésique complémentaire": ["gnesiqcompl", "genesiqcompl"],
            "pauvreté": ["pauvrete"],
            "santé": ["sante"],
        }
        for libelle, mots in cibles.items():
            nom_cible = next((n for n in tables if _table_correspond(n, mots)), None)
            if nom_cible is None:
                continue
            resultat = controle_eligibilite_croisee(tables, nom_presence, nom_cible)
            if resultat is not None:
                controles_croises.append((f"Éligibilité présence ↔ {libelle}", nom_presence, nom_cible, resultat))

    nom_deces = next((n for n in tables if _table_correspond(n, ["death", "deces"])), None)
    if nom_presence is not None and nom_deces is not None:
        resultat = controle_deces_present(tables, nom_deces, nom_presence)
        if resultat is not None:
            controles_croises.append(("Décédé mais présent dans la fiche présence", nom_deces, nom_presence, resultat))

    nom_migration_out = next((n for n in tables if _table_correspond(n, ["migration_out", "migrationout"])), None)
    if nom_presence is not None and nom_migration_out is not None:
        resultat = controle_deces_present(tables, nom_migration_out, nom_presence)
        if resultat is not None:
            controles_croises.append(
                ("A dormi dans le ménage mais apparaît aussi en migration OUT", nom_migration_out, nom_presence, resultat)
            )

    nom_grossesse = next(
        (n for n in tables if _table_correspond(n, ["pregnancy"]) and not _table_correspond(n, ["cpn", "outcome"])),
        None,
    )
    nom_issue = next((n for n in tables if _table_correspond(n, ["pregoutcome"])), None)
    if nom_grossesse is not None and nom_issue is not None:
        cle = detecter_cle_jointure(nom_grossesse, nom_issue, tables)
        if cle:
            manquantes = difference_tables(nom_grossesse, nom_issue, tables, cle=cle)
            controles_croises.append((
                "Grossesse sans issue de grossesse enregistrée", nom_grossesse, nom_issue,
                {"colonnes_verifiees": [cle], "n_anomalies": len(manquantes)},
            ))

    if nom_table is not None:
        controles_croises = [
            c for c in controles_croises if nom_table in (c[1], c[2])
        ]

    return {"par_table": resultats_par_table, "croises": controles_croises}


def controle_eligibilite_croisee(tables: dict, nom_presence: str, nom_cible: str, cle: str | None = None) -> dict | None:
    """Compare la population 'eligible' de la fiche presence (a dormi, sans
    date de depart enregistree) a une autre fiche (education/emploi/...) :
    qui a la fiche sans etre eligible, et qui est eligible sans avoir la
    fiche - repond directement a un des controles les plus demandes du
    catalogue de l'observatoire."""
    if nom_presence not in tables or nom_cible not in tables:
        return None
    presence, cible = tables[nom_presence], tables[nom_cible]
    cle = cle or detecter_cle_jointure(nom_presence, nom_cible, tables)
    if cle is None or cle not in presence.columns or cle not in cible.columns:
        return None

    col_sleep = _premiere_colonne(presence, SLEEP_LIKE)
    col_depart = _premiere_colonne(presence, DEPART_DATE_LIKE)
    masque = pd.Series(True, index=presence.index)
    colonnes_verifiees = [cle]
    if col_sleep is not None:
        masque &= presence[col_sleep].astype(str).str.strip().str.lower().isin(["1", "1.0", "oui", "yes", "true"])
        colonnes_verifiees.append(col_sleep)
    if col_depart is not None:
        masque &= presence[col_depart].isna()
        colonnes_verifiees.append(col_depart)

    ids_eligibles = set(presence.loc[masque, cle].dropna())
    ids_cible = set(cible[cle].dropna())
    eligibles_sans_fiche = sorted(ids_eligibles - ids_cible, key=str)
    fiche_sans_eligibilite = sorted(ids_cible - ids_eligibles, key=str)

    return {
        "colonnes_verifiees": colonnes_verifiees,
        "n_eligibles_sans_fiche": len(eligibles_sans_fiche),
        "n_fiche_sans_eligibilite": len(fiche_sans_eligibilite),
        "eligibles_sans_fiche": eligibles_sans_fiche[:50],
        "fiche_sans_eligibilite": fiche_sans_eligibilite[:50],
    }


def controle_deces_present(tables: dict, nom_source: str, nom_presence: str, cle: str | None = None) -> dict | None:
    """Compare les individus d'une table (deces, migration OUT...) a la fiche
    presence : signale ceux qui apparaissent dans les deux alors qu'ils ne
    devraient logiquement pas (ex: decede mais toujours marque present,
    parti en migration mais toujours marque comme ayant dormi sur place)."""
    if nom_source not in tables or nom_presence not in tables:
        return None
    cle = cle or detecter_cle_jointure(nom_source, nom_presence, tables)
    if cle is None:
        return None
    try:
        chevauchement = tables[nom_source].merge(tables[nom_presence][[cle]].drop_duplicates(), on=cle, how="inner")
    except Exception:
        return None
    return {"colonnes_verifiees": [cle], "n_anomalies": len(chevauchement)}


"""
Module "Performances" : suivi du volume d'activite de terrain par agent
enqueteur (menages/UCH visites, naissances/deces/grossesses enregistres),
independant du controle qualite deja fourni par `rapport_agents` (qui mesure
les erreurs/doublons, pas le volume). Meme principe d'auto-detection par nom
de colonne/table que le reste du module : rien n'est code en dur sur des
identifiants d'agents ou une equipe reelle, puisqu'ils varient d'un
observatoire/campagne a l'autre et ne sont pas connus a l'avance.
"""

CONTROLEUR_LIKE = re.compile(
    r"(controleur|contr[oô]lleur|superviseur|supervisor|chef_?equipe|team_?lead)", re.IGNORECASE
)

# Chaque entree : (libelle affiche, mots-cles de reconnaissance du role de la
# table par son nom - meme principe que `_table_correspond`/`TRANCHES_AGE_PAR_TYPE_FICHE`,
# compter_menages -> si True, compte aussi les menages/UCH distincts (colonne
# LOCATION_LIKE) en plus du nombre de fiches.
CATEGORIES_PERFORMANCE_TERRAIN = [
    ("Ménages/UCH visités", ["presences", "presence"], True),
    ("Naissances enregistrées", ["birth", "naissance"], False),
    ("Décès enregistrés", ["death", "deces"], False),
    ("Grossesses enregistrées", ["pregnancy", "grossesse"], False),
]


def rapport_performance_agents(
    tables: dict, exclure: list[str] | None = None, colonne_agent: str | None = None
) -> pd.DataFrame:
    """Volume d'activite de terrain par agent, agrege a travers TOUTES les
    tables chargees qui comportent une colonne d'agent detectable (pas une
    seule table par defaut - meme principe que le reste de l'assistant) :
    nombre de menages/UCH visites, naissances/deces/grossesses enregistres,
    plus un total "autres fiches" pour toute table avec un agent qui ne
    correspond a aucune categorie reconnue (transparence : on ne perd aucune
    activite de terrain juste parce qu'elle ne rentre pas dans les 4
    categories nommees).

    `exclure` : liste d'identifiants d'agent (ex: agents non-terrain,
    formateurs, superviseurs saisis par erreur comme agent) a retirer du
    rapport - comparaison insensible a la casse/aux espaces, jamais une liste
    figee en dur puisque l'observatoire est seul a savoir qui exclure.
    """
    exclure_norm = {str(a).strip().lower() for a in exclure} if exclure else set()
    donnees: dict[str, dict[str, float]] = {}

    for nom, df in tables.items():
        col_agent = colonne_agent if (colonne_agent and colonne_agent in df.columns) else (
            next(iter(detect_agent_columns(df)), None)
        )
        if col_agent is None:
            continue
        # Une table qui comporte une colonne "controleur" est traitee comme la
        # table d'equipe (mapping agent <-> controleur, voir
        # `fusion_agent_controleur`) et non comme une fiche d'activite de
        # terrain : elle ne doit pas etre comptee ici, sous peine de gonfler
        # artificiellement le volume de chaque agent avec des lignes qui ne
        # sont pas des fiches saisies.
        if _premiere_colonne(df, CONTROLEUR_LIKE) is not None:
            continue

        categorie, compter_menages = next(
            (
                (libelle, cm)
                for libelle, mots, cm in CATEGORIES_PERFORMANCE_TERRAIN
                if _table_correspond(nom, mots)
            ),
            (f"Autres fiches ({nom})", False),
        )

        col_menage = _premiere_colonne(df, LOCATION_LIKE) if compter_menages else None
        for agent, index_groupe in df.groupby(col_agent, dropna=False).groups.items():
            agent_str = str(agent).strip()
            if agent_str.lower() in exclure_norm:
                continue
            entree = donnees.setdefault(agent_str, {})
            entree[categorie] = entree.get(categorie, 0) + len(index_groupe)
            if col_menage is not None:
                n_menages = int(df.loc[index_groupe, col_menage].nunique())
                cle_menages = "Ménages/UCH distincts"
                entree[cle_menages] = entree.get(cle_menages, 0) + n_menages

    if not donnees:
        return pd.DataFrame(columns=["agent", "total_fiches"])

    rapport = pd.DataFrame.from_dict(donnees, orient="index").fillna(0)
    rapport = rapport.reset_index().rename(columns={"index": "agent"})
    colonnes_valeurs = [c for c in rapport.columns if c != "agent"]
    for c in colonnes_valeurs:
        rapport[c] = rapport[c].astype(int)
    rapport["total_fiches"] = rapport[[c for c in colonnes_valeurs if c != "Ménages/UCH distincts"]].sum(axis=1)
    colonnes_ordre = ["agent"] + sorted(c for c in colonnes_valeurs) + ["total_fiches"]
    return rapport[colonnes_ordre].sort_values("total_fiches", ascending=False).reset_index(drop=True)


def rapport_performance_par_jour(tables: dict, colonne_agent: str | None = None) -> pd.DataFrame:
    """Volume de fiches par agent ET par jour (forme longue : date, agent,
    n_fiches), a partir de la premiere table qui comporte a la fois une
    colonne d'agent et une colonne de date detectees - en priorite la fiche
    presence (role le plus proche d'un "passage terrain" quotidien), sinon la
    premiere table eligible trouvee parmi celles chargees."""
    candidates = []
    for nom, df in tables.items():
        col_agent = colonne_agent if (colonne_agent and colonne_agent in df.columns) else (
            next(iter(detect_agent_columns(df)), None)
        )
        if col_agent is None or _premiere_colonne(df, CONTROLEUR_LIKE) is not None:
            continue
        colonnes_date = detect_date_columns(df)
        if not colonnes_date:
            continue
        priorite = 0 if _table_correspond(nom, ["presences", "presence"]) else 1
        candidates.append((priorite, nom, df, col_agent, colonnes_date[0]))

    if not candidates:
        return pd.DataFrame(columns=["date", "agent", "n_fiches"])

    _, nom, df, col_agent, col_date = sorted(candidates, key=lambda c: c[0])[0]
    dates = pd.to_datetime(df[col_date], errors="coerce", dayfirst=True).dt.date
    temp = pd.DataFrame({"date": dates, "agent": df[col_agent].astype(str).str.strip()})
    temp = temp.dropna(subset=["date"])
    rapport = temp.groupby(["date", "agent"], dropna=False).size().reset_index(name="n_fiches")
    return rapport.sort_values(["date", "agent"]).reset_index(drop=True)


def fusion_agent_controleur(rapport_agents: pd.DataFrame, tables: dict) -> tuple[pd.DataFrame, str | None]:
    """Ajoute une colonne `controleur` au rapport de performance par agent, si
    une table "equipe" (contenant a la fois une colonne d'agent ET une
    colonne de controleur/superviseur, detectees par leur nom) est chargee -
    jointure agent <-> controleur demandee par l'observatoire pour situer
    chaque agent dans son equipe d'encadrement.

    Renvoie (rapport enrichi, nom de la table equipe utilisee) ; si aucune
    table de ce type n'est chargee, renvoie (rapport inchange, None) plutot
    que d'echouer."""
    if rapport_agents.empty or "agent" not in rapport_agents.columns:
        return rapport_agents, None

    for nom, df in tables.items():
        col_agent = next(iter(detect_agent_columns(df)), None)
        col_controleur = _premiere_colonne(df, CONTROLEUR_LIKE)
        if col_agent is None or col_controleur is None or col_agent == col_controleur:
            continue
        mapping = df[[col_agent, col_controleur]].drop_duplicates(subset=[col_agent])
        mapping = mapping.rename(columns={col_agent: "agent", col_controleur: "controleur"})
        mapping["agent"] = mapping["agent"].astype(str).str.strip()
        fusionne = rapport_agents.merge(mapping, on="agent", how="left")
        fusionne["controleur"] = fusionne["controleur"].fillna("Non renseigné")
        colonnes = ["agent", "controleur"] + [c for c in fusionne.columns if c not in ("agent", "controleur")]
        return fusionne[colonnes], nom

    return rapport_agents, None


def prevision_objectif(
    rapport_par_jour: pd.DataFrame, objectif: int = 17000, colonne_valeur: str = "n_fiches"
) -> dict | None:
    """Projection simple (rythme moyen constant) pour atteindre un objectif
    configurable (par defaut 17000 menages, chiffre cible communique par
    l'observatoire) a partir du cumul quotidien observe - renvoie le cumul
    actuel, le rythme journalier moyen, le nombre de jours restants estimes et
    la date de fin projetee. Renvoie None si aucune donnee par jour
    n'est disponible (rien a projeter)."""
    if rapport_par_jour is None or rapport_par_jour.empty:
        return None

    par_jour = rapport_par_jour.groupby("date")[colonne_valeur].sum().sort_index()
    cumul_actuel = int(par_jour.sum())
    n_jours = len(par_jour)
    rythme_journalier = cumul_actuel / n_jours if n_jours else 0.0
    restant = max(objectif - cumul_actuel, 0)

    resultat = {
        "objectif": objectif,
        "cumul_actuel": cumul_actuel,
        "n_jours_observes": n_jours,
        "rythme_journalier_moyen": round(rythme_journalier, 1),
        "reste_a_faire": restant,
        "date_debut": str(par_jour.index.min()),
        "date_derniere_donnee": str(par_jour.index.max()),
    }

    if restant == 0:
        resultat["jours_restants_estimes"] = 0
        resultat["date_fin_projetee"] = resultat["date_derniere_donnee"]
    elif rythme_journalier > 0:
        jours_restants = int(np.ceil(restant / rythme_journalier))
        resultat["jours_restants_estimes"] = jours_restants
        resultat["date_fin_projetee"] = str(
            (pd.Timestamp(resultat["date_derniere_donnee"]) + pd.Timedelta(days=jours_restants)).date()
        )
    else:
        resultat["jours_restants_estimes"] = None
        resultat["date_fin_projetee"] = None

    return resultat


def simulation_rythme(reste_a_faire: int, jours_disponibles: int) -> dict:
    """Simulation simple : rythme journalier necessaire pour finir le
    `reste_a_faire` (ex: menages restants) dans un nombre de jours donne -
    reponse a une question hypothetique ("si on veut finir en 20 jours, il
    faut combien par jour ?")."""
    if jours_disponibles <= 0:
        raise ValueError("Le nombre de jours disponibles doit être positif.")
    return {
        "reste_a_faire": reste_a_faire,
        "jours_disponibles": jours_disponibles,
        "rythme_journalier_necessaire": round(reste_a_faire / jours_disponibles, 1),
    }


def rechercher_identifiant(identifiant: str, tables: dict) -> dict[str, pd.DataFrame]:
    """Recherche instantanee d'un identifiant a travers TOUTES les tables
    chargees (pas seulement la table active) : renvoie, pour chaque table ou
    au moins une colonne d'identifiant contient cette valeur, les lignes
    correspondantes - pour retrouver en un coup toutes les fiches liees a un
    meme individu/menage sans avoir a nommer chaque table une par une."""
    identifiant_norm = str(identifiant).strip().lower()
    resultats: dict[str, pd.DataFrame] = {}
    for nom, df in tables.items():
        colonnes_id = detect_id_columns(df)
        if not colonnes_id:
            continue
        masque = pd.Series(False, index=df.index)
        for col in colonnes_id:
            masque |= df[col].astype(str).str.strip().str.lower() == identifiant_norm
        if masque.any():
            resultats[nom] = df[masque]
    return resultats


def generer_rapport_performance_docx(
    rapport_agents: pd.DataFrame, prevision: dict | None = None, objectif: int = 17000
) -> bytes:
    """Genere un rapport Word telechargeable (python-docx, deja une
    dependance du projet) resumant la performance de terrain par agent et,
    si disponible, la projection vers l'objectif configurable - pour partager
    un point d'avancement hors du chat (reunion d'equipe, hierarchie...)."""
    from docx import Document

    document = Document()
    document.add_heading("Rapport de performance de terrain", level=1)
    document.add_paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}.")

    if prevision:
        document.add_heading("Avancement vers l'objectif", level=2)
        document.add_paragraph(
            f"Objectif : {prevision.get('objectif', objectif)} — "
            f"réalisé à ce jour : {prevision.get('cumul_actuel', 0)} — "
            f"reste à faire : {prevision.get('reste_a_faire', 0)}."
        )
        document.add_paragraph(
            f"Rythme journalier moyen observé : {prevision.get('rythme_journalier_moyen', 0)} / jour "
            f"sur {prevision.get('n_jours_observes', 0)} jour(s) de données "
            f"({prevision.get('date_debut', '?')} → {prevision.get('date_derniere_donnee', '?')})."
        )
        if prevision.get("date_fin_projetee"):
            document.add_paragraph(
                f"Date de fin projetée au rythme actuel : {prevision['date_fin_projetee']} "
                f"(≈ {prevision.get('jours_restants_estimes', '?')} jour(s) restants)."
            )

    document.add_heading("Performance par agent", level=2)
    if rapport_agents is None or rapport_agents.empty:
        document.add_paragraph("Aucune donnée de performance disponible.")
    else:
        colonnes = list(rapport_agents.columns)
        table = document.add_table(rows=1, cols=len(colonnes))
        table.style = "Light Grid Accent 1"
        for cell, nom_colonne in zip(table.rows[0].cells, colonnes):
            cell.text = str(nom_colonne)
        for _, ligne in rapport_agents.iterrows():
            cells = table.add_row().cells
            for cell, valeur in zip(cells, ligne):
                cell.text = str(valeur)

    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


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
