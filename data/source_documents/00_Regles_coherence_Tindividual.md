# Règles de cohérence — Table Tindividual (pilote OPO)

Table source : `Tindividual` — identification individuelle de base (7 variables, aucune donnée nominative : pas de nom/prénom dans cette table).

Variables : `individid`, `sex` (1=Homme, 2=Femme), `ethnic_gp` (codes 1–13), `birth_date`, `fatherid`, `motherid`, `entry_date`.

## Règles de contrôle proposées

1. **Unicité de l'identifiant** — `individid` ne doit apparaître qu'une seule fois dans la table. Tout doublon est un signal d'erreur de saisie ou de double enregistrement.

2. **Modalités hors nomenclature** — `sex` doit être 1 ou 2 ; `ethnic_gp` doit être compris entre 1 et 13. Toute autre valeur (hors codes de "manquant" documentés) est à signaler.

3. **Dates invraisemblables** — `birth_date` ne doit être ni manquante, ni postérieure à aujourd'hui, ni antérieure à 1900. `entry_date` (date d'interview) ne peut pas précéder `birth_date`.

4. **Auto-référencement** — un individu ne peut pas être son propre père ou sa propre mère (`fatherid` ou `motherid` = `individid`).

5. **Filiation orpheline** — si `fatherid` ou `motherid` est renseigné, l'identifiant doit exister ailleurs dans la table (l'individu référencé doit être enregistré). Un `fatherid`/`motherid` sans correspondance est une incohérence à vérifier (individu hors concession, ou erreur de saisie).

6. **Cohérence du sexe des parents** — l'individu désigné par `fatherid` doit avoir `sex = 1` (Homme) ; celui désigné par `motherid` doit avoir `sex = 2` (Femme).

7. **Cohérence d'âge parent/enfant** — la date de naissance du père/de la mère doit précéder celle de l'enfant d'au moins ~12 ans (seuil indicatif, à ajuster). Un écart plus faible signale une inversion ou erreur d'identifiant.

## Traitement des anomalies détectées

Chaque contrôle produit une liste d'`individid` suspects, jamais une correction automatique. Cette liste est ce que l'assistant "Consultation" peut produire pour toute l'équipe. La validation et l'application des corrections restent réservées au projet "Correction", avec un journal (ID, champ, ancienne valeur, nouvelle valeur, justification, auteur, date).
