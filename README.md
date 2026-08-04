# Assistant OPO — prototype RAG

Agent qui répond aux questions sur le dictionnaire de données de l'Observatoire
de Population de Ouagadougou (RAG), et calcule des indicateurs, échantillons
et contrôles de cohérence sur une ou plusieurs tables (CSV, Excel — y compris
un classeur multi-feuilles — ou Stata), quelles que soient les colonnes
réellement présentes dans le fichier. Quand plusieurs tables sont chargées,
l'assistant détecte les colonnes qu'elles ont en commun et peut répondre à
des questions de relation, ou les fusionner sur demande.

**Les tables sont chargées automatiquement depuis le dossier Google Drive de
l'observatoire** (`opo_db_exports`), où un export de chaque table est déposé
chaque jour — il n'y a plus de dépôt manuel de fichier dans l'application :
voir la section [Connexion automatique à Google Drive](#connexion-automatique-à-google-drive-opo_db_exports).

## Installation

```bash
pip install -r requirements.txt
```

## Configuration du LLM (une seule clé nécessaire, "en arrière-plan")

Pour que personne n'ait besoin de saisir de clé dans l'application : copie
`.env.exemple` en `.env` (même dossier que `app.py`), colle ta clé dedans,
enlève le `#` en début de ligne, enregistre, puis lance/relance
l'application. La clé est alors chargée automatiquement à chaque démarrage,
et l'application n'affiche même plus les champs de saisie manuelle.

```
# fichier .env
ANTHROPIC_API_KEY=sk-ant-...    # recommandé, rendu le plus proche de Claude
# GROQ_API_KEY=gsk-...          # alternative gratuite
```

Ne jamais partager ce fichier une fois rempli (secret, comme un mot de
passe) ; ne pas le committer sur un dépôt Git public.

Alternative (utile en production sur un VPS, cf. section Déploiement) :
définir la clé comme variable d'environnement système plutôt que dans un
fichier `.env` :

```bash
export ANTHROPIC_API_KEY="..."
# ou
export GROQ_API_KEY="..."        # gratuit sur console.groq.com
```

Sans aucune clé configurée (ni `.env`, ni variable d'environnement), l'app
fonctionne quand même en mode dégradé : elle affiche le contexte récupéré du
dictionnaire mais sans réponse rédigée par un LLM, et propose alors une
saisie manuelle temporaire dans la barre latérale — utile pour tester la
pertinence de la recherche documentaire avant de brancher un modèle.

## Comptes / authentification

Un écran de connexion (`auth.py`, `streamlit-authenticator`) bloque
maintenant tout accès à l'application (tables, chat, documents) tant qu'un
compte valide n'a pas été utilisé — il n'est plus possible d'accéder aux
données simplement en ayant le lien de l'appli. Les comptes et leur rôle
(`correction` : accès complet ; `consultation` : rôle enregistré mais pas
encore utilisé pour restreindre l'interface) sont définis dans
`auth_config.yaml`.

**Pour un vrai déploiement (recommandé)**, les identifiants sont configurés
via le gestionnaire de secrets de Streamlit Cloud (Settings → Secrets de
l'appli), au format TOML documenté en tête de `auth_config.yaml.exemple`.
Tant que la configuration par défaut (mots de passe/clé de cookie d'exemple)
est détectée, un avertissement s'affiche directement dans l'application pour
que ça ne passe pas inaperçu.

`auth_config.yaml` (comptes réels) n'est **pas** suivi par Git — seul
`auth_config.yaml.exemple` (valeurs factices, même principe que
`.env`/`.env.exemple`) est versionné. Pour un usage local : copie
`auth_config.yaml.exemple` en `auth_config.yaml`, remplace les comptes et la
clé de cookie par de vraies valeurs.

Le fichier temporaire créé lors de la synchronisation d'une table depuis
Google Drive (le temps de la lire avec pandas) est systématiquement supprimé
du disque du serveur juste après son chargement, y compris en cas d'erreur
de lecture — aucune donnée synchronisée ne reste sur le disque au-delà du
strict temps de traitement.

## Connexion automatique à Google Drive (`opo_db_exports`)

Les tables ne se déposent plus manuellement dans l'application : un
processus externe exporte chaque jour les tables (CSV, Excel ou Stata) dans
le dossier Google Drive partagé `opo_db_exports`. À l'ouverture de
l'application (et toutes les 15 minutes ensuite, ou immédiatement via le
bouton **🔄 Recharger depuis Google Drive maintenant** dans la barre
latérale), l'assistant se connecte à ce dossier et ne garde que le
**dernier export**, quelle que soit la façon dont il est organisé dans le
Drive — trois organisations sont reconnues automatiquement, sans réglage :

- des fichiers directement dans le dossier, un par table, avec la date
  d'export dans le nom (ex. `FNewIndividual_2026-08-04.csv`) ;
- un **sous-dossier par export** (organisation observée en pratique sur
  `opo_db_exports`, ex. `export_2026-08-03_09-15-00`, contenant un fichier
  par table pour ce jour-là) : seul le sous-dossier le plus récent est
  utilisé, les autres sont ignorés ;
- une **archive `.zip`** par export (à la racine ou dans un sous-dossier),
  contenant les fichiers de chaque table : elle est téléchargée puis
  extraite en mémoire (rien n'est jamais écrit sur disque au-delà du temps
  de traitement), sans qu'il soit nécessaire de la dézipper manuellement.

Ces fichiers sont ensuite chargés exactement comme un dépôt manuel l'aurait
fait : mêmes contrôles (colonnes nom/prénom retirées, aucune correction
automatique), mêmes analyses disponibles ensuite dans le chat.

**Le dossier étant restreint à des comptes précis** (pas un lien public),
l'accès se fait via un **compte de service Google** (un compte technique,
sans interface de connexion humaine), à créer une seule fois :

1. Sur [console.cloud.google.com](https://console.cloud.google.com), créer un
   projet (ou réutiliser un projet existant), puis dans **API et services →
   Bibliothèque**, activer l'**API Google Drive**.
2. Dans **API et services → Identifiants → Créer des identifiants → Compte de
   service**, créer un compte de service (nom libre, ex. `assistant-opo-drive`).
   Une fois créé, ouvrir l'onglet **Clés** du compte de service → **Ajouter
   une clé → Créer une clé → JSON** : un fichier `.json` se télécharge — c'est
   la clé secrète de ce compte, à garder confidentielle comme un mot de passe.
3. Repérer l'adresse e-mail du compte de service (visible sur sa page,
   de la forme `assistant-opo-drive@<projet>.iam.gserviceaccount.com`), puis,
   **dans Google Drive**, ouvrir le dossier `opo_db_exports` → **Partager**
   → coller cette adresse e-mail avec le rôle **Lecteur** (aucun besoin
   d'accès en écriture, l'assistant ne fait que lire les exports).
4. Configurer la clé récupérée à l'étape 2 :
   - **Pour un vrai déploiement (recommandé)**, dans **Settings → Secrets**
     de l'application Streamlit Cloud, coller le contenu du fichier JSON
     sous la forme :
     ```toml
     [gdrive]
     folder_id = "1qjV_hHhGIE5klnQYUzxLT-OJQp827v0l"

     [gdrive.service_account]
     type = "service_account"
     project_id = "..."
     private_key_id = "..."
     private_key = "..."
     client_email = "assistant-opo-drive@....iam.gserviceaccount.com"
     client_id = "..."
     auth_uri = "https://accounts.google.com/o/oauth2/auth"
     token_uri = "https://oauth2.googleapis.com/token"
     auth_provider_x509_cert_url = "https://www.googleapis.com/oauth2/v1/certs"
     client_x509_cert_url = "..."
     universe_domain = "googleapis.com"
     ```
     (reprendre telles quelles les valeurs du fichier JSON téléchargé pour
     chaque champ).
   - **Pour un usage local**, copier `service_account.exemple.json` en
     `service_account.json` (déjà ignoré par Git) et y coller le contenu du
     vrai fichier JSON téléchargé.
5. `folder_id` n'a normalement rien à changer : le dossier `opo_db_exports`
   partagé par l'équipe est déjà la valeur par défaut du code
   (`drive_sync.FOLDER_ID_PAR_DEFAUT`). À ne renseigner que si un autre
   dossier doit être utilisé un jour.

Tant que ce compte de service n'est pas configuré, l'application reste
utilisable pour les questions sur le dictionnaire (chemin RAG), mais affiche
un message clair indiquant qu'aucune table n'est disponible pour l'instant,
plutôt que de planter.

## Ajouter un document de référence (fiche, manuel, PDF, Word...)

Deux façons d'ajouter un document, au choix :

**Depuis l'application elle-même (le plus simple)** : section "📚 Ajouter un
document de référence" dans la barre latérale — dépose un PDF, un Word, un
PowerPoint ou un texte, il est converti et ajouté en permanence à la base de
connaissances en quelques secondes, sans quitter le navigateur.

**Depuis le dossier du projet** : dépose le fichier dans
`data/source_documents/`, puis relance :

```bash
python ingest.py
```

Cette commande convertit automatiquement tout nouveau fichier de
`data/source_documents/` en texte (`.doc`, `.docx`, `.pdf`, `.ppt`, `.pptx`,
`.xlsx`, `.txt`, `.md` sont tous gérés — voir `prepare_corpus.py`), puis reconstruit l'index de
recherche. Un fichier déjà converti n'est pas retraité tant qu'il n'a pas été
modifié (la conversion d'un PDF ou d'un `.doc` via LibreOffice prend quelques
secondes, inutile de la relancer à chaque fois). Le dictionnaire de variables
Excel (`Dictionnaire_donnees_OPO_BaseDemographique.xlsx`) continue d'être
traité à part par sa propre logique (un chunk par variable) — ne pas le
déposer dans `data/source_documents/`, il y est ignoré automatiquement.

Note technique : la conversion des `.doc`/`.ppt` (anciens formats Word/PowerPoint)
et des `.pdf` s'appuie sur des outils externes (LibreOffice / poppler
`pdftotext`), déjà présents sur la plupart des serveurs Linux (`apt install
libreoffice poppler-utils` sinon) ; les `.docx`/`.pptx` modernes sont lus
directement en Python sans dépendance externe.

## Construire l'index (une seule fois, ou après modification du dictionnaire)

```bash
python ingest.py
```

`data/docs/00_schema_relations.txt` documente les clés primaires/étrangères et
les relations entre les 23 tables (entités centrales, épisodes, événements,
historiques), généré à partir du dictionnaire et du document de correspondance
des tables. Il est indexé comme les autres documents, avec un léger boost de
pertinence (`rag.SOURCES_PRIORITAIRES`) pour rester compétitif face aux courtes
définitions de variables sur les questions transversales ("comment X est
reliée à Y").

## Lancer l'application

```bash
streamlit run app.py
```

## Lancer les tests

```bash
pytest tests/ -v
```

210 tests couvrent `ingest.py` (découpage en chunks), `prepare_corpus.py`
(conversion Word/PowerPoint/PDF/Excel/texte, non-reconversion si déjà à jour, exclusion du
dictionnaire xlsx principal), `data_tools.py` (répartitions, échantillon reproductible,
doublons, dates invraisemblables, suppression des colonnes nominatives, export
CSV/Excel/Stata, résolution de la table ciblée, tableaux croisés, corrélation,
rapport par agent enquêteur, catalogue de contrôles de cohérence avancés et
leurs contrôles croisés entre tables, module Performances de terrain —
agrégation par agent, jointure contrôleur, prévision d'objectif, rapport
Word, recherche par identifiant), `rag.py` (récupération documentaire,
garde-fous index/clé manquants, classification d'intention), `auth_config.yaml`
(structure des comptes, hash/vérification des mots de passe), `drive_sync.py`
(détection de la date d'export dans le nom du fichier, sélection du dernier
export par table, synchronisation avec un client Drive simulé — sans appel
réseau réel), et `app.py` via des tests bout-en-bout (`tests/test_app.py`, avec
`streamlit.testing.v1.AppTest`) qui simulent de vraies conversations
multi-tours dans l'interface.

## Structure

- `data/source_documents/` — dépôt des documents bruts (Word, PDF, Excel, texte) à indexer : c'est ici qu'on ajoute une nouvelle fiche ou un nouveau manuel.
- `data/docs/` — textes prêts à être découpés en chunks : dictionnaire de données (généré depuis le fichier Excel source, un fichier par table) + tout document converti automatiquement depuis `data/source_documents/` (préfixe `source_`).
- `prepare_corpus.py` — convertit les fichiers de `data/source_documents/` (`.doc`, `.docx`, `.pdf`, `.xlsx`, `.txt`, `.md`) en texte dans `data/docs/`.
- `ingest.py` — construit l'index de recherche (TF-IDF, léger, sans téléchargement de modèle) ; appelle `prepare_corpus.py` en premier.
- `rag.py` — recherche + construction du prompt + appel au LLM.
- `data_tools.py` — analyse d'une ou plusieurs tables (CSV, Excel — multi-feuilles compris — ou Stata) : répartitions, échantillon reproductible, détection de doublons/dates invraisemblables (colonnes ID/date détectées automatiquement par leur nom/contenu), résolution automatique de la table ciblée par une question (nom de table ou colonne mentionnée), détection des colonnes communes entre tables chargées (relations, candidates de jointure) et fusion sur demande. Les colonnes de type nom/prénom sont systématiquement retirées avant toute analyse.
- `drive_sync.py` — synchronisation automatique des tables depuis le dossier Google Drive `opo_db_exports` (authentification par compte de service, détection du dernier export par table, téléchargement). Voir la section [Connexion automatique à Google Drive](#connexion-automatique-à-google-drive-opo_db_exports).
- `app.py` — interface de chat (Streamlit).
- `instructions_systeme.md` — rôle et garde-fous de l'assistant (ce qu'il fait, ce qu'il ne fait jamais).
- `auth_config.yaml` — comptes réels de l'équipe (identifiants, mots de passe, rôles), non suivi par Git. `auth_config.yaml.exemple` est le modèle versionné (valeurs factices).
- `service_account.json` — clé du compte de service Google utilisé pour lire le dossier Drive, non suivi par Git. `service_account.exemple.json` est le modèle versionné (valeurs factices).
- `.env` / `.env.exemple` — clé du modèle (Anthropic/Groq) chargée automatiquement au démarrage, jamais à ressaisir dans l'application.
- `packages.txt` — dépendances système (LibreOffice, poppler-utils) installées automatiquement par Streamlit Community Cloud au déploiement.
- `tests/` — suite de tests automatisés (pytest).
- `Rapport_Technique_Assistant_OPO.pdf` — rapport technique : architecture, méthodologie, limites et perspectives.

## Déploiement sur Streamlit Community Cloud (donner un lien à l'équipe)

1. **Mettre le code sur GitHub.** Créer un dépôt (peut être privé) et y
   pousser tout le contenu du dossier `assistant_opo_rag/`. Le `.env` rempli
   avec la vraie clé ne doit **jamais** être poussé sur GitHub — ajouter un
   fichier `.gitignore` contenant la ligne `.env` avant le premier commit.
2. Aller sur **share.streamlit.io**, se connecter (avec un compte GitHub),
   cliquer sur **"New app"**, choisir le dépôt, la branche, et indiquer
   `app.py` comme fichier principal.
3. **Configurer la clé du modèle** : dans les paramètres de l'app déployée
   (**Settings → Secrets**), coller :
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
   L'application la récupère automatiquement (pont `st.secrets` déjà prévu
   dans `app.py`) — pas besoin de fichier `.env` sur Streamlit Cloud.
4. `packages.txt` (déjà présent à la racine du projet, contenant `libreoffice`
   et `poppler-utils`) est détecté automatiquement par Streamlit Cloud pour
   installer les outils système nécessaires à la conversion des `.doc`/`.ppt`/`.pdf`.
5. Une fois le déploiement terminé, Streamlit fournit une URL du type
   `https://<nom-app>.streamlit.app` — c'est ce lien à partager avec l'équipe.

**Accès restreint (recommandé vu la sensibilité des données).** Dans les
paramètres de l'app (**Settings → Sharing**), passer l'app en **"Private"** et
ajouter les emails de l'équipe à la liste des personnes autorisées (connexion
via Google requise pour eux à l'ouverture du lien). Sans cette option, le
lien est accessible à quiconque le possède, sans aucune vérification —
l'écran de connexion ayant été retiré de l'application elle-même pour cette
phase de test (voir section précédente).

**Pour un usage institutionnel durable**, un VPS privé dédié (avec
reverse proxy HTTPS, éventuellement restriction par IP/VPN, et
réactivation du login applicatif) reste l'option la plus robuste à terme —
voir l'historique du projet pour le détail de cette alternative si besoin.

## Performance

Streamlit relance tout le script à chaque interaction (chaque question posée
dans le chat). Deux optimisations évitent que ça ralentisse l'appli : les
fichiers déjà déposés ne sont pas relus/re-parsés à chaque rerun (seuls les
nouveaux fichiers sont traités), et les exports CSV/Excel/Stata d'un résultat
sont calculés une seule fois à la création du message, pas à chaque
réaffichage de l'historique. Si une question déclenche un appel LLM (via
`rag.classifier_intention` puis `rag.answer`), jusqu'à deux appels réseau
séquentiels peuvent avoir lieu sur une même question sans mot-clé reconnu :
c'est la principale source de latence restante, dépendante de la vitesse du
fournisseur LLM choisi (Groq est nettement plus rapide qu'Anthropic sur ce
point).

## Combien de tables/feuilles sont chargées ?

Une question sur la session en cours elle-même (« combien de tables sont
chargées ? », « quelles feuilles sont disponibles ? », « liste des tables »,
mais aussi une formulation plus libre comme « je parle des tables que je
viens de vous envoyer ») est reconnue et répond directement, sans passer par
le dictionnaire (qui ne sait rien de ce qui est chargé à l'instant) : nombre
de tables, nom de chacune, nombre de lignes/colonnes. Plutôt que d'énumérer
indéfiniment de nouvelles formulations exactes (approche fragile), la
détection combine deux familles de mots : un mot générique désignant les
tables (« table(s) », « feuille(s) », « classeur(s) ») et un mot lié à
l'import/au dépôt (« importation », « envoyé », « reçu », « déposé »...) — les
deux réunis dans la même question déclenchent la vraie liste, quelle que soit
la tournure de phrase. Les mots trop génériques comme « combien » ou « liste »
sont volontairement exclus de cette combinaison (ils apparaissent aussi dans
d'innombrables questions de contenu classiques, ex. « combien d'individus
dans la table X » — les garder aurait fait basculer ces questions vers la
liste des tables au lieu de calculer la vraie réponse) ; ils restent gérés
uniquement via des formulations exactes sans ambiguïté (« combien de
table(s) », « liste des tables/feuilles »). Les formulations encore plus
inhabituelles passent en dernier recours par le même classifieur LLM que pour
une répartition/un échantillon/une cohérence (action `LISTE_TABLES`), pour
éviter que le modèle ne réponde à partir d'un historique qui ne mentionne
parfois qu'une seule table à la fois et ne devine à tort qu'une seule est
chargée.

## Relations entre tables et fusion

Dès que deux tables ou plus sont chargées (fichiers séparés, ou plusieurs
feuilles d'un même classeur Excel — traitées de façon identique), l'assistant
peut :

**Noms informels reconnus.** Pas besoin de citer le nom technique exact
d'une table pour la désigner : le préfixe technique commun (`FNew`, `F_New`)
et le singulier/pluriel sont automatiquement tolérés. « la table education »
ou « presence » sont reconnus pour les tables réellement chargées
`FNewEducation` et `FNewPresences`, sans avoir à écrire le nom complet.

- **Décrire une relation** : « quelle est la relation entre Tindividual et
  TMembership ? » ou, sans préciser de nom, « quelles tables sont reliées
  entre elles ? » — il compare les vraies colonnes chargées (pas seulement la
  documentation du dictionnaire) et indique les colonnes en commun,
  candidates comme clé de jointure.
- **Plus besoin de toujours nommer les deux tables.** Si une seule table est
citée et qu'il n'y en a que deux au total chargées, l'autre est évidente :
l'assistant complète tout seul, sans redemander. S'il y a plus de deux tables
et que la question reste ambiguë, la relance liste les **vraies** tables
actuellement chargées (jamais un exemple générique) — et si une seule est déjà
identifiée, ne redemande que l'autre plutôt que de tout redemander.

**Fusionner deux tables** : « fusionne Tindividual et TMembership » —
  effectue une vraie jointure (`pandas.merge`) sur la première colonne
  commune détectée, affiche un aperçu, **enregistre le résultat comme
  nouvelle table** (`fusion_Tindividual_TMembership`) interrogeable
  directement ensuite, et fournit la syntaxe R (`merge`) et Stata (`merge
  1:1 ... using`) équivalente.
- **Calculer une différence entre deux tables (anti-jointure)** : « combien
  d'individus sont dans Presence et pas dans Education, et vice versa ? » —
  calcule une vraie anti-jointure (`pandas.merge` avec indicateur) sur la clé
  commune détectée, donne le nombre exact de lignes concernées dans chaque
  sens (demande "vice versa"/"et inversement" pour obtenir les deux sens en
  une fois), affiche un aperçu de **la liste** des lignes concernées,
  **enregistre chaque résultat comme nouvelle table**
  (`difference_Presence_sans_Education`, et
  `difference_Education_sans_Presence` si les deux sens sont demandés) —
  interrogeable ensuite pour un indicateur ou un échantillon (« à travers
  cette base, donne-moi... »), et fournit la syntaxe R (`dplyr::anti_join`)
  et Stata (`merge` + `keep if _merge == 1`) équivalente.

Ces réponses (comme celles sur les doublons/incohérences/échantillons) sont
toujours calculées directement à partir des vraies données chargées — jamais
rédigées ou devinées par le LLM — pour rester précises. Le rapport de
cohérence indique en plus explicitement quelles colonnes d'identifiant et de
date ont été vérifiées, pas seulement les anomalies trouvées.

**Mémoire de la conversation, y compris sur les tables.** Une question de
suivi qui ne renomme pas la table ("et les doublons ?" après avoir parlé
d'une table précise) reste rattachée au bon contexte : la résolution de
table regarde d'abord la question en cours, puis les derniers échanges de la
conversation. Il n'y a plus de table par défaut choisie dans la barre
latérale : si rien ne se résout après la question et l'historique, l'analyse
est menée sur toutes les tables chargées (voir plus bas).

**Relance de calcul.** Si une réponse précédente a hésité ou n'a pas calculé
directement (rare, mais peut arriver si la question initiale n'a pas été bien
comprise), une relance courte qui ne répète pas le mot-clé initial ("il faut
analyser directement", "calcule-le vraiment", "sois précis"...) est reconnue
si le tour précédent portait déjà sur une différence/fusion/relation : le
vrai calcul déterministe est relancé plutôt que de laisser la question filer
vers une réponse générique du modèle de langage.

## Analyses statistiques (univariée, bivariée, multivariée) et code associé

Chaque analyse sur une table — répartition, échantillon, doublons, cohérence,
tableau croisé, corrélation — est **toujours calculée directement à partir
des vraies données chargées**, jamais rédigée ou devinée par le modèle de
langage, et **systématiquement accompagnée de la syntaxe R et Stata
équivalente** pour reproduire ou approfondir le calcul en dehors du chat.

- **Univariée** : « répartition de sex », « échantillon de 100 », « doublons »,
  « cohérence » — comme avant, avec en plus le code R/Stata à la fin de
  chaque réponse.
- **Bivariée** : « tableau croisé entre sex et education_level » — vrai
  `pandas.crosstab` (effectifs + marges), avec `table()`/`tab` en R/Stata.
- **Multivariée** : « corrélation entre individid, sex et age » (variables
  numériques → matrice de corrélation) ou « analyse multivariée de sex,
  education_level et field_wrkr » (3 colonnes catégorielles ou plus →
  effectifs croisés sur toutes les combinaisons observées).
- **Contrôle qualité des agents enquêteurs** : « performance des agents
  enquêteurs », « erreurs par agent » — détecte automatiquement la colonne
  d'agent de terrain (ex. `field_wrkr`) et donne, par agent : nombre de
  fiches saisies, fiches impliquées dans un doublon d'identifiant, dates
  invraisemblables détectées, taux moyen de valeurs manquantes — pour
  repérer une charge de travail inhabituelle ou un agent avec plus
  d'erreurs que les autres.

**Aucune table par défaut : toutes les tables travaillent dès le départ.**
La barre latérale n'impose plus de table active — elle liste simplement les
tables chargées et leurs colonnes (dans un menu déroulant repliable). Pour
cibler UNE table précise, il suffit de la nommer dans la question (ou de
nommer une colonne qui n'existe que dans cette table-là) ; sinon,
l'assistant considère que la question porte sur toutes les tables
concernées :

- Si une colonne mentionnée existe dans plusieurs tables à la fois (ex.
  `individid` présent dans 20 tables), l'analyse est calculée pour
  **chacune** des tables concernées plutôt que de silencieusement n'en
  garder qu'une. Ce principe couvre aussi les analyses à plusieurs colonnes
  (tableau croisé, corrélation, multivariée) : si les colonnes mentionnées
  (ex. « tableau croisé entre sex et education_level ») existent ensemble
  dans plusieurs tables, le résultat est calculé pour chacune, pas seulement
  la première chargée.
- Si la question ne nomme ni table ni colonne reconnaissable (ex. « les
  doublons ? », « vérifie la cohérence », « échantillon de 50 », « performance
  des agents ») et qu'aucun contexte de conversation ne permet de trancher,
  le calcul est fait sur **toutes** les tables chargées et les résultats sont
  présentés les uns après les autres.
- Si la question dépend forcément d'une colonne précise (répartition,
  tableau croisé, corrélation, analyse multivariée) et qu'aucune colonne
  reconnaissable n'a été citée, l'assistant ne devine pas : il répond en
  listant les vraies tables chargées et leurs colonnes pour demander de
  préciser.

**Un nom de colonne ne se confond jamais avec le nom d'une table.** La
reconnaissance d'une table mentionnée informellement (ex. « education » pour
`FNewEducation`) exige un mot entier — une colonne comme `education_level`
ne déclenche donc jamais à tort la table `FNewEducation`, même si son nom
commence par les mêmes lettres.

**Reconnaissance insensible aux accents et aux accords grammaticaux.** Une
question est reconnue qu'elle soit tapée avec ou sans accents
(« cohérence »/« coherence », « répartition »/« repartition »), et les mots-clés
plus ambigus (« relation », « corrélation »...) sont reconnus par mot entier
pour ne jamais se confondre entre eux (« corrélation » ne déclenche jamais à
tort une question de relation entre tables, par exemple).

## Audit de cohérence avancé (catalogue de contrôles métier de l'observatoire)

Au-delà du contrôle de cohérence générique (doublons + dates invraisemblables),
une question comme « audit complet de cohérence », « audit de cohérence
avancé » ou « toutes les incohérences » déclenche un **catalogue de contrôles
spécifiques au type d'enquête de l'OPO**, appliqué automatiquement à toutes les
tables chargées (ou à une seule si elle est nommée, ex. « audit complet de
FNewIndividual »). Chaque contrôle est **auto-détecté par le nom des colonnes
réellement présentes** (même principe que la détection des colonnes
d'identifiant ou d'agent) : rien n'est deviné ni supposé sur des colonnes
absentes, et la liste des contrôles non applicables à une table (colonnes non
reconnues) est affichée explicitement pour rester transparent sur ce qui a
vraiment été vérifié.

Contrôles par table :

- identifiants de longueur inhabituelle (par rapport à la longueur la plus
  fréquente de la colonne) ;
- auto-référence (un identifiant égal à son propre « ID2 », ex.
  `individid == individid2`) ;
- parents identiques (`fatherid == motherid`) ;
- jeune enfant (moins de 5 ans) sans `motherid` renseigné, et séparément sans
  `motherid` ni `fatherid` ;
- valeur sentinelle de non-réponse codée (poids = 9999, taille = 99) ;
- coordonnées GPS manquantes ou hors du territoire burkinabè ;
- format de téléphone invalide (segments qui ne font pas 8 chiffres) ;
- dates d'arrivée/départ incohérentes (départ antérieur ou égal à l'arrivée) ;
- résidence multiple pour un même individu (plusieurs ménages/localisations
  distincts) ;
- naissance postérieure au décès, enregistrement antérieur à la naissance ;
- âge hors de la tranche attendue selon le type de fiche détecté par le nom de
  la table (12-49 ans pour une fiche génésique/grossesse, 12-40 ans pour une
  fiche d'histoire matrimoniale, 5-34 ans pour l'éducation, 15-120 ans pour
  l'emploi).

Contrôles croisés entre tables (population « éligible » de la fiche présence —
a dormi dans le ménage, sans date de départ enregistrée) :

- éligibilité présence ↔ éducation/emploi/histoire génésique
  complémentaire/pauvreté/santé (qui est éligible sans avoir la fiche, et qui a
  la fiche sans être éligible) ;
- décédé mais toujours présent dans la fiche présence ;
- a dormi dans le ménage mais apparaît aussi en migration OUT ;
- grossesse sans issue de grossesse enregistrée.

**Non couvert pour l'instant, explicitement signalé comme tel dans le
rapport** : les contrôles les plus spécifiques au questionnaire (codes de
réponse détaillés d'une question de santé précise, par exemple), qui
nécessitent de connaître le nom exact d'une colonne de code de réponse propre
à l'observatoire et n'ont pas pu être vérifiés depuis cette session — à
compléter dès que ces colonnes précises sont confirmées.

## Performances de terrain (ménages/UCH, objectif, rapport Word)

Distinct du contrôle qualité par agent (ci-dessus, qui mesure les erreurs) :
une question comme « bilan de terrain », « avancement de la collecte » ou
« ménages visités par agent » déclenche un **rapport de volume d'activité de
terrain**, agrégé automatiquement sur **toutes les tables chargées** qui
comportent une colonne d'agent détectable (fiche présence, naissance, décès,
grossesse — et toute autre table avec un agent, classée en « Autres fiches »
pour ne rien perdre) :

- nombre de ménages/UCH visités (fiches + ménages **distincts**, via la
  colonne `menageid`/`locationid` détectée) ;
- naissances, décès, grossesses enregistrés par agent ;
- **exclusion configurable** d'agents non-terrain directement depuis la
  question (« ... en excluant les agents 12, 45 et 67 ») — aucune liste n'est
  codée en dur, l'observatoire reste seul décisionnaire de qui exclure ;
- **jointure agent ↔ contrôleur** automatique si une table « équipe »
  (colonnes agent + contrôleur/superviseur détectées) est chargée ;
- **performance par jour** (`n_fiches` par date et par agent) à partir de la
  première table pertinente (fiche présence en priorité) ;
- **prévision vers un objectif configurable** (17000 ménages par défaut,
  ajustable dans la question : « ... objectif 20000 ») : cumul actuel, reste à
  faire, rythme journalier moyen, date de fin projetée au rythme actuel ;
- **courbe de progression** (cumul de fiches par jour) affichée directement
  sous la réponse ;
- **rapport Word téléchargeable** (bouton dédié) reprenant le tableau par
  agent et la projection vers l'objectif, pour un partage hors chat.

**Historique des actualisations** : « historique des actualisations » liste,
pour la session en cours, chaque table (re)chargée avec son horodatage et son
nombre de lignes.

**Recherche instantanée par identifiant** : « recherche l'individu 1024 »
retrouve en un seul coup toutes les lignes correspondantes dans **toutes**
les tables chargées comportant une colonne d'identifiant, sans avoir à nommer
chaque table une par une.

_Non couvert pour l'instant_ : la simulation par hypothèse de rythme est
disponible comme fonction (`data_tools.simulation_rythme`) mais pas encore
reliée à une formulation en langage naturel dans le chat — à ajouter si
l'équipe confirme la formulation qu'elle utiliserait.

## Lire une image (photo, scan)

Section "🖼️ Lire une image" dans la barre latérale : dépose une photo ou un
scan (ex : une fiche terrain remplie), pose une question, l'assistant répond
en s'appuyant sur le contenu visuel de l'image. Nécessite une clé Anthropic
active (la lecture d'image n'est pas gérée par le modèle Groq gratuit). Cette
lecture est ponctuelle (pas d'indexation permanente de l'image, contrairement
aux documents texte) : pour une question ultérieure sur la même image, il
faut la redéposer.

## Conversation naturelle (mémoire, rédaction, reformulation)

Trois ajustements pour que l'assistant se comporte moins comme un moteur de
recherche et plus comme une vraie conversation :

- **Mémoire de conversation** : les 6 derniers échanges sont transmis au LLM à
  chaque question documentaire (`app.py:historique_recent` → `rag.answer(...,
  historique=...)`), pour que les questions de suivi ("et pour l'autre
  table ?", "peux-tu détailler ?") fonctionnent sans tout redemander depuis le
  début.
- **Rédaction naturelle** : le prompt (`rag.build_prompt`) demande
  explicitement au LLM de reformuler et d'expliquer avec ses propres mots,
  jamais de recopier les extraits du contexte tels quels.
- **Reformulation de requête en cas d'échec** : si le meilleur score de
  recherche TF-IDF est très faible (`rag.SEUIL_SCORE_FAIBLE`, question trop
  reformulée par rapport au vocabulaire du corpus), l'assistant demande une
  seule fois au LLM de reformuler la requête en termes plus techniques, puis
  retente la recherche. Ne se déclenche pas quand la recherche initiale
  fonctionne déjà bien (pas de coût supplémentaire dans le cas courant).

**Important : rien de tout cela ne fonctionne sans clé LLM renseignée.** Sans
clé Groq ou Anthropic dans la barre latérale, l'assistant se contente
d'afficher le contexte brut retrouvé — ce qui explique l'impression de ne pas
être un "vrai" assistant. Pour un rendu le plus proche de Claude, utiliser la
clé Anthropic (modèle `claude-haiku-4-5-20251001`) : elle est maintenant
prioritaire sur Groq quand les deux sont renseignées.

## Limites connues

- La recherche documentaire utilise du TF-IDF (mots-clés pondérés), pas des
  embeddings sémantiques : elle marche bien pour retrouver une variable par
  son nom ou sa description, moins bien pour des questions très reformulées.
  La reformulation de requête (ci-dessus) rattrape une partie des cas, mais un
  passage à `sentence-transformers` (voir commentaire dans `ingest.py`) reste
  la solution la plus robuste si le besoin se confirme.
- La détection d'intention sur une table déposée passe d'abord par des
  mots-clés simples (rapide, gratuit, sans LLM), puis, si aucune clé LLM n'est
  configurée et qu'aucun mot-clé ne correspond, la question est traitée comme
  une question documentaire avec un rappel des formulations reconnues. Avec
  une clé LLM configurée, une classification plus souple prend le relais
  (`rag.classifier_intention`) pour comprendre des formulations moins
  littérales — mais elle reste imparfaite sur des questions très ambiguës.
- L'authentification protège l'accès à l'application, mais le rôle
  "Correction" ne fait aujourd'hui qu'afficher un badge différent : il n'existe
  pas encore de workflow de validation/application des corrections dans
  l'outil lui-même (les corrections restent consignées et appliquées en
  dehors de l'application, par une personne habilitée).
