# Assistant OPO — prototype RAG

Agent qui répond aux questions sur le dictionnaire de données de l'Observatoire
de Population de Ouagadougou (RAG), et calcule des indicateurs, échantillons
et contrôles de cohérence sur une table déposée (CSV/Excel), quelles que
soient les colonnes réellement présentes dans le fichier.

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

## Comptes / authentification (actuellement désactivés)

Pour une phase de test avec l'équipe via un lien partagé, l'écran de connexion
est désactivé : quiconque a le lien accède directement à l'assistant, sans
identifiant ni mot de passe. La restriction d'accès repose alors sur la
diffusion du lien lui-même (ne le partager qu'à l'équipe) et, en cas de
déploiement sur Streamlit Community Cloud, sur son option d'app privée (accès
limité à une liste d'emails autorisés — voir section Déploiement).

Le mécanisme de comptes individuels (`streamlit-authenticator`,
`auth_config.yaml`, rôles `correction` / `consultation`) reste dans le projet
et testé (`tests/test_auth.py`), prêt à être réactivé dans `app.py` si l'usage
évolue vers un besoin de comptes nominatifs et de rôles différenciés.

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

65 tests couvrent `ingest.py` (découpage en chunks), `prepare_corpus.py`
(conversion Word/PowerPoint/PDF/Excel/texte, non-reconversion si déjà à jour, exclusion du
dictionnaire xlsx principal), `data_tools.py` (répartitions, échantillon reproductible,
doublons, dates invraisemblables, suppression des colonnes nominatives, export
CSV/Excel/Stata, résolution de la table ciblée), `rag.py` (récupération
documentaire, garde-fous index/clé manquants, classification d'intention) et
`auth_config.yaml` (structure des comptes, hash/vérification des mots de passe).

## Structure

- `data/source_documents/` — dépôt des documents bruts (Word, PDF, Excel, texte) à indexer : c'est ici qu'on ajoute une nouvelle fiche ou un nouveau manuel.
- `data/docs/` — textes prêts à être découpés en chunks : dictionnaire de données (généré depuis le fichier Excel source, un fichier par table) + tout document converti automatiquement depuis `data/source_documents/` (préfixe `source_`).
- `prepare_corpus.py` — convertit les fichiers de `data/source_documents/` (`.doc`, `.docx`, `.pdf`, `.xlsx`, `.txt`, `.md`) en texte dans `data/docs/`.
- `ingest.py` — construit l'index de recherche (TF-IDF, léger, sans téléchargement de modèle) ; appelle `prepare_corpus.py` en premier.
- `rag.py` — recherche + construction du prompt + appel au LLM.
- `data_tools.py` — analyse d'une table déposée : répartitions, échantillon reproductible, détection de doublons/dates invraisemblables (colonnes ID/date détectées automatiquement par leur nom/contenu), résolution automatique de la table ciblée par une question (nom de table ou colonne mentionnée), reconnaissance de toutes les feuilles d'un classeur Excel comme autant de tables distinctes. Les colonnes de type nom/prénom sont systématiquement retirées avant toute analyse.
- `app.py` — interface de chat (Streamlit).
- `instructions_systeme.md` — rôle et garde-fous de l'assistant (ce qu'il fait, ce qu'il ne fait jamais).
- `auth_config.yaml` — comptes de l'équipe (identifiants, mots de passe, rôles).
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
