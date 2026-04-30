# MiniKu - Mini-Dataiku MVP

Application Streamlit No-Code pour la Data Science, en 3 modules :

1. **Atelier de Data Prep** : upload CSV, exploration, nettoyage (suppression de colonnes, gestion des NaN, encodage).
2. **Studio de Modélisation (AutoML)** : choix de la cible et du type de tâche, entraînement via PyCaret (`compare_models`), leaderboard, sauvegarde du meilleur modèle en `.pkl`.
3. **Fabrique de Prédictions** : upload d'un nouveau CSV, prédictions via le modèle entraîné, téléchargement des résultats.

---

## :rocket: Méthode recommandée : Dev Container (Docker)

Cette méthode contourne tous les problèmes de version Python sur Windows
(Python 3.12+, build numpy, etc.). PyCaret 3.x exige Python 3.9-3.11, le
conteneur fournit **Python 3.11**.

### Pré-requis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé.
- [VS Code](https://code.visualstudio.com/) + extension **Dev Containers** (`ms-vscode-remote.remote-containers`).

### Configuration du proxy d'entreprise (si nécessaire)

Si tu es derrière un proxy d'entreprise, **définis les variables d'environnement
sur ta machine Windows avant de lancer VS Code**. Ouvre PowerShell :

```powershell
# Pour la session courante uniquement
$env:HTTP_PROXY  = "http://user:password@proxy.entreprise.com:8080"
$env:HTTPS_PROXY = "http://user:password@proxy.entreprise.com:8080"
$env:NO_PROXY    = "localhost,127.0.0.1,.entreprise.com"
code .
```

Pour rendre ces variables permanentes (recommandé) :

```powershell
[Environment]::SetEnvironmentVariable("HTTP_PROXY",  "http://user:password@proxy.entreprise.com:8080", "User")
[Environment]::SetEnvironmentVariable("HTTPS_PROXY", "http://user:password@proxy.entreprise.com:8080", "User")
[Environment]::SetEnvironmentVariable("NO_PROXY",    "localhost,127.0.0.1,.entreprise.com", "User")
```

Puis **redémarre VS Code et Docker Desktop** pour qu'ils prennent en compte les
variables. Les `${localEnv:HTTP_PROXY}` du `devcontainer.json` les
récupèrent automatiquement et les transmettent au conteneur (build + runtime).

> Si ton proxy intercepte HTTPS avec un certificat d'entreprise, il faudra
> en plus copier le certificat racine dans `/usr/local/share/ca-certificates/`
> et exécuter `update-ca-certificates` dans le Dockerfile. Demande à ton IT
> le `.crt` interne.

### Lancement

1. Ouvre le dossier `MiniKu` dans VS Code.
2. VS Code propose **"Reopen in Container"** (en bas à droite). Clique dessus.
   - Sinon : `Ctrl+Shift+P` -> `Dev Containers: Reopen in Container`.
3. Le build prend quelques minutes la première fois (téléchargement de
   l'image `python:3.11-slim` puis `pip install` des dépendances).
4. Une fois dans le conteneur, ouvre un terminal VS Code (`Ctrl+J`) et lance :

```bash
streamlit run app.py
```

5. VS Code te propose d'ouvrir le navigateur sur le port forwardé `8501`.

---

## :whale: Méthode alternative : Docker Compose (sans VS Code)

Si tu préfères ne pas utiliser VS Code Dev Containers :

```powershell
# Configure d'abord le proxy si besoin (voir section ci-dessus)
docker compose up --build
```

Puis ouvre <http://localhost:8501> dans ton navigateur.

Pour arrêter : `Ctrl+C` puis `docker compose down`.

---

## :snake: Méthode alternative : installation locale (Python 3.11 natif)

Si tu peux installer Python 3.11 sur ta machine (hors devcontainer) :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

> :warning: PyCaret 3.x ne supporte **pas** Python 3.12+. Utiliser Python
> 3.9, 3.10 ou 3.11 obligatoirement.

---

## Structure du projet

```
MiniKu/
├── .devcontainer/
│   ├── devcontainer.json     # Config VS Code Dev Containers + proxy
│   └── Dockerfile            # Image Python 3.11 + deps + proxy
├── docker-compose.yml        # Lancement standalone via docker compose
├── app.py                    # Application Streamlit (3 modules)
├── requirements.txt          # streamlit, pandas, pycaret, scikit-learn
└── README.md
```

## Navigation dans l'app

La navigation entre les 3 modules se fait via la `st.sidebar`. Des messages
d'avertissement bloquent l'accès aux étapes 2 et 3 tant que les prérequis
(dataset chargé, modèle entraîné) ne sont pas remplis.

---

## Dépannage

### Le build Docker échoue sur `pip install` (timeout / proxy)

Vérifie que `HTTP_PROXY` et `HTTPS_PROXY` sont bien définis **dans ta session
PowerShell** avant de lancer VS Code/Docker, et que l'URL inclut bien
`http://` (pas `https://`) et le port. Test rapide :

```powershell
echo $env:HTTP_PROXY
```

### Le build Docker échoue sur la vérification SSL (proxy MITM)

Ton proxy intercepte le TLS. Récupère le certificat racine de ton entreprise
(`.crt`), place-le dans `.devcontainer/corp-ca.crt`, puis ajoute dans le
Dockerfile (avant `pip install`) :

```dockerfile
COPY .devcontainer/corp-ca.crt /usr/local/share/ca-certificates/corp-ca.crt
RUN update-ca-certificates
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    PIP_CERT=/etc/ssl/certs/ca-certificates.crt
```

### Le port 8501 n'est pas accessible

Dans VS Code : onglet **Ports** -> vérifier que 8501 est forwardé. En
docker-compose : vérifier `docker compose ps` que le conteneur tourne.
