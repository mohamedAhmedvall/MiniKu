# MiniKu - Mini-Dataiku MVP

Application Streamlit No-Code pour la Data Science, en 3 modules :

1. **Atelier de Data Prep** : upload CSV, exploration, nettoyage (suppression de colonnes, gestion des NaN, encodage).
2. **Studio de Modélisation (AutoML)** : choix de la cible et du type de tâche, entraînement via PyCaret (`compare_models`), leaderboard, sauvegarde du meilleur modèle en `.pkl`.
3. **Fabrique de Prédictions** : upload d'un nouveau CSV, prédictions via le modèle entraîné, téléchargement des résultats.

## Pré-requis : version de Python

> **Important** : PyCaret 3.x supporte **Python 3.9, 3.10 ou 3.11**.
> Sur Python 3.12 / 3.13 / 3.14, pip essaiera de compiler `numpy` depuis les
> sources (et échouera sans Visual Studio Build Tools sur Windows).

Vérifie ta version :

```bash
python --version
```

Si tu n'as pas la bonne version, installe **Python 3.11** depuis
<https://www.python.org/downloads/release/python-3119/> puis recommence dans un
nouvel environnement virtuel.

## Installation (recommandée : environnement virtuel)

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

## Navigation

La navigation entre les 3 modules se fait via la `st.sidebar`. Des messages
d'avertissement bloquent l'accès aux étapes 2 et 3 tant que les prérequis
(dataset chargé, modèle entraîné) ne sont pas remplis.

## Dépannage

### `ERROR: Unknown compiler(s) [...]` lors de l'install de numpy

Symptôme : pip télécharge `numpy-1.26.4.tar.gz` puis tente une compilation qui
échoue. C'est que ta version de Python est trop récente pour les wheels de
PyCaret 3.x. Solution : installer **Python 3.11** et refaire l'installation
dans un nouvel environnement virtuel (voir ci-dessus).

### Alternative : conda

Si tu préfères conda, c'est encore plus simple côté binaires :

```bash
conda create -n miniku python=3.11 -y
conda activate miniku
pip install -r requirements.txt
```
