# MiniKu - POC Reflex

POC d'une refonte du **module Data Prep** sur Reflex, pour evaluer le pivot
depuis Streamlit.

## Pourquoi ce POC

Streamlit a montre ses limites pour ce type d'app :
- rerun complet a chaque interaction (etat fragile)
- pas de bouton "stop" propre pendant un training
- customisation UI limitee aux hacks CSS
- workflows multi-pages galere

Reflex (ex-Pynecone) compile en React/Next.js mais reste 100% Python. Tu
gardes pandas/pycaret tels quels, mais l'UI devient une vraie SPA.

## Ce que le POC demontre

- Layout Dataiku-like (sidebar sombre + contenu blanc)
- Upload CSV avec drag & drop
- Cartes de metriques globales
- Preview interactif (data_table avec recherche et tri)
- Recettes : suppression de colonnes, gestion NaN
- Etat persistant entre interactions (pas de rerun)
- Log des recettes appliquees

Le module Modelisation et Predictions ne sont **pas** dans ce POC : on
valide d'abord l'archi avant de tout reecrire.

## Lancement

### Pre-requis

Python 3.11 (deja en place dans le devcontainer du projet). Reflex
auto-installe Bun et Node.js a la premiere `reflex init`, donc bien
laisser passer le proxy si tu es en entreprise.

### Installation

```bash
cd reflex_poc
pip install -r requirements.txt
reflex init    # cree .web/ et installe les deps frontend (premier run uniquement)
reflex run
```

### Acces

- Frontend : http://localhost:3000
- Backend (API) : http://localhost:8000

Dans le devcontainer, ces ports sont a forwarder dans `.devcontainer/devcontainer.json`.

## Structure

```
reflex_poc/
├── README.md
├── requirements.txt
├── rxconfig.py           # config Reflex (ports, app name)
└── miniku/
    ├── __init__.py
    └── miniku.py         # app complete (state + composants)
```

## Comparaison rapide Streamlit vs Reflex

| Critere                    | Streamlit             | Reflex                       |
|----------------------------|-----------------------|------------------------------|
| Modele d'execution         | Rerun script complet  | Reactive (callbacks)         |
| Etat                       | session_state (fragile) | classe State (typage propre) |
| Customisation UI           | CSS hacks             | Composants Radix UI natifs   |
| Stop d'un job long         | Difficile             | Possible (tasks asynchrones) |
| Production deployment      | Streamlit Cloud only  | Docker, n'importe ou         |
| Time-to-MVP                | 1 jour                | 3-4 jours                    |
| Plafond                    | Bas                   | Eleve                        |

## Limitations connues du POC

- L'API Reflex evolue vite ; certains noms de composants peuvent varier
  selon la version installee (ex: `rx.upload_files` vs `rx.selected_files`).
  Si un import echoue, lance `reflex --version` et adapte.
- Le `_df: pd.DataFrame` interne au State n'est pas serialise au
  frontend mais reste en memoire sur le backend. Pour un deploiement
  multi-utilisateurs il faudra une vraie persistance (DB, cache).
- Le data_table est limite aux types simples (on stringifie tout en
  preview). Pour des dataframes massifs il faudra paginer cote backend.
