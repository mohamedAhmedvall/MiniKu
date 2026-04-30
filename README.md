# MiniKu - Mini-Dataiku MVP

Application Streamlit No-Code pour la Data Science, en 3 modules :

1. **Atelier de Data Prep** : upload CSV, exploration, nettoyage (suppression de colonnes, gestion des NaN, encodage).
2. **Studio de Modélisation (AutoML)** : choix de la cible et du type de tâche, entraînement via PyCaret (`compare_models`), leaderboard, sauvegarde du meilleur modèle en `.pkl`.
3. **Fabrique de Prédictions** : upload d'un nouveau CSV, prédictions via le modèle entraîné, téléchargement des résultats.

## Installation

```bash
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
