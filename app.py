"""
MiniKu - Mini-Dataiku MVP
Application Streamlit No-Code pour la Data Science : Data Prep, AutoML, Prédiction.

Lancement :
    streamlit run app.py
"""

import os
import io
import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Configuration globale de l'application
# =============================================================================
st.set_page_config(
    page_title="MiniKu - Mini-Dataiku",
    page_icon=":bar_chart:",
    layout="wide",
)

MODEL_PATH = "miniku_best_model"  # PyCaret ajoute automatiquement l'extension .pkl


# =============================================================================
# Initialisation du session_state
# =============================================================================
def init_session_state():
    """Initialise les clés du session_state utilisées par les 3 modules."""
    defaults = {
        "raw_df": None,         # DataFrame original tel qu'uploadé
        "clean_df": None,       # DataFrame nettoyé partagé entre les modules
        "task_type": None,      # "Classification" ou "Régression"
        "target_column": None,  # Nom de la colonne cible
        "leaderboard": None,    # Leaderboard PyCaret
        "model_trained": False, # Indique si un modèle a été entraîné et sauvegardé
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# Module 1 : Atelier de Data Prep
# =============================================================================
def page_data_prep():
    st.title(":wrench: Atelier de Data Prep")
    st.markdown(
        "Uploade un fichier CSV, explore-le, puis nettoie-le avant de passer "
        "à l'étape de modélisation."
    )

    # ---- Upload du fichier ----
    uploaded_file = st.file_uploader(
        "Charger un fichier CSV", type=["csv"], key="csv_uploader"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df.copy()
            # Au premier chargement, on initialise le clean_df au raw_df
            if st.session_state.clean_df is None or st.button(
                "Réinitialiser le nettoyage avec le fichier uploadé"
            ):
                st.session_state.clean_df = df.copy()
        except Exception as e:
            st.error(f"Erreur lors de la lecture du CSV : {e}")
            return

    # Si aucun dataset n'est encore chargé, on s'arrête ici
    if st.session_state.clean_df is None:
        st.info("En attente d'un fichier CSV...")
        return

    df = st.session_state.clean_df

    # ---- Aperçu interactif ----
    st.subheader("Aperçu des données")
    col1, col2, col3 = st.columns(3)
    col1.metric("Lignes", df.shape[0])
    col2.metric("Colonnes", df.shape[1])
    col3.metric("Valeurs manquantes", int(df.isna().sum().sum()))

    st.dataframe(df, use_container_width=True)

    with st.expander("Statistiques descriptives"):
        st.dataframe(df.describe(include="all").T, use_container_width=True)

    with st.expander("Types de colonnes et valeurs manquantes par colonne"):
        info_df = pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "missing": df.isna().sum(),
                "missing_pct": (df.isna().mean() * 100).round(2),
            }
        )
        st.dataframe(info_df, use_container_width=True)

    # ---- Suppression de colonnes ----
    st.subheader("1. Supprimer des colonnes")
    cols_to_drop = st.multiselect(
        "Sélectionner les colonnes à supprimer",
        options=list(df.columns),
        key="cols_to_drop",
    )
    if st.button("Supprimer les colonnes sélectionnées"):
        if cols_to_drop:
            st.session_state.clean_df = df.drop(columns=cols_to_drop)
            st.success(f"Colonnes supprimées : {', '.join(cols_to_drop)}")
            st.rerun()
        else:
            st.warning("Aucune colonne sélectionnée.")

    # ---- Gestion des valeurs manquantes ----
    st.subheader("2. Gérer les valeurs manquantes")
    missing_strategy = st.selectbox(
        "Stratégie",
        options=[
            "Ne rien faire",
            "Supprimer les lignes contenant des NaN",
            "Remplacer par la moyenne (numérique)",
            "Remplacer par la médiane (numérique)",
            "Remplacer par zéro (numérique)",
        ],
        key="missing_strategy",
    )
    if st.button("Appliquer la stratégie de NaN"):
        df_work = st.session_state.clean_df.copy()
        if missing_strategy == "Supprimer les lignes contenant des NaN":
            df_work = df_work.dropna()
        elif missing_strategy == "Remplacer par la moyenne (numérique)":
            num_cols = df_work.select_dtypes(include=np.number).columns
            df_work[num_cols] = df_work[num_cols].fillna(df_work[num_cols].mean())
        elif missing_strategy == "Remplacer par la médiane (numérique)":
            num_cols = df_work.select_dtypes(include=np.number).columns
            df_work[num_cols] = df_work[num_cols].fillna(df_work[num_cols].median())
        elif missing_strategy == "Remplacer par zéro (numérique)":
            num_cols = df_work.select_dtypes(include=np.number).columns
            df_work[num_cols] = df_work[num_cols].fillna(0)

        st.session_state.clean_df = df_work
        st.success(f"Stratégie appliquée : {missing_strategy}")
        st.rerun()

    # ---- Encodage simple des variables textuelles ----
    st.subheader("3. Encoder les variables textuelles")
    text_cols = list(
        st.session_state.clean_df.select_dtypes(include=["object", "category"]).columns
    )
    if not text_cols:
        st.info("Aucune colonne textuelle détectée.")
    else:
        cols_to_encode = st.multiselect(
            "Sélectionner les colonnes à encoder",
            options=text_cols,
            key="cols_to_encode",
        )
        encoding_method = st.selectbox(
            "Méthode d'encodage",
            options=["Label Encoding", "One-Hot Encoding"],
            key="encoding_method",
        )
        if st.button("Encoder les colonnes sélectionnées"):
            if not cols_to_encode:
                st.warning("Aucune colonne sélectionnée.")
            else:
                df_work = st.session_state.clean_df.copy()
                if encoding_method == "Label Encoding":
                    for col in cols_to_encode:
                        df_work[col] = df_work[col].astype("category").cat.codes
                else:  # One-Hot Encoding
                    df_work = pd.get_dummies(df_work, columns=cols_to_encode)
                st.session_state.clean_df = df_work
                st.success(
                    f"Encodage {encoding_method} appliqué sur : "
                    f"{', '.join(cols_to_encode)}"
                )
                st.rerun()

    # ---- Aperçu final + export ----
    st.subheader("Dataset nettoyé (état courant)")
    st.dataframe(st.session_state.clean_df, use_container_width=True)

    csv_bytes = st.session_state.clean_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Télécharger le dataset nettoyé (CSV)",
        data=csv_bytes,
        file_name="dataset_clean.csv",
        mime="text/csv",
    )


# =============================================================================
# Module 2 : Studio de Modélisation (AutoML)
# =============================================================================
def page_modeling():
    st.title(":robot_face: Studio de Modélisation (AutoML)")

    if st.session_state.clean_df is None:
        st.warning(
            "Aucun dataset nettoyé n'est disponible. "
            "Commence par l'étape **Atelier de Data Prep**."
        )
        return

    df = st.session_state.clean_df
    st.markdown(f"Dataset courant : **{df.shape[0]}** lignes x **{df.shape[1]}** colonnes")
    st.dataframe(df.head(20), use_container_width=True)

    # ---- Sélection cible et type de tâche ----
    target = st.selectbox(
        "Colonne cible (Target)",
        options=list(df.columns),
        key="target_select",
    )
    task_type = st.radio(
        "Type de tâche",
        options=["Classification", "Régression"],
        horizontal=True,
        key="task_type_select",
    )

    st.session_state.target_column = target
    st.session_state.task_type = task_type

    # ---- Lancement de l'entraînement ----
    if st.button(":rocket: Lancer l'entraînement"):
        # Vérifications de base
        if df[target].isna().any():
            st.error(
                "La colonne cible contient des valeurs manquantes. "
                "Retourne dans l'Atelier de Data Prep pour les nettoyer."
            )
            return

        with st.spinner("Entraînement en cours, cela peut prendre quelques minutes..."):
            try:
                if task_type == "Classification":
                    from pycaret.classification import (
                        setup, compare_models, pull, save_model,
                    )
                else:
                    from pycaret.regression import (
                        setup, compare_models, pull, save_model,
                    )

                setup(
                    data=df,
                    target=target,
                    session_id=42,
                    verbose=False,
                    html=False,
                )

                best_model = compare_models()
                leaderboard = pull()

                # Sauvegarde du meilleur modèle (PyCaret ajoute le suffixe .pkl)
                save_model(best_model, MODEL_PATH)

                st.session_state.leaderboard = leaderboard
                st.session_state.model_trained = True

                st.success("Entraînement terminé. Meilleur modèle sauvegardé.")
            except Exception as e:
                st.error(f"Erreur pendant l'entraînement PyCaret : {e}")
                return

    # ---- Affichage du leaderboard ----
    if st.session_state.leaderboard is not None:
        st.subheader("Leaderboard des modèles")
        st.dataframe(st.session_state.leaderboard, use_container_width=True)

        if os.path.exists(f"{MODEL_PATH}.pkl"):
            with open(f"{MODEL_PATH}.pkl", "rb") as f:
                st.download_button(
                    "Télécharger le modèle gagnant (.pkl)",
                    data=f.read(),
                    file_name="miniku_best_model.pkl",
                    mime="application/octet-stream",
                )


# =============================================================================
# Module 3 : Fabrique de Prédictions
# =============================================================================
def page_predict():
    st.title(":crystal_ball: Fabrique de Prédictions")

    if not st.session_state.model_trained or not os.path.exists(f"{MODEL_PATH}.pkl"):
        st.warning(
            "Aucun modèle entraîné n'est disponible. "
            "Commence par l'étape **Studio de Modélisation**."
        )
        return

    st.markdown(
        "Charge un nouveau fichier CSV (sans la colonne cible) pour générer "
        "les prédictions avec le modèle entraîné précédemment."
    )

    new_file = st.file_uploader(
        "Charger un fichier CSV pour prédiction", type=["csv"], key="predict_uploader"
    )
    if new_file is None:
        st.info("En attente d'un fichier CSV à prédire...")
        return

    try:
        new_df = pd.read_csv(new_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du CSV : {e}")
        return

    st.subheader("Aperçu du fichier à prédire")
    st.dataframe(new_df.head(20), use_container_width=True)

    if st.button("Générer les prédictions"):
        with st.spinner("Calcul des prédictions..."):
            try:
                if st.session_state.task_type == "Classification":
                    from pycaret.classification import load_model, predict_model
                else:
                    from pycaret.regression import load_model, predict_model

                model = load_model(MODEL_PATH)
                predictions = predict_model(model, data=new_df)
            except Exception as e:
                st.error(f"Erreur pendant la prédiction : {e}")
                return

        st.success("Prédictions générées.")
        st.subheader("Aperçu des prédictions")
        st.dataframe(predictions.head(50), use_container_width=True)

        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Télécharger les prédictions (CSV)",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
        )


# =============================================================================
# Navigation principale
# =============================================================================
def main():
    init_session_state()

    st.sidebar.title("MiniKu")
    st.sidebar.caption("Mini-Dataiku - MVP No-Code")

    page = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Atelier de Data Prep",
            "2. Studio de Modélisation",
            "3. Fabrique de Prédictions",
        ],
    )

    # Indicateurs d'état dans la sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("État de la session")
    st.sidebar.write(
        "- Dataset chargé : "
        + (":white_check_mark:" if st.session_state.clean_df is not None else ":x:")
    )
    st.sidebar.write(
        "- Modèle entraîné : "
        + (":white_check_mark:" if st.session_state.model_trained else ":x:")
    )

    if page.startswith("1"):
        page_data_prep()
    elif page.startswith("2"):
        page_modeling()
    else:
        page_predict()


if __name__ == "__main__":
    main()
