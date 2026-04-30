"""
MiniKu - Mini-Dataiku
Application Streamlit No-Code pour la Data Science.

Modules :
  1. Atelier de Data Prep     (upload + profilage + recettes + undo)
  2. Studio de Modelisation   (PyCaret AutoML)
  3. Fabrique de Predictions  (apply model + download CSV)

Lancement :
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


# =============================================================================
# Configuration
# =============================================================================
st.set_page_config(
    page_title="MiniKu",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "miniku_best_model"           # PyCaret ajoute le suffixe .pkl
PREVIEW_ROWS = 200                         # lignes max affichees dans st.dataframe
SAMPLE_THRESHOLD = 100_000                 # au-dela : sampling pour la preview
MAX_ROWS = 1_000_000                       # garde-fou : refus au-dela
HISTORY_LIMIT = 20                         # taille du stack d'undo


# =============================================================================
# Theme Dataiku-like (CSS)
# =============================================================================
CUSTOM_CSS = """
<style>
    /* Masquer la chrome Streamlit par defaut */
    #MainMenu, footer, header {visibility: hidden;}

    /* Conteneur principal plus aere */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    /* Sidebar sombre */
    section[data-testid="stSidebar"] {
        background: #1c2128;
        border-right: 1px solid #2d333b;
    }
    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2d333b;
    }

    /* Titres */
    h1 {
        color: #1c2128;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    h2, h3 {
        color: #2d333b;
        font-weight: 600;
    }

    /* Boutons */
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        border: 1px solid #d0d7de;
        transition: all 0.12s;
    }
    .stButton > button:hover {
        border-color: #2563eb;
        color: #2563eb;
    }
    .stButton > button[kind="primary"] {
        background: #2563eb;
        color: white;
        border: none;
    }
    .stButton > button[kind="primary"]:hover {
        background: #1d4ed8;
        color: white;
    }
    .stDownloadButton > button {
        border-radius: 6px;
        font-weight: 500;
    }

    /* Metriques */
    [data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
    }

    /* Pipeline steps dans la sidebar */
    .miniku-step {
        display: flex;
        align-items: center;
        padding: 8px 10px;
        margin: 4px 0;
        border-radius: 6px;
        font-size: 14px;
    }
    .miniku-step.active {
        background: rgba(37, 99, 235, 0.18);
        border-left: 3px solid #60a5fa;
        padding-left: 7px;
    }
    .miniku-step.done {
        color: #34d399 !important;
    }
    .miniku-step.pending {
        opacity: 0.55;
    }
    .miniku-step .icon {
        margin-right: 8px;
        font-weight: bold;
        width: 16px;
        display: inline-block;
    }

    /* Badge */
    .miniku-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 600;
        background: #dbeafe;
        color: #1e40af;
        margin-left: 6px;
    }

    /* Recipe item */
    .recipe-item {
        background: #f8fafc;
        border-left: 3px solid #2563eb;
        padding: 6px 12px;
        margin: 4px 0;
        font-size: 13px;
        border-radius: 4px;
        font-family: ui-monospace, Menlo, monospace;
    }

    /* Tabs plus discrets */
    button[role="tab"] {
        font-weight: 500;
    }
</style>
"""


# =============================================================================
# Etat de session
# =============================================================================
def init_state() -> None:
    defaults = {
        "raw_df": None,                # DataFrame original
        "clean_df": None,              # DataFrame courant (apres recettes)
        "history": [],                 # stack d'undo : list[(label, df)]
        "recipes": [],                 # liste de descriptions appliquees
        "source_info": None,           # dict : filename, rows, cols, size_kb
        "task_type": None,             # "Classification" / "Regression"
        "target_column": None,
        "leaderboard": None,
        "model_trained": False,
        "best_model_name": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def push_history(label: str) -> None:
    """Sauvegarde l'etat avant transformation (pour undo)."""
    if st.session_state.clean_df is not None:
        st.session_state.history.append((label, st.session_state.clean_df.copy()))
        if len(st.session_state.history) > HISTORY_LIMIT:
            st.session_state.history.pop(0)


def add_recipe(description: str) -> None:
    """Ajoute une etape a la liste des recettes appliquees."""
    st.session_state.recipes.append(description)


def undo_last() -> Optional[str]:
    """Revient en arriere d'un cran. Retourne le label de l'etape annulee."""
    if not st.session_state.history:
        return None
    label, df = st.session_state.history.pop()
    st.session_state.clean_df = df
    if st.session_state.recipes:
        st.session_state.recipes.pop()
    return label


def reset_all() -> None:
    """Reinitialise totalement la session (sauf raw_df)."""
    st.session_state.clean_df = (
        st.session_state.raw_df.copy() if st.session_state.raw_df is not None else None
    )
    st.session_state.history = []
    st.session_state.recipes = []
    st.session_state.task_type = None
    st.session_state.target_column = None
    st.session_state.leaderboard = None
    st.session_state.model_trained = False
    st.session_state.best_model_name = None


# =============================================================================
# Chargement de fichier (cache pour eviter de re-lire a chaque rerun)
# =============================================================================
@st.cache_data(show_spinner=False)
def read_csv_smart(content: bytes, sep: str, encoding: str) -> pd.DataFrame:
    if sep == "auto":
        # python engine + sep=None -> sniff
        return pd.read_csv(io.BytesIO(content), sep=None, engine="python", encoding=encoding)
    return pd.read_csv(io.BytesIO(content), sep=sep, encoding=encoding)


@st.cache_data(show_spinner=False)
def read_excel_file(content: bytes, sheet_name) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def read_parquet_file(content: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(content))


# =============================================================================
# Sidebar : pipeline visuel + navigation
# =============================================================================
def render_sidebar() -> str:
    st.sidebar.markdown("# :bar_chart: MiniKu")
    st.sidebar.caption("Mini-Dataiku - No-Code Data Science")
    st.sidebar.markdown("---")

    # Etat des etapes
    has_data = st.session_state.clean_df is not None
    has_model = st.session_state.model_trained

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        options=[
            "1. Atelier de Data Prep",
            "2. Studio de Modelisation",
            "3. Fabrique de Predictions",
        ],
        label_visibility="collapsed",
    )

    # Pipeline visuel (unicode direct car les shortcodes :emoji: ne sont pas
    # interpretes dans du HTML brut).
    st.sidebar.markdown("### Pipeline")
    steps = [
        ("Donnees chargees", has_data, page.startswith("1")),
        ("Modele entraine", has_model, page.startswith("2")),
        ("Predictions", False, page.startswith("3")),
    ]
    for label, done, active in steps:
        if active:
            cls, icon = "active", "●"
        elif done:
            cls, icon = "done", "✓"
        else:
            cls, icon = "pending", "○"
        st.sidebar.markdown(
            f'<div class="miniku-step {cls}"><span class="icon">{icon}</span>{label}</div>',
            unsafe_allow_html=True,
        )

    # Infos session
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Session")
    if st.session_state.source_info:
        info = st.session_state.source_info
        st.sidebar.caption(f":page_facing_up: **{info['filename']}**")
        st.sidebar.caption(
            f"{info['rows']:,} lignes - {info['cols']} colonnes - "
            f"{info['size_kb']:.0f} KB"
        )
    if st.session_state.clean_df is not None:
        df = st.session_state.clean_df
        st.sidebar.caption(
            f":scissors: Apres prep : **{df.shape[0]:,}** x **{df.shape[1]}**"
        )
    if st.session_state.target_column:
        st.sidebar.caption(
            f":dart: Cible : **{st.session_state.target_column}** "
            f"({st.session_state.task_type})"
        )

    if st.sidebar.button("Reinitialiser la session", width="stretch"):
        reset_all()
        st.rerun()

    return page


# =============================================================================
# Module 1 : Atelier de Data Prep
# =============================================================================
def page_data_prep() -> None:
    st.title("Atelier de Data Prep")
    st.caption("Charge un dataset, profile-le, applique des recettes de nettoyage.")

    tab_source, tab_explore, tab_recipes = st.tabs(
        [":inbox_tray: Source de donnees", ":mag: Exploration", ":scroll: Recettes appliquees"]
    )

    # --------- Onglet 1 : source ---------
    with tab_source:
        render_source_tab()

    # --------- Onglet 2 : exploration + recettes ---------
    with tab_explore:
        if st.session_state.clean_df is None:
            st.info("Charge un fichier dans l'onglet **Source de donnees** pour commencer.")
        else:
            render_explore_tab()

    # --------- Onglet 3 : log des recettes ---------
    with tab_recipes:
        render_recipes_tab()


def render_source_tab() -> None:
    """Upload + lecture intelligente du fichier."""
    col_upload, col_opts = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "Fichier de donnees",
            type=["csv", "xlsx", "xls", "parquet"],
            help="Formats supportes : CSV, Excel, Parquet. Limite : 1 000 000 lignes.",
        )

    with col_opts:
        sep = st.selectbox(
            "Separateur CSV", ["auto", ",", ";", "\\t", "|"],
            help="`auto` tente de detecter automatiquement.",
        )
        encoding = st.selectbox("Encodage", ["utf-8", "latin-1", "cp1252", "utf-16"])

    if uploaded is None:
        st.info(":information_source: En attente d'un fichier...")
        return

    content = uploaded.getvalue()
    size_kb = len(content) / 1024

    try:
        with st.spinner(f"Lecture de {uploaded.name}..."):
            ext = uploaded.name.split(".")[-1].lower()
            if ext == "csv":
                sep_val = "\t" if sep == "\\t" else sep
                df = read_csv_smart(content, sep_val, encoding)
            elif ext in ("xlsx", "xls"):
                df = read_excel_file(content, sheet_name=0)
            elif ext == "parquet":
                df = read_parquet_file(content)
            else:
                st.error(f"Extension non supportee : {ext}")
                return
    except Exception as e:
        st.error(f"Erreur a la lecture : {e}")
        return

    if df.shape[0] > MAX_ROWS:
        st.error(
            f"Fichier trop volumineux : {df.shape[0]:,} lignes "
            f"(limite {MAX_ROWS:,}). Reduis avec un sampling externe."
        )
        return

    # Premier chargement OU remplacement
    if (
        st.session_state.raw_df is None
        or st.session_state.source_info is None
        or st.session_state.source_info.get("filename") != uploaded.name
    ):
        st.session_state.raw_df = df.copy()
        st.session_state.clean_df = df.copy()
        st.session_state.history = []
        st.session_state.recipes = []
        st.session_state.source_info = {
            "filename": uploaded.name,
            "rows": df.shape[0],
            "cols": df.shape[1],
            "size_kb": size_kb,
        }

    # Resume
    st.success(f":white_check_mark: **{uploaded.name}** chargee.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{df.shape[0]:,}")
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Manquantes", f"{int(df.isna().sum().sum()):,}")
    c4.metric("Taille", f"{size_kb:.0f} KB")

    if df.shape[0] > SAMPLE_THRESHOLD:
        st.warning(
            f":warning: Dataset volumineux ({df.shape[0]:,} lignes). "
            f"Les apercus afficheront un echantillon de {PREVIEW_ROWS} lignes ; "
            f"toutes les operations restent appliquees sur le dataset complet."
        )

    st.subheader("Apercu")
    st.dataframe(df.head(PREVIEW_ROWS), width="stretch", height=350)


def render_explore_tab() -> None:
    """Profilage colonnes + recettes de nettoyage."""
    df = st.session_state.clean_df

    # Bandeau metriques
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes", f"{df.shape[0]:,}")
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Manquantes", f"{int(df.isna().sum().sum()):,}")
    mem_mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    c4.metric("Memoire", f"{mem_mb:.1f} MB")

    # Apercu (sample si volumineux)
    st.subheader("Apercu du dataset courant")
    if df.shape[0] > SAMPLE_THRESHOLD:
        preview = df.sample(min(PREVIEW_ROWS, df.shape[0]), random_state=0)
        st.caption(
            f":information_source: Echantillon aleatoire de {len(preview)} lignes "
            f"(dataset complet : {df.shape[0]:,})."
        )
    else:
        preview = df.head(PREVIEW_ROWS)
    st.dataframe(preview, width="stretch", height=320)

    # Profilage colonnes
    st.subheader("Profil des colonnes")
    profile = build_column_profile(df)
    st.dataframe(profile, width="stretch", height=280)

    st.markdown("---")

    # Actions rapides : undo + telechargement
    col_a, col_b, col_c = st.columns([1, 1, 2])
    with col_a:
        undo_label = (
            f"Annuler : {st.session_state.history[-1][0]}"
            if st.session_state.history else "Rien a annuler"
        )
        if st.button(undo_label, disabled=not st.session_state.history,
                     width="stretch"):
            label = undo_last()
            st.success(f"Annule : {label}")
            st.rerun()
    with col_b:
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger CSV nettoye", csv_bytes,
            file_name="dataset_clean.csv", mime="text/csv",
            width="stretch",
        )

    st.markdown("---")
    st.subheader("Recettes de nettoyage")

    # Sous-onglets pour eviter une page a rallonge
    sub_drop, sub_nan, sub_encode, sub_filter = st.tabs(
        ["Colonnes", "Valeurs manquantes", "Encodage", "Filtrage"]
    )

    with sub_drop:
        recipe_drop_columns(df)
    with sub_nan:
        recipe_handle_nan(df)
    with sub_encode:
        recipe_encode(df)
    with sub_filter:
        recipe_filter(df)


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Construit un dataframe de profilage par colonne.

    Toutes les colonnes statistiques sont stockees en string pour eviter
    le melange float/str qui casse la serialisation Arrow utilisee par
    st.dataframe.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        sample = s.dropna()
        rows.append({
            "Colonne": str(col),
            "Type": str(s.dtype),
            "Manquantes": int(s.isna().sum()),
            "% manquantes": round(float(s.isna().mean() * 100), 1),
            "Uniques": int(s.nunique(dropna=True)),
            "Min": _safe_stat(s, "min"),
            "Max": _safe_stat(s, "max"),
            "Moyenne": _safe_stat(s, "mean"),
            "Exemple": str(sample.iloc[0]) if len(sample) else "—",
        })
    return pd.DataFrame(rows)


def _safe_stat(s: pd.Series, op: str) -> str:
    """Retourne une stat numerique formatee en string, ou '—'."""
    if not pd.api.types.is_numeric_dtype(s):
        return "—"
    try:
        val = getattr(s, op)()
        if pd.isna(val):
            return "—"
        return f"{float(val):.3g}"
    except Exception:
        return "—"


def recipe_drop_columns(df: pd.DataFrame) -> None:
    cols = st.multiselect("Colonnes a supprimer", options=list(df.columns))
    if st.button("Appliquer la suppression", type="primary",
                 disabled=not cols, key="btn_drop"):
        push_history(f"drop {len(cols)} col(s)")
        st.session_state.clean_df = df.drop(columns=cols)
        add_recipe(f"Suppression : {', '.join(cols)}")
        st.success(f"Colonnes supprimees : {', '.join(cols)}")
        st.rerun()


def recipe_handle_nan(df: pd.DataFrame) -> None:
    strategy = st.selectbox(
        "Strategie",
        [
            "Supprimer les lignes contenant des NaN",
            "Remplacer par la moyenne (numerique)",
            "Remplacer par la mediane (numerique)",
            "Remplacer par zero (numerique)",
            "Remplacer par 'inconnu' (textuel)",
        ],
    )
    scope = st.multiselect(
        "Limiter aux colonnes (vide = toutes)",
        options=list(df.columns),
    )
    if st.button("Appliquer la strategie", type="primary", key="btn_nan"):
        push_history("traitement NaN")
        df_w = df.copy()
        target_cols = scope if scope else list(df_w.columns)
        if strategy.startswith("Supprimer"):
            before = len(df_w)
            df_w = df_w.dropna(subset=target_cols)
            recipe_msg = f"Drop NaN sur {len(target_cols)} col(s) ({before - len(df_w)} lignes retirees)"
        elif "moyenne" in strategy:
            num = df_w[target_cols].select_dtypes(include=np.number).columns
            df_w[num] = df_w[num].fillna(df_w[num].mean())
            recipe_msg = f"NaN -> moyenne ({len(num)} col)"
        elif "mediane" in strategy:
            num = df_w[target_cols].select_dtypes(include=np.number).columns
            df_w[num] = df_w[num].fillna(df_w[num].median())
            recipe_msg = f"NaN -> mediane ({len(num)} col)"
        elif "zero" in strategy:
            num = df_w[target_cols].select_dtypes(include=np.number).columns
            df_w[num] = df_w[num].fillna(0)
            recipe_msg = f"NaN -> 0 ({len(num)} col)"
        else:  # inconnu
            txt = df_w[target_cols].select_dtypes(include=["object", "category"]).columns
            for c in txt:
                df_w[c] = df_w[c].fillna("inconnu")
            recipe_msg = f"NaN -> 'inconnu' ({len(txt)} col)"
        st.session_state.clean_df = df_w
        add_recipe(recipe_msg)
        st.success(recipe_msg)
        st.rerun()


def recipe_encode(df: pd.DataFrame) -> None:
    text_cols = list(df.select_dtypes(include=["object", "category"]).columns)
    if not text_cols:
        st.info("Aucune colonne textuelle a encoder.")
        return
    cols = st.multiselect("Colonnes textuelles a encoder", options=text_cols)
    method = st.radio("Methode", ["Label Encoding", "One-Hot Encoding"], horizontal=True)
    if st.button("Appliquer l'encodage", type="primary",
                 disabled=not cols, key="btn_enc"):
        push_history(f"encode {method}")
        df_w = df.copy()
        if method == "Label Encoding":
            for c in cols:
                df_w[c] = df_w[c].astype("category").cat.codes
            msg = f"Label-encode : {', '.join(cols)}"
        else:
            df_w = pd.get_dummies(df_w, columns=cols)
            msg = f"One-hot : {', '.join(cols)}"
        st.session_state.clean_df = df_w
        add_recipe(msg)
        st.success(msg)
        st.rerun()


def recipe_filter(df: pd.DataFrame) -> None:
    """Filtrage simple : sur une colonne numerique, garder un intervalle."""
    num_cols = list(df.select_dtypes(include=np.number).columns)
    if not num_cols:
        st.info("Aucune colonne numerique a filtrer.")
        return
    col = st.selectbox("Colonne", options=num_cols)
    if col:
        s = df[col].dropna()
        if s.empty:
            st.info("Colonne vide.")
            return
        lo, hi = float(s.min()), float(s.max())
        rng = st.slider(f"Garder les lignes ou {col} est dans :",
                        lo, hi, (lo, hi))
        if st.button("Appliquer le filtre", type="primary", key="btn_filter"):
            push_history(f"filter {col}")
            mask = df[col].between(rng[0], rng[1])
            df_w = df[mask].copy()
            msg = f"Filtre : {col} in [{rng[0]:.3g}, {rng[1]:.3g}] ({len(df_w):,} lignes)"
            st.session_state.clean_df = df_w
            add_recipe(msg)
            st.success(msg)
            st.rerun()


def render_recipes_tab() -> None:
    if not st.session_state.recipes:
        st.info("Aucune recette appliquee pour le moment.")
        return
    st.markdown(f"**{len(st.session_state.recipes)}** etape(s) appliquee(s) :")
    for i, r in enumerate(st.session_state.recipes, start=1):
        st.markdown(
            f'<div class="recipe-item"><b>{i}.</b> {r}</div>',
            unsafe_allow_html=True,
        )


# =============================================================================
# Module 2 : Studio de Modelisation
# =============================================================================
def page_modeling() -> None:
    st.title("Studio de Modelisation")
    st.caption("AutoML via PyCaret : selectionne la cible, lance la comparaison.")

    if st.session_state.clean_df is None:
        st.warning(
            ":warning: Aucun dataset n'est disponible. "
            "Commence par l'**Atelier de Data Prep**."
        )
        return

    df = st.session_state.clean_df

    # Bandeau
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", f"{df.shape[0]:,}")
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Manquantes", f"{int(df.isna().sum().sum()):,}")

    # Validation prealable
    issues = validate_for_modeling(df)
    if issues:
        for issue in issues:
            st.error(f":x: {issue}")
        return

    # Configuration
    st.subheader("Configuration de l'experience")
    col_t, col_k = st.columns(2)
    with col_t:
        target = st.selectbox("Colonne cible", options=list(df.columns))
    with col_k:
        task = st.radio("Type de tache", ["Classification", "Regression"],
                        horizontal=True)

    st.session_state.target_column = target
    st.session_state.task_type = task

    # Apercu cible
    with st.expander("Apercu de la colonne cible"):
        s = df[target]
        if pd.api.types.is_numeric_dtype(s) and task == "Regression":
            st.write(s.describe())
        else:
            vc = s.value_counts(dropna=False).head(20)
            st.dataframe(vc.rename("count"), width="stretch")

    # Avertissement si pas le bon type
    if task == "Classification" and df[target].nunique() > 50:
        st.warning(
            f":warning: La cible a {df[target].nunique()} valeurs uniques. "
            f"Es-tu sur que c'est de la **classification** et pas de la regression ?"
        )

    # Lancement
    st.markdown("---")
    if st.button(":rocket: Lancer l'AutoML", type="primary",
                 width="stretch"):
        run_automl(df, target, task)

    # Resultats
    if st.session_state.leaderboard is not None:
        st.markdown("---")
        st.subheader("Leaderboard")
        if st.session_state.best_model_name:
            st.success(
                f":trophy: Meilleur modele : **{st.session_state.best_model_name}**"
            )
        st.dataframe(st.session_state.leaderboard, width="stretch")

        if os.path.exists(f"{MODEL_PATH}.pkl"):
            with open(f"{MODEL_PATH}.pkl", "rb") as f:
                st.download_button(
                    "Telecharger le modele gagnant (.pkl)",
                    data=f.read(),
                    file_name="miniku_best_model.pkl",
                    mime="application/octet-stream",
                )


def validate_for_modeling(df: pd.DataFrame) -> list[str]:
    """Detection precoce des problemes avant PyCaret."""
    issues: list[str] = []
    if df.shape[0] < 20:
        issues.append(f"Trop peu de lignes ({df.shape[0]}, min 20).")
    if df.shape[1] < 2:
        issues.append("Au moins 2 colonnes (1 feature + 1 target) sont requises.")
    return issues


def run_automl(df: pd.DataFrame, target: str, task: str) -> None:
    if df[target].isna().any():
        st.error("La colonne cible contient des NaN. Nettoie-la dans l'Atelier.")
        return

    with st.spinner("Entrainement (cela peut prendre quelques minutes)..."):
        t0 = time.time()
        try:
            if task == "Classification":
                from pycaret.classification import (
                    setup, compare_models, pull, save_model,
                )
            else:
                from pycaret.regression import (
                    setup, compare_models, pull, save_model,
                )

            setup(data=df, target=target, session_id=42, verbose=False, html=False)
            best = compare_models()
            leaderboard = pull()
            save_model(best, MODEL_PATH)

            st.session_state.leaderboard = leaderboard
            st.session_state.model_trained = True
            st.session_state.best_model_name = leaderboard.index[0] if len(leaderboard) else None
        except Exception as e:
            st.error(f":x: Erreur PyCaret : {e}")
            return

    st.success(f":white_check_mark: Entrainement termine en {time.time() - t0:.1f}s.")


# =============================================================================
# Module 3 : Fabrique de Predictions
# =============================================================================
def page_predict() -> None:
    st.title("Fabrique de Predictions")
    st.caption("Applique le modele entraine a un nouveau jeu de donnees.")

    if not st.session_state.model_trained or not os.path.exists(f"{MODEL_PATH}.pkl"):
        st.warning(
            ":warning: Aucun modele entraine. "
            "Passe par le **Studio de Modelisation** d'abord."
        )
        return

    st.info(
        f":dart: Modele charge : **{st.session_state.best_model_name or 'inconnu'}** "
        f"({st.session_state.task_type})"
    )

    uploaded = st.file_uploader("Fichier CSV a predire", type=["csv"])
    if uploaded is None:
        return

    try:
        new_df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Lecture impossible : {e}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Lignes", f"{new_df.shape[0]:,}")
    c2.metric("Colonnes", new_df.shape[1])

    st.subheader("Apercu")
    st.dataframe(new_df.head(PREVIEW_ROWS), width="stretch", height=300)

    if st.button(":sparkles: Generer les predictions", type="primary",
                 width="stretch"):
        with st.spinner("Calcul..."):
            try:
                if st.session_state.task_type == "Classification":
                    from pycaret.classification import load_model, predict_model
                else:
                    from pycaret.regression import load_model, predict_model
                model = load_model(MODEL_PATH)
                predictions = predict_model(model, data=new_df)
            except Exception as e:
                st.error(f":x: Erreur de prediction : {e}")
                return

        st.success(":white_check_mark: Predictions generees.")
        st.subheader("Resultats")
        st.dataframe(predictions.head(PREVIEW_ROWS), width="stretch", height=350)

        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Telecharger les predictions (CSV)",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            width="stretch",
        )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    init_state()
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    page = render_sidebar()

    if page.startswith("1"):
        page_data_prep()
    elif page.startswith("2"):
        page_modeling()
    else:
        page_predict()


if __name__ == "__main__":
    main()
