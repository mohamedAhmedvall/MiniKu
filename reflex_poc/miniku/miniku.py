"""
MiniKu - POC Reflex (Data Prep uniquement).

Demontre :
- Layout type Dataiku (sidebar sombre + contenu blanc)
- Upload + apercu d'un CSV
- Metriques globales (lignes, colonnes, manquantes)
- Recettes : suppression de colonnes, gestion NaN
- Etat persistant entre interactions (pas de rerun complet)

Lancement :
    cd reflex_poc
    reflex init    # premiere fois seulement
    reflex run

Frontend : http://localhost:3000
Backend  : http://localhost:8000
"""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import reflex as rx


# =============================================================================
# Etat
# =============================================================================
class State(rx.State):
    """Etat global de l'application. Persistant entre interactions."""

    # Nom du fichier charge
    file_name: str = ""
    # Dataset courant serialisable (liste de dicts pour le rendu)
    preview_rows: list[dict[str, Any]] = []
    columns: list[str] = []
    n_rows: int = 0
    n_cols: int = 0
    n_missing: int = 0
    is_loaded: bool = False

    # Selections recettes
    cols_to_drop: list[str] = []
    nan_strategy: str = "drop"

    # Log des recettes appliquees
    recipes_log: list[str] = []

    # Internal : on garde le DataFrame complet en attribut backend (non serialise
    # au frontend, juste utilise pour les operations).
    _df: pd.DataFrame = pd.DataFrame()

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        f = files[0]
        data = await f.read()
        try:
            df = pd.read_csv(io.BytesIO(data))
        except Exception:
            # fallback : sniff separateur
            df = pd.read_csv(io.BytesIO(data), sep=None, engine="python")
        self._df = df
        self.file_name = f.filename or "upload.csv"
        self.recipes_log = []
        self.cols_to_drop = []
        self._refresh_preview()

    def _refresh_preview(self):
        df = self._df
        self.preview_rows = df.head(50).fillna("").astype(str).to_dict("records")
        self.columns = list(df.columns)
        self.n_rows = int(df.shape[0])
        self.n_cols = int(df.shape[1])
        self.n_missing = int(df.isna().sum().sum())
        self.is_loaded = True

    @rx.event
    def toggle_drop_col(self, col: str):
        """Coche/decoche une colonne dans la liste a supprimer."""
        if col in self.cols_to_drop:
            self.cols_to_drop = [c for c in self.cols_to_drop if c != col]
        else:
            self.cols_to_drop = self.cols_to_drop + [col]

    @rx.event
    def apply_drop(self):
        if not self.cols_to_drop:
            return
        cols = list(self.cols_to_drop)
        self._df = self._df.drop(columns=cols)
        self.recipes_log = self.recipes_log + [f"Drop : {', '.join(cols)}"]
        self.cols_to_drop = []
        self._refresh_preview()

    @rx.event
    def set_nan_strategy(self, value: str):
        self.nan_strategy = value

    @rx.event
    def apply_nan(self):
        df = self._df.copy()
        s = self.nan_strategy
        if s == "drop":
            before = len(df)
            df = df.dropna()
            label = f"Drop NaN ({before - len(df)} lignes)"
        elif s == "mean":
            num = df.select_dtypes(include=np.number).columns
            df[num] = df[num].fillna(df[num].mean())
            label = f"NaN -> moyenne ({len(num)} col)"
        elif s == "median":
            num = df.select_dtypes(include=np.number).columns
            df[num] = df[num].fillna(df[num].median())
            label = f"NaN -> mediane ({len(num)} col)"
        elif s == "zero":
            num = df.select_dtypes(include=np.number).columns
            df[num] = df[num].fillna(0)
            label = f"NaN -> 0 ({len(num)} col)"
        else:
            return
        self._df = df
        self.recipes_log = self.recipes_log + [label]
        self._refresh_preview()


# =============================================================================
# Composants UI
# =============================================================================
SIDEBAR_BG = "#1c2128"
SIDEBAR_TEXT = "#e6edf3"
ACCENT = "#2563eb"
CARD_BORDER = "#e2e8f0"


def metric_card(label: str, value) -> rx.Component:
    return rx.box(
        rx.text(label, color="gray", size="2"),
        rx.text(value, size="6", weight="bold"),
        padding="1em 1.2em",
        border=f"1px solid {CARD_BORDER}",
        border_radius="8px",
        bg="white",
        flex="1",
    )


def sidebar_step(label: str, active: bool = False, done: bool = False) -> rx.Component:
    if active:
        icon, color = "●", "#60a5fa"
        bg = "rgba(37,99,235,0.18)"
    elif done:
        icon, color = "✓", "#34d399"
        bg = "transparent"
    else:
        icon, color = "○", "#6b7280"
        bg = "transparent"
    return rx.hstack(
        rx.text(icon, color=color, weight="bold", width="20px"),
        rx.text(label, color=SIDEBAR_TEXT),
        bg=bg,
        padding="6px 10px",
        border_radius="6px",
        width="100%",
    )


def sidebar() -> rx.Component:
    return rx.vstack(
        rx.heading("MiniKu", color="white", size="6"),
        rx.text("Mini-Dataiku - POC Reflex", color="#9ca3af", size="2"),
        rx.divider(border_color="#2d333b", margin_y="1em"),
        rx.text("Pipeline", color="white", weight="bold", size="3"),
        sidebar_step("Donnees chargees", active=True),
        sidebar_step("Modele entraine"),
        sidebar_step("Predictions"),
        rx.divider(border_color="#2d333b", margin_y="1em"),
        rx.cond(
            State.is_loaded,
            rx.vstack(
                rx.text("Session", color="white", weight="bold", size="3"),
                rx.text(State.file_name, color="#9ca3af", size="2"),
                rx.text(
                    State.n_rows.to_string() + " lignes - "
                    + State.n_cols.to_string() + " col",
                    color="#9ca3af", size="2",
                ),
                spacing="1",
                align_items="start",
                width="100%",
            ),
            rx.text("Aucun dataset", color="#6b7280", size="2"),
        ),
        bg=SIDEBAR_BG,
        color=SIDEBAR_TEXT,
        padding="1.5em",
        width="280px",
        height="100vh",
        spacing="2",
        align_items="start",
        overflow_y="auto",
    )


def upload_zone() -> rx.Component:
    return rx.vstack(
        rx.upload(
            rx.vstack(
                rx.icon("upload", size=32, color=ACCENT),
                rx.text("Glisse-depose un CSV ici ou clique"),
                rx.text("CSV jusqu'a 200 MB", size="1", color="gray"),
                spacing="2",
                align="center",
            ),
            id="upload1",
            border="2px dashed #cbd5e1",
            border_radius="8px",
            padding="2.5em",
            width="100%",
            bg="#f8fafc",
            on_drop=State.handle_upload(rx.upload_files(upload_id="upload1")),
            accept={"text/csv": [".csv"]},
        ),
        width="100%",
    )


def preview_table() -> rx.Component:
    return rx.box(
        rx.data_table(
            data=State.preview_rows,
            columns=State.columns,
            pagination=True,
            search=True,
            sort=True,
        ),
        border=f"1px solid {CARD_BORDER}",
        border_radius="8px",
        padding="0.5em",
        bg="white",
        width="100%",
    )


def column_checkbox(col: str) -> rx.Component:
    return rx.checkbox(
        col,
        on_change=lambda _: State.toggle_drop_col(col),
        size="2",
    )


def recipe_drop_columns() -> rx.Component:
    return rx.vstack(
        rx.text("Coche les colonnes a supprimer :", weight="medium"),
        rx.box(
            rx.foreach(State.columns, column_checkbox),
            display="grid",
            grid_template_columns="repeat(3, 1fr)",
            gap="0.5em",
            width="100%",
        ),
        rx.button(
            "Supprimer les colonnes selectionnees",
            on_click=State.apply_drop,
            color_scheme="blue",
            disabled=State.cols_to_drop.length() == 0,
        ),
        spacing="3",
        align_items="start",
        width="100%",
    )


def recipe_nan() -> rx.Component:
    return rx.vstack(
        rx.text("Strategie pour les valeurs manquantes :", weight="medium"),
        rx.radio_group.root(
            rx.vstack(
                rx.radio_group.item("drop", children=rx.text("Supprimer les lignes contenant des NaN")),
                rx.radio_group.item("mean", children=rx.text("Remplacer par la moyenne (numerique)")),
                rx.radio_group.item("median", children=rx.text("Remplacer par la mediane (numerique)")),
                rx.radio_group.item("zero", children=rx.text("Remplacer par 0 (numerique)")),
                spacing="2",
                align_items="start",
            ),
            value=State.nan_strategy,
            on_change=State.set_nan_strategy,
        ),
        rx.button("Appliquer", on_click=State.apply_nan, color_scheme="blue"),
        spacing="3",
        align_items="start",
        width="100%",
    )


def recipes_section() -> rx.Component:
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger("Colonnes", value="cols"),
            rx.tabs.trigger("Valeurs manquantes", value="nan"),
            rx.tabs.trigger("Historique", value="log"),
        ),
        rx.tabs.content(
            rx.box(recipe_drop_columns(), padding_y="1em"),
            value="cols",
        ),
        rx.tabs.content(
            rx.box(recipe_nan(), padding_y="1em"),
            value="nan",
        ),
        rx.tabs.content(
            rx.box(
                rx.cond(
                    State.recipes_log.length() == 0,
                    rx.text("Aucune recette appliquee.", color="gray"),
                    rx.foreach(
                        State.recipes_log,
                        lambda r, i: rx.box(
                            rx.text(f"{i + 1}. {r}", font_family="monospace", size="2"),
                            border_left=f"3px solid {ACCENT}",
                            padding="6px 12px",
                            bg="#f8fafc",
                            border_radius="4px",
                            margin_y="4px",
                        ),
                    ),
                ),
                padding_y="1em",
            ),
            value="log",
        ),
        default_value="cols",
        width="100%",
    )


def main_panel() -> rx.Component:
    return rx.vstack(
        rx.heading("Atelier de Data Prep", size="7"),
        rx.text(
            "POC Reflex - upload, profilage, recettes (avec etat persistant).",
            color="gray",
        ),
        rx.divider(),
        upload_zone(),
        rx.cond(
            State.is_loaded,
            rx.vstack(
                rx.heading(rx.text(State.file_name), size="4"),
                rx.hstack(
                    metric_card("Lignes", State.n_rows),
                    metric_card("Colonnes", State.n_cols),
                    metric_card("Manquantes", State.n_missing),
                    spacing="3",
                    width="100%",
                ),
                rx.heading("Apercu (50 premieres lignes)", size="4"),
                preview_table(),
                rx.heading("Recettes", size="4"),
                recipes_section(),
                spacing="4",
                width="100%",
                align_items="start",
            ),
        ),
        padding="2em 3em",
        spacing="4",
        width="100%",
        height="100vh",
        overflow_y="auto",
        align_items="start",
        bg="#f8fafc",
    )


def index() -> rx.Component:
    return rx.hstack(
        sidebar(),
        main_panel(),
        spacing="0",
        align_items="stretch",
        height="100vh",
        width="100%",
    )


# =============================================================================
# App
# =============================================================================
app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="blue",
        radius="medium",
    ),
)
app.add_page(index, title="MiniKu - Data Prep")
