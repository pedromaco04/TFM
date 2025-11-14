"""
engine_gen
-----------
Librería genérica para operaciones de EDA, selección de variables y utilidades
reutilizables. Pensada para integrarse con pipelines como `TFM_EDA.py` y
`engine_eda.py`, con énfasis en:
- Funciones genéricas y reutilizables (utils)
- Funciones de selección y reducción (vars_selection)
"""

from .utils import (
    summarize_missing,
    detect_column_types,
    impute_missing,
    winsorize_by_percentile,
    winsorize_by_iqr,
    coefficient_of_variation,
    compute_pca,
    compute_correlation_matrix,
)

from .vars_selection import (
    pca_lda_importance,
    select_by_correlation_graph,
)

__all__ = [
    # utils
    "summarize_missing",
    "detect_column_types",
    "impute_missing",
    "winsorize_by_percentile",
    "winsorize_by_iqr",
    "coefficient_of_variation",
    "compute_pca",
    "compute_correlation_matrix",
    # selection
    "pca_lda_importance",
    "select_by_correlation_graph",
]


