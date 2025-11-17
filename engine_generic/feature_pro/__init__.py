"""
feature_pro
-----------
Librería para operaciones de EDA, selección de variables y utilidades
reutilizables. Organización:
- common: utilidades genéricas y estables (utils)
- selection: funciones de selección de variables
"""

from .common.utils import (
    read_dataset,
    summarize_missing,
    detect_column_types,
    impute_missing,
    winsorize_by_percentile,
    winsorize_by_iqr,
    compute_correlation_matrix,
)

from .selection.numeric_selection import (
    compute_pca,
    compute_psi,
    coefficient_of_variation,
    pca_lda_importance,
    select_by_correlation_graph,
)
from .selection.categorical_selection import (
    count_unique_categorical,
    categorical_cumulative_frequency,
    calculate_woe_iv,
)
from .segmentation.segmentation import (
    build_regression_tree_segments,
    compute_tree_feature_importance,
    fit_numeric_preprocessor,
    transform_with_preprocessor,
    save_segmentation_pipeline,
    load_segmentation_pipeline,
    apply_segmentation_pipeline,
)

__all__ = [
    # common utils
    "read_dataset",
    "summarize_missing",
    "detect_column_types",
    "impute_missing",
    "winsorize_by_percentile",
    "winsorize_by_iqr",
    "compute_correlation_matrix",
    "compute_pca",
    "compute_psi",
    "coefficient_of_variation",
    "count_unique_categorical",
    "categorical_cumulative_frequency",
    "calculate_woe_iv",
    # segmentation
    "build_regression_tree_segments",
    "compute_tree_feature_importance",
    "fit_numeric_preprocessor",
    "transform_with_preprocessor",
    "save_segmentation_pipeline",
    "load_segmentation_pipeline",
    "apply_segmentation_pipeline",
    # selection
    "pca_lda_importance",
    "select_by_correlation_graph",
]


