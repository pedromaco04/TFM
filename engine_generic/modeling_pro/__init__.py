"""
modeling_pro
------------
Librería para modelamiento con regresión logística, verificación de direccionalidad
y validación de métricas (GINI) en el tiempo.
Organización:
- regression: funciones de regresión logística
- directionality: verificación de direccionalidad de variables
- validation: cálculo y validación de métricas (GINI)
"""

from .regression import (
    fit_logistic_regression,
    predict_logistic,
    get_coefficients_summary,
)

from .directionality import (
    check_directionality,
    verify_tasa_direction,
    get_directionality_report,
)

from .validation import (
    calculate_gini,
    calculate_gini_by_period,
    validate_gini_stability,
    plot_gini_over_time,
)

__all__ = [
    # regression
    "fit_logistic_regression",
    "predict_logistic",
    "get_coefficients_summary",
    # directionality
    "check_directionality",
    "verify_tasa_direction",
    "get_directionality_report",
    # validation
    "calculate_gini",
    "calculate_gini_by_period",
    "validate_gini_stability",
    "plot_gini_over_time",
]

