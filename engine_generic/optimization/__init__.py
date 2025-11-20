"""
Módulo genérico de optimización de pricing para productos financieros.

Este módulo proporciona funciones genéricas para optimizar variables de precio
y balancear profit y volumen.
"""

from optimization.spread_optimizer import (
    run_price_optimization,
    optimize_pilot_prices,
    calculate_efficient_frontier,
    # Funciones wrapper para compatibilidad hacia atrás
    run_spread_optimization,
    optimize_pilot_spreads,
)
from optimization.visualization import (
    plot_efficient_frontier_from_dataframe,
)

__all__ = [
    "run_price_optimization",
    "optimize_pilot_prices",
    "calculate_efficient_frontier",
    # Funciones wrapper para compatibilidad hacia atrás
    "run_spread_optimization",
    "optimize_pilot_spreads",
    "plot_efficient_frontier_from_dataframe",
]

