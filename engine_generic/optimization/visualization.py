"""
Módulo de visualización para optimización de pricing.

Genera gráficas de frontera eficiente mostrando el trade-off entre profit y volumen.
"""

import logging
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend no interactivo


def plot_efficient_frontier_from_dataframe(
    df_frontier: pd.DataFrame,
    price_variable_name: str = "spread",
    save_path: str = "optimization/efficient_frontier.png",
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Genera gráfica de frontera eficiente a partir de un DataFrame con datos calculados.
    
    Esta función es más generalizable y puede usarse con cualquier DataFrame que contenga:
    - alpha: Valores de alpha probados
    - optimal_{price_variable_name}: Valor óptimo de la variable de precio
    - total_profit: Profit total esperado agregado
    - total_volume: Volume total esperado agregado
    
    Args:
        df_frontier: DataFrame con datos de frontera eficiente calculados
                    Debe contener columnas: 'alpha', f'optimal_{price_variable_name}', 
                    'total_profit', 'total_volume'
        price_variable_name: Nombre de la variable de precio (ej: "spread", "tasa")
        save_path: Ruta donde guardar la gráfica
        logger: Logger opcional
    """
    if logger:
        logger.info("Generando gráfica de frontera eficiente desde DataFrame...")
    
    # Validar columnas requeridas
    required_cols = ['alpha', 'total_profit', 'total_volume']
    missing_cols = [col for col in required_cols if col not in df_frontier.columns]
    if missing_cols:
        raise ValueError(f"DataFrame de frontera eficiente debe contener columnas: {missing_cols}")
    
    # Ordenar por alpha para mejor visualización
    df_frontier = df_frontier.sort_values('alpha').copy()
    
    # Normalizar profit y volume respecto al máximo para mejor visualización
    max_profit = df_frontier['total_profit'].max()
    max_volume = df_frontier['total_volume'].max()
    
    # Evitar división por cero
    if max_profit == 0:
        max_profit = 1.0
    if max_volume == 0:
        max_volume = 1.0
    
    profit_norm = df_frontier['total_profit'] / max_profit
    volume_norm = df_frontier['total_volume'] / max_volume
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plotear curva de frontera eficiente
    ax.plot(volume_norm, profit_norm, 'o-', color='#1f77b4', 
            linewidth=2.5, markersize=8, label='Frontera Eficiente', zorder=3)
    
    # Etiquetar puntos con valores de alpha
    for idx, row in df_frontier.iterrows():
        alpha_val = row['alpha']
        x = volume_norm.loc[idx]
        y = profit_norm.loc[idx]
        
        # Etiquetar algunos puntos clave
        if alpha_val in [0.0, 0.5, 1.0] or (alpha_val == 0.7 and 'optimal_spread' in df_frontier.columns):
            ax.annotate(f"α={alpha_val:.1f}", (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=9, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='white', alpha=0.8, edgecolor='#1f77b4'))
    
    # Configurar ejes
    ax.set_xlabel('Volumen Total Normalizado (Propensión × Monto)', 
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Profit Total Normalizado (Precio × Monto × Propensión)', 
                  fontsize=12, fontweight='bold')
    ax.set_title(f'Frontera Eficiente: Trade-off Profit vs Volumen\n'
                 f'Variable: {price_variable_name}', 
                 fontsize=14, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Leyenda
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    
    # Guardar gráfica
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if logger:
        logger.info(f"Gráfica de frontera eficiente guardada en: {save_path}")
    
    return

