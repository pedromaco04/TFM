import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """
    Envía mensajes al logger si está disponible.
    """
    if logger is not None:
        logger.info(message)


def calculate_gini(
    y_true: pd.Series,
    y_pred_proba: pd.Series,
    logger: Optional[logging.Logger] = None,
) -> float:
    """
    Calcula el GINI a partir de probabilidades predichas y valores reales.
    Args:
        y_true: valores reales (binarios 0/1)
        y_pred_proba: probabilidades predichas
        logger: logger opcional
    Returns:
        Valor del GINI (entre -1 y 1)
    """
    # Validar que no haya missings
    valid_mask = y_true.notna() & y_pred_proba.notna()
    y_true_clean = y_true[valid_mask]
    y_pred_proba_clean = y_pred_proba[valid_mask]
    
    if len(y_true_clean) == 0:
        _log(logger, "ADVERTENCIA: No hay datos válidos para calcular GINI")
        return np.nan
    
    if y_true_clean.nunique() < 2:
        _log(logger, "ADVERTENCIA: Solo hay una clase en los datos. No se puede calcular GINI")
        return np.nan
    
    try:
        auc = roc_auc_score(y_true_clean, y_pred_proba_clean)
        gini = 2 * auc - 1
        _log(logger, f"GINI calculado: {gini:.4f} (AUC: {auc:.4f}, N: {len(y_true_clean)})")
        return float(gini)
    except Exception as e:
        _log(logger, f"Error calculando GINI: {e}")
        return np.nan


def calculate_gini_by_period(
    df: pd.DataFrame,
    y_true_column: str,
    y_pred_proba_column: str,
    period_column: str,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Calcula GINI por período temporal (útil para validación en el tiempo).
    Args:
        df: DataFrame con datos
        y_true_column: nombre de la columna con valores reales
        y_pred_proba_column: nombre de la columna con probabilidades predichas
        period_column: nombre de la columna con períodos temporales
        logger: logger opcional
    Returns:
        DataFrame con columnas ['period', 'gini', 'auc', 'n_samples']
    """
    _log(logger, f"Calculando GINI por período (columna: '{period_column}')")
    
    # Validar columnas
    required_cols = [y_true_column, y_pred_proba_column, period_column]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")
    
    periods = sorted(df[period_column].dropna().unique())
    _log(logger, f"Períodos encontrados: {len(periods)}")
    
    results = []
    for period in periods:
        period_mask = df[period_column] == period
        df_period = df[period_mask]
        
        y_true = df_period[y_true_column]
        y_pred_proba = df_period[y_pred_proba_column]
        
        # Calcular GINI
        gini = calculate_gini(y_true, y_pred_proba, logger=None)  # No loggear cada período
        
        # Calcular AUC
        valid_mask = y_true.notna() & y_pred_proba.notna()
        y_true_clean = y_true[valid_mask]
        y_pred_proba_clean = y_pred_proba[valid_mask]
        
        if len(y_true_clean) > 0 and y_true_clean.nunique() >= 2:
            try:
                auc = roc_auc_score(y_true_clean, y_pred_proba_clean)
            except:
                auc = np.nan
        else:
            auc = np.nan
        
        results.append({
            'period': period,
            'gini': gini,
            'auc': auc,
            'n_samples': int(len(df_period)),
        })
    
    result_df = pd.DataFrame(results)
    _log(logger, f"GINI calculado para {len(results)} períodos")
    _log(logger, f"GINI promedio: {result_df['gini'].mean():.4f}, "
         f"Min: {result_df['gini'].min():.4f}, Max: {result_df['gini'].max():.4f}")
    
    return result_df


def validate_gini_stability(
    gini_by_period_df: pd.DataFrame,
    threshold_degradation: float = 0.05,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Valida estabilidad del GINI en el tiempo.
    Detecta degradación significativa entre períodos.
    Args:
        gini_by_period_df: DataFrame retornado por calculate_gini_by_period
        threshold_degradation: umbral de degradación relativa para considerar inestable (default 5%)
        logger: logger opcional
    Returns:
        dict con:
          - 'is_stable': bool indicando si el GINI es estable
          - 'mean_gini': GINI promedio
          - 'std_gini': desviación estándar del GINI
          - 'min_gini': GINI mínimo
          - 'max_gini': GINI máximo
          - 'degradation_periods': lista de períodos con degradación significativa
          - 'summary': DataFrame con resumen de validación
    """
    _log(logger, f"Validando estabilidad del GINI (threshold degradación: {threshold_degradation})")
    
    if gini_by_period_df.empty:
        _log(logger, "ADVERTENCIA: DataFrame vacío para validación de GINI")
        return {
            'is_stable': False,
            'mean_gini': np.nan,
            'std_gini': np.nan,
            'min_gini': np.nan,
            'max_gini': np.nan,
            'degradation_periods': [],
            'summary': pd.DataFrame(),
        }
    
    gini_values = gini_by_period_df['gini'].dropna()
    
    if len(gini_values) < 2:
        _log(logger, "ADVERTENCIA: Menos de 2 períodos con GINI válido. No se puede validar estabilidad.")
        return {
            'is_stable': True,  # No hay suficiente información
            'mean_gini': float(gini_values.mean()) if len(gini_values) > 0 else np.nan,
            'std_gini': np.nan,
            'min_gini': float(gini_values.min()) if len(gini_values) > 0 else np.nan,
            'max_gini': float(gini_values.max()) if len(gini_values) > 0 else np.nan,
            'degradation_periods': [],
            'summary': pd.DataFrame(),
        }
    
    mean_gini = float(gini_values.mean())
    std_gini = float(gini_values.std())
    min_gini = float(gini_values.min())
    max_gini = float(gini_values.max())
    cv = std_gini / mean_gini if mean_gini != 0 else np.inf
    
    # Detectar degradación: si algún período tiene GINI significativamente menor que el promedio
    degradation_periods = []
    for _, row in gini_by_period_df.iterrows():
        if pd.isna(row['gini']):
            continue
        relative_diff = (mean_gini - row['gini']) / mean_gini if mean_gini != 0 else 0
        if relative_diff > threshold_degradation:
            degradation_periods.append(row['period'])
    
    # Considerar estable si:
    # 1. CV < threshold_degradation (variabilidad relativa baja)
    # 2. No hay períodos con degradación significativa
    is_stable = (cv < threshold_degradation) and (len(degradation_periods) == 0)
    
    _log(logger, f"Estabilidad del GINI:")
    _log(logger, f"  - Promedio: {mean_gini:.4f}")
    _log(logger, f"  - Desv. Est.: {std_gini:.4f}")
    _log(logger, f"  - CV: {cv:.4f}")
    _log(logger, f"  - Rango: [{min_gini:.4f}, {max_gini:.4f}]")
    _log(logger, f"  - Períodos con degradación: {len(degradation_periods)}")
    
    if is_stable:
        _log(logger, "✓ GINI es estable en el tiempo")
    else:
        _log(logger, f"✗ ADVERTENCIA: GINI muestra inestabilidad")
        if degradation_periods:
            _log(logger, f"  Períodos con degradación: {degradation_periods}")
    
    summary_df = pd.DataFrame([{
        'mean_gini': mean_gini,
        'std_gini': std_gini,
        'cv_gini': cv,
        'min_gini': min_gini,
        'max_gini': max_gini,
        'n_periods': len(gini_values),
        'n_degradation_periods': len(degradation_periods),
        'is_stable': is_stable,
    }])
    
    return {
        'is_stable': is_stable,
        'mean_gini': mean_gini,
        'std_gini': std_gini,
        'cv_gini': cv,
        'min_gini': min_gini,
        'max_gini': max_gini,
        'degradation_periods': degradation_periods,
        'summary': summary_df,
    }


def plot_gini_over_time(
    gini_by_period_df: pd.DataFrame,
    output_path: Optional[str] = None,
    return_fig: bool = True,
    figsize: tuple = (10, 6),
    logger: Optional[logging.Logger] = None,
) -> Optional[plt.Figure]:
    """
    Genera gráfico de GINI a lo largo del tiempo.
    Args:
        gini_by_period_df: DataFrame retornado por calculate_gini_by_period
        output_path: ruta donde guardar el gráfico (opcional)
        return_fig: si True, retorna la figura
        figsize: tamaño de la figura
        logger: logger opcional
    Returns:
        matplotlib.figure.Figure si return_fig=True, None en caso contrario
    """
    _log(logger, "Generando gráfico de GINI en el tiempo")
    
    if gini_by_period_df.empty:
        _log(logger, "ADVERTENCIA: DataFrame vacío. No se puede generar gráfico.")
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Convertir períodos a string para el eje X
    periods = gini_by_period_df['period'].astype(str)
    gini_values = gini_by_period_df['gini']
    
    # Línea principal
    ax.plot(periods, gini_values, marker='o', linewidth=2, markersize=6, label='GINI')
    
    # Línea de promedio
    mean_gini = gini_values.mean()
    ax.axhline(y=mean_gini, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Promedio: {mean_gini:.4f}')
    
    # Área de confianza (promedio ± 1 desv. est.)
    std_gini = gini_values.std()
    ax.fill_between(periods, 
                     mean_gini - std_gini, 
                     mean_gini + std_gini, 
                     alpha=0.2, 
                     color='gray', 
                     label=f'±1 Desv. Est.')
    
    ax.set_xlabel('Período', fontsize=12)
    ax.set_ylabel('GINI', fontsize=12)
    ax.set_title('Evolución del GINI en el Tiempo', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        _log(logger, f"Gráfico guardado en: {output_path}")
    
    if return_fig:
        return fig
    else:
        plt.close(fig)
        return None

