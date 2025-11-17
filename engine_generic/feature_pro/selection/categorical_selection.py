import logging
import numpy as np
from typing import Iterable, List, Optional, Tuple, Union
from typing import Dict, Any

import pandas as pd

from feature_pro.common.utils import (
    detect_column_types,
    _ensure_list,
    _log,
)


def count_unique_categorical(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    include_na: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con el conteo de valores únicos por variable categórica.
    Args:
        df: DataFrame de entrada.
        columns: columnas categóricas a evaluar; si None, se detectan automáticamente.
        include_na: si True, incluye NaN como categoría única en el conteo (considera 'MISSING').
        logger: logger opcional para registrar la operación.
    Returns:
        DataFrame con columnas ['variable','n_unique'] ordenado desc por n_unique
    """
    if columns is None:
        _, cat_cols = detect_column_types(df)
    else:
        cat_cols = _ensure_list(columns, df.columns.tolist())
    _log(logger, f"Contando valores únicos para {len(cat_cols)} variables categóricas (include_na={include_na})")
    records: List[Tuple[str, int]] = []
    for col in cat_cols:
        if col not in df.columns:
            continue
        n_unique = int(df[col].astype('object').nunique(dropna=not include_na))
        records.append((col, n_unique))
    out = pd.DataFrame(records, columns=['variable', 'n_unique']).sort_values('n_unique', ascending=False).reset_index(drop=True)
    return out


def categorical_cumulative_frequency(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    threshold: float = 0.8,
    include_na: bool = False,
    return_distributions: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Calcula la frecuencia acumulada de valores por variable categórica y evalúa dominancia.
    - Si el valor más frecuente de una variable supera 'threshold' (p.ej. 0.8), la variable es marcada como dominante.
    - Devuelve un resumen y la lista de columnas a mantener (no dominantes).
    Args:
        df: DataFrame de entrada.
        columns: columnas a evaluar; si None, detecta categóricas automáticamente.
        threshold: umbral (0-1) de dominancia del valor más frecuente para marcar variable como dominante.
        include_na: si True, considera NaN como categoría ('MISSING').
        return_distributions: si True, devuelve distribución detallada por variable (value, count, pct, cum_pct).
        logger: logger opcional.
    Returns:
        {
          'summary': DataFrame[variable, top_value, top_count, top_pct, num_unique, is_dominant],
          'kept_columns': List[str],
          'removed_columns': List[str],
          'distributions': Optional[Dict[str, DataFrame[value, count, pct, cum_pct]]]
        }
    """
    if columns is None:
        _, cat_cols = detect_column_types(df)
    else:
        cat_cols = _ensure_list(columns, df.columns.tolist())
    _log(logger, f"Evaluando dominancia categórica en {len(cat_cols)} variables (threshold={threshold}, include_na={include_na})")

    summary_rows: List[Tuple[str, Any, int, float, int, bool]] = []
    distributions: Dict[str, pd.DataFrame] = {}

    for col in cat_cols:
        if col not in df.columns:
            continue
        s = df[col].astype('object')
        if include_na:
            s = s.fillna('MISSING')
        total = len(s) if include_na else s.notna().sum()
        if total == 0:
            # Columna vacía tras exclusiones
            summary_rows.append((col, None, 0, 0.0, 0, True))
            if return_distributions:
                distributions[col] = pd.DataFrame(columns=['value', 'count', 'pct', 'cum_pct'])
            continue
        vc = s.value_counts(dropna=not include_na)
        pct = (vc / total).astype(float)
        dist = pd.DataFrame({'value': vc.index.astype('object'), 'count': vc.values, 'pct': pct.values})
        dist['cum_pct'] = dist['pct'].cumsum()
        # top
        top_value = dist.iloc[0]['value'] if len(dist) > 0 else None
        top_count = int(dist.iloc[0]['count']) if len(dist) > 0 else 0
        top_pct = float(dist.iloc[0]['pct']) if len(dist) > 0 else 0.0
        num_unique = int(dist.shape[0])
        is_dominant = top_pct >= threshold
        summary_rows.append((col, top_value, top_count, top_pct, num_unique, is_dominant))
        if return_distributions:
            distributions[col] = dist

    summary = pd.DataFrame(
        summary_rows,
        columns=['variable', 'top_value', 'top_count', 'top_pct', 'num_unique', 'is_dominant']
    ).sort_values('top_pct', ascending=False).reset_index(drop=True)

    removed_columns = summary.loc[summary['is_dominant'], 'variable'].tolist()
    kept_columns = summary.loc[~summary['is_dominant'], 'variable'].tolist()
    _log(logger, f"Variables dominantes detectadas: {len(removed_columns)}; variables a mantener: {len(kept_columns)}")

    result: Dict[str, Any] = {
        'summary': summary,
        'kept_columns': kept_columns,
        'removed_columns': removed_columns,
    }
    if return_distributions:
        result['distributions'] = distributions
    return result


def calculate_woe_iv(
    df: pd.DataFrame,
    target: str,
    columns: Optional[Union[str, Iterable[str]]] = None,
    include_na: bool = True,
    bin_numeric: bool = False,
    num_bins: int = 10,
    bin_strategy: str = 'quantile',
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Calcula WOE/IV para un conjunto de variables (principalmente categóricas).
    - Para variables numéricas, puede opcionalmente binar por cuantiles o anchos iguales.
    - Requiere que el target sea binario (0/1), donde por defecto 1=good, 0=bad.
    Args:
        df: DataFrame de entrada.
        target: nombre de la columna objetivo binaria (0/1).
        columns: columnas a evaluar; si None, detecta categóricas automáticamente.
        include_na: si True, considera NaN como categoría ('MISSING').
        bin_numeric: si True, binea variables numéricas antes de calcular WOE/IV.
        num_bins: número de bins si bin_numeric=True.
        bin_strategy: 'quantile' o 'equal' para bin_numeric.
        logger: logger opcional.
    Returns:
        {
          'summary': DataFrame ['variable','iv'] ordenado desc,
          'details': Dict[var_name -> DataFrame ['value','good','bad','dist_good','dist_bad','woe','iv_contrib']]
        }
    """
    if target not in df.columns:
        raise ValueError(f"Columna target '{target}' no encontrada en el DataFrame.")
    if df[target].dropna().nunique() != 2:
        raise ValueError("El target debe ser binario (dos clases).")

    if columns is None:
        # Por defecto: solo categóricas
        _, cat_cols = detect_column_types(df)
        cols = cat_cols
    else:
        cols = _ensure_list(columns, df.columns.tolist())

    _log(logger, f"Calculando WOE/IV para {len(cols)} variables; bin_numeric={bin_numeric} (num_bins={num_bins}, strategy={bin_strategy}).")

    y = df[target]
    # Mapeo explícito: 1 = good, 0 = bad
    total_good = (y == 1).sum()
    total_bad = (y == 0).sum()
    if total_good == 0 or total_bad == 0:
        raise ValueError("El target debe contener ambas clases 0 y 1 para calcular WOE/IV.")

    details: Dict[str, pd.DataFrame] = {}
    summary_rows: List[Tuple[str, float]] = []
    eps = 1e-9

    for col in cols:
        s = df[col]
        s_series = s
        # Binning opcional para numéricas
        if bin_numeric and pd.api.types.is_numeric_dtype(s):
            s_non_null = s.dropna()
            if s_non_null.empty:
                # Sin información
                summary_rows.append((col, np.nan))
                details[col] = pd.DataFrame(columns=['value','good','bad','dist_good','dist_bad','woe','iv_contrib'])
                continue
            if bin_strategy == 'quantile':
                try:
                    q = np.linspace(0, 1, num_bins + 1)
                    edges = np.unique(s_non_null.quantile(q).values)
                    if len(edges) <= 2:
                        # fallback a equal
                        edges = np.linspace(s_non_null.min(), s_non_null.max(), num_bins + 1)
                except Exception:
                    edges = np.linspace(s_non_null.min(), s_non_null.max(), num_bins + 1)
            else:
                edges = np.linspace(s_non_null.min(), s_non_null.max(), num_bins + 1)
            s_series = pd.cut(s, bins=edges, include_lowest=True, duplicates='drop')

        # Preparar serie (convertir categórica a object para permitir 'MISSING')
        s_obj = s_series.astype('object')
        if include_na:
            s_obj = s_obj.fillna('MISSING')

        # Conteos por categoría
        good_counts = df.loc[y == 1].groupby(s_obj.loc[y == 1]).size()
        bad_counts = df.loc[y == 0].groupby(s_obj.loc[y == 0]).size()
        all_cats = good_counts.index.union(bad_counts.index)
        good = good_counts.reindex(all_cats).fillna(0).astype(int)
        bad = bad_counts.reindex(all_cats).fillna(0).astype(int)

        dist_good = (good / max(total_good, 1)).astype(float) + eps
        dist_bad = (bad / max(total_bad, 1)).astype(float) + eps
        woe = np.log(dist_good / dist_bad)
        iv_contrib = (dist_good - dist_bad) * woe
        iv_value = float(iv_contrib.sum())

        det = pd.DataFrame({
            'value': all_cats.astype('object'),
            'good': good.values,
            'bad': bad.values,
            'dist_good': dist_good.values,
            'dist_bad': dist_bad.values,
            'woe': woe.values,
            'iv_contrib': iv_contrib.values,
        }).sort_values('iv_contrib', ascending=False).reset_index(drop=True)

        details[col] = det
        summary_rows.append((col, iv_value))

    summary = pd.DataFrame(summary_rows, columns=['variable', 'iv']).sort_values('iv', ascending=False).reset_index(drop=True)
    return {'summary': summary, 'details': details}

