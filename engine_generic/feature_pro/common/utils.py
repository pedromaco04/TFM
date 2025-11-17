import logging
from typing import Iterable, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


import matplotlib.pyplot as plt
import seaborn as sns


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """
    Envía mensajes al logger si está disponible.
    Args:
        logger: instancia de logging.Logger o None
        message: texto a registrar
    """
    if logger is not None:
        logger.info(message)


def _ensure_list(columns: Optional[Union[str, Iterable[str]]], all_columns: List[str]) -> List[str]:
    """
    Asegura que la entrada 'columns' sea una lista de nombres de columnas válida.
    - Si es None, devuelve todas las columnas.
    - Si es str, devuelve [str].
    - Si es iterable, filtra solo columnas presentes en all_columns.
    """
    if columns is None:
        return list(all_columns)
    if isinstance(columns, str):
        return [columns]
    return [c for c in columns if c in all_columns]


def read_dataset(
    path: str,
    fmt: Optional[str] = 'auto',
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    dtype_backend: Optional[str] = 'numpy_nullable',
    low_memory: bool = False,
    sheet_name: Union[int, str, None] = 0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Lector genérico de datasets con autodetección de formato por extensión.
    Args:
        path: ruta al archivo a leer.
        fmt: formato del archivo. Soportado: 'auto' | 'csv' | 'parquet' | 'txt' | 'tsv' | 'xlsx'.
             Si 'auto', se infiere desde la extensión del archivo.
        sep: separador para archivos de texto (csv/txt/tsv). Si None, usa ',' para csv/txt y '\\t' para tsv.
        encoding: codificación de texto para lectores basados en csv/txt/tsv.
        dtype_backend: backend de tipos para pandas (p.ej. 'numpy_nullable').
        low_memory: parámetro de pandas para csv; optimiza memoria a costa de tipos provisionales.
        sheet_name: nombre o índice de hoja para xlsx/xls.
        logger: logger opcional donde enviar mensajes.
    Returns:
        DataFrame leído desde el archivo.
    Raises:
        ValueError: si el formato no está soportado.
    """
    _log(logger, f"Leyendo dataset desde: {path} (fmt={fmt})")
    fmt_resolved = fmt
    if fmt_resolved is None or fmt_resolved == 'auto':
        lower = path.lower()
        if lower.endswith('.csv'):
            fmt_resolved = 'csv'
        elif lower.endswith('.parquet') or lower.endswith('.pq'):
            fmt_resolved = 'parquet'
        elif lower.endswith('.tsv'):
            fmt_resolved = 'tsv'
        elif lower.endswith('.txt'):
            fmt_resolved = 'txt'
        elif lower.endswith('.xlsx') or lower.endswith('.xls'):
            fmt_resolved = 'xlsx'
        else:
            fmt_resolved = 'csv'  # por defecto

    if fmt_resolved == 'csv':
        df = pd.read_csv(path, sep=sep if sep is not None else ',', encoding=encoding, low_memory=low_memory, dtype_backend=dtype_backend)
    elif fmt_resolved == 'parquet':
        df = pd.read_parquet(path)
    elif fmt_resolved == 'tsv':
        df = pd.read_csv(path, sep='\t' if sep is None else sep, encoding=encoding, low_memory=low_memory, dtype_backend=dtype_backend)
    elif fmt_resolved == 'txt':
        df = pd.read_csv(path, sep=sep if sep is not None else ',', encoding=encoding, low_memory=low_memory, dtype_backend=dtype_backend)
    elif fmt_resolved == 'xlsx':
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Formato no soportado: {fmt_resolved}")

    _log(logger, f"Dataset leído. Shape: {df.shape}")
    return df


def summarize_missing(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Genera una tabla resumen con conteo y porcentaje de valores faltantes por columna.
    Args:
        df: DataFrame a analizar
        columns: subconjunto de columnas (si None, usa todas)
    Returns:
        DataFrame con columnas ['column','dtype','num_missing','pct_missing']
    """
    cols = _ensure_list(columns, df.columns.tolist())
    subset = df[cols]
    miss = subset.isna().sum()
    pct = subset.isna().mean()
    out = pd.DataFrame({
        'column': miss.index,
        'dtype': subset.dtypes.astype(str).values,
        'num_missing': miss.values,
        'pct_missing': pct.values,
    })
    return out.sort_values(by='pct_missing', ascending=False).reset_index(drop=True)


def detect_column_types(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    treat_object_numeric: bool = True,
    sample_size_for_analysis: int = 100,
    max_code_length: int = 10,
    object_numeric_threshold: float = 0.95,
) -> Tuple[List[str], List[str]]:
    """
    Identifica columnas numéricas y categóricas, considerando objetos que representan números.
    Args:
        df: DataFrame
        columns: columnas a evaluar (si None, todas)
        treat_object_numeric: intenta convertir 'object' numéricos a numérico real
        sample_size_for_analysis: tamaño de muestra para validar patrones de strings
        max_code_length: longitud máxima para detectar códigos (no continuos)
        object_numeric_threshold: si treat_object_numeric=True, umbral mínimo (0-1)
            de fracción de valores no nulos convertibles a numérico para considerar la
            columna 'object' como numérica (default 0.95)
    Returns:
        (numeric_cols, categorical_cols)
    """
    cols = _ensure_list(columns, df.columns.tolist())
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in cols:
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        if treat_object_numeric and pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if non_null.empty:
                categorical_cols.append(col)
                continue
            sample = non_null.astype(str).head(sample_size_for_analysis)
            contains_letters = any(any(ch.isalpha() for ch in val) for val in sample)
            looks_like_code = any(len(str(val)) > max_code_length for val in sample)
            if not contains_letters and not looks_like_code:
                coerced = pd.to_numeric(non_null, errors='coerce')
                frac_numeric = coerced.notna().mean()
                if frac_numeric >= object_numeric_threshold:
                    numeric_cols.append(col)
                    continue

        categorical_cols.append(col)

    return numeric_cols, categorical_cols


def impute_missing(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    strategy: str = 'median',
    fill_value: Optional[Union[int, float, str]] = None,
    groupby: Optional[Union[str, Iterable[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Imputa valores faltantes de forma genérica, con opción de agrupación.
    Estrategias soportadas:
      - 'mean', 'median' (numéricas)
      - 'mode' (moda; útil para categóricas o numéricas)
      - 'zero' (rellena con 0)
      - 'constant' (usa fill_value)
    Args:
        df: DataFrame
        columns: columnas a imputar (si None, detecta numéricas)
        strategy: estrategia de imputación
        fill_value: valor para 'constant'
        groupby: columna(s) para imputación por grupos (opcional)
        logger: logger opcional
    Returns:
        DataFrame con imputación aplicada
    """
    result = df.copy()
    if columns is None:
        columns, _ = detect_column_types(df)
    else:
        columns = _ensure_list(columns, df.columns.tolist())

    def _impute_series(s: pd.Series) -> pd.Series:
        if strategy == 'zero':
            return s.fillna(0)
        if strategy == 'constant':
            return s.fillna(fill_value)
        if strategy == 'mode':
            # Moda por serie/grupo; si múltiples, toma la primera
            mode_vals = s.mode(dropna=True)
            if mode_vals.empty:
                return s
            mode_val = mode_vals.iloc[0]
            return s.fillna(mode_val)
        if not pd.api.types.is_numeric_dtype(s):
            return s if strategy not in ('mean', 'median') else s
        if strategy == 'mean':
            return s.fillna(s.mean())
        if strategy == 'median':
            return s.fillna(s.median())
        return s

    if groupby is None:
        _log(logger, f"Imputando {len(columns)} columnas con estrategia='{strategy}' (sin groupby)")
        for col in columns:
            result[col] = _impute_series(result[col])
    else:
        by = [groupby] if isinstance(groupby, str) else list(groupby)
        _log(logger, f"Imputando {len(columns)} columnas con estrategia='{strategy}' por grupos={by}")
        grouped = result.groupby(by, dropna=False)
        for col in columns:
            result[col] = grouped[col].transform(lambda s: _impute_series(s))
    return result


def winsorize_by_percentile(
    df: pd.DataFrame,
    columns: Union[str, Iterable[str]],
    lower: float = 0.01,
    upper: float = 0.99,
    logger: Optional[logging.Logger] = None,
    return_percentiles: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Winsorización por percentiles: recorta valores por debajo/encima de [lower, upper].
    Args:
        df: DataFrame
        columns: columnas numéricas a winsorizar
        lower: percentil inferior (0-1)
        upper: percentil superior (0-1)
        logger: logger opcional
        return_percentiles: si True, devuelve (df_wins, df_percentiles)
    Returns:
        DataFrame winsorizado o tupla con percentiles por columna
    """
    cols = _ensure_list(columns, df.columns.tolist())
    result = df.copy()
    _log(logger, f"Winsorizando por percentiles columnas={len(cols)} lower={lower} upper={upper}")
    percentiles_data: List[Tuple[str, float, float]] = []
    for col in cols:
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue
        q_low = result[col].quantile(lower)
        q_hi = result[col].quantile(upper)
        result[col] = result[col].clip(lower=q_low, upper=q_hi)
        percentiles_data.append((col, float(q_low) if pd.notna(q_low) else np.nan, float(q_hi) if pd.notna(q_hi) else np.nan))
    if return_percentiles:
        perc_df = pd.DataFrame(percentiles_data, columns=['column', f'Q{int(lower*100)}', f'Q{int(upper*100)}'])
        return result, perc_df
    return result


def winsorize_by_iqr(
    df: pd.DataFrame,
    columns: Union[str, Iterable[str]],
    factor: float = 1.5,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Winsorización por IQR: recorta a [Q1 - factor*IQR, Q3 + factor*IQR].
    Args:
        df: DataFrame
        columns: columnas numéricas a procesar
        factor: multiplicador del IQR
        logger: logger opcional
    """
    cols = _ensure_list(columns, df.columns.tolist())
    result = df.copy()
    _log(logger, f"Winsorizando por IQR columnas={len(cols)} factor={factor}")
    for col in cols:
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue
        q1 = result[col].quantile(0.25)
        q3 = result[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        result[col] = result[col].clip(lower=lower, upper=upper)
    return result


def _cramers_v(conf_mat: pd.DataFrame) -> float:
    """
    Cramér's V para asociación entre dos variables categóricas.
    Args:
        conf_mat: tabla de contingencia (DataFrame)
    Returns:
        Valor de Cramér's V en [0,1]
    """
    chi2 = stats.chi2_contingency(conf_mat)[0]
    n = conf_mat.values.sum()
    r, k = conf_mat.shape
    return float(np.sqrt(chi2 / (n * (min(k - 1, r - 1) if min(k, r) > 1 else 1))))


def _correlation_ratio(categories: np.ndarray, measurements: np.ndarray) -> float:
    """
    Correlation Ratio (eta) entre categórica y numérica.
    Args:
        categories: array-like categórico
        measurements: array-like numérico
    Returns:
        Eta en [0,1]
    """
    cats = pd.Categorical(categories)
    y = pd.Series(measurements).astype(float)
    groups = [y[cats == cat] for cat in cats.categories]
    n_total = float(len(y))
    if n_total == 0:
        return np.nan
    y_mean = float(y.mean())
    ss_between = sum([len(g) * (float(g.mean()) - y_mean) ** 2 for g in groups if len(g) > 0])
    ss_total = float(((y - y_mean) ** 2).sum())
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


def _point_biserial_safe(x: np.ndarray, y: np.ndarray) -> float:
    """
    Point-biserial para numérica vs binaria, con validaciones mínimas.
    Args:
        x: array-like numérico
        y: array-like binario (0/1)
    Returns:
        Coeficiente de correlación point-biserial o NaN
    """
    y_bin = pd.Series(y).dropna()
    if y_bin.nunique() != 2:
        return np.nan
    s = pd.Series(x).dropna()
    common = y_bin.index.intersection(s.index)
    if len(common) < 3:
        return np.nan
    r, _ = stats.pointbiserialr(s.loc[common], y_bin.loc[common])
    return float(r)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    method: str = 'pearson',
    plot: bool = False,
    return_fig: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[pd.DataFrame, Optional['plt.Figure']]]:
    """
    Calcula una matriz de correlación flexible.
    Métodos:
      - 'pearson' | 'spearman' | 'kendall' (solo numéricas)
      - 'auto' mixto: num-num (pearson), cat-cat (Cramér's V), cat-num (eta)
    Args:
        df: DataFrame
        columns: columnas a evaluar (si None, todas)
        method: método de correlación
        plot: si True, genera heatmap
        return_fig: si True, retorna la figura
        logger: logger opcional
    Returns:
        dict con {'matrix': DataFrame, 'figure': Optional[Figure]}
    """
    cols = _ensure_list(columns, df.columns.tolist())
    if method in ('pearson', 'spearman', 'kendall'):
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        mat = df[num_cols].corr(method=method)
    else:
        num_cols, cat_cols = detect_column_types(df, columns=cols)
        mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if j < i:
                    continue
                if ci == cj:
                    mat.loc[ci, cj] = 1.0
                    continue
                if (ci in num_cols) and (cj in num_cols):
                    r = df[[ci, cj]].corr(method='pearson').iloc[0, 1]
                elif (ci in cat_cols) and (cj in cat_cols):
                    cont = pd.crosstab(df[ci], df[cj])
                    r = _cramers_v(cont) if cont.size > 0 else np.nan
                else:
                    if ci in cat_cols and cj in num_cols:
                        r = _correlation_ratio(df[ci].values, df[cj].values)
                    else:
                        r = _correlation_ratio(df[cj].values, df[ci].values)
                mat.loc[ci, cj] = r
                mat.loc[cj, ci] = r

    fig = None
    if plot and (plt is not None) and (sns is not None):
        fig, ax = plt.subplots(figsize=(max(6, len(cols) * 0.5), max(4, len(cols) * 0.5)))
        sns.heatmap(mat.astype(float), cmap='coolwarm', center=0, annot=False, square=True, ax=ax)
        ax.set_title(f"Matriz de correlaciones ({method})")
        plt.tight_layout()

    _log(logger, f"Matriz de correlación ({method}) calculada para {len(cols)} columnas.")
    return {'matrix': mat, 'figure': fig if return_fig else None}


