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
    Identifica columnas numéricas y categóricas con reglas simples:
      - Si la columna es datetime, se ignora (no se devuelve en ninguna lista).
      - Si todos los valores no nulos pueden convertirse a float, se considera NUMÉRICA.
      - En caso contrario, se considera CATEGÓRICA.
    Args:
        df: DataFrame
        columns: columnas a evaluar (si None, todas)
        treat_object_numeric: ignorado en esta versión simplificada (mantenido por compatibilidad)
        sample_size_for_analysis: ignorado (compatibilidad)
        max_code_length: ignorado (compatibilidad)
        object_numeric_threshold: ignorado (compatibilidad)
    Returns:
        (numeric_cols, categorical_cols)
    """
    cols = _ensure_list(columns, df.columns.tolist())
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in cols:
        series = df[col]
        # Ignorar columnas datetime
        if pd.api.types.is_datetime64_any_dtype(series):
            continue

        # Si ya es numérica: entra como numérica
        if pd.api.types.is_numeric_dtype(series):
            numeric_cols.append(col)
            continue

        # Para object/string: si TODOS los no nulos son convertibles a float, es numérica
        non_null = series.dropna()
        if len(non_null) == 0:
            # sin evidencia; tratar como categórica por defecto
            categorical_cols.append(col)
            continue
        coerced = pd.to_numeric(non_null, errors='coerce')
        if coerced.notna().all():
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


def apply_standard_scaler(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    scaler: Optional[StandardScaler] = None,
    return_scaler: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, StandardScaler]]:
    """
    Aplica StandardScaler (normalización z-score) a columnas numéricas del DataFrame.
    
    Esta función permite:
    - Entrenar un nuevo scaler (fit) y aplicarlo (transform)
    - Aplicar un scaler pre-entrenado (solo transform)
    
    Args:
        df: DataFrame a transformar
        columns: columnas numéricas a escalar. Si None, detecta automáticamente todas las numéricas
        scaler: StandardScaler pre-entrenado. Si None, se entrena uno nuevo con los datos
        return_scaler: si True, retorna también el scaler entrenado (útil para guardarlo y reutilizarlo)
        logger: logger opcional
        
    Returns:
        DataFrame con columnas escaladas, o tupla (DataFrame, StandardScaler) si return_scaler=True
        
    Examples:
        >>> # Entrenar y aplicar scaler
        >>> df_scaled = apply_standard_scaler(df, columns=['age', 'income'])
        
        >>> # Entrenar y guardar scaler para reutilizar
        >>> df_train_scaled, scaler = apply_standard_scaler(df_train, return_scaler=True)
        >>> df_test_scaled = apply_standard_scaler(df_test, scaler=scaler)
    """
    result = df.copy()
    
    # Determinar columnas a escalar
    if columns is None:
        numeric_cols, _ = detect_column_types(df)
        cols_to_scale = numeric_cols
    else:
        cols_to_scale = _ensure_list(columns, df.columns.tolist())
        # Filtrar solo las que son numéricas
        cols_to_scale = [c for c in cols_to_scale if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(cols_to_scale) == 0:
        _log(logger, "No se encontraron columnas numéricas para escalar.")
        return (result, scaler) if return_scaler and scaler is not None else result
    
    _log(logger, f"Escalando {len(cols_to_scale)} columnas numéricas con StandardScaler")
    
    # Extraer datos numéricos
    data_to_scale = result[cols_to_scale].values
    
    # Entrenar o aplicar scaler
    if scaler is None:
        # Entrenar nuevo scaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data_to_scale)
        _log(logger, f"Scaler entrenado y aplicado a {len(cols_to_scale)} columnas")
    else:
        # Aplicar scaler pre-entrenado
        scaled_data = scaler.transform(data_to_scale)
        _log(logger, f"Scaler pre-entrenado aplicado a {len(cols_to_scale)} columnas")
    
    # Reemplazar columnas escaladas en el DataFrame
    result[cols_to_scale] = scaled_data
    
    if return_scaler:
        return result, scaler
    return result


def fit_standard_scaler(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    logger: Optional[logging.Logger] = None,
) -> StandardScaler:
    """
    Entrena un StandardScaler con los datos del DataFrame (solo fit, sin transformar).
    
    Útil cuando se quiere entrenar el scaler en un conjunto de datos pero aplicarlo después.
    
    Args:
        df: DataFrame con datos de entrenamiento
        columns: columnas numéricas a usar para entrenar. Si None, detecta automáticamente todas las numéricas
        logger: logger opcional
        
    Returns:
        StandardScaler entrenado
        
    Examples:
        >>> scaler = fit_standard_scaler(df_train, columns=['age', 'income'])
        >>> df_test_scaled = apply_standard_scaler(df_test, scaler=scaler)
    """
    # Determinar columnas a usar
    if columns is None:
        numeric_cols, _ = detect_column_types(df)
        cols_to_use = numeric_cols
    else:
        cols_to_use = _ensure_list(columns, df.columns.tolist())
        cols_to_use = [c for c in cols_to_use if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(cols_to_use) == 0:
        _log(logger, "No se encontraron columnas numéricas para entrenar el scaler.")
        raise ValueError("No hay columnas numéricas para entrenar el scaler")
    
    _log(logger, f"Entrenando StandardScaler con {len(cols_to_use)} columnas numéricas")
    
    # Entrenar scaler
    scaler = StandardScaler()
    scaler.fit(df[cols_to_use].values)
    
    _log(logger, f"StandardScaler entrenado exitosamente")
    return scaler


def plot_kde_by_hue(
    df: pd.DataFrame,
    variable: str,
    hue_column: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    logger: Optional[logging.Logger] = None,
) -> Optional[plt.Figure]:
    """
    Genera un gráfico de densidad (KDE) de una variable numérica separado por valores de una columna hue.
    
    Args:
        df: DataFrame con los datos
        variable: nombre de la variable numérica a graficar
        hue_column: nombre de la columna para separar por color (hue)
        output_path: ruta opcional para guardar el gráfico. Si None, no guarda
        title: título opcional del gráfico
        figsize: tamaño de la figura (ancho, alto)
        logger: logger opcional
        
    Returns:
        Figura de matplotlib si output_path es None, None si se guardó
    """
    import seaborn as sns
    
    if variable not in df.columns:
        _log(logger, f"Variable '{variable}' no encontrada en DataFrame.")
        return None
    
    if hue_column not in df.columns:
        _log(logger, f"Columna hue '{hue_column}' no encontrada en DataFrame.")
        return None
    
    # Filtrar datos válidos
    df_plot = df[[variable, hue_column]].dropna(subset=[variable, hue_column])
    
    if df_plot.empty:
        _log(logger, f"No hay datos válidos para graficar '{variable}'.")
        return None
    
    # Verificar que hay suficientes valores únicos en hue
    unique_hues = df_plot[hue_column].nunique()
    if unique_hues < 2:
        _log(logger, f"Columna hue '{hue_column}' tiene menos de 2 valores únicos. Se omite.")
        return None
    
    # Convertir hue a string si no es categórica
    if df_plot[hue_column].dtype not in ['object', 'category']:
        df_plot[hue_column] = df_plot[hue_column].astype(str)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=figsize)
    
    # Obtener valores únicos del hue
    hue_values = sorted(df_plot[hue_column].unique())
    
    # Generar kde para cada valor del hue
    for hue_val in hue_values:
        data_subset = df_plot[df_plot[hue_column] == hue_val][variable].dropna()
        if len(data_subset) > 0:
            sns.kdeplot(
                data=data_subset,
                label=f'{hue_column}={hue_val}',
                ax=ax,
                fill=True,
                alpha=0.6
            )
    
    ax.set_xlabel(variable)
    ax.set_ylabel('Densidad')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Distribución de {variable} por {hue_column}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar o retornar
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _log(logger, f"Gráfico KDE guardado: {output_path}")
        return None
    else:
        return fig


def plot_categorical_target_mean(
    df: pd.DataFrame,
    categorical_var: str,
    target_column: str,
    output_path: Optional[str] = None,
    title: Optional[str] = None,
    max_categories: int = 20,
    figsize: Tuple[int, int] = (12, 6),
    logger: Optional[logging.Logger] = None,
) -> Optional[plt.Figure]:
    """
    Genera un gráfico de barras mostrando el promedio de una variable objetivo por categoría.
    
    Args:
        df: DataFrame con los datos
        categorical_var: nombre de la variable categórica
        target_column: nombre de la variable objetivo (numérica)
        output_path: ruta opcional para guardar el gráfico. Si None, no guarda
        title: título opcional del gráfico
        max_categories: máximo número de categorías a mostrar (top por promedio)
        figsize: tamaño de la figura (ancho, alto)
        logger: logger opcional
        
    Returns:
        Figura de matplotlib si output_path es None, None si se guardó
    """
    if categorical_var not in df.columns:
        _log(logger, f"Variable categórica '{categorical_var}' no encontrada en DataFrame.")
        return None
    
    if target_column not in df.columns:
        _log(logger, f"Variable objetivo '{target_column}' no encontrada en DataFrame.")
        return None
    
    # Filtrar datos válidos
    df_plot = df[[categorical_var, target_column]].dropna(subset=[categorical_var, target_column])
    
    if df_plot.empty:
        _log(logger, f"No hay datos válidos para graficar '{categorical_var}'.")
        return None
    
    # Convertir target a numérico
    target_numeric = pd.to_numeric(df_plot[target_column], errors='coerce')
    if target_numeric.isna().all():
        _log(logger, f"No se puede convertir '{target_column}' a numérico. Se omite.")
        return None
    
    # Calcular promedio del target por categoría
    avg_by_cat = df_plot.groupby(categorical_var)[target_column].apply(
        lambda x: pd.to_numeric(x, errors='coerce').mean()
    ).reset_index()
    avg_by_cat.columns = [categorical_var, 'promedio_target']
    avg_by_cat = avg_by_cat.sort_values('promedio_target', ascending=False)
    
    # Limitar a top categorías si hay muchas
    total_categories = len(avg_by_cat)
    if total_categories > max_categories:
        avg_by_cat = avg_by_cat.head(max_categories)
        _log(logger, f"Mostrando solo top {max_categories} categorías (de {total_categories} totales)")
    
    # Crear gráfico de barras
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(range(len(avg_by_cat)), avg_by_cat['promedio_target'], 
                 color='steelblue', alpha=0.7)
    
    ax.set_xlabel(categorical_var)
    ax.set_ylabel(f'Promedio de {target_column}')
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Promedio de {target_column} por {categorical_var}')
    ax.set_xticks(range(len(avg_by_cat)))
    ax.set_xticklabels(avg_by_cat[categorical_var], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Agregar valores en las barras
    for i, (idx, row) in enumerate(avg_by_cat.iterrows()):
        ax.text(i, row['promedio_target'], f"{row['promedio_target']:.3f}",
               ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar o retornar
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        _log(logger, f"Gráfico de barras guardado: {output_path}")
        return None
    else:
        return fig
def convert_to_logarithm(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    base: Union[str, float] = 'natural',
    handle_zeros: str = 'add_one',
    handle_negatives: str = 'skip',
    add_constant: float = 1.0,
    auto_detect: bool = False,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Convierte columnas numéricas a su logaritmo.
    
    Esta función permite:
    - Convertir columnas numéricas específicas a logaritmo
    - Elegir la base del logaritmo (natural, base 10, base 2, o base personalizada)
    - Manejar valores cero y negativos de diferentes formas
    
    Args:
        df: DataFrame a transformar
        columns: columnas numéricas a convertir. Si None y auto_detect=True, detecta todas las numéricas
        base: base del logaritmo:
            - 'natural' o 'ln' o 'e': logaritmo natural (np.log)
            - '10' o 'log10': logaritmo base 10 (np.log10)
            - '2' o 'log2': logaritmo base 2 (np.log2)
            - float: base personalizada (usa np.log(x) / np.log(base))
        handle_zeros: cómo manejar valores cero:
            - 'add_one': suma 1 antes de aplicar log (log(x+1)) - default
            - 'add_constant': suma add_constant antes de aplicar log
            - 'skip': no convierte columnas con ceros
            - 'nan': convierte ceros a NaN
        handle_negatives: cómo manejar valores negativos:
            - 'skip': no convierte columnas con negativos - default
            - 'abs': usa valor absoluto antes de aplicar log
            - 'nan': convierte negativos a NaN
            - 'add_constant': suma add_constant a todos los valores antes de aplicar log
        add_constant: constante a sumar cuando handle_zeros='add_constant' o handle_negatives='add_constant'
        auto_detect: si True y columns=None, detecta automáticamente todas las columnas numéricas
        logger: logger opcional
        
    Returns:
        DataFrame con columnas convertidas a logaritmo (nombres originales se mantienen)
        
    Examples:
        >>> # Convertir columnas específicas a logaritmo natural
        >>> df = convert_to_logarithm(df, columns=['income', 'sales'], base='natural')
        
        >>> # Convertir a logaritmo base 10, manejando ceros sumando 1
        >>> df = convert_to_logarithm(df, columns=['amount'], base='10', handle_zeros='add_one')
        
        >>> # Convertir todas las numéricas automáticamente
        >>> df = convert_to_logarithm(df, auto_detect=True, base='natural')
    """
    result = df.copy()
    
    # Determinar columnas a convertir
    if columns is None:
        if auto_detect:
            numeric_cols, _ = detect_column_types(df)
            cols_to_convert = numeric_cols
            _log(logger, f"Auto-detección: {len(cols_to_convert)} columnas numéricas detectadas para conversión a logaritmo")
        else:
            cols_to_convert = []
            _log(logger, "No se especificaron columnas y auto_detect=False. No se realizará conversión.")
    else:
        cols_to_convert = _ensure_list(columns, df.columns.tolist())
        # Filtrar solo las que son numéricas
        cols_to_convert = [c for c in cols_to_convert if pd.api.types.is_numeric_dtype(result[c])]
    
    if len(cols_to_convert) == 0:
        _log(logger, "No hay columnas numéricas para convertir a logaritmo.")
        return result
    
    # Determinar función de logaritmo según la base
    if isinstance(base, str):
        base_lower = base.lower()
        if base_lower in ('natural', 'ln', 'e'):
            log_func = np.log
            base_name = 'natural (ln)'
        elif base_lower in ('10', 'log10'):
            log_func = np.log10
            base_name = '10'
        elif base_lower in ('2', 'log2'):
            log_func = np.log2
            base_name = '2'
        else:
            raise ValueError(f"Base no reconocida: {base}. Use 'natural', '10', '2', o un float")
    else:
        # Base personalizada
        base_float = float(base)
        if base_float <= 0 or base_float == 1:
            raise ValueError(f"Base debe ser positiva y diferente de 1. Recibido: {base_float}")
        log_func = lambda x: np.log(x) / np.log(base_float)
        base_name = str(base_float)
    
    _log(logger, f"Convirtiendo {len(cols_to_convert)} columnas a logaritmo base {base_name}")
    
    # Convertir cada columna
    converted_count = 0
    skipped_count = 0
    
    for col in cols_to_convert:
        if col not in result.columns:
            _log(logger, f"Advertencia: Columna '{col}' no existe en el DataFrame. Se omite.")
            continue
        
        try:
            series = result[col].copy()
            
            # Verificar y manejar valores negativos
            has_negatives = (series < 0).any()
            if has_negatives:
                if handle_negatives == 'skip':
                    _log(logger, f"Columna '{col}' tiene valores negativos. Se omite (handle_negatives='skip').")
                    skipped_count += 1
                    continue
                elif handle_negatives == 'abs':
                    series = series.abs()
                    _log(logger, f"Columna '{col}': valores negativos convertidos a absoluto")
                elif handle_negatives == 'nan':
                    series = series.where(series >= 0)
                    _log(logger, f"Columna '{col}': valores negativos convertidos a NaN")
                elif handle_negatives == 'add_constant':
                    min_val = series.min()
                    if min_val < 0:
                        constant_to_add = abs(min_val) + add_constant
                        series = series + constant_to_add
                        _log(logger, f"Columna '{col}': se sumó {constant_to_add} para manejar negativos")
            
            # Verificar y manejar valores cero o negativos
            has_zeros = (series <= 0).any()
            if has_zeros:
                if handle_zeros == 'skip':
                    _log(logger, f"Columna '{col}' tiene valores <= 0. Se omite (handle_zeros='skip').")
                    skipped_count += 1
                    continue
                elif handle_zeros == 'add_one':
                    series = series + 1.0
                    _log(logger, f"Columna '{col}': se sumó 1 para manejar ceros (log1p)")
                elif handle_zeros == 'add_constant':
                    series = series + add_constant
                    _log(logger, f"Columna '{col}': se sumó {add_constant} para manejar ceros")
                elif handle_zeros == 'nan':
                    series = series.where(series > 0)
                    _log(logger, f"Columna '{col}': valores <= 0 convertidos a NaN")
            
            # Aplicar logaritmo
            result[col] = log_func(series)
            converted_count += 1
            _log(logger, f"Columna '{col}' convertida a logaritmo base {base_name}")
            
        except Exception as e:
            _log(logger, f"Error convirtiendo columna '{col}' a logaritmo: {e}")
            skipped_count += 1
    
    _log(logger, f"Conversión a logaritmo completada: {converted_count} convertidas, {skipped_count} omitidas")
    
    return result


