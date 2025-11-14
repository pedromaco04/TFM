import logging
from typing import Iterable, List, Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Plotting is optional; figures can be returned to ser guardados por el usuario
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None


def _log(logger: Optional[logging.Logger], message: str) -> None:
    if logger is not None:
        logger.info(message)


def _ensure_list(columns: Optional[Union[str, Iterable[str]]], all_columns: List[str]) -> List[str]:
    if columns is None:
        return list(all_columns)
    if isinstance(columns, str):
        return [columns]
    return [c for c in columns if c in all_columns]


def summarize_missing(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame resumen con conteo y porcentaje de missings por columna.
    Args:
        df: DataFrame a analizar
        columns: Columnas a incluir (si None, toma todas)
    Returns:
        DataFrame con columnas: ['column', 'dtype', 'num_missing', 'pct_missing']
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
) -> Tuple[List[str], List[str]]:
    """
    Identifica columnas numéricas y categóricas, incluso si numéricas están como 'object'.
    Args:
        df: DataFrame fuente
        columns: subconjunto opcional de columnas a analizar
        treat_object_numeric: si True, intenta convertir objetos numéricos a numérico
        sample_size_for_analysis: tamaño de muestra para validar patrones de string numéricos
        max_code_length: si un valor excede, se asume código identificador (no numérico continuo)
    Returns:
        (numeric_columns, categorical_columns)
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
                # Vacía → tratar como categórica
                categorical_cols.append(col)
                continue
            sample = non_null.astype(str).head(sample_size_for_analysis)
            contains_letters = any(any(ch.isalpha() for ch in val) for val in sample)
            looks_like_code = any(len(str(val)) > max_code_length for val in sample)
            if not contains_letters and not looks_like_code:
                # Probar conversión estricta
                coerced = pd.to_numeric(non_null, errors='coerce')
                if coerced.notna().all():
                    numeric_cols.append(col)
                    continue

        # Por defecto: categórica
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
    Imputa valores faltantes de forma genérica. Soporta imputación con groupby.
    Estrategias soportadas (para columnas numéricas):
      - 'mean' | 'median' | 'zero' | 'constant'
    Para 'constant' use fill_value. En columnas no numéricas, 'zero', 'mean' y 'median'
    no aplican; para ellas use 'constant' o llene por fuera.
    Args:
        df: DataFrame de entrada
        columns: columnas a imputar (por defecto detecta numéricas)
        strategy: estrategia de imputación
        fill_value: valor a usar si strategy='constant'
        groupby: columna(s) para imputación por grupo (opcional)
        logger: logger para enviar mensajes al log (no imprime a consola)
    Returns:
        DataFrame con imputación aplicada (copia)
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
        # mean / median solo tienen sentido numéricamente
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
) -> pd.DataFrame:
    """
    Winsorización por percentiles; recorta por debajo de 'lower' y por encima de 'upper'.
    Args:
        df: DataFrame
        columns: columnas numéricas a winsorizar
        lower: percentil inferior (0-1)
        upper: percentil superior (0-1)
    """
    cols = _ensure_list(columns, df.columns.tolist())
    result = df.copy()
    _log(logger, f"Winsorizando por percentiles columnas={len(cols)} lower={lower} upper={upper}")
    for col in cols:
        if not pd.api.types.is_numeric_dtype(result[col]):
            continue
        q_low = result[col].quantile(lower)
        q_hi = result[col].quantile(upper)
        result[col] = result[col].clip(lower=q_low, upper=q_hi)
    return result


def winsorize_by_iqr(
    df: pd.DataFrame,
    columns: Union[str, Iterable[str]],
    factor: float = 1.5,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Winsorización por IQR; recorta a [Q1 - factor*IQR, Q3 + factor*IQR].
    Args:
        df: DataFrame
        columns: columnas numéricas a winsorizar
        factor: multiplicador del IQR
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


def coefficient_of_variation(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Calcula el coeficiente de variación (CV = |std/mean|) por columna.
    Args:
        df: DataFrame
        columns: columnas a evaluar (por defecto, numéricas detectadas)
    Returns:
        DataFrame con ['column', 'cv']
    """
    if columns is None:
        cols, _ = detect_column_types(df)
    else:
        cols = _ensure_list(columns, df.columns.tolist())
    data = []
    for col in cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        mean = s.mean()
        std = s.std()
        cv = np.nan if mean == 0 or pd.isna(mean) else abs(float(std) / float(mean))
        data.append((col, cv))
    return pd.DataFrame(data, columns=['column', 'cv']).sort_values('cv', ascending=True).reset_index(drop=True)


def _plot_variance_curve(cum_var: np.ndarray, title: str = "Varianza acumulada PCA"):
    if plt is None or sns is None:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return fig


def compute_pca(
    df: pd.DataFrame,
    columns: Union[str, Iterable[str]],
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    scale: bool = True,
    plot: bool = False,
    return_fig: bool = True,
    random_state: Optional[int] = None,
) -> Dict[str, Union[int, np.ndarray, PCA, Optional['plt.Figure']]]:
    """
    Calcula PCA genérico. Permite:
      - fijar n_components, o
      - calcular n para alcanzar variance_threshold (p.ej. 0.95)
      - opcionalmente escalar (StandardScaler)
      - opcionalmente devolver figura del gráfico de varianza acumulada
    Returns:
      dict con:
        'pca': objeto PCA ajustado
        'n_components': int usado
        'explained_variance_ratio': np.ndarray
        'cumulative_variance': np.ndarray
        'figure': figura (opcional)
    """
    cols = _ensure_list(columns, df.columns.tolist())
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    if scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

    # Si se pide variance_threshold, primero PCA completo para estimar n
    pca_full = PCA(random_state=random_state)
    pca_full.fit(X)
    cum_var = pca_full.explained_variance_ratio_.cumsum()
    chosen_n = n_components
    if variance_threshold is not None:
        if not (0 < variance_threshold <= 1):
            raise ValueError("variance_threshold debe estar en (0, 1].")
        chosen_n = int(np.argmax(cum_var >= variance_threshold) + 1)
        chosen_n = max(1, chosen_n)
    if chosen_n is None:
        chosen_n = min(10, X.shape[1])  # default conservador

    pca = PCA(n_components=chosen_n, random_state=random_state)
    pca.fit(X)
    cum_var_chosen = pca_full.explained_variance_ratio_.cumsum()  # curva completa

    fig = _plot_variance_curve(cum_var_chosen) if plot and return_fig else None
    return {
        'pca': pca,
        'n_components': chosen_n,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cum_var_chosen,
        'figure': fig,
    }


# ------------------------- Correlaciones avanzadas -------------------------

def _cramers_v(conf_mat: pd.DataFrame) -> float:
    """Calcula Cramér's V a partir de una tabla de contingencia."""
    chi2 = stats.chi2_contingency(conf_mat)[0]
    n = conf_mat.values.sum()
    r, k = conf_mat.shape
    return float(np.sqrt(chi2 / (n * (min(k - 1, r - 1) if min(k, r) > 1 else 1))))


def _correlation_ratio(categories: np.ndarray, measurements: np.ndarray) -> float:
    """
    Correlation Ratio (eta) para categórica vs numérica.
    categories: array-like categórico
    measurements: array-like numérico
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
    """Point-biserial para binaria vs numérica (manejo robusto si y no es estrictamente 0/1)."""
    y_bin = pd.Series(y).dropna()
    if y_bin.nunique() != 2:
        return np.nan
    s = pd.Series(x).dropna()
    # Alinear índices
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
    Calcula una matriz de correlaciones flexible.
    Métodos:
      - 'pearson' (solo numéricas)
      - 'spearman' (solo numéricas)
      - 'kendall' (solo numéricas)
      - 'auto' (mixto): 
           num-num: pearson
           cat-cat: Cramér's V
           cat-num: correlation ratio (eta)
    Args:
        df: DataFrame
        columns: subconjunto de columnas (por defecto todas)
        method: ver arriba
        plot: si True, genera heatmap (devuelto como figura)
        return_fig: si True, retorna la figura en el dict de salida
        logger: logger para mensajes
    Returns:
        {'matrix': DataFrame, 'figure': Optional[Figure]}
    """
    cols = _ensure_list(columns, df.columns.tolist())
    if method in ('pearson', 'spearman', 'kendall'):
        num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        mat = df[num_cols].corr(method=method)
    else:
        # modo 'auto' mixto
        num_cols, cat_cols = detect_column_types(df, columns=cols)
        mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if j < i:
                    continue
                if ci == cj:
                    mat.loc[ci, cj] = 1.0
                    continue
                # Casos
                if (ci in num_cols) and (cj in num_cols):
                    r = df[[ci, cj]].corr(method='pearson').iloc[0, 1]
                elif (ci in cat_cols) and (cj in cat_cols):
                    cont = pd.crosstab(df[ci], df[cj])
                    r = _cramers_v(cont) if cont.size > 0 else np.nan
                else:
                    # mixto
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


