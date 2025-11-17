import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from feature_pro.common.utils import (
    detect_column_types,
    compute_correlation_matrix,
    _point_biserial_safe,
    _cramers_v,
    _correlation_ratio,
)

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """
    Envía un mensaje al logger si está disponible.
    """
    if logger is not None:
        logger.info(message)

def _ensure_list(columns: Optional[Union[str, Iterable[str]]], all_columns: List[str]) -> List[str]:
    """
    Normaliza 'columns' a lista válida dentro de 'all_columns'.
    """
    if columns is None:
        return list(all_columns)
    if isinstance(columns, str):
        return [columns]
    return [c for c in columns if c in all_columns]

def _plot_variance_curve(cum_var: np.ndarray, title: str = "Varianza acumulada PCA"):
    """
    Genera figura con la curva de varianza acumulada del PCA.
    Args:
        cum_var: arreglo con la varianza acumulada por número de componentes (cumsum).
        title: título del gráfico.
    Returns:
        matplotlib.figure.Figure generado.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza acumulada")
    ax.set_ylim(0, 1.01)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    return fig

def coefficient_of_variation(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
) -> pd.DataFrame:
    """
    Calcula el coeficiente de variación (CV = |std/mean|) por columna numérica.
    Si columns es None, detecta numéricas automáticamente.
    Args:
        df: DataFrame de entrada.
        columns: columnas a evaluar; si None, se detectan numéricas automáticamente.
    Returns:
        DataFrame con columnas ['column','cv'] ordenado ascendentemente por cv.
    """
    if columns is None:
        cols, _ = detect_column_types(df)
    else:
        cols = _ensure_list(columns, df.columns.tolist())
    data: List[Tuple[str, float]] = []
    for col in cols:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        mean = s.mean()
        std = s.std()
        cv = np.nan if mean == 0 or pd.isna(mean) else abs(float(std) / float(mean))
        data.append((col, cv))
    return pd.DataFrame(data, columns=['column', 'cv']).sort_values('cv', ascending=True).reset_index(drop=True)

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
    Ajusta PCA con opciones de:
      - n_components fijo, o
      - selección por varianza acumulada (variance_threshold).
    Puede escalar, devolver figura de varianza acumulada y marcar el n elegido.
    Args:
        df: DataFrame con las variables.
        columns: columnas a incluir en el PCA.
        n_components: número fijo de componentes a retener. Si None y variance_threshold se define, se ignora y se calcula.
        variance_threshold: proporción de varianza acumulada objetivo (0,1]; se calcula el n mínimo que la alcanza.
        scale: si True, aplica StandardScaler antes del PCA.
        plot: si True, genera la figura de varianza acumulada.
        return_fig: si True, incluye la figura en el retorno.
        random_state: semilla para reproducibilidad del PCA.
    Returns:
        dict con:
          - 'pca': objeto PCA ajustado
          - 'n_components': n usado
          - 'explained_variance_ratio': vector de varianza por componente
          - 'cumulative_variance': varianza acumulada completa
          - 'figure': figura opcional
    """
    cols = _ensure_list(columns, df.columns.tolist())
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    if scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

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
        chosen_n = min(10, X.shape[1])

    pca = PCA(n_components=chosen_n, random_state=random_state)
    pca.fit(X)
    cum_var_chosen = pca_full.explained_variance_ratio_.cumsum()

    fig = _plot_variance_curve(cum_var_chosen) if plot and return_fig else None
    if fig is not None and variance_threshold is not None and chosen_n is not None and chosen_n >= 1:
        ax = fig.axes[0] if fig.axes else None
        if ax is not None:
            ax.axvline(chosen_n, color='red', linestyle='--', linewidth=1.5, alpha=0.8)
            y_val = cum_var_chosen[chosen_n - 1] if chosen_n - 1 < len(cum_var_chosen) else variance_threshold
            ax.annotate(
                f"n={chosen_n} (>= {int(variance_threshold*100)}%)",
                xy=(chosen_n, y_val),
                xytext=(chosen_n + max(1, int(0.02*len(cum_var_chosen))), min(1.0, y_val + 0.05)),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.0),
                fontsize=9,
                color='red'
            )
    return {
        'pca': pca,
        'n_components': chosen_n,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': cum_var_chosen,
        'figure': fig,
    }


def pca_lda_importance(
    df: pd.DataFrame,
    feature_columns: Union[str, Iterable[str]],
    target_column: str,
    n_pca_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    scale: bool = True,
    plot: bool = False,
    return_fig: bool = True,
    random_state: Optional[int] = None,
    top_n: Optional[int] = None,
) -> Dict[str, Union[pd.DataFrame, Optional['plt.Figure']]]:
    """
    Calcula importancia de variables mediante PCA+LDA:
      1) PCA sobre features (n fijo o por varianza acumulada)
      2) LDA en espacio PCA
      3) Proyección de coeficientes LDA a espacio original (|coef_LDA · loadings|)
    Retorna ranking y, opcionalmente, un gráfico de barras.
    Args:
        df: DataFrame con variables y target.
        feature_columns: columnas a considerar como features.
        target_column: nombre de la columna objetivo (clasificación binaria o multiclase).
        n_pca_components: número de componentes PCA a usar (si None y variance_threshold definido, se estima).
        variance_threshold: varianza acumulada objetivo para estimar n (si n_pca_components es None).
        scale: si True, aplica StandardScaler a X antes de PCA.
        plot: si True, genera gráfico de barras con importancias.
        return_fig: si True, incluye la figura en el retorno.
        random_state: semilla para reproducibilidad.
        top_n: si no es None, recorta el ranking al top_n.
    Returns:
        dict con:
          - 'ranking': DataFrame ['variable','importance'] ordenado desc
          - 'figure': figura opcional del ranking
    """
    cols = [feature_columns] if isinstance(feature_columns, str) else list(feature_columns)
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    if scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

    pca_full = PCA(random_state=random_state)
    pca_full.fit(X)
    cum_var = pca_full.explained_variance_ratio_.cumsum()
    chosen_n = n_pca_components
    if variance_threshold is not None:
        if not (0 < variance_threshold <= 1):
            raise ValueError("variance_threshold debe estar en (0, 1].")
        chosen_n = int(np.argmax(cum_var >= variance_threshold) + 1)
        chosen_n = max(1, chosen_n)
    if chosen_n is None:
        chosen_n = min(10, X.shape[1])

    pca = PCA(n_components=chosen_n, random_state=random_state)
    X_pca = pca.fit_transform(X)

    y = df[target_column]
    lda = LDA(n_components=1)
    lda.fit(X_pca, y)
    lda_coef = lda.coef_[0]
    loadings = pca.components_
    importance = np.abs(np.dot(lda_coef, loadings))
    ranking = pd.DataFrame({'variable': cols, 'importance': importance}).sort_values(
        by='importance', ascending=False
    )
    if top_n is not None:
        ranking = ranking.head(top_n).reset_index(drop=True)
    else:
        ranking = ranking.reset_index(drop=True)

    fig = None
    if plot and (plt is not None) and (sns is not None):
        fig, ax = plt.subplots(figsize=(8, max(3, len(ranking) * 0.3)))
        sns.barplot(data=ranking, x='importance', y='variable', orient='h', ax=ax, color='#4C78A8')
        ax.set_title("Importancia de variables (PCA+LDA)")
        ax.set_xlabel("Importancia (|coef_LDA · loadings|)")
        ax.set_ylabel("")
        plt.tight_layout()

    return {'ranking': ranking, 'figure': fig}


def _feature_target_association(
    df: pd.DataFrame,
    feature: str,
    target: str,
    method: str = 'auto',
) -> float:
    """
    Mide asociación absoluta feature-target según el método:
      - 'pointbiserial' para binaria vs numérica
      - 'pearson'/'spearman'/'kendall' para num-num
      - auto: num-num (pearson), cat-cat (Cramér's V), cat-num (eta)
    Args:
        df: DataFrame.
        feature: nombre de la variable a evaluar.
        target: nombre de la variable objetivo.
        method: método de asociación ('pointbiserial', 'pearson', 'spearman', 'kendall' o 'auto').
    Returns:
        Valor absoluto de la asociación (float) o NaN si no es aplicable.
    """
    x = df[feature]
    y = df[target]
    x_num = pd.api.types.is_numeric_dtype(x)
    y_num = pd.api.types.is_numeric_dtype(y)

    # Punto-biserial explícito
    if method == 'pointbiserial':
        return float(abs(_point_biserial_safe(x.values, y.values)))

    if method in ('pearson', 'spearman', 'kendall'):
        if not (x_num and y_num):
            return np.nan
        corr = df[[feature, target]].corr(method=method).iloc[0, 1]
        return float(abs(corr))

    if x_num and y_num:
        corr = df[[feature, target]].corr(method='pearson').iloc[0, 1]
        return float(abs(corr))
    if (not x_num) and (not y_num):
        cont = pd.crosstab(x, y)
        if cont.size == 0:
            return np.nan
        return float(abs(_cramers_v(cont)))
    if x_num and (not y_num):
        return float(abs(_correlation_ratio(y.values, x.values)))
    if (not x_num) and y_num:
        return float(abs(_correlation_ratio(x.values, y.values)))
    return np.nan


def select_by_correlation_graph(
    df: pd.DataFrame,
    columns: Optional[Union[str, Iterable[str]]] = None,
    target_column: Optional[str] = None,
    method: str = 'auto',
    target_method: str = 'auto',
    threshold: float = 0.6,
    prioritize: Optional[Iterable[str]] = None,
    plot_matrix: bool = False,
    return_matrix_fig: bool = True,
    return_graph: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[List[str], pd.DataFrame, Optional['plt.Figure'], Optional['nx.Graph']]]:
    """
    Selección por redundancia mediante grafo de correlaciones:
      - Crea aristas con |correlación| >= threshold (método entre variables: 'method')
      - Por cada componente conexo, conserva:
         • variable priorizada si existe, o
         • variable con mayor asociación al target (target_method), o
         • mayor conectividad como fallback
      - Retorna variables seleccionadas, matriz, figura y grafo (opcionales)
    Args:
        df: DataFrame con datos.
        columns: columnas a evaluar; si None, usa todas.
        target_column: objetivo para decidir dentro de cada grupo.
        method: método de correlación entre variables ('auto','pearson','spearman','kendall').
        target_method: método de asociación con el objetivo (p.ej. 'pointbiserial').
        threshold: umbral de |correlación| para crear aristas.
        prioritize: lista de variables a priorizar (si están en un grupo, se eligen).
        plot_matrix: si True, genera heatmap de correlación.
        return_matrix_fig: si True, retorna la figura del heatmap.
        return_graph: si True, retorna el grafo networkx.
        logger: logger opcional.
    Returns:
        dict con:
          - 'selected': lista de variables seleccionadas
          - 'matrix': DataFrame de correlación
          - 'matrix_figure': figura opcional
          - 'graph': grafo opcional
    """
    cols = [columns] if isinstance(columns, str) else (list(columns) if columns is not None else df.columns.tolist())
    corr_out = compute_correlation_matrix(df, columns=cols, method=method, plot=plot_matrix, return_fig=return_matrix_fig, logger=logger)
    mat = corr_out['matrix']
    fig = corr_out['figure']

    selected: List[str] = []
    prioritized = set(prioritize or [])

    graph_obj = None
    if nx is not None:
        G = nx.Graph()
        G.add_nodes_from(cols)
        for i, ci in enumerate(cols):
            for j, cj in enumerate(cols):
                if j <= i:
                    continue
                val = mat.loc[ci, cj]
                if pd.isna(val):
                    continue
                if abs(float(val)) >= threshold:
                    G.add_edge(ci, cj, weight=float(val))
        graph_obj = G
        components = list(nx.connected_components(G))
    else:
        seen = set()
        components = []
        for c in cols:
            if c in seen:
                continue
            group = {c}
            queue = [c]
            seen.add(c)
            while queue:
                v = queue.pop()
                related = [
                    u for u in cols
                    if (u not in group)
                    and (not pd.isna(mat.loc[v, u]))
                    and (abs(float(mat.loc[v, u])) >= threshold)
                ]
                for u in related:
                    if u not in seen:
                        seen.add(u)
                        group.add(u)
                        queue.append(u)
            components.append(group)

    for comp in components:
        comp_list = list(comp)
        prios = [v for v in comp_list if v in prioritized]
        if len(prios) > 0:
            selected.append(prios[0])
            continue

        if target_column is None:
            if nx is not None and graph_obj is not None:
                degrees = [(v, graph_obj.degree[v]) for v in comp_list]
                chosen = sorted(degrees, key=lambda x: x[1], reverse=True)[0][0]
            else:
                chosen = comp_list[0]
            selected.append(chosen)
            continue

        scores = []
        for v in comp_list:
            s = _feature_target_association(df, v, target_column, method=target_method if target_method else method)
            scores.append((v, (s if not pd.isna(s) else -np.inf)))
        chosen = sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
        selected.append(chosen)

    for v in prioritized:
        if v in cols and v not in selected:
            selected.append(v)

    _log(logger, f"Seleccionadas {len(selected)} variables tras redundancia por correlación (threshold={threshold}).")
    return {
        'selected': selected,
        'matrix': mat,
        'matrix_figure': fig if return_matrix_fig else None,
        'graph': graph_obj if return_graph else None,
    }

def compute_psi(
    df: pd.DataFrame,
    temporal_column: str,
    variables: Optional[Union[str, Iterable[str]]] = None,
    num_bins: int = 10,
    bin_strategy: str = 'quantile',
    reference_period: Optional[Union[str, int, float, pd.Timestamp]] = None,
    return_detail: bool = False,
    logger: Optional[logging.Logger] = None,
    epsilon: float = 1e-8,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Calcula PSI (Population Stability Index) para evaluar estabilidad a través del tiempo.
    - Para numéricas, binea a partir del periodo de referencia (quantile/equal con fallback).
    - Para categóricas, compara proporciones por categoría (incluye 'MISSING' si aplica).
    Args:
        df: DataFrame con datos.
        temporal_column: columna que define los periodos (agrupador temporal).
        variables: columnas a evaluar; si None, todas excepto la temporal.
        num_bins: número de bins para variables numéricas.
        bin_strategy: estrategia de binning para numéricas: 'quantile' o 'equal'.
        reference_period: periodo de referencia; si None, se usa el primero (ordenado).
        return_detail: si True, retorna detalle por periodo además del resumen.
        logger: logger opcional.
        epsilon: suavizado numérico para evitar log(0).
    Returns:
        Si return_detail=False:
          DataFrame ['variable','psi'] con el PSI medio vs referencia.
        Si return_detail=True:
          (summary_df, detail_df) donde detail_df = ['variable','period','psi'].
    """
    _log(logger, "Iniciando cálculo de PSI...")
    if temporal_column not in df.columns:
        raise ValueError(f"Columna temporal '{temporal_column}' no encontrada en el DataFrame.")
    periods_sorted = pd.Index(sorted(df[temporal_column].dropna().unique()))
    if len(periods_sorted) < 2:
        raise ValueError("Se requieren al menos 2 periodos para calcular PSI.")
    ref_period = reference_period if reference_period is not None else periods_sorted[0]
    if ref_period not in periods_sorted:
        raise ValueError(f"reference_period '{ref_period}' no existe en la columna temporal.")
    _log(logger, f"Periodo de referencia PSI: {ref_period}")

    if variables is None:
        variables_list = [c for c in df.columns if c != temporal_column]
    else:
        variables_list = _ensure_list(variables, df.columns.tolist())
        variables_list = [c for c in variables_list if c != temporal_column]
    _log(logger, f"Variables a evaluar PSI: {len(variables_list)}")

    df_ref = df[df[temporal_column] == ref_period]
    compare_periods = [p for p in periods_sorted if p != ref_period]

    def _psi_from_proportions(p_ref: pd.Series, p_cur: pd.Series) -> float:
        idx = p_ref.index.union(p_cur.index)
        pr = p_ref.reindex(idx).fillna(0.0) + epsilon
        pc = p_cur.reindex(idx).fillna(0.0) + epsilon
        return float(((pr - pc) * np.log(pr / pc)).sum())

    detail_records: List[Tuple[str, Union[str, int, float, pd.Timestamp], float]] = []
    summary_records: List[Tuple[str, float]] = []

    for col in variables_list:
        s_ref = df_ref[col]
        is_num = pd.api.types.is_numeric_dtype(s_ref)

        if is_num:
            ref_non_null = s_ref.dropna()
            if len(ref_non_null) == 0:
                per_var_psis = [np.nan] * len(compare_periods)
            else:
                if bin_strategy == 'quantile':
                    try:
                        quantiles = np.linspace(0, 1, num_bins + 1)
                        edges = np.unique(ref_non_null.quantile(quantiles).values)
                        if len(edges) <= 2:
                            bin_strategy_use = 'equal'
                            _log(logger, f"[PSI] Bins por quantiles colapsaron en '{col}'. Fallback a 'equal'.")
                        else:
                            bin_strategy_use = 'quantile'
                    except Exception as e:
                        _log(logger, f"[PSI] Error creando bins por quantiles para '{col}': {e}. Fallback a 'equal'.")
                        bin_strategy_use = 'equal'
                else:
                    bin_strategy_use = 'equal'

                if bin_strategy_use == 'equal':
                    mn, mx = ref_non_null.min(), ref_non_null.max()
                    if pd.isna(mn) or pd.isna(mx) or mn == mx:
                        edges = np.array([mn - 1, mx + 1], dtype=float)
                        _log(logger, f"[PSI] Rango degenerado para '{col}' en referencia. Usando bordes extendidos para bins.")
                    else:
                        edges = np.linspace(mn, mx, num_bins + 1)

                ref_bins = pd.cut(s_ref, bins=edges, include_lowest=True, duplicates='drop')
                ref_counts = ref_bins.value_counts(normalize=True, dropna=False)
                ref_counts.index = ref_counts.index.astype(object)
                ref_counts = ref_counts.rename(index={np.nan: 'MISSING'})

                per_var_psis = []
                for period in compare_periods:
                    s_cur = df.loc[df[temporal_column] == period, col]
                    cur_bins = pd.cut(s_cur, bins=edges, include_lowest=True, duplicates='drop')
                    cur_counts = cur_bins.value_counts(normalize=True, dropna=False)
                    cur_counts.index = cur_counts.index.astype(object)
                    cur_counts = cur_counts.rename(index={np.nan: 'MISSING'})
                    psi_val = _psi_from_proportions(ref_counts, cur_counts)
                    per_var_psis.append(psi_val)
                    detail_records.append((col, period, psi_val))
        else:
            sref = df_ref[col]
            if pd.api.types.is_categorical_dtype(sref):
                sref = sref.astype('object')
            ref_counts = sref.fillna('MISSING').value_counts(normalize=True)
            per_var_psis = []
            for period in compare_periods:
                scur = df.loc[df[temporal_column] == period, col]
                if pd.api.types.is_categorical_dtype(scur):
                    scur = scur.astype('object')
                cur_counts = scur.fillna('MISSING').value_counts(normalize=True)
                psi_val = _psi_from_proportions(ref_counts, cur_counts)
                per_var_psis.append(psi_val)
                detail_records.append((col, period, psi_val))

        psi_mean = float(np.nanmean(per_var_psis)) if len(per_var_psis) > 0 else np.nan
        summary_records.append((col, psi_mean))

    summary_df = pd.DataFrame(summary_records, columns=['variable', 'psi']).sort_values('psi', ascending=False).reset_index(drop=True)
    detail_df = pd.DataFrame(detail_records, columns=['variable', 'period', 'psi']).sort_values(['variable', 'period']).reset_index(drop=True)
    _log(logger, "Cálculo de PSI finalizado.")
    return (summary_df, detail_df) if return_detail else summary_df


