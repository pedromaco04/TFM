import logging
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler

from .utils import (
    detect_column_types,
    compute_correlation_matrix,
    _point_biserial_safe,  # internal helper from utils
    _cramers_v,            # internal helper from utils
    _correlation_ratio,    # internal helper from utils
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover
    plt = None
    sns = None

try:
    import networkx as nx
except Exception:  # pragma: no cover
    nx = None


def _log(logger: Optional[logging.Logger], message: str) -> None:
    if logger is not None:
        logger.info(message)


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
    Calcula importancia de variables via PCA + LDA.
    Pasos:
      1) Escalado opcional
      2) PCA (n fijo o determinado por variance_threshold)
      3) LDA sobre componentes PCA (clasificación)
      4) Importancia en espacio original: |coef_LDA · loadings_PCA|
    Args:
        df: DataFrame con features y target
        feature_columns: columnas de características (numéricas)
        target_column: variable objetivo (clase)
        n_pca_components: número de componentes PCA
        variance_threshold: si se indica, se calcula n para alcanzar ese umbral en varianza
        scale: aplica StandardScaler antes de PCA
        plot: si True, grafica barras de importancia
        return_fig: si True, retorna figura
        random_state: semilla
        top_n: si se indica, recorta a top_n variables más importantes en la salida
    Returns:
        {'ranking': DataFrame(variable, importance), 'figure': Optional[Figure]}
    """
    cols = [feature_columns] if isinstance(feature_columns, str) else list(feature_columns)
    X = df[cols].copy()
    X = X.fillna(X.median(numeric_only=True))
    if scale:
        X = StandardScaler().fit_transform(X)
    else:
        X = X.values

    # Ajuste PCA completo para curva y determinar n si aplica
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
    lda_coef = lda.coef_[0]  # tamaño: n_components
    loadings = pca.components_  # shape: (n_components, n_features)
    # Importancia en variables originales (abs proyección)
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
    """Calcula asociación absoluta feature-target siguiendo la misma lógica de correlación 'auto'."""
    x = df[feature]
    y = df[target]
    x_num = pd.api.types.is_numeric_dtype(x)
    y_num = pd.api.types.is_numeric_dtype(y)

    # Métodos numéricos explícitos
    if method in ('pearson', 'spearman', 'kendall'):
        if not (x_num and y_num):
            return np.nan
        corr = df[[feature, target]].corr(method=method).iloc[0, 1]
        return float(abs(corr))

    # auto
    if x_num and y_num:
        corr = df[[feature, target]].corr(method='pearson').iloc[0, 1]
        return float(abs(corr))
    if (not x_num) and (not y_num):
        cont = pd.crosstab(x, y)
        if cont.size == 0:
            return np.nan
        return float(abs(_cramers_v(cont)))
    # mixto
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
    threshold: float = 0.6,
    prioritize: Optional[Iterable[str]] = None,
    plot_matrix: bool = False,
    return_matrix_fig: bool = True,
    return_graph: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Union[List[str], pd.DataFrame, Optional['plt.Figure'], Optional['nx.Graph']]]:
    """
    Selección por redundancia usando grafo de correlaciones:
      1) Calcula matriz de asociación (método 'auto' o numérico clásico)
      2) Construye grafo con aristas si |asociación| >= threshold
      3) Encuentra componentes conectados (grupos)
      4) De cada grupo, elige UNA variable:
           - Si el grupo contiene variable priorizada, se elige esa.
           - En caso contrario, se elige la de mayor asociación absoluta con el target.
      5) Incluye también nodos aislados.
    Args:
        df: DataFrame
        columns: columnas a evaluar (por defecto todas)
        target_column: objetivo (opcional, pero necesario si no hay priorizadas y hay empate)
        method: 'auto' (mixto), o numérico ('pearson'/'spearman'/'kendall')
        threshold: umbral de asociación para crear aristas
        prioritize: lista de variables a mantener sí o sí
        plot_matrix: si True, genera el heatmap de la matriz
        return_matrix_fig: si True, retorna la figura
        return_graph: si True, retorna el grafo (networkx)
        logger: logger para mensajes
    Returns:
        {
          'selected': List[str],
          'matrix': pd.DataFrame,
          'matrix_figure': Optional[Figure],
          'graph': Optional[nx.Graph],
        }
    """
    cols = [columns] if isinstance(columns, str) else (list(columns) if columns is not None else df.columns.tolist())
    corr_out = compute_correlation_matrix(df, columns=cols, method=method, plot=plot_matrix, return_fig=return_matrix_fig, logger=logger)
    mat = corr_out['matrix']
    fig = corr_out['figure']

    selected: List[str] = []
    prioritized = set(prioritize or [])

    # Crear grafo de redundancias
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
        # Fallback sin networkx: agrupar por conectividad simple
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

    # Selección por grupo
    for comp in components:
        comp_list = list(comp)
        # Si alguno priorizado está en el grupo, elíjase
        prios = [v for v in comp_list if v in prioritized]
        if len(prios) > 0:
            selected.append(prios[0])
            continue

        if target_column is None:
            # Sin target, elegir la variable con mayor grado de conectividad (o la primera)
            if nx is not None and graph_obj is not None:
                degrees = [(v, graph_obj.degree[v]) for v in comp_list]
                chosen = sorted(degrees, key=lambda x: x[1], reverse=True)[0][0]
            else:
                chosen = comp_list[0]
            selected.append(chosen)
            continue

        # Elegir la de mayor asociación con el target
        scores = []
        for v in comp_list:
            s = _feature_target_association(df, v, target_column, method=method)
            scores.append((v, (s if not pd.isna(s) else -np.inf)))
        chosen = sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
        selected.append(chosen)

    # Asegurar inclusión de priorizadas aunque sean componentes independientes
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


