from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree


def build_regression_tree_segments(
    df: pd.DataFrame,
    features: Union[List[str], Tuple[str, ...]],
    target: str,
    max_depth: int = 4,
    min_samples_leaf: int = 1000,
    random_state: Optional[int] = 42,
    plot: bool = False,
    return_fig: bool = True,
    merge_leaves: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Entrena un árbol de regresión para segmentar observaciones maximizando varianza entre grupos
    y minimizando varianza intra-grupo (criterio MSE). Cada hoja define un segmento; opcionalmente
    se pueden fusionar hojas en segmentos mayores.

    Args:
        df: DataFrame con las variables de entrada y el target.
        features: lista de nombres de columnas numéricas usadas como predictores.
        target: nombre de la variable objetivo (numérica), p.ej. 'tasa'.
        max_depth: profundidad máxima del árbol.
        min_samples_leaf: mínimo de observaciones por hoja.
        random_state: semilla para reproducibilidad.
        plot: si True, genera una figura del árbol.
        return_fig: si True, devuelve la figura en el resultado.
        merge_leaves: mapeo opcional {leaf_id -> segment_id} para fusionar hojas en segmentos.

    Returns:
        dict con:
          - 'model': objeto DecisionTreeRegressor entrenado
          - 'leaf_id': Serie con el id de hoja para cada fila utilizada en el ajuste
          - 'segment_id': Serie con el id de segmento (igual a leaf_id o fusionado si merge_leaves)
          - 'summary': DataFrame por segmento con métricas (n, mean_target, std_target)
          - 'figure': figura del árbol (si plot y return_fig)
    """
    if target not in df.columns:
        raise ValueError(f"Target '{target}' no encontrado en DataFrame.")
    for f in features:
        if f not in df.columns:
            raise ValueError(f"Feature '{f}' no encontrado en DataFrame.")

    work = df[features + [target]].copy()
    work = work.dropna(subset=[target])
    X = work[features]
    y = work[target].astype(float)

    # Reemplazar NaN en X con medianas columna a columna (simple safeguard)
    X_filled = X.copy()
    for c in X_filled.columns:
        if X_filled[c].isna().any():
            X_filled[c] = X_filled[c].fillna(X_filled[c].median())

    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X_filled, y)

    # Obtener id de hoja por fila usada en el fit
    leaf_ids = model.apply(X_filled)
    leaf_series = pd.Series(leaf_ids, index=work.index, name="leaf_id")

    # Construir segment_id (posible fusión de hojas)
    if merge_leaves:
        seg_series = leaf_series.map(lambda lid: merge_leaves.get(lid, lid))
    else:
        seg_series = leaf_series.copy()
        seg_series.name = "segment_id"

    # Resumen por segmento
    summ = (
        pd.DataFrame({target: y, "segment_id": seg_series})
        .groupby("segment_id")
        .agg(n=(target, "size"), mean_target=(target, "mean"), std_target=(target, "std"))
        .reset_index()
        .sort_values("mean_target", ascending=False)
        .reset_index(drop=True)
    )

    fig = None
    if plot and return_fig:
        fig, ax = plt.subplots(figsize=(14, 8))
        plot_tree(
            model,
            feature_names=list(features),
            filled=True,
            rounded=True,
            impurity=True,
            fontsize=8,
            ax=ax,
        )
        ax.set_title(f"DecisionTreeRegressor (max_depth={max_depth}, min_leaf={min_samples_leaf})")
        fig.tight_layout()

    return {
        "model": model,
        "leaf_id": leaf_series,
        "segment_id": seg_series,
        "summary": summ,
        "figure": fig,
    }


def compute_tree_feature_importance(
    model: DecisionTreeRegressor,
    feature_names: Union[List[str], Tuple[str, ...]],
    normalize: bool = True,
) -> pd.DataFrame:
    """
    Calcula la importancia de variables a partir de un DecisionTreeRegressor entrenado.
    Usa `model.feature_importances_` y devuelve un DataFrame ordenado descendentemente.
    Opcionalmente normaliza para que sumen 1.0 (si no lo están).

    Args:
        model: instancia entrenada de sklearn.tree.DecisionTreeRegressor.
        feature_names: lista/tupla de nombres de variables en el mismo orden que X.
        normalize: si True, normaliza importancias para que sumen 1.0 (si la suma > 0).

    Returns:
        DataFrame con columnas ['feature', 'importance'], ordenado desc.
    """
    importances = np.asarray(getattr(model, "feature_importances_", None))
    if importances is None:
        raise ValueError("El modelo no contiene 'feature_importances_'. ¿Está entrenado correctamente?")
    if normalize:
        s = importances.sum()
        if s > 0:
            importances = importances / s
    df_imp = pd.DataFrame({"feature": list(feature_names), "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).reset_index(drop=True)
    return df_imp


def fit_numeric_preprocessor(
    df: pd.DataFrame,
    numeric_columns: Union[List[str], Tuple[str, ...]],
    impute_strategy: str = "median",
    groupby: Optional[str] = None,
    winsorize: bool = False,
    lower: float = 0.01,
    upper: float = 0.99,
) -> Dict[str, Any]:
    """
    Ajusta un preprocesador numérico para:
      - Imputación por estrategia ('median' o 'mean' o 'mode'), global o por grupos (groupby).
      - Winsorización opcional con percentiles [lower, upper] por columna (global).
    Devuelve un dict serializable con los parámetros necesarios para luego transformar cualquier DataFrame.
    Args:
        df: DataFrame de entrada.
        numeric_columns: columnas numéricas a procesar.
        impute_strategy: 'median' | 'mean' | 'mode'.
        groupby: nombre de columna categórica para imputación por grupo (opcional).
        winsorize: si True, calcula percentiles por columna para recorte.
        lower: percentil inferior para winsorización (0-1).
        upper: percentil superior para winsorización (0-1).
    Returns:
        dict con claves:
          - 'numeric_columns': lista de columnas numéricas
          - 'impute': {'strategy','groupby','values': mapping}
          - 'winsor': {'enabled', 'lower','upper','quantiles': mapping}
    """
    cols = [c for c in numeric_columns if c in df.columns]
    impute_values: Dict[str, Any] = {}
    if groupby is None:
        impute_values["__global__"] = {}
        for c in cols:
            s = df[c]
            if impute_strategy == "median":
                impute_values["__global__"][c] = float(s.median())
            elif impute_strategy == "mean":
                impute_values["__global__"][c] = float(s.mean())
            elif impute_strategy == "mode":
                m = s.mode(dropna=True)
                impute_values["__global__"][c] = None if m.empty else float(m.iloc[0]) if pd.api.types.is_numeric_dtype(s) else m.iloc[0]
            else:
                raise ValueError(f"Estrategia de imputación no soportada: {impute_strategy}")
    else:
        impute_values["__groups__"] = {}
        for g, part in df.groupby(groupby, dropna=False):
            impute_values["__groups__"][g] = {}
            for c in cols:
                s = part[c]
                if impute_strategy == "median":
                    impute_values["__groups__"][g][c] = float(s.median())
                elif impute_strategy == "mean":
                    impute_values["__groups__"][g][c] = float(s.mean())
                elif impute_strategy == "mode":
                    m = s.mode(dropna=True)
                    impute_values["__groups__"][g][c] = None if m.empty else float(m.iloc[0]) if pd.api.types.is_numeric_dtype(s) else m.iloc[0]
                else:
                    raise ValueError(f"Estrategia de imputación no soportada: {impute_strategy}")

    quantiles: Dict[str, Tuple[float, float]] = {}
    if winsorize:
        for c in cols:
            s = df[c]
            ql = s.quantile(lower)
            qu = s.quantile(upper)
            quantiles[c] = (float(ql) if pd.notna(ql) else None, float(qu) if pd.notna(qu) else None)

    preprocessor = {
        "numeric_columns": cols,
        "impute": {
            "strategy": impute_strategy,
            "groupby": groupby,
            "values": impute_values,
        },
        "winsor": {
            "enabled": bool(winsorize),
            "lower": float(lower),
            "upper": float(upper),
            "quantiles": quantiles,
        },
    }
    return preprocessor


def transform_with_preprocessor(
    df: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> pd.DataFrame:
    """
    Aplica el preprocesador a un DataFrame:
      - Imputación (global o por grupo)
      - Winsorización (si fue ajustada)
    No modifica el DataFrame original (retorna copia).
    """
    result = df.copy()
    cols = preprocessor.get("numeric_columns", [])
    imp = preprocessor.get("impute", {})
    wins = preprocessor.get("winsor", {})

    # Imputación
    groupby = imp.get("groupby", None)
    values = imp.get("values", {})
    if groupby is None or "__global__" in values:
        glob = values.get("__global__", {})
        for c in cols:
            if c in result.columns and c in glob and glob[c] is not None:
                result[c] = result[c].fillna(glob[c])
    else:
        groups_map = values.get("__groups__", {})
        if groupby in result.columns:
            for g, colmap in groups_map.items():
                mask = result[groupby].astype('object').eq(g)
                for c in cols:
                    if c in result.columns and c in colmap and colmap[c] is not None:
                        result.loc[mask, c] = result.loc[mask, c].fillna(colmap[c])
        else:
            glob = values.get("__global__", {})
            for c in cols:
                if c in result.columns and c in glob and glob[c] is not None:
                    result[c] = result[c].fillna(glob[c])

    # Winsorización
    if wins.get("enabled", False):
        quantiles = wins.get("quantiles", {})
        for c in cols:
            if c in result.columns and c in quantiles:
                ql, qu = quantiles[c]
                if ql is not None and qu is not None:
                    result[c] = result[c].clip(lower=ql, upper=qu)

    return result


def save_segmentation_pipeline(
    path: str,
    preprocessor: Dict[str, Any],
    model: DecisionTreeRegressor,
    features: Union[List[str], Tuple[str, ...]],
    merge_leaves: Optional[Dict[int, int]] = None,
) -> str:
    """
    Guarda en .pkl un pipeline de segmentación con:
      - preprocesador numérico
      - modelo (árbol de regresión)
      - lista de features y mapa de fusión de hojas (opcional)
    Args:
        path: ruta del archivo .pkl a guardar.
        preprocessor: dict serializable devuelto por fit_numeric_preprocessor.
        model: DecisionTreeRegressor entrenado.
        features: lista/tupla de features usadas.
        merge_leaves: dict opcional {leaf_id: segment_id}.
    Returns:
        Ruta final guardada.
    """
    import pickle, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "preprocessor": preprocessor,
        "model": model,
        "features": list(features),
        "merge_leaves": merge_leaves or {},
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return path


def load_segmentation_pipeline(path: str) -> Dict[str, Any]:
    """
    Carga un pipeline de segmentación previamente guardado con save_segmentation_pipeline.
    """
    import pickle
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload


def apply_segmentation_pipeline(
    pipeline: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Aplica un pipeline de segmentación a un DataFrame:
      1) Aplica preprocesador (imputación y winsorización)
      2) Evalúa el árbol y retorna segment_id por fila
    Returns:
      dict con:
        - 'segment_id': Serie con los segmentos asignados
        - 'leaf_id': Serie con ids de hoja
    """
    pre = pipeline["preprocessor"]
    model: DecisionTreeRegressor = pipeline["model"]
    features: List[str] = pipeline["features"]
    merge_leaves: Dict[int, int] = pipeline.get("merge_leaves", {})

    df_t = transform_with_preprocessor(df, pre)
    X = df_t[features].copy()
    # Fill residual NaNs defensivamente
    for c in X.columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    leaf_ids = model.apply(X)
    leaf_series = pd.Series(leaf_ids, index=df.index, name="leaf_id")
    if merge_leaves:
        seg_series = leaf_series.map(lambda lid: merge_leaves.get(lid, lid)).rename("segment_id")
    else:
        seg_series = leaf_series.rename("segment_id")

    return {"segment_id": seg_series, "leaf_id": leaf_series}


