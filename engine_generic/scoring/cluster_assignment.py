"""
Módulo para asignar clusters a clientes usando el pipeline de segmentación.

Este módulo:
1. Carga el pipeline de segmentación desde pickle
2. Preprocesa las variables necesarias
3. Asigna la etiqueta de cluster a cada cliente
"""

import pickle
from typing import Dict, Any, Optional

import pandas as pd

from feature_pro import apply_segmentation_pipeline, load_segmentation_pipeline


def assign_clusters(
    df: pd.DataFrame,
    segmentation_pipeline_path: str,
    cluster_column: str = "cluster",
    pipeline: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Asigna clusters a clientes usando el pipeline de segmentación.
    
    El pipeline contiene:
    - preprocessor: Preprocesador numérico (imputación, winsorización)
    - model: DecisionTreeRegressor entrenado
    - features: Lista de features necesarias para la segmentación
    - merge_leaves: Diccionario opcional para fusionar hojas
    
    Args:
        df: DataFrame con datos de clientes (debe contener las variables de segmentación)
        segmentation_pipeline_path: Ruta al archivo .pkl del pipeline de segmentación
        cluster_column: Nombre de la columna donde se guardará el cluster asignado
        pipeline: Pipeline pre-cargado (opcional, si no se proporciona se carga desde path)
        
    Returns:
        DataFrame con la columna de cluster asignada
        
    Raises:
        ValueError: Si faltan columnas necesarias para la segmentación
    """
    # Cargar pipeline si no se proporciona
    if pipeline is None:
        pipeline = load_segmentation_pipeline(segmentation_pipeline_path)
    
    # Obtener las features necesarias del pipeline
    features = pipeline.get("features", [])
    
    if not features:
        raise ValueError("Pipeline de segmentación no contiene 'features'")
    
    # Verificar que las columnas necesarias estén presentes
    missing_cols = [col for col in features if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columnas faltantes para segmentación: {missing_cols}\n"
            f"Columnas disponibles en DataFrame: {sorted(list(df.columns))}\n"
            f"Columnas requeridas por el pipeline: {sorted(features)}"
        )
    
    # Aplicar pipeline de segmentación
    # Esto internamente:
    # 1. Aplica el preprocessor (imputación, winsorización)
    # 2. Selecciona las features
    # 3. Usa el model para predecir leaf_id
    # 4. Aplica merge_leaves si existe
    # 5. Renumera clusters secuencialmente (1, 2, 3, ...)
    applied_result = apply_segmentation_pipeline(
        pipeline=pipeline,
        df=df,
    )
    
    # Obtener los segment_id (que son los clusters renumerados)
    df_result = df.copy()
    df_result[cluster_column] = applied_result["segment_id"]
    
    return df_result

