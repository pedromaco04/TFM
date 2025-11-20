"""
Módulo para asignar propensiones a clientes usando el modelo unificado.

Este módulo:
1. Carga el modelo unificado desde pickle
2. Para cada cliente, aplica el pipeline de preprocesamiento de su cluster
3. Usa el modelo del cluster correspondiente para predecir la propensión
"""

import pickle
import io
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import pandas as pd
import sys
import os

from modeling_pro import predict_logistic

# Importar apply_preprocessing_pipeline desde process_modeling_engine
# Nota: Esta función está definida en process_modeling_engine.py
# En el futuro podría moverse a feature_pro para mejor modularidad
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from process_modeling_engine import apply_preprocessing_pipeline


def load_unified_model(unified_model_path: str, logger: Optional[Any] = None) -> Dict[str, Any]:
    """
    Carga el modelo unificado desde un archivo .pkl.
    
    Soporta rutas locales y S3.
    
    Args:
        unified_model_path: Ruta al archivo .pkl del modelo unificado
        logger: Logger opcional para mensajes
        
    Returns:
        Diccionario con el modelo unificado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el archivo no contiene un modelo válido
    """
    if logger:
        logger.info(f"Cargando modelo unificado desde: {unified_model_path}")
    
    if unified_model_path.startswith('s3://'):
        # Cargar desde S3
        try:
            import boto3
            
            parsed = urlparse(unified_model_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            s3_client = boto3.client('s3')
            buffer = io.BytesIO()
            s3_client.download_fileobj(bucket, key, buffer)
            buffer.seek(0)
            unified_model = pickle.load(buffer)
            
            if logger:
                logger.info(f"Modelo unificado cargado desde S3 exitosamente")
        except ImportError:
            raise ImportError("boto3 no está disponible. No se puede cargar desde S3.")
        except Exception as e:
            if logger:
                logger.error(f"Error cargando modelo desde S3: {e}")
            raise
    else:
        # Cargar desde ruta local
        with open(unified_model_path, 'rb') as f:
            unified_model = pickle.load(f)
        
        if logger:
            logger.info(f"Modelo unificado cargado desde ruta local exitosamente")
    
    # Validar estructura del modelo
    if not isinstance(unified_model, dict):
        raise ValueError("El archivo no contiene un diccionario válido")
    
    if 'clusters' not in unified_model:
        raise ValueError("El modelo unificado no contiene la clave 'clusters'")
    
    if 'metadata' not in unified_model:
        raise ValueError("El modelo unificado no contiene la clave 'metadata'")
    
    if logger:
        clusters = list(unified_model['clusters'].keys())
        logger.info(f"Modelo unificado contiene {len(clusters)} clusters: {clusters}")
    
    return unified_model


def assign_propensities(
    df: pd.DataFrame,
    unified_model_path: str,
    cluster_column: str = "cluster",
    propensity_column: str = "propensity",
    unified_model: Optional[Dict[str, Any]] = None,
    logger: Optional[Any] = None,
) -> pd.DataFrame:
    """
    Asigna propensiones a clientes usando el modelo unificado.
    
    Para cada cliente:
    1. Obtiene su cluster asignado
    2. Aplica el pipeline de preprocesamiento del cluster correspondiente
    3. Usa el modelo del cluster para predecir la propensión
    
    Args:
        df: DataFrame con datos de clientes (debe contener la columna de cluster)
        unified_model_path: Ruta al archivo .pkl del modelo unificado
        cluster_column: Nombre de la columna que contiene el cluster asignado
        propensity_column: Nombre de la columna donde se guardará la propensión
        unified_model: Modelo unificado pre-cargado (opcional, si no se proporciona se carga desde path)
        logger: Logger opcional para mensajes
        
    Returns:
        DataFrame con la columna de propensión asignada
        
    Raises:
        ValueError: Si faltan columnas necesarias o el cluster no tiene modelo
    """
    if logger:
        logger.info(f"Iniciando asignación de propensiones para {len(df)} clientes")
    
    # Cargar modelo unificado si no se proporciona
    if unified_model is None:
        unified_model = load_unified_model(unified_model_path, logger=logger)
    
    # Verificar que existe la columna de cluster
    if cluster_column not in df.columns:
        raise ValueError(
            f"Columna '{cluster_column}' no encontrada en el DataFrame. "
            f"Columnas disponibles: {sorted(list(df.columns))}"
        )
    
    # Obtener clusters disponibles en el modelo
    available_clusters = list(unified_model['clusters'].keys())
    
    # Verificar que todos los clusters en el DataFrame tienen modelo
    unique_clusters = df[cluster_column].unique()
    missing_clusters = [c for c in unique_clusters if c not in available_clusters]
    if missing_clusters:
        raise ValueError(
            f"Los siguientes clusters no tienen modelo en el unified_model: {missing_clusters}\n"
            f"Clusters disponibles en el modelo: {available_clusters}"
        )
    
    if logger:
        logger.info(f"Clusters a procesar: {sorted(unique_clusters)}")
    
    # Crear DataFrame de resultados
    df_result = df.copy()
    df_result[propensity_column] = None
    
    # Procesar cada cluster
    for cluster_num in sorted(unique_clusters):
        cluster_mask = df[cluster_column] == cluster_num
        n_clients = cluster_mask.sum()
        
        if n_clients == 0:
            continue
        
        if logger:
            logger.info(f"Procesando cluster {cluster_num}: {n_clients} clientes")
        
        # Obtener datos del cluster
        cluster_data = unified_model['clusters'][cluster_num]
        
        # Obtener pipeline y modelo del cluster
        preprocessing_pipeline = cluster_data.get('preprocessing_pipeline')
        models_dict = cluster_data.get('models_dict')
        feature_columns = cluster_data.get('feature_columns')
        
        if preprocessing_pipeline is None:
            raise ValueError(f"Cluster {cluster_num} no tiene 'preprocessing_pipeline'")
        
        if models_dict is None:
            raise ValueError(f"Cluster {cluster_num} no tiene 'models_dict'")
        
        if feature_columns is None:
            raise ValueError(f"Cluster {cluster_num} no tiene 'feature_columns'")
        
        # Obtener DataFrame del cluster
        df_cluster = df[cluster_mask].copy()
        
        # Aplicar pipeline de preprocesamiento
        if logger:
            logger.debug(f"Aplicando pipeline de preprocesamiento para cluster {cluster_num}")
        
        try:
            df_cluster_transformed = apply_preprocessing_pipeline(
                df=df_cluster,
                pipeline=preprocessing_pipeline,
                logger=logger,
            )
        except Exception as e:
            if logger:
                logger.error(f"Error aplicando pipeline para cluster {cluster_num}: {e}")
            raise
        
        # Verificar que las columnas de features estén presentes
        missing_features = [f for f in feature_columns if f not in df_cluster_transformed.columns]
        if missing_features:
            raise ValueError(
                f"Cluster {cluster_num}: Faltan features después del preprocesamiento: {missing_features}\n"
                f"Features requeridas: {feature_columns}\n"
                f"Features disponibles: {sorted(list(df_cluster_transformed.columns))}"
            )
        
        # Predecir propensiones
        if logger:
            logger.debug(f"Prediciendo propensiones para cluster {cluster_num}")
        
        try:
            predictions = predict_logistic(
                models_dict=models_dict,
                df=df_cluster_transformed,
                segment_column=None,
                return_proba=True,
                logger=logger,
            )
            
            # Asignar propensiones al DataFrame de resultados
            df_result.loc[cluster_mask, propensity_column] = predictions['pred_proba'].values
            
            if logger:
                avg_propensity = predictions['pred_proba'].mean()
                logger.info(
                    f"Cluster {cluster_num}: Propensiones asignadas. "
                    f"Promedio: {avg_propensity:.4f}, "
                    f"Min: {predictions['pred_proba'].min():.4f}, "
                    f"Max: {predictions['pred_proba'].max():.4f}"
                )
        except Exception as e:
            if logger:
                logger.error(f"Error prediciendo propensiones para cluster {cluster_num}: {e}")
            raise
    
    # Verificar que todas las propensiones fueron asignadas
    missing_propensities = df_result[propensity_column].isna().sum()
    if missing_propensities > 0:
        if logger:
            logger.warning(f"{missing_propensities} clientes no tienen propensión asignada")
    
    if logger:
        logger.info(
            f"Asignación de propensiones completada. "
            f"Promedio global: {df_result[propensity_column].mean():.4f}, "
            f"Min: {df_result[propensity_column].min():.4f}, "
            f"Max: {df_result[propensity_column].max():.4f}"
        )
    
    return df_result

