"""
Módulo genérico de optimización de pricing para productos financieros.

Este módulo optimiza variables de precio genéricas, balanceando profit y volumen usando
una función objetivo con parámetro alpha.

Profit = price_variable × amount × propensity
Volume = propensity × amount
Utility = alpha × profit_norm + (1 - alpha) × volume_norm
"""

import logging
import os
from typing import Dict, Any, Optional, Tuple, Callable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

EPSILON = 1e-9


def utilidad_lineal(alpha: float, monto: np.ndarray, ingreso: np.ndarray, **_: Any) -> np.ndarray:
    """Combinación lineal clásica."""
    return alpha * monto + (1 - alpha) * ingreso


def utilidad_pesos_gamma(alpha: float, monto: np.ndarray, ingreso: np.ndarray, gamma: float = 1.0, **_: Any) -> np.ndarray:
    """Utilidad con pesos ajustados por potencia gamma."""
    gamma = np.clip(gamma, EPSILON, None)
    peso_monto = np.power(alpha, gamma)
    peso_ingreso = np.power(1 - alpha, gamma)
    total = peso_monto + peso_ingreso
    total = total if total > 0 else EPSILON
    peso_monto /= total
    peso_ingreso = 1 - peso_monto
    return peso_monto * monto + peso_ingreso * ingreso


def utilidad_sesgo_k(alpha: float, monto: np.ndarray, ingreso: np.ndarray, k: float = 1.0, **_: Any) -> np.ndarray:
    """Utilidad con sesgo controlado por parámetro k."""
    k = np.clip(k, EPSILON, None)
    denom = alpha + k * (1 - alpha)
    denom = denom if denom > 0 else EPSILON
    peso_monto = alpha / denom
    peso_ingreso = 1 - peso_monto
    return peso_monto * monto + peso_ingreso * ingreso


def utilidad_cobb_douglas(alpha: float, monto: np.ndarray, ingreso: np.ndarray, **_: Any) -> np.ndarray:
    """Función de utilidad tipo Cobb-Douglas."""
    monto_safe = np.maximum(monto, EPSILON)
    ingreso_safe = np.maximum(ingreso, EPSILON)
    return np.power(monto_safe, alpha) * np.power(ingreso_safe, 1 - alpha)


def utilidad_ces(alpha: float, monto: np.ndarray, ingreso: np.ndarray, rho: float = 0.5, **_: Any) -> np.ndarray:
    """Función CES (Constant Elasticity of Substitution)."""
    if np.isclose(rho, 0.0):
        return utilidad_cobb_douglas(alpha, monto, ingreso)
    monto_safe = np.maximum(monto, EPSILON)
    ingreso_safe = np.maximum(ingreso, EPSILON)
    inner = alpha * np.power(monto_safe, rho) + (1 - alpha) * np.power(ingreso_safe, rho)
    return np.power(np.maximum(inner, EPSILON), 1 / rho)


def utilidad_ratio_simple(alpha: float, monto: np.ndarray, ingreso: np.ndarray, **_: Any) -> np.ndarray:
    """Utilidad basada en ratios simples monto/ingreso e ingreso/monto."""
    ratio_monto = monto / np.maximum(ingreso, EPSILON)
    ratio_ingreso = ingreso / np.maximum(monto, EPSILON)
    return alpha * ratio_monto + (1 - alpha) * ratio_ingreso


def utilidad_ratio_log(alpha: float, monto: np.ndarray, ingreso: np.ndarray, **_: Any) -> np.ndarray:
    """Versión logarítmica de la utilidad basada en ratios."""
    ratio_monto = np.log1p(monto / np.maximum(ingreso, EPSILON))
    ratio_ingreso = np.log1p(ingreso / np.maximum(monto, EPSILON))
    return alpha * ratio_monto + (1 - alpha) * ratio_ingreso


def utilidad_normalizada(
    alpha: float,
    monto: np.ndarray,
    ingreso: np.ndarray,
    sigma_monto: float = 1.0,
    sigma_ingreso: float = 1.0,
    **_: Any,
) -> np.ndarray:
    """Utilidad ponderada por desviaciones estándar."""
    base_monto = 1 / np.clip(sigma_monto, EPSILON, None)
    base_ingreso = 1 / np.clip(sigma_ingreso, EPSILON, None)
    peso_monto = alpha * base_monto
    peso_ingreso = (1 - alpha) * base_ingreso
    total = peso_monto + peso_ingreso
    total = total if total > 0 else EPSILON
    peso_monto /= total
    peso_ingreso = 1 - peso_monto
    return peso_monto * monto + peso_ingreso * ingreso


def utilidad_piecewise_monto(
    alpha: float,
    monto: np.ndarray,
    ingreso: np.ndarray,
    alpha_bajo: float = 0.3,
    alpha_alto: float = 0.7,
    tau: float = 0.5,
    **_: Any,
) -> np.ndarray:
    """Utilidad por tramos basada en el valor de monto."""
    alpha_bajo = np.clip(alpha_bajo, 0.0, 1.0)
    alpha_alto = np.clip(alpha_alto, 0.0, 1.0)
    pesos = np.where(monto < tau, alpha_bajo, alpha_alto)
    pesos = np.clip(pesos, 0.0, 1.0)
    return pesos * monto + (1 - pesos) * ingreso


UTILITY_FUNCTIONS: Dict[str, Callable[..., np.ndarray]] = {
    "lineal": utilidad_lineal,
    "pesos_gamma": utilidad_pesos_gamma,
    "sesgo_k": utilidad_sesgo_k,
    "cobb_douglas": utilidad_cobb_douglas,
    "ces": utilidad_ces,
    "ratio_simple": utilidad_ratio_simple,
    "ratio_log": utilidad_ratio_log,
    "normalizada": utilidad_normalizada,
    "piecewise_monto": utilidad_piecewise_monto,
}


def _process_one_price_scenario(
    scenario_idx: int,
    price_variable_value: float,
    price_variable_name: str,
    df_val: pd.DataFrame,
    unified_model: Dict[str, Any],
    segmentation_pipeline_path: str,
    unified_model_path: Optional[str] = None,
    segmentation_pipeline: Optional[Dict[str, Any]] = None,
    amount_column: str = "importe",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Procesa un escenario de optimización de precio y retorna propensiones, profit y volume.
    
    Esta función:
    1. Asigna clusters usando assign_clusters (usa .pkl de segmentación)
    2. Asigna propensiones usando assign_propensities (usa .pkl de modelado)
    Ambos usan los .pkl correctamente sin transformar columnas que no deberían transformarse.
    
    Args:
        scenario_idx: Índice del escenario
        price_variable_value: Valor de la variable de precio a optimizar
        price_variable_name: Nombre de la columna de la variable de precio (ej: "spread", "tasa", etc.)
        df_val: DataFrame con datos de clientes
        unified_model: Modelo unificado con pipelines y modelos por cluster
        segmentation_pipeline_path: Ruta al pipeline de segmentación (.pkl)
        unified_model_path: Ruta al modelo unificado (opcional, solo para logging en assign_propensities)
        segmentation_pipeline: Pipeline de segmentación pre-cargado (opcional, para evitar cargar múltiples veces)
        amount_column: Nombre de la columna que contiene el monto/importe (default: "importe")
        logger: Logger opcional
        
    Returns:
        Diccionario con:
        - propensities: array de propensiones
        - profit: array de profit (price_variable × amount × propensity)
        - volume: array de volume (propensity × amount)
        - price_variable: array de valores de la variable de precio
        - amount: array de montos
    """
    try:
        # Crear copia del DataFrame y asignar el valor de la variable de precio
        df_scenario = df_val.copy()
        df_scenario[price_variable_name] = price_variable_value
        
        # Paso 1: Asignar clusters usando el pipeline de segmentación
        from scoring.cluster_assignment import assign_clusters
        
        # Usar logger=None para evitar logs repetitivos durante optimización
        df_with_clusters = assign_clusters(
            df=df_scenario,
            segmentation_pipeline_path=segmentation_pipeline_path,
            cluster_column="cluster",
            pipeline=segmentation_pipeline,  # Pasar pipeline pre-cargado si está disponible
        )
        
        # Paso 2: Asignar propensiones usando el modelo unificado
        from scoring.propensity_assignment import assign_propensities
        
        df_with_propensities = assign_propensities(
            df=df_with_clusters,
            unified_model_path=unified_model_path or "",  # No se usará si se pasa unified_model
            cluster_column="cluster",
            propensity_column="propensity",
            unified_model=unified_model,
            logger=None,  # Sin logger para evitar logs repetitivos
        )
        
        # Obtener monto
        if amount_column not in df_with_propensities.columns:
            # Intentar alternativas
            if 'monto' in df_with_propensities.columns:
                amount_column = 'monto'
            else:
                raise ValueError(f"No se encontró columna '{amount_column}' ni 'monto' en el DataFrame")
        
        amount = pd.to_numeric(df_with_propensities[amount_column], errors='coerce').fillna(0).values
        
        # Obtener propensiones
        propensities = pd.to_numeric(df_with_propensities['propensity'], errors='coerce').fillna(0).values
        
        # Calcular profit y volume
        n_customers = len(df_with_propensities)
        price_variable_array = np.full(n_customers, price_variable_value, dtype=np.float32)
        profit = price_variable_array * amount * propensities  # Profit = price_variable × amount × propensity
        volume = propensities * amount  # Volume = propensity × amount
        
        return {
            'scenario_idx': scenario_idx,
            'propensities': propensities,
            'profit': profit,
            'volume': volume,
            'price_variable': price_variable_array,
            'amount': amount,
        }
        
    except Exception as e:
        if logger:
            logger.error(f"Error procesando escenario {scenario_idx} ({price_variable_name}={price_variable_value:.4f}): {e}", exc_info=True)
        raise


def run_price_optimization(
    unified_model: Dict[str, Any],
    df_val: pd.DataFrame,
    price_variable_name: str,
    price_variable_values: np.ndarray,
    segmentation_pipeline_path: str,
    amount_column: str = "importe",
    unified_model_path: Optional[str] = None,
    n_jobs: int = 1,
    save_results_path: Optional[str] = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/05.Optimization/escenarios_x_cliente.parquet",
    compute_efficient_frontier: bool = False,
    alpha_values: Optional[np.ndarray] = None,
    objective_type: str = "lineal",
    objective_params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Ejecuta optimización de una variable de precio generando múltiples escenarios.
    
    PASO 1: Calcula propensiones para cada escenario de price_variable
    - Para cada escenario:
      1. Asigna el valor de la variable de precio al DataFrame
      2. Asigna clusters usando el pipeline de segmentación (.pkl de process_segmentation.py)
      3. Asigna propensiones usando el modelo unificado (.pkl de process_modeling_engine.py)
      4. Calcula profit = price_variable × amount × propensity
      5. Calcula volume = propensity × amount
    - Guarda resultados en archivo si se especifica save_results_path
    
    PASO 2 (opcional): Calcula frontera eficiente
    - Carga resultados guardados del paso 1
    - Para cada alpha, maximiza función de utilidad
    - Retorna datos agregados (profit total, volume total) para la curva
    
    Args:
        unified_model: Modelo unificado con pipelines y modelos por cluster
        df_val: DataFrame con datos de clientes
        price_variable_name: Nombre de la variable de precio a optimizar (ej: "spread", "tasa")
        price_variable_values: Array de valores de la variable de precio a evaluar
        segmentation_pipeline_path: Ruta al pipeline de segmentación (.pkl generado en process_segmentation.py)
        amount_column: Nombre de la columna que contiene el monto/importe (default: "importe")
        unified_model_path: Ruta al modelo unificado (opcional, solo para logging en assign_propensities)
        n_jobs: Número de trabajos paralelos (1 = secuencial)
        save_results_path: Ruta donde guardar resultados del paso 1 (.parquet o .csv). 
                          Puede ser local o S3. Por defecto: ruta de S3 para escenarios_x_cliente.parquet
                          Si None, no guarda.
        compute_efficient_frontier: Si True, calcula la frontera eficiente después del paso 1
        alpha_values: Array de valores de alpha para calcular frontera eficiente (requerido si compute_efficient_frontier=True)
        objective_type: Nombre de la función objetivo para frontera eficiente (default: "lineal")
        objective_params: Parámetros adicionales para la función objetivo
        logger: Logger opcional
        
    Returns:
        Diccionario con:
        - optimization_results: Resultados del paso 1 (propensities, profit, volume, etc.)
        - efficient_frontier: DataFrame con frontera eficiente (solo si compute_efficient_frontier=True)
            - alpha: Valor de alpha
            - optimal_{price_variable_name}: Price variable óptimo promedio
            - total_profit: Profit total esperado agregado
            - total_volume: Volume total esperado agregado
    """
    if logger:
        logger.info("=" * 80)
        logger.info("INICIO OPTIMIZACIÓN DE PRECIO")
        logger.info("=" * 80)
        logger.info(f"Variable de precio: {price_variable_name}")
        logger.info(f"Rango: {price_variable_values.min():.4f} a {price_variable_values.max():.4f}")
        logger.info(f"Total de escenarios: {len(price_variable_values)}")
        logger.info(f"Clientes a procesar: {len(df_val):,}")
        logger.info(f"Pipeline de segmentación: {segmentation_pipeline_path}")
        logger.info(f"Paralelización: {n_jobs} jobs")
    
    # Convertir a numpy array si es necesario
    if not isinstance(price_variable_values, np.ndarray):
        price_variable_values = np.asarray(price_variable_values, dtype=float)
    
    if logger:
        logger.info(f"Escenarios: {price_variable_values}")
    
    # Cargar pipeline de segmentación una vez para reutilizarlo en todos los escenarios
    # (solo en modo secuencial, en paralelo cada worker lo carga)
    segmentation_pipeline = None
    if n_jobs == 1:
        try:
            from feature_pro import load_segmentation_pipeline
            segmentation_pipeline = load_segmentation_pipeline(segmentation_pipeline_path)
            if logger:
                logger.info("Pipeline de segmentación cargado (se reutilizará en todos los escenarios)")
        except Exception as e:
            if logger:
                logger.warning(f"No se pudo precargar el pipeline de segmentación: {e}. Se cargará en cada escenario.")
    
    # Inicializar resultados
    results = {
        'propensities': {},
        'profit': {},
        'volume': {},
        'price_variable': {},
        'amount': {},
        'price_variable_values': price_variable_values.tolist(),
    }
    
    # Procesar escenarios
    if n_jobs > 1:
        # Procesamiento paralelo
        if logger:
            logger.info("Procesando escenarios en paralelo...")
        
        results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_one_price_scenario)(
                scenario_idx=i,
                price_variable_value=price_val,
                price_variable_name=price_variable_name,
                df_val=df_val,
                unified_model=unified_model,
                segmentation_pipeline_path=segmentation_pipeline_path,
                unified_model_path=unified_model_path,
                segmentation_pipeline=None,  # Cada worker carga su propio pipeline
                amount_column=amount_column,
                logger=logger,
            )
            for i, price_val in enumerate(price_variable_values)
        )
        
        # Organizar resultados
        for result in results_list:
            scenario_idx = result['scenario_idx']
            results['propensities'][scenario_idx] = result['propensities']
            results['profit'][scenario_idx] = result['profit']
            results['volume'][scenario_idx] = result['volume']
            results['price_variable'][scenario_idx] = result['price_variable']
            results['amount'][scenario_idx] = result['amount']
    else:
        # Procesamiento secuencial
        if logger:
            logger.info("Procesando escenarios secuencialmente...")
        
        for i, price_val in enumerate(price_variable_values):
            if logger:
                logger.info(f"Procesando escenario {i+1}/{len(price_variable_values)}: {price_variable_name}={price_val:.4f}")
            
            result = _process_one_price_scenario(
                scenario_idx=i,
                price_variable_value=price_val,
                price_variable_name=price_variable_name,
                df_val=df_val,
                unified_model=unified_model,
                segmentation_pipeline_path=segmentation_pipeline_path,
                unified_model_path=unified_model_path,
                segmentation_pipeline=segmentation_pipeline,  # Reutilizar pipeline precargado
                amount_column=amount_column,
                logger=logger,
            )
            
            results['propensities'][i] = result['propensities']
            results['profit'][i] = result['profit']
            results['volume'][i] = result['volume']
            results['price_variable'][i] = result['price_variable']
            results['amount'][i] = result['amount']
    
    if logger:
        logger.info("=" * 80)
        logger.info("PASO 1 COMPLETADO: OPTIMIZACIÓN DE PRECIO")
        logger.info("=" * 80)
        logger.info(f"Total de escenarios procesados: {len(results['propensities'])}")
    
    # PASO 1: Guardar resultados si se especifica
    if save_results_path:
        _save_optimization_results(
            optimization_results=results,
            df_val=df_val,
            save_path=save_results_path,
            logger=logger,
        )
    
    # Preparar resultado final
    final_results = {
        'optimization_results': results,
    }
    
    # PASO 2: Calcular frontera eficiente si se solicita
    if compute_efficient_frontier:
        if alpha_values is None:
            raise ValueError("alpha_values es requerido cuando compute_efficient_frontier=True")
        
        if save_results_path is None:
            raise ValueError("save_results_path es requerido cuando compute_efficient_frontier=True (para cargar resultados)")
        
        if logger:
            logger.info("\n" + "=" * 80)
            logger.info("INICIANDO PASO 2: CÁLCULO DE FRONTERA EFICIENTE")
            logger.info("=" * 80)
        
        df_frontier = calculate_efficient_frontier(
            optimization_results_path=save_results_path,
            price_variable_name=price_variable_name,
            alpha_values=alpha_values,
            objective_type=objective_type,
            objective_params=objective_params,
            logger=logger,
        )
        
        final_results['efficient_frontier'] = df_frontier
    
    return final_results


def _save_optimization_results(
    optimization_results: Dict[str, Any],
    df_val: pd.DataFrame,
    save_path: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Guarda los resultados de optimización en formato Parquet o CSV (local o S3).
    
    Crea un DataFrame con todos los escenarios donde cada fila es un cliente
    y cada columna corresponde a un escenario de price_variable.
    
    Args:
        optimization_results: Resultados de run_price_optimization
        df_val: DataFrame original con datos de clientes
        save_path: Ruta donde guardar (debe terminar en .parquet o .csv). Puede ser local o S3.
        logger: Logger opcional
    """
    if logger:
        logger.info(f"Guardando resultados de optimización en: {save_path}")
    
    # Obtener número de clientes y escenarios
    n_customers = len(df_val)
    n_scenarios = len(optimization_results['propensities'])
    
    # Crear DataFrame base con datos originales
    df_results = df_val.copy()
    
    # Agregar columnas por escenario
    propensities = optimization_results['propensities']
    profit = optimization_results['profit']
    volume = optimization_results['volume']
    price_variable = optimization_results['price_variable']
    price_variable_values = optimization_results['price_variable_values']
    
    for scenario_idx in range(n_scenarios):
        price_val = price_variable_values[scenario_idx]
        df_results[f'propensity_scenario_{scenario_idx}'] = propensities[scenario_idx]
        df_results[f'profit_scenario_{scenario_idx}'] = profit[scenario_idx]
        df_results[f'volume_scenario_{scenario_idx}'] = volume[scenario_idx]
        df_results[f'price_variable_scenario_{scenario_idx}'] = price_variable[scenario_idx]
    
    # Agregar metadata
    df_results['n_scenarios'] = n_scenarios
    df_results['price_variable_values'] = str(price_variable_values)
    
    # Guardar según extensión y ubicación (local o S3)
    if save_path.startswith('s3://'):
        # Guardar en S3
        try:
            # Intentar guardar directamente (pandas/pyarrow puede manejar rutas S3)
            if save_path.endswith('.parquet'):
                df_results.to_parquet(save_path, index=False, engine='pyarrow')
            elif save_path.endswith('.csv'):
                df_results.to_csv(save_path, index=False)
            else:
                raise ValueError(f"Formato no soportado. Use .parquet o .csv. Recibido: {save_path}")
            
            if logger:
                logger.info(f"Resultados guardados exitosamente en S3. Shape: {df_results.shape}")
        except Exception as e:
            # Si falla, intentar con boto3
            if logger:
                logger.warning(f"Error guardando directamente en S3: {e}")
                logger.info("Intentando guardar usando boto3...")
            
            try:
                import boto3
                from urllib.parse import urlparse
                import io
                
                # Parsear URL de S3
                parsed = urlparse(save_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                # Guardar en buffer
                buffer = io.BytesIO()
                if save_path.endswith('.parquet'):
                    df_results.to_parquet(buffer, index=False, engine='pyarrow')
                elif save_path.endswith('.csv'):
                    df_results.to_csv(buffer, index=False)
                buffer.seek(0)
                
                # Subir a S3
                s3_client = boto3.client('s3')
                s3_client.upload_fileobj(buffer, bucket, key)
                
                if logger:
                    logger.info(f"Resultados guardados exitosamente en S3 usando boto3. Shape: {df_results.shape}")
            except Exception as e2:
                if logger:
                    logger.error(f"Error guardando en S3 con boto3: {e2}")
                raise
    else:
        # Guardar localmente
        # Crear directorio si no existe
        dir_path = os.path.dirname(save_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            if logger:
                logger.info(f"Directorio creado: {dir_path}")
        
        if save_path.endswith('.parquet'):
            df_results.to_parquet(save_path, index=False, engine='pyarrow')
        elif save_path.endswith('.csv'):
            df_results.to_csv(save_path, index=False)
        else:
            raise ValueError(f"Formato no soportado. Use .parquet o .csv. Recibido: {save_path}")
        
        if logger:
            logger.info(f"Resultados guardados exitosamente localmente. Shape: {df_results.shape}")


def _load_optimization_results(
    load_path: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Carga los resultados de optimización guardados previamente (local o S3).
    
    Args:
        load_path: Ruta al archivo guardado (.parquet o .csv). Puede ser local o S3.
        logger: Logger opcional
        
    Returns:
        Tuple con:
        - df_results: DataFrame con todos los datos
        - optimization_results: Diccionario en formato de run_price_optimization
    """
    if logger:
        logger.info(f"Cargando resultados de optimización desde: {load_path}")
    
    # Cargar según ubicación (S3 o local)
    if load_path.startswith('s3://'):
        # Cargar desde S3
        try:
            # Intentar cargar directamente (pandas/pyarrow puede manejar rutas S3)
            if load_path.endswith('.parquet'):
                df_results = pd.read_parquet(load_path, engine='pyarrow')
            elif load_path.endswith('.csv'):
                df_results = pd.read_csv(load_path)
            else:
                raise ValueError(f"Formato no soportado. Use .parquet o .csv. Recibido: {load_path}")
            
            if logger:
                logger.info("Resultados cargados exitosamente desde S3")
        except Exception as e:
            # Si falla, intentar con boto3
            if logger:
                logger.warning(f"Error cargando directamente desde S3: {e}")
                logger.info("Intentando cargar usando boto3...")
            
            try:
                import boto3
                from urllib.parse import urlparse
                import io
                
                # Parsear URL de S3
                parsed = urlparse(load_path)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                # Descargar desde S3
                s3_client = boto3.client('s3')
                buffer = io.BytesIO()
                s3_client.download_fileobj(bucket, key, buffer)
                buffer.seek(0)
                
                # Cargar desde buffer
                if load_path.endswith('.parquet'):
                    df_results = pd.read_parquet(buffer, engine='pyarrow')
                elif load_path.endswith('.csv'):
                    df_results = pd.read_csv(buffer)
                
                if logger:
                    logger.info("Resultados cargados exitosamente desde S3 usando boto3")
            except Exception as e2:
                if logger:
                    logger.error(f"Error cargando desde S3 con boto3: {e2}")
                raise
    else:
        # Cargar localmente
        if load_path.endswith('.parquet'):
            df_results = pd.read_parquet(load_path, engine='pyarrow')
        elif load_path.endswith('.csv'):
            df_results = pd.read_csv(load_path)
        else:
            raise ValueError(f"Formato no soportado. Use .parquet o .csv. Recibido: {load_path}")
        
        if logger:
            logger.info("Resultados cargados exitosamente desde local")
    
    # Extraer número de escenarios
    n_scenarios = int(df_results['n_scenarios'].iloc[0])
    
    # Extraer price_variable_values
    price_variable_values_str = df_results['price_variable_values'].iloc[0]
    # Convertir string a lista
    import ast
    price_variable_values = ast.literal_eval(price_variable_values_str)
    
    # Reconstruir optimization_results
    optimization_results = {
        'propensities': {},
        'profit': {},
        'volume': {},
        'price_variable': {},
        'amount': {},
        'price_variable_values': price_variable_values,
    }
    
    # Obtener amount (debería estar en el DataFrame original)
    amount_array = None
    if 'importe' in df_results.columns:
        amount_array = df_results['importe'].values
    elif 'monto' in df_results.columns:
        amount_array = df_results['monto'].values
    else:
        # Intentar obtener de las columnas de profit o volume (amount es el mismo para todos los escenarios)
        # amount = profit / (price_variable * propensity) o amount = volume / propensity
        # Usar el primer escenario como referencia
        if n_scenarios > 0:
            profit_scenario_0 = df_results[f'profit_scenario_0'].values
            price_var_scenario_0 = df_results[f'price_variable_scenario_0'].values
            propensity_scenario_0 = df_results[f'propensity_scenario_0'].values
            # amount = profit / (price_variable * propensity)
            denominator = price_var_scenario_0 * propensity_scenario_0
            denominator = np.where(denominator == 0, 1, denominator)  # Evitar división por cero
            amount_array = profit_scenario_0 / denominator
    
    if amount_array is None:
        raise ValueError("No se pudo determinar 'amount'. Asegúrese de que el DataFrame tenga 'importe' o 'monto'")
    
    for scenario_idx in range(n_scenarios):
        optimization_results['propensities'][scenario_idx] = df_results[f'propensity_scenario_{scenario_idx}'].values
        optimization_results['profit'][scenario_idx] = df_results[f'profit_scenario_{scenario_idx}'].values
        optimization_results['volume'][scenario_idx] = df_results[f'volume_scenario_{scenario_idx}'].values
        optimization_results['price_variable'][scenario_idx] = df_results[f'price_variable_scenario_{scenario_idx}'].values
        optimization_results['amount'][scenario_idx] = amount_array  # Amount es el mismo para todos los escenarios
    
    if logger:
        logger.info(f"Resultados cargados exitosamente. Shape: {df_results.shape}, Escenarios: {n_scenarios}")
    
    return df_results, optimization_results


def calculate_efficient_frontier(
    optimization_results_path: str,
    price_variable_name: str,
    alpha_values: np.ndarray,
    objective_type: str = "lineal",
    objective_params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Calcula la frontera eficiente probando distintos valores de alpha.
    
    Para cada alpha:
    1. Maximiza la función de utilidad
    2. Obtiene el price_variable óptimo
    3. Calcula profit esperado total y volume esperado total agregados
    
    Args:
        optimization_results_path: Ruta a los resultados guardados del primer paso
        price_variable_name: Nombre de la variable de precio (ej: "spread", "tasa")
        alpha_values: Array de valores de alpha a probar
        objective_type: Nombre de la función objetivo a utilizar
        objective_params: Parámetros adicionales para la función objetivo
        logger: Logger opcional
        
    Returns:
        DataFrame con columnas:
        - alpha: Valor de alpha probado
        - optimal_price_variable: Valor óptimo de price_variable (promedio o agregado)
        - total_profit: Profit total esperado agregado
        - total_volume: Volume total esperado agregado
    """
    if logger:
        logger.info("=" * 80)
        logger.info("INICIO CÁLCULO DE FRONTERA EFICIENTE")
        logger.info("=" * 80)
        logger.info(f"Resultados a cargar: {optimization_results_path}")
        logger.info(f"Valores de alpha a probar: {alpha_values}")
        logger.info(f"Función objetivo: {objective_type}")
    
    # Cargar resultados del primer paso
    df_results, optimization_results = _load_optimization_results(optimization_results_path, logger=logger)
    
    # Seleccionar función objetivo
    objective_type = objective_type.lower()
    if objective_type not in UTILITY_FUNCTIONS:
        raise ValueError(f"Función objetivo '{objective_type}' no soportada. Opciones: {list(UTILITY_FUNCTIONS.keys())}")
    objective_fn = UTILITY_FUNCTIONS[objective_type]
    objective_params = objective_params or {}
    
    # Obtener datos de resultados
    propensities = optimization_results['propensities']
    profit = optimization_results['profit']
    volume = optimization_results['volume']
    price_variable = optimization_results['price_variable']
    price_variable_values = np.asarray(optimization_results['price_variable_values'], dtype=float)
    
    # Obtener número de clientes y escenarios
    n_customers = len(df_results)
    n_scenarios = len(propensities)
    
    # Construir matrices (clientes × escenarios)
    profit_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    volume_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    price_variable_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    
    for scenario_idx in range(n_scenarios):
        profit_matrix[:, scenario_idx] = profit[scenario_idx]
        volume_matrix[:, scenario_idx] = volume[scenario_idx]
        price_variable_matrix[:, scenario_idx] = price_variable[scenario_idx]
    
    # Normalizar por cliente (dividir por máximo de cada cliente)
    profit_max = profit_matrix.max(axis=1, keepdims=True)
    volume_max = volume_matrix.max(axis=1, keepdims=True)
    
    # Evitar división por cero
    profit_max = np.where(profit_max == 0, 1, profit_max)
    volume_max = np.where(volume_max == 0, 1, volume_max)
    
    profit_norm = profit_matrix / profit_max
    volume_norm = volume_matrix / volume_max
    
    # Calcular frontera eficiente para cada alpha
    frontier_data = []
    
    for alpha in alpha_values:
        if logger:
            logger.info(f"Procesando alpha={alpha:.3f}")
        
        # Calcular utilidad con función seleccionada
        utility_matrix = objective_fn(
            alpha,
            profit_norm,
            volume_norm,
            monto_raw=profit_matrix,
            ingreso_raw=volume_matrix,
            **objective_params,
        )
        
        # Encontrar escenario óptimo por cliente
        optimal_scenario_idx = utility_matrix.argmax(axis=1)
        
        # Calcular valores agregados (totales)
        total_profit = 0.0
        total_volume = 0.0
        optimal_price_variable_sum = 0.0
        
        for i in range(n_customers):
            scenario_idx = optimal_scenario_idx[i]
            total_profit += profit_matrix[i, scenario_idx]
            total_volume += volume_matrix[i, scenario_idx]
            optimal_price_variable_sum += price_variable_matrix[i, scenario_idx]
        
        # Calcular promedio de price_variable óptimo
        optimal_price_variable_avg = optimal_price_variable_sum / n_customers
        
        frontier_data.append({
            'alpha': alpha,
            f'optimal_{price_variable_name}': optimal_price_variable_avg,
            'total_profit': total_profit,
            'total_volume': total_volume,
        })
    
    df_frontier = pd.DataFrame(frontier_data)
    
    if logger:
        logger.info("=" * 80)
        logger.info("CÁLCULO DE FRONTERA EFICIENTE COMPLETADO")
        logger.info("=" * 80)
        logger.info(f"Puntos en la frontera: {len(df_frontier)}")
        logger.info(f"Profit total - Min: {df_frontier['total_profit'].min():,.0f}, Max: {df_frontier['total_profit'].max():,.0f}")
        logger.info(f"Volume total - Min: {df_frontier['total_volume'].min():,.0f}, Max: {df_frontier['total_volume'].max():,.0f}")
    
    return df_frontier


def optimize_pilot_prices(
    optimization_results: Dict[str, Any],
    df_pilot: pd.DataFrame,
    price_variable_name: str,
    optimal_alpha: float,
    max_price_change_pct: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    objective_type: str = "lineal",
    objective_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Optimiza variable de precio por cliente usando función objetivo con alpha.
    
    Para cada cliente:
    1. Normaliza profit y volume por máximo del cliente
    2. Calcula utility usando la función objetivo seleccionada
    3. Selecciona valor óptimo de la variable de precio: argmax(utility)
    
    Args:
        optimization_results: Resultados de run_price_optimization
        df_pilot: DataFrame con datos de clientes piloto
        price_variable_name: Nombre de la variable de precio a optimizar (ej: "spread", "tasa")
        optimal_alpha: Peso del profit en función objetivo (0=solo volumen, 1=solo profit)
        max_price_change_pct: Porcentaje máximo de cambio de la variable de precio permitido (opcional)
        logger: Logger opcional
        objective_type: Nombre de la función objetivo a utilizar
        objective_params: Parámetros adicionales para la función objetivo
        
    Returns:
        DataFrame con columnas:
        - Todas las columnas originales de df_pilot
        - optimal_{price_variable_name}: Valor óptimo de la variable de precio recomendado
        - optimal_profit: Profit esperado con valor óptimo
        - optimal_volume: Volume esperado con valor óptimo
        - optimal_utility: Utilidad con valor óptimo
    """
    if logger:
        logger.info("=" * 80)
        logger.info("INICIO OPTIMIZACIÓN DE PRECIOS POR CLIENTE")
        logger.info("=" * 80)
        logger.info(f"Variable de precio: {price_variable_name}")
        logger.info(f"Alpha: {optimal_alpha:.2f} (profit: {optimal_alpha*100:.0f}%, volume: {(1-optimal_alpha)*100:.0f}%)")
        logger.info(f"Clientes a optimizar: {len(df_pilot):,}")
    
    # Validar alpha
    if not 0 <= optimal_alpha <= 1:
        raise ValueError(f"Alpha debe estar en [0, 1], recibido: {optimal_alpha}")
    
    # Seleccionar función objetivo
    objective_type = objective_type.lower()
    if objective_type not in UTILITY_FUNCTIONS:
        raise ValueError(f"Función objetivo '{objective_type}' no soportada. Opciones: {list(UTILITY_FUNCTIONS.keys())}")
    objective_fn = UTILITY_FUNCTIONS[objective_type]
    objective_params = objective_params or {}

    if logger:
        logger.info(f"Función objetivo seleccionada: {objective_type} | Parámetros: {objective_params}")

    # Obtener datos de resultados
    propensities = optimization_results['propensities']
    profit = optimization_results['profit']
    volume = optimization_results['volume']
    price_variable = optimization_results['price_variable']
    # Convertir price_variable_values a numpy array para facilitar cálculos
    price_variable_values = np.asarray(optimization_results['price_variable_values'], dtype=float)
    
    # Obtener número de clientes y escenarios
    n_customers = len(df_pilot)
    n_scenarios = len(propensities)
    
    # Construir matrices (clientes × escenarios)
    profit_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    volume_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    price_variable_matrix = np.zeros((n_customers, n_scenarios), dtype=np.float32)
    
    for scenario_idx in range(n_scenarios):
        profit_matrix[:, scenario_idx] = profit[scenario_idx]
        volume_matrix[:, scenario_idx] = volume[scenario_idx]
        price_variable_matrix[:, scenario_idx] = price_variable[scenario_idx]
    
    # Normalizar por cliente (dividir por máximo de cada cliente)
    profit_max = profit_matrix.max(axis=1, keepdims=True)
    volume_max = volume_matrix.max(axis=1, keepdims=True)
    
    # Evitar división por cero
    profit_max = np.where(profit_max == 0, 1, profit_max)
    volume_max = np.where(volume_max == 0, 1, volume_max)
    
    profit_norm = profit_matrix / profit_max
    volume_norm = volume_matrix / volume_max
    
    # Calcular utilidad con función seleccionada
    utility_matrix = objective_fn(
        optimal_alpha,
        profit_norm,
        volume_norm,
        monto_raw=profit_matrix,
        ingreso_raw=volume_matrix,
        **objective_params,
    )
    
    # Aplicar límite de cambio de precio si se especifica
    if max_price_change_pct is not None and price_variable_name in df_pilot.columns:
        if logger:
            logger.info(f"Aplicando límite de cambio de {price_variable_name}: ±{max_price_change_pct*100:.0f}%")
        
        baseline_price = pd.to_numeric(df_pilot[price_variable_name], errors='coerce').fillna(0).values
        
        # Diagnosticar baseline_price
        if logger:
            logger.info(f"Baseline {price_variable_name} - Min: {baseline_price.min():.4f}, Max: {baseline_price.max():.4f}, "
                       f"Promedio: {baseline_price.mean():.4f}, Median: {np.median(baseline_price):.4f}")
            logger.info(f"Escenarios de {price_variable_name} disponibles: {price_variable_values}")
        
        # Crear máscara de precios válidos
        valid_mask = np.ones_like(utility_matrix, dtype=bool)
        
        for scenario_idx in range(n_scenarios):
            scenario_price = price_variable_matrix[:, scenario_idx]
            
            # Calcular cambio de precio
            # Si baseline_price es 0 o muy pequeño, usar cambio absoluto en lugar de porcentual
            baseline_abs = np.abs(baseline_price)
            is_small_baseline = baseline_abs < 0.1  # Si baseline < 0.1, usar cambio absoluto
            
            # Cambio porcentual (solo si baseline no es muy pequeño)
            price_change_pct = np.abs(scenario_price - baseline_price) / (baseline_abs + 1e-8)
            
            # Cambio absoluto (para baselines pequeños)
            price_change_abs = np.abs(scenario_price - baseline_price)
            
            # Para cambio absoluto, permitir un rango más amplio
            # Si max_price_change_pct = 0.3 (30%), permitir cambios absolutos de hasta 30% del máximo
            # O usar un mínimo basado en el rango de valores
            max_abs_change = max(max_price_change_pct * price_variable_values.max(), price_variable_values.max() * 0.1)
            
            # Usar cambio porcentual si baseline no es muy pequeño, sino usar cambio absoluto
            valid_mask[:, scenario_idx] = np.where(
                is_small_baseline,
                price_change_abs <= max_abs_change,  # Cambio absoluto máximo más generoso
                price_change_pct <= max_price_change_pct  # Cambio porcentual máximo
            )
        
        # Verificar que cada cliente tenga al menos un precio válido
        valid_per_customer = valid_mask.any(axis=1)
        if not valid_per_customer.all():
            invalid_customers = np.where(~valid_per_customer)[0]
            if logger:
                logger.warning(
                    f"{len(invalid_customers)} clientes no tienen ningún {price_variable_name} válido dentro del rango. "
                    f"Deshabilitando límite de cambio para estos clientes."
                )
            # Para clientes sin precios válidos, permitir todos los precios
            valid_mask[invalid_customers, :] = True
        
        # Penalizar precios fuera del rango (poner utilidad muy negativa)
        utility_matrix = np.where(valid_mask, utility_matrix, -1e10)
        
        if logger:
            n_valid = valid_mask.sum()
            n_total = valid_mask.size
            logger.info(f"Precios válidos: {n_valid}/{n_total} ({n_valid/n_total*100:.1f}%)")
    
    # Encontrar precio óptimo por cliente
    optimal_scenario_idx = utility_matrix.argmax(axis=1)
    
    # Verificar que no todos los valores sean -1e10
    max_utility_per_customer = utility_matrix.max(axis=1)
    if (max_utility_per_customer <= -1e9).any():
        if logger:
            n_invalid = (max_utility_per_customer <= -1e9).sum()
            logger.warning(
                f"{n_invalid} clientes tienen utilidad máxima <= -1e9. "
                f"Esto indica que todos los precios fueron penalizados. "
                f"Revisar límite de cambio de {price_variable_name} o datos de entrada."
            )
    
    # Diagnosticar distribución de escenarios óptimos
    if logger:
        unique_scenarios, counts = np.unique(optimal_scenario_idx, return_counts=True)
        logger.info(f"Distribución de escenarios óptimos seleccionados:")
        for sc_idx, count in zip(unique_scenarios, counts):
            price_val = price_variable_values[sc_idx] if sc_idx < len(price_variable_values) else 0
            logger.info(f"  Escenario {sc_idx} ({price_variable_name}={price_val:.4f}): {count} clientes ({count/n_customers*100:.1f}%)")
    
    # Extraer valores óptimos
    optimal_price = np.array([price_variable_matrix[i, optimal_scenario_idx[i]] for i in range(n_customers)])
    optimal_profit = np.array([profit_matrix[i, optimal_scenario_idx[i]] for i in range(n_customers)])
    optimal_volume = np.array([volume_matrix[i, optimal_scenario_idx[i]] for i in range(n_customers)])
    optimal_utility = np.array([utility_matrix[i, optimal_scenario_idx[i]] for i in range(n_customers)])
    
    # Diagnosticar si todos tienen el mismo precio
    if logger:
        unique_prices = np.unique(optimal_price)
        if len(unique_prices) == 1:
            logger.warning(
                f"TODOS los clientes tienen el mismo {price_variable_name} óptimo: {unique_prices[0]:.4f}. "
                f"Esto puede indicar un problema con la optimización o el límite de cambio."
            )
    
    # Crear DataFrame de resultados
    df_result = df_pilot.copy()
    df_result[f'optimal_{price_variable_name}'] = optimal_price
    df_result['optimal_profit'] = optimal_profit
    df_result['optimal_volume'] = optimal_volume
    df_result['optimal_utility'] = optimal_utility
    
    if logger:
        logger.info("=" * 80)
        logger.info("OPTIMIZACIÓN DE PRECIOS COMPLETADA")
        logger.info("=" * 80)
        logger.info(f"{price_variable_name} óptimo - Promedio: {optimal_price.mean():.4f}, Min: {optimal_price.min():.4f}, Max: {optimal_price.max():.4f}")
        logger.info(f"Profit óptimo - Promedio: {optimal_profit.mean():,.0f}, Total: {optimal_profit.sum():,.0f}")
        logger.info(f"Volume óptimo - Promedio: {optimal_volume.mean():,.0f}, Total: {optimal_volume.sum():,.0f}")
        logger.info(f"Utility óptimo - Promedio: {optimal_utility.mean():.4f}")
    
    return df_result


# ============================================================================
# Funciones wrapper para compatibilidad hacia atrás (deprecated)
# Estas funciones mantienen la interfaz antigua pero internamente llaman
# a las funciones genéricas.
# ============================================================================

def run_spread_optimization(
    unified_model: Dict[str, Any],
    df_val: pd.DataFrame,
    di_value: float,
    spread_range: Dict[str, float],
    segmentation_pipeline_path: str,
    n_jobs: int = 1,
    calculate_rate_fn: Optional[Callable[[float, float], float]] = None,
    save_results_path: Optional[str] = None,
    compute_efficient_frontier: bool = False,
    alpha_values: Optional[np.ndarray] = None,
    objective_type: str = "lineal",
    objective_params: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    [DEPRECATED] Función wrapper para compatibilidad hacia atrás.
    
    Esta función mantiene la interfaz antigua pero internamente llama a
    run_price_optimization. Se recomienda usar run_price_optimization directamente.
    
    Args:
        unified_model: Modelo unificado con pipelines y modelos por cluster
        df_val: DataFrame con datos de clientes
        di_value: [DEPRECATED] No se usa, se mantiene solo para compatibilidad
        spread_range: Diccionario con 'spread_min', 'spread_max', 'step_size'
        segmentation_pipeline_path: Ruta al pipeline de segmentación (.pkl)
        n_jobs: Número de trabajos paralelos
        calculate_rate_fn: [DEPRECATED] No se usa, se mantiene solo para compatibilidad
        save_results_path: Ruta donde guardar resultados del paso 1 (opcional)
        compute_efficient_frontier: Si True, calcula la frontera eficiente
        alpha_values: Array de valores de alpha para frontera eficiente
        objective_type: Tipo de función objetivo para frontera eficiente
        objective_params: Parámetros adicionales para función objetivo
        logger: Logger opcional
        
    Returns:
        Diccionario con resultados (usando nombres antiguos: 'spread', 'spread_values')
        Si compute_efficient_frontier=True, también incluye 'efficient_frontier'
    """
    # Generar valores de spread
    spread_min = spread_range['spread_min']
    spread_max = spread_range['spread_max']
    step_size = spread_range['step_size']
    spread_values = np.arange(spread_min, spread_max + step_size, step_size)
    
    # Llamar a la función genérica
    final_results = run_price_optimization(
        unified_model=unified_model,
        df_val=df_val,
        price_variable_name="spread",
        price_variable_values=spread_values,
        segmentation_pipeline_path=segmentation_pipeline_path,
        amount_column="importe",
        n_jobs=n_jobs,
        save_results_path=save_results_path,
        compute_efficient_frontier=compute_efficient_frontier,
        alpha_values=alpha_values,
        objective_type=objective_type,
        objective_params=objective_params,
        logger=logger,
    )
    
    # Extraer optimization_results
    results = final_results['optimization_results']
    
    # Mapear nombres de resultados para compatibilidad hacia atrás
    results['spread'] = results.pop('price_variable')
    results['spread_values'] = results.pop('price_variable_values')
    
    # Si se calculó frontera eficiente, agregarla al resultado
    if 'efficient_frontier' in final_results:
        results['efficient_frontier'] = final_results['efficient_frontier']
    
    return results


def optimize_pilot_spreads(
    optimization_results: Dict[str, Any],
    df_pilot: pd.DataFrame,
    optimal_alpha: float,
    max_spread_change_pct: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
    objective_type: str = "lineal",
    objective_params: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    [DEPRECATED] Función wrapper para compatibilidad hacia atrás.
    
    Esta función mantiene la interfaz antigua pero internamente llama a
    optimize_pilot_prices. Se recomienda usar optimize_pilot_prices directamente.
    
    Args:
        optimization_results: Resultados de run_spread_optimization
        df_pilot: DataFrame con datos de clientes
        optimal_alpha: Peso del profit en función objetivo
        max_spread_change_pct: Porcentaje máximo de cambio de spread permitido
        logger: Logger opcional
        objective_type: Nombre de la función objetivo
        objective_params: Parámetros adicionales para la función objetivo
        
    Returns:
        DataFrame con columnas optimal_spread, optimal_profit, optimal_volume, optimal_utility
    """
    # Mapear nombres de resultados para compatibilidad
    if 'spread' in optimization_results:
        optimization_results['price_variable'] = optimization_results.pop('spread')
    if 'spread_values' in optimization_results:
        optimization_results['price_variable_values'] = optimization_results.pop('spread_values')
    
    # Llamar a la función genérica
    df_result = optimize_pilot_prices(
        optimization_results=optimization_results,
        df_pilot=df_pilot,
        price_variable_name="spread",
        optimal_alpha=optimal_alpha,
        max_price_change_pct=max_spread_change_pct,
        logger=logger,
        objective_type=objective_type,
        objective_params=objective_params,
    )
    
    # Mapear nombres de columnas para compatibilidad hacia atrás
    if 'optimal_spread' not in df_result.columns and 'optimal_spread' in df_result.columns:
        # Ya está mapeado por optimize_pilot_prices
        pass
    
    return df_result

