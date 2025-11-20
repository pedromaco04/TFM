import os
import logging
import pickle
import io
from typing import Dict, Optional, Any, List
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importar funciones de feature_pro
from feature_pro import (
    read_dataset,
    detect_column_types,
    impute_missing,
    fit_numeric_preprocessor,
    transform_with_preprocessor,
    apply_standard_scaler,
    fit_standard_scaler,
    convert_to_logarithm,
)

# Importar funciones de modeling_pro
from modeling_pro import (
    fit_logistic_regression,
    predict_logistic,
    get_coefficients_summary,
    check_directionality,
    verify_tasa_direction,
    get_directionality_report,
    calculate_gini,
    calculate_gini_by_period,
    validate_gini_stability,
    plot_gini_over_time,
)

# ===== CONFIGURACIÓN =====
# Ruta del archivo unificado con todos los clusters
UNIFIED_DATA_PATH = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/02.Feature/seleccion_de_var_unificado.parquet'

# Ruta base para guardar resultados en S3
OUTPUT_BASE_PATH = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/03.Modelling'

# Columna objetivo
TARGET_COLUMN = 'flag_vta'

VARS_BY_CLUSTER_PATH = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/02.Feature/vars_finales_x_cluster.csv'


def setup_logger(log_file: str) -> logging.Logger:
    """
    Configura el logger para el proceso de modelamiento.
    """
    logger = logging.getLogger("modeling_engine")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str,
    exclude_cols: Optional[List[str]] = None,
    impute_strategy: str = 'median',
    winsorize: bool = True,
    winsorize_lower: float = 0.01,
    winsorize_upper: float = 0.99,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Construye un pipeline completo de preprocesamiento.
    
    Incluye:
    - Detección de tipos (numéricas/categóricas)
    - Imputación para numéricas y categóricas
    - Winsorización para numéricas
    - Conversión a logaritmo para numéricas (después de imputación y winsorización)
    - StandardScaler para numéricas (después de conversión a logaritmo)
    - One-Hot Encoding para categóricas
    
    Args:
        df: DataFrame de entrada
        target_column: nombre de la columna objetivo (se excluye del pipeline)
        exclude_cols: columnas adicionales a excluir
        impute_strategy: estrategia de imputación para numéricas ('median', 'mean', 'mode')
        winsorize: si True, aplica winsorización
        winsorize_lower: percentil inferior para winsorización
        winsorize_upper: percentil superior para winsorización
        logger: logger opcional
        
    Returns:
        dict con el pipeline completo
    """
    if exclude_cols is None:
        exclude_cols = []
    
    # Columnas a excluir
    cols_to_exclude = [target_column] + exclude_cols
    
    # Detectar tipos
    num_cols, cat_cols = detect_column_types(df)
    
    # Filtrar columnas que están en el DataFrame y no están excluidas
    num_cols = [c for c in num_cols if c in df.columns and c not in cols_to_exclude]
    cat_cols = [c for c in cat_cols if c in df.columns and c not in cols_to_exclude]
    
    if logger:
        logger.info(f"Columnas numéricas detectadas: {len(num_cols)}")
        logger.info(f"Columnas categóricas detectadas: {len(cat_cols)}")
    
    # Construir preprocesador numérico usando feature_pro
    numeric_preprocessor = fit_numeric_preprocessor(
        df=df,
        numeric_columns=num_cols,
        impute_strategy=impute_strategy,
        groupby=None,  # Sin agrupación por segmento en este caso
        winsorize=winsorize,
        lower=winsorize_lower,
        upper=winsorize_upper,
    )
    
    # Aplicar transformaciones numéricas (imputación y winsorización) para entrenar el scaler
    # Esto asegura que el scaler se entrene con datos ya imputados y winsorizados
    df_numeric_processed = transform_with_preprocessor(df, numeric_preprocessor)
    
    # Aplicar conversión a logaritmo a las variables numéricas
    # Esto se hace después de imputación y winsorización, pero antes del StandardScaler
    log_transform_params = {
        'enabled': True,
        'base': 'natural',
        'handle_zeros': 'add_one',  # log1p para manejar ceros
        'handle_negatives': 'skip',  # Omitir columnas con negativos
        'add_constant': 1.0,
        'columns_to_convert': num_cols.copy() if num_cols else [],
    }
    
    # Rastrear qué columnas se convertirán realmente (excluyendo las que tienen negativos si handle_negatives='skip')
    columns_converted = []
    if num_cols:
        # Verificar qué columnas tienen negativos (se omitirán si handle_negatives='skip')
        for col in num_cols:
            if col in df_numeric_processed.columns:
                has_negatives = (df_numeric_processed[col] < 0).any()
                if not has_negatives:
                    columns_converted.append(col)
                elif logger:
                    logger.warning(f"Columna '{col}' tiene valores negativos. Se omitirá de la conversión a logaritmo.")
        
        # Aplicar conversión a logaritmo
        df_numeric_processed = convert_to_logarithm(
            df=df_numeric_processed,
            columns=num_cols,
            base=log_transform_params['base'],
            handle_zeros=log_transform_params['handle_zeros'],
            handle_negatives=log_transform_params['handle_negatives'],
            add_constant=log_transform_params['add_constant'],
            auto_detect=False,
            logger=logger,
        )
        if logger:
            logger.info(f"Conversión a logaritmo aplicada a {len(columns_converted)} columnas numéricas (de {len(num_cols)} totales)")
    
    log_transform_params['columns_converted'] = columns_converted
    
    # Entrenar StandardScaler con las variables numéricas ya procesadas (imputadas, winsorizadas y logarítmicas)
    standard_scaler = None
    if num_cols:
        standard_scaler = fit_standard_scaler(
            df=df_numeric_processed,
            columns=num_cols,
            logger=logger,
        )
        if logger:
            logger.info(f"StandardScaler entrenado para {len(num_cols)} columnas numéricas")
    
    # One-Hot Encoding para categóricas
    onehot_encoder = None
    categorical_feature_names = []
    valid_cat_cols = []
    
    if cat_cols:
        # Filtrar categóricas con pocos valores únicos (evitar explosión de features)
        max_categories = 20
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= max_categories:
                valid_cat_cols.append(col)
            elif logger:
                logger.warning(f"Columna categórica '{col}' tiene {n_unique} valores únicos (> {max_categories}), se excluye del OHE")
        
        if valid_cat_cols:
            # Ajustar OneHotEncoder
            onehot_encoder = OneHotEncoder(
                drop='first',  # Evitar multicolinealidad
                sparse_output=False,
                handle_unknown='ignore',  # Ignorar categorías nuevas en test
            )
            
            # Imputar categóricas antes de ajustar OHE
            cat_data = df[valid_cat_cols].copy()
            for col in valid_cat_cols:
                # Imputar con moda
                cat_data[col] = impute_missing(
                    cat_data[[col]], 
                    columns=[col], 
                    strategy='mode',
                    logger=logger
                )[col]
            
            # Ajustar con datos de entrenamiento
            onehot_encoder.fit(cat_data)
            
            # Obtener nombres de features generadas
            for i, col in enumerate(valid_cat_cols):
                categories = onehot_encoder.categories_[i]
                # drop='first' elimina la primera categoría
                for cat in categories[1:]:
                    categorical_feature_names.append(f"{col}_{cat}")
            
            if logger:
                logger.info(f"One-Hot Encoding aplicado a {len(valid_cat_cols)} columnas categóricas")
                logger.info(f"Features categóricas generadas: {len(categorical_feature_names)}")
        else:
            if logger:
                logger.info("No hay columnas categóricas válidas para One-Hot Encoding")
    
    # Lista final de features
    feature_columns = num_cols + categorical_feature_names
    
    pipeline = {
        'numeric_preprocessor': numeric_preprocessor,  # Incluye imputación y winsorización
        'log_transform': log_transform_params,  # Parámetros y columnas convertidas a logaritmo
        'standard_scaler': standard_scaler,  # StandardScaler para normalización
        'numeric_columns': num_cols,
        'categorical_columns': valid_cat_cols,
        'onehot_encoder': onehot_encoder,
        'categorical_feature_names': categorical_feature_names,
        'feature_columns': feature_columns,
        'target_column': target_column,
    }
    
    if logger:
        logger.info("Pipeline construido con:")
        logger.info(f"  - Imputación: {impute_strategy}")
        logger.info(f"  - Winsorización: {'Sí' if winsorize else 'No'} (percentiles [{winsorize_lower}, {winsorize_upper}])")
        logger.info(f"  - Conversión a logaritmo: Sí (base natural, log1p para ceros)")
        logger.info(f"  - StandardScaler: {'Sí' if standard_scaler is not None else 'No'}")
    
    return pipeline


def load_cluster_variables(
    csv_path: str,
    target_column: str,
    logger: Optional[logging.Logger] = None,
) -> Dict[int, List[str]]:
    """
    Lee el CSV consolidado de variables por cluster y construye un dict {cluster: [variables]}.
    Asegura que la columna objetivo esté incluida.
    """
    df_vars = read_dataset(csv_path, fmt='csv', logger=logger)
    required_cols = {'cluster', 'variables'}
    if not required_cols.issubset(df_vars.columns):
        missing = required_cols - set(df_vars.columns)
        raise ValueError(f"El archivo de variables por cluster no tiene las columnas requeridas: {missing}")

    cluster_map: Dict[int, List[str]] = {}
    for _, row in df_vars.iterrows():
        cluster_id = int(row['cluster'])
        raw_variables = str(row['variables']) if pd.notna(row['variables']) else ''
        variable_list = [v.strip() for v in raw_variables.split(',') if v.strip()]
        if target_column not in variable_list:
            variable_list.append(target_column)
        cluster_map[cluster_id] = variable_list

    if logger:
        logger.info(f"Variables por cluster cargadas desde {csv_path}. Total clusters definidos: {len(cluster_map)}")
    return cluster_map


def apply_preprocessing_pipeline(
    df: pd.DataFrame,
    pipeline: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Aplica el pipeline de preprocesamiento a un DataFrame.
    
    Args:
        df: DataFrame a transformar
        pipeline: pipeline construido con build_preprocessing_pipeline
        logger: logger opcional
        
    Returns:
        DataFrame transformado con solo las features (numéricas + categóricas codificadas)
    """
    result = df.copy()
    
    # Obtener columnas numéricas y preprocesador
    numeric_cols = pipeline['numeric_columns']
    numeric_preprocessor = pipeline['numeric_preprocessor']
    standard_scaler = pipeline.get('standard_scaler')
    
    # Aplicar transformación numérica usando feature_pro
    # Esto aplica: 1) Conversión a float, 2) Imputación, 3) Winsorización
    if numeric_cols:
        result = transform_with_preprocessor(result, numeric_preprocessor)
        if logger:
            logger.debug(f"Transformaciones numéricas aplicadas (imputación + winsorización) a {len(numeric_cols)} columnas")
        
        # Aplicar conversión a logaritmo después de imputación y winsorización
        # Usar los parámetros guardados en el pipeline
        log_transform = pipeline.get('log_transform', {})
        if log_transform.get('enabled', False):
            result = convert_to_logarithm(
                df=result,
                columns=numeric_cols,
                base=log_transform.get('base', 'natural'),
                handle_zeros=log_transform.get('handle_zeros', 'add_one'),
                handle_negatives=log_transform.get('handle_negatives', 'skip'),
                add_constant=log_transform.get('add_constant', 1.0),
                auto_detect=False,
                logger=logger,
            )
            columns_converted = log_transform.get('columns_converted', [])
            if logger:
                logger.debug(f"Conversión a logaritmo aplicada a {len(columns_converted)} columnas numéricas (usando parámetros del pipeline)")
        
        # Aplicar StandardScaler después de imputación, winsorización y conversión a logaritmo
        if standard_scaler is not None:
            result = apply_standard_scaler(
                df=result,
                columns=numeric_cols,
                scaler=standard_scaler,
                logger=logger,
            )
            if logger:
                logger.info(f"StandardScaler aplicado a {len(numeric_cols)} columnas numéricas")
    
    # Aplicar One-Hot Encoding a categóricas
    onehot_encoder = pipeline.get('onehot_encoder')
    cat_cols = pipeline.get('categorical_columns', [])
    
    if onehot_encoder and cat_cols:
        # Filtrar solo las categóricas válidas que se usaron para entrenar
        valid_cat_cols = [c for c in cat_cols if c in result.columns]
        if valid_cat_cols:
            # Imputar categóricas antes de transformar
            cat_data = result[valid_cat_cols].copy()
            for col in valid_cat_cols:
                cat_data[col] = impute_missing(
                    cat_data[[col]], 
                    columns=[col], 
                    strategy='mode',
                    logger=logger
                )[col]
            
            cat_encoded = onehot_encoder.transform(cat_data)
            
            # Crear DataFrame con nombres de columnas
            cat_feature_names = pipeline['categorical_feature_names']
            cat_df = pd.DataFrame(
                cat_encoded,
                columns=cat_feature_names,
                index=result.index
            )
            
            # Eliminar columnas categóricas originales y agregar las codificadas
            result = result.drop(columns=valid_cat_cols)
            result = pd.concat([result, cat_df], axis=1)
            
            if logger:
                logger.info(f"One-Hot Encoding aplicado: {len(valid_cat_cols)} columnas -> {len(cat_feature_names)} features")
    
    # Retornar solo las features
    feature_columns = pipeline['feature_columns']
    available_features = [f for f in feature_columns if f in result.columns]
    
    return result[available_features]


def save_pipeline(
    pipeline: Dict[str, Any],
    output_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Guarda el pipeline de preprocesamiento en un archivo .pkl.
    Soporta rutas locales y S3.
    """
    # Crear directorio si no existe (solo para rutas locales)
    if not output_path.startswith('s3://'):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(pipeline, f)
    else:
        # Para S3, usar boto3 si está disponible, sino guardar localmente
        try:
            import boto3
            
            # Parsear URL de S3
            parsed = urlparse(output_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Guardar en S3 usando boto3
            s3_client = boto3.client('s3')
            buffer = io.BytesIO()
            pickle.dump(pipeline, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket, key)
            
            if logger:
                logger.info(f"Pipeline guardado en S3: {output_path}")
        except ImportError:
            # Si boto3 no está disponible, guardar localmente con advertencia
            local_path = os.path.join(os.path.dirname(__file__), os.path.basename(output_path))
            with open(local_path, 'wb') as f:
                pickle.dump(pipeline, f)
            if logger:
                logger.warning(f"boto3 no disponible. Pipeline guardado localmente en: {local_path}")
                logger.warning(f"Ruta S3 original: {output_path}")
    
    return output_path


def save_csv_to_s3(
    df: pd.DataFrame,
    output_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Guarda un DataFrame en CSV. Soporta rutas locales y S3.
    """
    if not output_path.startswith('s3://'):
        # Guardar localmente
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        df.to_csv(output_path, index=False)
    else:
        # Para S3, usar boto3
        try:
            import boto3
            
            # Parsear URL de S3
            parsed = urlparse(output_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Guardar en S3 usando boto3
            s3_client = boto3.client('s3')
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            s3_client.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())
            
            if logger:
                logger.info(f"CSV guardado en S3: {output_path}")
        except ImportError:
            # Si boto3 no está disponible, guardar localmente con advertencia
            local_path = os.path.join(os.path.dirname(__file__), os.path.basename(output_path))
            df.to_csv(local_path, index=False)
            if logger:
                logger.warning(f"boto3 no disponible. CSV guardado localmente en: {local_path}")
                logger.warning(f"Ruta S3 original: {output_path}")
        except Exception as e:
            if logger:
                logger.error(f"Error guardando CSV en S3: {e}")
            raise
    
    return output_path


def plot_spread_probability_curve(
    models_dict: Dict[str, Any],
    preprocessing_pipeline: Dict[str, Any],
    df_reference: pd.DataFrame,
    spread_column: str = 'spread',
    n_points: int = 100,
    logger: Optional[logging.Logger] = None,
) -> Optional[plt.Figure]:
    """
    Genera una gráfica que muestra la relación entre spread y probabilidad predicha.
    La curva debería mostrar que a menor spread, mayor probabilidad.
    
    Args:
        models_dict: diccionario con el modelo entrenado
        preprocessing_pipeline: pipeline de preprocesamiento
        df_reference: DataFrame de referencia para obtener valores promedio de otras variables
        spread_column: nombre de la columna de spread
        n_points: número de puntos para la curva
        logger: logger opcional
        
    Returns:
        Figura de matplotlib o None si hay error
    """
    try:
        # Obtener el modelo desde el diccionario retornado por fit_logistic_regression
        models_container = models_dict.get('models')
        if not models_container:
            if logger:
                logger.warning("No se encontró la clave 'models' en models_dict. No se puede generar gráfica.")
            return None
        
        # Usar el modelo global si existe, de lo contrario tomar el primero disponible
        if 'global' in models_container:
            model = models_container['global']
            model_segment = 'global'
        else:
            model_segment, model = next(iter(models_container.items()))
            if logger:
                logger.info(f"Usando modelo del segmento '{model_segment}' para la gráfica de spread.")
        
        feature_columns = models_dict.get('feature_columns', preprocessing_pipeline.get('feature_columns', []))
        
        # Verificar que spread está en las features
        if spread_column not in feature_columns:
            if logger:
                logger.warning(f"'{spread_column}' no está en las feature_columns. No se puede generar gráfica.")
            return None
        
        # Obtener valores de referencia (promedios) de todas las features excepto spread
        # Usar el DataFrame de referencia ya transformado
        df_ref_transformed = apply_preprocessing_pipeline(df_reference, preprocessing_pipeline, logger=logger)
        
        # Verificar que spread está en el DataFrame transformado
        if spread_column not in df_ref_transformed.columns:
            if logger:
                logger.warning(f"'{spread_column}' no encontrada en DataFrame transformado. No se puede generar gráfica.")
            return None
        
        # Obtener valores promedio de todas las features (excepto spread)
        other_features = [f for f in feature_columns if f != spread_column]
        reference_values = {}
        for feat in other_features:
            if feat in df_ref_transformed.columns:
                reference_values[feat] = df_ref_transformed[feat].mean()
            else:
                # Si la feature no está, usar 0 como default
                reference_values[feat] = 0.0
                if logger:
                    logger.warning(f"Feature '{feat}' no encontrada en DataFrame transformado. Usando 0.0")
        
        # Crear rango de valores de spread usando los valores ya transformados
        spread_min = df_ref_transformed[spread_column].min()
        spread_max = df_ref_transformed[spread_column].max()
        
        # Crear array de valores de spread (ya transformados)
        spread_values = np.linspace(spread_min, spread_max, n_points)
        
        # Para cada valor de spread, crear un DataFrame con todas las features
        probabilities = []
        valid_spread_values = []
        
        for spread_val in spread_values:
            try:
                # Crear DataFrame con una fila
                row_data = reference_values.copy()
                row_data[spread_column] = spread_val
                
                # Crear DataFrame con todas las features en el orden correcto
                X = pd.DataFrame([row_data], columns=feature_columns)
                
                # Asegurar que todas las features estén presentes
                for feat in feature_columns:
                    if feat not in X.columns:
                        X[feat] = 0.0
                
                # Reordenar columnas para que coincidan con el orden del modelo
                X = X[feature_columns]
                
                # Predecir probabilidad
                proba = model.predict_proba(X)[0, 1]  # Probabilidad de clase positiva
                probabilities.append(proba)
                valid_spread_values.append(spread_val)
                
            except Exception as e:
                if logger:
                    logger.warning(f"Error prediciendo para spread={spread_val}: {e}")
                continue
        
        if len(probabilities) == 0:
            if logger:
                logger.warning("No se pudieron generar predicciones. No se puede crear gráfica.")
            return None
        
        # Crear gráfica
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Ordenar por spread para tener una curva suave
        sorted_indices = np.argsort(valid_spread_values)
        sorted_spread = np.array(valid_spread_values)[sorted_indices]
        sorted_probs = np.array(probabilities)[sorted_indices]
        
        # Graficar curva
        ax.plot(sorted_spread, sorted_probs, linewidth=2, color='#2E86AB', label='Probabilidad predicha')
        ax.fill_between(sorted_spread, sorted_probs, alpha=0.3, color='#2E86AB')
        
        # Agregar línea de referencia en el punto medio
        mid_idx = len(sorted_spread) // 2
        ax.axvline(x=sorted_spread[mid_idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.axhline(y=sorted_probs[mid_idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # Configurar ejes y título
        ax.set_xlabel('Spread', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probabilidad de Venta (flag_vta=1)', fontsize=12, fontweight='bold')
        ax.set_title('Relación entre Spread y Probabilidad de Venta\n(Mayor spread → Menor probabilidad)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        # Agregar anotación con la dirección esperada
        ax.text(0.02, 0.98, 
                'Dirección esperada: ↓ spread → ↑ probabilidad',
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        if logger:
            logger.info(f"Gráfica de probabilidad vs spread generada con {len(probabilities)} puntos")
            logger.info(f"Rango de spread: [{spread_min:.4f}, {spread_max:.4f}]")
            logger.info(f"Rango de probabilidad: [{min(probabilities):.4f}, {max(probabilities):.4f}]")
        
        return fig
        
    except Exception as e:
        if logger:
            logger.error(f"Error generando gráfica de probabilidad vs spread: {e}", exc_info=True)
        return None


def save_model(
    models_dict: Dict[str, Any],
    output_path: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """
    Guarda el modelo de regresión logística en un archivo .pkl.
    Soporta rutas locales y S3.
    """
    # Crear directorio si no existe (solo para rutas locales)
    if not output_path.startswith('s3://'):
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(models_dict, f)
    else:
        # Para S3, usar boto3 si está disponible, sino guardar localmente
        try:
            import boto3
            
            # Parsear URL de S3
            parsed = urlparse(output_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip('/')
            
            # Guardar en S3 usando boto3
            s3_client = boto3.client('s3')
            buffer = io.BytesIO()
            pickle.dump(models_dict, buffer)
            buffer.seek(0)
            s3_client.upload_fileobj(buffer, bucket, key)
            
            if logger:
                logger.info(f"Modelo guardado en S3: {output_path}")
        except ImportError:
            # Si boto3 no está disponible, guardar localmente con advertencia
            local_path = os.path.join(os.path.dirname(__file__), os.path.basename(output_path))
            with open(local_path, 'wb') as f:
                pickle.dump(models_dict, f)
            if logger:
                logger.warning(f"boto3 no disponible. Modelo guardado localmente en: {local_path}")
                logger.warning(f"Ruta S3 original: {output_path}")
    
    return output_path


def process_cluster(
    cluster_num: int,
    df_cluster: pd.DataFrame,
    cluster_variables: List[str],
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    Procesa un cluster completo: entrena modelo, valida y retorna resultados.
    
    Args:
        cluster_num: número del cluster (1, 2, 3, 4, 5)
        df_cluster: DataFrame filtrado para este cluster con solo las variables relevantes
        cluster_variables: lista de variables específicas para este cluster (incluye flag_vta)
        logger: logger
        
    Returns:
        dict con pipeline, modelo, métricas y resultados del cluster
    """
    cluster_name = f'cluster_{cluster_num}'
    logger.info(f"== INICIO PROCESAMIENTO {cluster_name.upper()} ==")
    print(f"\n== Procesando {cluster_name} ==")
    
    logger.info(f"Datos del cluster. Shape: {df_cluster.shape}")
    logger.info(f"Variables del cluster ({len(cluster_variables)}): {cluster_variables}")
    
    # Validar que existe la columna objetivo
    if TARGET_COLUMN not in df_cluster.columns:
        logger.error(f"Columna objetivo '{TARGET_COLUMN}' no encontrada. Abortando.")
        return None
    
    # Filtrar DataFrame para incluir solo las variables del cluster (y columnas auxiliares si existen)
    # Mantener columnas auxiliares como codmes, quarter_period si existen
    aux_cols = ['codmes', 'quarter_period']
    cols_to_keep = [c for c in cluster_variables if c in df_cluster.columns]
    cols_to_keep += [c for c in aux_cols if c in df_cluster.columns]
    # Asegurar que flag_vta esté presente
    if TARGET_COLUMN in df_cluster.columns and TARGET_COLUMN not in cols_to_keep:
        cols_to_keep.append(TARGET_COLUMN)
    df_cluster = df_cluster[cols_to_keep].copy()
    
    logger.info(f"DataFrame filtrado. Shape: {df_cluster.shape}, Columnas: {list(df_cluster.columns)}")
    
    # 2. Construir pipeline de preprocesamiento
    print("== Construyendo pipeline de preprocesamiento ==")
    exclude_cols = ['codmes', 'quarter_period']  # Columnas a excluir del pipeline
    
    preprocessing_pipeline = build_preprocessing_pipeline(
        df=df_cluster,
        target_column=TARGET_COLUMN,
        exclude_cols=exclude_cols,
        impute_strategy='median',
        winsorize=True,
        winsorize_lower=0.01,
        winsorize_upper=0.99,
        logger=logger,
    )
    
    feature_columns = preprocessing_pipeline['feature_columns']
    logger.info(f"Pipeline construido. Features finales: {len(feature_columns)}")
    logger.info(f"  - Numéricas: {len(preprocessing_pipeline['numeric_columns'])}")
    logger.info(f"  - Categóricas codificadas: {len(preprocessing_pipeline['categorical_feature_names'])}")
    
    # 3. Aplicar pipeline y dividir datos
    print("== Aplicando pipeline y dividiendo datos ==")
    df_transformed = apply_preprocessing_pipeline(df_cluster, preprocessing_pipeline, logger=logger)
    
    # Agregar target de vuelta (usando índice para asegurar alineación)
    df_transformed = df_transformed.copy()
    df_transformed[TARGET_COLUMN] = df_cluster[TARGET_COLUMN].values
    
    # Agregar columnas auxiliares de vuelta (codmes, quarter_period) para análisis temporal
    aux_cols_to_add = ['codmes', 'quarter_period']
    for col in aux_cols_to_add:
        if col in df_cluster.columns:
            # Usar el índice para asegurar que los valores se alineen correctamente
            df_transformed[col] = df_cluster[col].values
            logger.info(f"Columna auxiliar '{col}' agregada al DataFrame transformado")
    
    logger.info(f"DataFrame transformado final. Shape: {df_transformed.shape}, Columnas: {list(df_transformed.columns)}")
    
    # Split train/validation
    y = df_transformed[TARGET_COLUMN]
    df_train, df_validation, y_train, y_valid = train_test_split(
        df_transformed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    df_train[TARGET_COLUMN] = y_train
    df_validation[TARGET_COLUMN] = y_valid
    
    logger.info(f"Split realizado - Train: {df_train.shape}, Validation: {df_validation.shape}")
    
    # 5. Entrenar modelo de regresión logística
    print("== Entrenando modelo de regresión logística ==")
    models_dict = fit_logistic_regression(
        df=df_train,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
        segment_column=None,  # Modelo global por cluster
        random_state=42,
        max_iter=1000,
        logger=logger,
    )
    
    # Mostrar métricas
    summary = models_dict['summary']
    print("\n== Resumen de métricas ==")
    print(summary.to_string(index=False))
    logger.info("Resumen de métricas:\n" + summary.to_string(index=False))
    
    # 6. Validaciones de direccionalidad
    print("== Verificando direccionalidad ==")
    
    # Verificar TASA
    tasa_check_result = None
    if 'tasa' in feature_columns:
        tasa_check_result = verify_tasa_direction(
            models_dict=models_dict,
            tasa_column='tasa',
            expected_negative=True,
            logger=logger,
        )
        
        if tasa_check_result.get('all_correct', False):
            print("[OK] Direccionalidad de TASA correcta")
        else:
            print("[ERROR] Direccionalidad de TASA incorrecta")
    
    # Verificar otras direccionalidades
    expected_directions = {
        'tasa': 'negative',
        'spread':'negative'
    }
    
    directionality_check = check_directionality(
        models_dict=models_dict,
        feature_columns=feature_columns,
        expected_directions=expected_directions,
        logger=logger,
    )
    
    # Obtener reporte de direccionalidad
    directionality_report = None
    if not directionality_check['summary'].empty:
        directionality_report = get_directionality_report(
            models_dict=models_dict,
            feature_columns=feature_columns,
            expected_directions=expected_directions,
            logger=logger,
        )
    
    # 8. Predicciones en validación
    print("== Generando predicciones en validación ==")
    predictions = predict_logistic(
        models_dict=models_dict,
        df=df_validation,
        segment_column=None,
        return_proba=True,
        logger=logger,
    )
    
    df_validation['pred_proba'] = predictions['pred_proba'].values
    
    # Calcular GINI global
    gini_validation = calculate_gini(
        y_true=df_validation[TARGET_COLUMN],
        y_pred_proba=df_validation['pred_proba'],
        logger=logger,
    )
    print(f"GINI en validación: {gini_validation:.4f}")
    
    # 8.1. Generar gráfica de probabilidad vs spread
    spread_probability_figure = None
    if 'spread' in feature_columns:
        print("== Generando gráfica de probabilidad vs spread ==")
        logger.info(f"Generando gráfica de probabilidad vs spread (spread está en feature_columns)")
        spread_probability_figure = plot_spread_probability_curve(
            models_dict=models_dict,
            preprocessing_pipeline=preprocessing_pipeline,
            df_reference=df_validation,
            spread_column='spread',
            n_points=100,
            logger=logger,
        )
        if spread_probability_figure is not None:
            logger.info("✓ Gráfica de probabilidad vs spread generada exitosamente")
            print("✓ Gráfica de probabilidad vs spread generada")
        else:
            logger.warning("✗ No se pudo generar la gráfica de probabilidad vs spread")
            print("✗ No se pudo generar la gráfica de probabilidad vs spread")
    else:
        logger.info(f"Spread no está en feature_columns. No se generará gráfica de probabilidad vs spread.")
        logger.info(f"Feature columns disponibles: {feature_columns}")
    
    # 9. GINI por período (si existe columna temporal)
    gini_by_period = None
    stability = None
    period_column = None
    
    # Verificar columnas disponibles en df_validation
    logger.info(f"Columnas disponibles en df_validation: {list(df_validation.columns)}")
    
    if 'codmes' in df_validation.columns:
        period_column = 'codmes'
        logger.info("Columna 'codmes' encontrada en df_validation. Se usará para análisis temporal.")
    elif 'quarter_period' in df_validation.columns:
        period_column = 'quarter_period'
        logger.info("Columna 'quarter_period' encontrada en df_validation. Se usará para análisis temporal.")
    else:
        logger.warning("No se encontró 'codmes' ni 'quarter_period' en df_validation. No se calculará GINI por período.")
        logger.warning(f"Columnas en df_validation: {list(df_validation.columns)}")
    
    gini_plot_figure = None
    if period_column:
        print("== Calculando GINI por período ==")
        logger.info(f"Calculando GINI por período usando columna: {period_column}")
        gini_by_period = calculate_gini_by_period(
            df=df_validation,
            y_true_column=TARGET_COLUMN,
            y_pred_proba_column='pred_proba',
            period_column=period_column,
            logger=logger,
        )
        
        # Validar estabilidad
        stability = validate_gini_stability(
            gini_by_period_df=gini_by_period,
            threshold_degradation=0.05,
            logger=logger,
        )
        
        if stability['is_stable']:
            print("[OK] GINI es estable en el tiempo")
        else:
            print("[ERROR] ADVERTENCIA: GINI muestra inestabilidad")
        
        # Generar gráfico de GINI en el tiempo
        print("== Generando gráfico de GINI en el tiempo ==")
        logger.info("Generando gráfico de GINI en el tiempo")
        gini_plot_figure = plot_gini_over_time(
            gini_by_period_df=gini_by_period,
            output_path=None,  # No guardar aquí, se guardará después
            return_fig=True,
            logger=logger,
        )
        if gini_plot_figure is not None:
            logger.info("✓ Gráfico de GINI en el tiempo generado exitosamente")
            print("✓ Gráfico de GINI en el tiempo generado")
        else:
            logger.warning("✗ No se pudo generar el gráfico de GINI en el tiempo")
            print("✗ No se pudo generar el gráfico de GINI en el tiempo")
    else:
        logger.info("No se generará gráfico de GINI en el tiempo (no hay columna temporal disponible)")
    
    # 10. Obtener coeficientes
    print("== Obteniendo coeficientes ==")
    coefficients = get_coefficients_summary(
        models_dict=models_dict,
        feature_columns=feature_columns,
        logger=logger,
    )
    
    # Construir diccionario de resultados
    cluster_results = {
        'cluster_num': cluster_num,
        'cluster_name': cluster_name,
        'preprocessing_pipeline': preprocessing_pipeline,  # Pipeline completo guardado aquí
        'models_dict': models_dict,
        'feature_columns': feature_columns,
        'metrics_summary': summary,
        'gini_validation': gini_validation,
        'gini_by_period': gini_by_period,
        'gini_plot_figure': gini_plot_figure,  # Figura del gráfico de GINI
        'spread_probability_figure': spread_probability_figure,  # Figura de probabilidad vs spread
        'stability': stability,
        'coefficients': coefficients,  # Coeficientes (betas) del modelo
        'directionality_check': directionality_check,
        'directionality_report': directionality_report,
        'tasa_check': tasa_check_result,
        'cluster_variables': cluster_variables,
    }
    
    # Logging para confirmar que el pipeline se guarda
    logger.info(f"Pipeline guardado en resultados: {type(preprocessing_pipeline)}")
    logger.info(f"Modelo guardado en resultados: {type(models_dict)}")
    logger.info(f"Coeficientes guardados: {len(coefficients) if coefficients is not None and not coefficients.empty else 0} variables")
    
    logger.info(f"== FIN PROCESAMIENTO {cluster_name.upper()} ==")
    print(f"== {cluster_name} procesado exitosamente ==")
    
    return cluster_results


def main():
    """
    Función principal que procesa todos los clusters desde un archivo unificado.
    """
    # Configurar logger
    log_path = os.path.join(os.path.dirname(__file__), 'process_modeling_engine.log')
    logger = setup_logger(log_path)
    logger.info("== INICIO PROCESO DE MODELAMIENTO ==")
    
    # 1. Cargar archivo unificado
    print("== Cargando archivo unificado ==")
    logger.info(f"Cargando datos desde: {UNIFIED_DATA_PATH}")
    df_unified = read_dataset(UNIFIED_DATA_PATH, fmt='parquet', logger=logger)
    logger.info(f"Datos cargados. Shape: {df_unified.shape}")
    
    # Validar que existe la columna 'cluster'
    if 'cluster' not in df_unified.columns:
        logger.error("La columna 'cluster' no existe en el dataset. Abortando.")
        print("Error: La columna 'cluster' no existe en el dataset.")
        return
    
    # Validar que existe la columna objetivo
    if TARGET_COLUMN not in df_unified.columns:
        logger.error(f"La columna objetivo '{TARGET_COLUMN}' no existe en el dataset. Abortando.")
        print(f"Error: La columna objetivo '{TARGET_COLUMN}' no existe en el dataset.")
        return
    
    # 1.1. Cargar variables por cluster desde CSV
    cluster_variables_map = load_cluster_variables(
        csv_path=VARS_BY_CLUSTER_PATH,
        target_column=TARGET_COLUMN,
        logger=logger,
    )

    # Obtener clusters únicos
    clusters = sorted(df_unified['cluster'].unique().tolist())
    logger.info(f"Clusters encontrados: {clusters}")
    print(f"== Procesando {len(clusters)} clusters ==")
    
    # 2. Procesar cada cluster y acumular resultados
    all_cluster_results = {}
    
    for cluster_num in clusters:
        cluster_name = f'cluster_{cluster_num}'
        try:
            # Filtrar datos del cluster
            df_cluster = df_unified[df_unified['cluster'] == cluster_num].copy()
            logger.info(f"Registros en {cluster_name}: {len(df_cluster)}")
            
            if len(df_cluster) == 0:
                logger.warning(f"{cluster_name} está vacío. Se omite.")
                continue
            
            # Obtener variables específicas del cluster desde el CSV
            if cluster_num not in cluster_variables_map:
                logger.warning(f"No se encontraron variables definidas para cluster {cluster_num} en el CSV. Se omite.")
                continue
            
            cluster_variables = cluster_variables_map[cluster_num]
            
            # Procesar cluster
            cluster_results = process_cluster(
                cluster_num=cluster_num,
                df_cluster=df_cluster,
                cluster_variables=cluster_variables,
                logger=logger,
            )
            
            if cluster_results is not None:
                all_cluster_results[cluster_num] = cluster_results
                logger.info(f"{cluster_name} procesado exitosamente")
            else:
                logger.warning(f"{cluster_name} no produjo resultados válidos.")
                
        except Exception as e:
            logger.error(f"Error procesando {cluster_name}: {e}", exc_info=True)
            print(f"Error procesando {cluster_name}: {e}")
            continue
    
    # 3. Guardar archivos individuales por cluster (CSV, gráficos) - LOCALMENTE
    print("\n== Guardando archivos individuales por cluster (local) ==")
    
    # Crear carpeta local de modelling
    local_modelling_dir = os.path.join(os.path.dirname(__file__), 'modelling')
    os.makedirs(local_modelling_dir, exist_ok=True)
    logger.info(f"Directorio local de modelling: {local_modelling_dir}")
    
    for cluster_num, cluster_results in all_cluster_results.items():
        cluster_name = cluster_results['cluster_name']
        cluster_dir = os.path.join(local_modelling_dir, cluster_name)
        os.makedirs(cluster_dir, exist_ok=True)
        
        try:
            # Guardar coeficientes (betas) en CSV - LOCAL
            if cluster_results['coefficients'] is not None and not cluster_results['coefficients'].empty:
                coef_path = os.path.join(cluster_dir, 'coefficients_summary.csv')
                cluster_results['coefficients'].to_csv(coef_path, index=False)
                logger.info(f"Coeficientes de {cluster_name} guardados en: {coef_path}")
                print(f"  ✓ Coeficientes de {cluster_name} guardados: {coef_path}")
            
            # Guardar GINI por período en CSV - LOCAL
            if cluster_results['gini_by_period'] is not None and not cluster_results['gini_by_period'].empty:
                gini_period_path = os.path.join(cluster_dir, 'gini_by_period.csv')
                cluster_results['gini_by_period'].to_csv(gini_period_path, index=False)
                logger.info(f"GINI por período de {cluster_name} guardado en: {gini_period_path}")
                print(f"  ✓ GINI por período de {cluster_name} guardado: {gini_period_path}")
            
            # Guardar gráfico de GINI - LOCAL
            if cluster_results['gini_plot_figure'] is not None:
                gini_plot_path = os.path.join(cluster_dir, 'gini_over_time.png')
                cluster_results['gini_plot_figure'].savefig(gini_plot_path, dpi=150, bbox_inches='tight')
                plt.close(cluster_results['gini_plot_figure'])  # Cerrar figura para liberar memoria
                logger.info(f"Gráfico de GINI de {cluster_name} guardado en: {gini_plot_path}")
                print(f"  ✓ Gráfico de GINI de {cluster_name} guardado: {gini_plot_path}")
            
            # Guardar gráfico de probabilidad vs spread - LOCAL
            if cluster_results.get('spread_probability_figure') is not None:
                spread_prob_path = os.path.join(cluster_dir, 'spread_probability_curve.png')
                cluster_results['spread_probability_figure'].savefig(spread_prob_path, dpi=150, bbox_inches='tight')
                plt.close(cluster_results['spread_probability_figure'])  # Cerrar figura para liberar memoria
                logger.info(f"Gráfico de probabilidad vs spread de {cluster_name} guardado en: {spread_prob_path}")
                print(f"  ✓ Gráfico de probabilidad vs spread de {cluster_name} guardado: {spread_prob_path}")
            
            # Guardar reporte de direccionalidad - LOCAL
            if cluster_results['directionality_report'] is not None and not cluster_results['directionality_report'].empty:
                directionality_path = os.path.join(cluster_dir, 'directionality_report.csv')
                cluster_results['directionality_report'].to_csv(directionality_path, index=False)
                logger.info(f"Reporte de direccionalidad de {cluster_name} guardado en: {directionality_path}")
                print(f"  ✓ Reporte de direccionalidad de {cluster_name} guardado: {directionality_path}")
            
            # Guardar métricas en CSV - LOCAL
            if cluster_results['metrics_summary'] is not None and not cluster_results['metrics_summary'].empty:
                metrics_path = os.path.join(cluster_dir, 'metrics_summary.csv')
                cluster_results['metrics_summary'].to_csv(metrics_path, index=False)
                logger.info(f"Métricas de {cluster_name} guardadas en: {metrics_path}")
                print(f"  ✓ Métricas de {cluster_name} guardadas: {metrics_path}")
                
        except Exception as e:
            logger.error(f"Error guardando archivos individuales para {cluster_name}: {e}", exc_info=True)
            print(f"  ✗ Error guardando archivos de {cluster_name}: {e}")
    
    # 4. Guardar modelo unificado en un solo .pkl - LOCAL Y S3
    if all_cluster_results:
        print("\n== Guardando modelo unificado ==")
        logger.info(f"Guardando modelo unificado con {len(all_cluster_results)} clusters")
        
        # Crear diccionario unificado (sin las figuras de matplotlib que no se pueden serializar bien)
        unified_model = {
            'metadata': {
                'target_column': TARGET_COLUMN,
                'clusters': list(all_cluster_results.keys()),
                'creation_timestamp': pd.Timestamp.now().isoformat(),
            },
            'clusters': {},
        }
        
        # Copiar resultados sin las figuras (se guardan por separado)
        for cluster_num, cluster_results in all_cluster_results.items():
            cluster_data = cluster_results.copy()
            # Remover las figuras de matplotlib (no se serializan bien en pickle)
            cluster_data.pop('gini_plot_figure', None)
            cluster_data.pop('spread_probability_figure', None)
            unified_model['clusters'][cluster_num] = cluster_data
        
        # Guardar localmente
        local_unified_model_path = os.path.join(local_modelling_dir, 'unified_model.pkl')
        with open(local_unified_model_path, 'wb') as f:
            pickle.dump(unified_model, f)
        logger.info(f"Modelo unificado guardado localmente en: {local_unified_model_path}")
        print(f"  ✓ Modelo unificado guardado localmente: {local_unified_model_path}")
        
        # También guardar en S3
        s3_unified_model_path = os.path.join(OUTPUT_BASE_PATH, 'unified_model.pkl')
        save_model(unified_model, s3_unified_model_path, logger=logger)
        
        logger.info(f"Modelo unificado guardado en S3: {s3_unified_model_path}")
        logger.info("El modelo unificado incluye:")
        logger.info("  - preprocessing_pipeline: Pipeline completo de preprocesamiento")
        logger.info("  - models_dict: Modelo de regresión logística entrenado")
        logger.info("  - coefficients: Coeficientes (betas) del modelo")
        logger.info("  - metrics_summary: Métricas de evaluación")
        logger.info("  - gini_by_period: GINI por período")
        logger.info("  - directionality_report: Reporte de direccionalidad")
        logger.info("  - Y otros resultados del modelamiento")
        print(f"  ✓ Modelo unificado guardado en S3: {s3_unified_model_path}")
        print("  ✓ Pipeline de preprocesamiento incluido")
        print("  ✓ Modelo de regresión logística incluido")
        print("  ✓ Coeficientes (betas) incluidos")
        print("  ✓ Métricas y reportes incluidos")
    else:
        logger.warning("No se generaron resultados para ningún cluster.")
        print("Advertencia: No se generaron resultados para ningún cluster.")
    
    logger.info("== FIN PROCESO DE MODELAMIENTO ==")
    print("\n== Proceso completado. Ver log para detalles. ==")


if __name__ == "__main__":
    main()
