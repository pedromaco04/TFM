"""
Script para asignar clusters y propensiones a base de clientes.

Este script:
1. Carga la base de clientes
2. Asigna clusters usando el pipeline de segmentación
3. Asigna propensiones usando el modelo unificado
4. Guarda los resultados
"""

import os
import logging
from datetime import datetime

import pandas as pd

from feature_pro import read_dataset
from scoring import assign_clusters, assign_propensities, load_unified_model
from optimization import run_price_optimization, optimize_pilot_prices
from optimization.visualization import plot_efficient_frontier_from_dataframe
import numpy as np


def setup_logger(log_file: str = "optimization/optimization_process.log") -> logging.Logger:
    """Configura el logger para el proceso de asignación de clusters."""
    logger = logging.getLogger("cluster_assignment")
    logger.setLevel(logging.INFO)
    
    # Limpiar handlers existentes
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    
    # Formato de logs
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para archivo
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    
    return logger


def main():
    """Función principal para ejecutar la asignación de clusters."""
    
    # ===== CONFIGURACIÓN =====
    
    # Path del pipeline de segmentación
    SEGMENTATION_PIPELINE_PATH = "MOMA/segmentation_pipeline.pkl"
    
    # Path del modelo unificado
    UNIFIED_MODEL_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/03.Modelling/unified_model.pkl"
    
    # Path de datos de entrada
    CUSTOMER_BASE_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/01.Universo/universo_bot_cotiza_vars_comport_v4_backup"
    
    # Paths de salida en S3
    OUTPUT_CLUSTERED_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/04.Scoring/universo_with_cluster.parquet"
    OUTPUT_FINAL_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/04.Scoring/universo_with_cluster_and_propensity.parquet"
    OUTPUT_OPTIMIZED_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/04.Scoring/universo_with_optimal_spread.parquet"
    
    # Path para guardar resultados del paso 1 (optimización de escenarios) - S3
    OPTIMIZATION_RESULTS_PATH = "s3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/05.Optimization/escenarios_x_cliente.parquet"
    
    # Parámetros de optimización
    DI_VALUE = 8.0  # Costo de fondos (DI) en porcentaje
    SPREAD_RANGE = {
        "spread_min": 0.0,   # 0%
        "spread_max": 7.0,   # 7%
        "step_size": 0.1     # 0.1%
    }
    OPTIMAL_ALPHA = 0.7  # 70% peso en profit, 30% en volumen
    MAX_SPREAD_CHANGE_PCT = 0.3  # 30% máximo cambio de spread permitido
    
    # Parámetros para frontera eficiente
    ALPHA_VALUES = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])  # Valores de alpha a probar
    EFFICIENT_FRONTIER_PATH = "optimization/efficient_frontier.png"
    
    # ==========================
    
    # Configurar logger
    logger = setup_logger()
    logger.info("=" * 80)
    logger.info("INICIO PROCESO DE SCORING (CLUSTERS Y PROPENSIONES)")
    logger.info("=" * 80)
    logger.info(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Paso 1: Cargar base de clientes
        logger.info("\n" + "=" * 80)
        logger.info("PASO 1: Cargando base de clientes")
        logger.info("=" * 80)
        logger.info(f"Ruta: {CUSTOMER_BASE_PATH}")
        
        df_customers = read_dataset(CUSTOMER_BASE_PATH, fmt="parquet", logger=logger)
        logger.info(f"Base cargada exitosamente. Shape: {df_customers.shape}")
        logger.info(f"Total de clientes: {len(df_customers):,}")
        logger.info(f"Columnas disponibles: {list(df_customers.columns)}")
        
        # Paso 2: Asignar clusters
        logger.info("\n" + "=" * 80)
        logger.info("PASO 2: Asignando clusters")
        logger.info("=" * 80)
        logger.info(f"Pipeline de segmentación: {SEGMENTATION_PIPELINE_PATH}")
        
        df_clustered = assign_clusters(
            df=df_customers,
            segmentation_pipeline_path=SEGMENTATION_PIPELINE_PATH,
            cluster_column="cluster",
        )
        
        logger.info(f"Clusters asignados exitosamente")
        logger.info(f"Distribución de clusters:")
        cluster_counts = df_clustered["cluster"].value_counts().sort_index()
        for cluster, count in cluster_counts.items():
            logger.info(f"  Cluster {cluster}: {count:,} clientes ({count/len(df_clustered)*100:.2f}%)")
        
        # Paso 3: Guardar resultados en S3
        logger.info("\n" + "=" * 80)
        logger.info("PASO 3: Guardando resultados en S3")
        logger.info("=" * 80)
        logger.info(f"Ruta S3: {OUTPUT_CLUSTERED_PATH}")
        
        try:
            # Intentar guardar directamente en S3 (pandas/pyarrow puede manejar rutas S3)
            df_clustered.to_parquet(OUTPUT_CLUSTERED_PATH, index=False, engine='pyarrow')
            logger.info(f"Base con clusters guardada exitosamente en S3: {OUTPUT_CLUSTERED_PATH}")
        except Exception as e:
            # Si falla, intentar con boto3
            logger.warning(f"Error guardando directamente en S3: {e}")
            logger.info("Intentando guardar usando boto3...")
            
            try:
                import boto3
                from urllib.parse import urlparse
                import io
                
                # Parsear URL de S3
                parsed = urlparse(OUTPUT_CLUSTERED_PATH)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                # Guardar parquet en buffer
                buffer = io.BytesIO()
                df_clustered.to_parquet(buffer, index=False, engine='pyarrow')
                buffer.seek(0)
                
                # Subir a S3
                s3_client = boto3.client('s3')
                s3_client.upload_fileobj(buffer, bucket, key)
                
                logger.info(f"Base con clusters guardada en S3 usando boto3: {OUTPUT_CLUSTERED_PATH}")
            except Exception as e2:
                logger.error(f"Error guardando en S3 con boto3: {e2}")
                # Fallback: guardar localmente
                local_path = "scoring/output/clientes_con_clusters.parquet"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                df_clustered.to_parquet(local_path, index=False, engine='pyarrow')
                logger.warning(f"Guardado localmente en: {local_path}")
                logger.warning(f"   Ruta S3 original: {OUTPUT_CLUSTERED_PATH}")
        
        logger.info(f"Total de clientes procesados: {len(df_clustered):,}")
        
        # Paso 4: Asignar propensiones
        logger.info("\n" + "=" * 80)
        logger.info("PASO 4: Asignando propensiones")
        logger.info("=" * 80)
        logger.info(f"Modelo unificado: {UNIFIED_MODEL_PATH}")
        
        df_final = assign_propensities(
            df=df_clustered,
            unified_model_path=UNIFIED_MODEL_PATH,
            cluster_column="cluster",
            propensity_column="propensity",
            logger=logger,
        )
        
        logger.info(f"Propensiones asignadas exitosamente")
        logger.info(f"Estadísticas de propensiones:")
        logger.info(f"  Promedio: {df_final['propensity'].mean():.4f}")
        logger.info(f"  Mínimo: {df_final['propensity'].min():.4f}")
        logger.info(f"  Máximo: {df_final['propensity'].max():.4f}")
        logger.info(f"  Mediana: {df_final['propensity'].median():.4f}")
        
        # Paso 5: Guardar resultado final en S3
        logger.info("\n" + "=" * 80)
        logger.info("PASO 5: Guardando resultado final en S3")
        logger.info("=" * 80)
        logger.info(f"Ruta S3: {OUTPUT_FINAL_PATH}")
        
        try:
            # Intentar guardar directamente en S3 (pandas/pyarrow puede manejar rutas S3)
            df_final.to_parquet(OUTPUT_FINAL_PATH, index=False, engine='pyarrow')
            logger.info(f"Base con clusters y propensiones guardada exitosamente en S3: {OUTPUT_FINAL_PATH}")
        except Exception as e:
            # Si falla, intentar con boto3
            logger.warning(f"Error guardando directamente en S3: {e}")
            logger.info("Intentando guardar usando boto3...")
            
            try:
                import boto3
                from urllib.parse import urlparse
                import io
                
                # Parsear URL de S3
                parsed = urlparse(OUTPUT_FINAL_PATH)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                # Guardar parquet en buffer
                buffer = io.BytesIO()
                df_final.to_parquet(buffer, index=False, engine='pyarrow')
                buffer.seek(0)
                
                # Subir a S3
                s3_client = boto3.client('s3')
                s3_client.upload_fileobj(buffer, bucket, key)
                
                logger.info(f"Base con clusters y propensiones guardada en S3 usando boto3: {OUTPUT_FINAL_PATH}")
            except Exception as e2:
                logger.error(f"Error guardando en S3 con boto3: {e2}")
                # Fallback: guardar localmente
                local_path = "scoring/output/universo_with_cluster_and_propensity.parquet"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                df_final.to_parquet(local_path, index=False, engine='pyarrow')
                logger.warning(f"Guardado localmente en: {local_path}")
                logger.warning(f"   Ruta S3 original: {OUTPUT_FINAL_PATH}")
        
        # Paso 6: Optimización de SPREAD
        logger.info("\n" + "=" * 80)
        logger.info("PASO 6: Optimización de SPREAD")
        logger.info("=" * 80)
        logger.info(f"Cargando modelo unificado: {UNIFIED_MODEL_PATH}")
        
        unified_model = load_unified_model(UNIFIED_MODEL_PATH, logger=logger)
        
        # Generar valores de spread a evaluar
        spread_values = np.arange(
            SPREAD_RANGE['spread_min'],
            SPREAD_RANGE['spread_max'] + SPREAD_RANGE['step_size'],
            SPREAD_RANGE['step_size']
        )
        
        logger.info("Ejecutando optimización de SPREAD (PASO 1: Cálculo de propensiones por escenario)...")
        final_results = run_price_optimization(
            unified_model=unified_model,
            df_val=df_final,
            price_variable_name="spread",
            price_variable_values=spread_values,
            segmentation_pipeline_path=SEGMENTATION_PIPELINE_PATH,
            amount_column="importe",
            unified_model_path=UNIFIED_MODEL_PATH,
            n_jobs=1,  # Usar 1 para evitar problemas de memoria, ajustar según necesidad
            save_results_path=OPTIMIZATION_RESULTS_PATH,
            compute_efficient_frontier=True,
            alpha_values=ALPHA_VALUES,
            objective_type="lineal",
            logger=logger,
        )
        
        # Extraer resultados del paso 1
        optimization_results = final_results['optimization_results']
        
        logger.info("Optimizando SPREAD por cliente...")
        df_optimized = optimize_pilot_prices(
            optimization_results=optimization_results,
            df_pilot=df_final,
            price_variable_name="spread",
            optimal_alpha=OPTIMAL_ALPHA,
            max_price_change_pct=MAX_SPREAD_CHANGE_PCT,
            logger=logger,
        )
        
        logger.info("Optimización completada exitosamente")
        logger.info(f"Estadísticas de SPREAD óptimo:")
        logger.info(f"  Promedio: {df_optimized['optimal_spread'].mean():.2f}%")
        logger.info(f"  Mínimo: {df_optimized['optimal_spread'].min():.2f}%")
        logger.info(f"  Máximo: {df_optimized['optimal_spread'].max():.2f}%")
        logger.info(f"  Mediana: {df_optimized['optimal_spread'].median():.2f}%")
        logger.info(f"Profit total óptimo: {df_optimized['optimal_profit'].sum():,.0f}")
        logger.info(f"Volumen total óptimo: {df_optimized['optimal_volume'].sum():,.0f}")
        
        # Paso 7: Generar gráfica de frontera eficiente
        logger.info("\n" + "=" * 80)
        logger.info("PASO 7: Generando gráfica de frontera eficiente")
        logger.info("=" * 80)
        
        # Obtener datos de frontera eficiente del paso 2
        if 'efficient_frontier' in final_results:
            df_frontier = final_results['efficient_frontier']
            logger.info(f"Frontera eficiente calculada con {len(df_frontier)} puntos")
            logger.info(f"Datos de frontera eficiente:")
            logger.info(df_frontier.to_string(index=False))
            
            # Usar función generalizable de visualization.py
            plot_efficient_frontier_from_dataframe(
                df_frontier=df_frontier,
                price_variable_name="spread",
                save_path=EFFICIENT_FRONTIER_PATH,
                logger=logger,
            )
        else:
            logger.warning("No se calcularon datos de frontera eficiente. Omitiendo gráfica.")
        
        # Paso 8: Guardar resultado optimizado en S3
        logger.info("\n" + "=" * 80)
        logger.info("PASO 8: Guardando resultado optimizado en S3")
        logger.info("=" * 80)
        logger.info(f"Ruta S3: {OUTPUT_OPTIMIZED_PATH}")
        
        try:
            # Intentar guardar directamente en S3
            df_optimized.to_parquet(OUTPUT_OPTIMIZED_PATH, index=False, engine='pyarrow')
            logger.info(f"Base con SPREAD óptimo guardada exitosamente en S3: {OUTPUT_OPTIMIZED_PATH}")
        except Exception as e:
            # Si falla, intentar con boto3
            logger.warning(f"Error guardando directamente en S3: {e}")
            logger.info("Intentando guardar usando boto3...")
            
            try:
                import boto3
                from urllib.parse import urlparse
                import io
                
                # Parsear URL de S3
                parsed = urlparse(OUTPUT_OPTIMIZED_PATH)
                bucket = parsed.netloc
                key = parsed.path.lstrip('/')
                
                # Guardar parquet en buffer
                buffer = io.BytesIO()
                df_optimized.to_parquet(buffer, index=False, engine='pyarrow')
                buffer.seek(0)
                
                # Subir a S3
                s3_client = boto3.client('s3')
                s3_client.upload_fileobj(buffer, bucket, key)
                
                logger.info(f"Base con SPREAD óptimo guardada en S3 usando boto3: {OUTPUT_OPTIMIZED_PATH}")
            except Exception as e2:
                logger.error(f"Error guardando en S3 con boto3: {e2}")
                # Fallback: guardar localmente
                local_path = "optimization/output/universo_with_optimal_spread.parquet"
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                df_optimized.to_parquet(local_path, index=False, engine='pyarrow')
                logger.warning(f"Guardado localmente en: {local_path}")
                logger.warning(f"   Ruta S3 original: {OUTPUT_OPTIMIZED_PATH}")
        
        # Resumen final
        logger.info("\n" + "=" * 80)
        logger.info("RESUMEN FINAL")
        logger.info("=" * 80)
        logger.info(f"Clientes procesados: {len(df_optimized):,}")
        logger.info(f"Clusters asignados: {df_optimized['cluster'].nunique()}")
        logger.info(f"Propensiones asignadas: {df_optimized['propensity'].notna().sum():,}")
        logger.info(f"SPREAD óptimo asignado: {df_optimized['optimal_spread'].notna().sum():,}")
        logger.info(f"Archivo intermedio (solo clusters): {OUTPUT_CLUSTERED_PATH}")
        logger.info(f"Archivo intermedio (clusters + propensiones): {OUTPUT_FINAL_PATH}")
        logger.info(f"Archivo final (clusters + propensiones + SPREAD óptimo): {OUTPUT_OPTIMIZED_PATH}")
        logger.info(f"Gráfica de frontera eficiente: {EFFICIENT_FRONTIER_PATH}")
        logger.info("=" * 80)
        logger.info("PROCESO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error en el proceso de scoring: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
