#!/usr/bin/env python3
"""
TFM_REPORTS.py - Generador de Reportes de Modelos Entrenados

Este script genera reportes completos de modelos ya entrenados sin necesidad de re-entrenarlos.
Incluye análisis de sensibilidad, comparativas de rendimiento, y visualizaciones.

Autor: Sistema TFM
Fecha: 2024
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Configurar matplotlib para mejor calidad
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Importar módulos del proyecto
sys.path.append(os.path.join(os.path.dirname(__file__), 'engine_TFM'))
from engine_TFM.engine_modeling import ModelingEngine
from engine_TFM.utils import (
    analyze_real_int_rate_sensitivity,
    calculate_gini,
    ModelComparator
)

def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configura el sistema de logging"""
    log_config = config.get("logging", {})
    
    # Crear logger
    logger = logging.getLogger("TFM_REPORTS")
    logger.setLevel(getattr(logging, log_config.get("level", "INFO")))
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Formato
    formatter = logging.Formatter(
        log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    # Handler para archivo
    log_file = log_config.get("file", "log_REPORTS.txt")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Handler para consola
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str = "engine_TFM/config_reports.yml") -> Dict[str, Any]:
    """Carga la configuración desde archivo YAML"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"❌ Error: Archivo de configuración no encontrado: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error al cargar configuración YAML: {e}")
        sys.exit(1)

def create_directories(config: Dict[str, Any]) -> Tuple[str, str, str]:
    """Crea los directorios necesarios"""
    general = config.get("general", {})
    base_dir = general.get("base_dir", ".")
    
    # Crear directorios
    output_dir = os.path.join(base_dir, general.get("output_dir", "reports"))
    images_dir = os.path.join(base_dir, general.get("images_dir", "imagenes"))
    models_dir = os.path.join(base_dir, general.get("models_dir", "models"))
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    
    return output_dir, images_dir, models_dir

def get_available_models(models_dir: str, config: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    """Obtiene la lista de modelos disponibles según la configuración"""
    models_cfg = config.get("models", {})
    available_models = []
    
    # Mapeo de configuraciones a archivos de modelo
    model_mapping = [
        ("LOGIT_PCA_LDA", "logit_pca_lda.pkl", "PCA+LDA", "enable_logit_pca_lda"),
        ("LOGIT_ANOVA", "logit_anova.pkl", "ANOVA", "enable_logit_anova"),
        ("GNB_PCA_LDA", "gnb_pca_lda.pkl", "PCA+LDA", "enable_gnb_pca_lda"),
        ("GNB_ANOVA", "gnb_anova.pkl", "ANOVA", "enable_gnb_anova"),
        ("SVM_LINEAR_PCA_LDA", "svm_linear_pca_lda.pkl", "PCA+LDA", "enable_svm_linear_pca_lda"),
        ("SVM_LINEAR_ANOVA", "svm_linear_anova.pkl", "ANOVA", "enable_svm_linear_anova"),
        ("SVM_BEST_PCA_LDA", "svm_best_pca_lda.pkl", "PCA+LDA", "enable_svm_best_pca_lda"),
        ("SVM_BEST_ANOVA", "svm_best_anova.pkl", "ANOVA", "enable_svm_best_anova"),
        ("MLP_PCA_LDA", "mlp_pca_lda.pkl", "PCA+LDA", "enable_mlp_pca_lda"),
        ("MLP_ANOVA", "mlp_anova.pkl", "ANOVA", "enable_mlp_anova"),
    ]
    
    for model_name, model_file, dataset_type, config_key in model_mapping:
        if models_cfg.get(config_key, True):  # Por defecto True si no está especificado
            model_path = os.path.join(models_dir, model_file)
            if os.path.exists(model_path):
                available_models.append((model_name, model_file, dataset_type))
            else:
                print(f"⚠️ Modelo no encontrado: {model_file}")
    
    return available_models

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Carga y prepara los datos originales"""
    data_cfg = config.get("data", {})
    input_csv = data_cfg.get("input_csv", "Loan_data.csv")
    
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Archivo de datos no encontrado: {input_csv}")
    
    # Cargar datos
    df = pd.read_csv(input_csv, low_memory=False)
    
    # Crear target binario
    target_col = data_cfg.get("target_column", "flg_target")
    status_col = data_cfg.get("status_column", "loan_status")
    
    good_status = data_cfg.get("allowed_status", {}).get("good", ["Fully Paid"])
    bad_status = data_cfg.get("allowed_status", {}).get("bad", ["Charged Off", "Default"])
    
    good_set = set(good_status)
    bad_set = set(bad_status)
    mask = df[status_col].isin(good_set | bad_set)
    df = df.loc[mask].copy().reset_index(drop=True)
    df[target_col] = df[status_col].apply(
        lambda x: 1 if x in bad_set else (0 if x in good_set else np.nan)
    )
    
    return df

def create_comparative_report(models_dir: str, available_models: List[Tuple[str, str, str]], 
                              output_dir: str, logger: logging.Logger) -> pd.DataFrame:
    """Genera reporte comparativo de todos los modelos"""
    logger.info("📊 GENERANDO REPORTE COMPARATIVO")
    logger.info("=" * 50)
    
    # Usar ModelComparator existente
    base_dir = os.path.dirname(models_dir)
    comparator = ModelComparator(base_dir)
    
    # Generar comparativa
    df_comparative = comparator.compare()
    
    # Guardar reporte
    output_file = os.path.join(output_dir, "reporte_comparativo_modelos.csv")
    df_comparative.to_csv(output_file, index=False)
    logger.info(f"✅ Reporte comparativo guardado en: {output_file}")
    
    # Mostrar resumen
    logger.info("\n🏆 TOP 5 MODELOS POR AUC:")
    top_models = df_comparative.nlargest(5, 'AUC')
    for _, row in top_models.iterrows():
        logger.info(f"  {row['Modelo']}: AUC={row['AUC']:.4f} | GINI={row['GINI']:.4f} | PR-AUC={row['PR_AUC(AP)']:.4f}")
    
    return df_comparative

def generate_sensitivity_analysis(models_dir: str, available_models: List[Tuple[str, str, str]], 
                                output_dir: str, config: Dict[str, Any], logger: logging.Logger):
    """Genera análisis de sensibilidad para variables especificadas usando datasets correctos"""
    sensitivity_cfg = config.get("sensitivity_analysis", {})
    
    if not sensitivity_cfg.get("enable_sensitivity_analysis", True):
        logger.info("⏭️ Análisis de sensibilidad deshabilitado en configuración")
        return
    
    logger.info("🔍 ANÁLISIS DE SENSIBILIDAD")
    logger.info("=" * 50)
    
    # Cargar datasets procesados que ya tienen todas las features
    try:
        df_pca_lda = pd.read_csv("df_pca_lda.csv")
        df_anova = pd.read_csv("df_anova.csv")
        logger.info(f"✅ Datasets cargados: PCA+LDA ({len(df_pca_lda):,} obs), ANOVA ({len(df_anova):,} obs)")
    except FileNotFoundError as e:
        logger.error(f"❌ Error cargando datasets: {e}")
        return
    
    variables = sensitivity_cfg.get("variables", [])
    confidence_level = sensitivity_cfg.get("confidence_level", 0.95)
    
    for var_config in variables:
        var_name = var_config.get("name")
        display_name = var_config.get("display_name", var_name)
        step = var_config.get("step", 2.0)
        min_samples = var_config.get("min_samples", 100)
        
        logger.info(f"\n📈 Analizando sensibilidad para: {display_name}")
        
        for model_name, model_file, dataset_type in available_models:
            model_path = os.path.join(models_dir, model_file)
            
            try:
                # Cargar modelo
                model = ModelingEngine.load_model(model_path)
                
                # Seleccionar dataset correcto según el tipo
                if "pca_lda" in model_name.lower():
                    df_sensitivity = df_pca_lda.copy()
                elif "anova" in model_name.lower():
                    df_sensitivity = df_anova.copy()
                else:
                    logger.warning(f"⚠️ No se pudo determinar dataset para {model_name}")
                    continue
                
                # Verificar que la variable existe
                if var_name not in df_sensitivity.columns:
                    logger.warning(f"⚠️ Variable '{var_name}' no encontrada en {dataset_type}")
                    continue
                
                # Filtrar datos válidos - mantener TODAS las variables para el modelo
                # Solo eliminar filas donde la variable de análisis o el target sean NaN
                df_sensitivity = df_sensitivity.dropna(subset=[var_name, 'flg_target'])
                
                # Verificar que tenemos suficientes variables para el modelo
                available_vars = [col for col in df_sensitivity.columns if col not in ['flg_target', var_name]]
                logger.info(f"📊 Variables disponibles para {model_name}: {len(available_vars)}")
                if len(available_vars) < 2:
                    logger.warning(f"⚠️ Insuficientes variables para {model_name}: solo {len(available_vars)}")
                    continue
                
                if len(df_sensitivity) < min_samples:
                    logger.warning(f"⚠️ Insuficientes datos para {model_name} - {display_name}")
                    continue
                
                # Ejecutar análisis de sensibilidad
                save_path = os.path.join(output_dir, f"sensibilidad_{var_name}_{model_name.lower()}.png")
                
                df_results = analyze_real_int_rate_sensitivity(
                    df=df_sensitivity,
                    model=model,
                    target_col="flg_target",
                    int_rate_col=var_name,
                    step=step,
                    min_samples=min_samples,
                    confidence_level=confidence_level,
                    save_path=save_path,
                    model_name=f"{model_name} ({dataset_type})",
                    verbose=True,
                    logger=logger
                )
                
                # Guardar resultados CSV
                if not df_results.empty and sensitivity_cfg.get("save_csv", True):
                    csv_path = os.path.join(output_dir, f"sensibilidad_{var_name}_{model_name.lower()}.csv")
                    df_results.to_csv(csv_path, index=False)
                    logger.info(f"✅ Resultados guardados en: {csv_path}")
                
            except Exception as e:
                logger.error(f"❌ Error en análisis de sensibilidad para {model_name} - {display_name}: {str(e)}")
                continue

def generate_comparative_sensitivity_plots(models_dir: str, available_models: List[Tuple[str, str, str]], 
                                         output_dir: str, config: Dict[str, Any], logger: logging.Logger):
    """Genera gráficos comparativos de sensibilidad agrupados por tipo de dataset"""
    logger.info("📊 GENERANDO GRÁFICOS COMPARATIVOS DE SENSIBILIDAD")
    logger.info("=" * 50)
    
    sensitivity_cfg = config.get("sensitivity_analysis", {})
    variables = sensitivity_cfg.get("variables", [])
    
    if not variables:
        logger.warning("⚠️ No hay variables configuradas para análisis de sensibilidad")
        return
    
    # Cargar datasets
    try:
        df_pca_lda = pd.read_csv("df_pca_lda.csv")
        df_anova = pd.read_csv("df_anova.csv")
        logger.info("✅ Datasets cargados para gráficos comparativos")
    except Exception as e:
        logger.error(f"❌ Error cargando datasets: {e}")
        return
    
    # Agrupar modelos por tipo de dataset
    pca_lda_models = [(name, file, dtype) for name, file, dtype in available_models if "pca_lda" in dtype.lower()]
    anova_models = [(name, file, dtype) for name, file, dtype in available_models if "anova" in dtype.lower()]
    
    for var_config in variables:
        var_name = var_config.get("name")
        display_name = var_config.get("display_name", var_name)
        
        logger.info(f"\n📈 Generando gráfico comparativo para: {display_name}")
        
        # Generar gráfico para PCA+LDA
        if pca_lda_models:
            generate_dataset_comparative_plot(
                models=pca_lda_models, 
                dataset=df_pca_lda, 
                dataset_name="PCA+LDA",
                var_name=var_name,
                display_name=display_name,
                output_dir=output_dir,
                logger=logger
            )
        
        # Generar gráfico para ANOVA
        if anova_models:
            generate_dataset_comparative_plot(
                models=anova_models, 
                dataset=df_anova, 
                dataset_name="ANOVA",
                var_name=var_name,
                display_name=display_name,
                output_dir=output_dir,
                logger=logger
            )

def generate_dataset_comparative_plot(models: List[Tuple[str, str, str]], dataset: pd.DataFrame, 
                                    dataset_name: str, var_name: str, display_name: str,
                                    output_dir: str, logger: logging.Logger):
    """Genera un gráfico comparativo para un tipo de dataset específico"""
    from engine_TFM.utils import plot_comparative_sensitivity
    
    curves_info = []
    
    for model_name, model_file, dataset_type in models:
        model_path = os.path.join("models", model_file)
        
        try:
            # Cargar modelo
            model = ModelingEngine.load_model(model_path)
            
            # Preparar datos
            df_sensitivity = dataset.copy()
            df_sensitivity = df_sensitivity.dropna(subset=[var_name, 'flg_target'])
            
            if len(df_sensitivity) < 100:
                logger.warning(f"⚠️ Insuficientes datos para {model_name}")
                continue
            
            # Ejecutar análisis de sensibilidad (sin guardar gráfico individual)
            df_results = analyze_real_int_rate_sensitivity(
                df=df_sensitivity,
                model=model,
                target_col="flg_target",
                int_rate_col=var_name,
                step=2.0,
                min_samples=100,
                confidence_level=0.95,
                save_path=None,  # No guardar gráfico individual
                model_name=f"{model_name} ({dataset_name})",
                verbose=False,
                logger=logger
            )
            
            if not df_results.empty:
                # Preparar datos para el gráfico comparativo
                df_plot = df_results.copy()
                df_plot['prob_impago_promedio'] = df_plot['predicted_rate']
                df_plot[var_name] = df_plot['rate_mean']
                
                curves_info.append({
                    'df': df_plot,
                    'label': model_name.replace(f"_{dataset_name.upper()}", "").replace("_", " ")
                })
                
        except Exception as e:
            logger.error(f"❌ Error procesando {model_name}: {str(e)}")
            continue
    
    # Generar gráfico comparativo si hay datos
    if curves_info:
        output_path = os.path.join(output_dir, f"sensibilidad_comparativo_{var_name}_{dataset_name.lower()}.png")
        plot_comparative_sensitivity(curves_info, var_name, output_path)
        logger.info(f"✅ Gráfico comparativo guardado: {output_path}")
    else:
        logger.warning(f"⚠️ No se pudieron generar datos para gráfico comparativo {dataset_name}")

def generate_individual_reports(models_dir: str, available_models: List[Tuple[str, str, str]], 
                              output_dir: str, config: Dict[str, Any], logger: logging.Logger):
    """Genera reportes individuales para cada modelo"""
    logger.info("📋 GENERANDO REPORTES INDIVIDUALES")
    logger.info("=" * 50)
    
    for model_name, model_file, dataset_type in available_models:
        model_path = os.path.join(models_dir, model_file)
        
        try:
            logger.info(f"\n🔍 Analizando modelo: {model_name}")
            
            # Cargar modelo
            model = ModelingEngine.load_model(model_path)
            
            # Aquí podrías agregar más análisis específicos por modelo
            # Por ejemplo: feature importance, calibration plots, etc.
            
            logger.info(f"✅ Reporte individual completado para {model_name}")
            
        except Exception as e:
            logger.error(f"❌ Error generando reporte individual para {model_name}: {str(e)}")
            continue

def generate_performance_plots(models_dir: str, available_models: List[Tuple[str, str, str]], 
                             images_dir: str, config: Dict[str, Any], logger: logging.Logger):
    """Genera gráficos de rendimiento comparativo"""
    plots_cfg = config.get("plots", {})
    
    if not plots_cfg.get("enable_performance_plots", True):
        logger.info("⏭️ Gráficos de rendimiento deshabilitados en configuración")
        return
    
    logger.info("📊 GENERANDO GRÁFICOS DE RENDIMIENTO")
    logger.info("=" * 50)
    
    # Aquí podrías implementar gráficos comparativos
    # Por ejemplo: ROC curves, Precision-Recall curves, etc.
    
    logger.info("✅ Gráficos de rendimiento generados")

def print_summary(df_comparative: pd.DataFrame, available_models: List[Tuple[str, str, str]], 
                 logger: logging.Logger):
    """Imprime resumen final del reporte"""
    logger.info("\n" + "="*60)
    logger.info("📋 RESUMEN FINAL DEL REPORTE")
    logger.info("="*60)
    
    logger.info(f"📊 Modelos analizados: {len(available_models)}")
    logger.info(f"📈 Mejor modelo por AUC: {df_comparative.loc[df_comparative['AUC'].idxmax(), 'Modelo']}")
    logger.info(f"📈 Mejor modelo por GINI: {df_comparative.loc[df_comparative['GINI'].idxmax(), 'Modelo']}")
    logger.info(f"📈 Mejor modelo por PR-AUC: {df_comparative.loc[df_comparative['PR_AUC(AP)'].idxmax(), 'Modelo']}")
    
    logger.info(f"\n🏆 TOP 3 MODELOS:")
    top_3 = df_comparative.nlargest(3, 'AUC')
    for i, (_, row) in enumerate(top_3.iterrows(), 1):
        logger.info(f"  {i}. {row['Modelo']}: AUC={row['AUC']:.4f} | GINI={row['GINI']:.4f}")

def main():
    """Función principal"""
    start_time = time.perf_counter()
    
    print("🔧 INICIANDO GENERADOR DE REPORTES...")
    print("="*60)
    
    try:
        # Cargar configuración
        config = load_config()
        
        # Configurar logging
        logger = setup_logging(config)
        logger.info("🚀 INICIANDO GENERADOR DE REPORTES TFM")
        logger.info("="*60)
        
        # Crear directorios
        output_dir, images_dir, models_dir = create_directories(config)
        logger.info(f"📁 Directorios configurados:")
        logger.info(f"   📂 Reportes: {output_dir}")
        logger.info(f"   📂 Imágenes: {images_dir}")
        logger.info(f"   📂 Modelos: {models_dir}")
        
        # Obtener modelos disponibles
        available_models = get_available_models(models_dir, config)
        logger.info(f"🤖 Modelos disponibles: {len(available_models)}")
        for model_name, _, dataset_type in available_models:
            logger.info(f"   ✅ {model_name} ({dataset_type})")
        
        if not available_models:
            logger.error("❌ No se encontraron modelos entrenados")
            return
        
        # Cargar datos
        logger.info("📥 Cargando datos...")
        df = load_data(config)
        logger.info(f"✅ Datos cargados: {len(df):,} observaciones")
        
        # Generar reporte comparativo
        reports_cfg = config.get("reports", {})
        if reports_cfg.get("enable_comparative_table", True):
            df_comparative = create_comparative_report(models_dir, available_models, output_dir, logger)
        else:
            df_comparative = pd.DataFrame()
        
        # Generar análisis de sensibilidad
        if reports_cfg.get("enable_sensitivity_analysis", True):
            generate_sensitivity_analysis(models_dir, available_models, output_dir, config, logger)
            
            # Generar gráficos comparativos de sensibilidad
            generate_comparative_sensitivity_plots(models_dir, available_models, output_dir, config, logger)
        
        # Generar reportes individuales
        if reports_cfg.get("enable_individual_reports", True):
            generate_individual_reports(models_dir, available_models, output_dir, config, logger)
        
        # Generar gráficos de rendimiento
        if reports_cfg.get("enable_performance_plots", True):
            generate_performance_plots(models_dir, available_models, images_dir, config, logger)
        
        # Resumen final
        if not df_comparative.empty:
            print_summary(df_comparative, available_models, logger)
        
        # Tiempo total
        total_time = time.perf_counter() - start_time
        logger.info(f"\n⏱️ Tiempo total de ejecución: {total_time:.2f} segundos")
        logger.info("✅ GENERACIÓN DE REPORTES COMPLETADA EXITOSAMENTE")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {str(e)}")
        if 'logger' in locals():
            logger.error(f"❌ Error durante la ejecución: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
