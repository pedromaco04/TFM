"""
TFM_MODELING.py

Script equivalente al notebook TFM/TFM_MODELING.ipynb con salidas organizadas por secciones.
Incluye múltiples prints para rastrear el progreso y entender resultados en consola.

Ejecución:
    python TFM_MODELING.py

Requisitos:
    - Archivos CSV: df_pca_lda.csv y df_anova.csv ubicados en el directorio TFM
    - Módulo: engine_TFM.engine_modeling (ModelingEngine)
"""

import os
import sys
import time
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from engine_TFM.engine_modeling import ModelingEngine
from engine_TFM.utils import (
    load_config,
    section,
    prepare_model_dataframe,
    stratified_sample_xy,
    evaluate_at_threshold,
    print_metrics_block,
    plot_comparative_curves,
    plot_comparative_sensitivity,
    sensitivity_int_rate,
    ModelComparator,
    WOETransformer,
)


def print_section_progress(current, total, section_name='', prefix='', suffix='', length=50):
    """
    Función para mostrar barra de progreso con título de sección en consola.
    """
    percent = float(current) / float(total)
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)

    if section_name:
        section_title = f"{section_name}: "
        sys.stdout.write(f'\r{section_title}|{bar}| {percent:.1%} {suffix}')
    else:
        sys.stdout.write(f'\r{prefix} |{bar}| {percent:.1%} {suffix}')
    sys.stdout.flush()
    if current == total:
        print()  # Nueva línea al final


def setup_logging(config_path: str = 'engine_TFM/config_modeling.yml') -> logging.Logger:
    """
    Configura el sistema de logging para el pipeline de modelado.
    """
    config = load_config(config_path)
    logging_cfg = config.get('logging', {})

    if not logging_cfg.get('enable_file_logging', True):
        return None

    log_filename = logging_cfg.get('log_filename', 'log_MODELING.txt')

    # Crear logger
    logger = logging.getLogger('ModelingPipeline')
    logger.setLevel(logging.INFO)

    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Handler para archivo
    log_file = os.path.join(os.getcwd(), log_filename)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Log inicial
    logger.info("=" * 80)
    logger.info("MODELING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)

    return logger


def log_message(logger: logging.Logger, msg: str, level: str = 'info') -> None:
    """
    Función unificada para logging que escribe tanto en archivo como en consola si es necesario.
    """
    if logger:
        if level == 'info':
            logger.info(msg)
        elif level == 'warning':
            logger.warning(msg)
        elif level == 'error':
            logger.error(msg)
        elif level == 'debug':
            logger.debug(msg)


def preparar_df_modelo_sin_fillna(df: pd.DataFrame, num_vars: List[str], target: str = 'flg_target') -> pd.DataFrame:
    # Deprecated in favor of utils.prepare_model_dataframe; kept for backward compatibility where used below
    return prepare_model_dataframe(df, num_vars, target)


def evaluar_a_umbral(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Dict[str, Any]:
    # Delegate to utils for consistency
    return evaluate_at_threshold(y_true, y_score, thr)


# keep local alias to preserve current calls
_print_metrics_block = print_metrics_block
def print_metrics_block(nombre: str, metrics: Dict[str, Any], logger=None) -> None:
    _print_metrics_block(nombre, metrics, logger)


def _stratified_sample_Xy(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if (max_rows is None) or (len(X) <= max_rows):
        return X
    frac_eff = max_rows / float(len(X))
    df_aux = X.copy()
    df_aux['_y_'] = y.values
    sampled = df_aux.groupby('_y_', group_keys=False).apply(
        lambda g: g.sample(frac=frac_eff, random_state=random_state),
        include_groups=False
    ).drop(columns=['_y_'])
    return sampled


def sensibilidad_int_rate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str, images_dir: str, max_rows: int = 50000) -> None:
    # Delegate to utils.sensitivity_int_rate ensuring same signature
    return sensitivity_int_rate(model, X_test, y_test, label, images_dir, max_rows=max_rows)


def plot_sensibilidad_comparativa(curvas_info: list, var_name: str, out_path: str) -> None:
    return plot_comparative_sensitivity(curvas_info, var_name, out_path)


def plot_curvas_comparativas(models_info: list, out_path: str) -> None:
    return plot_comparative_curves(models_info, out_path)


def setup_logging(config_path: str = 'engine_TFM/config_modeling.yml') -> logging.Logger:
    """
    Configura el sistema de logging para el pipeline de modelado.
    """
    config = load_config(config_path)
    logging_cfg = config.get('logging', {})

    if not logging_cfg.get('enable_file_logging', True):
        return None

    log_filename = logging_cfg.get('log_filename', 'log_MODELING.txt')

    # Crear logger
    logger = logging.getLogger('ModelingPipeline')
    logger.setLevel(logging.INFO)

    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Handler para archivo
    log_file = os.path.join(os.getcwd(), log_filename)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


def log_message(logger: logging.Logger, msg: str, level: str = 'info') -> None:
    """
    Función unificada para logging que escribe en archivo.
    """
    if logger:
        if level == 'info':
            logger.info(msg)
        elif level == 'warning':
            logger.warning(msg)
        elif level == 'error':
            logger.error(msg)
        elif level == 'debug':
            logger.debug(msg)


def main() -> None:
    """
    Función principal del pipeline de modelado con logging y configuración desde YML.
    """
    # Cargar configuración y logging
    config_path = 'engine_TFM/config_modeling.yml'
    config = load_config(config_path)
    logger = setup_logging(config_path)
    
    log_message(logger, "=" * 80)
    log_message(logger, "INICIANDO PIPELINE DE MODELADO")
    log_message(logger, "=" * 80)
    log_message(logger, f"Archivo de configuración: {config_path}")

    # Mostrar configuración de modelos
    models_cfg = config.get('models', {})
    log_message(logger, "Configuración de modelos a ejecutar:")
    for model_name, enabled in models_cfg.items():
        status = "✅ HABILITADO" if enabled else "❌ DESHABILITADO"
        log_message(logger, f"  • {model_name}: {status}")

    # Directorios
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pca_lda = os.path.join(base_dir, 'df_pca_lda.csv')
    csv_anova = os.path.join(base_dir, 'df_anova.csv')
    models_dir = os.path.join(base_dir, 'models')
    images_dir = os.path.join(base_dir, 'imagenes')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Barra de progreso inicial
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(0, 5, section_name='🔧 INICIANDO MODELING PIPELINE...')

    # ===== PASO 1: CARGA DE DATOS =====
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(1, 5, section_name='📥 Cargando datos')
    else:
        section('1) CARGA DE DATOS')

    t0_total = time.perf_counter()
    t0 = time.perf_counter()
    
    log_message(logger, f"Leyendo archivos de datos:")
    log_message(logger, f"  • PCA+LDA: {csv_pca_lda}")
    log_message(logger, f"  • ANOVA: {csv_anova}")
    log_message(logger, f"Leyendo: {csv_pca_lda}")
    
    df_pca_lda = pd.read_csv(csv_pca_lda)
    log_message(logger, f"Leyendo: {csv_anova}")
    df_anova = pd.read_csv(csv_anova)
    
    log_message(logger, f"Datos cargados correctamente:")
    log_message(logger, f"  • PCA+LDA: {df_pca_lda.shape[0]:,} filas × {df_pca_lda.shape[1]} columnas")
    log_message(logger, f"  • ANOVA: {df_anova.shape[0]:,} filas × {df_anova.shape[1]} columnas")
    log_message(logger, f"Tamaños -> PCA+LDA: {df_pca_lda.shape} | ANOVA: {df_anova.shape}")
    log_message(logger, f"[TIMER] Carga de datos: {time.perf_counter()-t0:.2f}s")

    # Variables (se asume última columna es target)
    all_vars_pca_lda = df_pca_lda.columns.tolist()[:-1]
    all_vars_anova = df_anova.columns.tolist()[:-1]

    # Identificar variables categóricas vs numéricas
    cat_cols_pca_lda = [col for col in all_vars_pca_lda if df_pca_lda[col].dtype == 'object']
    cat_cols_anova = [col for col in all_vars_anova if df_anova[col].dtype == 'object']
    num_vars_pca_lda = [col for col in all_vars_pca_lda if col not in cat_cols_pca_lda]
    num_vars_anova = [col for col in all_vars_anova if col not in cat_cols_anova]

    log_message(logger, f"Análisis de variables:")
    log_message(logger, f"  • PCA+LDA -> Numéricas: {len(num_vars_pca_lda)} | Categóricas: {len(cat_cols_pca_lda)}")
    log_message(logger, f"  • ANOVA -> Numéricas: {len(num_vars_anova)} | Categóricas: {len(cat_cols_anova)}")
    log_message(logger, f"Variables PCA+LDA -> Numéricas: {len(num_vars_pca_lda)} | Categóricas: {len(cat_cols_pca_lda)}")
    log_message(logger, f"Variables ANOVA -> Numéricas: {len(num_vars_anova)} | Categóricas: {len(cat_cols_anova)}")
    
    if cat_cols_pca_lda:
        log_message(logger, f"  • Variables categóricas PCA+LDA: {cat_cols_pca_lda}")
        log_message(logger, f"Categóricas PCA+LDA: {cat_cols_pca_lda}")
    if cat_cols_anova:
        log_message(logger, f"  • Variables categóricas ANOVA: {cat_cols_anova}")
        log_message(logger, f"Categóricas ANOVA: {cat_cols_anova}")

    # ===== PASO 2: PREPARACIÓN CON WOE =====
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(2, 5, section_name='🔢 Preparando WOE')
    else:
        section('2) PREPARACIÓN DE DATAFRAMES CON WOE')

    t0 = time.perf_counter()
    log_message(logger, "Iniciando preparación de dataframes con transformación WOE...")

    # Preparar dataframes base (todas las variables)
    df_model_pca_lda = preparar_df_modelo_sin_fillna(df_pca_lda, all_vars_pca_lda)
    df_model_anova = preparar_df_modelo_sin_fillna(df_anova, all_vars_anova)

    # Variables finales: numéricas + WOE de categóricas (SOLO LAS CATEGÓRICAS SE TRANSFORMAN)
    final_vars_pca_lda = num_vars_pca_lda.copy()
    final_vars_anova = num_vars_anova.copy()

    # Aplicar WOE SOLO a variables categóricas
    woe_mappings_dir = os.path.join(base_dir, 'woe_mappings')
    os.makedirs(woe_mappings_dir, exist_ok=True)

    if cat_cols_pca_lda:
        log_message(logger, f"Aplicando transformación WOE a {len(cat_cols_pca_lda)} variables categóricas PCA+LDA...")
        # Solo log; sin print en consola
        
        woe_pca_lda = WOETransformer(target_col='flg_target')
        df_model_pca_lda = woe_pca_lda.fit_transform(df_model_pca_lda, cat_cols_pca_lda)
        
        # Agregar las nuevas columnas WOE a las variables finales
        woe_cols_pca_lda = [f"{col}_woe" for col in cat_cols_pca_lda]
        final_vars_pca_lda.extend(woe_cols_pca_lda)
        
        # Guardar mapeos WOE
        woe_pca_lda.final_vars = final_vars_pca_lda
        woe_mapping_file = os.path.join(woe_mappings_dir, 'woe_mappings_pca_lda.json')
        woe_pca_lda.save_mappings(woe_mapping_file)

        # Guardar dataframe con WOE para ModelComparator
        df_pca_lda_woe = df_model_pca_lda.copy()
        df_pca_lda_woe_path = os.path.join(base_dir, 'df_pca_lda_woe.csv')
        df_pca_lda_woe.to_csv(df_pca_lda_woe_path, index=False)
        log_message(logger, f"[SAVE] DataFrame PCA+LDA con WOE guardado: {df_pca_lda_woe_path}")
        
        # Mostrar estadísticas WOE
        woe_stats = woe_pca_lda.get_summary_stats()
        log_message(logger, f"Estadísticas WOE PCA+LDA:")
        for _, row in woe_stats.iterrows():
            log_message(logger, f"  • {row['variable']} | {row['categoria']}: WOE={row['woe']:.3f} (N={row['total_count']:,})")
        # Detalle solo en log

    if cat_cols_anova:
        log_message(logger, f"Aplicando transformación WOE a {len(cat_cols_anova)} variables categóricas ANOVA...")
        # Solo log; sin print en consola
        
        woe_anova = WOETransformer(target_col='flg_target')
        df_model_anova = woe_anova.fit_transform(df_model_anova, cat_cols_anova)
        
        # Agregar las nuevas columnas WOE a las variables finales
        woe_cols_anova = [f"{col}_woe" for col in cat_cols_anova]
        final_vars_anova.extend(woe_cols_anova)
        
        # Guardar mapeos WOE
        woe_anova.final_vars = final_vars_anova
        woe_mapping_file = os.path.join(woe_mappings_dir, 'woe_mappings_anova.json')
        woe_anova.save_mappings(woe_mapping_file)

        # Guardar dataframe con WOE para ModelComparator
        df_anova_woe = df_model_anova.copy()
        df_anova_woe_path = os.path.join(base_dir, 'df_anova_woe.csv')
        df_anova_woe.to_csv(df_anova_woe_path, index=False)
        log_message(logger, f"[SAVE] DataFrame ANOVA con WOE guardado: {df_anova_woe_path}")
        
        # Mostrar estadísticas WOE
        woe_stats = woe_anova.get_summary_stats()
        log_message(logger, f"Estadísticas WOE ANOVA:")
        for _, row in woe_stats.iterrows():
            log_message(logger, f"  • {row['variable']} | {row['categoria']}: WOE={row['woe']:.3f} (N={row['total_count']:,})")
        # Detalle solo en log

    log_message(logger, f"Preparación completada:")
    log_message(logger, f"  • Variables finales PCA+LDA: {len(final_vars_pca_lda)} ({len(num_vars_pca_lda)} numéricas + {len(cat_cols_pca_lda)} WOE)")
    log_message(logger, f"  • Variables finales ANOVA: {len(final_vars_anova)} ({len(num_vars_anova)} numéricas + {len(cat_cols_anova)} WOE)")
    log_message(logger, f"  • DataFrames: PCA+LDA {df_model_pca_lda.shape} | ANOVA {df_model_anova.shape}")
    log_message(logger, f"Variables finales PCA+LDA: {len(final_vars_pca_lda)} | ANOVA: {len(final_vars_anova)}")
    log_message(logger, f"[TIMER] Preparación con WOE: {time.perf_counter()-t0:.2f}s")

    # ===== PASO 3: REGRESIÓN LOGÍSTICA =====
    if models_cfg.get('enable_logit', True):
        if config.get('logging', {}).get('enable_progress_bar', True):
            print_section_progress(3, 5, section_name='🤖 Aplicando modelos')

        log_message(logger, "Iniciando entrenamiento de modelos LOGIT...")
        
        # LOGIT PCA+LDA
        log_message(logger, "Entrenando LOGIT con PCA+LDA...")
        # Solo log; sin print en consola
    t0 = time.perf_counter()
    model_pca_lda, metrics_pca_lda, y_pred_pca_lda, y_proba_pca_lda, \
    X_train_pca_lda, X_test_pca_lda, y_train_pca_lda, y_test_pca_lda = \
        ModelingEngine.fit_logit_predict(
            df=df_model_pca_lda,
                    num_cols=final_vars_pca_lda,
                    cat_cols=[],  # WOE ya aplicado, no hay categóricas originales
            target='flg_target',
            model_params={'solver': 'lbfgs'},
            show_confusion=False,
            verbose=True,
            save_confusion_path=os.path.join(images_dir, 'logit_pca_lda_confusion.png')
        )
        log_message(logger, f"LOGIT PCA+LDA completado:")
        log_message(logger, f"  • AUC: {metrics_pca_lda['AUC_test']:.4f}")
        log_message(logger, f"  • Accuracy: {metrics_pca_lda['Accuracy_test']:.4f}")
        log_message(logger, f"  • F1: {metrics_pca_lda['F1_test']:.4f}")
        log_message(logger, f"  • PR-AUC: {metrics_pca_lda['PR_AUC_test']:.4f}")
        log_message(logger, f"  • KS: {metrics_pca_lda['KS_test']:.4f}")
        log_message(logger, f"  • Brier: {metrics_pca_lda['Brier_test']:.4f}")
        # Métricas detalladas solo al log
    ModelingEngine.save_model(model_pca_lda, os.path.join(models_dir, 'logit_pca_lda.pkl'))
        log_message(logger, f"Modelo LOGIT PCA+LDA guardado: {os.path.join(models_dir, 'logit_pca_lda.pkl')}")
        log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'logit_pca_lda.pkl')}")
        log_message(logger, f"[TIMER] LOGIT PCA+LDA: {time.perf_counter()-t0:.2f}s")

        # LOGIT ANOVA
        log_message(logger, "Entrenando LOGIT con ANOVA...")
        # Solo log; sin print en consola
    t0 = time.perf_counter()
    model_anova, metrics_anova, y_pred_anova, y_proba_anova, \
    X_train_anova, X_test_anova, y_train_anova, y_test_anova = \
        ModelingEngine.fit_logit_predict(
            df=df_model_anova,
                    num_cols=final_vars_anova,
                    cat_cols=[],  # WOE ya aplicado, no hay categóricas originales
            target='flg_target',
            model_params={'solver': 'lbfgs'},
            show_confusion=False,
            verbose=True,
            save_confusion_path=os.path.join(images_dir, 'logit_anova_confusion.png')
        )
        log_message(logger, f"LOGIT ANOVA completado:")
        log_message(logger, f"  • AUC: {metrics_anova['AUC_test']:.4f}")
        log_message(logger, f"  • Accuracy: {metrics_anova['Accuracy_test']:.4f}")
        log_message(logger, f"  • F1: {metrics_anova['F1_test']:.4f}")
        log_message(logger, f"  • PR-AUC: {metrics_anova['PR_AUC_test']:.4f}")
        log_message(logger, f"  • KS: {metrics_anova['KS_test']:.4f}")
        log_message(logger, f"  • Brier: {metrics_anova['Brier_test']:.4f}")
        # Métricas detalladas solo al log
    ModelingEngine.save_model(model_anova, os.path.join(models_dir, 'logit_anova.pkl'))
        log_message(logger, f"Modelo LOGIT ANOVA guardado: {os.path.join(models_dir, 'logit_anova.pkl')}")
        log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'logit_anova.pkl')}")
        log_message(logger, f"[TIMER] LOGIT ANOVA: {time.perf_counter()-t0:.2f}s")

        # Comparativa rápida LOGIT
        t0 = time.perf_counter()
        log_message(logger, f"Métricas LOGIT - PCA+LDA AUC: {metrics_pca_lda['AUC_test']:.4f}, ANOVA AUC: {metrics_anova['AUC_test']:.4f}")
        log_message(logger, 
            f"PCA+LDA:  AUC={metrics_pca_lda['AUC_test']:.4f}, Acc={metrics_pca_lda['Accuracy_test']:.4f}, "
            f"KS={metrics_pca_lda['KS_test']:.4f}, PR_AUC(AP)={metrics_pca_lda['PR_AUC_test']:.4f}, "
            f"Brier={metrics_pca_lda['Brier_test']:.4f}"
        )
        log_message(logger,
            f"ANOVA:    AUC={metrics_anova['AUC_test']:.4f}, Acc={metrics_anova['Accuracy_test']:.4f}, "
            f"KS={metrics_anova['KS_test']:.4f}, PR_AUC(AP)={metrics_anova['PR_AUC_test']:.4f}, "
            f"Brier={metrics_anova['Brier_test']:.4f}"
        )
        log_message(logger, f"[TIMER] Comparativa LOGIT: {time.perf_counter()-t0:.2f}s")
    else:
        log_message(logger, "LOGIT deshabilitado en configuración - saltando...")
        # Sin prints en consola

    # section('4) SELECCIÓN DE UMBRALES (recall-first, F2 y Top-k)')
    # TARGET_RECALL = 0.90
    # TOPK_RATIO = 0.10
    # print(f"Objetivos -> Recall: {TARGET_RECALL:.0%} | Top-k: {int(TOPK_RATIO*100)}%")
    #
    # thr_rec90_pca = ModelingEngine.threshold_for_target_recall(y_test_pca_lda, y_proba_pca_lda, target_recall=TARGET_RECALL, verbose=True)
    # thr_rec90_anova = ModelingEngine.threshold_for_target_recall(y_test_anova, y_proba_anova, target_recall=TARGET_RECALL, verbose=True)
    #
    # thr_f2_pca = ModelingEngine.threshold_max_fbeta(y_test_pca_lda, y_proba_pca_lda, beta=2.0, verbose=True)
    # thr_f2_anova = ModelingEngine.threshold_max_fbeta(y_test_anova, y_proba_anova, beta=2.0, verbose=True)
    #
    # thr_topk_pca = ModelingEngine.threshold_for_topk(y_proba_pca_lda, topk_ratio=TOPK_RATIO, verbose=True)
    # thr_topk_anova = ModelingEngine.threshold_for_topk(y_proba_anova, topk_ratio=TOPK_RATIO, verbose=True)
    #
    # section('5) EVALUACIÓN POR UMBRAL (reporte operativo)')
    # evals_pca = pd.DataFrame([
    #     evaluar_a_umbral(y_test_pca_lda, y_proba_pca_lda, thr_rec90_pca),
    #     evaluar_a_umbral(y_test_pca_lda, y_proba_pca_lda, thr_f2_pca),
    #     evaluar_a_umbral(y_test_pca_lda, y_proba_pca_lda, thr_topk_pca),
    # ]).assign(criterio=[f'Recall≥{int(TARGET_RECALL*100)}%', 'Max F2', f'Top {int(TOPK_RATIO*100)}%'])
    # print("PCA+LDA | Evaluación por umbral:")
    # print(evals_pca.to_string(index=False))
    #
    # evals_anova = pd.DataFrame([
    #     evaluar_a_umbral(y_test_anova, y_proba_anova, thr_rec90_anova),
    #     evaluar_a_umbral(y_test_anova, y_proba_anova, thr_f2_anova),
    #     evaluar_a_umbral(y_test_anova, y_proba_anova, thr_topk_anova),
    # ]).assign(criterio=[f'Recall≥{int(TARGET_RECALL*100)}%', 'Max F2', f'Top {int(TOPK_RATIO*100)}%'])
    # print("ANOVA | Evaluación por umbral:")
    # print(evals_anova.to_string(index=False))

    section('6) COMPARATIVA RÁPIDA DE MÉTRICAS BASE (LOGIT)')
    t0 = time.perf_counter()
    log_message(logger,
        f"PCA+LDA:  AUC={metrics_pca_lda['AUC_test']:.4f}, Acc={metrics_pca_lda['Accuracy_test']:.4f}, "
        f"KS={metrics_pca_lda['KS_test']:.4f}, PR_AUC(AP)={metrics_pca_lda['PR_AUC_test']:.4f}, "
        f"Brier={metrics_pca_lda['Brier_test']:.4f}"
    )
    log_message(logger,
        f"ANOVA:    AUC={metrics_anova['AUC_test']:.4f}, Acc={metrics_anova['Accuracy_test']:.4f}, "
        f"KS={metrics_anova['KS_test']:.4f}, PR_AUC(AP)={metrics_anova['PR_AUC_test']:.4f}, "
        f"Brier={metrics_anova['Brier_test']:.4f}"
    )
    log_message(logger, f"[TIMER] Comparativa LOGIT: {time.perf_counter()-t0:.2f}s")

    section("7) SENSIBILIDAD 'int_rate' (LOGIT)")
    t0 = time.perf_counter()
    sensibilidad_int_rate(model_pca_lda, X_test_pca_lda, y_test_pca_lda, 'LOGIT | PCA+LDA', images_dir, max_rows=50000)
    sensibilidad_int_rate(model_anova, X_test_anova, y_test_anova, 'LOGIT | ANOVA', images_dir, max_rows=50000)
    log_message(logger, f"[TIMER] Sensibilidad LOGIT: {time.perf_counter()-t0:.2f}s")

    # ===== SECCIONES COMENTADAS TEMPORALMENTE PARA EVITAR ERRORES DE INDENTACIÓN =====
    # section('8) GAUSSIAN NAIVE BAYES (GNB)')
    # log_message(logger, "Entrenando GNB con PCA+LDA+correlación + WOE...")
    t0 = time.perf_counter()
    # gnb_pca_lda, gnb_metrics_pca_lda, gnb_y_pred_pca_lda, gnb_y_proba_pca_lda, \
    # Xtr_gnb_pca_lda, Xte_gnb_pca_lda, ytr_gnb_pca_lda, yte_gnb_pca_lda = ModelingEngine.fit_gaussian_nb_predict(
    # df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'gnb_pca_lda_confusion.png')
    # )
    # print_metrics_block('GNB | PCA+LDA', gnb_metrics_pca_lda, logger)
    # ModelingEngine.save_model(gnb_pca_lda, os.path.join(models_dir, 'gnb_pca_lda.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'gnb_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] GNB PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # log_message(logger, "Entrenando GNB con ANOVA+correlación + WOE...")
    t0 = time.perf_counter()
    # gnb_anova, gnb_metrics_anova, gnb_y_pred_anova, gnb_y_proba_anova, \
    # Xtr_gnb_anova, Xte_gnb_anova, ytr_gnb_anova, yte_gnb_anova = ModelingEngine.fit_gaussian_nb_predict(
    # df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'gnb_anova_confusion.png')
    # )
    # print_metrics_block('GNB | ANOVA', gnb_metrics_anova, logger)
    # ModelingEngine.save_model(gnb_anova, os.path.join(models_dir, 'gnb_anova.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'gnb_anova.pkl')}")
    # log_message(logger, f"[TIMER] GNB ANOVA: {time.perf_counter()-t0:.2f}s")

    # section('9) GNB SMART SEARCH (var_smoothing)')
    # print("Búsqueda rápida (sin CV) sobre PCA+LDA...")
    t0 = time.perf_counter()
    # gnb_best_pca_lda, best_params_pca_lda, gnb_metrics_best_pca_lda, gnb_y_pred_best_pca_lda, gnb_y_proba_best_pca_lda, \
    # Xtr_gnb_best_pca_lda, Xte_gnb_best_pca_lda, ytr_gnb_best_pca_lda, yte_gnb_best_pca_lda = ModelingEngine.fit_gaussian_nb_smartsearch(
    #     df_model_pca_lda,
    #     num_cols=vars_corr_pca_lda,
    #     cat_cols=[],
    #     target='flg_target',
    #     standardize_numeric=True,
    #     verbose=True
    # )
    # print(f"Mejor var_smoothing (PCA+LDA): {best_params_pca_lda['classifier__var_smoothing']}")
    # print_metrics_block('GNB | PCA+LDA (best)', gnb_metrics_best_pca_lda)
    # ModelingEngine.save_model(gnb_best_pca_lda, os.path.join(models_dir, 'gnb_best_pca_lda.pkl'))
    # print(f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'gnb_best_pca_lda.pkl')}")
    # print(f"[TIMER] GNB smartsearch PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # print("Búsqueda rápida (sin CV) sobre ANOVA...")
    t0 = time.perf_counter()
    # gnb_best, best_params, gnb_metrics_best, gnb_y_pred_best, gnb_y_proba_best, \
    # Xtr_gnb_best, Xte_gnb_best, ytr_gnb_best, yte_gnb_best = ModelingEngine.fit_gaussian_nb_smartsearch(
    #     df_model_anova,
    #     num_cols=vars_corr_anova,
    #     cat_cols=[],
    #     target='flg_target',
    #     standardize_numeric=True,
    #     verbose=True
    # )
    # print(f"Mejor var_smoothing (ANOVA): {best_params['classifier__var_smoothing']}")
    # print_metrics_block('GNB | ANOVA (best)', gnb_metrics_best)
    # ModelingEngine.save_model(gnb_best, os.path.join(models_dir, 'gnb_best_anova.pkl'))
    # print(f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'gnb_best_anova.pkl')}")
    # print(f"[TIMER] GNB smartsearch ANOVA: {time.perf_counter()-t0:.2f}s")

    # section("10) SENSIBILIDAD 'int_rate' (GNB)")
    t0 = time.perf_counter()
    # sensibilidad_int_rate(gnb_pca_lda, Xte_gnb_pca_lda, yte_gnb_pca_lda, 'GNB | PCA+LDA', images_dir, max_rows=50000)
    # sensibilidad_int_rate(gnb_anova, Xte_gnb_anova, yte_gnb_anova, 'GNB | ANOVA', images_dir, max_rows=50000)
    # log_message(logger, f"[TIMER] Sensibilidad GNB: {time.perf_counter()-t0:.2f}s")

    # section('11) LINEAR SVM (calibrado)')
    # log_message(logger, "Entrenando Linear SVM calibrado con PCA+LDA+correlación + WOE...")
    t0 = time.perf_counter()
    # svm_pca_lda, svm_metrics_pca_lda, svm_y_pred_pca_lda, svm_y_proba_pca_lda, \
    # Xtr_svm_pca_lda, Xte_svm_pca_lda, ytr_svm_pca_lda, yte_svm_pca_lda = ModelingEngine.fit_linear_svm_predict(
    # df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_pca_lda_confusion.png')
    # )
    # print_metrics_block('Linear SVM | PCA+LDA', svm_metrics_pca_lda, logger)
    # ModelingEngine.save_model(svm_pca_lda, os.path.join(models_dir, 'svm_pca_lda.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] SVM PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # log_message(logger, "Entrenando Linear SVM calibrado con ANOVA+correlación + WOE...")
    t0 = time.perf_counter()
    # svm_anova, svm_metrics_anova, svm_y_pred_anova, svm_y_proba_anova, \
    # Xtr_svm_anova, Xte_svm_anova, ytr_svm_anova, yte_svm_anova = ModelingEngine.fit_linear_svm_predict(
    # df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_anova_confusion.png')
    # )
    # print_metrics_block('Linear SVM | ANOVA', svm_metrics_anova, logger)
    # ModelingEngine.save_model(svm_anova, os.path.join(models_dir, 'svm_anova.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_anova.pkl')}")
    # log_message(logger, f"[TIMER] SVM ANOVA: {time.perf_counter()-t0:.2f}s")

    # section("12) SVM SMART SEARCH (Linear SVM)")
    # log_message(logger, "Búsqueda rápida (smart) sobre PCA+LDA + WOE...")
    t0 = time.perf_counter()
    # svm_best_pca_lda, best_svm_params_pca_lda, svm_metrics_best_pca_lda, svm_y_pred_best_pca_lda, svm_y_proba_best_pca_lda, \
    # Xtr_svm_best_pca_lda, Xte_svm_best_pca_lda, ytr_svm_best_pca_lda, yte_svm_best_pca_lda = ModelingEngine.fit_linear_svm_smartsearch(
    # df=df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_best_pca_lda_confusion.png'),
    # title_suffix='PCA+LDA'
    # )
    # log_message(logger, f"Mejor params SVM (PCA+LDA): {best_svm_params_pca_lda}")
    # print_metrics_block('SVM | PCA+LDA (best)', svm_metrics_best_pca_lda, logger)
    # ModelingEngine.save_model(svm_best_pca_lda, os.path.join(models_dir, 'svm_best_pca_lda.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_best_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] SVM smartsearch PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # log_message(logger, "Búsqueda rápida (smart) sobre ANOVA + WOE...")
    t0 = time.perf_counter()
    # svm_best_anova, best_svm_params_anova, svm_metrics_best_anova, svm_y_pred_best_anova, svm_y_proba_best_anova, \
    # Xtr_svm_best_anova, Xte_svm_best_anova, ytr_svm_best_anova, yte_svm_best_anova = ModelingEngine.fit_linear_svm_smartsearch(
    # df=df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_best_anova_confusion.png'),
    # title_suffix='ANOVA'
    # )
    # log_message(logger, f"Mejor params SVM (ANOVA): {best_svm_params_anova}")
    # print_metrics_block('SVM | ANOVA (best)', svm_metrics_best_anova, logger)
    # ModelingEngine.save_model(svm_best_anova, os.path.join(models_dir, 'svm_best_anova.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_best_anova.pkl')}")
    # log_message(logger, f"[TIMER] SVM smartsearch ANOVA: {time.perf_counter()-t0:.2f}s")

    # section("13) SVM RBF (no lineal) – SMART SEARCH")
    # log_message(logger, "Búsqueda rápida (smart) SVM RBF sobre PCA+LDA + WOE...")
    t0 = time.perf_counter()
    # svm_rbf_best_pca_lda, best_rbf_params_pca_lda, svm_rbf_metrics_best_pca_lda, svm_rbf_y_pred_best_pca_lda, svm_rbf_y_proba_best_pca_lda, \
    # Xtr_svm_rbf_best_pca_lda, Xte_svm_rbf_best_pca_lda, ytr_svm_rbf_best_pca_lda, yte_svm_rbf_best_pca_lda = ModelingEngine.fit_rbf_svm_smartsearch(
    # df=df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_rbf_best_pca_lda_confusion.png'),
    # title_suffix='PCA+LDA'
    # )
    # log_message(logger, f"Mejor params SVM RBF (PCA+LDA): {best_rbf_params_pca_lda}")
    # print_metrics_block('SVM RBF | PCA+LDA (best)', svm_rbf_metrics_best_pca_lda, logger)
    # ModelingEngine.save_model(svm_rbf_best_pca_lda, os.path.join(models_dir, 'svm_rbf_best_pca_lda.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_rbf_best_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] SVM RBF smartsearch PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # log_message(logger, "Búsqueda rápida (smart) SVM RBF sobre ANOVA + WOE...")
    t0 = time.perf_counter()
    # svm_rbf_best_anova, best_rbf_params_anova, svm_rbf_metrics_best_anova, svm_rbf_y_pred_best_anova, svm_rbf_y_proba_best_anova, \
    # Xtr_svm_rbf_best_anova, Xte_svm_rbf_best_anova, ytr_svm_rbf_best_anova, yte_svm_rbf_best_anova = ModelingEngine.fit_rbf_svm_smartsearch(
    # df=df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'svm_rbf_best_anova_confusion.png'),
    # title_suffix='ANOVA'
    # )
    # log_message(logger, f"Mejor params SVM RBF (ANOVA): {best_rbf_params_anova}")
    # print_metrics_block('SVM RBF | ANOVA (best)', svm_rbf_metrics_best_anova, logger)
    # ModelingEngine.save_model(svm_rbf_best_anova, os.path.join(models_dir, 'svm_rbf_best_anova.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'svm_rbf_best_anova.pkl')}")
    # log_message(logger, f"[TIMER] SVM RBF smartsearch ANOVA: {time.perf_counter()-t0:.2f}s")

    # section("14) RED NEURONAL (MLP)")
    # log_message(logger, "Entrenando MLP con PCA+LDA+correlación + WOE...")
    t0 = time.perf_counter()
    # mlp_best_pca_lda, best_mlp_params_pca_lda, mlp_metrics_pca_lda, mlp_y_pred_pca_lda, mlp_y_proba_pca_lda, \
    # Xtr_mlp_pca_lda, Xte_mlp_pca_lda, ytr_mlp_pca_lda, yte_mlp_pca_lda = ModelingEngine.fit_mlp_smartsearch(
    # df=df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'mlp_pca_lda_confusion.png'),
    # title_suffix='PCA+LDA'
    # )
    # log_message(logger, f"Mejor params MLP (PCA+LDA): {best_mlp_params_pca_lda}")
    # print_metrics_block('MLP | PCA+LDA', mlp_metrics_pca_lda, logger)
    # ModelingEngine.save_model(mlp_best_pca_lda, os.path.join(models_dir, 'mlp_pca_lda.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'mlp_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] MLP PCA+LDA: {time.perf_counter()-t0:.2f}s")

    # log_message(logger, "Entrenando MLP con ANOVA+correlación + WOE...")
    t0 = time.perf_counter()
    # mlp_best_anova, best_mlp_params_anova, mlp_metrics_anova, mlp_y_pred_anova, mlp_y_proba_anova, \
    # Xtr_mlp_anova, Xte_mlp_anova, ytr_mlp_anova, yte_mlp_anova = ModelingEngine.fit_mlp_smartsearch(
    # df=df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado
    # target='flg_target',
    # standardize_numeric=True,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'mlp_anova_confusion.png'),
    # title_suffix='ANOVA'
    # )
    # log_message(logger, f"Mejor params MLP (ANOVA): {best_mlp_params_anova}")
    # print_metrics_block('MLP | ANOVA', mlp_metrics_anova, logger)
    # ModelingEngine.save_model(mlp_best_anova, os.path.join(models_dir, 'mlp_anova.pkl'))
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'mlp_anova.pkl')}")
    # log_message(logger, f"[TIMER] MLP ANOVA: {time.perf_counter()-t0:.2f}s")

    # section("15) SENSIBILIDAD 'int_rate' (SVM/MLP)")
    t0 = time.perf_counter()
    # sensibilidad_int_rate(svm_best_pca_lda, Xte_svm_best_pca_lda, yte_svm_best_pca_lda, 'SVM best | PCA+LDA', images_dir, max_rows=50000)
    # sensibilidad_int_rate(svm_best_anova, Xte_svm_best_anova, yte_svm_best_anova, 'SVM best | ANOVA', images_dir, max_rows=50000)
    # sensibilidad_int_rate(mlp_best_pca_lda, Xte_mlp_pca_lda, yte_mlp_pca_lda, 'MLP | PCA+LDA', images_dir, max_rows=50000)
    # sensibilidad_int_rate(mlp_best_anova, Xte_mlp_anova, yte_mlp_anova, 'MLP | ANOVA', images_dir, max_rows=50000)
    # log_message(logger, f"[TIMER] Sensibilidad SVM: {time.perf_counter()-t0:.2f}s")

    # section('16) COMPARATIVA RÁPIDA DE MODELOS (AUC / PR-AUC)')
    t0 = time.perf_counter()
    # print(
    # f"GNB PCA+LDA: AUC={gnb_metrics_pca_lda['AUC_test']:.4f}, PR-AUC(AP)={gnb_metrics_pca_lda['PR_AUC_test']:.4f}\n"
    # f"GNB ANOVA:   AUC={gnb_metrics_anova['AUC_test']:.4f}, PR-AUC(AP)={gnb_metrics_anova['PR_AUC_test']:.4f}\n"
        # f"GNB BEST PCA+LDA: AUC={gnb_metrics_best_pca_lda['AUC_test']:.4f}, PR-AUC(AP)={gnb_metrics_best_pca_lda['PR_AUC_test']:.4f}\n"
        # f"GNB BEST ANOVA:   AUC={gnb_metrics_best['AUC_test']:.4f}, PR-AUC(AP)={gnb_metrics_best['PR_AUC_test']:.4f}\n"
    # f"SVM best PCA+LDA: AUC={svm_metrics_best_pca_lda['AUC_test']:.4f}, PR-AUC(AP)={svm_metrics_best_pca_lda['PR_AUC_test']:.4f}\n"
    # f"SVM best ANOVA:   AUC={svm_metrics_best_anova['AUC_test']:.4f}, PR-AUC(AP)={svm_metrics_best_anova['PR_AUC_test']:.4f}\n"
    # f"SVM RBF best PCA+LDA: AUC={svm_rbf_metrics_best_pca_lda['AUC_test']:.4f}, PR-AUC(AP)={svm_rbf_metrics_best_pca_lda['PR_AUC_test']:.4f}\n"
    # f"SVM RBF best ANOVA:   AUC={svm_rbf_metrics_best_anova['AUC_test']:.4f}, PR-AUC(AP)={svm_rbf_metrics_best_anova['PR_AUC_test']:.4f}\n"
    # f"MLP PCA+LDA: AUC={mlp_metrics_pca_lda['AUC_test']:.4f}, PR-AUC(AP)={mlp_metrics_pca_lda['PR_AUC_test']:.4f}\n"
    # f"MLP ANOVA:   AUC={mlp_metrics_anova['AUC_test']:.4f}, PR-AUC(AP)={mlp_metrics_anova['PR_AUC_test']:.4f}"
    # )
    # log_message(logger, f"[TIMER] Comparativa final: {time.perf_counter()-t0:.2f}s")

    # section('17) CURVAS ROC y PR EN UN ÚNICO GRÁFICO')
    t0 = time.perf_counter()
    # modelos_para_curvas = [
    # {'label': 'LOGIT | PCA+LDA', 'y_true': y_test_pca_lda, 'y_proba': y_proba_pca_lda},
    # {'label': 'LOGIT | ANOVA', 'y_true': y_test_anova, 'y_proba': y_proba_anova},
    # {'label': 'GNB | PCA+LDA', 'y_true': yte_gnb_pca_lda, 'y_proba': gnb_y_proba_pca_lda},
    # {'label': 'GNB | ANOVA', 'y_true': yte_gnb_anova, 'y_proba': gnb_y_proba_anova},
        # {'label': 'GNB (best) | PCA+LDA', 'y_true': yte_gnb_best_pca_lda, 'y_proba': gnb_y_proba_best_pca_lda},
        # {'label': 'GNB (best) | ANOVA', 'y_true': yte_gnb_best, 'y_proba': gnb_y_proba_best},
    # {'label': 'SVM best | PCA+LDA', 'y_true': yte_svm_best_pca_lda, 'y_proba': svm_y_proba_best_pca_lda},
    # {'label': 'SVM best | ANOVA', 'y_true': yte_svm_best_anova, 'y_proba': svm_y_proba_best_anova},
    # {'label': 'SVM RBF best | PCA+LDA', 'y_true': yte_svm_rbf_best_pca_lda, 'y_proba': svm_rbf_y_proba_best_pca_lda},
    # {'label': 'SVM RBF best | ANOVA', 'y_true': yte_svm_rbf_best_anova, 'y_proba': svm_rbf_y_proba_best_anova},
    # {'label': 'MLP | PCA+LDA', 'y_true': yte_mlp_pca_lda, 'y_proba': mlp_y_proba_pca_lda},
    # {'label': 'MLP | ANOVA', 'y_true': yte_mlp_anova, 'y_proba': mlp_y_proba_anova},
    # ]
    # plot_curvas_comparativas(modelos_para_curvas, os.path.join(images_dir, 'comparativa_curvas_modelos.png'))
    # log_message(logger, f"[TIMER] Curvas comparativas: {time.perf_counter()-t0:.2f}s")

    # section("18) SENSIBILIDAD COMPARATIVA EN UN SOLO GRÁFICO")
    t0 = time.perf_counter()
    # Recalcular dataframes de sensibilidad para tenerlos en memoria
    # tasas = np.arange(10, 95, 5)
    # sens_list = []
    # sens_list.append({
    # 'label': 'LOGIT | PCA+LDA',
    # 'df': ModelingEngine.variable_sensitivity(model_pca_lda, _stratified_sample_Xy(X_test_pca_lda.copy(), y_test_pca_lda, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'LOGIT | ANOVA',
    # 'df': ModelingEngine.variable_sensitivity(model_anova, _stratified_sample_Xy(X_test_anova.copy(), y_test_anova, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'GNB | PCA+LDA',
    # 'df': ModelingEngine.variable_sensitivity(gnb_pca_lda, _stratified_sample_Xy(Xte_gnb_pca_lda.copy(), yte_gnb_pca_lda, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'GNB | ANOVA',
    # 'df': ModelingEngine.variable_sensitivity(gnb_anova, _stratified_sample_Xy(Xte_gnb_anova.copy(), yte_gnb_anova, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'SVM best | PCA+LDA',
    # 'df': ModelingEngine.variable_sensitivity(svm_best_pca_lda, _stratified_sample_Xy(Xte_svm_best_pca_lda.copy(), yte_svm_best_pca_lda, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'SVM best | ANOVA',
    # 'df': ModelingEngine.variable_sensitivity(svm_best_anova, _stratified_sample_Xy(Xte_svm_best_anova.copy(), yte_svm_best_anova, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'SVM RBF best | PCA+LDA',
    # 'df': ModelingEngine.variable_sensitivity(svm_rbf_best_pca_lda, _stratified_sample_Xy(Xte_svm_rbf_best_pca_lda.copy(), yte_svm_rbf_best_pca_lda, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'SVM RBF best | ANOVA',
    # 'df': ModelingEngine.variable_sensitivity(svm_rbf_best_anova, _stratified_sample_Xy(Xte_svm_rbf_best_anova.copy(), yte_svm_rbf_best_anova, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'MLP | PCA+LDA',
    # 'df': ModelingEngine.variable_sensitivity(mlp_best_pca_lda, _stratified_sample_Xy(Xte_mlp_pca_lda.copy(), yte_mlp_pca_lda, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # sens_list.append({
    # 'label': 'MLP | ANOVA',
    # 'df': ModelingEngine.variable_sensitivity(mlp_best_anova, _stratified_sample_Xy(Xte_mlp_anova.copy(), yte_mlp_anova, max_rows=50000), 'int_rate', tasas, plot=False, verbose=False, max_rows=None)
    # })
    # plot_sensibilidad_comparativa(sens_list, 'int_rate', os.path.join(images_dir, 'comparativa_sensibilidad_int_rate.png'))
    # log_message(logger, f"[TIMER] Sensibilidad comparativa: {time.perf_counter()-t0:.2f}s")

    # section('19) TABLA COMPARATIVA (AUC, PR-AUC, Brier, Precision, Recall, etc.)')
    t0 = time.perf_counter()
    # comparator = ModelComparator(base_dir)
    # df_cmp = comparator.compare()
    # No imprimir comparativa en consola
    # out_csv = comparator.save(df_cmp)
    # log_message(logger, f"[SAVE] Comparativa guardada en: {out_csv}")
    # log_message(logger, f"[TIMER] Tabla comparativa: {time.perf_counter()-t0:.2f}s")

    # section('FIN')
    # log_message(logger, 'Ejecución completada correctamente.')
    # log_message(logger, f"[TIMER] Tiempo total: {time.perf_counter()-t0_total:.2f}s")


    def main() -> None:
    """
    Función principal del pipeline de modelado.
    """
    # Cargar configuración
    config_path = 'engine_TFM/config_modeling.yml'
    config = load_config(config_path)

    # Configurar logging
    logger = setup_logging(config_path)
    log_message(logger, "Iniciando pipeline de modelado con configuración:")
    log_message(logger, f"  • Archivo de configuración: {config_path}")

    # Mostrar configuración de modelos
    models_cfg = config.get('models', {})
    log_message(logger, "\nConfiguración de modelos a ejecutar:")
    for model_name, enabled in models_cfg.items():
        status = "✅ HABILITADO" if enabled else "❌ DESHABILITADO"
        log_message(logger, f"  • {model_name}: {status}")

    # Barra de progreso inicial
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(0, 5, section_name='🔧 INICIANDO MODELING PIPELINE...')

    # Aquí va todo el código existente del pipeline
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pca_lda = os.path.join(base_dir, 'df_pca_lda.csv')
    csv_anova = os.path.join(base_dir, 'df_anova.csv')
    models_dir = os.path.join(base_dir, 'models')
    images_dir = os.path.join(base_dir, 'imagenes')
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Barra de progreso - Paso 1
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(1, 5, section_name='📥 Cargando datos')
    else:
        section('1) CARGA DE DATOS')

    t0_total = time.perf_counter()
    t0 = time.perf_counter()
    log_message(logger, f"Leyendo: {csv_pca_lda}")
    df_pca_lda = pd.read_csv(csv_pca_lda)
    log_message(logger, f"Leyendo: {csv_anova}")
    df_anova = pd.read_csv(csv_anova)
    log_message(logger, f"Tamaños -> PCA+LDA: {df_pca_lda.shape} | ANOVA: {df_anova.shape}")
    log_message(logger, f"[TIMER] Carga de datos: {time.perf_counter()-t0:.2f}s")

    # Continuar con el resto del código...
    # Variables (se asume última columna es target)
    all_vars_pca_lda = df_pca_lda.columns.tolist()[:-1]
    all_vars_anova = df_anova.columns.tolist()[:-1]

    # Identificar variables categóricas vs numéricas
    cat_cols_pca_lda = [col for col in all_vars_pca_lda if df_pca_lda[col].dtype == 'object']
    cat_cols_anova = [col for col in all_vars_anova if df_anova[col].dtype == 'object']
    num_vars_pca_lda = [col for col in all_vars_pca_lda if col not in cat_cols_pca_lda]
    num_vars_anova = [col for col in all_vars_anova if col not in cat_cols_anova]

    log_message(logger, f"Variables PCA+LDA -> Numéricas: {len(num_vars_pca_lda)} | Categóricas: {len(cat_cols_pca_lda)}")
    log_message(logger, f"Variables ANOVA -> Numéricas: {len(num_vars_anova)} | Categóricas: {len(cat_cols_anova)}")

    if cat_cols_pca_lda:
    log_message(logger, f"Categóricas PCA+LDA: {cat_cols_pca_lda}")
    if cat_cols_anova:
    log_message(logger, f"Categóricas ANOVA: {cat_cols_anova}")

    # Paso 2: Preparación de dataframes
    if config.get('logging', {}).get('enable_progress_bar', True):
    print_section_progress(2, 19, section_name='2) PREPARACIÓN WOE')
    else:
    section('2) PREPARACIÓN DE DATAFRAMES CON WOE')

    t0 = time.perf_counter()

    # Preparar dataframes base (solo numéricas por ahora)
    df_model_pca_lda = preparar_df_modelo_sin_fillna(df_pca_lda, num_vars_pca_lda + cat_cols_pca_lda)
    df_model_anova = preparar_df_modelo_sin_fillna(df_anova, num_vars_anova + cat_cols_anova)

    # Aplicar WOE a variables categóricas si existen
    woe_mappings_dir = os.path.join(base_dir, 'woe_mappings')
    os.makedirs(woe_mappings_dir, exist_ok=True)

    if cat_cols_pca_lda:
    log_message(logger, "Aplicando WOE a variables categóricas PCA+LDA...")
    woe_pca_lda = WOETransformer(target_col='flg_target')
    df_model_pca_lda = woe_pca_lda.fit_transform(df_model_pca_lda, cat_cols_pca_lda)

        # Guardar mapeos WOE
    woe_mapping_file = os.path.join(woe_mappings_dir, 'woe_mappings_pca_lda.json')
    woe_pca_lda.save_mappings(woe_mapping_file)

        # Guardar dataframe con WOE para evaluación posterior
    df_pca_lda_woe = df_model_pca_lda.copy()
    df_pca_lda_woe_path = os.path.join(base_dir, 'df_pca_lda_woe.csv')
    df_pca_lda_woe.to_csv(df_pca_lda_woe_path, index=False)
    log_message(logger, f"[SAVE] DataFrame PCA+LDA con WOE guardado: {df_pca_lda_woe_path}")

        # Mostrar estadísticas WOE
    woe_stats = woe_pca_lda.get_summary_stats()
    log_message(logger, "Estadísticas WOE PCA+LDA generadas")
        # detallado solo en log

    if cat_cols_anova:
    log_message(logger, "Aplicando WOE a variables categóricas ANOVA...")
    woe_anova = WOETransformer(target_col='flg_target')
    df_model_anova = woe_anova.fit_transform(df_model_anova, cat_cols_anova)

        # Guardar mapeos WOE
    woe_mapping_file = os.path.join(woe_mappings_dir, 'woe_mappings_anova.json')
    woe_anova.save_mappings(woe_mapping_file)

        # Guardar dataframe con WOE para evaluación posterior
    df_anova_woe = df_model_anova.copy()
    df_anova_woe_path = os.path.join(base_dir, 'df_anova_woe.csv')
    df_anova_woe.to_csv(df_anova_woe_path, index=False)
    log_message(logger, f"[SAVE] DataFrame ANOVA con WOE guardado: {df_anova_woe_path}")

        # Mostrar estadísticas WOE
    woe_stats = woe_anova.get_summary_stats()
    log_message(logger, "Estadísticas WOE ANOVA generadas")
        # detallado solo en log

    # Variables finales (numéricas + WOE)
    final_vars_pca_lda = num_vars_pca_lda + ([f"{col}_woe" for col in cat_cols_pca_lda] if cat_cols_pca_lda else [])
    final_vars_anova = num_vars_anova + ([f"{col}_woe" for col in cat_cols_anova] if cat_cols_anova else [])

    log_message(logger, f"Variables finales PCA+LDA: {len(final_vars_pca_lda)} | ANOVA: {len(final_vars_anova)}")

    # ===== PASO 3: REGRESIÓN LOGÍSTICA =====
    if models_cfg.get('enable_logit', True):
        if config.get('logging', {}).get('enable_progress_bar', True):
    # print_section_progress(3, 19, section_name='3) LOGIT')
    else:
    # section('3) REGRESIÓN LOGÍSTICA (con WOE)')

    # log_message(logger, "Entrenando LOGIT con PCA+LDA+correlación + WOE...")
        # sin prints
    t0 = time.perf_counter()
    # model_pca_lda, metrics_pca_lda, y_pred_pca_lda, y_proba_pca_lda, \
    # X_train_pca_lda, X_test_pca_lda, y_train_pca_lda, y_test_pca_lda = \
    # ModelingEngine.fit_logit_predict(
    # df=df_model_pca_lda,
    # num_cols=final_vars_pca_lda,
    # cat_cols=[],  # WOE ya aplicado, no hay categóricas originales
    # target='flg_target',
    # model_params={'solver': 'lbfgs'},
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'logit_pca_lda_confusion.png')
    # )
    # log_message(logger, f"LOGIT PCA+LDA completado - AUC: {metrics_pca_lda['AUC_test']:.4f}")
        # métricas solo al log
    # ModelingEngine.save_model(model_pca_lda, os.path.join(models_dir, 'logit_pca_lda.pkl'))
    # log_message(logger, f"Modelo LOGIT PCA+LDA guardado")
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'logit_pca_lda.pkl')}")
    # log_message(logger, f"[TIMER] LOGIT PCA+LDA: {time.perf_counter()-t0:.2f}s")
        # timer solo al log

    # log_message(logger, "Entrenando LOGIT con ANOVA+correlación + WOE...")
        # sin prints
    t0 = time.perf_counter()
    # model_anova, metrics_anova, y_pred_anova, y_proba_anova, \
    # X_train_anova, X_test_anova, y_train_anova, y_test_anova = \
    # ModelingEngine.fit_logit_predict(
    # df=df_model_anova,
    # num_cols=final_vars_anova,
    # cat_cols=[],  # WOE ya aplicado, no hay categóricas originales
    # target='flg_target',
    # model_params={'solver': 'lbfgs'},
    # show_confusion=False,
    # verbose=True,
    # save_confusion_path=os.path.join(images_dir, 'logit_anova_confusion.png')
    # )
    # log_message(logger, f"LOGIT ANOVA completado - AUC: {metrics_anova['AUC_test']:.4f}")
        # métricas solo al log
    # ModelingEngine.save_model(model_anova, os.path.join(models_dir, 'logit_anova.pkl'))
    # log_message(logger, f"Modelo LOGIT ANOVA guardado")
    # log_message(logger, f"[SAVE] Modelo guardado: {os.path.join(models_dir, 'logit_anova.pkl')}")
    # log_message(logger, f"[TIMER] LOGIT ANOVA: {time.perf_counter()-t0:.2f}s")
        # timer solo al log



    # Continuar con otros modelos según configuración...
    # Aquí irían los demás modelos (GNB, SVM, MLP) con la misma lógica

    # ===== PASO 19: TABLA COMPARATIVA FINAL =====
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(4, 5, section_name='💾 Guardando resultados')
    else:
        section('19) TABLA COMPARATIVA (AUC, PR-AUC, Brier, Precision, Recall, etc.)')

    t0 = time.perf_counter()
    log_message(logger, "Generando tabla comparativa final de todos los modelos...")
    comparator = ModelComparator(base_dir)
    df_cmp = comparator.compare()
    
    # Registrar resultados completos en el log
    log_message(logger, "RESULTADOS FINALES - COMPARATIVA DE MODELOS:")
    log_message(logger, "=" * 60)
    for _, row in df_cmp.iterrows():
        log_message(logger, f"{row['Modelo']}:")
        log_message(logger, f"  • AUC: {row['AUC']:.4f} | PR-AUC: {row['PR_AUC(AP)']:.4f} | Brier: {row['Brier']:.4f}")
        log_message(logger, f"  • Accuracy: {row['Accuracy']:.4f} | F1: {row['F1']:.4f} | Balanced Acc: {row['BalancedAcc']:.4f}")
        log_message(logger, f"  • Precision: {row['Precision']:.4f} | Recall: {row['Recall']:.4f} | KS: {row['KS']:.4f}")
    
    out_csv = comparator.save(df_cmp)
    log_message(logger, f"Tabla comparativa guardada en: {out_csv}")
    log_message(logger, f"[TIMER] Tabla comparativa: {time.perf_counter()-t0:.2f}s")

    # Finalización
    log_message(logger, "MODELING PIPELINE COMPLETED SUCCESSFULLY")
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(5, 5, section_name='🎉 COMPLETADO')

    log_message(logger, 'Ejecución completada correctamente.')
    log_message(logger, f"[TIMER] Tiempo total: {time.perf_counter()-t0_total:.2f}s")
    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(4, 5, section_name='💾 Guardando resultados')
    else:
        section('19) TABLA COMPARATIVA (AUC, PR-AUC, Brier, Precision, Recall, etc.)')

        t0 = time.perf_counter()
        log_message(logger, "Generando tabla comparativa final de todos los modelos...")
    comparator = ModelComparator(base_dir)
    df_cmp = comparator.compare()
    
    # Registrar resultados completos en el log
        log_message(logger, "RESULTADOS FINALES - COMPARATIVA DE MODELOS:")
        log_message(logger, "=" * 60)
    for _, row in df_cmp.iterrows():
        log_message(logger, f"{row['Modelo']}:")
        log_message(logger, f"  • AUC: {row['AUC']:.4f} | PR-AUC: {row['PR_AUC(AP)']:.4f} | Brier: {row['Brier']:.4f}")
        log_message(logger, f"  • Accuracy: {row['Accuracy']:.4f} | F1: {row['F1']:.4f} | Balanced Acc: {row['BalancedAcc']:.4f}")
        log_message(logger, f"  • Precision: {row['Precision']:.4f} | Recall: {row['Recall']:.4f} | KS: {row['KS']:.4f}")
    
        log_message(logger, "=" * 60)
        log_message(logger, f"🏆 MEJOR MODELO POR AUC: {df_cmp.iloc[0]['Modelo']} (AUC: {df_cmp.iloc[0]['AUC']:.4f})")
        log_message(logger, f"🥈 SEGUNDO MEJOR: {df_cmp.iloc[1]['Modelo']} (AUC: {df_cmp.iloc[1]['AUC']:.4f})")
        log_message(logger, f"🥉 TERCER MEJOR: {df_cmp.iloc[2]['Modelo']} (AUC: {df_cmp.iloc[2]['AUC']:.4f})")
    
    # no prints en consola
    out_csv = comparator.save(df_cmp)
        log_message(logger, f"Tabla comparativa guardada en: {out_csv}")
        log_message(logger, f"[SAVE] Comparativa guardada en: {out_csv}")
        log_message(logger, f"[TIMER] Tabla comparativa: {time.perf_counter()-t0:.2f}s")

    # Finalización
        log_message(logger, "MODELING PIPELINE COMPLETED SUCCESSFULLY")
        log_message(logger, "=" * 80)

    if config.get('logging', {}).get('enable_progress_bar', True):
        print_section_progress(5, 5, section_name='🎉 COMPLETADO')

    # Fin: solo log
        log_message(logger, 'Ejecución completada correctamente.')
        log_message(logger, f"[TIMER] Tiempo total: {time.perf_counter()-t0_total:.2f}s")


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print("Error en la ejecución:", repr(exc))  # Solo errores críticos van a consola
        sys.exit(1)


