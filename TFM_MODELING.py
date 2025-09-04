"""
TFM_MODELING.py

Pipeline de modelado con logging completo a archivo y salida limpia en consola.
- Consola: solo barras de progreso y encabezado
- Archivo: logs detallados, warnings y métricas
"""

import os
import sys
import time
import logging
import warnings
import io
from contextlib import redirect_stdout
from datetime import datetime
from typing import List

import pandas as pd

from engine_TFM.engine_modeling import ModelingEngine
from engine_TFM.utils import (
    load_config,
    prepare_model_dataframe,
    ModelComparator,
    WOETransformer,
)

# Suprimir warnings en terminal; se capturan y registran en el log
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def print_section_progress(current: int, total: int, section_name: str = "", prefix: str = "", suffix: str = "", length: int = 50) -> None:
    percent = float(current) / float(total)
    filled_length = int(length * percent)
    bar = "█" * filled_length + "-" * (length - filled_length)
    if section_name:
        section_title = f"{section_name}: "
        sys.stdout.write(f"\r{section_title}|{bar}| {percent:.1%} {suffix}")
    else:
        sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1%} {suffix}")
    sys.stdout.flush()
    if current == total:
        print()


def setup_logging(config_path: str) -> logging.Logger:
    config = load_config(config_path)
    logging_cfg = config.get("logging", {})

    if not logging_cfg.get("enable_file_logging", True):
        return None

    log_filename = logging_cfg.get("log_filename", "log_MODELING.txt")

    logger = logging.getLogger("ModelingPipeline")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_file = os.path.join(os.getcwd(), log_filename)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.setLevel(logging.WARNING)
    warnings_logger.addHandler(file_handler)

    logger.info("=" * 80)
    logger.info("MODELING PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Configuration file: {config_path}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log file: {log_file}")
    logger.info("=" * 80)
    return logger


def main() -> None:
    config_path = "engine_TFM/config_modeling.yml"
    config = load_config(config_path)
    logger = setup_logging(config_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_pca_lda = os.path.join(base_dir, "df_pca_lda.csv")
    csv_anova = os.path.join(base_dir, "df_anova.csv")
    models_dir = os.path.join(base_dir, "models")
    images_dir = os.path.join(base_dir, "imagenes")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

    # Configuración de pasos dinámicos según modelos habilitados
    models_cfg = config.get("models", {})
    steps = [
        (True, "📥 Cargando datos"),
        (True, "🔢 Preparando WOE"),
        (models_cfg.get("enable_logit", True), "🤖 Entrenando LOGIT"),
        (models_cfg.get("enable_gnb", False), "🧮 Entrenando GNB"),
        (models_cfg.get("enable_svm_linear", False), "🧷 SVM Lineal"),
        (models_cfg.get("enable_svm_linear_smart", False), "🧷 SVM Lineal (smart)"),
        (models_cfg.get("enable_mlp", False), "🧠 Entrenando MLP"),
        (config.get("execution", {}).get("run_comparator", True), "💾 Guardando resultados"),
    ]
    enabled_steps = [s for enabled, s in steps if enabled]
    total_steps = len(enabled_steps)
    current_step = 0

    print("🔧 INICIANDO MODELING PIPELINE...")

    start_time = time.perf_counter()

    with warnings.catch_warnings(record=True) as pipeline_warnings:
        warnings.simplefilter("always")

        # Paso 1: Carga de datos
        print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
        current_step += 1
        t0 = time.perf_counter()
        logger.info("Leyendo archivos de datos:")
        logger.info(f"  • PCA+LDA: {csv_pca_lda}")
        logger.info(f"  • ANOVA: {csv_anova}")
        with warnings.catch_warnings(record=True) as w_read:
            warnings.simplefilter("always")
            df_pca_lda = pd.read_csv(csv_pca_lda, low_memory=False)
            df_anova = pd.read_csv(csv_anova, low_memory=False)
            if w_read:
                logger.warning("Warnings during data loading:")
                for w in w_read:
                    logger.warning(f"  {w.category.__name__}: {w.message}")
        logger.info(f"Datos cargados: PCA+LDA {df_pca_lda.shape} | ANOVA {df_anova.shape}")
        logger.info(f"[TIMER] Carga de datos: {time.perf_counter()-t0:.2f}s")
        time.sleep(0.1)
        print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 2: Preparación con WOE
        print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
        current_step += 1
        t0 = time.perf_counter()
        all_vars_pca_lda = df_pca_lda.columns.tolist()[:-1]
        all_vars_anova = df_anova.columns.tolist()[:-1]
        cat_cols_pca_lda = [c for c in all_vars_pca_lda if df_pca_lda[c].dtype == "object"]
        cat_cols_anova = [c for c in all_vars_anova if df_anova[c].dtype == "object"]
        num_vars_pca_lda = [c for c in all_vars_pca_lda if c not in cat_cols_pca_lda]
        num_vars_anova = [c for c in all_vars_anova if c not in cat_cols_anova]

        df_model_pca_lda = prepare_model_dataframe(df_pca_lda, all_vars_pca_lda, "flg_target")
        df_model_anova = prepare_model_dataframe(df_anova, all_vars_anova, "flg_target")

        final_vars_pca_lda = num_vars_pca_lda.copy()
        final_vars_anova = num_vars_anova.copy()

        woe_dir = os.path.join(base_dir, "woe_mappings")
        os.makedirs(woe_dir, exist_ok=True)

        if cat_cols_pca_lda:
            woe_pca_lda = WOETransformer(target_col="flg_target")
            df_model_pca_lda = woe_pca_lda.fit_transform(df_model_pca_lda, cat_cols_pca_lda)
            final_vars_pca_lda += [f"{c}_woe" for c in cat_cols_pca_lda]
            woe_pca_lda.final_vars = final_vars_pca_lda
            woe_pca_lda.save_mappings(os.path.join(woe_dir, "woe_mappings_pca_lda.json"))
            df_model_pca_lda.to_csv(os.path.join(base_dir, "df_pca_lda_woe.csv"), index=False)

        if cat_cols_anova:
            woe_anova = WOETransformer(target_col="flg_target")
            df_model_anova = woe_anova.fit_transform(df_model_anova, cat_cols_anova)
            final_vars_anova += [f"{c}_woe" for c in cat_cols_anova]
            woe_anova.final_vars = final_vars_anova
            woe_anova.save_mappings(os.path.join(woe_dir, "woe_mappings_anova.json"))
            df_model_anova.to_csv(os.path.join(base_dir, "df_anova_woe.csv"), index=False)

        logger.info(f"[TIMER] Preparación WOE: {time.perf_counter()-t0:.2f}s")
        time.sleep(0.1)
        print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 3: Entrenamiento LOGIT
        if models_cfg.get("enable_logit", True):
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
            current_step += 1
            t0 = time.perf_counter()
            # Silenciar posibles prints internos durante el entrenamiento
            buf1 = io.StringIO()
            with redirect_stdout(buf1):
                model_pca_lda, metrics_pca_lda, y_pred_pca_lda, y_proba_pca_lda, \
                X_train_pca_lda, X_test_pca_lda, y_train_pca_lda, y_test_pca_lda = \
                    ModelingEngine.fit_logit_predict(
                        df=df_model_pca_lda,
                        num_cols=final_vars_pca_lda,
                        cat_cols=[],
                        target="flg_target",
                        model_params={"solver": "lbfgs"},
                        show_confusion=False,
                        verbose=True,
                        save_confusion_path=os.path.join(images_dir, "logit_pca_lda_confusion.png"),
                    )
            captured1 = buf1.getvalue()
            if captured1:
                for line in captured1.splitlines():
                    line = line.strip()
                    if line:
                        logger.info(line)

            buf2 = io.StringIO()
            with redirect_stdout(buf2):
                model_anova, metrics_anova, y_pred_anova, y_proba_anova, \
                X_train_anova, X_test_anova, y_train_anova, y_test_anova = \
                    ModelingEngine.fit_logit_predict(
                        df=df_model_anova,
                        num_cols=final_vars_anova,
                        cat_cols=[],
                        target="flg_target",
                        model_params={"solver": "lbfgs"},
                        show_confusion=False,
                        verbose=True,
                        save_confusion_path=os.path.join(images_dir, "logit_anova_confusion.png"),
                    )
            captured2 = buf2.getvalue()
            if captured2:
                for line in captured2.splitlines():
                    line = line.strip()
                    if line:
                        logger.info(line)
            ModelingEngine.save_model(model_pca_lda, os.path.join(models_dir, "logit_pca_lda.pkl"))
            ModelingEngine.save_model(model_anova, os.path.join(models_dir, "logit_anova.pkl"))
            logger.info(f"LOGIT | PCA+LDA AUC: {metrics_pca_lda['AUC_test']:.4f}")
            logger.info(f"LOGIT | ANOVA   AUC: {metrics_anova['AUC_test']:.4f}")
            logger.info(f"[TIMER] LOGIT: {time.perf_counter()-t0:.2f}s")
            time.sleep(0.1)
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 4: GNB
        if models_cfg.get("enable_gnb", False):
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
            current_step += 1
            t0 = time.perf_counter()
            buf = io.StringIO()
            with redirect_stdout(buf):
                gnb_pca_lda, gnb_metrics_pca_lda, _, _, _, _, _, _ = ModelingEngine.fit_gaussian_nb_predict(
                    df=df_model_pca_lda,
                    num_cols=final_vars_pca_lda,
                    cat_cols=[],
                    target="flg_target",
                    show_confusion=False,
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "gnb_pca_lda_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            buf = io.StringIO()
            with redirect_stdout(buf):
                gnb_anova, gnb_metrics_anova, _, _, _, _, _, _ = ModelingEngine.fit_gaussian_nb_predict(
                    df=df_model_anova,
                    num_cols=final_vars_anova,
                    cat_cols=[],
                    target="flg_target",
                    show_confusion=False,
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "gnb_anova_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            ModelingEngine.save_model(gnb_pca_lda, os.path.join(models_dir, "gnb_pca_lda.pkl"))
            ModelingEngine.save_model(gnb_anova, os.path.join(models_dir, "gnb_anova.pkl"))
            logger.info(f"GNB | PCA+LDA AUC: {gnb_metrics_pca_lda['AUC_test']:.4f}")
            logger.info(f"GNB | ANOVA   AUC: {gnb_metrics_anova['AUC_test']:.4f}")
            logger.info(f"[TIMER] GNB: {time.perf_counter()-t0:.2f}s")
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 5: SVM Lineal (predict)
        if models_cfg.get("enable_svm_linear", False):
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
            current_step += 1
            t0 = time.perf_counter()
            buf = io.StringIO()
            with redirect_stdout(buf):
                svm_lin_pca_lda, svm_lin_metrics_pca_lda, _, _, _, _, _, _ = ModelingEngine.fit_linear_svm_predict(
                    df=df_model_pca_lda,
                    num_cols=final_vars_pca_lda,
                    cat_cols=[],
                    target="flg_target",
                    show_confusion=False,
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "svm_linear_pca_lda_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            buf = io.StringIO()
            with redirect_stdout(buf):
                svm_lin_anova, svm_lin_metrics_anova, _, _, _, _, _, _ = ModelingEngine.fit_linear_svm_predict(
                    df=df_model_anova,
                    num_cols=final_vars_anova,
                    cat_cols=[],
                    target="flg_target",
                    show_confusion=False,
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "svm_linear_anova_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            # Guardar (para trazabilidad; el comparador usa los smart-search si existen)
            ModelingEngine.save_model(svm_lin_pca_lda, os.path.join(models_dir, "svm_linear_pca_lda.pkl"))
            ModelingEngine.save_model(svm_lin_anova, os.path.join(models_dir, "svm_linear_anova.pkl"))
            logger.info(f"SVM Linear | PCA+LDA AUC: {svm_lin_metrics_pca_lda['AUC_test']:.4f}")
            logger.info(f"SVM Linear | ANOVA   AUC: {svm_lin_metrics_anova['AUC_test']:.4f}")
            logger.info(f"[TIMER] SVM Linear: {time.perf_counter()-t0:.2f}s")
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 6: SVM Lineal (smart search)
        if models_cfg.get("enable_svm_linear_smart", False):
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
            current_step += 1
            t0 = time.perf_counter()
            buf = io.StringIO()
            with redirect_stdout(buf):
                svm_best_pca_lda, best_cfg_pca, svm_best_metrics_pca_lda, _, _, _, _, _, _ = ModelingEngine.fit_linear_svm_smartsearch(
                    df=df_model_pca_lda,
                    num_cols=final_vars_pca_lda,
                    cat_cols=[],
                    target="flg_target",
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "svm_best_pca_lda_confusion.png"),
                    title_suffix="PCA+LDA",
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            buf = io.StringIO()
            with redirect_stdout(buf):
                svm_best_anova, best_cfg_anova, svm_best_metrics_anova, _, _, _, _, _, _ = ModelingEngine.fit_linear_svm_smartsearch(
                    df=df_model_anova,
                    num_cols=final_vars_anova,
                    cat_cols=[],
                    target="flg_target",
                    verbose=True,
                    save_confusion_path=os.path.join(images_dir, "svm_best_anova_confusion.png"),
                    title_suffix="ANOVA",
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            # Guardar con nombres esperados por el comparador
            ModelingEngine.save_model(svm_best_pca_lda, os.path.join(models_dir, "svm_best_pca_lda.pkl"))
            ModelingEngine.save_model(svm_best_anova, os.path.join(models_dir, "svm_best_anova.pkl"))
            logger.info(f"SVM best | PCA+LDA AUC: {svm_best_metrics_pca_lda['AUC_test']:.4f} | params: {best_cfg_pca}")
            logger.info(f"SVM best | ANOVA   AUC: {svm_best_metrics_anova['AUC_test']:.4f} | params: {best_cfg_anova}")
            logger.info(f"[TIMER] SVM Linear smart: {time.perf_counter()-t0:.2f}s")
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso 7: MLP
        if models_cfg.get("enable_mlp", False):
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
            current_step += 1
            t0 = time.perf_counter()
            buf = io.StringIO()
            with redirect_stdout(buf):
                mlp_pca_lda, mlp_metrics_pca_lda, _, _, _, _, _, _ = ModelingEngine.fit_mlp_predict(
                    df=df_model_pca_lda,
                    num_cols=final_vars_pca_lda,
                    cat_cols=[],
                    target="flg_target",
                    verbose=True,
                    show_confusion=False,
                    save_confusion_path=os.path.join(images_dir, "mlp_pca_lda_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            buf = io.StringIO()
            with redirect_stdout(buf):
                mlp_anova, mlp_metrics_anova, _, _, _, _, _, _ = ModelingEngine.fit_mlp_predict(
                    df=df_model_anova,
                    num_cols=final_vars_anova,
                    cat_cols=[],
                    target="flg_target",
                    verbose=True,
                    show_confusion=False,
                    save_confusion_path=os.path.join(images_dir, "mlp_anova_confusion.png"),
                )
            for line in buf.getvalue().splitlines():
                line = line.strip()
                if line:
                    logger.info(line)
            ModelingEngine.save_model(mlp_pca_lda, os.path.join(models_dir, "mlp_pca_lda.pkl"))
            ModelingEngine.save_model(mlp_anova, os.path.join(models_dir, "mlp_anova.pkl"))
            logger.info(f"MLP | PCA+LDA AUC: {mlp_metrics_pca_lda['AUC_test']:.4f}")
            logger.info(f"MLP | ANOVA   AUC: {mlp_metrics_anova['AUC_test']:.4f}")
            logger.info(f"[TIMER] MLP: {time.perf_counter()-t0:.2f}s")
            print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step-1], suffix="Completado")

        # Paso final: Comparativa y guardado
        print_section_progress(current_step, total_steps, section_name=enabled_steps[current_step], suffix="")
        current_step += 1
        t0 = time.perf_counter()
        comparator = ModelComparator(base_dir)
        df_cmp = comparator.compare()
        # Guardado tolerante: usar método save si existe; si no, guardar manualmente
        out_csv = None
        try:
            out_csv = comparator.save(df_cmp)  # type: ignore[attr-defined]
        except Exception:
            reports_dir = os.path.join(base_dir, "reports")
            os.makedirs(reports_dir, exist_ok=True)
            out_csv = os.path.join(reports_dir, "model_comparison.csv")
            df_cmp.to_csv(out_csv, index=False)
        logger.info("RESULTADOS FINALES - COMPARATIVA DE MODELOS:")
        logger.info("=" * 60)
        for _, row in df_cmp.iterrows():
            logger.info(f"{row['Modelo']}: AUC={row['AUC']:.4f} | PR-AUC={row['PR_AUC(AP)']:.4f} | Brier={row['Brier']:.4f}")
        logger.info(f"Tabla comparativa guardada en: {out_csv}")
        logger.info(f"[TIMER] Comparativa: {time.perf_counter()-t0:.2f}s")
        time.sleep(0.1)
        print_section_progress(current_step, total_steps, section_name="💾 Guardando resultados", suffix="Completado")

        if pipeline_warnings:
            logger.warning("\n⚠️  WARNINGS CAPTURADOS DURANTE LA EJECUCIÓN:")
            for w in pipeline_warnings:
                logger.warning(f"  {w.category.__name__}: {w.message}")

    logger.info("MODELING PIPELINE COMPLETED SUCCESSFULLY")
    logger.info(f"[TIMER] Tiempo total: {time.perf_counter()-start_time:.2f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error en la ejecución:", repr(exc))
        sys.exit(1)
