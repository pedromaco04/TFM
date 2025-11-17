import os
import logging

import pandas as pd
import matplotlib.pyplot as plt

from feature_pro import (
    read_dataset,
    summarize_missing,
    detect_column_types,
    impute_missing,
    build_regression_tree_segments,
    compute_tree_feature_importance,
    fit_numeric_preprocessor,
    save_segmentation_pipeline,
)


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("feature_pro_segmentation")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def main():
    # Paths
    data_path = os.path.join(os.path.dirname(__file__), 'borrar_prueba_engine.csv')
    log_path = os.path.join(os.path.dirname(__file__), 'process_segmentation.log')
    graphs_dir = os.path.join(os.path.dirname(__file__), 'segmentation_graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    out_tree_plot = os.path.join(graphs_dir, 'regression_tree.png')

    logger = setup_logger(log_path)
    logger.info("== INICIO SEGMENTATION ==")

    # 1) Leer dataset
    print("== Leyendo dataset ==")
    logger.info("Leyendo dataset de entrada...")
    df = read_dataset(data_path, fmt='csv', logger=logger)
    logger.info(f"Dataset shape: {df.shape}")

    # 2) Eliminar variables con >60% nulos
    print("== Eliminando variables con > 60% nulos ==")
    miss = summarize_missing(df)
    logger.info("Resumen de nulos (top 20):\n" + miss.head(20).to_string(index=False))
    keep_cols = miss.loc[miss['pct_missing'] <= 0.60, 'column'].tolist()
    removed_cols = [c for c in df.columns if c not in keep_cols]
    logger.info(f"Variables removidas por nulos >60% ({len(removed_cols)}): {removed_cols}")
    df = df[keep_cols].copy()
    logger.info(f"Shape tras filtro de nulos: {df.shape}")

    # 3) Trabajar solo con numéricas
    print("== Preparando variables numéricas ==")
    num_cols, cat_cols = detect_column_types(df)
    logger.info(f"Numéricas detectadas ({len(num_cols)}): {num_cols}")
    logger.info(f"Categóricas detectadas ({len(cat_cols)}): {cat_cols}")
    if not num_cols:
        logger.error("No se detectaron variables numéricas. Abortando.")
        return

    # 4) Imputar medianas por grupos de 'segmento' (si existe), solo para numéricas
    print("== Imputando medianas por 'segmento' ==")
    group_col = 'segmento' if 'segmento' in df.columns else None
    if group_col is None:
        logger.warning("Columna 'segmento' no encontrada. Se imputará sin agrupación.")
    df[num_cols] = impute_missing(df, columns=num_cols, strategy='median', groupby=group_col)[num_cols]
    logger.info("Imputación por medianas completada.")

    # 5) Árbol de regresión para segmentación (target='tasa')
    print("== Entrenando árbol de regresión para segmentación (target='tasa') ==")
    if 'tasa' not in df.columns:
        logger.error("Columna objetivo 'tasa' no encontrada en el dataset. Abortando.")
        return

    # Features iniciales: unas cuantas variables numéricas (priorizando 'variable_*' hasta 20)
    candidate_vars = [c for c in num_cols if c.startswith('variable_') and c != 'tasa']
    if not candidate_vars:
        candidate_vars = [c for c in num_cols if c != 'tasa']
    features = candidate_vars[:20] if len(candidate_vars) > 20 else candidate_vars
    if not features:
        logger.error("No hay variables numéricas (excluyendo 'tasa') para entrenar el árbol.")
        return
    logger.info(f"Features iniciales para el árbol ({len(features)}): {features}")

    seg_out = build_regression_tree_segments(
        df=df,
        features=features,
        target='tasa',
        max_depth=4,
        min_samples_leaf=1000,
        random_state=42,
        plot=True,
        return_fig=True,
        merge_leaves=None,  # se puede pasar un dict {leaf_id: segment_id} para fusionar
    )
    logger.info("Árbol de regresión entrenado.")
    logger.info("Resumen por segmento (ordenado por media de 'tasa'):\n" + seg_out['summary'].to_string(index=False))

    if seg_out.get('figure') is not None:
        seg_out['figure'].savefig(out_tree_plot, dpi=150, bbox_inches='tight')
        plt.close(seg_out['figure'])
        logger.info(f"Gráfico del árbol guardado en: {out_tree_plot}")

    # 6) Importancia de variables del árbol y re-entrenar con finales
    print("== Calculando importancias de variables del árbol ==")
    imp_df = compute_tree_feature_importance(seg_out['model'], features, normalize=True)
    logger.info("Importancias de variables (ordenadas):\n" + imp_df.to_string(index=False))
    # Guardar gráfico de importancias
    print("== Guardando gráfico de importancias ==")
    try:
        import matplotlib.pyplot as plt
        fig_imp, ax_imp = plt.subplots(figsize=(8, max(3, len(imp_df) * 0.25)))
        ax_imp.barh(imp_df['feature'][::-1], imp_df['importance'][::-1], color='#4C78A8')
        ax_imp.set_title("Importancia de variables (árbol inicial)")
        ax_imp.set_xlabel("Importancia (normalizada)")
        ax_imp.set_ylabel("Variable")
        fig_imp.tight_layout()
        out_imp_plot = os.path.join(graphs_dir, 'feature_importances.png')
        fig_imp.savefig(out_imp_plot, dpi=150, bbox_inches='tight')
        plt.close(fig_imp)
        logger.info(f"Gráfico de importancias guardado en: {out_imp_plot}")
    except Exception as e:
        logger.warning(f"No se pudo guardar el gráfico de importancias: {e}")

    # seleccionar variables finales: todas con importancia > 0 (si ninguna, tomar top 5)
    final_feats = imp_df.loc[imp_df['importance'] > 0, 'feature'].tolist()
    if not final_feats:
        final_feats = imp_df['feature'].head(5).tolist()
    logger.info(f"Variables finales seleccionadas para reentrenar el árbol ({len(final_feats)}): {final_feats}")

    print("== Re-entrenando árbol con variables finales ==")
    out_tree_plot_final = os.path.join(graphs_dir, 'regression_tree_final.png')
    seg_out_final = build_regression_tree_segments(
        df=df,
        features=final_feats,
        target='tasa',
        max_depth=4,
        min_samples_leaf=1000,
        random_state=42,
        plot=True,
        return_fig=True,
        merge_leaves=None,
    )
    logger.info("Árbol final entrenado (con variables filtradas por importancia).")
    logger.info("Resumen por segmento (final):\n" + seg_out_final['summary'].to_string(index=False))
    if seg_out_final.get('figure') is not None:
        seg_out_final['figure'].savefig(out_tree_plot_final, dpi=150, bbox_inches='tight')
        plt.close(seg_out_final['figure'])
        logger.info(f"Gráfico del árbol final guardado en: {out_tree_plot_final}")

    # Guardar asignaciones finales
    assignments_final = pd.DataFrame({
        'segment_id': seg_out_final['segment_id']
    })
    out_assign_final = os.path.join(os.path.dirname(__file__), 'segments_assignment_final.csv')
    assignments_final.to_csv(out_assign_final, index=True, index_label='row_id')
    logger.info(f"Asignación de segmentos (final) guardada en: {out_assign_final}")

    # 7) Guardar pipeline (imputación + modelo final)
    print("== Guardando pipeline de segmentación ==")
    preprocessor = fit_numeric_preprocessor(
        df=df,
        numeric_columns=final_feats,
        impute_strategy='median',
        groupby=('segmento' if 'segmento' in df.columns else None),
        winsorize=False,
        lower=0.01,
        upper=0.99,
    )
    pipeline_path = os.path.join(os.path.dirname(__file__), 'segmentation_pipeline.pkl')
    save_segmentation_pipeline(
        path=pipeline_path,
        preprocessor=preprocessor,
        model=seg_out_final['model'],
        features=final_feats,
        merge_leaves=None,
    )
    logger.info(f"Pipeline de segmentación guardado en: {pipeline_path}")

    # Guardar asignaciones de segmento (opcional)
    assignments = pd.DataFrame({
        'segment_id': seg_out['segment_id']
    })
    out_assign = os.path.join(os.path.dirname(__file__), 'segments_assignment.csv')
    assignments.to_csv(out_assign, index=True, index_label='row_id')
    logger.info(f"Asignación de segmentos guardada en: {out_assign}")

    logger.info("== FIN SEGMENTATION ==")
    print("== Proceso de Segmentación completado. Ver log y gráficos generados. ==")


if __name__ == "__main__":
    main()


