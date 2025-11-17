import os
import logging

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from feature_pro import (
    read_dataset,
    summarize_missing,
    detect_column_types,
    impute_missing,
    build_regression_tree_segments,
    compute_tree_feature_importance,
    fit_numeric_preprocessor,
    save_segmentation_pipeline,
    apply_segmentation_pipeline,
    load_segmentation_pipeline,
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
    data_path = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/01.Universo/universo_bot_cotiza_vars_comport_v4'
    log_path = os.path.join(os.path.dirname(__file__), 'process_segmentation.log')
    graphs_dir = os.path.join(os.path.dirname(__file__), 'segmentation_graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    out_tree_plot = os.path.join(graphs_dir, 'regression_tree.png')

    logger = setup_logger(log_path)
    logger.info("== INICIO SEGMENTATION ==")

    # 1) Leer dataset
    print("== Leyendo dataset ==")
    logger.info("Leyendo dataset de entrada...")
    df = read_dataset(data_path, fmt='parquet', logger=logger)
    logger.info(f"Dataset shape: {df.shape}")

    # 1.1) Eliminar columnas que contengan "_id" o "_type"
    print("== Eliminando columnas *_id, *_type, *cutoff*,  u otras ==")
    id_type_cols = [c for c in df.columns if ('_id' in c) or ('_type' in c) or ('cutoff' in c) or ('codmes' in c)]
    id_type_cols = ['plazo','coddiv','di','spread','plazo_dias','numcnt','flag_vta','plazo_mes'] + id_type_cols
    if id_type_cols:
        df = df.drop(columns=id_type_cols)
        logger.info(f"Columnas removidas por patrón  ({len(id_type_cols)}): {id_type_cols}")
        logger.info(f"Shape tras remover  {df.shape}")
    else:
        logger.info("No se encontraron columnas que coincidan con **_id, *_type, *cutoff* u otras .")

    # 2) Eliminar variables con >60% nulos
    print("== Eliminando variables con > 60% nulos ==")
    miss = summarize_missing(df)
    logger.info("Resumen de nulos (top 20):\n" + miss.head(20).to_string(index=False))
    keep_cols = miss.loc[miss['pct_missing'] <= 0.40, 'column'].tolist()
    removed_cols = [c for c in df.columns if c not in keep_cols]
    logger.info(f"Variables removidas por nulos >60% ({len(removed_cols)}): {removed_cols}")
    df = df[keep_cols].copy()
    logger.info(f"Shape tras filtro de nulos: {df.shape}")

   # 3) Trabajar solo con numéricas (detección robusta incorporada en detect_column_types)
    print("== Preparando variables numéricas ==")
    num_cols, cat_cols = detect_column_types(df)
    logger.info(f"Numéricas detectadas ({len(num_cols)}): {num_cols}")
    logger.info(f"Categóricas detectadas ({len(cat_cols)}): {cat_cols}")
    if not num_cols:
        logger.error("No se detectaron variables numéricas. Abortando.")
        return

    # 3.1) Convertir las columnas numéricas detectadas a float antes de cualquier operación matemática (simple)
    print("== Convirtiendo numéricas a float ==")
    df[num_cols] = df[num_cols].copy().astype(float)
    logger.info(f"Columnas convertidas a float: {num_cols}")

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
    #candidate_vars = [c for c in num_cols if c.startswith('variable_') and c != 'tasa']
    #if not candidate_vars:
    candidate_vars = [c for c in num_cols if c != 'tasa']
    features = candidate_vars #[:20] if len(candidate_vars) > 20 else candidate_vars
    if not features:
        logger.error("No hay variables numéricas (excluyendo 'tasa') para entrenar el árbol.")
        return
    logger.info(f"Features iniciales para el árbol ({len(features)}): {features}")

    seg_out = build_regression_tree_segments(
        df=df,
        features=features,
        target='tasa',
        max_depth=4,
        min_samples_leaf=7000,
        random_state=42,
        plot=True,
        return_fig=True,
        merge_leaves=None,  # se puede pasar un dict {leaf_id: segment_id} para fusionar
    )
    logger.info("Árbol de regresión entrenado.")
    logger.info("Resumen por cluster:\n" + seg_out['summary'].to_string(index=False))
    
    # Mostrar reglas de cada cluster
    logger.info("\n=== REGLAS DE DECISIÓN POR CLUSTER ===")
    rules = seg_out.get('rules', {})
    summary = seg_out['summary']
    for _, row in summary.iterrows():
        cluster_id = int(row['cluster_id'])
        n = int(row['n'])
        mean_tasa = row['mean_target']
        std_tasa = row['std_target']
        rule = rules.get(cluster_id, "Regla no encontrada")
        logger.info(f"\nCluster {cluster_id}:")
        logger.info(f"  - N observaciones: {n}")
        logger.info(f"  - Tasa promedio: {mean_tasa:.4f} (std: {std_tasa:.4f})")
        logger.info(f"  - Regla: {rule}")
    logger.info("=" * 50)

    if seg_out.get('figure') is not None:
        seg_out['figure'].savefig(out_tree_plot, dpi=150, bbox_inches='tight')
        plt.close(seg_out['figure'])
        logger.info(f"Gráfico del árbol guardado en: {out_tree_plot}")


    # 7.1) Releer dataset desde cero y reentrenar árbol solo con selected_features
    print("== Releyendo dataset y reentrenando árbol con features seleccionadas ==")
    # Seleccionar solo variables relevantes del árbol (importancia > 0)
    imp_df = compute_tree_feature_importance(seg_out['model'], features, normalize=True)
    selected_features = imp_df.loc[imp_df['importance'] > 0, 'feature'].tolist()
    if not selected_features:
        selected_features = imp_df['feature'].head(5).tolist()
    logger.info(f"Variables seleccionadas por el árbol ({len(selected_features)}): {selected_features}")

    # Releer dataset en limpio
    df_fresh = read_dataset(data_path, fmt='parquet', logger=logger)
    # Subset mínimo: selected_features + 'segmento' si existe
    work_cols = [c for c in selected_features if c in df_fresh.columns]
    seg_col = 'segmento' if 'segmento' in df_fresh.columns else None
    if seg_col is not None:
        work_cols = work_cols + [seg_col]
    df_work = df_fresh[work_cols].copy()
    # Conversión a float para selected_features
    df_work[selected_features] = df_work[selected_features].copy().astype(float)
    # Imputación (esta imputación es previa al reentrenamiento; no va dentro del pipeline)
    df_work[selected_features] = impute_missing(
        df_work, columns=selected_features, strategy='median', groupby=seg_col
    )[selected_features]
    # Añadir 'tasa' para reentrenar árbol
    if 'tasa' not in df_fresh.columns:
        logger.error("Columna objetivo 'tasa' no encontrada al re-leer el dataset. Abortando.")
        return
    df_train = df_work.copy()
    df_train['tasa'] = df_fresh['tasa']

    # Reentrenar árbol con selected_features
    seg_out_refit = build_regression_tree_segments(
        df=df_train,
        features=selected_features,
        target='tasa',
        max_depth=4,
        min_samples_leaf=7000,
        random_state=42,
        plot=True,
        return_fig=True,
        merge_leaves=None,
    )
    logger.info("Árbol reentrenado con features seleccionadas.")
    logger.info("Resumen (refit) por cluster:\n" + seg_out_refit['summary'].to_string(index=False))
    
    # Mostrar reglas de cada cluster (refit)
    logger.info("\n=== REGLAS DE DECISIÓN POR CLUSTER (REFIT) ===")
    rules_refit = seg_out_refit.get('rules', {})
    summary_refit = seg_out_refit['summary']
    for _, row in summary_refit.iterrows():
        cluster_id = int(row['cluster_id'])
        n = int(row['n'])
        mean_tasa = row['mean_target']
        std_tasa = row['std_target']
        rule = rules_refit.get(cluster_id, "Regla no encontrada")
        logger.info(f"\nCluster {cluster_id}:")
        logger.info(f"  - N observaciones: {n}")
        logger.info(f"  - Tasa promedio: {mean_tasa:.4f} (std: {std_tasa:.4f})")
        logger.info(f"  - Regla: {rule}")
    logger.info("=" * 50)
    
    # Permitir fusionar clusters basado en reglas
    # Ejemplo: para fusionar el cluster 7 en el cluster 6, usar: merge_clusters = {7: 6}
    # Esto significa que todas las observaciones del cluster 7 pasarán al cluster 6
    merge_clusters = None  # Cambiar a dict si se quieren fusionar, ej: {7: 6} fusiona 7 en 6
    # merge_clusters = {7: 6}  # Descomentar y ajustar para fusionar clusters
    
    if merge_clusters:
        # Convertir cluster_id -> leaf_id para merge_leaves
        # Necesitamos mapear cluster_id a leaf_id original
        cluster_to_leaf = dict(zip(summary_refit['cluster_id'], summary_refit['leaf_id']))
        merge_leaves_dict = {}
        for source_cluster, target_cluster in merge_clusters.items():
            # source_cluster se fusiona en target_cluster
            target_leaf = cluster_to_leaf.get(target_cluster)
            source_leaf = cluster_to_leaf.get(source_cluster)
            if target_leaf is not None and source_leaf is not None:
                # Mapear source_leaf -> target_leaf (el leaf_id fuente se mapea al leaf_id destino)
                merge_leaves_dict[source_leaf] = target_leaf
                logger.info(f"Fusionando cluster {source_cluster} (leaf_id={source_leaf}) en cluster {target_cluster} (leaf_id={target_leaf})")
        if merge_leaves_dict:
            logger.info(f"Especificación de fusión: {merge_clusters}")
            logger.info(f"Mapeo de hojas para fusión: {merge_leaves_dict}")
            # Reaplicar segmentación con merge
            leaf_ids_merged = seg_out_refit['leaf_id'].map(lambda lid: merge_leaves_dict.get(lid, lid))
            seg_series_merged = pd.Series(leaf_ids_merged, index=seg_out_refit['leaf_id'].index, name="segment_id")
            # Renumerar después del merge
            unique_segments_merged = seg_series_merged.unique()
            segment_to_cluster_merged = {old_id: new_id for new_id, old_id in enumerate(sorted(unique_segments_merged), start=1)}
            seg_out_refit['segment_id'] = seg_series_merged.map(segment_to_cluster_merged)
            # Actualizar summary con clusters fusionados
            # Necesitamos mantener el leaf_id para las reglas
            df_summary = pd.DataFrame({
                'tasa': df_train['tasa'], 
                'segment_id': seg_out_refit['segment_id'],
                'leaf_id': seg_out_refit['leaf_id']
            })
            summ_merged = (
                df_summary
                .groupby('segment_id')
                .agg(
                    n=('tasa', 'size'), 
                    mean_target=('tasa', 'mean'), 
                    std_target=('tasa', 'std'),
                    leaf_id=('leaf_id', 'first')  # Tomar el primer leaf_id del grupo
                )
                .reset_index()
                .sort_values('segment_id')
                .reset_index(drop=True)
            )
            summ_merged.columns = ['cluster_id', 'n', 'mean_target', 'std_target', 'leaf_id']
            seg_out_refit['summary'] = summ_merged
            logger.info("Clusters fusionados. Nuevo resumen:")
            logger.info(summ_merged.to_string(index=False))
    else:
        merge_leaves_dict = None
    out_tree_plot_selected = os.path.join(graphs_dir, 'regression_tree_selected_features.png')
    if seg_out_refit.get('figure') is not None:
        seg_out_refit['figure'].savefig(out_tree_plot_selected, dpi=150, bbox_inches='tight')
        plt.close(seg_out_refit['figure'])
        logger.info(f"Gráfico del árbol (selected features) guardado en: {out_tree_plot_selected}")

    # 7.2) Guardar pipeline (float + imputación + árbol refit con selected_features)
    print("== Guardando pipeline de segmentación ==")
    preprocessor = fit_numeric_preprocessor(
        df=df_train,
        numeric_columns=selected_features,
        impute_strategy='median',
        groupby=seg_col,
        winsorize=False,
        lower=0.01,
        upper=0.99,
    )
    # Guardar pipeline (float + imputación + árbol)
    pipeline_path = os.path.join(os.path.dirname(__file__), 'MOMA', 'segmentation_pipeline.pkl')
    save_segmentation_pipeline(
        path=pipeline_path,
        preprocessor=preprocessor,
        model=seg_out_refit['model'],
        features=selected_features,
        merge_leaves=merge_leaves_dict,
    )
    logger.info(f"Pipeline de segmentación guardado en: {pipeline_path}")

    # 7.3) Puntuar toda la base desde cero con el pipeline y guardar CSV requerido
    print("== Puntuar clusters y guardar base ==")
    # Releer base para validar el .pkl
    df_apply = read_dataset(data_path, fmt='parquet', logger=logger)
    pipeline_loaded = load_segmentation_pipeline(pipeline_path)
    applied = apply_segmentation_pipeline(
        pipeline=pipeline_loaded,
        df=df_apply,
    )
    df_scored = df_apply.copy()
    df_scored['cluster'] = applied['segment_id']
    out_scored = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/01.Universo/universo_bot_cotiza_vars_comport_v4'
    df_scored.to_csv(out_scored, index=False)
    logger.info(f"Base clusterizada guardada en: {out_scored}")

    # 7.2) Gráficos de distribución de tasa por cluster: Violin y KDE
    print("== Graficando tasa por cluster (violin y KDE) ==")
    try:
        # Violin: eje y = tasa, eje x = cluster
        fig_v, ax_v = plt.subplots(figsize=(10, max(4, df_scored['cluster'].nunique() * 0.5)))
        sns.violinplot(data=df_scored, x='cluster', y='tasa', inner='quartile', cut=0, ax=ax_v)
        ax_v.set_title('Distribución de tasa por cluster (Violin)')
        fig_v.tight_layout()
        out_violin = os.path.join(graphs_dir, 'violin_tasa_by_cluster.png')
        fig_v.savefig(out_violin, dpi=150, bbox_inches='tight')
        plt.close(fig_v)
        logger.info(f"Gráfico violin guardado en: {out_violin}")

        # KDE: hue = cluster, x = tasa
        fig_k, ax_k = plt.subplots(figsize=(10, 5))
        sns.kdeplot(data=df_scored, x='tasa', hue='cluster', common_norm=False, fill=True, alpha=0.25, ax=ax_k)
        ax_k.set_title('Distribución de tasa por cluster (KDE)')
        fig_k.tight_layout()
        out_kde = os.path.join(graphs_dir, 'kde_tasa_by_cluster.png')
        fig_k.savefig(out_kde, dpi=150, bbox_inches='tight')
        plt.close(fig_k)
        logger.info(f"Gráfico KDE guardado en: {out_kde}")
    except Exception as e:
        logger.warning(f"No se pudieron generar los gráficos de distribución: {e}")

    # Guardar asignaciones de segmento (opcional)
    assignments = pd.DataFrame({
        'cluster_id': seg_out_refit['segment_id']  # Usar los clusters del refit final
    })
    out_assign = os.path.join(os.path.dirname(__file__), 'segments_assignment.csv')
    assignments.to_csv(out_assign, index=True, index_label='row_id')
    logger.info(f"Asignación de segmentos guardada en: {out_assign}")

    logger.info("== FIN SEGMENTATION ==")
    print("== Proceso de Segmentación completado. Ver log y gráficos generados. ==")


if __name__ == "__main__":
    main()


