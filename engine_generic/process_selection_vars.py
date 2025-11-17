import os
import logging
from datetime import datetime

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from feature_pro import (
    read_dataset,
    summarize_missing,
    detect_column_types,
    impute_missing,
    winsorize_by_percentile,
    coefficient_of_variation,
    compute_pca,
    pca_lda_importance,
    select_by_correlation_graph,
    compute_psi,
    count_unique_categorical,
    categorical_cumulative_frequency,
    calculate_woe_iv,
)
import networkx as nx


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger("feature_pro_demo")
    logger.setLevel(logging.INFO)
    # Clear existing handlers to avoid duplicates
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    return logger


def main():
    # Config paths
    data_path = os.path.join(os.path.dirname(__file__), 'borrar_prueba_engine.csv')
    log_path = os.path.join(os.path.dirname(__file__), 'process_borrar_prueba_engine.log')
    graphs_dir = os.path.join(os.path.dirname(__file__), 'selection_graphs')
    os.makedirs(graphs_dir, exist_ok=True)
    out_cv_plot = os.path.join(graphs_dir, 'cv_kde.png')

    logger = setup_logger(log_path)
    logger.info("== INICIO DEL PROCESO ==")

    # 1) Leer dataset
    print("== Leyendo dataset ==")
    logger.info("Leyendo dataset...")
    df = read_dataset(data_path, fmt='csv', logger=logger)
    logger.info(f"Leído con shape: {df.shape}")
    # Copia de base original para construir la base final (sin transformaciones)
    df_raw = df.copy()

    # 2) Resumen de missings y filtrado (< 60%)
    print("== Resumen de missings y filtrado < 60% ==")
    logger.info("Calculando resumen de missings...")
    miss_table = summarize_missing(df)
    logger.info("Resumen de missings (primeros 20):\n" + miss_table.head(20).to_string(index=False))
    keep_cols = miss_table.loc[miss_table['pct_missing'] < 0.60, 'column'].tolist()
    df = df[keep_cols].copy()
    logger.info(f"Columnas retenidas tras filtro <60% missings: {len(keep_cols)}")

    # 2.1) Definir universo de variables objetivo (variable_1 .. variable_22) y separar por tipo
    print("== Definiendo universo de variables objetivo y separando por tipo ==")
    target_vars = [f"variable_{i}" for i in range(1, 23)]
    target_vars = [c for c in target_vars if c in df.columns]
    logger.info(f"Universo de variables objetivo ({len(target_vars)}): {target_vars}")

    num_detected, cat_detected = detect_column_types(df, columns=target_vars)
    # Variables numéricas de trabajo (seguirán el pipeline numérico existente)
    working_numeric_vars = list(num_detected)
    # Variables categóricas de trabajo (guardar para pasos posteriores)
    working_categorical_vars = list(cat_detected)

    print(f"== Variables categóricas detectadas en universo objetivo: {len(working_categorical_vars)} ==")
    logger.info(f"Variables categóricas (objetivo) ({len(working_categorical_vars)}): {working_categorical_vars}")
    logger.info(f"Variables numéricas (objetivo) ({len(working_numeric_vars)}): {working_numeric_vars}")

    # 3) Variables numéricas de trabajo (desde el universo objetivo)
    print("== Seleccionando variables numéricas objetivo ==")
    num_vars = working_numeric_vars
    if not num_vars:
        logger.warning("No se encontraron variables numéricas en el universo objetivo.")
        print("Advertencia: No se encontraron variables numéricas. Proceso finalizará.")
        return
    logger.info(f"Variables numéricas seleccionadas: {num_vars}")

    # 4) Crear columna de cortes de percentiles (20 en 20) para groupby
    print("== Creando cortes de percentiles (20 en 20) para imputación por grupo ==")
    base_col = 'variable_1' if 'variable_1' in df.columns else num_vars[0]
    logger.info(f"Base para cortes de percentiles: {base_col}")
    try:
        bins = pd.qcut(df[base_col], q=5, duplicates='drop')
    except Exception as e:
        logger.warning(f"No fue posible crear qcut por '{base_col}': {e}. Se imputará sin groupby.")
        bins = None

    if bins is not None:
        df['percentile_bin_20'] = bins
        group_col = 'percentile_bin_20'
        logger.info("Columna 'percentile_bin_20' creada para imputación por grupo.")
    else:
        group_col = None

    # 5) Imputación de medianas por grupo (si hay bin), si no, global
    print("== Imputando medianas ==")
    logger.info("Aplicando imputación de medianas...")
    df[num_vars] = impute_missing(df, columns=num_vars, strategy='median', groupby=group_col)[num_vars]
    logger.info("Imputación finalizada.")

    # 6) Winsorización percentil [1, 99]
    print("== Winsorización por percentiles [1, 99] ==")
    logger.info("Aplicando winsorización por percentiles [1, 99]...")
    wins_df, perc_df = winsorize_by_percentile(df, columns=num_vars, lower=0.01, upper=0.99, return_percentiles=True)
    df[num_vars] = wins_df[num_vars]
    logger.info("Percentiles usados por variable (primeros 20):\n" + perc_df.head(20).to_string(index=False))
    logger.info("Winsorización finalizada.")

    # 7) Coeficiente de variación y tabla de salida
    print("== Calculando coeficiente de variación ==")
    logger.info("Calculando coeficiente de variación...")
    cv_table = coefficient_of_variation(df, columns=num_vars)
    logger.info("Tabla de CV (primeros 20):\n" + cv_table.head(20).to_string(index=False))

    # 8) KDE plot del CV (antes del filtrado de CV)
    print("== Generando KDE plot de CV ==")
    logger.info("Generando KDE plot de CV...")
    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=cv_table, x='cv', fill=True, color='#4C78A8')
    plt.title('Distribución del Coeficiente de Variación (CV)')
    plt.xlabel('CV')
    plt.ylabel('Densidad')
    plt.tight_layout()
    plt.savefig(out_cv_plot, dpi=150)
    plt.close()
    logger.info(f"KDE plot guardado en: {out_cv_plot}")

    # 9) Filtrar variables por CV (0.2 < CV < 4)
    print("== Filtrando por CV (0.2 < CV < 4) ==")
    logger.info("Filtrando variables por CV en (0.2, 4)...")
    cv_keep = cv_table[(cv_table['cv'] > 0.2) & (cv_table['cv'] < 4)]
    kept_vars = cv_keep['column'].tolist()
    logger.info(f"Variables retenidas por CV ({len(kept_vars)}): {kept_vars}")
    if not kept_vars:
        logger.warning("No quedaron variables tras filtrar por CV. Proceso finaliza antes de PCA.")
        print("Advertencia: No quedaron variables tras filtro de CV; se detiene antes de PCA.")
        return

    # 10) PCA sobre variables retenidas, varianza acumulada 90%
    print("== Ejecutando PCA (varianza 90%) ==")
    logger.info("Ejecutando PCA con threshold de varianza acumulada 90%...")
    pca_out = compute_pca(
        df=df,
        columns=kept_vars,
        n_components=None,
        variance_threshold=0.90,
        scale=True,
        plot=True,
        return_fig=True,
        random_state=None,
    )
    chosen_n = pca_out['n_components']
    cum_var = pca_out['cumulative_variance']
    var90 = float(cum_var[chosen_n - 1]) if chosen_n >= 1 else float('nan')
    logger.info(f"Componentes seleccionados: n_components={chosen_n}; varianza acumulada={var90:.4f}")

    # Guardar gráfico PCA varianza acumulada
    out_pca_plot = os.path.join(graphs_dir, 'pca_variance.png')
    print("== Generando gráfico de varianza acumulada de PCA ==")
    if pca_out['figure'] is not None:
        pca_out['figure'].savefig(out_pca_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Gráfico de varianza acumulada guardado en: {out_pca_plot}")

    # 11) PCA+LDA con n_components seleccionado; ranking top-20 y gráfico
    print("== Ejecutando PCA+LDA (con n recomendado por 90% varianza) ==")
    logger.info("Ejecutando PCA+LDA usando n_components recomendado (90% varianza)...")
    if 'flag_vta' not in df.columns:
        logger.error("La columna 'flag_vta' no existe en el DataFrame. No se puede ejecutar LDA.")
        print("Advertencia: No existe 'flag_vta'; se detiene antes de PCA+LDA.")
        return
    # Si n_components == número total de variables, usar n-1 para evitar problemas numéricos
    n_for_lda = chosen_n
    if n_for_lda >= len(kept_vars):
        n_for_lda = max(1, len(kept_vars) - 1)
        logger.info(f"n_components ajustado para LDA: {chosen_n} -> {n_for_lda} (porque igualaba el número de variables)")
    lda_out = pca_lda_importance(
        df=df,
        feature_columns=kept_vars,
        target_column='flag_vta',
        n_pca_components=n_for_lda,
        variance_threshold=None,
        scale=True,
        plot=True,
        return_fig=True,
        random_state=None,
        top_n=20,
    )
    ranking = lda_out['ranking']
    logger.info("Top 20 coeficientes PCA+LDA:\n" + ranking.to_string(index=False))

    out_lda_plot = os.path.join(graphs_dir, 'pca_lda_importance.png')
    print("== Generando gráfico de coeficientes PCA+LDA ==")
    if lda_out['figure'] is not None:
        lda_out['figure'].savefig(out_lda_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Gráfico de coeficientes PCA+LDA guardado en: {out_lda_plot}")

    # 12) Filtrado por importancia PCA+LDA (> 0.05)
    print("== Filtrando por importancia PCA+LDA (> 0.05) ==")
    logger.info("Filtrando variables por importancia PCA+LDA > 0.05...")
    # Para asegurar que evaluamos todas, volvemos a pedir ranking completo (sin top_n)
    lda_all = pca_lda_importance(
        df=df,
        feature_columns=kept_vars,
        target_column='flag_vta',
        n_pca_components=n_for_lda,
        variance_threshold=None,
        scale=True,
        plot=False,
        return_fig=False,
        random_state=None,
        top_n=None,
    )
    ranking_all = lda_all['ranking']
    lda_keep = ranking_all[ranking_all['importance'] > 0.05]['variable'].tolist()
    logger.info(f"Variables retenidas por PCA+LDA > 0.05 ({len(lda_keep)}): {lda_keep}")
    if not lda_keep:
        logger.warning("No quedaron variables tras filtrar por importancia PCA+LDA. Proceso se detiene antes de correlación.")
        print("Advertencia: No quedaron variables tras filtro PCA+LDA; se detiene antes de correlación.")
        return

    # 13) Selección por correlación con grafos (incluyendo 'tasa' y priorizándola)
    print("== Ejecutando selección por correlación (grafos) ==")
    logger.info("Iniciando selección por correlación (grafos)...")
    corr_vars = list(lda_keep)
    if 'tasa' in df.columns and 'tasa' not in corr_vars:
        corr_vars.append('tasa')
        logger.info("Variable 'tasa' agregada para análisis de correlación y priorizada.")

    sel_out = select_by_correlation_graph(
        df=df,
        columns=corr_vars,
        target_column='flag_vta',
        method='spearman',
        target_method='pointbiserial',
        threshold=0.6,
        prioritize=['tasa'] if 'tasa' in corr_vars else None,
        plot_matrix=True,
        return_matrix_fig=True,
        return_graph=True,
        logger=logger,
    )

    # Guardar matriz de correlación
    print("== Generando matriz de correlación ==")
    corr_plot_path = os.path.join(graphs_dir, 'corr_matrix.png')
    if sel_out['matrix_figure'] is not None:
        sel_out['matrix_figure'].savefig(corr_plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Matriz de correlación guardada en: {corr_plot_path}")

    # Reportar grupos formados (componentes conexos)
    if sel_out['graph'] is not None:
        comps = list(nx.connected_components(sel_out['graph']))
        logger.info(f"Grupos de variables correlacionadas ({len(comps)} grupos):")
        for idx, comp in enumerate(comps, start=1):
            logger.info(f"Grupo {idx}: {sorted(list(comp))}")

    selected_final = sel_out['selected']
    logger.info(f"Variables seleccionadas finales ({len(selected_final)}): {selected_final}")
    print("== Selección por correlación completada ==")

    # PSI por trimestre (solo numéricas seleccionadas finales)
    print("== Calculando PSI por trimestre ==")
    if 'codmes' not in df.columns:
        logger.warning("No existe la columna 'codmes' para calcular PSI por trimestre. Se omite esta sección.")
    else:
        logger.info("Generando columna trimestral a partir de 'codmes' (YYYYMM -> YYYYQn)...")
        codmes_str = df['codmes'].astype(str).str.zfill(6)
        year = codmes_str.str.slice(0, 4)
        month = pd.to_numeric(codmes_str.str.slice(4, 6), errors='coerce')
        quarter_num = ((month - 1) // 3 + 1).astype('Int64')
        quarter_label = year + "Q" + quarter_num.astype(str)
        df['quarter_period'] = quarter_label
        logger.info("Columna 'quarter_period' creada.")

        logger.info("Calculando PSI por variable (numéricas finales) tomando el primer trimestre como referencia...")
        logger.info(f"Variables consideradas para PSI (post-selección final) ({len(selected_final)}): {selected_final}")
        psi_summary = compute_psi(
            df=df,
            temporal_column='quarter_period',
            variables=selected_final,     # usar solo variables numéricas finales seleccionadas
            num_bins=10,
            bin_strategy='quantile',
            reference_period=None,       # primer periodo ordenado
            return_detail=False,
            logger=logger,
        )
        logger.info("PSI por variable (trimestre):\n" + psi_summary.to_string(index=False))

        # Filtro por estabilidad: PSI < 0.10 (10%)
        print("== Filtrando variables numéricas por PSI < 10% ==")
        psi_threshold = 10
        psi_keep = psi_summary[psi_summary['psi'] < psi_threshold]['variable'].tolist()
        psi_removed = [v for v in selected_final if v not in psi_keep]
        logger.info(f"Variables retenidas por PSI < {psi_threshold:.2f} ({len(psi_keep)}): {psi_keep}")
        if psi_removed:
            logger.info(f"Variables removidas por PSI >= {psi_threshold:.2f} ({len(psi_removed)}): {psi_removed}")
        # Actualizar conjunto numérico final
        selected_final = psi_keep

    # 14) Categóricas: únicos < 20
    print("== Seleccionando categóricas por # únicos < 20 ==")
    logger.info("Seleccionando variables categóricas con menos de 20 valores únicos...")
    if working_categorical_vars:
        uniq_df = count_unique_categorical(df, columns=working_categorical_vars, include_na=False, logger=logger)
        cat_less_20 = uniq_df[uniq_df['n_unique'] < 20]['variable'].tolist()
        logger.info(f"Categóricas con <20 únicos ({len(cat_less_20)}): {cat_less_20}")
    else:
        cat_less_20 = []
        logger.info("No hay variables categóricas identificadas en el universo objetivo.")

    # 15) Categóricas: filtro por dominancia (umbral 95%)
    print("== Filtrando categóricas por dominancia (umbral 95%) ==")
    logger.info("Eliminando variables categóricas dominantes con un valor único >= 95%...")
    if cat_less_20:
        dom_out = categorical_cumulative_frequency(
            df=df,
            columns=cat_less_20,
            threshold=0.95,
            include_na=False,
            return_distributions=False,
            logger=logger,
        )
        cat_kept = dom_out['kept_columns']
        cat_removed_dom = dom_out['removed_columns']
        logger.info(f"Categóricas removidas por dominancia ({len(cat_removed_dom)}): {cat_removed_dom}")
        logger.info(f"Categóricas retenidas tras dominancia ({len(cat_kept)}): {cat_kept}")
    else:
        cat_kept = []

    # 16) Categóricas: IV y selección por IV > 0.1
    print("== Calculando IV de categóricas ==")
    if 'flag_vta' not in df.columns:
        logger.warning("No existe 'flag_vta'; se omite el cálculo de IV para categóricas.")
        cat_iv_keep = []
    else:
        if cat_kept:
            iv_out = calculate_woe_iv(
                df=df,
                target='flag_vta',
                columns=cat_kept,
                include_na=True,
                bin_numeric=False,
                logger=logger,
            )
            iv_summary = iv_out['summary']
            logger.info("Resumen IV de categóricas:\n" + iv_summary.to_string(index=False))
            cat_iv_keep = iv_summary[iv_summary['iv'] > 0.02]['variable'].tolist()
            logger.info(f"Categóricas seleccionadas por IV > 0.1 ({len(cat_iv_keep)}): {cat_iv_keep}")
        else:
            cat_iv_keep = []
            logger.info("No hay categóricas para evaluar IV tras filtros previos.")

    # 17) Construir base final (desde la base original), solo variables seleccionadas + target
    print("== Construyendo base final de variables seleccionadas ==")
    final_numeric = selected_final if 'selected_final' in locals() else []
    final_categorical = cat_iv_keep if 'cat_iv_keep' in locals() else []
    final_vars_all = list(dict.fromkeys(final_numeric + final_categorical))
    final_vars_all = [c for c in final_vars_all if c in df_raw.columns]
    target_col = 'flag_vta' if 'flag_vta' in df_raw.columns else None
    if target_col is None:
        logger.warning("La columna objetivo 'flag_vta' no existe en la base original. La base final no incluirá target.")
        final_cols = final_vars_all
    else:
        final_cols = final_vars_all + [target_col]

    if final_cols:
        final_df = df_raw[final_cols].copy()
        out_final_path = os.path.join(os.path.dirname(__file__), 'final_selected_variables.csv')
        final_df.to_csv(out_final_path, index=False)
        logger.info(f"Base final construida desde la base original. Shape: {final_df.shape}. Guardada en: {out_final_path}")
        logger.info(f"Variables finales incluidas ({len(final_vars_all)}) + target: {final_cols}")
    else:
        logger.warning("No hay variables finales para construir la base final.")

    logger.info("== FIN DEL PROCESO ==")
    print("== Proceso completado. Ver log y gráfico generado. ==")


if __name__ == "__main__":
    main()


