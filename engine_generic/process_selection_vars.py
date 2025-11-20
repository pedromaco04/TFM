import os
import logging
from datetime import datetime
from typing import Optional

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
    compute_correlation_matrix,
    plot_kde_by_hue,
    plot_categorical_target_mean,
)
import networkx as nx

VARS_BY_CLUSTER_OUTPUT = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/02.Feature/vars_finales_x_cluster.csv'
DATA_PATH = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/01.Universo/universo_bot_cotiza_vars_clusterizado.csv'
TARGET_COLUMN = 'flag_vta'
PRIORITY_VAR_CORRELATION = 'spread'

SELECTION_PARAMS = {
    'missing_threshold': 0.60,
    'imputation_group_column': 'segmento',
    'winsor_percentiles': {'lower': 0.01, 'upper': 0.99},
    'cv': {'min': 0.15, 'max': 10.0},
    'pca_variance_threshold': 0.90,
    'lda_importance_threshold': 0.05,
    'numeric_corr_threshold': 0.60,
    'psi': {
        'threshold': 0.10,
        'bins': 10,
        'bin_strategy': 'quantile',
        'priority_extras': ['importe'],
    },
    'categorical_unique_max': 10,
    'categorical_dominance_threshold': 0.95,
    'categorical_iv_threshold': 0.02,
    'categorical_corr_threshold': 0.70,
}

def setup_logger(log_file: str, logger_name: str = "feature_pro_demo") -> logging.Logger:
    logger = logging.getLogger(logger_name)
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


def generate_final_vars_graphs(
    df_final: pd.DataFrame,
    cluster_value: str,
    base_dir: str,
    target_column: str = 'flag_vta',
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Genera gráficos de densidad (kde) para variables numéricas y gráficos de barras
    para variables categóricas de las variables finales seleccionadas.
    
    Args:
        df_final: DataFrame con las variables finales seleccionadas + target
        cluster_value: Valor del cluster (para nombres de archivos y carpetas)
        base_dir: Directorio base para guardar archivos
        target_column: Nombre de la columna objetivo (default: 'flag_vta')
        logger: Logger opcional
    """
    if df_final.empty:
        if logger:
            logger.info("DataFrame final está vacío. No se generarán gráficos de variables finales.")
        return
    
    if target_column not in df_final.columns:
        if logger:
            logger.info(f"Variable objetivo '{target_column}' no encontrada. No se generarán gráficos de variables finales.")
        return
    
    if logger:
        logger.info("== Generando gráficos de variables finales ==")
    
    # Crear directorio para este cluster
    vars_finales_dir = os.path.join(base_dir, 'selection_graphs', 'vars_finales', f'cluster_{cluster_value}')
    os.makedirs(vars_finales_dir, exist_ok=True)
    if logger:
        logger.info(f"Directorio creado: {vars_finales_dir}")
    
    # Obtener variables (excluyendo target y cluster si existe)
    vars_to_plot = [c for c in df_final.columns if c not in [target_column, 'cluster']]
    
    if not vars_to_plot:
        if logger:
            logger.info("No hay variables para graficar (solo target/cluster).")
        return
    
    # Detectar tipos de variables
    num_detected, cat_detected = detect_column_types(df_final, columns=vars_to_plot)
    numeric_vars = [v for v in vars_to_plot if v in num_detected]
    categorical_vars = [v for v in vars_to_plot if v in cat_detected]
    
    if logger:
        logger.info(f"Variables numéricas a graficar: {len(numeric_vars)}")
        logger.info(f"Variables categóricas a graficar: {len(categorical_vars)}")
    
    # Filtrar datos válidos (sin NaN en target)
    df_plot = df_final[[target_column] + vars_to_plot].copy()
    df_plot = df_plot.dropna(subset=[target_column])
    
    if df_plot.empty:
        if logger:
            logger.info("No hay datos válidos (todos tienen NaN en target). No se generarán gráficos.")
        return
    
    # 1. Gráficos de densidad (kde) para variables numéricas usando función genérica
    for var in numeric_vars:
        try:
            var_safe = var.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = os.path.join(vars_finales_dir, f'{var_safe}_kde.png')
            title = f'Distribución de {var} por {target_column} - Cluster {cluster_value}'
            
            plot_kde_by_hue(
                df=df_plot,
                variable=var,
                hue_column=target_column,
                output_path=output_path,
                title=title,
                logger=logger
            )
            
        except Exception as e:
            if logger:
                logger.error(f"Error generando gráfico KDE para '{var}': {str(e)}")
            plt.close('all')
            continue
    
    # 2. Gráficos de barras para variables categóricas usando función genérica
    for var in categorical_vars:
        try:
            var_safe = var.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = os.path.join(vars_finales_dir, f'{var_safe}_bar.png')
            title = f'Promedio de {target_column} por {var} - Cluster {cluster_value}'
            
            plot_categorical_target_mean(
                df=df_plot,
                categorical_var=var,
                target_column=target_column,
                output_path=output_path,
                title=title,
                max_categories=20,
                logger=logger
            )
            
        except Exception as e:
            if logger:
                logger.error(f"Error generando gráfico de barras para '{var}': {str(e)}")
            plt.close('all')
            continue
    
    if logger:
        logger.info(f"== Gráficos de variables finales completados. Guardados en: {vars_finales_dir} ==")


def process_cluster_selection(
    df_cluster: pd.DataFrame,
    cluster_value: str,
    base_dir: str,
    target_column: str = 'flag_vta',
    priority_var_correlation: str = 'spread'
) -> pd.DataFrame:
    """
    Ejecuta el proceso completo de selección de variables para un cluster específico.
    
    Args:
        df_cluster: DataFrame filtrado para un cluster específico
        cluster_value: Valor del cluster (para nombres de archivos)
        base_dir: Directorio base para guardar archivos
        target_column: Nombre de la columna objetivo (default: 'flag_vta')
        priority_var_correlation: Variable a priorizar en selección por correlación (default: 'spread')
        
    Returns:
        DataFrame final con variables seleccionadas + target
    """
    # Crear logger específico para este cluster
    logs_dir = os.path.join(base_dir, 'selection_logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, f'selection_cluster_{cluster_value}.log')
    logger = setup_logger(log_file, logger_name=f"cluster_{cluster_value}")
    
    logger.info(f"== PROCESANDO CLUSTER: {cluster_value} ==")
    print(f"== PROCESANDO CLUSTER: {cluster_value} ==")
    
    # Copia de base original para construir la base final (sin transformaciones)
    df_raw = df_cluster.copy()
    df = df_cluster.copy()
    
    # Guardar la variable objetivo separadamente (no debe participar en selección)
    target_data = None
    if target_column in df.columns:
        target_data = df[[target_column]].copy()
        logger.info(f"Variable objetivo '{target_column}' guardada separadamente (no participará en selección)")
    
    # Configurar rutas específicas del cluster - estructura organizada por tipo de gráfico
    graphs_base_dir = os.path.join(base_dir, 'selection_graphs')
    
    # Crear subdirectorios para cada tipo de gráfico
    cv_kde_dir = os.path.join(graphs_base_dir, 'cv_kde')
    pca_variance_dir = os.path.join(graphs_base_dir, 'pca_variance')
    pca_lda_dir = os.path.join(graphs_base_dir, 'pca_lda_importance')
    corr_matrix_dir = os.path.join(graphs_base_dir, 'corr_matrix')
    iv_categorical_dir = os.path.join(graphs_base_dir, 'iv_categorical')
    corr_categorical_dir = os.path.join(graphs_base_dir, 'corr_categorical')
    
    os.makedirs(cv_kde_dir, exist_ok=True)
    os.makedirs(pca_variance_dir, exist_ok=True)
    os.makedirs(pca_lda_dir, exist_ok=True)
    os.makedirs(corr_matrix_dir, exist_ok=True)
    os.makedirs(iv_categorical_dir, exist_ok=True)
    os.makedirs(corr_categorical_dir, exist_ok=True)
    
    # Rutas de salida para cada gráfico
    out_cv_plot = os.path.join(cv_kde_dir, f'cv_kde_cluster_{cluster_value}.png')
    out_pca_plot = os.path.join(pca_variance_dir, f'pca_variance_cluster_{cluster_value}.png')
    out_lda_plot = os.path.join(pca_lda_dir, f'pca_lda_importance_cluster_{cluster_value}.png')
    out_corr_plot = os.path.join(corr_matrix_dir, f'corr_matrix_cluster_{cluster_value}.png')
    out_iv_plot = os.path.join(iv_categorical_dir, f'iv_categorical_cluster_{cluster_value}.png')
    out_corr_cat_plot = os.path.join(corr_categorical_dir, f'corr_categorical_cluster_{cluster_value}.png')

    # 2) Resumen de missings y filtrado (< 60%) - EXCLUYENDO la variable objetivo
    print("== Resumen de missings y filtrado < 60% ==")
    logger.info("Calculando resumen de missings (excluyendo variable objetivo)...")
    # Excluir target_column del análisis de missings
    cols_for_missing = [c for c in df.columns if c != target_column]
    df_for_missing = df[cols_for_missing].copy()
    miss_table = summarize_missing(df_for_missing)
    logger.info("Resumen de missings (primeros 5):\n" + miss_table.head(5).to_string(index=False))
    missing_threshold = SELECTION_PARAMS['missing_threshold']
    keep_cols = miss_table.loc[miss_table['pct_missing'] < missing_threshold, 'column'].tolist()
    # Agregar target_column de vuelta si existe
    if target_column in df.columns:
        keep_cols = keep_cols + [target_column]
    df = df[keep_cols].copy()
    logger.info(f"Columnas retenidas tras filtro <{missing_threshold*100:.1f}% missings: {len(keep_cols)} (incluye target si existe)")

    # 2.1) Definir universo de variables objetivo y separar por tipo
    print("== Definiendo universo de variables objetivo y separando por tipo ==")
    target_vars = ['ac_pmt_typ_crcdpy_tot_amount_transaccional_maestra',
    'additional_dbcd_bal_tl_amount_transaccional_maestra',
    'additional_dbcd_exp_tl_amount_transaccional_maestra',
    'addls_card_expns_mov_tl_number_transaccional_maestra',
    'addls_cards_icm_mov_tl_number_transaccional_maestra',
    'app_collection_use_company_ind_type_transaccional_maestra',
    'audtiminsert_date_transaccional_maestra',
    'aut_count_productos',
    'available_risk_cons_cr_number_rcc',
    'available_risk_credits_number_rcc',
    'bbva_cons_credits_amount_rcc',
    'bills_payment_cr_total_amount_transaccional_maestra',
    'birth_date_transaccional_maestra',
    'card_pmt_shft_iss_tr_tl_amount_transaccional_maestra',
    'card_pmt_shft_rv_tr_tl_amount_transaccional_maestra',
    'cash_pmt_typ_crcdpy_tot_amount_transaccional_maestra',
    'clct_beauty_services_payments_tl_amount_transaccional_maestra',
    'clct_bitel_entel_reld_pcr_pymt_tl_amount_transaccional_maestra',
    'clct_municipalities_payments_tot_amount_transaccional_maestra',
    'clt_app_co_acc_charge_pymt_total_amount_transaccional_maestra',
    'clt_app_co_checks_bbva_pymt_total_amount_transaccional_maestra',
    'clt_app_co_checks_oth_ent_pymt_tl_amount_transaccional_maestra',
    'clt_app_co_credir_card_pymt_total_amount_transaccional_maestra',
    'clt_app_companies_cash_pymt_total_amount_transaccional_maestra',
    'clt_app_companies_payments_total_amount_transaccional_maestra',
    'clt_app_companies_payments_total_number_transaccional_maestra',
    'coddiv',
    'collect_account_charge_pymt_total_amount_transaccional_maestra',
    'collect_afp_payments_total_amount_transaccional_maestra',
    'collect_bbva_check_payments_total_amount_transaccional_maestra',
    'collect_cash_payments_total_amount_transaccional_maestra',
    'collect_claro_pho_carrier_pymt_tl_amount_transaccional_maestra',
    'collect_clubs_payments_total_amount_transaccional_maestra',
    'collect_companies_payments_total_amount_transaccional_maestra',
    'collect_credit_card_payments_tot_amount_transaccional_maestra',
    'collect_customer_ind_type_transaccional_maestra',
    'collect_health_insrnc_payments_tl_amount_transaccional_maestra',
    'collect_institutes_payments_total_amount_transaccional_maestra',
    'collect_institutions_payments_tot_amount_transaccional_maestra',
    'collect_othent_check_payments_tot_amount_transaccional_maestra',
    'collect_payments_total_amount_transaccional_maestra',
    'collect_payments_total_number_transaccional_maestra',
    'collect_public_svc_payments_total_amount_transaccional_maestra',
    'collect_schools_payments_total_amount_transaccional_maestra',
    'collect_sunat_payments_total_amount_transaccional_maestra',
    'collect_universities_payments_tot_amount_transaccional_maestra',
    'cons_cr_unused_lines_number_rcc',
    'cons_revolving_card_number_rcc',
    'consm_cr_available_risk_amount_rcc',
    'consm_cr_crdtr_entities_number_rcc',
    'consm_cr_unuse_lines_amount_rcc',
    'consumption_line_amount_rcc',
    'cptl_form_charge_total_amount_transaccional_maestra',
    'crcd_bars_expns_total_amount_transaccional_maestra',
    'crcd_bazaar_stor_exp_tl_amount_transaccional_maestra',
    'crcd_clothing_exp_tl_amount_transaccional_maestra',
    'crcd_co_prch_exp_tra_tl_number_transaccional_maestra',
    'crcd_comm_prch_exp_tot_amount_transaccional_maestra',
    'crcd_digital_svc_exp_tl_amount_transaccional_maestra',
    'crcd_dlvry_svc_exp_tl_amount_transaccional_maestra',
    'crcd_entrt_svc_exp_tl_amount_transaccional_maestra',
    'crcd_fin_ent_exp_total_amount_transaccional_maestra',
    'crcd_genrl_svc_exp_tl_amount_transaccional_maestra',
    'crcd_health_svc_exp_tl_amount_transaccional_maestra',
    'crcd_housewares_exp_tl_amount_transaccional_maestra',
    'crcd_insur_svc_exp_tl_amount_transaccional_maestra',
    'crcd_major_svs_exp_tl_amount_transaccional_maestra',
    'crcd_minority_sv_exp_tl_amount_transaccional_maestra',
    'crcd_more_prch_province_name_transaccional_maestra',
    'crcd_mr_prch_first_dstrc_name_transaccional_maestra',
    'crcd_mr_prch_secnd_dstrc_name_transaccional_maestra',
    'crcd_mr_prch_third_dstrc_name_transaccional_maestra',
    'crcd_pblc_svc_expns_tl_amount_transaccional_maestra',
    'crcd_persl_svc_exp_tl_amount_transaccional_maestra',
    'crcd_phone_splys_exp_tl_amount_transaccional_maestra',
    'crcd_store_expns_total_amount_transaccional_maestra',
    'crcd_stream_svc_exp_tl_amount_transaccional_maestra',
    'crcd_svc_station_exp_tl_amount_transaccional_maestra',
    'crcd_tax_svc_expns_tl_amount_transaccional_maestra',
    'crcd_trnsp_psg_exp_tl_amount_transaccional_maestra',
    'credit_card_expns_tot_amount_transaccional_maestra',
    'credit_card_origin_ind_type_transaccional_maestra',
    'credit_card_transactions_number_transaccional_maestra',
    'credit_cards_total_amount_rcc',
    'credit_cd_exp_trans_tl_number_transaccional_maestra',
    'credit_cd_genrl_exp_tl_number_transaccional_maestra',
    'credit_cd_genrl_exp_tot_amount_transaccional_maestra',
    'credit_cd_pmt_trans_tl_number_transaccional_maestra',
    'creditor_entities1_number_rcc',
    'creditor_entities_number_rcc',
    'ctg_ac_avail_risk_tl_amount_rcc',
    'ctsPEN',
    'ctsUSD',
    'cts_pmt_shft_iss_tr_tl_amount_transaccional_maestra',
    'cts_pmt_shft_rv_tr_tl_amount_transaccional_maestra',
    'curncy_purchase_exch_rate_amount_transaccional_maestra',
    'curncy_sale_exch_rate_amount_transaccional_maestra',
    'cust_estimated_income_amount_ingresos',
    'dbcd_bars_expns_total_amount_transaccional_maestra',
    'dbcd_bazaar_stor_exp_tl_amount_transaccional_maestra',
    'dbcd_cash_dpst_icm_tot_amount_transaccional_maestra',
    'dbcd_cash_dpst_icm_tot_number_transaccional_maestra',
    'dbcd_cash_wthdr_exp_tot_amount_transaccional_maestra',
    'dbcd_cash_wthdr_exp_tot_number_transaccional_maestra',
    'dbcd_check_chrg_exp_tl_amount_transaccional_maestra',
    'dbcd_check_cr_icm_total_amount_transaccional_maestra',
    'dbcd_clothing_exp_total_amount_transaccional_maestra',
    'dbcd_co_prch_exp_tra_tl_number_transaccional_maestra',
    'dbcd_comex_expns_total_amount_transaccional_maestra',
    'dbcd_comex_incomes_tot_amount_transaccional_maestra',
    'dbcd_comm_prch_total_amount_transaccional_maestra',
    'dbcd_confirming_exp_tl_amount_transaccional_maestra',
    'dbcd_confirming_incm_tl_amount_transaccional_maestra',
    'dbcd_crcd_autdbt_exp_tl_amount_transaccional_maestra',
    'dbcd_crcd_pmt_exp_tot_amount_transaccional_maestra',
    'dbcd_cts_expenses_total_amount_transaccional_maestra',
    'dbcd_cts_incomes_total_amount_transaccional_maestra',
    'dbcd_digital_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_divd_shr_incm_tl_amount_transaccional_maestra',
    'dbcd_dlvry_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_entrt_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_exprs_agnt_exp_tot_amount_transaccional_maestra',
    'dbcd_exprs_agnt_icm_tot_amount_transaccional_maestra',
    'dbcd_fc_prchsl_exp_tot_amount_transaccional_maestra',
    'dbcd_fc_prchsl_incm_tot_amount_transaccional_maestra',
    'dbcd_fees_exp_total_amount_transaccional_maestra',
    'dbcd_fees_incomes_total_amount_transaccional_maestra',
    'dbcd_flows_income_total_amount_transaccional_maestra',
    'dbcd_genrl_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_health_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_housewares_exp_tl_amount_transaccional_maestra',
    'dbcd_insrc_cr_icm_tot_amount_transaccional_maestra',
    'dbcd_insrc_pymt_exp_tot_amount_transaccional_maestra',
    'dbcd_insur_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_int_charge_exp_tl_amount_transaccional_maestra',
    'dbcd_int_cr_icm_total_amount_transaccional_maestra',
    'dbcd_invc_clt_income_tl_amount_transaccional_maestra',
    'dbcd_invc_othclt_icm_tl_amount_transaccional_maestra',
    'dbcd_invc_tx_pmt_exp_tl_amount_transaccional_maestra',
    'dbcd_invse_pmt_incre_tl_amount_transaccional_maestra',
    'dbcd_itf_charge_exp_tot_amount_transaccional_maestra',
    'dbcd_itf_rturn_icm_tot_amount_transaccional_maestra',
    'dbcd_jdcl_wthld_exp_tot_amount_transaccional_maestra',
    'dbcd_jdcl_wthld_incm_tl_amount_transaccional_maestra',
    'dbcd_leasing_exp_total_amount_transaccional_maestra',
    'dbcd_leasing_incm_total_amount_transaccional_maestra',
    'dbcd_letters_exp_total_amount_transaccional_maestra',
    'dbcd_letters_incm_total_amount_transaccional_maestra',
    'dbcd_loans_pyment_tl_amount_transaccional_maestra',
    'dbcd_major_svs_exp_tl_amount_transaccional_maestra',
    'dbcd_mfund_exp_total_amount_transaccional_maestra',
    'dbcd_mfund_incm_total_amount_transaccional_maestra',
    'dbcd_minority_sv_exp_tl_amount_transaccional_maestra',
    'dbcd_more_prch_province_name_transaccional_maestra',
    'dbcd_more_prch_sec_dstrc_name_transaccional_maestra',
    'dbcd_mr_prch_first_dstrc_name_transaccional_maestra',
    'dbcd_mr_prch_third_dstrc_name_transaccional_maestra',
    'dbcd_oth_charge_exp_tl_amount_transaccional_maestra',
    'dbcd_oth_credits_icm_tl_amount_transaccional_maestra',
    'dbcd_othac_tr_exp_tot_amount_transaccional_maestra',
    'dbcd_othac_tr_exp_tot_number_transaccional_maestra',
    'dbcd_othent_ccpt_exp_tl_amount_transaccional_maestra',
    'dbcd_owac_tr_exp_tot_amount_transaccional_maestra',
    'dbcd_owac_tr_incm_total_amount_transaccional_maestra',
    'dbcd_pblc_svc_expns_tl_amount_transaccional_maestra',
    'dbcd_persl_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_phone_splys_exp_tl_amount_transaccional_maestra',
    'dbcd_plin_tra_exp_total_amount_transaccional_maestra',
    'dbcd_pmt_cdy_shr_exp_tl_amount_transaccional_maestra',
    'dbcd_pnts_earcnl_exp_tl_amount_transaccional_maestra',
    'dbcd_pub_svc_pmt_exp_tl_amount_transaccional_maestra',
    'dbcd_pymt_state_exp_tot_amount_transaccional_maestra',
    'dbcd_splys_pmt_icm_tot_amount_transaccional_maestra',
    'dbcd_splys_pyment_tot_amount_transaccional_maestra',
    'dbcd_store_expns_total_amount_transaccional_maestra',
    'dbcd_stream_svc_exp_tl_amount_transaccional_maestra',
    'dbcd_surety_exp_total_amount_transaccional_maestra',
    'dbcd_svc_invc_exp_tl_amount_transaccional_maestra',
    'dbcd_svc_station_exp_tl_amount_transaccional_maestra',
    'dbcd_tax_svc_expns_tl_amount_transaccional_maestra',
    'dbcd_taxes_payrl_exp_tl_amount_transaccional_maestra',
    'dbcd_taxes_payrl_icm_tl_amount_transaccional_maestra',
    'dbcd_term_dpst_exp_tot_amount_transaccional_maestra',
    'dbcd_term_dpst_icm_tot_amount_transaccional_maestra',
    'dbcd_trans_income_total_amount_transaccional_maestra',
    'dbcd_trans_income_total_number_transaccional_maestra',
    'dbcd_trnsp_psg_exp_tl_amount_transaccional_maestra',
    'dbcd_visanet_pmt_icm_tl_amount_transaccional_maestra',
    'debit_card_exp_trans_tl_number_transaccional_maestra',
    'debit_card_expns_total_amount_transaccional_maestra',
    'debit_card_icm_trans_tl_number_transaccional_maestra',
    'debit_card_income_total_amount_transaccional_maestra',
    'debit_card_loans_icm_tl_amount_transaccional_maestra',
    'debit_card_origin_ind_type_transaccional_maestra',
    'debit_card_trans_total_number_transaccional_maestra',
    'dplazoPEN',
    'dplazoUSD',
    'earcnl_credit_total_amount_transaccional_maestra',
    'edad_sociodemo',
    'ent_othac_iss_imt_tl_amount_transaccional_maestra',
    'ent_othac_rv_imt_tl_amount_transaccional_maestra',
    'ent_owac_iss_imt_tl_amount_transaccional_maestra',
    'ent_owac_rv_imt_tl_amount_transaccional_maestra',
    'ent_shift_iss_tr_total_amount_transaccional_maestra',
    'ent_shift_rv_tr_total_amount_transaccional_maestra',
    'fin_othac_iss_imt_total_amount_transaccional_maestra',
    'fin_othac_rv_imt_total_amount_transaccional_maestra',
    'fin_owac_iss_imt_total_amount_transaccional_maestra',
    'fin_shift_iss_tr_total_amount_transaccional_maestra',
    'fin_shift_rv_tr_total_amount_transaccional_maestra',
    'finance_system_balance_amount_rcc',
    'first_importe',
    'first_sched_cr_cd_tra_number_transaccional_maestra',
    'first_sched_dbt_cd_tra_number_transaccional_maestra',
    'first_week_clt_app_co_pymt_total_number_transaccional_maestra',
    'first_week_dbt_card_tra_number_transaccional_maestra',
    'first_week_tl_loans_pmt_number_transaccional_maestra',
    'first_week_total_collect_payments_number_transaccional_maestra',
    'first_wk_crcd_exp_trans_number_transaccional_maestra',
    'first_wk_crcd_pmt_trans_number_transaccional_maestra',
    'flag_activity_count_number_transaccional_maestra',
    'flg_tiene_hipoteca',
    'fonPEN',
    'fonUSD',
    'fourth_sched_cr_cd_tra_number_transaccional_maestra',
    'fourth_sched_dbt_cd_tra_number_transaccional_maestra',
    'fourth_week_clt_app_co_pymt_total_number_transaccional_maestra',
    'fourth_week_dbt_cd_tra_number_transaccional_maestra',
    'fourth_wk_tl_loans_pmt_number_transaccional_maestra',
    'frth_week_total_collect_payments_number_transaccional_maestra',
    'frth_wk_crcd_expn_trans_number_transaccional_maestra',
    'frth_wk_crcd_pmt_trans_number_transaccional_maestra',
    'immediate_issued_tr_tl_number_transaccional_maestra',
    'immediate_recv_tr_tl_number_transaccional_maestra',
    'immediate_tr_iss_total_amount_transaccional_maestra',
    'immediate_tr_recv_tl_amount_transaccional_maestra',
    'importe',
    'in_type_pmt_loan_tot_amount_transaccional_maestra',
    'instl_adv_credit_total_amount_transaccional_maestra',
    'interntnl_tr_origin_ind_type_transaccional_maestra',
    'iss_inmediate_tr_ori_ind_type_transaccional_maestra',
    'iss_intl_transfers_tot_amount_transaccional_maestra',
    'iss_not_customer_tr_tl_number_transaccional_maestra',
    'iss_schedule_tr_orig_ind_type_transaccional_maestra',
    'issued_tot_interntnl_tr_number_transaccional_maestra',
    'l6m_liability_sav_medn_amount_ingresos',
    'last_name_transaccional_maestra',
    'loans_charge_total_amount_transaccional_maestra',
    'loans_charges_total_number_transaccional_maestra',
    'loans_credit_total_amount_transaccional_maestra',
    'loans_credits_total_number_transaccional_maestra',
    'loans_origin_ind_type_transaccional_maestra',
    'mortgage_credit_loan_amount_rcc',
    'munics_othac_iss_imt_tl_amount_transaccional_maestra',
    'munics_othac_rv_imt_tl_amount_transaccional_maestra',
    'munics_owac_iss_imt_tl_amount_transaccional_maestra',
    'munics_owac_rv_imt_tl_amount_transaccional_maestra',
    'munics_shift_iss_tr_tl_amount_transaccional_maestra',
    'munics_shift_rv_tr_tl_amount_transaccional_maestra',
    'ncust_fn_ent_iss_tr_tot_amount_transaccional_maestra',
    'ncust_fn_ent_rv_tr_tot_amount_transaccional_maestra',
    'ncust_intbk_iss_tr_tot_amount_transaccional_maestra',
    'ncust_intbk_rv_tr_tot_amount_transaccional_maestra',
    'ncust_munics_iss_tr_tot_amount_transaccional_maestra',
    'ncust_munics_rv_tr_tot_amount_transaccional_maestra',
    'ncust_oe_iss_tr_tot_amount_transaccional_maestra',
    'ncust_oe_rv_tr_total_amount_transaccional_maestra',
    'ncust_scotb_iss_tr_tot_amount_transaccional_maestra',
    'ncust_scotb_rv_tr_tot_amount_transaccional_maestra',
    'no_customers_tr_orig_ind_type_transaccional_maestra',
    'non_cst_bcp_iss_tr_tot_amount_transaccional_maestra',
    'non_cust_bcp_rv_tr_tot_amount_transaccional_maestra',
    'non_cust_iss_transf_tot_amount_transaccional_maestra',
    'non_cust_rv_transf_tot_amount_transaccional_maestra',
    'os_crcd_payments_tot_amount_transaccional_maestra',
    'os_loan_payments_tot_amount_transaccional_maestra',
    'oth_pmt_typ_crcdpy_tot_amount_transaccional_maestra',
    'othac_crcd_payments_tot_amount_transaccional_maestra',
    'othac_loan_payments_tot_amount_transaccional_maestra',
    'other_cards_origin_ind_type_transaccional_maestra',
    'other_debits_amount_transaccional_maestra',
    'other_income_amount_transaccional_maestra',
    'other_lines_amount_rcc',
    'owac_cr_shift_iss_tr_tl_amount_transaccional_maestra',
    'owac_cr_shift_rv_tr_tl_amount_transaccional_maestra',
    'owac_crcd_payments_tot_amount_transaccional_maestra',
    'owac_loan_payments_tot_amount_transaccional_maestra',
    'pld_sum_amount_productos',
    'prtamor_credit_total_amount_transaccional_maestra',
    'recv_not_customer_tr_tl_number_transaccional_maestra',
    'recv_total_interntnl_tr_number_transaccional_maestra',
    'revolving_cards_number_rcc',
    'rv_inmediate_tr_ori_ind_type_transaccional_maestra',
    'rv_intl_transfers_tot_amount_transaccional_maestra',
    'rv_schedule_tr_origin_ind_type_transaccional_maestra',
    'sec_week_crcd_expn_tran_number_transaccional_maestra',
    'sec_wk_crcd_pmt_trans_number_transaccional_maestra',
    'secnd_week_total_collect_payments_number_transaccional_maestra',
    'second_last_name_transaccional_maestra',
    'second_sched_cr_cd_tra_number_transaccional_maestra',
    'second_sched_dbt_cd_tra_number_transaccional_maestra',
    'second_week_clt_app_co_pymt_total_number_transaccional_maestra',
    'second_week_dbt_cd_tra_number_transaccional_maestra',
    'second_wk_tl_loans_pmt_number_transaccional_maestra',
    'segment_desc_segmento',
    'segment_group_desc_segmento',
    'segmento',
    'shift_iss_transf_total_amount_transaccional_maestra',
    'shift_issued_tot_transf_number_transaccional_maestra',
    'shift_received_total_tr_number_transaccional_maestra',
    'shift_rv_transf_total_amount_transaccional_maestra',
    'slrypmt_shift_iss_tr_tl_amount_transaccional_maestra',
    'slrypmt_shift_rv_tr_tl_amount_transaccional_maestra',
    'splys_pmt_shft_is_tr_tl_amount_transaccional_maestra',
    'splys_pmt_shft_rv_tr_tl_amount_transaccional_maestra',
    'spread',
    'thd_week_crcd_expn_tran_number_transaccional_maestra',
    'thd_wk_crcd_pmt_trans_number_transaccional_maestra',
    'third_sched_cr_cd_tra_number_transaccional_maestra',
    'third_sched_dbt_cd_tra_number_transaccional_maestra',
    'third_week_clt_app_co_pymt_total_number_transaccional_maestra',
    'third_week_dbt_card_tra_number_transaccional_maestra',
    'third_week_tl_loans_pmt_number_transaccional_maestra',
    'third_week_total_collect_payments_number_transaccional_maestra',
    'top_bk_entity_debts_fnl_amount_rcc',
    'top_ent_othac_is_imt_tl_amount_transaccional_maestra',
    'top_ent_othac_rv_imt_tl_amount_transaccional_maestra',
    'top_ent_owac_iss_imt_tl_amount_transaccional_maestra',
    'top_ent_owac_rv_imt_tl_amount_transaccional_maestra',
    'top_ent_shift_iss_tr_tl_amount_transaccional_maestra',
    'top_ent_shift_rv_tr_tl_amount_transaccional_maestra',
    'total_balance_amount_rcc',
    'total_balance_amount_without_mortgage_rcc',
    'total_cr_card_payment_amount_transaccional_maestra',
    'total_risk_amount_rcc',
    'unused_credit_line_amount_rcc',
    'unused_line_revlv_card_amount_rcc',
    'unused_lines_credit_number_rcc',
    'visahoPEN',
    'visahoUSD']
    # Excluir la variable objetivo de las variables a procesar
    target_vars = [c for c in target_vars if c in df.columns and c != target_column]
    logger.info(f"Universo de variables objetivo ({len(target_vars)}): {target_vars}")
    logger.info(f"Variable objetivo '{target_column}' excluida del proceso de selección")

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
        return pd.DataFrame()

    logger.info(f"Variables numéricas seleccionadas: {num_vars}")

    # 4) Crear columna de cortes de percentiles (20 en 20) para groupby
    print("== Creando cortes de percentiles (20 en 20) para imputación por grupo ==")

    group_col = SELECTION_PARAMS['imputation_group_column']

    # 5) Imputación de medianas por grupo (si hay bin), si no, global
    print("== Imputando medianas ==")
    logger.info("Aplicando imputación de medianas...")
    df[num_vars] = impute_missing(df, columns=num_vars, strategy='median', groupby=group_col)[num_vars]
    logger.info("Imputación finalizada.")

    # 6) Winsorización percentil [1, 99]
    wins_lower = SELECTION_PARAMS['winsor_percentiles']['lower']
    wins_upper = SELECTION_PARAMS['winsor_percentiles']['upper']
    print(f"== Winsorización por percentiles [{wins_lower*100:.2f}, {wins_upper*100:.2f}] ==")
    logger.info(f"Aplicando winsorización por percentiles [{wins_lower*100:.2f}, {wins_upper*100:.2f}]...")
    wins_df, perc_df = winsorize_by_percentile(
        df,
        columns=num_vars,
        lower=wins_lower,
        upper=wins_upper,
        return_percentiles=True,
    )
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
    # Validar que todos los valores de CV sean no negativos (el CV siempre es >= 0)
    cv_valid = cv_table['cv'].dropna()
    if (cv_valid < 0).any():
        logger.warning(f"Se encontraron {sum(cv_valid < 0)} valores negativos de CV. Se convertirán a valor absoluto.")
        cv_table['cv'] = cv_table['cv'].abs()
    # Filtrar NaN para el gráfico
    cv_for_plot = cv_table['cv'].dropna()
    if len(cv_for_plot) == 0:
        logger.warning("No hay valores válidos de CV para graficar.")
    else:
        plt.figure(figsize=(6, 4))
        sns.kdeplot(data=cv_for_plot, fill=True, color='#4C78A8')
        plt.title(f'Distribución del Coeficiente de Variación (CV) - Cluster {cluster_value}')
        plt.xlabel('CV')
        plt.ylabel('Densidad')
        plt.xlim(left=0)  # Asegurar que el eje x comience en 0 (CV siempre >= 0)
        plt.tight_layout()
        plt.savefig(out_cv_plot, dpi=150)
        plt.close()
        logger.info(f"KDE plot guardado en: {out_cv_plot}")
        logger.info(f"CV válidos para gráfico: {len(cv_for_plot)}, rango: [{cv_for_plot.min():.4f}, {cv_for_plot.max():.4f}]")

    cv_min = SELECTION_PARAMS['cv']['min']
    cv_max = SELECTION_PARAMS['cv']['max']
    # 9) Filtrar variables por CV (configurable)
    print(f"== Filtrando por CV ({cv_min} < CV < {cv_max}) ==")
    logger.info(f"Filtrando variables por CV en ({cv_min}, {cv_max})...")
    cv_keep = cv_table[(cv_table['cv'] > cv_min) & (cv_table['cv'] < cv_max)]
    kept_vars = cv_keep['column'].tolist()
    logger.info(f"Variables retenidas por CV ({len(kept_vars)}): {kept_vars}")
    if not kept_vars:
        logger.warning("No quedaron variables tras filtrar por CV. Proceso finaliza antes de PCA.")
        print("Advertencia: No quedaron variables tras filtro de CV; se detiene antes de PCA.")
        return pd.DataFrame()

    # 10) PCA sobre variables retenidas, varianza acumulada 90%
    variance_threshold = SELECTION_PARAMS['pca_variance_threshold']
    print(f"== Ejecutando PCA (varianza {variance_threshold*100:.1f}%) ==")
    logger.info(f"Ejecutando PCA con threshold de varianza acumulada {variance_threshold*100:.1f}%...")
    pca_out = compute_pca(
        df=df,
        columns=kept_vars,
        n_components=None,
        variance_threshold=variance_threshold,
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
    print("== Generando gráfico de varianza acumulada de PCA ==")
    if pca_out['figure'] is not None:
        pca_out['figure'].savefig(out_pca_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Gráfico de varianza acumulada guardado en: {out_pca_plot}")

    # 11) PCA+LDA con n_components seleccionado; ranking top-20 y gráfico
    print("== Ejecutando PCA+LDA (con n recomendado por 90% varianza) ==")
    logger.info("Ejecutando PCA+LDA usando n_components recomendado (90% varianza)...")
    if target_column not in df.columns:
        logger.error(f"La columna objetivo '{target_column}' no existe en el DataFrame. No se puede ejecutar LDA.")
        print(f"Advertencia: No existe '{target_column}'; se detiene antes de PCA+LDA.")
        return pd.DataFrame()
    # Si n_components == número total de variables, usar n-1 para evitar problemas numéricos
    n_for_lda = chosen_n
    if n_for_lda >= len(kept_vars):
        n_for_lda = max(1, len(kept_vars) - 1)
        logger.info(f"n_components ajustado para LDA: {chosen_n} -> {n_for_lda} (porque igualaba el número de variables)")
    lda_out = pca_lda_importance(
        df=df,
        feature_columns=kept_vars,
        target_column=target_column,
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

    print("== Generando gráfico de coeficientes PCA+LDA ==")
    if lda_out['figure'] is not None:
        lda_out['figure'].savefig(out_lda_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Gráfico de coeficientes PCA+LDA guardado en: {out_lda_plot}")

    # 12) Filtrado por importancia PCA+LDA (> 0.05)
    lda_importance_threshold = SELECTION_PARAMS['lda_importance_threshold']
    print(f"== Filtrando por importancia PCA+LDA (> {lda_importance_threshold}) ==")
    logger.info(f"Filtrando variables por importancia PCA+LDA > {lda_importance_threshold}...")
    # Para asegurar que evaluamos todas, volvemos a pedir ranking completo (sin top_n)
    lda_all = pca_lda_importance(
        df=df,
        feature_columns=kept_vars,
        target_column=target_column,
        n_pca_components=n_for_lda,
        variance_threshold=None,
        scale=True,
        plot=False,
        return_fig=False,
        random_state=None,
        top_n=None,
    )
    ranking_all = lda_all['ranking']
    lda_keep = ranking_all[ranking_all['importance'] > lda_importance_threshold]['variable'].tolist()
    logger.info(f"Variables retenidas por PCA+LDA > {lda_importance_threshold} ({len(lda_keep)}): {lda_keep}")
    if not lda_keep:
        logger.warning("No quedaron variables tras filtrar por importancia PCA+LDA. Proceso se detiene antes de correlación.")
        print("Advertencia: No quedaron variables tras filtro PCA+LDA; se detiene antes de correlación.")
        return pd.DataFrame()

    # 13) Selección por correlación con grafos (priorizando variable configurable)
    print("== Ejecutando selección por correlación (grafos) ==")
    logger.info("Iniciando selección por correlación (grafos)...")
    corr_vars = list(lda_keep)
    # Agregar variable prioritaria si existe y no está ya en la lista
    prioritize_list = None
    if priority_var_correlation in df.columns and priority_var_correlation not in corr_vars:
        corr_vars.append(priority_var_correlation)
        prioritize_list = [priority_var_correlation]
        logger.info(f"Variable '{priority_var_correlation}' agregada para análisis de correlación y priorizada.")
    elif priority_var_correlation in corr_vars:
        prioritize_list = [priority_var_correlation]
        logger.info(f"Variable '{priority_var_correlation}' será priorizada en análisis de correlación.")

    numeric_corr_threshold = SELECTION_PARAMS['numeric_corr_threshold']
    sel_out = select_by_correlation_graph(
        df=df,
        columns=corr_vars,
        target_column=target_column,
        method='spearman',
        target_method='pointbiserial',
        threshold=numeric_corr_threshold,
        prioritize=prioritize_list,
        plot_matrix=True,
        return_matrix_fig=True,
        return_graph=True,
        logger=logger,
    )

    # Guardar matriz de correlación
    print("== Generando matriz de correlación ==")
    if sel_out['matrix_figure'] is not None:
        sel_out['matrix_figure'].savefig(out_corr_plot, dpi=150, bbox_inches='tight')
        logger.info(f"Matriz de correlación guardada en: {out_corr_plot}")

    # Reportar grupos formados (componentes conexos)
    if sel_out['graph'] is not None:
        comps = list(nx.connected_components(sel_out['graph']))
        logger.info(f"Grupos de variables numéricas correlacionadas ({len(comps)} grupos):")
        for idx, comp in enumerate(comps, start=1):
            comp_list = sorted(list(comp))
            logger.info(f"  Grupo {idx} ({len(comp_list)} variables): {comp_list}")

    selected_final = sel_out['selected']
    logger.info(f"Variables seleccionadas finales ({len(selected_final)}) con threshold de correlación {numeric_corr_threshold}: {selected_final}")
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
        psi_params = SELECTION_PARAMS['psi']
        psi_summary = compute_psi(
            df=df,
            temporal_column='quarter_period',
            variables=selected_final,     # usar solo variables numéricas finales seleccionadas
            num_bins=psi_params['bins'],
            bin_strategy=psi_params['bin_strategy'],
            reference_period=None,       # primer periodo ordenado
            return_detail=False,
            logger=logger,
        )
        logger.info("PSI por variable (trimestre):\n" + psi_summary.to_string(index=False))

        # Filtro por estabilidad: PSI < 0.1, priorizando 'spread' e 'importe'
        psi_threshold = psi_params['threshold']
        print(f"== Filtrando variables numéricas por PSI < {psi_threshold} ==")
        priority_vars_psi = list(dict.fromkeys([priority_var_correlation] + psi_params['priority_extras']))
        
        psi_keep = psi_summary[psi_summary['psi'] < psi_threshold]['variable'].tolist()
        psi_removed = [v for v in selected_final if v not in psi_keep]
        
        # Agregar variables priorizadas aunque no pasen el umbral de PSI
        for var in priority_vars_psi:
            if var in selected_final and var not in psi_keep:
                psi_keep.append(var)
                logger.info(f"Variable prioritaria '{var}' agregada aunque PSI >= {psi_threshold} (PSI={psi_summary[psi_summary['variable']==var]['psi'].values[0] if var in psi_summary['variable'].values else 'N/A'})")
        
        logger.info(f"Variables retenidas por PSI < {psi_threshold:.2f} ({len(psi_keep)}): {psi_keep}")
        if psi_removed:
            psi_removed_final = [v for v in psi_removed if v not in priority_vars_psi]
            if psi_removed_final:
                logger.info(f"Variables removidas por PSI >= {psi_threshold:.2f} ({len(psi_removed_final)}): {psi_removed_final}")
        # Actualizar conjunto numérico final
        selected_final = psi_keep

    # 14) Categóricas: únicos < 20
    cat_unique_max = SELECTION_PARAMS['categorical_unique_max']
    print(f"== Seleccionando categóricas por # únicos < {cat_unique_max} ==")
    logger.info(f"Seleccionando variables categóricas con menos de {cat_unique_max} valores únicos...")
    if working_categorical_vars:
        uniq_df = count_unique_categorical(df, columns=working_categorical_vars, include_na=False, logger=logger)
        cat_less_10 = uniq_df[uniq_df['n_unique'] < cat_unique_max]['variable'].tolist()
        logger.info(f"Categóricas con <{cat_unique_max} únicos ({len(cat_less_10)}): {cat_less_10}")
    else:
        cat_less_10 = []
        logger.info("No hay variables categóricas identificadas en el universo objetivo.")

    # 15) Categóricas: filtro por dominancia (umbral 95%)
    cat_dom_threshold = SELECTION_PARAMS['categorical_dominance_threshold']
    print(f"== Filtrando categóricas por dominancia (umbral {cat_dom_threshold*100:.1f}%) ==")
    logger.info(f"Eliminando variables categóricas dominantes con un valor único >= {cat_dom_threshold*100:.1f}%...")
    if cat_less_10:
        dom_out = categorical_cumulative_frequency(
            df=df,
            columns=cat_less_10,
            threshold=cat_dom_threshold,
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

    # 16) Categóricas: IV y selección por IV > 0.02
    print("== Calculando IV de categóricas ==")
    if target_column not in df.columns:
        logger.warning(f"No existe '{target_column}'; se omite el cálculo de IV para categóricas.")
        cat_iv_keep = []
    else:
        if cat_kept:
            iv_out = calculate_woe_iv(
                df=df,
                target=target_column,
                columns=cat_kept,
                include_na=True,
                bin_numeric=False,
                logger=logger,
            )
            iv_summary = iv_out['summary']
            logger.info("Resumen IV de categóricas:\n" + iv_summary.to_string(index=False))
            
            # Generar gráfico de IV
            print("== Generando gráfico de IV para categóricas ==")
            if len(iv_summary) > 0:
                fig, ax = plt.subplots(figsize=(10, max(6, len(iv_summary) * 0.3)))
                iv_sorted = iv_summary.sort_values('iv', ascending=True)
                iv_threshold = SELECTION_PARAMS['categorical_iv_threshold']
                colors = ['#d62728' if iv < iv_threshold else '#2ca02c' if iv < 0.1 else '#1f77b4' 
                         for iv in iv_sorted['iv']]
                ax.barh(iv_sorted['variable'], iv_sorted['iv'], color=colors)
                ax.axvline(x=iv_threshold, color='red', linestyle='--', linewidth=1, label=f'Umbral selección ({iv_threshold})')
                ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, label='IV débil (0.1)')
                ax.axvline(x=0.3, color='green', linestyle='--', linewidth=1, label='IV medio (0.3)')
                ax.set_xlabel('Information Value (IV)')
                ax.set_title(f'Information Value - Variables Categóricas (Cluster {cluster_value})')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                plt.savefig(out_iv_plot, dpi=150, bbox_inches='tight')
                plt.close()
                logger.info(f"Gráfico de IV guardado en: {out_iv_plot}")
            
            cat_iv_keep = iv_summary[iv_summary['iv'] > iv_threshold]['variable'].tolist()
            logger.info(f"Categóricas seleccionadas por IV > {iv_threshold} ({len(cat_iv_keep)}): {cat_iv_keep}")
        else:
            cat_iv_keep = []
            logger.info("No hay categóricas para evaluar IV tras filtros previos.")

    # 17) Selección de categóricas por correlación Chi-cuadrado (Cramér's V)
    print("== Selección de categóricas por correlación (Cramér's V) ==")
    if cat_iv_keep:
        logger.info("Calculando matriz de correlación para variables categóricas (método: Cramér's V)...")
        logger.info(f"Variables categóricas antes de selección por correlación ({len(cat_iv_keep)}): {cat_iv_keep}")
        
        # Selección por grafo de correlación usando Cramér's V
        # Esta función:
        # 1. Calcula la matriz de correlación (Cramér's V para cat-cat)
        # 2. Crea un grafo donde las aristas conectan variables altamente correlacionadas
        # 3. De cada grupo (componente conexo), selecciona la variable más correlacionada con el target
        cat_corr_threshold = SELECTION_PARAMS['categorical_corr_threshold']
        logger.info(f"Aplicando selección por grafo de correlación con threshold={cat_corr_threshold}...")
        logger.info("Metodología: Cramér's V (basado en Chi-cuadrado) para correlación cat-cat")
        logger.info("De cada grupo de variables correlacionadas, se selecciona la más asociada al target (correlation ratio)")
        
        cat_sel_out = select_by_correlation_graph(
            df=df,
            columns=cat_iv_keep,
            target_column=target_column,
            method='auto',  # Cramér's V para cat-cat, correlation ratio para cat-target
            threshold=cat_corr_threshold,
            prioritize=None,  # Sin priorizar ninguna variable categórica
            plot_matrix=True,
            return_matrix_fig=True,
            return_graph=True,  # Retornar grafo para analizar grupos
            logger=logger,
        )
        
        # Guardar gráfico de matriz de correlación categórica
        if 'matrix_figure' in cat_sel_out and cat_sel_out['matrix_figure'] is not None:
            cat_sel_out['matrix_figure'].savefig(out_corr_cat_plot, dpi=150, bbox_inches='tight')
            plt.close(cat_sel_out['matrix_figure'])
            logger.info(f"Matriz de correlación categórica guardada en: {out_corr_cat_plot}")
        else:
            logger.warning("No se generó figura de matriz de correlación categórica.")
        
        # Reportar grupos formados (componentes conexos)
        if cat_sel_out['graph'] is not None:
            cat_comps = list(nx.connected_components(cat_sel_out['graph']))
            logger.info(f"Grupos de variables categóricas correlacionadas ({len(cat_comps)} grupos):")
            for idx, comp in enumerate(cat_comps, start=1):
                comp_list = sorted(list(comp))
                logger.info(f"  Grupo {idx} ({len(comp_list)} variables): {comp_list}")
        
        cat_final_selected = cat_sel_out['selected']
        cat_corr_removed = [c for c in cat_iv_keep if c not in cat_final_selected]
        logger.info(f"Categóricas seleccionadas tras correlación ({len(cat_final_selected)}): {cat_final_selected}")
        if cat_corr_removed:
            logger.info(f"Categóricas removidas por correlación ({len(cat_corr_removed)}): {cat_corr_removed}")
        
        # Actualizar lista final de categóricas
        cat_iv_keep = cat_final_selected
    else:
        logger.info("No hay variables categóricas para analizar correlación.")

    # 18) Construir base final (desde la base original), solo variables seleccionadas + target
    print("== Construyendo base final de variables seleccionadas ==")
    final_numeric = selected_final if 'selected_final' in locals() else []
    final_categorical = cat_iv_keep if 'cat_iv_keep' in locals() else []
    final_vars_all = list(dict.fromkeys(final_numeric + final_categorical))
    final_vars_all = [c for c in final_vars_all if c in df_raw.columns and c != target_column]
    
    # === RESUMEN FINAL DE VARIABLES SELECCIONADAS ===
    logger.info("\n" + "="*80)
    logger.info(f"RESUMEN FINAL DE VARIABLES SELECCIONADAS - CLUSTER {cluster_value}")
    logger.info("="*80)
    
    # Variables numéricas
    if final_numeric:
        logger.info(f"\nVARIABLES NUMERICAS SELECCIONADAS ({len(final_numeric)}):")
        for i, var in enumerate(final_numeric, 1):
            logger.info(f"  {i}. {var}")
    else:
        logger.info("\nVARIABLES NUMERICAS SELECCIONADAS: Ninguna")
    
    # Variables categóricas
    if final_categorical:
        logger.info(f"\nVARIABLES CATEGORICAS SELECCIONADAS ({len(final_categorical)}):")
        for i, var in enumerate(final_categorical, 1):
            logger.info(f"  {i}. {var}")
    else:
        logger.info("\nVARIABLES CATEGORICAS SELECCIONADAS: Ninguna")
    
    # Total
    logger.info(f"\nTOTAL DE VARIABLES SELECCIONADAS: {len(final_vars_all)}")
    logger.info(f"   - Numericas: {len(final_numeric)}")
    logger.info(f"   - Categoricas: {len(final_categorical)}")
    logger.info(f"Variable objetivo: {target_column}")
    logger.info("="*80 + "\n")
    
    # Agregar variable objetivo al final
    if target_column in df_raw.columns:
        final_cols = final_vars_all + [target_column]
        logger.info(f"Variable objetivo '{target_column}' agregada al resultado final.")
    else:
        logger.warning(f"La columna objetivo '{target_column}' no existe en la base original. La base final no incluirá target.")
        final_cols = final_vars_all

    if final_cols:
        # Generar gráficos de variables finales usando el DataFrame PROCESADO (con imputación, winsorización, etc.)
        print("== Generando gráficos de variables finales (con variables procesadas) ==")
        try:
            # Crear DataFrame temporal con variables procesadas para los gráficos
            # Usar df (procesado) en lugar de df_raw (original)
            cols_for_graphs = [c for c in final_cols if c in df.columns]
            if cols_for_graphs:
                df_processed_for_graphs = df[cols_for_graphs].copy()
                logger.info(f"Generando gráficos con {len(cols_for_graphs)} variables procesadas (imputación, winsorización, etc.)")
                
                generate_final_vars_graphs(
                    df_final=df_processed_for_graphs,
                    cluster_value=cluster_value,
                    base_dir=base_dir,
                    target_column=target_column,
                    logger=logger
                )
            else:
                logger.warning("No hay variables procesadas disponibles para generar gráficos.")
        except Exception as e:
            logger.error(f"Error generando gráficos de variables finales: {str(e)}", exc_info=True)
            print(f"Advertencia: Error generando gráficos de variables finales: {str(e)}")
        
        # Construir base final desde la base ORIGINAL (sin transformaciones) para guardar
        final_df = df_raw[final_cols].copy()
        logger.info(f"Base final construida desde la base original (sin transformaciones). Shape: {final_df.shape}")
        logger.info(f"Columnas en el DataFrame final: {final_cols}")
        
        return final_df
    else:
        logger.warning("No hay variables finales para construir la base final.")
        return pd.DataFrame()


def main():
    # Config paths
    data_path = DATA_PATH
    base_dir = os.path.dirname(__file__)
    log_path = os.path.join(base_dir, 'process_selection_vars.log')
    
    # Configuración de variables
    target_column = TARGET_COLUMN  # Variable objetivo (no participa en selección)
    priority_var_correlation = PRIORITY_VAR_CORRELATION  # Variable a priorizar en selección por correlación
    
    logger = setup_logger(log_path)
    logger.info("== INICIO DEL PROCESO ==")
    logger.info(f"Variable objetivo: {target_column}")
    logger.info(f"Variable prioritaria en correlaciones: {priority_var_correlation}")

    # 1) Leer dataset
    print("== Leyendo dataset ==")
    logger.info("Leyendo dataset...")
    df = read_dataset(data_path, fmt='csv', logger=logger)
    logger.info(f"Leído con shape: {df.shape}")
    
    # Verificar que existe la columna 'cluster'
    if 'cluster' not in df.columns:
        logger.error("La columna 'cluster' no existe en el dataset. Proceso finaliza.")
        print("Error: La columna 'cluster' no existe en el dataset.")
        return
    
    # 2) Eliminar columnas según criterios
    print("== Eliminando columnas según criterios ==")
    logger.info("Eliminando columnas con patrones específicos...")
    cols_to_drop = []
    for col in df.columns:
        col_lower = col.lower()
        
        # Eliminar columnas con estos patrones
        if any(pattern in col_lower for pattern in ['_id', '_type', 'plazo', 'cutoff', 'date']):
            cols_to_drop.append(col)
            continue
        
        # Eliminar columnas que contengan "codmes" pero NO la columna exacta "codmes"
        if 'codmes' in col_lower and col_lower != 'codmes':
            cols_to_drop.append(col)
            continue
    
    # Eliminar columnas específicas
    specific_drops = ['di', 'tasa']
    for col in specific_drops:
        if col in df.columns and col not in cols_to_drop:
            cols_to_drop.append(col)
    
    # Log detallado de eliminación
    logger.info(f"Columnas a eliminar ({len(cols_to_drop)}):")
    logger.info(f"- Columnas con '_id': {[c for c in cols_to_drop if '_id' in c.lower()]}")
    logger.info(f"- Columnas con '_type': {[c for c in cols_to_drop if '_type' in c.lower()]}")
    logger.info(f"- Columnas con 'date': {[c for c in cols_to_drop if 'date' in c.lower()]}")
    logger.info(f"- Columnas con variantes de 'codmes' (excepto 'codmes'): {[c for c in cols_to_drop if 'codmes' in c.lower() and c.lower() != 'codmes']}")
    logger.info(f"- Columnas con 'plazo': {[c for c in cols_to_drop if 'plazo' in c.lower()]}")
    logger.info(f"- Columnas con 'cutoff': {[c for c in cols_to_drop if 'cutoff' in c.lower()]}")
    logger.info(f"- Columnas específicas: {[c for c in cols_to_drop if c in specific_drops]}")
    
    # Verificar que 'codmes' no esté en la lista
    if 'codmes' in cols_to_drop:
        logger.warning("ADVERTENCIA: 'codmes' estaba marcada para eliminación, se remueve de la lista.")
        cols_to_drop.remove('codmes')
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    logger.info(f"Columnas eliminadas: {len(cols_to_drop)}")
    logger.info(f"Shape después de eliminar columnas: {df.shape}")
    
    # Verificar que codmes sigue en el dataset
    if 'codmes' in df.columns:
        logger.info("✓ Columna 'codmes' preservada correctamente")
    else:
        logger.warning("✗ Columna 'codmes' no está en el dataset")
    
    # 3) Obtener clusters únicos
    clusters = sorted(df['cluster'].unique().tolist())
    logger.info(f"Clusters encontrados: {clusters}")
    print(f"== Procesando {len(clusters)} clusters ==")
    
    # 4) Procesar cada cluster
    all_results = []
    cluster_variables_records = []
    for cluster_val in clusters:
        logger.info(f"\n{'='*60}")
        logger.info(f"INICIANDO PROCESAMIENTO PARA CLUSTER: {cluster_val}")
        logger.info(f"{'='*60}")
        logger.info(f"Log detallado del cluster en: selection_logs/selection_cluster_{cluster_val}.log")
        
        # Filtrar datos del cluster
        df_cluster = df[df['cluster'] == cluster_val].copy()
        logger.info(f"Registros en cluster {cluster_val}: {len(df_cluster)}")
        
        if len(df_cluster) == 0:
            logger.warning(f"Cluster {cluster_val} está vacío. Se omite.")
            continue
        
        # Ejecutar proceso de selección para este cluster
        try:
            final_df = process_cluster_selection(
                df_cluster=df_cluster,
                cluster_value=str(cluster_val),
                base_dir=base_dir,
                target_column=target_column,
                priority_var_correlation=priority_var_correlation
            )
            
            if not final_df.empty:
                vars_for_cluster = final_df.columns.tolist()
                if target_column not in vars_for_cluster and target_column in df_cluster.columns:
                    vars_for_cluster.append(target_column)
                cluster_variables_records.append({
                    'cluster': cluster_val,
                    'variables': ', '.join(vars_for_cluster)
                })
                
                # Agregar columna cluster al resultado
                final_df['cluster'] = cluster_val
                all_results.append(final_df)
                logger.info(f"Cluster {cluster_val} procesado exitosamente. Variables seleccionadas: {len(final_df.columns) - 1}")
            else:
                logger.warning(f"Cluster {cluster_val} no produjo variables seleccionadas.")
        except Exception as e:
            logger.error(f"Error procesando cluster {cluster_val}: {str(e)}", exc_info=True)
            print(f"Error procesando cluster {cluster_val}: {str(e)}")
            continue
    
    # Guardar CSV consolidado de variables por cluster
    if cluster_variables_records:
        logger.info("Generando CSV con variables finales por cluster...")
        try:
            vars_by_cluster_df = (
                pd.DataFrame(cluster_variables_records)
                .sort_values('cluster')
                .reset_index(drop=True)
            )
            vars_by_cluster_df.to_csv(VARS_BY_CLUSTER_OUTPUT, index=False)
            logger.info(f"Variables por cluster guardadas en: {VARS_BY_CLUSTER_OUTPUT}")
            print(f"  ✓ Variables por cluster guardadas en: {VARS_BY_CLUSTER_OUTPUT}")
        except Exception as e:
            logger.error(f"Error guardando variables por cluster: {str(e)}", exc_info=True)
            print(f"Error guardando variables por cluster: {str(e)}")
    else:
        logger.warning("No se generó resumen de variables por cluster (no hubo resultados).")

    # 5) Guardar resultados en S3
    if all_results:
        print("== Guardando resultados en S3 ==")
        logger.info(f"Guardando {len(all_results)} clusters en S3...")
        
        # Guardar cada cluster por separado en S3 (sin concatenar porque cada cluster tiene diferentes variables)
        for i, cluster_df in enumerate(all_results):
            if not cluster_df.empty:
                # Obtener el valor del cluster desde el DataFrame
                cluster_val = cluster_df['cluster'].iloc[0]
                s3_output_path = f's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/02.Feature/seleccion_de_var_{cluster_val}'
                logger.info(f"Guardando cluster {cluster_val} en: {s3_output_path}")
                logger.info(f"Variables en cluster {cluster_val}: {len(cluster_df.columns) - 1} (sin contar 'cluster')")
                try:
                    cluster_df.to_parquet(s3_output_path, index=False, engine='pyarrow')
                    logger.info(f"Cluster {cluster_val} guardado exitosamente en S3. Shape: {cluster_df.shape}")
                    print(f"Cluster {cluster_val} guardado en S3")
                except Exception as e:
                    logger.error(f"Error guardando cluster {cluster_val} en S3: {str(e)}", exc_info=True)
                    print(f"Error guardando cluster {cluster_val} en S3: {str(e)}")
        
        # 6) Crear y guardar base unificada con todos los clusters y todas las variables
        print("== Creando base unificada con todos los clusters ==")
        logger.info("Creando base unificada que contiene todos los clusters con todas las variables seleccionadas...")
        
        try:
            # Filtrar solo DataFrames no vacíos
            valid_dfs = [cluster_df for cluster_df in all_results if not cluster_df.empty]
            
            if not valid_dfs:
                logger.warning("No se pudo crear la base unificada: no hay DataFrames válidos.")
            else:
                # Usar pd.concat directamente - pandas manejará automáticamente la alineación de columnas y tipos
                logger.info(f"Concatenando {len(valid_dfs)} DataFrames de clusters...")
                unified_df = pd.concat(valid_dfs, ignore_index=True, sort=False)
                
                logger.info(f"Base unificada creada. Shape: {unified_df.shape}")
                logger.info(f"Total de registros: {len(unified_df)}")
                logger.info(f"Total de columnas: {len(unified_df.columns)}")
                
                # Convertir todos los tipos numéricos a float64 y tipos nullable a estándar
                logger.info("Convirtiendo todas las variables numéricas a float64...")
                for col in unified_df.columns:
                    # Saltar la columna cluster si es categórica
                    if col == 'cluster':
                        continue
                    
                    dtype = str(unified_df[col].dtype)
                    
                    # Si es numérica (int, int64, Int64, float, float64, Float64, etc.), convertir a float64
                    if pd.api.types.is_numeric_dtype(unified_df[col]):
                        unified_df[col] = unified_df[col].astype('float64')
                    # Convertir tipos nullable no numéricos
                    elif dtype == 'boolean':
                        unified_df[col] = unified_df[col].astype('object')  # object para mantener True/False/NaN
                    elif dtype == 'string':
                        unified_df[col] = unified_df[col].astype('object')  # object para strings con NaN
                
                logger.info("Todas las variables numéricas convertidas a float64")
                
                # Guardar base unificada en S3
                s3_unified_path = 's3://ada-us-east-1-sbx-live-pe-intc-data/01DataAnalytics/01Pricing/07.DepositoPlazo/02.Feature/seleccion_de_var_unificado.parquet'
                logger.info(f"Guardando base unificada en: {s3_unified_path}")
                unified_df.to_parquet(s3_unified_path, index=False, engine='pyarrow')
                logger.info(f"Base unificada guardada exitosamente en S3. Shape: {unified_df.shape}")
                print(f"Base unificada guardada en S3: {s3_unified_path}")
        except Exception as e:
            logger.error(f"Error creando/guardando base unificada: {str(e)}", exc_info=True)
            print(f"Error creando base unificada: {str(e)}")
    else:
        logger.warning("No hay resultados para guardar.")
        print("Advertencia: No se generaron resultados para ningún cluster.")
    
    logger.info("== FIN DEL PROCESO ==")
    print("== Proceso completado. Ver log para detalles. ==")


if __name__ == "__main__":
    main()
