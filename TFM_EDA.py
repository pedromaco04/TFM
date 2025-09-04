import os
import argparse
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging
from datetime import datetime
import warnings
import sys

from engine_TFM.engine_eda import Exploracion, SelectVarsNumerics, SelectVarsCategoricals
from engine_TFM.utils import (
    load_config,
    section,
    safe_drop_columns,
    build_binary_target,
    derive_features,
    proteger_int_rate,
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time

# Configurar pandas para suprimir warnings en terminal
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


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

def print_pipeline_header():
    """Mostrar solo el header inicial del pipeline en terminal"""
    print("🔧 INICIANDO EDA PIPELINE...")

def print_pipeline_summary(execution_time, total_rows, pca_vars, anova_vars, files_saved):
    """Mostrar solo el resumen final en terminal"""
    print("\n🔧 RESUMEN FINAL:")
    print(f"   📊 Dataset: {total_rows:,} filas × 50 columnas")  
    print(f"   ⏱️  Tiempo: {execution_time}")
    print(f"   ✅ Variables PCA+LDA: {pca_vars}")
    print(f"   ✅ Variables ANOVA: {anova_vars}")
    print(f"   🛡️  int_rate PROTEGIDO: ✅")
    print(f"   📁 Archivos guardados: {', '.join(files_saved)}")
    print(f"   📋 Log detallado: log_EDA.txt")


def calculate_pca_components_by_variance(df, num_cols, variance_threshold=0.95):
    """
    Calcula el número de componentes PCA que explican un porcentaje específico de varianza.
    
    Args:
        df: DataFrame con los datos
        num_cols: Lista de columnas numéricas
        variance_threshold: Porcentaje de varianza a explicar (ej: 0.95 para 95%)
    
    Returns:
        int: Número de componentes que explican el porcentaje especificado
    """
    # Preparar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
    
    # Ejecutar PCA con todos los componentes
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calcular varianza acumulada
    varianza_acumulada = pca.explained_variance_ratio_.cumsum()
    
    # Encontrar el número de componentes que alcanza el threshold
    n_components = np.argmax(varianza_acumulada >= variance_threshold) + 1
    
    # Asegurar que al menos sea 1
    n_components = max(1, n_components)
    
    return n_components, varianza_acumulada[n_components-1]


class EDAPipeline:
    """
    Object-oriented EDA pipeline that replicates the notebook flow with flexibility.

    Steps (config-driven):
      1) Load CSV
      2) Build binary target from status
      3) Drop columns by nulls threshold and manual lists
      4) Drop ex-post columns
      5) Derive numeric features
      6) Separate numeric/categorical
      7) Numeric summary and filtering by missing and CV
      8) PCA diagnostics and PCA+LDA variable importance selection
      9) ANOVA F-test variable selection
     10) Correlation-based redundancy removal
     11) Save final CSVs for PCA+LDA and ANOVA sets
    """

    def __init__(self, config_path: str):
        defaults = {
            'data': {
                'input_csv': 'Loan_data.csv',
                'status_column': 'loan_status',
                'target_column': 'flg_target',
                'allowed_status': {'good': [], 'bad': []},
                'drop_missing_status': True,
                'output_dir': '.',
                'output_pca_lda_csv': 'df_pca_lda.csv',
                'output_anova_csv': 'df_anova.csv',
            },
            'cleaning': {
                'enable_drop_null_columns': True,
                'drop_null_threshold': 0.95,
                'enable_drop_irrelevant_columns': True,
                'drop_columns': [],
                'enable_drop_ex_post': True,
                'drop_ex_post_columns': [],
            },
            'features': {
                'derive': {
                    'enable_term_integer': True,
                    'enable_emp_length_months': True,
                    'enable_ratio_loan_income': True,
                    'enable_installment_pct_loan': True,
                    'enable_fico_avg': True,
                    'enable_revol_util_ratio': True,
                }
            },
            'selection': {
                'filter_missing': {'enable': True, 'threshold': 0.5, 'rescue_by_correlation': True, 'min_corr': 0.1, 'min_samples': 30},
                'filter_cv': {'enable': True, 'threshold': 0.1, 'rescue_by_correlation': True, 'min_corr': 0.1, 'min_samples': 30},
                'pca': {
                    'enable': True, 
                    'mode': 'variance_percentage',  # 'fixed_components' o 'variance_percentage'
                    'n_components': 35,  # Solo se usa si mode='fixed_components'
                    'variance_threshold': 0.95,  # Solo se usa si mode='variance_percentage'
                    'plot_variance': False
                },
                'lda_importance': {'enable': True, 'importance_threshold': 0.05, 'plot_importance': False},
                'anova': {'enable': True, 'min_f_score': None, 'max_p_value': 0.05, 'top_n': None},
                'correlation_redundancy': {'enable': True, 'threshold': 0.6, 'plot_heatmap': False},
            },
            'logging': {'verbose': True, 'log_file': 'log_EDA.txt'}
        }
        self.config = load_config(config_path, defaults=defaults)
        self.verbose = bool(self.config.get('logging', {}).get('verbose', True))
        
        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging to both console and file"""
        log_file = self.config.get('logging', {}).get('log_file', 'log_EDA.txt')
        
        # Create logger
        self.logger = logging.getLogger('EDAPipeline')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_formatter = logging.Formatter('%(message)s')
        
        # File handler (detailed)
        file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Solo agregar file handler - no console handler
        # para que los logs detallados vayan solo al archivo
        self.logger.addHandler(file_handler)
        
        # Configurar logging de warnings para que vayan al archivo
        warnings_logger = logging.getLogger('py.warnings')
        warnings_logger.setLevel(logging.WARNING)
        warnings_logger.addHandler(file_handler)
        
        # Log pipeline start
        self.logger.info("=" * 80)
        self.logger.info("EDA PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration file: {self.config.get('_config_path', 'Unknown')}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)

    def _print(self, msg: str, level: str = "INFO") -> None:
        """Enhanced print method that logs ONLY to file, not console"""
        if self.verbose:
            if level == "INFO":
                self.logger.info(msg)
            elif level == "WARNING":
                self.logger.warning(msg)
            elif level == "ERROR":
                self.logger.error(msg)
            else:
                self.logger.info(msg)

    def _log_step_start(self, step_name: str, step_number: int):
        """Log the start of a pipeline step"""
        self.logger.info("")
        self.logger.info(f"STEP {step_number}: {step_name}")
        self.logger.info("-" * 60)

    def _log_step_end(self, step_name: str, step_number: int, result_info: str = ""):
        """Log the end of a pipeline step"""
        if result_info:
            self.logger.info(f"STEP {step_number} COMPLETED: {step_name} - {result_info}")
        else:
            self.logger.info(f"STEP {step_number} COMPLETED: {step_name}")
        self.logger.info("-" * 60)

    def _ensure_output_dir(self) -> None:
        out_dir = self.config['data']['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        self.logger.info(f"Output directory ensured: {out_dir}")

    # --------------------------- Pipeline steps ----------------------------
    def load_data(self) -> pd.DataFrame:
        self._log_step_start("Load data", 1)
        path = self.config['data']['input_csv']
        self.logger.info(f"Loading data from: {path}")
        
        try:
            # Capturar warnings y enviarlos al log en lugar del terminal
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                # Cargar datos con parámetros para evitar DtypeWarning
                df = pd.read_csv(
                    path, 
                    low_memory=False,  # Evita DtypeWarning
                    dtype_backend='numpy_nullable'  # Mejor manejo de tipos
                )
                
                # Log de cualquier warning capturado
                if w:
                    self.logger.warning(f"Warnings during data loading:")
                    for warning in w:
                        self.logger.warning(f"  {warning.category.__name__}: {warning.message}")
            
            self.logger.info(f"Data loaded successfully")
            self.logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
            self.logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            # Log basic info about the dataset
            self.logger.info(f"Data types:")
            for col, dtype in df.dtypes.items():
                self.logger.info(f"  {col}: {dtype}")
            
            self._log_step_end("Load data", 1, f"Shape: {df.shape}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_step_start("Build binary target", 2)
        data_cfg = self.config['data']
        good = data_cfg['allowed_status'].get('good', [])
        bad = data_cfg['allowed_status'].get('bad', [])
        
        self.logger.info(f"Status column: {data_cfg['status_column']}")
        self.logger.info(f"Target column: {data_cfg['target_column']}")
        self.logger.info(f"Good status values: {good}")
        self.logger.info(f"Bad status values: {bad}")
        self.logger.info(f"Drop missing status: {data_cfg.get('drop_missing_status', True)}")
        
        # Log original status distribution
        if data_cfg['status_column'] in df.columns:
            status_counts = df[data_cfg['status_column']].value_counts()
            self.logger.info(f"Original status distribution:")
            for status, count in status_counts.items():
                self.logger.info(f"  {status}: {count} ({count/len(df)*100:.2f}%)")
        
        df = build_binary_target(
            df,
            status_column=data_cfg['status_column'],
            bad_status=bad,
            good_status=good,
            target_column=data_cfg['target_column'],
            drop_missing_status=bool(data_cfg.get('drop_missing_status', True)),
            verbose=False  # Solo log, no prints en terminal
        )
        
        # Log target distribution
        if data_cfg['target_column'] in df.columns:
            target_counts = df[data_cfg['target_column']].value_counts()
            self.logger.info(f"Target distribution after building:")
            for target_val, count in target_counts.items():
                self.logger.info(f"  {target_val}: {count} ({count/len(df)*100:.2f}%)")
        
        self._log_step_end("Build binary target", 2, f"Shape: {df.shape}")
        return df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_step_start("Clean columns", 3)
        cfg = self.config['cleaning']
        df2 = df.copy()
        
        self.logger.info(f"Starting shape: {df2.shape}")
        
        if cfg.get('enable_drop_null_columns', True):
            thr = float(cfg.get('drop_null_threshold', 0.95))
            self.logger.info(f"Null threshold: {thr*100:.0f}%")
            
            # Calculate null percentages for all columns
            null_percentages = df2.isnull().mean().sort_values(ascending=False)
            cols_above_threshold = null_percentages[null_percentages > thr]
            
            self.logger.info(f"Columns with >{thr*100:.0f}% nulls ({len(cols_above_threshold)}):")
            for col, pct in cols_above_threshold.items():
                self.logger.info(f"  {col}: {pct*100:.2f}%")
            
            cols_muchos_nulos = cols_above_threshold.index.tolist()
            df2 = safe_drop_columns(df2, cols_muchos_nulos, verbose=False)
            self.logger.info(f"After dropping null columns: {df2.shape}")
        
        if cfg.get('enable_drop_irrelevant_columns', True):
            drop_cols = cfg.get('drop_columns', [])
            self.logger.info(f"Dropping irrelevant columns ({len(drop_cols)}): {drop_cols}")
            df2 = safe_drop_columns(df2, drop_cols, verbose=False)
            self.logger.info(f"After dropping irrelevant columns: {df2.shape}")
        
        if cfg.get('enable_drop_ex_post', True):
            ex_post_cols = cfg.get('drop_ex_post_columns', [])
            self.logger.info(f"Dropping ex-post columns ({len(ex_post_cols)}): {ex_post_cols}")
            df2 = safe_drop_columns(df2, ex_post_cols, verbose=False)
            self.logger.info(f"After dropping ex-post columns: {df2.shape}")
        
        self._log_step_end("Clean columns", 3, f"Final shape: {df2.shape}")
        return df2

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_step_start("Derive features", 4)
        toggles = self.config.get('features', {}).get('derive', {})
        
        self.logger.info(f"Feature derivation toggles:")
        for feature, enabled in toggles.items():
            self.logger.info(f"  {feature}: {'✓' if enabled else '✗'}")
        
        df2 = derive_features(df, toggles, verbose=False)
        
        # Log new columns added
        new_cols = set(df2.columns) - set(df.columns)
        if new_cols:
            self.logger.info(f"New derived columns ({len(new_cols)}): {list(new_cols)}")
        else:
            self.logger.info("No new columns were derived")
        
        self._log_step_end("Derive features", 4, f"Shape: {df2.shape}")
        return df2

    def separate_variables(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        self._log_step_start("Separate variables by type (INTELIGENTE)", 5)

        # Usar la nueva función inteligente de separación
        config = self.config.get('separation', {})
        max_unique_cats = config.get('max_unique_categorical', 20)
        max_concentration = config.get('max_concentration_categorical', 0.98)
        
        # Obtener variables protegidas
        protection_config = self.config.get('int_rate_protection', {})
        protected_variables = []
        if protection_config.get('enable_protection', True):
            protected_var = protection_config.get('protected_variable', 'int_rate')
            protected_variables = [protected_var]

        vars_dict = Exploracion.separar_variables_inteligente(
            df,
            target=self.config['data']['target_column'],
            max_unique_cats=max_unique_cats,
            max_concentration=max_concentration,
            protected_variables=protected_variables,
            verbose=False,  # Solo log, no prints en terminal
            logger=self.logger
        )

        cat_cols = vars_dict['categoricas']
        num_cols = vars_dict['numericas']
        descartadas = vars_dict['descartadas']
        conversiones = vars_dict['conversiones_automaticas']

        # Logging adicional para el pipeline
        self.logger.info(f"\n📈 RESUMEN PARA PIPELINE:")
        self.logger.info(f"   ✅ Numéricas procesables: {len(num_cols)}")
        self.logger.info(f"   ✅ Categóricas procesables: {len(cat_cols)}")
        self.logger.info(f"   ❌ Variables descartadas: {len(descartadas)}")
        if conversiones:
            self.logger.info(f"   🔄 Conversiones automáticas: {len(conversiones)}")

        if descartadas:
            self.logger.warning(f"\n⚠️  Variables descartadas por calidad:")
            for col in descartadas[:5]:
                self.logger.warning(f"   • {col}")
            if len(descartadas) > 5:
                self.logger.warning(f"   • ... y {len(descartadas)-5} más")

        if cat_cols:
            self.logger.info(f"\n🎯 Variables categóricas seleccionadas:")
            for col in cat_cols[:5]:
                n_unique = df[col].nunique()
                self.logger.info(f"   • {col} ({n_unique} valores únicos)")
            if len(cat_cols) > 5:
                self.logger.info(f"   • ... y {len(cat_cols)-5} más")

        self._log_step_end("Separate variables by type (INTELIGENTE)", 5,
                          f"Numéricas: {len(num_cols)}, Categóricas: {len(cat_cols)}, Descartadas: {len(descartadas)}")
        return cat_cols, num_cols

    def select_numeric_variables(self, df: pd.DataFrame, num_cols: List[str]) -> List[str]:
        self._log_step_start("Numeric summary and filtering", 6)

        # PROTEGER int_rate de cualquier filtro
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if protected_var in num_cols:
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' detectada - será preservada")
        else:
            self.logger.warning(f"⚠️ Variable protegida '{protected_var}' NO encontrada en variables numéricas")

        self.logger.info(f"Starting with {len(num_cols)} numerical variables")
        
        resumen = Exploracion.resumen_numericas(df, num_cols)
        cfg = self.config['selection']
        selected = num_cols
        
        # Log initial summary
        self.logger.info("Initial numerical variables summary:")
        self.logger.info(f"  Total variables: {len(num_cols)}")
        
        if cfg['filter_missing'].get('enable', True):
            thr = float(cfg['filter_missing'].get('threshold', 0.5))
            self.logger.info(f"Applying missing value filter (threshold: {thr*100:.0f}%)")
            
            selected = SelectVarsNumerics.filtrar_por_nulos(
                df,
                resumen.loc[selected],
                target=self.config['data']['target_column'],
                umbral=thr,
                rescatar=bool(cfg['filter_missing'].get('rescue_by_correlation', True)),
                min_corr=float(cfg['filter_missing'].get('min_corr', 0.1)),
                min_samples=int(cfg['filter_missing'].get('min_samples', 30)),
                logger=self.logger
            )
            self.logger.info(f"After missing value filter: {len(selected)} variables")

        if cfg['filter_cv'].get('enable', True):
            thr = float(cfg['filter_cv'].get('threshold', 0.1))
            self.logger.info(f"Applying coefficient of variation filter (threshold: {thr})")

            resumen_sel = resumen.loc[selected]
            selected = SelectVarsNumerics.filtrar_por_cv(
                df,
                resumen_sel,
                target=self.config['data']['target_column'],
                umbral=thr,
                rescatar=bool(cfg['filter_cv'].get('rescue_by_correlation', True)),
                min_corr=float(cfg['filter_cv'].get('min_corr', 0.1)),
                min_samples=int(cfg['filter_cv'].get('min_samples', 30)),
            )
            self.logger.info(f"After CV filter: {len(selected)} variables")
        
        # ASEGURAR QUE LA VARIABLE PROTEGIDA ESTÉ INCLUIDA
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if protected_var not in selected and protected_var in df.columns:
            selected.append(protected_var)
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en selección final")
        elif protected_var in selected:
            self.logger.info(f"✅ Variable protegida '{protected_var}' mantenida en selección")

        self.logger.info(f"Final selected numerical variables: {selected}")
        self._log_step_end("Numeric summary and filtering", 6, f"Selected: {len(selected)}")
        return selected

    def select_categorical_variables(self, df: pd.DataFrame, cat_cols: List[str]) -> Tuple[List[str], pd.DataFrame]:
        self._log_step_start("Categorical variables selection", 6.5)

        if not cat_cols:
            self.logger.info("No categorical variables to process")
            self._log_step_end("Categorical variables selection", 6.5, "No variables")
            return [], df

        # Obtener configuración
        config = self.config.get('categorical_selection', {})
        iv_threshold = config.get('iv_threshold', 0.3)  # Cambiado a 0.3 (MEDIA en adelante)
        chi_square_threshold = config.get('chi_square_threshold', 0.05)
        enable_chi_square = config.get('enable_chi_square', True)
        enable_woe_iv = config.get('enable_woe_iv', True)

        self.logger.info(f"Configuración selección categórica:")
        self.logger.info(f"  • IV threshold: {iv_threshold} (MEDIA en adelante)")
        self.logger.info(f"  • Chi-square: {'✅' if enable_chi_square else '❌'}")
        self.logger.info(f"  • WOE/IV: {'✅' if enable_woe_iv else '❌'}")

        # Ejecutar selección categórica (detallado en log)
        if enable_woe_iv:
            selected_vars, woe_iv_results, ranking = SelectVarsCategoricals.select_categorical_variables(
                df, cat_cols, target=self.config['data']['target_column'],
                iv_threshold=iv_threshold, verbose=True  # Detallado en log
            )
            
            # PASO NUEVO: Conversión a WOE
            if selected_vars:
                df_woe, woe_columns = SelectVarsCategoricals.convert_categorical_to_woe(
                    df, selected_vars, woe_iv_results, 
                    target=self.config['data']['target_column'],
                    verbose=True, logger=self.logger
                )
                self.logger.info(f"\n📊 RESULTADO SELECCIÓN CATEGÓRICA:")
                self.logger.info(f"   ✅ Variables iniciales: {len(cat_cols)}")
                self.logger.info(f"   ✅ Variables seleccionadas: {len(selected_vars)}")
                self.logger.info(f"   🔄 Variables convertidas a WOE: {len(woe_columns)}")
                self.logger.info(f"   ❌ Variables descartadas: {len(cat_cols) - len(selected_vars)}")
                self.logger.info(f"   🏆 Variables WOE finales: {woe_columns}")
                
                self._log_step_end("Categorical variables selection", 6.5, f"Selected: {len(woe_columns)} WOE variables")
                return woe_columns, df_woe
            else:
                self.logger.info(f"\n📊 RESULTADO SELECCIÓN CATEGÓRICA:")
                self.logger.info(f"   ✅ Variables iniciales: {len(cat_cols)}")
                self.logger.info(f"   ❌ Variables seleccionadas: 0 (ninguna superó IV threshold)")
                self.logger.info(f"   ❌ Variables descartadas: {len(cat_cols)}")
                
                self._log_step_end("Categorical variables selection", 6.5, "No variables selected")
                return [], df
        else:
            # Solo Chi-square si WOE/IV está deshabilitado
            selected_vars = SelectVarsCategoricals.chi_square_test(
                df, cat_cols, target=self.config['data']['target_column'],
                threshold=chi_square_threshold, verbose=True  # Detallado en log
            )
            
            self.logger.info(f"\n📊 RESULTADO SELECCIÓN CATEGÓRICA:")
            self.logger.info(f"   ✅ Variables iniciales: {len(cat_cols)}")
            self.logger.info(f"   ✅ Variables seleccionadas: {len(selected_vars)}")
            self.logger.info(f"   ❌ Variables descartadas: {len(cat_cols) - len(selected_vars)}")

            if selected_vars:
                self.logger.info(f"   🏆 Variables finales: {selected_vars}")

            self._log_step_end("Categorical variables selection", 6.5, f"Selected: {len(selected_vars)}")
            return selected_vars, df

    def pca_lda_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        self._log_step_start("PCA diagnostics and LDA importance", 7)

        # PROTEGER int_rate en PCA+LDA
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if protected_var in num_cols:
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' incluida en análisis PCA+LDA")

        cfg = self.config['selection']
        
        # Determinar el número de componentes PCA según el modo configurado
        pca_cfg = cfg['pca']
        pca_mode = pca_cfg.get('mode', 'variance_percentage')
        
        if pca_mode == 'fixed_components':
            n_components = int(pca_cfg.get('n_components', 35))
            self.logger.info(f"🔧 Modo PCA: Componentes fijos ({n_components})")
        elif pca_mode == 'variance_percentage':
            variance_threshold = float(pca_cfg.get('variance_threshold', 0.95))
            n_components, actual_variance = calculate_pca_components_by_variance(
                df, num_cols, variance_threshold
            )
            self.logger.info(f"🔧 Modo PCA: Porcentaje de varianza ({variance_threshold*100:.1f}%)")
            self.logger.info(f"   📊 Componentes calculados: {n_components}")
            self.logger.info(f"   📈 Varianza real explicada: {actual_variance*100:.2f}%")
        else:
            # Fallback a modo fijo si el modo no es válido
            n_components = int(pca_cfg.get('n_components', 35))
            self.logger.warning(f"⚠️ Modo PCA inválido '{pca_mode}', usando componentes fijos ({n_components})")
        
        if cfg['pca'].get('enable', True):
            plot_variance = bool(cfg['pca'].get('plot_variance', False))
            self.logger.info(f"Running PCA with {n_components} components, plot_variance: {plot_variance}")
            
            SelectVarsNumerics.pca_con_grafico(
                df,
                num_cols,
                plot_variance=plot_variance,
                verbose=self.verbose,
            )
            self.logger.info("PCA completed successfully")
        
        if not cfg['lda_importance'].get('enable', True):
            self.logger.info("LDA importance selection skipped")
            self._log_step_end("PCA diagnostics and LDA importance", 7, "Skipped")
            return None
        
        importance_threshold = float(cfg['lda_importance'].get('importance_threshold', 0.05))
        plot_importance = bool(cfg['lda_importance'].get('plot_importance', False))
        
        self.logger.info(f"Running LDA importance selection (threshold: {importance_threshold})")
        
        vars_pca_lda, _ = SelectVarsNumerics.pca_lda_importancia(
            df,
            num_cols,
            target=self.config['data']['target_column'],
            n_pca_components=n_components,  # Usar el número calculado dinámicamente
            umbral_importancia=importance_threshold,
            plot_graph=plot_importance,
            verbose=self.verbose,
        )
        
        # ASEGURAR QUE int_rate ESTÉ EN LOS RESULTADOS FINALES
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if vars_pca_lda and protected_var not in vars_pca_lda and protected_var in df.columns:
            vars_pca_lda.append(protected_var)
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en resultados PCA+LDA")

        if vars_pca_lda:
            self.logger.info(f"LDA importance selection completed. Selected variables: {vars_pca_lda}")
        else:
            self.logger.warning("No variables selected by LDA importance")

        self._log_step_end("PCA diagnostics and LDA importance", 7, f"Selected: {len(vars_pca_lda) if vars_pca_lda else 0}")
        return vars_pca_lda

    def anova_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        self._log_step_start("ANOVA F-test selection", 8)

        # PROTEGER int_rate en ANOVA
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if protected_var in num_cols:
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' incluida en análisis ANOVA")

        cfg = self.config['selection']
        if not cfg['anova'].get('enable', True):
            self.logger.info("ANOVA selection skipped")
            self._log_step_end("ANOVA F-test selection", 8, "Skipped")
            return None
        
        min_f_score = cfg['anova'].get('min_f_score', None)
        max_p_value = cfg['anova'].get('max_p_value', 0.05)
        top_n = cfg['anova'].get('top_n', None)
        
        self.logger.info(f"ANOVA parameters: min_f_score={min_f_score}, max_p_value={max_p_value}, top_n={top_n}")
        
        en_anova_vars, _ = SelectVarsNumerics.anova_feature_selection(
            df,
            num_cols,
            target=self.config['data']['target_column'],
            min_f_score=min_f_score,
            max_p_value=max_p_value,
            top_n=top_n,
            verbose=self.verbose,
        )
        
        # ASEGURAR QUE int_rate ESTÉ EN LOS RESULTADOS FINALES
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if en_anova_vars and protected_var not in en_anova_vars and protected_var in df.columns:
            en_anova_vars.append(protected_var)
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en resultados ANOVA")

        if en_anova_vars:
            self.logger.info(f"ANOVA selection completed. Selected variables: {en_anova_vars}")
        else:
            self.logger.warning("No variables selected by ANOVA")

        self._log_step_end("ANOVA F-test selection", 8, f"Selected: {len(en_anova_vars) if en_anova_vars else 0}")
        return en_anova_vars

    def correlation_redundancy(self, df: pd.DataFrame, vars_list: Optional[List[str]], label: str) -> Optional[List[str]]:
        self._log_step_start(f"Correlation redundancy removal - {label}", 9)

        if not vars_list:
            self.logger.info(f"No variables provided for {label} correlation analysis")
            self._log_step_end(f"Correlation redundancy removal - {label}", 9, "No variables")
            return None

        # PROTEGER int_rate en análisis de correlación
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if protected_var in vars_list:
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' incluida en análisis de correlación {label}")

        cfg = self.config['selection']['correlation_redundancy']
        if not cfg.get('enable', True):
            self.logger.info("Correlation redundancy removal skipped")
            self._log_step_end(f"Correlation redundancy removal - {label}", 9, "Skipped")
            return vars_list
        
        threshold = float(cfg.get('threshold', 0.6))
        plot_heatmap = bool(cfg.get('plot_heatmap', False))
        
        self.logger.info(f"Correlation threshold: {threshold}, plot_heatmap: {plot_heatmap}")
        self.logger.info(f"Starting with {len(vars_list)} variables for {label}")
        
        vars_corr, _ = SelectVarsNumerics.correlacion_feature_selection(
            df,
            vars_list,
            target=self.config['data']['target_column'],
            threshold=threshold,
            plot_heatmap=plot_heatmap,
            verbose=self.verbose,
        )
        
        # ASEGURAR QUE int_rate NO SEA ELIMINADA POR CORRELACIÓN
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')
        if vars_corr and protected_var not in vars_corr and protected_var in vars_list:
            vars_corr.append(protected_var)
            self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en resultados de correlación {label}")

        removed_count = len(vars_list) - len(vars_corr)
        self.logger.info(f"Removed {removed_count} redundant variables")
        self.logger.info(f"Remaining variables: {vars_corr}")

        self._log_step_end(f"Correlation redundancy removal - {label}", 9, f"Remaining: {len(vars_corr)}")
        return vars_corr

    def save_outputs(self, df: pd.DataFrame, vars_pca_lda: Optional[List[str]], vars_anova: Optional[List[str]]) -> None:
        self._log_step_start("Save outputs", 10)

        self._ensure_output_dir()
        out_dir = self.config['data']['output_dir']
        target = self.config['data']['target_column']

        # ASEGURAR QUE int_rate ESTÉ EN LOS ARCHIVOS FINALES
        protected_var = self.config.get('int_rate_protection', {}).get('protected_variable', 'int_rate')

        if vars_pca_lda:
            # Forzar inclusión de int_rate si no está
            if protected_var not in vars_pca_lda and protected_var in df.columns:
                vars_pca_lda = vars_pca_lda + [protected_var]
                self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en archivo PCA+LDA")

            out1 = os.path.join(out_dir, self.config['data']['output_pca_lda_csv'])
            df_pca_lda = df[vars_pca_lda + [target]].copy()
            df_pca_lda.to_csv(out1, index=False)
            self.logger.info(f"PCA+LDA dataset saved: {out1}")
            self.logger.info(f"  Shape: {df_pca_lda.shape}")
            self.logger.info(f"  Variables: {vars_pca_lda}")
            self.logger.info(f"  Target: {target}")
            if protected_var in vars_pca_lda:
                self.logger.info(f"  ✅ {protected_var} INCLUIDA en dataset final")

        if vars_anova:
            # Forzar inclusión de int_rate si no está
            if protected_var not in vars_anova and protected_var in df.columns:
                vars_anova = vars_anova + [protected_var]
                self.logger.info(f"🛡️ Variable protegida '{protected_var}' FORZADA en archivo ANOVA")

            out2 = os.path.join(out_dir, self.config['data']['output_anova_csv'])
            df_anova = df[vars_anova + [target]].copy()
            df_anova.to_csv(out2, index=False)
            self.logger.info(f"ANOVA dataset saved: {out2}")
            self.logger.info(f"  Shape: {df_anova.shape}")
            self.logger.info(f"  Variables: {vars_anova}")
            self.logger.info(f"  Target: {target}")
            if protected_var in vars_anova:
                self.logger.info(f"  ✅ {protected_var} INCLUIDA en dataset final")

        self._log_step_end("Save outputs", 10, f"Files saved: {len([v for v in [vars_pca_lda, vars_anova] if v])}")

    # ------------------------------- Runner --------------------------------
    def run(self) -> None:
        # Solo mostrar header en terminal
        print_pipeline_header()
        
        # Todo lo demás va al log
        self.logger.info("🚀 INICIANDO EDA PIPELINE...")
        self.logger.info("Starting EDA Pipeline execution...")
        start_time = datetime.now()

        total_steps = 10
        current_step = 0

        # Configurar captura global de warnings para el pipeline
        with warnings.catch_warnings(record=True) as pipeline_warnings:
            warnings.simplefilter("always")
            
            try:
                # PASO 1: Carga de datos
                print_section_progress(current_step, total_steps, section_name='📥 Cargando datos', suffix='')
                current_step += 1
                df = self.load_data()
                time.sleep(0.2)  # Pausa breve para ver la barra
                print_section_progress(current_step, total_steps, section_name='📥 Cargando datos', suffix='Completado')

                # PASO 2: Creación de target
                print_section_progress(current_step, total_steps, section_name='🎯 Creando target', suffix='')
                current_step += 1
                df = self.build_target(df)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🎯 Creando target', suffix='Completado')

                # PASO 3: Limpieza de columnas
                print_section_progress(current_step, total_steps, section_name='🧹 Limpiando columnas', suffix='')
                current_step += 1
                df = self.clean_columns(df)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🧹 Limpiando columnas', suffix='Completado')

                # PASO 4: Feature engineering
                print_section_progress(current_step, total_steps, section_name='🔧 Feature Engineering', suffix='')
                current_step += 1
                df = self.derive_features(df)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🔧 Feature Engineering', suffix='Completado')

                # PASO 5: Separación inteligente de variables
                print_section_progress(current_step, total_steps, section_name='📊 Separando variables', suffix='')
                current_step += 1
                cat_cols, num_cols = self.separate_variables(df)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='📊 Separando variables', suffix='Completado')

                # PASO 6: Protección de int_rate
                print_section_progress(current_step, total_steps, section_name='🛡️ Protegiendo int_rate', suffix='')
                current_step += 1
                protection_config = self.config.get('int_rate_protection', {})
                if protection_config.get('enable_protection', True):
                    protection_result = proteger_int_rate(
                        df=df,
                        variables_numericas=num_cols,
                        variables_categoricas=cat_cols,
                        protected_var=protection_config.get('protected_variable', 'int_rate'),
                        correlation_threshold=protection_config.get('correlation_threshold', 0.7),
                        verbose=False,  # Solo log, no prints en terminal
                        logger=self.logger
                    )

                    # Actualizar listas con variables protegidas
                    num_cols = protection_result['variables_numericas_finales']
                    cat_cols = protection_result['variables_categoricas_finales']
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🛡️ Protegiendo int_rate', suffix='Completado')

                # PASO 7: Selección categórica → Conversión a WOE
                print_section_progress(current_step, total_steps, section_name='📋 Selección categórica', suffix='')
                current_step += 1
                cat_woe_cols, df = self.select_categorical_variables(df, cat_cols) if cat_cols else ([], df)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='📋 Selección categórica', suffix='Completado')

                # PASO 8: Selección numérica
                print_section_progress(current_step, total_steps, section_name='🔢 Selección numérica', suffix='')
                current_step += 1
                num_sel = self.select_numeric_variables(df, num_cols)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🔢 Selección numérica', suffix='Completado')

                # PASO 9: PCA+LDA y ANOVA (incluye numéricas + WOE)
                print_section_progress(current_step, total_steps, section_name='🤖 Aplicando modelos', suffix='')
                current_step += 1
                # Combinar variables numéricas + WOE para ambos flujos
                all_numeric_vars = num_sel + cat_woe_cols
                vars_pca_lda = self.pca_lda_selection(df, all_numeric_vars)
                vars_anova = self.anova_selection(df, all_numeric_vars)
                time.sleep(0.2)
                print_section_progress(current_step, total_steps, section_name='🤖 Aplicando modelos', suffix='Completado')

                # PASO 10: Correlación y guardado
                print_section_progress(current_step, total_steps, section_name='💾 Guardando resultados', suffix='')
                current_step += 1
                vars_corr_pca_lda = self.correlation_redundancy(df, vars_pca_lda, label='PCA+LDA') if vars_pca_lda else None
                vars_corr_anova = self.correlation_redundancy(df, vars_anova, label='ANOVA') if vars_anova else None

                # Variables finales (ya incluyen numéricas + WOE)
                final_vars_pca_lda = vars_corr_pca_lda if vars_corr_pca_lda else []
                final_vars_anova = vars_corr_anova if vars_corr_anova else []

                self.save_outputs(df, final_vars_pca_lda, final_vars_anova)
                time.sleep(0.2)
                print_section_progress(total_steps, total_steps, section_name='💾 Guardando resultados', suffix='Completado')

                # Mostrar resumen final en log
                end_time = datetime.now()
                duration = end_time - start_time

                # Resumen solo en log (detallado)
                self.logger.info(f"\n🎯 RESUMEN FINAL:")
                self.logger.info(f"   📊 Dataset: {df.shape[0]:,} filas × {df.shape[1]} columnas")
                self.logger.info(f"   ⏱️  Tiempo: {duration}")
                self.logger.info(f"   ✅ Variables PCA+LDA: {len(final_vars_pca_lda)}")
                self.logger.info(f"   ✅ Variables ANOVA: {len(final_vars_anova)}")
                self.logger.info(f"   🛡️  int_rate PROTEGIDO: ✅")
                self.logger.info(f"   📁 Archivos guardados: df_pca_lda.csv, df_anova.csv")
                self.logger.info(f"   📋 Log detallado: log_EDA.txt")

                # Log de warnings capturados durante la ejecución
                if pipeline_warnings:
                    self.logger.warning(f"\n⚠️  WARNINGS CAPTURADOS DURANTE LA EJECUCIÓN:")
                    for warning in pipeline_warnings:
                        self.logger.warning(f"  {warning.category.__name__}: {warning.message}")
                
                self.logger.info("")
                self.logger.info("=" * 80)
                self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
                self.logger.info("=" * 80)
                self.logger.info(f"Total execution time: {duration}")
                self.logger.info(f"Final dataset shape: {df.shape}")
                self.logger.info(f"Variables numéricas PCA+LDA: {len(vars_corr_pca_lda) if vars_corr_pca_lda else 0}")
                self.logger.info(f"Variables numéricas ANOVA: {len(vars_corr_anova) if vars_corr_anova else 0}")
                self.logger.info(f"Variables categóricas WOE: {len(cat_woe_cols)}")
                self.logger.info(f"TOTAL PCA+LDA (Num + Cat): {len(final_vars_pca_lda)}")
                self.logger.info(f"TOTAL ANOVA (Num + Cat): {len(final_vars_anova)}")
                self.logger.info("=" * 80)
                
                # Resumen limpio en terminal
                files_saved = []
                if final_vars_pca_lda:
                    files_saved.append("df_pca_lda.csv")
                if final_vars_anova:
                    files_saved.append("df_anova.csv")
                    
                print_pipeline_summary(
                    execution_time=duration,
                    total_rows=df.shape[0],
                    pca_vars=len(final_vars_pca_lda),
                    anova_vars=len(final_vars_anova),
                    files_saved=files_saved
                )
                
            except Exception as e:
                self.logger.error(f"Pipeline failed with error: {str(e)}")
                self.logger.error("=" * 80)
                raise


def main():
    # Prueba rápida de la barra de progreso antes de ejecutar el pipeline completo
    print("🔧 Probando barra de progreso...")
    total_steps = 5
    for i in range(total_steps):
        print_section_progress(i, total_steps, section_name=f'Paso {i+1}', suffix='')
        time.sleep(0.1)
        print_section_progress(i+1, total_steps, section_name=f'Paso {i+1}', suffix='Completado')
        time.sleep(0.1)
    print("✅ Barra de progreso funcionando correctamente!")

    parser = argparse.ArgumentParser(description='Config-driven EDA pipeline')
    default_cfg = os.path.join(os.path.dirname(__file__), 'engine_TFM', 'config_eda.yml')
    parser.add_argument('--config', type=str, default=default_cfg)
    args = parser.parse_args()

    try:
        pipeline = EDAPipeline(args.config)
        pipeline.run()
    except Exception as e:
        print(f"❌ Pipeline execution failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()


