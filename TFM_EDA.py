import os
import argparse
import pandas as pd
from typing import List, Optional, Tuple
import logging
from datetime import datetime

from engine_TFM.engine_eda import Exploracion, SelectVarsNumerics
from engine_TFM.utils import (
    load_config,
    section,
    safe_drop_columns,
    build_binary_target,
    derive_features,
)


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
                'pca': {'enable': True, 'n_components': 35, 'plot_variance': False},
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
        
        # Console handler (simple)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log pipeline start
        self.logger.info("=" * 80)
        self.logger.info("EDA PIPELINE STARTED")
        self.logger.info("=" * 80)
        self.logger.info(f"Configuration file: {self.config.get('_config_path', 'Unknown')}")
        self.logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Log file: {log_file}")
        self.logger.info("=" * 80)

    def _print(self, msg: str, level: str = "INFO") -> None:
        """Enhanced print method that logs to both console and file"""
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
            df = pd.read_csv(path)
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
            verbose=self.verbose
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
            df2 = safe_drop_columns(df2, cols_muchos_nulos, verbose=self.verbose)
            self.logger.info(f"After dropping null columns: {df2.shape}")
        
        if cfg.get('enable_drop_irrelevant_columns', True):
            drop_cols = cfg.get('drop_columns', [])
            self.logger.info(f"Dropping irrelevant columns ({len(drop_cols)}): {drop_cols}")
            df2 = safe_drop_columns(df2, drop_cols, verbose=self.verbose)
            self.logger.info(f"After dropping irrelevant columns: {df2.shape}")
        
        if cfg.get('enable_drop_ex_post', True):
            ex_post_cols = cfg.get('drop_ex_post_columns', [])
            self.logger.info(f"Dropping ex-post columns ({len(ex_post_cols)}): {ex_post_cols}")
            df2 = safe_drop_columns(df2, ex_post_cols, verbose=self.verbose)
            self.logger.info(f"After dropping ex-post columns: {df2.shape}")
        
        self._log_step_end("Clean columns", 3, f"Final shape: {df2.shape}")
        return df2

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._log_step_start("Derive features", 4)
        toggles = self.config.get('features', {}).get('derive', {})
        
        self.logger.info(f"Feature derivation toggles:")
        for feature, enabled in toggles.items():
            self.logger.info(f"  {feature}: {'✓' if enabled else '✗'}")
        
        df2 = derive_features(df, toggles, verbose=self.verbose)
        
        # Log new columns added
        new_cols = set(df2.columns) - set(df.columns)
        if new_cols:
            self.logger.info(f"New derived columns ({len(new_cols)}): {list(new_cols)}")
        else:
            self.logger.info("No new columns were derived")
        
        self._log_step_end("Derive features", 4, f"Shape: {df2.shape}")
        return df2

    def separate_variables(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        self._log_step_start("Separate variables by type", 5)
        
        vars_dict = Exploracion.separar_variables(df, target=self.config['data']['target_column'])
        cat_cols = vars_dict['categoricas']
        num_cols = vars_dict['numericas']
        
        self.logger.info(f"Categorical variables ({len(cat_cols)}): {cat_cols}")
        self.logger.info(f"Numerical variables ({len(num_cols)}): {num_cols}")

        # Log some statistics about categorical variables
        if cat_cols:
            self.logger.info("Categorical variables summary:")
            for col in cat_cols[:5]:  # Show first 5
                if col in df.columns:
                    unique_vals = df[col].nunique()
                    self.logger.info(f"  {col}: {unique_vals} unique values")

        # ⚠️ WARNING about ignored categorical variables
        if cat_cols:
            self.logger.warning(f"⚠️  CRITICAL WARNING: {len(cat_cols)} categorical variables DETECTED but NOT PROCESSED!")
            self.logger.warning("   These variables may contain CRUCIAL predictive information:")
            important_cats = [col for col in cat_cols if col in ['grade', 'sub_grade', 'purpose', 'home_ownership', 'verification_status']]
            if important_cats:
                self.logger.warning(f"   🔴 Important categoricals ignored: {important_cats}")
            self.logger.warning("   💡 RECOMMENDATION: Implement categorical variable selection!")

        self._log_step_end("Separate variables by type", 5, f"Categorical: {len(cat_cols)}, Numerical: {len(num_cols)}")
        return cat_cols, num_cols

    def select_numeric_variables(self, df: pd.DataFrame, num_cols: List[str]) -> List[str]:
        self._log_step_start("Numeric summary and filtering", 6)
        
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
        
        self.logger.info(f"Final selected numerical variables: {selected}")
        self._log_step_end("Numeric summary and filtering", 6, f"Selected: {len(selected)}")
        return selected

    def pca_lda_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        self._log_step_start("PCA diagnostics and LDA importance", 7)
        
        cfg = self.config['selection']
        
        if cfg['pca'].get('enable', True):
            n_components = int(cfg['pca'].get('n_components', 35))
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
            n_pca_components=int(cfg['pca'].get('n_components', 35)),
            umbral_importancia=importance_threshold,
            plot_graph=plot_importance,
            verbose=self.verbose,
        )
        
        if vars_pca_lda:
            self.logger.info(f"LDA importance selection completed. Selected variables: {vars_pca_lda}")
        else:
            self.logger.warning("No variables selected by LDA importance")
        
        self._log_step_end("PCA diagnostics and LDA importance", 7, f"Selected: {len(vars_pca_lda) if vars_pca_lda else 0}")
        return vars_pca_lda

    def anova_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        self._log_step_start("ANOVA F-test selection", 8)
        
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
        
        if vars_pca_lda:
            out1 = os.path.join(out_dir, self.config['data']['output_pca_lda_csv'])
            df_pca_lda = df[vars_pca_lda + [target]].copy()
            df_pca_lda.to_csv(out1, index=False)
            self.logger.info(f"PCA+LDA dataset saved: {out1}")
            self.logger.info(f"  Shape: {df_pca_lda.shape}")
            self.logger.info(f"  Variables: {vars_pca_lda}")
            self.logger.info(f"  Target: {target}")
        
        if vars_anova:
            out2 = os.path.join(out_dir, self.config['data']['output_anova_csv'])
            df_anova = df[vars_anova + [target]].copy()
            df_anova.to_csv(out2, index=False)
            self.logger.info(f"ANOVA dataset saved: {out2}")
            self.logger.info(f"  Shape: {df_anova.shape}")
            self.logger.info(f"  Variables: {vars_anova}")
            self.logger.info(f"  Target: {target}")
        
        self._log_step_end("Save outputs", 10, f"Files saved: {len([v for v in [vars_pca_lda, vars_anova] if v])}")

    # ------------------------------- Runner --------------------------------
    def run(self) -> None:
        self.logger.info("Starting EDA Pipeline execution...")
        start_time = datetime.now()
        
        try:
            df = self.load_data()
            df = self.build_target(df)
            df = self.clean_columns(df)
            df = self.derive_features(df)
            cat_cols, num_cols = self.separate_variables(df)
            num_sel = self.select_numeric_variables(df, num_cols)
            vars_pca_lda = self.pca_lda_selection(df, num_sel)
            vars_anova = self.anova_selection(df, num_sel)
            vars_corr_pca_lda = self.correlation_redundancy(df, vars_pca_lda, label='PCA+LDA') if vars_pca_lda else None
            vars_corr_anova = self.correlation_redundancy(df, vars_anova, label='ANOVA') if vars_anova else None
            self.save_outputs(df, vars_corr_pca_lda, vars_corr_anova)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            self.logger.info("")
            self.logger.info("=" * 80)
            self.logger.info("EDA PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            self.logger.info(f"Total execution time: {duration}")
            self.logger.info(f"Final dataset shape: {df.shape}")
            self.logger.info(f"PCA+LDA variables: {len(vars_corr_pca_lda) if vars_corr_pca_lda else 0}")
            self.logger.info(f"ANOVA variables: {len(vars_corr_anova) if vars_corr_anova else 0}")
            self.logger.info("=" * 80)
            
        except Exception as e:
            self.logger.error(f"Pipeline failed with error: {str(e)}")
            self.logger.error("=" * 80)
            raise


def main():
    parser = argparse.ArgumentParser(description='Config-driven EDA pipeline')
    default_cfg = os.path.join(os.path.dirname(__file__), 'engine_TFM', 'config_eda.yml')
    parser.add_argument('--config', type=str, default=default_cfg)
    args = parser.parse_args()
    
    try:
        pipeline = EDAPipeline(args.config)
        pipeline.run()
    except Exception as e:
        print(f"Pipeline execution failed: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()


