import os
import argparse
import pandas as pd
from typing import List, Optional, Tuple

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
            'logging': {'verbose': True}
        }
        self.config = load_config(config_path, defaults=defaults)
        self.verbose = bool(self.config.get('logging', {}).get('verbose', True))

    def _print(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _ensure_output_dir(self) -> None:
        out_dir = self.config['data']['output_dir']
        os.makedirs(out_dir, exist_ok=True)

    # --------------------------- Pipeline steps ----------------------------
    def load_data(self) -> pd.DataFrame:
        section("1) Load data")
        path = self.config['data']['input_csv']
        df = pd.read_csv(path)
        self._print(f"[DATA] Loaded shape: {df.shape}")
        return df

    def build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        section("2) Build target")
        data_cfg = self.config['data']
        good = data_cfg['allowed_status'].get('good', [])
        bad = data_cfg['allowed_status'].get('bad', [])
        df = build_binary_target(
            df,
            status_column=data_cfg['status_column'],
            bad_status=bad,
            good_status=good,
            target_column=data_cfg['target_column'],
            drop_missing_status=bool(data_cfg.get('drop_missing_status', True)),
            verbose=self.verbose
        )
        return df

    def clean_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        section("3) Clean columns")
        cfg = self.config['cleaning']
        df2 = df.copy()
        if cfg.get('enable_drop_null_columns', True):
            thr = float(cfg.get('drop_null_threshold', 0.95))
            cols_muchos_nulos = df2.columns[df2.isnull().mean() > thr].tolist()
            self._print(f"[CLEAN] Columns with >{thr*100:.0f}% nulls: {len(cols_muchos_nulos)}")
            df2 = safe_drop_columns(df2, cols_muchos_nulos, verbose=self.verbose)
        if cfg.get('enable_drop_irrelevant_columns', True):
            df2 = safe_drop_columns(df2, cfg.get('drop_columns', []), verbose=self.verbose)
        if cfg.get('enable_drop_ex_post', True):
            df2 = safe_drop_columns(df2, cfg.get('drop_ex_post_columns', []), verbose=self.verbose)
        self._print(f"[CLEAN] Resulting shape: {df2.shape}")
        return df2

    def derive_features(self, df: pd.DataFrame) -> pd.DataFrame:
        section("4) Derive features")
        toggles = self.config.get('features', {}).get('derive', {})
        df2 = derive_features(df, toggles, verbose=self.verbose)
        self._print(f"[FEATURES] Shape: {df2.shape}")
        return df2

    def separate_variables(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        section("5) Separate variables by type")
        vars_dict = Exploracion.separar_variables(df, target=self.config['data']['target_column'])
        cat_cols = vars_dict['categoricas']
        num_cols = vars_dict['numericas']
        self._print(f"[VARS] Categorical: {len(cat_cols)} | Numerical: {len(num_cols)}")
        return cat_cols, num_cols

    def select_numeric_variables(self, df: pd.DataFrame, num_cols: List[str]) -> List[str]:
        section("6) Numeric summary and filtering")
        resumen = Exploracion.resumen_numericas(df, num_cols)
        cfg = self.config['selection']
        selected = num_cols
        if cfg['filter_missing'].get('enable', True):
            selected = SelectVarsNumerics.filtrar_por_nulos(
                df,
                resumen.loc[selected],
                target=self.config['data']['target_column'],
                umbral=float(cfg['filter_missing'].get('threshold', 0.5)),
                rescatar=bool(cfg['filter_missing'].get('rescue_by_correlation', True)),
                min_corr=float(cfg['filter_missing'].get('min_corr', 0.1)),
                min_samples=int(cfg['filter_missing'].get('min_samples', 30)),
            )
        if cfg['filter_cv'].get('enable', True):
            resumen_sel = resumen.loc[selected]
            selected = SelectVarsNumerics.filtrar_por_cv(
                df,
                resumen_sel,
                target=self.config['data']['target_column'],
                umbral=float(cfg['filter_cv'].get('threshold', 0.1)),
                rescatar=bool(cfg['filter_cv'].get('rescue_by_correlation', True)),
                min_corr=float(cfg['filter_cv'].get('min_corr', 0.1)),
                min_samples=int(cfg['filter_cv'].get('min_samples', 30)),
            )
        self._print(f"[FILTER] Numerical after filters: {len(selected)}")
        return selected

    def pca_lda_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        section("7) PCA diagnostics and LDA importance")
        cfg = self.config['selection']
        if cfg['pca'].get('enable', True):
            SelectVarsNumerics.pca_con_grafico(
                df,
                num_cols,
                plot_variance=bool(cfg['pca'].get('plot_variance', False)),
                verbose=self.verbose,
            )
        if not cfg['lda_importance'].get('enable', True):
            self._print("[PCA+LDA] Skipped LDA importance selection.")
            return None
        vars_pca_lda, _ = SelectVarsNumerics.pca_lda_importancia(
            df,
            num_cols,
            target=self.config['data']['target_column'],
            n_pca_components=int(cfg['pca'].get('n_components', 35)),
            umbral_importancia=float(cfg['lda_importance'].get('importance_threshold', 0.05)),
            plot_graph=bool(cfg['lda_importance'].get('plot_importance', False)),
            verbose=self.verbose,
        )
        return vars_pca_lda

    def anova_selection(self, df: pd.DataFrame, num_cols: List[str]) -> Optional[List[str]]:
        section("8) ANOVA F-test selection")
        cfg = self.config['selection']
        if not cfg['anova'].get('enable', True):
            self._print("[ANOVA] Skipped.")
            return None
        en_anova_vars, _ = SelectVarsNumerics.anova_feature_selection(
            df,
            num_cols,
            target=self.config['data']['target_column'],
            min_f_score=cfg['anova'].get('min_f_score', None),
            max_p_value=cfg['anova'].get('max_p_value', 0.05),
            top_n=cfg['anova'].get('top_n', None),
            verbose=self.verbose,
        )
        self._print(f"[ANOVA] Selected: {len(en_anova_vars) if en_anova_vars is not None else 0}")
        return en_anova_vars

    def correlation_redundancy(self, df: pd.DataFrame, vars_list: Optional[List[str]], label: str) -> Optional[List[str]]:
        section(f"9) Correlation redundancy – {label}")
        if not vars_list:
            self._print(f"[CORR] No variables provided for {label}.")
            return None
        cfg = self.config['selection']['correlation_redundancy']
        if not cfg.get('enable', True):
            self._print("[CORR] Skipped redundancy removal.")
            return vars_list
        vars_corr, _ = SelectVarsNumerics.correlacion_feature_selection(
            df,
            vars_list,
            target=self.config['data']['target_column'],
            threshold=float(cfg.get('threshold', 0.6)),
            plot_heatmap=bool(cfg.get('plot_heatmap', False)),
            verbose=self.verbose,
        )
        self._print(f"[CORR] Remaining after redundancy – {label}: {len(vars_corr)}")
        return vars_corr

    def save_outputs(self, df: pd.DataFrame, vars_pca_lda: Optional[List[str]], vars_anova: Optional[List[str]]) -> None:
        section("10) Save outputs")
        self._ensure_output_dir()
        out_dir = self.config['data']['output_dir']
        target = self.config['data']['target_column']
        if vars_pca_lda:
            out1 = os.path.join(out_dir, self.config['data']['output_pca_lda_csv'])
            df_pca_lda = df[vars_pca_lda + [target]].copy()
            df_pca_lda.to_csv(out1, index=False)
            self._print(f"[SAVE] PCA+LDA CSV: {out1} | cols={len(vars_pca_lda)}")
        if vars_anova:
            out2 = os.path.join(out_dir, self.config['data']['output_anova_csv'])
            df_anova = df[vars_anova + [target]].copy()
            df_anova.to_csv(out2, index=False)
            self._print(f"[SAVE] ANOVA CSV: {out2} | cols={len(vars_anova)}")

    # ------------------------------- Runner --------------------------------
    def run(self) -> None:
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


def main():
    parser = argparse.ArgumentParser(description='Config-driven EDA pipeline')
    default_cfg = os.path.join(os.path.dirname(__file__), 'engine_TFM', 'config_eda.yml')
    parser.add_argument('--config', type=str, default=default_cfg)
    args = parser.parse_args()
    pipeline = EDAPipeline(args.config)
    pipeline.run()


if __name__ == '__main__':
    main()


