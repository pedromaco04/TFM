# utils.py
# Funciones genéricas y utilidades para el engine TFM

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
import seaborn as sns
import networkx as nx
from scipy.stats import pearsonr
import os
import json
from datetime import datetime
from typing import Dict, Any, List
import logging

def log_message(logger, message):
    """Helper para logging compatible"""
    if logger:
        logger.info(message)
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
    precision_recall_curve, average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, roc_curve
)

try:
    import yaml
except Exception:  # yaml optional; handle gracefully
    yaml = None

def correlacion_con_target(df, cols, target, min_samples=30):
    """
    Calcula la correlación de cada columna en cols con el target, solo si hay suficientes datos válidos.
    Devuelve un DataFrame con la correlación absoluta.
    """
    correlaciones = {}
    for col in cols:
        try:
            serie_valida = df[[col, target]].dropna()
            if len(serie_valida) >= min_samples:
                correlaciones[col] = serie_valida.corr().iloc[0,1]
        except Exception as e:
            continue
    
    if not correlaciones:
        return pd.DataFrame(columns=['correlacion_con_target'])
    
    df_corr = pd.DataFrame.from_dict(correlaciones, orient='index', columns=['correlacion_con_target'])
    df_corr['correlacion_con_target'] = df_corr['correlacion_con_target'].abs()
    df_corr = df_corr.sort_values(by='correlacion_con_target', ascending=False)
    return df_corr 

def plot_pca_variance(varianza_acumulada):
    """
    Grafica la varianza acumulada de un PCA.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(varianza_acumulada)+1), varianza_acumulada, marker='o')
    plt.xlabel('Número de Componentes Principales')
    plt.ylabel('Varianza Acumulada')
    plt.title('PCA - Varianza Acumulada')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

def plot_importancia_variables(df_importancia, top_n=20):
    """
    Grafica la importancia de las variables (no acumulada) tras PCA+LDA.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    top = df_importancia.head(top_n)
    plt.plot(top['importancia'].values, marker='o')
    plt.xticks(range(len(top)), top['variable'], rotation=45, ha='right')
    plt.ylabel('Importancia de las Variables')
    plt.title('PCA * LDA - Importancia de las Variables')
    plt.grid(True)
    plt.tight_layout()
    plt.show() 

def anova_f_scores(df, num_cols, target):
    """
    Calcula el score ANOVA F para cada variable numérica respecto a una target categórica.
    Devuelve un DataFrame ordenado por score.
    """
    X = df[num_cols].fillna(df[num_cols].median())
    y = df[target]
    f_scores, p_values = f_classif(X, y)
    df_anova = pd.DataFrame({
        'variable': num_cols,
        'f_score': f_scores,
        'p_value': p_values
    }).sort_values(by='f_score', ascending=False).reset_index(drop=True)
    return df_anova 

def plot_corr_heatmap(corr_matrix, annot=True):
    """
    Grafica un heatmap de la matriz de correlación.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=annot, fmt=".2f", cmap='coolwarm', square=True, linewidths=0.5)
    plt.title("Matriz de Correlación entre Variables Numéricas")
    plt.tight_layout()
    plt.show()


def select_least_redundant_vars(corr_matrix, df, target, threshold=0.6):
    """
    Selecciona variables menos redundantes usando grafo de correlaciones altas.
    Para cada grupo de variables correlacionadas (>threshold), se queda con la más correlacionada con el target.
    Devuelve la lista de variables seleccionadas.
    """
    G = nx.Graph()
    for i in corr_matrix.columns:
        for j in corr_matrix.columns:
            if i != j and corr_matrix.loc[i, j] > threshold:
                G.add_edge(i, j, weight=corr_matrix.loc[i, j])
    componentes = list(nx.connected_components(G))
    seleccionadas = []
    for grupo in componentes:
        correlaciones = {}
        for var in grupo:
            correlaciones[var] = abs(pearsonr(df[var], df[target])[0])
        mejor = max(correlaciones, key=correlaciones.get)
        seleccionadas.append(mejor)
    todas_relacionadas = set().union(*componentes) if componentes else set()
    no_conectadas = set(corr_matrix.columns) - todas_relacionadas
    seleccionadas.extend(no_conectadas)
    return list(seleccionadas) 


def load_config(config_path: str = None, defaults: dict = None) -> dict:
    """
    Load configuration from YAML and shallow-merge with defaults.
    If YAML is not available or does not exist, return defaults (or {}).
    """
    if defaults is None:
        defaults = {}
    if (config_path is None) or (yaml is None) or (not os.path.exists(config_path)):
        return defaults
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            user_cfg = yaml.safe_load(f) or {}
        merged = defaults.copy()
        # Shallow merge dicts
        for k, v in user_cfg.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged
    except Exception:
        return defaults
# ===== Reubicadas desde TFM/utils/utils.py para unificar utils =====
def section(title: str) -> None:
    # Solo log, no prints en consola
    pass


def prepare_model_dataframe(df: pd.DataFrame, numeric_vars: List[str], target: str = "flg_target") -> pd.DataFrame:
    return df[numeric_vars + [target]].copy()


def stratified_sample_xy(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if (max_rows is None) or (len(X) <= max_rows):
        return X
    fraction = max_rows / float(len(X))
    aux = X.copy()
    aux["_y_"] = y.values
    sampled = aux.groupby("_y_", group_keys=False).apply(
        lambda g: g.sample(frac=fraction, random_state=random_state),
        include_groups=False
    ).drop(columns=["_y_"])
    return sampled


def evaluate_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_hat = (y_score >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_hat)
    precision = precision_score(y_true, y_hat, zero_division=0)
    recall = recall_score(y_true, y_hat, zero_division=0)
    f1 = f1_score(y_true, y_hat, zero_division=0)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn + 1e-12)
    intervened = float(y_hat.mean())
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "FPR": float(fpr),
        "%_intervenido": intervened,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def print_metrics_block(model_name: str, metrics: Dict[str, Any], logger=None) -> None:
    # Solo log, no prints en consola
    if logger:
        log_message(logger, f"[{model_name}] Test metrics:")
        log_message(logger,
            f"  AUC={metrics['AUC_test']:.4f} | Acc={metrics['Accuracy_test']:.4f} | "
            f"F1={metrics['F1_test']:.4f} | BalAcc={metrics['BalancedAcc_test']:.4f}"
        )
        log_message(logger,
            f"  PR_AUC(AP)={metrics['PR_AUC_test']:.4f} | KS={metrics['KS_test']:.4f} | "
            f"Brier={metrics['Brier_test']:.4f}"
        )
        cm = metrics["ConfusionMatrix"]
        log_message(logger, f"  ConfusionMatrix = [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")


def plot_comparative_curves(models_info: list, out_path: str) -> None:
    plt.figure(figsize=(12, 5))
    ax1 = plt.subplot(1, 2, 1)
    for info in models_info:
        fpr, tpr, _ = roc_curve(info['y_true'], info['y_proba'])
        roc_auc = roc_auc_score(info['y_true'], info['y_proba'])
        ax1.plot(fpr, tpr, label=f"{info['label']} (AUC={roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax1.set_title('ROC – Model comparison')
    ax1.set_xlabel('FPR (1 – Specificity)')
    ax1.set_ylabel('TPR (Sensitivity)')
    ax1.grid(True)
    ax1.legend(fontsize=8)
    ax2 = plt.subplot(1, 2, 2)
    for info in models_info:
        prec, rec, _ = precision_recall_curve(info['y_true'], info['y_proba'])
        ap = average_precision_score(info['y_true'], info['y_proba'])
        ax2.plot(rec, prec, label=f"{info['label']} (AP={ap:.3f})")
    ax2.set_title('Precision–Recall – Model comparison')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(True)
    ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    _log_info(f"[SAVE] Saved comparative curves: {out_path}")


def plot_comparative_sensitivity(curves_info: list, var_name: str, out_path: str) -> None:
    plt.figure(figsize=(8, 6))
    for info in curves_info:
        df = info['df']
        plt.plot(df[var_name], df['prob_impago_promedio'], marker='o', label=info['label'])
    plt.title(f"Sensitivity (ceteris paribus) – Avg. default prob vs {var_name}")
    plt.xlabel(var_name)
    plt.ylabel('Average predicted default probability')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
    plt.grid(True)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    _log_info(f"[SAVE] Saved comparative sensitivity: {out_path}")


def sensitivity_int_rate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str, images_dir: str, max_rows: int = 50000) -> None:
    from engine_TFM.engine_modeling import ModelingEngine
    if 'int_rate' not in X_test.columns:
        _log_info(f"[{label}] 'int_rate' not present in X_test. Skipping sensitivity.")
        return
    _log_info(f"[{label}] Sensitivity for 'int_rate'...")
    tasas = np.arange(10, 95, 5)
    safe_label = label.replace(' | ', '_').replace(' ', '_').lower()
    save_path = os.path.join(images_dir, f'sensibilidad_int_rate_{safe_label}.png')
    X_used = stratified_sample_xy(X_test, y_test, max_rows=max_rows)
    ModelingEngine.variable_sensitivity(
        model,
        X_used,
        'int_rate',
        tasas,
        plot=True,
        verbose=False,
        save_path=save_path,
        title_suffix=label,
        show_plot=False,
        max_rows=None
    )
    _log_info(f"[SAVE] Sensitivity saved: {save_path}")


class ModelComparator:
    def __init__(self, base_dir: str, test_size: float = 0.3, random_state: int = 42):
        self.base_dir = base_dir
        self.models_dir = os.path.join(base_dir, 'models')
        self.reports_dir = os.path.join(base_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        self.test_size = test_size
        self.random_state = random_state

    def _evaluate_loaded_model(self, model, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str], target: str = 'flg_target') -> Dict[str, Any]:
        X = df[num_cols + cat_cols]
        y = df[target]
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
            y_proba = 1.0 / (1.0 + np.exp(-scores))
        auc = roc_auc_score(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        pr_auc = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = float(np.max(tpr - fpr))
        brier = brier_score_loss(y_test, y_proba)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        return {
            'AUC': auc,
            'PR_AUC(AP)': pr_auc,
            'Brier': brier,
            'Accuracy': acc,
            'F1': f1,
            'BalancedAcc': bal_acc,
            'Precision': float(prec),
            'Recall': float(rec),
            'KS': ks,
            'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp),
        }

    def compare(self) -> pd.DataFrame:
        from engine_TFM.engine_modeling import ModelingEngine

        # Intentar usar dataframes con WOE si existen, sino usar originales
        csv_pca_lda_woe = os.path.join(self.base_dir, 'df_pca_lda_woe.csv')
        csv_anova_woe = os.path.join(self.base_dir, 'df_anova_woe.csv')
        csv_pca_lda = os.path.join(self.base_dir, 'df_pca_lda.csv')
        csv_anova = os.path.join(self.base_dir, 'df_anova.csv')

        # Cargar dataframes con WOE si existen
        if os.path.exists(csv_pca_lda_woe):
            df_pca_lda = pd.read_csv(csv_pca_lda_woe)
            _log_info(f"[ModelComparator] Usando dataframe PCA+LDA con WOE: {csv_pca_lda_woe}")
        else:
            df_pca_lda = pd.read_csv(csv_pca_lda)
            _log_info(f"[ModelComparator] Usando dataframe PCA+LDA original: {csv_pca_lda}")

        if os.path.exists(csv_anova_woe):
            df_anova = pd.read_csv(csv_anova_woe)
            _log_info(f"[ModelComparator] Usando dataframe ANOVA con WOE: {csv_anova_woe}")
        else:
            df_anova = pd.read_csv(csv_anova)
            _log_info(f"[ModelComparator] Usando dataframe ANOVA original: {csv_anova}")

        # Intentar cargar las variables finales desde los mapeos WOE
        num_pca_lda = self._get_final_vars_from_woe('pca', df_pca_lda)
        num_anova = self._get_final_vars_from_woe('anova', df_anova)
        cat_cols: List[str] = []
        entries = [
            ('LOGIT | PCA+LDA', 'logit_pca_lda.pkl', 'pca'),
            ('LOGIT | ANOVA', 'logit_anova.pkl', 'anova'),
            ('GNB | PCA+LDA', 'gnb_pca_lda.pkl', 'pca'),
            ('GNB | ANOVA', 'gnb_anova.pkl', 'anova'),
            ('SVM best | PCA+LDA', 'svm_best_pca_lda.pkl', 'pca'),
            ('SVM best | ANOVA', 'svm_best_anova.pkl', 'anova'),
            ('SVM RBF best | PCA+LDA', 'svm_rbf_best_pca_lda.pkl', 'pca'),
            ('SVM RBF best | ANOVA', 'svm_rbf_best_anova.pkl', 'anova'),
            ('MLP | PCA+LDA', 'mlp_pca_lda.pkl', 'pca'),
            ('MLP | ANOVA', 'mlp_anova.pkl', 'anova'),
        ]
        rows: List[Dict[str, Any]] = []
        for label, fname, which in entries:
            fpath = os.path.join(self.models_dir, fname)
            if not os.path.exists(fpath):
                # silencioso, no prints
                continue
            model = ModelingEngine.load_model(fpath)
            metrics = self._evaluate_loaded_model(
                model,
                df_pca_lda if which == 'pca' else df_anova,
                num_pca_lda if which == 'pca' else num_anova,
                []
            )
            row = {'Modelo': label}
            row.update(metrics)
            rows.append(row)
        return pd.DataFrame(rows).sort_values(by=['AUC', 'PR_AUC(AP)'], ascending=[False, False]).reset_index(drop=True)

    def _get_final_vars_from_woe(self, dataset_type: str, df: pd.DataFrame) -> List[str]:
        """
        Obtiene las variables finales desde los mapeos WOE guardados.

        Args:
            dataset_type: 'pca' o 'anova'
            df: DataFrame con las columnas disponibles

        Returns:
            Lista de variables finales a usar
        """
        woe_file = os.path.join(self.base_dir, f'woe_mappings/woe_mappings_{dataset_type}.json')

        if os.path.exists(woe_file):
            try:
                with open(woe_file, 'r', encoding='utf-8') as f:
                    woe_data = json.load(f)

                final_vars = woe_data.get('final_vars')
                if final_vars:
                    # Verificar que todas las variables estén disponibles en el dataframe
                    available_vars = [var for var in final_vars if var in df.columns]
                    if len(available_vars) == len(final_vars):
                        _log_info(f"[ModelComparator] Usando variables finales desde WOE ({dataset_type}): {len(available_vars)} variables")
                        return available_vars
                    else:
                        _log_info(f"[ModelComparator] Algunas variables finales no disponibles en {dataset_type}, usando todas menos target")
            except Exception as e:
                _log_info(f"[ModelComparator] Error leyendo mapeos WOE para {dataset_type}: {e}")

        # Fallback: usar todas las columnas menos el target
        all_cols = df.columns.tolist()
        if 'flg_target' in all_cols:
            all_cols.remove('flg_target')
        _log_info(f"[ModelComparator] Usando todas las columnas disponibles para {dataset_type}: {len(all_cols)} variables")
        return all_cols


def _log_info(message: str) -> None:
    """
    Helper para enviar info al logger global si existe; evita prints en consola.
    """
    logger = logging.getLogger('ModelingPipeline')
    if logger and logger.handlers:
        logger.info(message)
    # no prints

    def save(self, df_cmp: pd.DataFrame, filename: str = 'models_comparison.csv') -> str:
        out_csv = os.path.join(self.reports_dir, filename)
        df_cmp.to_csv(out_csv, index=False)
        return out_csv

# ===== Helpers for safe EDA operations =====

def safe_drop_columns(df: pd.DataFrame, columns: List[str], verbose: bool = True) -> pd.DataFrame:
    """
    Drop columns only if present. Returns a new DataFrame.
    """
    present = [c for c in columns if c in df.columns]
    if present:
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
        return df.drop(columns=present)
    if verbose:
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass
    return df


def build_binary_target(
    df: pd.DataFrame,
    status_column: str,
    bad_status: List[str],
    good_status: List[str],
    target_column: str = "flg_target",
    drop_missing_status: bool = True,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Construct a binary target column from a status column using provided lists.
    Optionally drop rows with status not in good/bad sets.
    """
    if status_column not in df.columns:
        raise ValueError(f"Status column '{status_column}' not found in DataFrame.")
    if verbose:
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass
    good_set = set(good_status or [])
    bad_set = set(bad_status or [])
    if drop_missing_status:
        mask = df[status_column].isin(good_set | bad_set)
        before = len(df)
        df = df.loc[mask].copy().reset_index(drop=True)
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
    df[target_column] = df[status_column].apply(lambda x: 1 if x in bad_set else (0 if x in good_set else np.nan))
    if verbose:
        pct_bad = df[target_column].mean()
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass
    return df


def term_to_int(series: pd.Series, verbose: bool = False) -> pd.Series:
    """
    Convert strings like '36 months' to integer 36. Leaves integers unchanged.
    """
    if series.dtype == int or series.dtype == float:
        return series.astype(float).round().astype(int)
    converted = series.astype(str).str.replace(' months', '', regex=False)
    converted = converted.str.extract(r"(\d+)")[0].astype(float).round().astype(float)
    if verbose:
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass
    return converted


def emp_length_to_months(series: pd.Series, verbose: bool = False) -> pd.Series:
    """
    Map '10+ years'->10, '< 1 year'->0, '1 year'->1, then multiply by 12. Returns float.
    """
    mapping = {"10+ years": "10", "< 1 year": "0", "1 year": "1"}
    s = series.astype(str).replace(mapping, regex=False)
    s = s.str.replace(r"\D", "", regex=True).replace("", np.nan)
    months = s.astype(float) * 12.0
    if verbose:
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass
    return months


def intentar_conversion_numerica(series: pd.Series, verbose: bool = False) -> tuple:
    """
    Intenta convertir una serie a numérica y devuelve el resultado.

    Args:
        series: Serie de pandas a convertir
        verbose: Si mostrar mensajes detallados

    Returns:
        tuple: (éxito_conversion, serie_convertida, tipo_original)
    """
    tipo_original = series.dtype
    valores_no_nulos = series.dropna()

    if len(valores_no_nulos) == 0:
        return False, series, tipo_original

    try:
        # Intentar conversión a numérico
        numeric_series = pd.to_numeric(valores_no_nulos, errors='coerce')

        # Verificar si la conversión fue exitosa (sin NaN introducidos)
        if not numeric_series.isna().any():
            if verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass
            return True, numeric_series, tipo_original
        else:
            if verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass
            return False, series, tipo_original

    except Exception as e:
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
        return False, series, tipo_original


def analizar_cardinalidad_categorica(series: pd.Series, max_unique: int = 20, max_concentration: float = 0.98, verbose: bool = False) -> dict:
    """
    Analiza las características de cardinalidad de una variable categórica.

    Args:
        series: Serie categórica a analizar
        max_unique: Máximo número de valores únicos permitidos
        max_concentration: Máxima concentración de un solo valor
        verbose: Si mostrar análisis detallado

    Returns:
        dict: Resultado del análisis con métricas y decisión
    """
    valores_no_nulos = series.dropna()

    if len(valores_no_nulos) == 0:
        return {
            'decision': 'descartar',
            'razon': 'columna_vacia',
            'n_unique': 0,
            'concentracion': 0.0,
            'valores_unicos': []
        }

    n_unique = valores_no_nulos.nunique()
    valor_counts = valores_no_nulos.value_counts()
    concentracion = valor_counts.iloc[0] / len(valores_no_nulos) if len(valor_counts) > 0 else 0.0

    # Tomar decisión
    if n_unique > max_unique:
        decision = 'descartar'
        razon = f'demasiados_valores_unicos_{n_unique}_>_max_{max_unique}'
    elif concentracion > max_concentration:
        decision = 'descartar'
        razon = f'concentracion_excesiva_{concentracion:.1%}_>_max_{max_concentration:.1%}'
    else:
        decision = 'mantener'
        razon = 'aprobada'

    resultado = {
        'decision': decision,
        'razon': razon,
        'n_unique': n_unique,
        'concentracion': concentracion,
        'valores_unicos': valor_counts.index.tolist()[:10]  # Primeros 10 valores
    }

    if verbose:
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        pass

    return resultado


def proteger_int_rate(df: pd.DataFrame, variables_numericas: list, variables_categoricas: list,
                     protected_var: str = 'int_rate', correlation_threshold: float = 0.7,
                     verbose: bool = True, logger=None) -> dict:
    """
    Protege la variable int_rate asegurando que no sea eliminada y manejando
    variables altamente correlacionadas con ella.

    Args:
        df: DataFrame con los datos
        variables_numericas: Lista de variables numéricas
        variables_categoricas: Lista de variables categóricas
        protected_var: Variable a proteger (int_rate)
        correlation_threshold: Umbral de correlación alta
        verbose: Si mostrar logs detallados

    Returns:
        dict: Resultado con variables finales y análisis de correlación
    """
    def log_message(msg):
        if logger:
            logger.info(msg)
        # No print a terminal, solo a log

    if verbose:
        log_message(f"\\n🛡️ PROTECCIÓN DE {protected_var.upper()}")
        log_message("=" * 50)

    # Verificar que int_rate existe
    if protected_var not in df.columns:
        if verbose:
            log_message(f"⚠️ Variable protegida '{protected_var}' no encontrada en el dataset")
        return {
            'variables_numericas_finales': variables_numericas,
            'variables_categoricas_finales': variables_categoricas,
            'variables_correlacionadas_eliminadas': [],
            'protected_variable_presente': False
        }

    if verbose:
        log_message(f"✅ Variable protegida '{protected_var}' encontrada")

    # Variables a analizar (numéricas + int_rate)
    vars_para_correlacion = [v for v in variables_numericas if v in df.columns]
    if protected_var not in vars_para_correlacion:
        vars_para_correlacion.append(protected_var)

    # Filtrar solo variables realmente numéricas para correlación
    vars_numericas_validas = []
    for var in vars_para_correlacion:
        if pd.api.types.is_numeric_dtype(df[var]):
            vars_numericas_validas.append(var)
        else:
            if verbose:
                log_message(f"⚠️ Variable {var} no es numérica para correlación: {df[var].dtype}")

    if verbose:
        log_message(f"Variables válidas para correlación: {len(vars_numericas_validas)}")

    # Calcular matriz de correlación
    try:
        corr_matrix = df[vars_numericas_validas].corr()
        corr_con_protegida = corr_matrix[protected_var].abs()

        if verbose:
            log_message(f"\\n📊 Análisis de correlación con {protected_var}:")
            # Mostrar top 10 correlaciones más altas
            top_corr = corr_con_protegida.sort_values(ascending=False).head(10)
            for var, corr in top_corr.items():
                if var != protected_var:
                    log_message(f"  {var}: {corr:.3f}")

        # Identificar variables altamente correlacionadas
        variables_altamente_correlacionadas = []
        for var in corr_con_protegida.index:
            if var != protected_var and var in variables_numericas and var in vars_numericas_validas:
                corr_abs = corr_con_protegida[var]
                if corr_abs >= correlation_threshold:
                    variables_altamente_correlacionadas.append({
                        'variable': var,
                        'correlacion': corr_abs,
                        'decision': 'eliminar_por_correlacion_alta'
                    })

        if verbose and variables_altamente_correlacionadas:
            log_message(f"\\n🚨 VARIABLES ALTAMENTE CORRELACIONADAS CON {protected_var} (>{correlation_threshold}):")
            for item in variables_altamente_correlacionadas:
                log_message(f"  ❌ {item['variable']}: {item['correlacion']:.3f}")

        # También revisar variables categóricas que podrían estar relacionadas
        variables_cat_correlacionadas = []
        for cat_var in variables_categoricas:
            if cat_var in df.columns:
                # Para variables categóricas, podemos hacer un análisis básico
                # Variables como 'grade' y 'sub_grade' están inherentemente relacionadas con int_rate
                if cat_var in ['grade', 'sub_grade'] and protected_var in df.columns:
                    variables_cat_correlacionadas.append({
                        'variable': cat_var,
                        'razon': f'relacion_inherente_con_{protected_var}'
                    })

        if verbose and variables_cat_correlacionadas:
            log_message(f"\\n🎯 VARIABLES CATEGÓRICAS RELACIONADAS CON {protected_var}:")
            for item in variables_cat_correlacionadas:
                log_message(f"  ⚠️ {item['variable']}: {item['razon']}")

        # Variables finales (mantener int_rate y eliminar correlacionadas)
        variables_numericas_finales = [v for v in variables_numericas if v not in
                                     [item['variable'] for item in variables_altamente_correlacionadas]]

        # Asegurar que int_rate esté incluida
        if protected_var not in variables_numericas_finales:
            variables_numericas_finales.append(protected_var)

        # Para categóricas, por ahora las mantenemos todas (luego el WOE/IV decidirá)
        variables_categoricas_finales = variables_categoricas.copy()

        if verbose:
            log_message(f"\\n🎯 RESULTADO PROTECCIÓN {protected_var.upper()}:")
            log_message(f"   ✅ {protected_var} PROTEGIDA Y MANTENIDA")
            log_message(f"   ❌ Variables numéricas eliminadas por correlación: {len(variables_altamente_correlacionadas)}")
            log_message(f"   📊 Variables numéricas finales: {len(variables_numericas_finales)}")
            log_message(f"   📊 Variables categóricas finales: {len(variables_categoricas_finales)}")

        return {
            'variables_numericas_finales': variables_numericas_finales,
            'variables_categoricas_finales': variables_categoricas_finales,
            'variables_correlacionadas_eliminadas': variables_altamente_correlacionadas,
            'variables_cat_relacionadas': variables_cat_correlacionadas,
            'protected_variable_presente': True,
            'correlacion_matrix': corr_matrix
        }

    except Exception as e:
        if verbose:
            log_message(f"❌ Error en análisis de correlación: {str(e)}")
        return {
            'variables_numericas_finales': variables_numericas,
            'variables_categoricas_finales': variables_categoricas,
            'variables_correlacionadas_eliminadas': [],
            'error': str(e),
            'protected_variable_presente': True
        }


def derive_features(df: pd.DataFrame, toggles: dict, verbose: bool = True) -> pd.DataFrame:
    """
    Create derived numeric features according to toggles and column availability.
    Safely handles missing columns.
    """
    df = df.copy()
    if toggles.get('enable_term_integer', True) and ('term' in df.columns):
        df['term'] = term_to_int(df['term'], verbose=verbose)
    if toggles.get('enable_emp_length_months', True) and ('emp_length' in df.columns):
        df['emp_length_months'] = emp_length_to_months(df['emp_length'], verbose=verbose)
    if toggles.get('enable_ratio_loan_income', True) and all(c in df.columns for c in ['loan_amnt', 'annual_inc']):
        df['ratio_loan_income'] = df['loan_amnt'] / (df['annual_inc'].astype(float) + 1e-5)
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
    if toggles.get('enable_installment_pct_loan', True) and all(c in df.columns for c in ['installment', 'loan_amnt']):
        df['installment_pct_loan'] = df['installment'] / (df['loan_amnt'].astype(float) + 1e-5)
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
    if toggles.get('enable_fico_avg', True) and all(c in df.columns for c in ['fico_range_low', 'fico_range_high']):
        df['fico_avg'] = (df['fico_range_low'].astype(float) + df['fico_range_high'].astype(float)) / 2.0
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
    if toggles.get('enable_revol_util_ratio', True) and all(c in df.columns for c in ['revol_bal', 'total_rev_hi_lim']):
        df['revol_util_ratio'] = df['revol_bal'].astype(float) / (df['total_rev_hi_lim'].astype(float) + 1e-5)
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
    return df


class WOETransformer:
    """
    Transformer para Weight of Evidence (WOE) encoding de variables categóricas.

    Maneja automáticamente los valores faltantes como una categoría separada y
    guarda los mapeos para replicación posterior en producción.
    """

    def __init__(self, target_col: str = 'flg_target', min_samples: int = 30):
        self.target_col = target_col
        self.min_samples = min_samples
        self.woe_mappings: Dict[str, Dict[str, float]] = {}
        self.category_stats: Dict[str, Dict[str, Any]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str]) -> 'WOETransformer':
        """
        Calcula los mapeos WOE para las variables categóricas especificadas.

        Args:
            df: DataFrame de entrenamiento
            cat_cols: Lista de columnas categóricas a procesar

        Returns:
            self: Transformer entrenado
        """
        for col in cat_cols:
            if col not in df.columns:
                continue

            # Crear tabla de contingencia
            contingency_table = pd.crosstab(df[col].fillna('MISSING'), df[self.target_col])
            total_good = contingency_table[0].sum() if 0 in contingency_table.columns else 0
            total_bad = contingency_table[1].sum() if 1 in contingency_table.columns else 0
            total_samples = len(df)

            woe_dict = {}
            stats_dict = {}

            for category in contingency_table.index:
                good_count = contingency_table.loc[category, 0] if 0 in contingency_table.columns else 0
                bad_count = contingency_table.loc[category, 1] if 1 in contingency_table.columns else 0

                # Calcular tasas con suavizado para evitar división por cero
                good_rate = (good_count + 0.5) / (total_good + 1)
                bad_rate = (bad_count + 0.5) / (total_bad + 1)

                # Calcular WOE
                woe = np.log(good_rate / bad_rate)
                woe_dict[str(category)] = float(woe)

                # Estadísticas para análisis
                stats_dict[str(category)] = {
                    'good_count': int(good_count),
                    'bad_count': int(bad_count),
                    'total_count': int(good_count + bad_count),
                    'good_rate': float(good_rate),
                    'bad_rate': float(bad_rate),
                    'woe': float(woe)
                }

            self.woe_mappings[col] = woe_dict
            self.category_stats[col] = stats_dict

        return self

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """
        Aplica la transformación WOE a las variables categóricas.

        Args:
            df: DataFrame a transformar
            cat_cols: Lista de columnas categóricas a transformar

        Returns:
            DataFrame con variables transformadas
        """
        df_transformed = df.copy()

        for col in cat_cols:
            if col not in df.columns or col not in self.woe_mappings:
                continue

            # Crear nueva columna con sufijo _woe
            new_col = f"{col}_woe"
            df_transformed[new_col] = df[col].fillna('MISSING').astype(str).map(self.woe_mappings[col])

            # Manejar categorías no vistas (asignar WOE promedio)
            nan_mask = df_transformed[new_col].isna()
            if nan_mask.any():
                mean_woe = np.mean(list(self.woe_mappings[col].values()))
                df_transformed.loc[nan_mask, new_col] = mean_woe

        return df_transformed

    def fit_transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """
        Ajusta el transformer y transforma los datos en un solo paso.

        Args:
            df: DataFrame a procesar
            cat_cols: Lista de columnas categóricas

        Returns:
            DataFrame transformado
        """
        return self.fit(df, cat_cols).transform(df, cat_cols)

    def save_mappings(self, filepath: str) -> None:
        """
        Guarda los mapeos WOE en un archivo JSON para replicación posterior.

        Args:
            filepath: Ruta donde guardar el archivo
        """
        save_data = {
            'woe_mappings': self.woe_mappings,
            'category_stats': self.category_stats,
            'target_col': self.target_col,
            'min_samples': self.min_samples,
            'final_vars': getattr(self, 'final_vars', None),
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        _log_info(f"[WOE] Mapeos guardados en: {filepath}")

    def load_mappings(self, filepath: str) -> 'WOETransformer':
        """
        Carga los mapeos WOE desde un archivo JSON.

        Args:
            filepath: Ruta del archivo a cargar

        Returns:
            self: Transformer con mapeos cargados
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            save_data = json.load(f)

        self.woe_mappings = save_data['woe_mappings']
        self.category_stats = save_data.get('category_stats', {})
        self.target_col = save_data['target_col']
        self.min_samples = save_data['min_samples']
        if 'final_vars' in save_data:
            self.final_vars = save_data['final_vars']

        _log_info(f"[WOE] Mapeos cargados desde: {filepath}")
        return self

    def get_feature_names_out(self, cat_cols: List[str]) -> List[str]:
        """
        Devuelve los nombres de las nuevas columnas WOE.

        Args:
            cat_cols: Lista de columnas categóricas originales

        Returns:
            Lista de nombres de columnas WOE
        """
        return [f"{col}_woe" for col in cat_cols]

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Devuelve un resumen estadístico de las transformaciones WOE realizadas.

        Returns:
            DataFrame con estadísticas resumidas
        """
        summary_data = []

        for col, stats in self.category_stats.items():
            for category, cat_stats in stats.items():
                summary_data.append({
                    'variable': col,
                    'categoria': category,
                    'good_count': cat_stats['good_count'],
                    'bad_count': cat_stats['bad_count'],
                    'total_count': cat_stats['total_count'],
                    'woe': cat_stats['woe']
                })

        return pd.DataFrame(summary_data)