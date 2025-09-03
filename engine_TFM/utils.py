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
from typing import Dict, Any, List
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
        serie_valida = df[[col, target]].dropna()
        if len(serie_valida) >= min_samples:
            correlaciones[col] = serie_valida.corr().iloc[0,1]
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
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def prepare_model_dataframe(df: pd.DataFrame, numeric_vars: List[str], target: str = "flg_target") -> pd.DataFrame:
    return df[numeric_vars + [target]].copy()


def stratified_sample_xy(X: pd.DataFrame, y: pd.Series, max_rows: int, random_state: int = 42) -> pd.DataFrame:
    if (max_rows is None) or (len(X) <= max_rows):
        return X
    fraction = max_rows / float(len(X))
    aux = X.copy()
    aux["_y_"] = y.values
    sampled = aux.groupby("_y_", group_keys=False).apply(
        lambda g: g.sample(frac=fraction, random_state=random_state)
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


def print_metrics_block(model_name: str, metrics: Dict[str, Any]) -> None:
    print(f"[{model_name}] Test metrics:")
    print(
        f"  AUC={metrics['AUC_test']:.4f} | Acc={metrics['Accuracy_test']:.4f} | "
        f"F1={metrics['F1_test']:.4f} | BalAcc={metrics['BalancedAcc_test']:.4f}"
    )
    print(
        f"  PR_AUC(AP)={metrics['PR_AUC_test']:.4f} | KS={metrics['KS_test']:.4f} | "
        f"Brier={metrics['Brier_test']:.4f}"
    )
    cm = metrics["ConfusionMatrix"]
    print(f"  ConfusionMatrix = [[{cm[0,0]}, {cm[0,1]}], [{cm[1,0]}, {cm[1,1]}]]")


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
    print(f"[SAVE] Saved comparative curves: {out_path}")


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
    print(f"[SAVE] Saved comparative sensitivity: {out_path}")


def sensitivity_int_rate(model, X_test: pd.DataFrame, y_test: pd.Series, label: str, images_dir: str, max_rows: int = 50000) -> None:
    from engine_TFM.engine_modeling import ModelingEngine
    if 'int_rate' not in X_test.columns:
        print(f"[{label}] 'int_rate' not present in X_test. Skipping sensitivity.")
        return
    print(f"[{label}] Sensitivity for 'int_rate'...")
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
    print(f"[SAVE] Sensitivity saved: {save_path}")


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
        csv_pca_lda = os.path.join(self.base_dir, 'df_pca_lda.csv')
        csv_anova = os.path.join(self.base_dir, 'df_anova.csv')
        df_pca_lda = pd.read_csv(csv_pca_lda)
        df_anova = pd.read_csv(csv_anova)
        num_pca_lda = df_pca_lda.columns.tolist()[:-1]
        num_anova = df_anova.columns.tolist()[:-1]
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
                print(f"[SKIP] No existe: {fpath}")
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
            print(f"[DROP] Dropping {len(present)} columns: {present}")
        return df.drop(columns=present)
    if verbose:
        print("[DROP] No matching columns to drop.")
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
        print(f"[TARGET] Building target '{target_column}' from '{status_column}'...")
    good_set = set(good_status or [])
    bad_set = set(bad_status or [])
    if drop_missing_status:
        mask = df[status_column].isin(good_set | bad_set)
        before = len(df)
        df = df.loc[mask].copy().reset_index(drop=True)
        if verbose:
            print(f"[TARGET] Filtered by allowed statuses: {before} -> {len(df)} rows")
    df[target_column] = df[status_column].apply(lambda x: 1 if x in bad_set else (0 if x in good_set else np.nan))
    if verbose:
        pct_bad = df[target_column].mean()
        print(f"[TARGET] '{target_column}' created. Bad rate={pct_bad:.3f}")
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
        print("[FEATURE] Converted 'term' to integers (nullable).")
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
        print("[FEATURE] Derived 'emp_length_months'.")
    return months


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
            print("[FEATURE] Created 'ratio_loan_income'.")
    if toggles.get('enable_installment_pct_loan', True) and all(c in df.columns for c in ['installment', 'loan_amnt']):
        df['installment_pct_loan'] = df['installment'] / (df['loan_amnt'].astype(float) + 1e-5)
        if verbose:
            print("[FEATURE] Created 'installment_pct_loan'.")
    if toggles.get('enable_fico_avg', True) and all(c in df.columns for c in ['fico_range_low', 'fico_range_high']):
        df['fico_avg'] = (df['fico_range_low'].astype(float) + df['fico_range_high'].astype(float)) / 2.0
        if verbose:
            print("[FEATURE] Created 'fico_avg'.")
    if toggles.get('enable_revol_util_ratio', True) and all(c in df.columns for c in ['revol_bal', 'total_rev_hi_lim']):
        df['revol_util_ratio'] = df['revol_bal'].astype(float) / (df['total_rev_hi_lim'].astype(float) + 1e-5)
        if verbose:
            print("[FEATURE] Created 'revol_util_ratio'.")
    return df
