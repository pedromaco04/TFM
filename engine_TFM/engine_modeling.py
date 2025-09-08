# engine_modeling.py
# Modelado y evaluación de modelos para el engine TFM

import joblib
import os
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, balanced_accuracy_score, precision_recall_curve, brier_score_loss, average_precision_score
from engine_TFM.utils import calculate_gini
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
import warnings
from sklearn.exceptions import ConvergenceWarning


class _SigmoidProbaWrapper:
    """
    Envuelve un estimador con decision_function para exponer predict_proba≈sigmoid(decision_function).
    """
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator.predict(X)

    def decision_function(self, X):
        return self.estimator.decision_function(X)

    def predict_proba(self, X):
        scores = self.decision_function(X)
        probs_pos = 1.0 / (1.0 + np.exp(-scores))
        probs_neg = 1.0 - probs_pos
        return np.column_stack([probs_neg, probs_pos])

class ModelingEngine:
    @staticmethod
    def fit_logit_predict(
        df,
        num_cols,
        cat_cols,
        target='flg_target',
        test_size=0.3,
        random_state=42,
        model_params=None,
        config=None,
        show_confusion=True,
        verbose=False,
        save_confusion_path=None,
        title_suffix=None
    ):
        """
        Entrena y evalúa un modelo de regresión logística con pipeline de preprocesamiento mejorado.
        Devuelve: pipeline entrenado, métricas, predicciones, probabilidades, X_train, X_test, y_train, y_test
        """
        if model_params is None:
            model_params = {}
        
        # Preprocesamiento mejorado
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])
        
        # Usar parámetros del config si están disponibles
        if config and 'logit' in config:
            logit_cfg = config['logit']
            max_iter = logit_cfg.get('max_iter', 1000)
            random_state = logit_cfg.get('random_state', random_state)
            if model_params is None:
                model_params = {}
            model_params.update({
                'max_iter': max_iter,
                'random_state': random_state
            })
        else:
            if model_params is None:
                model_params = {}
            model_params.setdefault('max_iter', 1000)
            
        model = LogisticRegression(**model_params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        
        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        if verbose:
            print("[fit_logit_predict] Iniciando entrenamiento Logit...")
            print(f"[fit_logit_predict] Tamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")
            print(f"[fit_logit_predict] Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:,1]
        
        # Métricas extendidas
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # KS statistic
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        
        # Matriz de confusión (mostrar y/o guardar)
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – Regresión logística"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_logit_predict] Matriz de confusión guardada en: {save_confusion_path}")
        if show_confusion:
            plt.show()
        else:
            plt.close()
        
        if verbose:
            print("[fit_logit_predict] Métricas en test:")
            print(f"  AUC={auc:.4f} | GINI={gini:.4f} | Acc={acc:.4f} | F1={f1:.4f} | BalAcc={bal_acc:.4f}")
            print(f"  PR_AUC(AP)={pr_auc:.4f} | KS={ks:.4f} | Brier={brier:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return pipeline, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def threshold_for_target_recall(y_true, y_score, target_recall=0.9, verbose=False):
        thresholds = np.unique(np.sort(y_score))[::-1]
        best_thr = thresholds[-1]
        from sklearn.metrics import recall_score
        for thr in thresholds:
            y_pred = (y_score >= thr).astype(int)
            rec = recall_score(y_true, y_pred)
            if rec >= target_recall:
                if verbose:
                    print(f"[threshold_for_target_recall] Umbral={thr:.4f} alcanza Recall={rec:.4f} (objetivo {target_recall:.2f})")
                return thr
        if verbose:
            print(f"[threshold_for_target_recall] No se alcanzó Recall objetivo. Devolviendo umbral mínimo {best_thr:.4f}")
        return best_thr

    @staticmethod
    def threshold_max_fbeta(y_true, y_score, beta=2.0, verbose=False):
        prec, rec, thr = precision_recall_curve(y_true, y_score)
        fbeta = (1+beta**2) * (prec*rec) / (beta**2*prec + rec + 1e-12)
        fbeta = fbeta[1:]
        idx = np.nanargmax(fbeta)
        best_thr = thr[idx]
        if verbose:
            print(f"[threshold_max_fbeta] Umbral*={best_thr:.4f} con F{beta:.0f}_max={np.nanmax(fbeta):.4f}")
        return best_thr

    @staticmethod
    def threshold_for_topk(y_score, topk_ratio=0.10, verbose=False):
        cutoff_idx = int(np.ceil((1-topk_ratio) * len(y_score))) - 1
        thr = np.partition(y_score, cutoff_idx)[cutoff_idx]
        if verbose:
            print(f"[threshold_for_topk] Umbral para Top-{int(topk_ratio*100)}% = {thr:.4f}")
        return thr

    @staticmethod
    def threshold_min_cost(y_true, y_score, cost_fn=10.0, cost_fp=1.0, n=200, verbose=False):
        thrs = np.linspace(0, 1, n)
        best_thr, best_cost = 0.5, float('inf')
        for t in thrs:
            y_pred = (y_score >= t).astype(int)
            fn = np.sum((y_true==1) & (y_pred==0))
            fp = np.sum((y_true==0) & (y_pred==1))
            cost = cost_fn*fn + cost_fp*fp
            if cost < best_cost:
                best_cost, best_thr = cost, t
        if verbose:
            print(f"[threshold_min_cost] Mejor umbral={best_thr:.4f} con costo={best_cost:.2f} (FN*{cost_fn} + FP*{cost_fp})")
        return best_thr

    @staticmethod
    def fit_gaussian_nb_predict(
        df,
        num_cols,
        cat_cols,
        target='flg_target',
        test_size=0.3,
        random_state=42,
        model_params=None,
        config=None,
            show_confusion=True,
            standardize_numeric=True,
            verbose=False,
            save_confusion_path=None,
            title_suffix=None
        ):
        """
        Entrena y evalúa un Gaussian Naive Bayes con pipeline de preprocesamiento.
        - Numéricas: imputación (mediana) + (opcional) StandardScaler.
        - Categóricas: OneHotEncoder(drop='first', handle_unknown='ignore').
        Devuelve: (pipeline, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """

        if model_params is None:
            model_params = {}

        # Numeric pipeline
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)

        # Categorical pipeline
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        # Usar parámetros del config si están disponibles
        if config and 'gnb' in config:
            gnb_cfg = config['gnb']
            var_smoothing = float(gnb_cfg.get('var_smoothing', 1e-9))
            if model_params is None:
                model_params = {}
            model_params['var_smoothing'] = var_smoothing
            
        clf = GaussianNB(**model_params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if verbose:
            print("[fit_gaussian_nb_predict] Iniciando entrenamiento GaussianNB...")
            print(f"[fit_gaussian_nb_predict] Tamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")
            print(f"[fit_gaussian_nb_predict] Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Métricas extendidas
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        
        # KS statistic
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Matriz de confusión (mostrar y/o guardar)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – Gaussian Naive Bayes"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_gaussian_nb_predict] Matriz de confusión guardada en: {save_confusion_path}")
        if show_confusion:
            plt.show()
        else:
            plt.close()

        if verbose:
            print("[fit_gaussian_nb_predict] Métricas en test:")
            print(f"  AUC={auc:.4f} | GINI={gini:.4f} | Acc={acc:.4f} | F1={f1:.4f} | BalAcc={bal_acc:.4f}")
            print(f"  PR_AUC(AP)={pr_auc:.4f} | KS={ks:.4f} | Brier={brier:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return pipeline, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_gaussian_nb_smartsearch(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            standardize_numeric=True,
            search_grid=None,
            verbose=False
        ):
        """
        Smart search SIN cross-validation para GaussianNB:
        - Mantiene el mismo preprocesamiento que Logit (imputer + [scaler] + OHE).
        - Busca el mejor var_smoothing en un grid logarítmico sobre un único split train/test.
        Devuelve: (best_pipeline, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        import numpy as np
        from sklearn.naive_bayes import GaussianNB
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
            precision_recall_curve, brier_score_loss, confusion_matrix, roc_curve
        )

        # Grid por defecto si no se pasa uno
        if search_grid is None:
            search_grid = np.logspace(-12, -6, 20)

        # Pipelines de preprocesamiento (mismo estilo que Logit)
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)

        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        best_auc = -np.inf
        best_vs = None
        best_pipeline = None
        best_preds = None
        best_proba = None

        # Barrido simple sin CV
        if verbose:
            print(f"[fit_gaussian_nb_smartsearch] Barrido de var_smoothing con {len(search_grid)} valores...")
        for vs in search_grid:
            pipe = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', GaussianNB(var_smoothing=vs))
            ])
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            if auc > best_auc:
                best_auc = auc
                best_vs = vs
                best_pipeline = pipe
                best_proba = proba
                best_preds = (proba >= 0.5).astype(int)  # umbral base; luego puedes optimizar

        if verbose and best_vs is not None:
            print(f"[fit_gaussian_nb_smartsearch] Mejor var_smoothing={best_vs:g} con AUC={best_auc:.4f}")

        # Métricas extendidas (igual que Logit)
        y_pred = best_preds
        y_proba = best_proba
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        best_params = {'classifier__var_smoothing': best_vs}

        return best_pipeline, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test


    @staticmethod
    def save_model(model, filename):
        joblib.dump(model, filename)

    @staticmethod
    def load_model(filename):
        return joblib.load(filename)

    @staticmethod
    def variable_sensitivity(
            model,
            X_test,
            var_name,
            values,
            proba_col_name='prob_impago_promedio',
            plot=True,
            verbose=False,
            save_path=None,
            title_suffix=None,
            show_plot=False,
            max_rows=None,
            random_state=42
        ):
        """
        Evalúa la sensibilidad de la predicción respecto a una variable numérica.
        Devuelve un DataFrame con los valores simulados y la probabilidad promedio de impago.
        """
        if verbose:
            print(f"[variable_sensitivity] Evaluando sensibilidad para '{var_name}' en {len(values)} valores...")
        # Muestreo para acelerar en test grandes
        X_base = X_test
        if (max_rows is not None) and (len(X_base) > max_rows):
            X_base = X_base.sample(n=max_rows, random_state=random_state)

        resultados = []
        for v in values:
            X_temp = X_base.copy()
            X_temp[var_name] = v
            y_proba_temp = model.predict_proba(X_temp)[:,1]
            resultados.append({var_name: v, proba_col_name: y_proba_temp.mean()})
        df_sens = pd.DataFrame(resultados)
        if verbose:
            print(f"[variable_sensitivity] Listo. Prob promedio: min={df_sens[proba_col_name].min():.4f}, max={df_sens[proba_col_name].max():.4f}")
        if plot:
            plt.figure(figsize=(8,6))
            plt.plot(df_sens[var_name], df_sens[proba_col_name], marker='o')
            title = f"Sensibilidad (ceteris paribus): probabilidad media de impago vs {var_name}"
            if title_suffix:
                title += f" – {title_suffix}"
            plt.title(title)
            plt.xlabel(f"{var_name}")
            plt.ylabel("Probabilidad media de impago (predicha)")
            plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
            plt.grid(True)
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, bbox_inches='tight')
                if verbose:
                    print(f"[variable_sensitivity] Figura guardada en: {save_path}")
            if show_plot:
                plt.show()
            else:
                plt.close()
        return df_sens 

    @staticmethod
    def fit_linear_svm_predict(
        df,
        num_cols,
        cat_cols,
        target='flg_target',
        test_size=0.3,
        random_state=42,
        model_params=None,
        config=None,
            show_confusion=True,
            standardize_numeric=True,
            verbose=False,
            save_confusion_path=None,
            title_suffix=None
        ):
        """
        Entrena y evalúa un SVM lineal con calibración (probabilidades) usando CalibratedClassifierCV.
        - Preprocesamiento: imputación (mediana) + (opcional) StandardScaler para numéricas, OHE para categóricas.
        - Base estimator: LinearSVC (rápido en alta dimensionalidad), calibrado con Platt (sigmoid) y cv=3.
        Devuelve: (modelo_calibrado, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """

        if model_params is None:
            model_params = {}

        # Numeric pipeline
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)

        # Categorical pipeline
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        base_svm = LinearSVC(random_state=random_state, **model_params)
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', base_svm)
        ])
        model = _SigmoidProbaWrapper(pipeline)

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if verbose:
            print("[fit_linear_svm_predict] Iniciando entrenamiento Linear SVM calibrado...")
            print(f"[fit_linear_svm_predict] Tamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")
            print(f"[fit_linear_svm_predict] Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Métricas extendidas
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Matriz de confusión (mostrar y/o guardar)
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – Linear SVM"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_linear_svm_predict] Matriz de confusión guardada en: {save_confusion_path}")
        if show_confusion:
            plt.show()
        else:
            plt.close()

        if verbose:
            print("[fit_linear_svm_predict] Métricas en test:")
            print(f"  AUC={auc:.4f} | GINI={gini:.4f} | Acc={acc:.4f} | F1={f1:.4f} | BalAcc={bal_acc:.4f}")
            print(f"  PR_AUC(AP)={pr_auc:.4f} | KS={ks:.4f} | Brier={brier:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return model, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_linear_svm_smartsearch(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            standardize_numeric=True,
            search_space=None,
            verbose=False,
            save_confusion_path=None,
            config=None,
            title_suffix=None,
            max_train_samples=None,
            max_test_samples=None
        ):
        """
        Smart search SIN CV (solo split train/test) para Linear SVM.
        - Recorre un espacio de hiperparámetros y elige el mejor por AUC en test usando y_proba≈sigmoid(decision_function).
        - Para acelerar en datasets grandes, muestrea hasta max_train_samples/max_test_samples.
        - Devuelve: (best_model, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        # Default search space
        if search_space is None:
            search_space = {
                'C': np.logspace(-2, 2, 5),  # 0.01..100
                'loss': ['squared_hinge'],   # más estable que 'hinge'
                'class_weight': [None, 'balanced'],
            }

        # Numeric pipeline
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)

        # Categorical pipeline
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')

        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Muestreo para acelerar (solo si se especifica un límite)
        if max_train_samples is not None and len(X_train) > max_train_samples:
            rs = np.random.RandomState(random_state)
            idx = rs.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
        if max_test_samples is not None and len(X_test) > max_test_samples:
            rs = np.random.RandomState(random_state + 1)
            idx = rs.choice(len(X_test), size=max_test_samples, replace=False)
            X_test = X_test.iloc[idx]
            y_test = y_test.iloc[idx]

        best_auc = -np.inf
        best_cfg = None
        best_model = None
        best_pred = None
        best_proba = None

        # Silenciar warnings de convergencia para acelerar logs
        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        dual_flag = X_train.shape[0] < X_train.shape[1]

        # Iterate grid
        for C in search_space['C']:
            for loss in search_space['loss']:
                for cw in search_space['class_weight']:
                    params = {
                        'C': C,
                        'loss': loss,
                        'class_weight': cw,
                        'random_state': random_state,
                        'max_iter': 3000,
                        'tol': 1e-3,
                        'dual': dual_flag if loss == 'squared_hinge' else True
                    }
                    if verbose:
                        print(f"[fit_linear_svm_smartsearch] Probando params: {params}")
                    base = LinearSVC(**params)
                    pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', base)
                    ])
                    model = _SigmoidProbaWrapper(pipe)
                    model.fit(X_train, y_train)
                    proba = model.predict_proba(X_test)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    auc_val = roc_auc_score(y_test, proba)
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_cfg = params
                        best_model = model
                        best_proba = proba
                        best_pred = (proba >= 0.5).astype(int)

        # Metrics
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
            precision_recall_curve, average_precision_score, roc_curve
        )
        y_pred = best_pred
        y_proba = best_proba
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Confusion matrix save/show
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            y_pred,
            display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – Linear SVM (smart)"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_linear_svm_smartsearch] Matriz de confusión guardada en: {save_confusion_path}")
        plt.close()

        if verbose:
            print(f"[fit_linear_svm_smartsearch] Mejor params: {best_cfg} con AUC={best_auc:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }

        return best_model, best_cfg, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_mlp_predict(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            model_params=None,
            show_confusion=True,
            standardize_numeric=True,
            verbose=False,
            config=None,
            save_confusion_path=None,
            title_suffix=None
        ):
        """
        Entrena y evalúa una red neuronal MLP (clasificación binaria) con salida probabilística.
        - Preprocesamiento: imputación (mediana) + (opcional) StandardScaler + OHE.
        - MLPClassifier con early_stopping para acelerar.
        Devuelve: (pipeline, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        if model_params is None:
            model_params = {}

        # Pipelines
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        # Usar parámetros del config si están disponibles
        if config and 'mlp' in config:
            mlp_cfg = config['mlp']
            default_params = dict(
                hidden_layer_sizes=tuple(mlp_cfg.get('hidden_layer_sizes', [64, 32])),
                activation=mlp_cfg.get('activation', 'relu'),
                solver=mlp_cfg.get('solver', 'adam'),
                alpha=mlp_cfg.get('alpha', 1e-4),
                batch_size=mlp_cfg.get('batch_size', 'auto'),
                learning_rate=mlp_cfg.get('learning_rate', 'adaptive'),
                max_iter=mlp_cfg.get('max_iter', 50),
                early_stopping=mlp_cfg.get('early_stopping', True),
                n_iter_no_change=mlp_cfg.get('n_iter_no_change', 5),
                random_state=mlp_cfg.get('random_state', random_state),
                verbose=False
            )
        else:
            # Parámetros por defecto razonables para dataset grande
            default_params = dict(
                hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
                alpha=1e-4, batch_size='auto', learning_rate='adaptive',
                max_iter=50, early_stopping=True, n_iter_no_change=5,
                random_state=random_state, verbose=False
            )
        
        if model_params is None:
            model_params = {}
        default_params.update(model_params)
        clf = MLPClassifier(**default_params)

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if verbose:
            print("[fit_mlp_predict] Entrenando MLP...")
            print(f"[fit_mlp_predict] Tamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Métricas
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        # Matriz de confusión
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – MLP"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_mlp_predict] Matriz de confusión guardada en: {save_confusion_path}")
        if show_confusion:
            plt.show()
        else:
            plt.close()

        if verbose:
            print("[fit_mlp_predict] Métricas en test:")
            print(f"  AUC={auc:.4f} | GINI={gini:.4f} | Acc={acc:.4f} | F1={f1:.4f} | BalAcc={bal_acc:.4f}")
            print(f"  PR_AUC(AP)={pr_auc:.4f} | KS={ks:.4f} | Brier={brier:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return pipeline, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_mlp_smartsearch(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            standardize_numeric=True,
            search_space=None,
            verbose=False,
            save_confusion_path=None,
            title_suffix=None,
            max_train_samples=None,
            max_test_samples=None
        ):
        """
        Smart search SIN CV para MLPClassifier con early_stopping, sobre un único split train/test.
        - Muestrea train/test para acelerar en datasets grandes.
        - Selecciona por AUC en test (usando predict_proba[:,1]).
        Devuelve: (best_pipeline, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        if search_space is None:
            search_space = {
                'hidden_layer_sizes': [(64,), (64, 32), (128, 64)],
                'alpha': [1e-5, 1e-4, 1e-3],
                'learning_rate_init': [1e-3, 3e-3],
            }

        # Preprocesamiento
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Muestreo (solo si se especifica un límite)
        if max_train_samples is not None and len(X_train) > max_train_samples:
            rs = np.random.RandomState(random_state)
            idx = rs.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
        if max_test_samples is not None and len(X_test) > max_test_samples:
            rs = np.random.RandomState(random_state + 1)
            idx = rs.choice(len(X_test), size=max_test_samples, replace=False)
            X_test = X_test.iloc[idx]
            y_test = y_test.iloc[idx]

        best_auc = -np.inf
        best_params = None
        best_pipeline = None
        best_y_pred = None
        best_y_proba = None

        for hls in search_space['hidden_layer_sizes']:
            for alpha in search_space['alpha']:
                for lr in search_space['learning_rate_init']:
                    params = dict(
                        hidden_layer_sizes=hls,
                        activation='relu',
                        solver='adam',
                        alpha=alpha,
                        batch_size='auto',
                        learning_rate='adaptive',
                        learning_rate_init=lr,
                        max_iter=50,
                        early_stopping=True,
                        n_iter_no_change=5,
                        random_state=random_state,
                        verbose=False,
                    )
                    if verbose:
                        print(f"[fit_mlp_smartsearch] Probando params: {params}")
                    clf = MLPClassifier(**params)
                    pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', clf)
                    ])
                    pipe.fit(X_train, y_train)
                    proba = pipe.predict_proba(X_test)[:, 1]
                    auc_val = roc_auc_score(y_test, proba)
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_params = params
                        best_pipeline = pipe
                        best_y_proba = proba
                        best_y_pred = (proba >= 0.5).astype(int)

        # Métricas del mejor
        y_pred = best_y_pred
        y_proba = best_y_proba
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – MLP (smart)"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_mlp_smartsearch] Matriz de confusión guardada en: {save_confusion_path}")
        plt.close()

        if verbose:
            print(f"[fit_mlp_smartsearch] Mejor params: {best_params} con AUC={best_auc:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return best_pipeline, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_rbf_svm_predict(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            model_params=None,
            show_confusion=True,
            standardize_numeric=True,
            verbose=False,
            save_confusion_path=None,
            title_suffix=None
        ):
        """
        Entrena y evalúa un SVM no lineal (RBF) sin CV. Probabilidades aproximadas via sigmoide(decision_function).
        Devuelve: (modelo_envuelto, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        if model_params is None:
            model_params = {}

        # Pipelines
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        clf = SVC(kernel='rbf', **model_params)
        pipe = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])
        model = _SigmoidProbaWrapper(pipe)

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        if verbose:
            print("[fit_rbf_svm_predict] Entrenando SVM RBF...")
            print(f"[fit_rbf_svm_predict] Tamaños -> X_train: {X_train.shape}, X_test: {X_test.shape}")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – SVM RBF"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_rbf_svm_predict] Matriz de confusión guardada en: {save_confusion_path}")
        if show_confusion:
            plt.show()
        else:
            plt.close()

        if verbose:
            print("[fit_rbf_svm_predict] Métricas en test:")
            print(f"  AUC={auc:.4f} | GINI={gini:.4f} | Acc={acc:.4f} | F1={f1:.4f} | BalAcc={bal_acc:.4f}")
            print(f"  PR_AUC(AP)={pr_auc:.4f} | KS={ks:.4f} | Brier={brier:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }
        return model, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test

    @staticmethod
    def fit_rbf_svm_smartsearch(
            df,
            num_cols,
            cat_cols,
            target='flg_target',
            test_size=0.3,
            random_state=42,
            standardize_numeric=True,
            search_space=None,
            verbose=False,
            save_confusion_path=None,
            title_suffix=None,
            max_train_samples=None,
            max_test_samples=None
        ):
        """
        Smart search SIN CV para SVM RBF (usa decision_function + sigmoide). Muestrea para acelerar.
        Devuelve: (best_model, best_params, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test)
        """
        if search_space is None:
            search_space = {
                'C': [0.1, 1.0, 10.0],
                'gamma': ['scale', 'auto'],
                'class_weight': [None, 'balanced']
            }

        # Pipelines
        num_steps = [('imputer', SimpleImputer(strategy='median'))]
        if standardize_numeric:
            num_steps.append(('scaler', StandardScaler()))
        num_pipeline = Pipeline(num_steps)
        cat_pipeline = OneHotEncoder(drop='first', handle_unknown='ignore')
        preprocessor = ColumnTransformer([
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ])

        X = df[num_cols + cat_cols]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Muestreo (solo si se especifica un límite)
        if max_train_samples is not None and len(X_train) > max_train_samples:
            rs = np.random.RandomState(random_state)
            idx = rs.choice(len(X_train), size=max_train_samples, replace=False)
            X_train = X_train.iloc[idx]
            y_train = y_train.iloc[idx]
        if max_test_samples is not None and len(X_test) > max_test_samples:
            rs = np.random.RandomState(random_state + 1)
            idx = rs.choice(len(X_test), size=max_test_samples, replace=False)
            X_test = X_test.iloc[idx]
            y_test = y_test.iloc[idx]

        best_auc = -np.inf
        best_cfg = None
        best_model = None
        best_pred = None
        best_proba = None

        warnings.filterwarnings("ignore", category=ConvergenceWarning)

        for C in search_space['C']:
            for gamma in search_space['gamma']:
                for cw in search_space['class_weight']:
                    params = {
                        'C': C,
                        'gamma': gamma,
                        'class_weight': cw,
                        'kernel': 'rbf',
                        'shrinking': True,
                        'tol': 1e-3,
                        'max_iter': 2000,
                        'cache_size': 500
                    }
                    if verbose:
                        print(f"[fit_rbf_svm_smartsearch] Probando params: {params}")
                    clf = SVC(**params)
                    pipe = Pipeline([
                        ('preprocessor', preprocessor),
                        ('classifier', clf)
                    ])
                    model = _SigmoidProbaWrapper(pipe)
                    model.fit(X_train, y_train)
                    proba = model.predict_proba(X_test)[:, 1]
                    from sklearn.metrics import roc_auc_score
                    auc_val = roc_auc_score(y_test, proba)
                    if auc_val > best_auc:
                        best_auc = auc_val
                        best_cfg = params
                        best_model = model
                        best_proba = proba
                        best_pred = (proba >= 0.5).astype(int)

        # Metrics
        from sklearn.metrics import (
            roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score,
            precision_recall_curve, average_precision_score, roc_curve
        )
        y_pred = best_pred
        y_proba = best_proba
        auc = roc_auc_score(y_test, y_proba)
        gini = calculate_gini(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ks = np.max(tpr - fpr)
        brier = brier_score_loss(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, display_labels=["No impago (0)", "Impago (1)"]
        )
        title = "Matriz de confusión (test) – SVM RBF (smart)"
        if title_suffix:
            title += f" – {title_suffix}"
        plt.title(title)
        plt.xlabel("Etiqueta predicha")
        plt.ylabel("Etiqueta verdadera")
        if save_confusion_path is not None:
            os.makedirs(os.path.dirname(save_confusion_path), exist_ok=True)
            plt.savefig(save_confusion_path, bbox_inches='tight')
            if verbose:
                print(f"[fit_rbf_svm_smartsearch] Matriz de confusión guardada en: {save_confusion_path}")
        plt.close()

        if verbose:
            print(f"[fit_rbf_svm_smartsearch] Mejor params: {best_cfg} con AUC={best_auc:.4f}")

        metrics = {
            'AUC_test': auc,
            'GINI_test': gini,
            'Accuracy_test': acc,
            'F1_test': f1,
            'BalancedAcc_test': bal_acc,
            'PR_AUC_test': pr_auc,
            'KS_test': ks,
            'Brier_test': brier,
            'ConfusionMatrix': cm
        }

        return best_model, best_cfg, metrics, y_pred, y_proba, X_train, X_test, y_train, y_test