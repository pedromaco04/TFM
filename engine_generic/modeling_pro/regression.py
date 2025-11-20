import logging
from typing import Dict, List, Optional, Union, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """
    Envía mensajes al logger si está disponible.
    Args:
        logger: instancia de logging.Logger o None
        message: texto a registrar
    """
    if logger is not None:
        logger.info(message)


def fit_logistic_regression(
    df: pd.DataFrame,
    feature_columns: Union[str, List[str]],
    target_column: str,
    segment_column: Optional[str] = None,
    random_state: Optional[int] = None,
    max_iter: int = 1000,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Ajusta modelos de regresión logística por segmento (o global si no hay segmento).
    Args:
        df: DataFrame con datos de entrenamiento
        feature_columns: lista de columnas a usar como features
        target_column: nombre de la columna objetivo (binaria 0/1)
        segment_column: columna de segmentación (opcional). Si se proporciona, ajusta un modelo por segmento
        random_state: semilla para reproducibilidad
        max_iter: número máximo de iteraciones para el solver
        logger: logger opcional
    Returns:
        dict con:
          - 'models': Dict[segmento -> LogisticRegression] (o 'global' si no hay segmento)
          - 'metrics': Dict[segmento -> dict con métricas]
          - 'summary': DataFrame con resumen de métricas por segmento
          - 'feature_columns': lista de features usadas
    """
    if isinstance(feature_columns, str):
        feature_columns = [feature_columns]
    
    # Validar que existan las columnas
    missing_cols = [c for c in feature_columns + [target_column] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}")
    
    if segment_column is not None and segment_column not in df.columns:
        raise ValueError(f"Columna de segmento '{segment_column}' no encontrada")
    
    _log(logger, f"Ajustando modelos de regresión logística (features: {len(feature_columns)})")
    
    models_dict: Dict[str, LogisticRegression] = {}
    metrics_dict: Dict[str, Dict[str, float]] = {}
    
    # Preparar datos
    # NOTA: Se asume que las variables categóricas ya vienen codificadas (One-Hot) desde feature_pro
    X = df[feature_columns].copy()
    y = df[target_column].copy()
    
    # Eliminar filas con target faltante
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    
    # NOTA: Se asume que feature_pro ya manejó:
    # - Codificación de variables categóricas (One-Hot)
    # - Imputación de missings
    # - Todas las variables son numéricas
    
    # Usar todas las columnas directamente (ya están preprocesadas)
    X = X[feature_columns]
    
    # Verificación silenciosa: rellenar cualquier NaN residual (no debería haber, pero por seguridad)
    if X.isna().any().any():
        X = X.fillna(0)
    
    feature_names_out = feature_columns
    
    if segment_column is None:
        # Modelo global
        _log(logger, "Ajustando modelo global (sin segmentación)")
        model = LogisticRegression(
            random_state=random_state,
            max_iter=max_iter,
            solver='lbfgs' if len(feature_names_out) > 1 else 'liblinear'
        )
        model.fit(X, y)
        models_dict['global'] = model
        
        # Métricas
        y_pred_proba = model.predict_proba(X)[:, 1]
        y_pred = model.predict(X)
        metrics = _calculate_metrics(y, y_pred, y_pred_proba, logger)
        metrics_dict['global'] = metrics
        _log(logger, f"Métricas globales - GINI: {metrics['gini']:.4f}, AUC: {metrics['auc']:.4f}")
    else:
        # Modelos por segmento
        segments = df.loc[valid_mask, segment_column].unique()
        _log(logger, f"Ajustando modelos por segmento ({len(segments)} segmentos)")
        
        for segment in segments:
            segment_mask = (df.loc[valid_mask, segment_column] == segment)
            X_seg = X[segment_mask]
            y_seg = y[segment_mask]
            
            if len(y_seg) < 10:
                _log(logger, f"Segmento '{segment}' tiene muy pocos registros ({len(y_seg)}). Se omite.")
                continue
            
            if y_seg.nunique() < 2:
                _log(logger, f"Segmento '{segment}' no tiene ambas clases. Se omite.")
                continue
            
            # Usar directamente (variables ya preprocesadas desde feature_pro)
            X_seg = X_seg[feature_columns]
            
            model = LogisticRegression(
                random_state=random_state,
                max_iter=max_iter,
                solver='lbfgs' if len(feature_names_out) > 1 else 'liblinear'
            )
            model.fit(X_seg, y_seg)
            models_dict[str(segment)] = model
            
            # Métricas
            y_pred_proba = model.predict_proba(X_seg)[:, 1]
            y_pred = model.predict(X_seg)
            metrics = _calculate_metrics(y_seg, y_pred, y_pred_proba, logger)
            metrics_dict[str(segment)] = metrics
            _log(logger, f"Segmento '{segment}' - GINI: {metrics['gini']:.4f}, AUC: {metrics['auc']:.4f}, N: {len(y_seg)}")
    
    # Resumen de métricas
    summary_rows = []
    for seg, metrics in metrics_dict.items():
        summary_rows.append({
            'segment': seg,
            'gini': metrics['gini'],
            'auc': metrics['auc'],
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'n_samples': metrics['n_samples'],
        })
    summary_df = pd.DataFrame(summary_rows)
    
    return {
        'models': models_dict,
        'metrics': metrics_dict,
        'summary': summary_df,
        'feature_columns': feature_columns,  # Columnas (ya codificadas desde feature_pro)
        'feature_names_out': feature_names_out,  # Mismo que feature_columns
    }


def _calculate_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, float]:
    """
    Calcula métricas de evaluación del modelo.
    """
    try:
        auc = roc_auc_score(y_true, y_pred_proba)
        gini = 2 * auc - 1
    except Exception as e:
        _log(logger, f"Error calculando AUC/GINI: {e}")
        auc = np.nan
        gini = np.nan
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'gini': float(gini),
        'auc': float(auc),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'n_samples': int(len(y_true)),
    }


def predict_logistic(
    models_dict: Dict[str, Any],
    df: pd.DataFrame,
    segment_column: Optional[str] = None,
    return_proba: bool = True,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Genera predicciones usando los modelos ajustados.
    Args:
        models_dict: diccionario retornado por fit_logistic_regression
        df: DataFrame con datos para predecir
        segment_column: columna de segmentación (debe coincidir con la usada en el ajuste)
        return_proba: si True, retorna probabilidades; si False, retorna clases predichas
        logger: logger opcional
    Returns:
        DataFrame con columnas 'pred_proba' (o 'pred_class') y 'segment' (si aplica)
    """
    models = models_dict['models']
    feature_columns = models_dict['feature_columns']
    feature_names_out = models_dict.get('feature_names_out', feature_columns)
    
    _log(logger, f"Generando predicciones para {len(df)} registros")
    
    # Validar columnas
    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}")
    
    # NOTA: Se asume que feature_pro ya manejó:
    # - Codificación de variables categóricas (One-Hot)
    # - Imputación de missings
    # - Todas las variables son numéricas
    
    X = df[feature_columns].copy()
    
    # Usar directamente (variables ya preprocesadas desde feature_pro)
    X = X[feature_columns]
    
    # Verificación silenciosa: rellenar cualquier NaN residual (no debería haber, pero por seguridad)
    if X.isna().any().any():
        X = X.fillna(0)
    
    results = pd.DataFrame(index=df.index)
    
    if segment_column is None or 'global' in models:
        # Predicción global
        model = models.get('global', list(models.values())[0])
        if return_proba:
            results['pred_proba'] = model.predict_proba(X)[:, 1]
        else:
            results['pred_class'] = model.predict(X)
    else:
        # Predicción por segmento
        if segment_column not in df.columns:
            raise ValueError(f"Columna de segmento '{segment_column}' no encontrada")
        
        pred_proba_list = []
        pred_class_list = []
        segment_list = []
        
        for segment in df[segment_column].unique():
            segment_mask = df[segment_column] == segment
            X_seg = X[segment_mask]
            
            if str(segment) not in models:
                _log(logger, f"Segmento '{segment}' no tiene modelo. Usando modelo global si existe.")
                model = models.get('global', list(models.values())[0])
            else:
                model = models[str(segment)]
            
            if return_proba:
                proba = model.predict_proba(X_seg)[:, 1]
                pred_proba_list.extend(proba)
            else:
                pred = model.predict(X_seg)
                pred_class_list.extend(pred)
            
            segment_list.extend([segment] * len(X_seg))
        
        if return_proba:
            results['pred_proba'] = pred_proba_list
        else:
            results['pred_class'] = pred_class_list
        results['segment'] = segment_list
    
    _log(logger, "Predicciones generadas")
    return results


def get_coefficients_summary(
    models_dict: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Extrae y resume coeficientes de todos los modelos.
    Args:
        models_dict: diccionario retornado por fit_logistic_regression
        feature_columns: lista de features (si None, usa las del models_dict después del encoding)
        logger: logger opcional
    Returns:
        DataFrame con columnas ['segment', 'variable', 'coefficient', 'abs_coefficient']
    """
    models = models_dict['models']
    # Usar feature_names_out (después del encoding) en lugar de feature_columns originales
    if feature_columns is None:
        feature_columns = models_dict.get('feature_names_out', models_dict['feature_columns'])
    
    _log(logger, "Extrayendo coeficientes de modelos")
    
    rows = []
    for segment, model in models.items():
        coefficients = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        intercept = model.intercept_[0] if hasattr(model.intercept_, '__len__') else model.intercept_
        
        # Intercept
        rows.append({
            'segment': segment,
            'variable': 'intercept',
            'coefficient': float(intercept),
            'abs_coefficient': float(abs(intercept)),
        })
        
        # Features (usar feature_names_out que incluye las columnas codificadas)
        for var, coef in zip(feature_columns, coefficients):
            rows.append({
                'segment': segment,
                'variable': var,
                'coefficient': float(coef),
                'abs_coefficient': float(abs(coef)),
            })
    
    summary = pd.DataFrame(rows)
    _log(logger, f"Coeficientes extraídos para {len(models)} modelos")
    return summary

