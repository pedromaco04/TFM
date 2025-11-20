import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np


def _log(logger: Optional[logging.Logger], message: str) -> None:
    """
    Envía mensajes al logger si está disponible.
    """
    if logger is not None:
        logger.info(message)


def verify_tasa_direction(
    models_dict: Dict[str, Any],
    tasa_column: str = 'tasa',
    expected_negative: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Verificación específica para la variable TASA (debe ser negativa típicamente).
    Args:
        models_dict: diccionario retornado por fit_logistic_regression
        tasa_column: nombre de la columna TASA
        expected_negative: si True, espera coeficiente negativo; si False, positivo
        logger: logger opcional
    Returns:
        dict con:
          - 'verification': Dict[segmento -> {'coefficient': float, 'expected': str, 'correct': bool}]
          - 'summary': DataFrame con resumen por segmento
          - 'all_correct': bool indicando si todos los segmentos tienen direccionalidad correcta
    """
    from .regression import get_coefficients_summary
    
    _log(logger, f"Verificando direccionalidad de TASA (esperada: {'negativa' if expected_negative else 'positiva'})")
    
    coef_summary = get_coefficients_summary(models_dict, logger=logger)
    tasa_coefs = coef_summary[coef_summary['variable'] == tasa_column].copy()
    
    if tasa_coefs.empty:
        _log(logger, f"ADVERTENCIA: Variable '{tasa_column}' no encontrada en los modelos")
        return {
            'verification': {},
            'summary': pd.DataFrame(),
            'all_correct': False,
            'error': f"Variable '{tasa_column}' no encontrada"
        }
    
    verification = {}
    summary_rows = []
    
    for _, row in tasa_coefs.iterrows():
        segment = row['segment']
        coef = row['coefficient']
        is_negative = coef < 0
        is_correct = (is_negative == expected_negative)
        
        verification[segment] = {
            'coefficient': float(coef),
            'expected': 'negative' if expected_negative else 'positive',
            'actual': 'negative' if is_negative else 'positive',
            'correct': is_correct,
        }
        
        summary_rows.append({
            'segment': segment,
            'coefficient': float(coef),
            'expected_direction': 'negative' if expected_negative else 'positive',
            'actual_direction': 'negative' if is_negative else 'positive',
            'is_correct': is_correct,
        })
        
        status = "✓" if is_correct else "✗"
        _log(logger, f"Segmento '{segment}': TASA coef={coef:.6f} ({'negativo' if is_negative else 'positivo'}) {status}")
    
    summary_df = pd.DataFrame(summary_rows)
    all_correct = summary_df['is_correct'].all() if not summary_df.empty else False
    
    if all_correct:
        _log(logger, "✓ Todas las direccionalidades de TASA son correctas")
    else:
        incorrect = summary_df[~summary_df['is_correct']]['segment'].tolist()
        _log(logger, f"✗ ADVERTENCIA: Segmentos con direccionalidad incorrecta de TASA: {incorrect}")
    
    return {
        'verification': verification,
        'summary': summary_df,
        'all_correct': all_correct,
    }


def check_directionality(
    models_dict: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    expected_directions: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Verifica que los signos de los coeficientes coincidan con direcciones esperadas.
    Args:
        models_dict: diccionario retornado por fit_logistic_regression
        feature_columns: lista de features a verificar (si None, usa todas)
        expected_directions: dict {variable: 'positive'|'negative'} con direcciones esperadas
        logger: logger opcional
    Returns:
        dict con:
          - 'verification': Dict[segmento -> Dict[variable -> {'coefficient': float, 'expected': str, 'correct': bool}]]
          - 'summary': DataFrame con resumen por variable y segmento
          - 'all_correct': bool indicando si todas las verificaciones son correctas
    """
    from .regression import get_coefficients_summary
    
    if expected_directions is None:
        expected_directions = {}
    
    if feature_columns is None:
        feature_columns = models_dict['feature_columns']
    
    _log(logger, f"Verificando direccionalidad de {len(feature_columns)} variables")
    
    coef_summary = get_coefficients_summary(models_dict, feature_columns=feature_columns, logger=logger)
    
    # Filtrar solo las variables de interés
    coef_summary = coef_summary[coef_summary['variable'].isin(feature_columns)].copy()
    
    verification: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summary_rows = []
    
    for segment in coef_summary['segment'].unique():
        verification[segment] = {}
        seg_coefs = coef_summary[coef_summary['segment'] == segment]
        
        for _, row in seg_coefs.iterrows():
            var = row['variable']
            coef = row['coefficient']
            is_negative = coef < 0
            actual_dir = 'negative' if is_negative else 'positive'
            
            # Si no hay dirección esperada, marcar como 'not_specified'
            expected_dir = expected_directions.get(var, 'not_specified')
            is_correct = True
            if expected_dir != 'not_specified':
                is_correct = (actual_dir == expected_dir)
            
            verification[segment][var] = {
                'coefficient': float(coef),
                'expected': expected_dir,
                'actual': actual_dir,
                'correct': is_correct,
            }
            
            summary_rows.append({
                'segment': segment,
                'variable': var,
                'coefficient': float(coef),
                'expected_direction': expected_dir,
                'actual_direction': actual_dir,
                'is_correct': is_correct,
            })
    
    summary_df = pd.DataFrame(summary_rows)
    
    # Calcular estadísticas
    if not summary_df.empty:
        total_checks = len(summary_df)
        specified_checks = summary_df[summary_df['expected_direction'] != 'not_specified']
        correct_checks = specified_checks['is_correct'].sum() if not specified_checks.empty else 0
        total_specified = len(specified_checks)
        
        _log(logger, f"Verificación completada: {correct_checks}/{total_specified} correctas (de {total_specified} especificadas)")
        
        if total_specified > 0:
            incorrect = summary_df[
                (summary_df['expected_direction'] != 'not_specified') & 
                (~summary_df['is_correct'])
            ]
            if not incorrect.empty:
                _log(logger, f"Variables con direccionalidad incorrecta:")
                for _, row in incorrect.iterrows():
                    _log(logger, f"  - Segmento '{row['segment']}', Variable '{row['variable']}': "
                         f"esperado {row['expected_direction']}, actual {row['actual_direction']}")
        
        all_correct = (correct_checks == total_specified) if total_specified > 0 else True
    else:
        all_correct = True
    
    return {
        'verification': verification,
        'summary': summary_df,
        'all_correct': all_correct,
    }


def get_directionality_report(
    models_dict: Dict[str, Any],
    feature_columns: Optional[List[str]] = None,
    expected_directions: Optional[Dict[str, str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Genera reporte tabular de direccionalidad por variable y segmento.
    Args:
        models_dict: diccionario retornado por fit_logistic_regression
        feature_columns: lista de features (si None, usa todas)
        expected_directions: dict {variable: 'positive'|'negative'} con direcciones esperadas
        logger: logger opcional
    Returns:
        DataFrame pivotado con segmentos como columnas y variables como filas,
        mostrando coeficientes y estado de verificación
    """
    check_result = check_directionality(
        models_dict=models_dict,
        feature_columns=feature_columns,
        expected_directions=expected_directions,
        logger=logger
    )
    
    summary = check_result['summary']
    
    if summary.empty:
        return pd.DataFrame()
    
    # Crear reporte pivotado
    report_data = []
    for var in summary['variable'].unique():
        var_data = summary[summary['variable'] == var]
        row = {'variable': var}
        
        for segment in summary['segment'].unique():
            seg_data = var_data[var_data['segment'] == segment]
            if not seg_data.empty:
                coef = seg_data.iloc[0]['coefficient']
                is_correct = seg_data.iloc[0]['is_correct']
                expected = seg_data.iloc[0]['expected_direction']
                actual = seg_data.iloc[0]['actual_direction']
                
                # Formato: coeficiente (dirección) [estado]
                status = "✓" if is_correct else "✗"
                if expected != 'not_specified':
                    row[f"{segment}_coef"] = f"{coef:.6f}"
                    row[f"{segment}_dir"] = f"{actual} {status}"
                else:
                    row[f"{segment}_coef"] = f"{coef:.6f}"
                    row[f"{segment}_dir"] = actual
            else:
                row[f"{segment}_coef"] = np.nan
                row[f"{segment}_dir"] = np.nan
        
        report_data.append(row)
    
    report_df = pd.DataFrame(report_data)
    return report_df

