# engine_eda.py
# Análisis exploratorio y visualizaciones para el engine TFM

import pandas as pd
import numpy as np
from engine_TFM.utils import correlacion_con_target
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from engine_TFM.utils import plot_pca_variance, plot_importancia_variables, anova_f_scores, plot_corr_heatmap, select_least_redundant_vars
import warnings

class Exploracion:
    @staticmethod
    def schema_summary(df):
        """
        Devuelve un DataFrame resumen con nombre de columna, tipo, % nulos y ejemplo de valor.
        """
        return pd.DataFrame({
            'columna': df.columns,
            'tipo': df.dtypes.values,
            '% nulos': df.isnull().mean().round(3),
            'ejemplo': df.iloc[0].values
        })

    @staticmethod
    def separar_variables_inteligente(df, target='flg_target', max_unique_cats=20, max_concentration=0.98, verbose=True, logger=None):
        """
        Separación inteligente de variables basada en contenido real, no solo tipos pandas.

        PASOS:
        1. Analizar contenido real de cada columna
        2. Convertir automáticamente columnas numéricas viables
        3. Filtrar categóricas por cardinalidad (< max_unique_cats)
        4. Filtrar categóricas por distribución (< max_concentration)
        5. Validar consistencia y retornar resultado

        Args:
            df: DataFrame a analizar
            target: nombre de la columna objetivo
            max_unique_cats: máximo número de valores únicos para mantener categórica
            max_concentration: máximo porcentaje que puede cubrir un solo valor
            verbose: si mostrar logs detallados

        Returns:
            dict: {'categoricas': [...], 'numericas': [...], 'descartadas': [...]}
        """
        def log_message(msg):
            if logger:
                logger.info(msg)
            elif verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass

        if verbose:
            log_message(f"\n🔍 ANALIZANDO SEPARACIÓN INTELIGENTE DE VARIABLES")
            log_message("=" * 60)
            log_message(f"Dataset: {df.shape[0]} filas × {df.shape[1]} columnas")
            log_message(f"Máximo valores únicos para categóricas: {max_unique_cats}")
            log_message(f"Máximo concentración por valor: {max_concentration*100:.1f}%")

        # Inicializar listas
        cat_cols = []
        num_cols = []
        descartadas_cols = []
        conversiones_num = []

        # Analizar cada columna
        for col in df.columns:
            if col == target:
                continue

            if verbose:
                log_message(f"\n📊 Analizando columna: {col}")

            # PASO 1: Analizar tipo actual
            tipo_actual = df[col].dtype
            valores_no_nulos = df[col].dropna()

            if len(valores_no_nulos) == 0:
                if verbose:
                    log_message(f"  ❌ Columna vacía - DESCARTADA")
                descartadas_cols.append(col)
                continue

            # PASO 2: Intentar conversión automática a numérico (solo si es realmente numérica continua)
            es_numerica = False
            if tipo_actual in ['int64', 'float64']:
                # Ya es numérica
                es_numerica = True
                num_cols.append(col)
                continue
            elif tipo_actual == 'object':
                # Solo convertir si TODOS los valores son números reales (no códigos/letras)
                try:
                    # Verificar si contiene letras o valores no numéricos
                    sample_values = valores_no_nulos.head(100).astype(str)
                    contiene_letras = any(any(c.isalpha() for c in str(val)) for val in sample_values)
                    contiene_codigos = any(len(str(val)) > 10 for val in sample_values)  # Probablemente códigos largos

                    if not contiene_letras and not contiene_codigos:
                        numeric_series = pd.to_numeric(valores_no_nulos, errors='coerce')
                        if not numeric_series.isna().any():
                            # Conversión exitosa y es realmente numérica continua
                            es_numerica = True
                            conversiones_num.append(col)
                            if verbose:
                                log_message(f"  ✅ Conversión automática a NUMÉRICA exitosa")
                                log_message(f"     Tipo original: {tipo_actual} → numérica")
                            num_cols.append(col)
                            continue
                except:
                    pass

            # PASO 3: Si no es numérica, analizar como categórica
            if not es_numerica:
                # Calcular métricas de cardinalidad
                n_unique = valores_no_nulos.nunique()
                total_valores = len(valores_no_nulos)

                if n_unique == 0:
                    if verbose:
                        log_message(f"  ❌ Sin valores únicos - DESCARTADA")
                    descartadas_cols.append(col)
                    continue

                # Calcular concentración del valor más común
                valor_mas_comun = valores_no_nulos.value_counts().iloc[0]
                concentracion = valor_mas_comun / total_valores

                if verbose:
                    log_message(f"  📈 Cardinalidad: {n_unique} valores únicos")
                    log_message(f"  📊 Concentración: {concentracion*100:.1f}% (valor más común)")

                # PASO 4: Aplicar filtros de calidad
                if n_unique > max_unique_cats:
                    if verbose:
                        log_message(f"  ❌ DESCARTADA: Demasiados valores únicos ({n_unique} > {max_unique_cats})")
                    descartadas_cols.append(col)
                elif concentracion > max_concentration:
                    if verbose:
                        log_message(f"  ❌ DESCARTADA: Concentración excesiva ({concentracion*100:.1f}% > {max_concentration*100:.1f}%)")
                    descartadas_cols.append(col)
                else:
                    if verbose:
                        log_message(f"  ✅ APROBADA como CATEGÓRICA")
                    cat_cols.append(col)

        # PASO 5: Resumen final
        if verbose:
            log_message(f"\n🎯 RESULTADO FINAL DE SEPARACIÓN:")
            log_message("=" * 60)
            log_message(f"✅ Variables NUMÉRICAS: {len(num_cols)}")
            for col in num_cols[:5]:  # Mostrar primeras 5
                log_message(f"   • {col}")
            if len(num_cols) > 5:
                log_message(f"   • ... y {len(num_cols)-5} más")

            log_message(f"\n✅ Variables CATEGÓRICAS: {len(cat_cols)}")
            for col in cat_cols[:5]:  # Mostrar primeras 5
                n_unique = df[col].nunique() if col in df.columns else 0
                log_message(f"   • {col} ({n_unique} valores únicos)")
            if len(cat_cols) > 5:
                log_message(f"   • ... y {len(cat_cols)-5} más")

            log_message(f"\n❌ Variables DESCARTADAS: {len(descartadas_cols)}")
            for col in descartadas_cols[:3]:  # Mostrar primeras 3
                log_message(f"   • {col}")
            if len(descartadas_cols) > 3:
                log_message(f"   • ... y {len(descartadas_cols)-3} más")

            if conversiones_num:
                log_message(f"\n🔄 Conversiones automáticas realizadas: {len(conversiones_num)}")
                for col in conversiones_num[:3]:
                    log_message(f"   • {col}")
                if len(conversiones_num) > 3:
                    log_message(f"   • ... y {len(conversiones_num)-3} más")

        return {
            'categoricas': cat_cols,
            'numericas': num_cols,
            'descartadas': descartadas_cols,
            'conversiones_automaticas': conversiones_num
        }

    @staticmethod
    def separar_variables(df, target='flg_target'):
        """
        MÉTODO LEGACY: Mantener compatibilidad hacia atrás.
        Redirige a la versión inteligente con parámetros conservadores.
        """
        if logger:
            logger.warning("Usando método legacy 'separar_variables()'")
            logger.info("💡 Recomendación: usar 'separar_variables_inteligente()' para mejor análisis")
        else:
            # Mensajes de warning solo en log (manejo interno)
            pass
        return Exploracion.separar_variables_inteligente(df, target, verbose=False)

    @staticmethod
    def resumen_numericas(df, num_cols):
        """
        Devuelve un DataFrame resumen estadístico extendido de las variables numéricas.
        """
        resumen = df[num_cols].describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99]).T
        resumen['pcrt_nulos'] = df[num_cols].isnull().mean().round(3)
        resumen['rango'] = resumen['max'] - resumen['min']
        resumen['CV'] = np.abs(resumen['std'] / resumen['mean']).round(3)
        return resumen

class SelectVarsCategoricals:
    @staticmethod
    def chi_square_test(df, cat_cols, target='flg_target', threshold=0.05, verbose=True, logger=None):
        """
        Realiza test de Chi-cuadrado para cada variable categórica vs target.
        Devuelve lista de variables que pasan el test estadístico.
        """
        from scipy.stats import chi2_contingency

        def log_message(msg):
            if logger:
                logger.info(msg)
            elif verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass

        selected_vars = []

        if verbose:
            log_message(f"\\n🎯 CHI-SQUARE TEST ANALYSIS")
            log_message("=" * 50)
            log_message(f"Analizando {len(cat_cols)} variables categóricas")
            log_message(f"Umbral p-value: {threshold}")

        for col in cat_cols:
            if col not in df.columns:
                continue

            if verbose:
                log_message(f"\\n📊 Analizando: {col}")

            # Crear tabla de contingencia
            contingency_table = pd.crosstab(df[col], df[target])

            if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                if verbose:
                    log_message(f"  ❌ Tabla muy pequeña - Omitida")
                continue

            # Calcular Chi-square
            chi2, p_value, dof, expected = chi2_contingency(contingency_table)

            if verbose:
                log_message(f"  📈 Chi2: {chi2:.2f}")
                log_message(f"  📊 p-value: {p_value:.4f}")
                log_message(f"  🎯 Decisión: {'✅ PASÓ' if p_value < threshold else '❌ FALLÓ'}")

            if p_value < threshold:
                selected_vars.append(col)

        if verbose:
            log_message(f"\\n🎯 RESULTADO CHI-SQUARE:")
            log_message(f"   ✅ Variables seleccionadas: {len(selected_vars)}")
            log_message(f"   ❌ Variables descartadas: {len(cat_cols) - len(selected_vars)}")

        return selected_vars

    @staticmethod
    def calculate_woe_iv(df, cat_cols, target='flg_target', verbose=True, logger=None):
        """
        Calcula Weight of Evidence (WOE) e Information Value (IV) para variables categóricas.
        Devuelve ranking de variables por poder predictivo.
        """
        woe_iv_results = {}

        def log_message(msg):
            if logger:
                logger.info(msg)
            elif verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass

        if verbose:
            log_message(f"\\n📊 WOE/IV ANALYSIS")
            log_message("=" * 50)
            log_message(f"Analizando {len(cat_cols)} variables categóricas")

        for col in cat_cols:
            if col not in df.columns:
                continue

            if verbose:
                log_message(f"\\n📊 Analizando: {col}")

            # Crear tabla de contingencia
            contingency_table = pd.crosstab(df[col], df[target], margins=True)

            # Calcular WOE e IV
            woe_dict = {}
            iv = 0

            for category in contingency_table.index[:-1]:  # Excluir 'All'
                if category in contingency_table.index:
                    good = contingency_table.loc[category, 1] if 1 in contingency_table.columns else 0
                    bad = contingency_table.loc[category, 0] if 0 in contingency_table.columns else 0
                    total_good = contingency_table.loc['All', 1] if 1 in contingency_table.columns else 0
                    total_bad = contingency_table.loc['All', 0] if 0 in contingency_table.columns else 0

                    # Evitar división por cero
                    good_rate = good / total_good if total_good > 0 else 0
                    bad_rate = bad / total_bad if total_bad > 0 else 0

                    if good_rate > 0 and bad_rate > 0:
                        woe = np.log(good_rate / bad_rate)
                        woe_dict[category] = woe

                        # Contribución al IV
                        iv += (good_rate - bad_rate) * woe
                    else:
                        woe_dict[category] = 0

            woe_iv_results[col] = {
                'woe_values': woe_dict,
                'iv': iv,
                'iv_category': SelectVarsCategoricals._categorize_iv(iv)
            }

            if verbose:
                log_message(f"  📈 Information Value: {iv:.4f}")
                log_message(f"  🎯 Categoría IV: {woe_iv_results[col]['iv_category']}")

        # Ranking por IV
        ranking = sorted(woe_iv_results.items(), key=lambda x: x[1]['iv'], reverse=True)

        if verbose:
            log_message(f"\\n🎯 RANKING POR IV (top 5):")
            for i, (col, data) in enumerate(ranking[:5]):
                log_message(f"  {i+1}. {col}: IV={data['iv']:.4f} ({data['iv_category']})")

        return woe_iv_results, ranking

    @staticmethod
    def _categorize_iv(iv_value):
        """Categoriza el Information Value según estándares de la industria."""
        if iv_value < 0.02:
            return "No predictiva"
        elif iv_value < 0.1:
            return "Débil"
        elif iv_value < 0.3:
            return "Media"
        elif iv_value < 0.5:
            return "Fuerte"
        else:
            return "Muy fuerte"

    @staticmethod
    def select_categorical_variables(df, cat_cols, target='flg_target', iv_threshold=0.1, verbose=True, logger=None):
        """
        Pipeline completo de selección de variables categóricas.
        Combina Chi-square + WOE/IV para selección robusta.
        """
        def log_message(msg):
            if logger:
                logger.info(msg)
            elif verbose:
                # Información detallada solo en log (manejo desde TFM_EDA.py)
                pass

        if verbose:
            log_message(f"\\n🎯 PIPELINE COMPLETO: SELECCIÓN DE VARIABLES CATEGÓRICAS")
            log_message("=" * 70)

        # Paso 1: Chi-square test
        if verbose:
            log_message(f"\\nPASO 1: Chi-square Test")
        chi_selected = SelectVarsCategoricals.chi_square_test(df, cat_cols, target, verbose=verbose, logger=logger)

        # Paso 2: WOE/IV Analysis
        if verbose:
            log_message(f"\\nPASO 2: WOE/IV Analysis")
        woe_iv_results, ranking = SelectVarsCategoricals.calculate_woe_iv(df, chi_selected, target, verbose=verbose, logger=logger)

        # Paso 3: Selección final por IV
        final_selected = []
        for col, data in woe_iv_results.items():
            if data['iv'] >= iv_threshold:
                final_selected.append(col)

        if verbose:
            log_message(f"\\n🎯 SELECCIÓN FINAL:")
            log_message(f"   📊 Umbral IV mínimo: {iv_threshold}")
            log_message(f"   ✅ Variables seleccionadas: {len(final_selected)}")
            log_message(f"   ❌ Variables descartadas: {len(chi_selected) - len(final_selected)}")

            if final_selected:
                log_message(f"\\n🏆 VARIABLES CATEGÓRICAS SELECCIONADAS:")
                for col in final_selected:
                    iv = woe_iv_results[col]['iv']
                    category = woe_iv_results[col]['iv_category']
                    log_message(f"   • {col}: IV={iv:.4f} ({category})")

        return final_selected, woe_iv_results, ranking


class SelectVarsNumerics:
    @staticmethod
    def filtrar_por_nulos(df, resumen, target='flg_target', umbral=0.5, rescatar=False, min_corr=0.1, min_samples=30, logger=None):
        """
        Filtra variables con % nulos > umbral. Si rescatar=True, rescata las que superan min_corr con el target.
        Imprime el proceso y devuelve la lista final de variables.
        """
        def log_message(msg):
            if logger:
                logger.info(msg)
            # No print a terminal, solo a log

        var_wNull = resumen[resumen['pcrt_nulos'] > umbral].index.tolist()
        log_message(f"Variables con más de {umbral*100:.0f}% de nulos: {len(var_wNull)}")
        rescatar_vars = []
        if rescatar and var_wNull:
            log_message("Calculando correlación con el target para posibles rescates...")
            df_corr = correlacion_con_target(df, var_wNull, target, min_samples)
            rescatar_vars = df_corr[df_corr['correlacion_con_target'] > min_corr].index.tolist()
            log_message(f"Variables rescatadas por correlación > {min_corr}: {len(rescatar_vars)}")
            log_message(str(df_corr.loc[rescatar_vars]))
        vars_final = [v for v in resumen.index if v not in var_wNull or v in rescatar_vars]
        log_message(f"Variables numéricas tras filtrar nulos y rescatar: {len(vars_final)}")
        return vars_final

    @staticmethod
    def filtrar_por_cv(df, resumen, target='flg_target', umbral=0.1, rescatar=False, min_corr=0.1, min_samples=30, logger=None):
        """
        Filtra variables con CV <= umbral. Si rescatar=True, rescata las que superan min_corr con el target.
        Imprime el proceso y devuelve la lista final de variables.
        """
        def log_message(msg):
            if logger:
                logger.info(msg)
            # No print a terminal, solo a log

        var_bajo_cv = resumen[resumen['CV'] <= umbral].index.tolist()
        log_message(f"Variables con CV <= {umbral}: {len(var_bajo_cv)}")
        rescatar_vars = []
        if rescatar and var_bajo_cv:
            log_message("Calculando correlación con el target para posibles rescates...")
            df_corr = correlacion_con_target(df, var_bajo_cv, target, min_samples)
            rescatar_vars = df_corr[df_corr['correlacion_con_target'] > min_corr].index.tolist()
            log_message(f"Variables rescatadas por correlación > {min_corr}: {len(rescatar_vars)}")
            log_message(str(df_corr.loc[rescatar_vars]))
        vars_final = [v for v in resumen.index if v not in var_bajo_cv or v in rescatar_vars]
        log_message(f"Variables numéricas tras filtrar CV y rescatar: {len(vars_final)}")
        return vars_final

    @staticmethod
    def pca_con_grafico(df, num_cols, plot_variance: bool = True, verbose: bool = True):
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = varianza_explicada.cumsum()
        if verbose:
            # Información detallada solo en log (manejo desde TFM_EDA.py)
            pass
        if plot_variance:
            plot_pca_variance(varianza_acumulada)
        return pca, varianza_acumulada

    @staticmethod
    def pca_lda_importancia(df, num_cols, target='flg_target', n_pca_components=10, umbral_importancia=0.1, plot_graph=False, verbose: bool = True):
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        pca = PCA(n_components=n_pca_components)
        X_pca = pca.fit_transform(X_scaled)
        lda = LDA(n_components=1)
        lda.fit(X_pca, df[target])
        lda_coef = lda.coef_[0]
        pca_loadings = pca.components_
        importancia_vars = np.abs(np.dot(lda_coef, pca_loadings))
        df_importancia = pd.DataFrame({
            'variable': num_cols,
            'importancia': importancia_vars
        }).sort_values(by='importancia', ascending=False).reset_index(drop=True)
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        seleccionadas = df_importancia[df_importancia['importancia'] > umbral_importancia]['variable'].tolist()
        if plot_graph:
            plot_importancia_variables(df_importancia)
        return seleccionadas, df_importancia

    @staticmethod
    def anova_feature_selection(df, num_cols, target='flg_target', min_f_score=None, max_p_value=0.05, top_n=None, verbose: bool = True):
        """
        Selecciona variables numéricas usando ANOVA F-test. Permite filtrar por f_score mínimo, p_value máximo o top_n.
        Imprime el resumen y devuelve la lista de variables seleccionadas y el DataFrame de scores.
        """
        df_anova = anova_f_scores(df, num_cols, target)
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        seleccionadas = df_anova.copy()
        if min_f_score is not None:
            seleccionadas = seleccionadas[seleccionadas['f_score'] >= min_f_score]
        if max_p_value is not None:
            seleccionadas = seleccionadas[seleccionadas['p_value'] <= max_p_value]
        if top_n is not None:
            seleccionadas = seleccionadas.head(top_n)
        return seleccionadas['variable'].tolist(), df_anova

    @staticmethod
    def correlacion_feature_selection(df, num_cols, target='flg_target', threshold=0.6, plot_heatmap=True, verbose: bool = True):
        """
        Selecciona variables menos redundantes usando matriz de correlación y grafo.
        Grafica el heatmap si plot_heatmap=True.
        Devuelve la lista de variables seleccionadas y la matriz de correlación.
        """
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        corr_matrix = df[num_cols].corr().abs()
        if plot_heatmap:
            plot_corr_heatmap(corr_matrix, annot=False)
        seleccionadas = select_least_redundant_vars(corr_matrix, df, target, threshold=threshold)
        # Información detallada solo en log (manejo desde TFM_EDA.py)
        return seleccionadas, corr_matrix

# Las clases CleanData y SelectionVars se agregarán aquí en el futuro. 