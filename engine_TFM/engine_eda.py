# engine_eda.py
# Análisis exploratorio y visualizaciones para el engine TFM

import pandas as pd
import numpy as np
from engine_TFM.utils import correlacion_con_target
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.preprocessing import StandardScaler
from engine_TFM.utils import plot_pca_variance, plot_importancia_variables, anova_f_scores, plot_corr_heatmap, select_least_redundant_vars

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
    def separar_variables(df, target='flg_target'):
        """
        Devuelve un diccionario con listas de columnas categóricas y numéricas, excluyendo la variable objetivo.
        """
        cat_cols = df.select_dtypes(include='object').columns.tolist()
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if target in num_cols:
            num_cols.remove(target)
        return {'categoricas': cat_cols, 'numericas': num_cols}

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

class SelectVarsNumerics:
    @staticmethod
    def filtrar_por_nulos(df, resumen, target='flg_target', umbral=0.5, rescatar=False, min_corr=0.1, min_samples=30):
        """
        Filtra variables con % nulos > umbral. Si rescatar=True, rescata las que superan min_corr con el target.
        Imprime el proceso y devuelve la lista final de variables.
        """
        var_wNull = resumen[resumen['pcrt_nulos'] > umbral].index.tolist()
        print(f"Variables con más de {umbral*100:.0f}% de nulos: {len(var_wNull)}")
        rescatar_vars = []
        if rescatar and var_wNull:
            print("Calculando correlación con el target para posibles rescates...")
            df_corr = correlacion_con_target(df, var_wNull, target, min_samples)
            rescatar_vars = df_corr[df_corr['correlacion_con_target'] > min_corr].index.tolist()
            print(f"Variables rescatadas por correlación > {min_corr}: {len(rescatar_vars)}")
            print(df_corr.loc[rescatar_vars])
        vars_final = [v for v in resumen.index if v not in var_wNull or v in rescatar_vars]
        print(f"Variables numéricas tras filtrar nulos y rescatar: {len(vars_final)}")
        return vars_final

    @staticmethod
    def filtrar_por_cv(df, resumen, target='flg_target', umbral=0.1, rescatar=False, min_corr=0.1, min_samples=30):
        """
        Filtra variables con CV <= umbral. Si rescatar=True, rescata las que superan min_corr con el target.
        Imprime el proceso y devuelve la lista final de variables.
        """
        var_bajo_cv = resumen[resumen['CV'] <= umbral].index.tolist()
        print(f"Variables con CV <= {umbral}: {len(var_bajo_cv)}")
        rescatar_vars = []
        if rescatar and var_bajo_cv:
            print("Calculando correlación con el target para posibles rescates...")
            df_corr = correlacion_con_target(df, var_bajo_cv, target, min_samples)
            rescatar_vars = df_corr[df_corr['correlacion_con_target'] > min_corr].index.tolist()
            print(f"Variables rescatadas por correlación > {min_corr}: {len(rescatar_vars)}")
            print(df_corr.loc[rescatar_vars])
        vars_final = [v for v in resumen.index if v not in var_bajo_cv or v in rescatar_vars]
        print(f"Variables numéricas tras filtrar CV y rescatar: {len(vars_final)}")
        return vars_final

    @staticmethod
    def pca_con_grafico(df, num_cols, plot_variance: bool = True, verbose: bool = True):
        if verbose:
            print("Realizando PCA...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[num_cols].fillna(df[num_cols].median()))
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        varianza_explicada = pca.explained_variance_ratio_
        varianza_acumulada = varianza_explicada.cumsum()
        if verbose:
            print(f"Varianza explicada por los primeros 10 componentes: {varianza_acumulada[:10]}")
        if plot_variance:
            plot_pca_variance(varianza_acumulada)
        return pca, varianza_acumulada

    @staticmethod
    def pca_lda_importancia(df, num_cols, target='flg_target', n_pca_components=10, umbral_importancia=0.1, plot_graph=False, verbose: bool = True):
        if verbose:
            print(f"Realizando PCA (n_components={n_pca_components}) + LDA...")
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
        if verbose:
            print("Importancia de variables (top 10):")
            print(df_importancia.head(10))
        seleccionadas = df_importancia[df_importancia['importancia'] > umbral_importancia]['variable'].tolist()
        if verbose:
            print(f"Variables seleccionadas con importancia > {umbral_importancia}: {len(seleccionadas)}")
            print(df_importancia[df_importancia['importancia'] > umbral_importancia])
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
        if verbose:
            print("Resultados ANOVA F-test (top 10):")
            print(df_anova.head(10))
        seleccionadas = df_anova.copy()
        if min_f_score is not None:
            seleccionadas = seleccionadas[seleccionadas['f_score'] >= min_f_score]
            if verbose:
                print(f"Variables seleccionadas con f_score >= {min_f_score}: {len(seleccionadas)}")
        if max_p_value is not None:
            seleccionadas = seleccionadas[seleccionadas['p_value'] <= max_p_value]
            if verbose:
                print(f"Variables seleccionadas con p_value <= {max_p_value}: {len(seleccionadas)}")
        if top_n is not None:
            seleccionadas = seleccionadas.head(top_n)
            if verbose:
                print(f"Variables seleccionadas (top {top_n}): {len(seleccionadas)}")
        if verbose:
            print(seleccionadas)
        return seleccionadas['variable'].tolist(), df_anova

    @staticmethod
    def correlacion_feature_selection(df, num_cols, target='flg_target', threshold=0.6, plot_heatmap=True, verbose: bool = True):
        """
        Selecciona variables menos redundantes usando matriz de correlación y grafo.
        Grafica el heatmap si plot_heatmap=True.
        Devuelve la lista de variables seleccionadas y la matriz de correlación.
        """
        if verbose:
            print(f"Calculando matriz de correlación para {len(num_cols)} variables...")
        corr_matrix = df[num_cols].corr().abs()
        if plot_heatmap:
            plot_corr_heatmap(corr_matrix, annot=False)
        seleccionadas = select_least_redundant_vars(corr_matrix, df, target, threshold=threshold)
        if verbose:
            print(f"Variables seleccionadas tras eliminar redundancia (umbral={threshold}): {len(seleccionadas)}")
            print(seleccionadas)
        return seleccionadas, corr_matrix

# Las clases CleanData y SelectionVars se agregarán aquí en el futuro. 