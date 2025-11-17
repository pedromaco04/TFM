## feature_pro

Librería modular para EDA, utilidades de preprocesamiento y selección de variables. Su objetivo es ofrecer funciones genéricas y reutilizables, con una organización clara para crecer hacia pipelines más avanzados.

### Estructura actual

```text
feature_pro/
├── __init__.py                 # Reexporta la API pública
├── common/
│   ├── __init__.py
│   └── utils.py                # Utilidades genéricas y estables (EDA/preprocesamiento/estadística)
└── selection/
    ├── __init__.py
    └── feature_selection.py    # Funciones de selección de variables
```

Sugeridos a futuro (no obligatorios aún):

```text
feature_pro/
├── preprocessing/              # Pipelines de preprocesamiento (orquestación)
├── transformation/             # Ingeniería de variables
└── analysis/                   # Visualizaciones y análisis descriptivo
```

### Principios de diseño

- Utilidades realmente genéricas viven en `common/utils.py` para evitar duplicación y facilitar el reuso.
- Funciones con foco en selección/reducción quedan bajo `selection/`.
- Impresión en consola mínima (solo títulos/secciones). El detalle se envía al log mediante `logging` (coherente con la preferencia de no saturar la terminal).

### Módulos y funciones

#### common/utils.py

- `read_dataset(path, fmt='auto', sep=None, encoding=None, dtype_backend='numpy_nullable', low_memory=False, sheet_name=0, logger=None) -> pd.DataFrame`
  - Lector genérico: `csv`, `parquet`, `txt`, `tsv`, `xlsx` (detección por extensión si `fmt='auto'`).

- `summarize_missing(df, columns=None) -> pd.DataFrame`
  - Tabla resumen por columna con conteo y porcentaje de missings, y dtype.

- `detect_column_types(df, columns=None, treat_object_numeric=True, sample_size_for_analysis=100, max_code_length=10) -> (numeric_cols, categorical_cols)`
  - Identifica numéricas y categóricas incluso si numéricas están almacenadas como `object` (detecta si parecen números reales).

- `impute_missing(df, columns=None, strategy='median', fill_value=None, groupby=None, logger=None) -> pd.DataFrame`
  - Imputación genérica. Estrategias: `mean`, `median`, `zero`, `constant`. Soporta imputación por `groupby`.

- `winsorize_by_percentile(df, columns, lower=0.01, upper=0.99, logger=None, return_percentiles=False) -> pd.DataFrame | (pd.DataFrame, pd.DataFrame)`
  - Winsorización por percentiles para columnas numéricas. Si `return_percentiles=True`, devuelve también un DataFrame con los percentiles usados por variable.

- `winsorize_by_iqr(df, columns, factor=1.5, logger=None) -> pd.DataFrame`
  - Winsorización por IQR: recorte a `[Q1 - factor*IQR, Q3 + factor*IQR]`.

- `compute_correlation_matrix(df, columns=None, method='pearson', plot=False, return_fig=True, logger=None) -> dict`
  - Matriz de correlaciones flexible. Métodos numéricos (`pearson`, `spearman`, `kendall`) o `auto` mixto:
    - num-num: Pearson
    - cat-cat: Cramér’s V
    - num-cat: Correlation Ratio (eta)
  - Puede devolver un heatmap (figura) opcional.

- `count_unique_categorical(df, columns=None, include_na=False, logger=None) -> DataFrame`
  - Devuelve `['variable','n_unique']` con el conteo de valores únicos por variable categórica.
  - Si `columns` es None, detecta categóricas automáticamente; con `include_na=True` cuenta `NaN` como categoría.

#### selection/numeric_selection.py

- `compute_pca(df, columns, n_components=None, variance_threshold=None, scale=True, plot=False, return_fig=True, random_state=None) -> dict`
  - Ajuste flexible de PCA (selección de componentes). Determina `n_components` por `variance_threshold` o fijo. Puede devolver figura con varianza acumulada y marca el n elegido.

- `compute_psi(df, temporal_column, variables=None, num_bins=10, bin_strategy='quantile', reference_period=None, return_detail=False, logger=None, epsilon=1e-8) -> DataFrame | (DataFrame, DataFrame)`
  - Population Stability Index (PSI) para evaluar estabilidad de variables a través del tiempo. Si `variables` es None, toma todas excepto la temporal. Maneja numéricas (bins por cuantiles o anchos iguales con fallback) y categóricas (incluye 'MISSING').

- `coefficient_of_variation(df, columns=None) -> pd.DataFrame`
  - Calcula CV = |std/mean| por columna numérica.

- `pca_lda_importance(df, feature_columns, target_column, n_pca_components=None, variance_threshold=None, scale=True, plot=False, return_fig=True, random_state=None, top_n=None) -> dict`
  - Importancia de variables vía PCA + LDA. Proyecta los coeficientes de LDA al espacio original para obtener un ranking por variable. Puede graficar el ranking.

- `select_by_correlation_graph(df, columns=None, target_column=None, method='auto', threshold=0.6, prioritize=None, plot_matrix=False, return_matrix_fig=True, return_graph=False, logger=None) -> dict`
  - Selección por redundancia mediante grafo de correlaciones (aristas cuando |corr| ≥ threshold). De cada componente conexo conserva:
    - Una variable priorizada si pertenece al grupo, en caso contrario
    - La variable con mayor asociación absoluta respecto al `target_column` (si se proporciona), o la de mayor conectividad como fallback.
  - Devuelve la lista seleccionada, la matriz, la figura del heatmap (opcional) y el grafo (opcional).
  - Parámetro adicional soportado: `target_method` para especificar la métrica de asociación con el objetivo (p.ej. `pointbiserial`). Para correlación entre variables, se recomienda `method='spearman'` en casos no lineales.

#### selection/categorical_selection.py

- `count_unique_categorical(df, columns=None, include_na=False, logger=None) -> DataFrame`
  - Devuelve `['variable','n_unique']` con el conteo de valores únicos por variable categórica.
  - Si `columns` es None, detecta categóricas automáticamente; con `include_na=True` cuenta `NaN` como categoría.

- `categorical_cumulative_frequency(df, columns=None, threshold=0.8, include_na=False, return_distributions=False, logger=None) -> dict`
  - Calcula la frecuencia acumulada de valores para cada variable categórica y evalúa dominancia.
  - Si la proporción del valor más frecuente de una variable ≥ `threshold`, esa variable es marcada como dominante.
  - Retorna un diccionario con:
    - `summary`: DataFrame [`variable`, `top_value`, `top_count`, `top_pct`, `num_unique`, `is_dominant`]
    - `kept_columns`: lista de variables no dominantes
    - `removed_columns`: lista de variables dominantes
    - `distributions` (opcional): dict variable → DataFrame [`value`, `count`, `pct`, `cum_pct`]

- `calculate_woe_iv(df, target, columns=None, include_na=True, bin_numeric=False, num_bins=10, bin_strategy='quantile', logger=None) -> dict`
  - Calcula WOE/IV para variables (principalmente categóricas). Para numéricas, opcionalmente binea (cuantiles o anchos iguales).
  - Requiere target binario 0/1 (1=good, 0=bad). Devuelve:
    - `summary`: DataFrame [`variable`, `iv`] ordenado desc
    - `details`: Dict[var → DataFrame con [`value`,`good`,`bad`,`dist_good`,`dist_bad`,`woe`,`iv_contrib`]]

### API pública

Se puede importar desde `feature_pro` directamente:

```python
from feature_pro import (
    read_dataset,
    summarize_missing,
    detect_column_types,
    impute_missing,
    winsorize_by_percentile,
    winsorize_by_iqr,
    coefficient_of_variation,
    compute_pca,
    compute_correlation_matrix,
    pca_lda_importance,
    select_by_correlation_graph,
)
```

### Ejemplos de uso

Lectura, resumen de missings y filtrado simple:

```python
df = read_dataset("data.csv")
miss = summarize_missing(df)
cols_to_keep = miss.loc[miss["pct_missing"] < 0.6, "column"].tolist()
df = df[cols_to_keep]
```

Imputación por grupos (percentiles) y winsorización con percentiles devueltos:

```python
# Crear bins por percentiles externos a la librería (dependiente del caso de uso)
df["bin20"] = pd.qcut(df["variable_1"], q=5, duplicates="drop")

# Imputación mediana por grupo
df = impute_missing(df, columns=["variable_1","variable_2"], strategy="median", groupby="bin20")

# Winsorización [1,99] y percentiles usados
df_winz, perc = winsorize_by_percentile(df, columns=["variable_1","variable_2"], lower=0.01, upper=0.99, return_percentiles=True)
```

Selección por correlación/grafos con priorización:

```python
sel = select_by_correlation_graph(
    df,
    columns=["variable_1","variable_2","variable_3"],
    target_column="flg_target",
    method="spearman",            # correlación entre variables
    target_method="pointbiserial",# asociación con el objetivo
    threshold=0.6,
    prioritize=["variable_2"],
    plot_matrix=False,
)
selected_vars = sel["selected"]
```

PSI por variable vs período de referencia:

```python
# temporal_column define el agrupador temporal (ej. 'codmes' o fecha)
psi_summary, psi_detail = compute_psi(
    df,
    temporal_column="codmes",
    variables=None,          # si None, toma todas las columnas salvo 'codmes'
    num_bins=10,
    bin_strategy="quantile", # fallback automático a 'equal' si los cuantiles colapsan (registrado en log)
    reference_period=None,   # si None, toma el primer período ordenado
    return_detail=True,
)
# psi_summary: ['variable','psi']
# psi_detail:  ['variable','period','psi']
```

### Convenciones de logging y consola

- Las funciones de la librería no imprimen resultados detallados a la consola. Use `logger` (si se pasa) para registrar progreso y detalles. Los scripts de orquestación (p.ej., en el proyecto raíz) deben imprimir únicamente títulos/secciones y enviar los resultados al log.

### Roadmap corto

- `preprocessing/`: orquestadores de imputación/topeo/normalización/codificación.
- `transformation/`: ingeniería de variables (derivación de rasgos, binning supervisado/no supervisado).
- `analysis/`: visualizaciones “bonitas” (pares, perfiles, drift, etc.) con figuras retornadas para ser guardadas por el usuario.

Si necesitas ampliar funciones o mover alguna utilidad entre módulos, seguimos esta guía para mantener cohesión (genéricas en `common`, específicas por dominio en su subpaquete). 


